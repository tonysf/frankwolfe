#!/usr/bin/env python3
"""Bounded Slurm allocation accounting; offline by default, one query if explicit.

ElapsedRaw * allocated GPUs measures reservation time, not utilization, solver
work, elapsed campaign makespan, billed project credits, or replay-controller
runtime. Original source/results/ledger are never edited.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
from collections import Counter

FIELDS = ['JobID', 'JobIDRaw', 'JobName', 'Partition', 'Account', 'State',
          'ExitCode', 'ElapsedRaw', 'AllocTRES', 'Submit', 'Start', 'End',
          'NodeList', 'AllocCPUS', 'NNodes']
FORMAT = ['JobID%128', 'JobIDRaw%40', 'JobName%100', 'Partition%40',
          'Account%40', 'State%40', 'ExitCode', 'ElapsedRaw', 'AllocTRES%300',
          'Submit', 'Start', 'End', 'NodeList%100', 'AllocCPUS', 'NNodes']
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
            'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED',
          'RESIZING', 'REQUEUED', 'REQUEUE_FED', 'REQUEUE_HOLD', 'SPECIAL_EXIT',
          'STAGE_OUT', 'SIGNALING', 'STOPPED'}
UNSTARTED = {'', 'None', 'Unknown', 'N/A'}
REPLAY_JOBS = {1961687, 1961715, 1961716}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def nonnegative(value: str) -> int:
    if not re.fullmatch(r'\d+', value):
        raise ValueError(f'not a nonnegative integer: {value!r}')
    return int(value)


def expected_slots(ledger: dict) -> dict[int, set[str]]:
    result = {}
    for job in ledger['jobs']:
        job_id = job['job_id']
        if type(job_id) is not int or job_id <= 0 or job_id in result:
            raise ValueError('ledger has invalid/duplicate job IDs')
        if 'array_task_indices' in job:
            indices = job['array_task_indices']
            if not indices or len(set(indices)) != len(indices) or any(type(i) is not int or i < 0 for i in indices):
                raise ValueError('invalid explicit array indices')
            result[job_id] = {f'{job_id}_{i}' for i in indices}
        elif 'array_tasks' in job:
            count = job['array_tasks']
            if type(count) is not int or not 0 < count <= 10000:
                raise ValueError('invalid array task count')
            result[job_id] = {f'{job_id}_{i}' for i in range(count)}
        else:
            result[job_id] = {str(job_id)}
    if not result:
        raise ValueError('empty ledger')
    return result


def parse_rows(raw: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(raw), delimiter='|')
    if reader.fieldnames != FIELDS:
        raise ValueError(f'unexpected sacct header: {reader.fieldnames!r}')
    rows = []
    for index, row in enumerate(reader, 2):
        if None in row or any(v is None for v in row.values()):
            raise ValueError(f'malformed sacct row {index}')
        rows.append({**{k: v.strip() for k, v in row.items()}, 'raw_line_number': index})
    return rows


def expand_indices(text: str) -> list[int]:
    """Expand a bounded Slurm pending-array expression, including stride/cap."""
    if not re.fullmatch(r'[0-9,:-]+(?:%[1-9][0-9]*)?', text):
        raise ValueError('unsupported compressed array expression')
    result = []
    for item in text.split('%')[0].split(','):
        match = re.fullmatch(r'(\d+)(?:-(\d+)(?::(\d+))?)?', item)
        if not match:
            raise ValueError('malformed compressed array interval')
        first = int(match[1]); last = int(match[2] or first); step = int(match[3] or 1)
        if step <= 0 or last < first or last > 100000 or (last-first)//step > 10000:
            raise ValueError('invalid/oversized compressed array interval')
        result.extend(range(first, last+1, step))
    if len(set(result)) != len(result) or len(result) > 10000:
        raise ValueError('overlapping/oversized compressed array expression')
    return result


def gpu_count(row: dict) -> tuple[int, dict]:
    """Generic gres/gpu is a total, not an extra GPU alongside typed TRES."""
    values = {}
    for token in filter(None, row['AllocTRES'].split(',')):
        key, sep, value = token.partition('=')
        if not sep or key in values:
            raise ValueError('malformed/duplicate allocated TRES')
        values[key] = value
    typed = {k: nonnegative(v) for k, v in values.items() if k.startswith('gres/gpu:')}
    total = nonnegative(values['gres/gpu']) if 'gres/gpu' in values else None
    if total is not None and typed and sum(typed.values()) != total:
        raise ValueError('generic and typed allocated GPU counts disagree')
    if total is None:
        total = sum(typed.values()) if typed else 0
    state = row['State'].split()[0]
    unstarted_zero = (row['Start'] in UNSTARTED and row['ElapsedRaw'] == '0'
                      and state in {'PENDING', 'CANCELLED'})
    if not values and not unstarted_zero:
        raise ValueError('missing allocated TRES for a started/otherwise unresolved job')
    if row['Partition'] == 'gpua100' and not total and not unstarted_zero:
        raise ValueError('started gpua100 allocation has no allocated GPU count')
    if row['Partition'] == 'gpua100' and any('a100' not in k.lower() for k in typed):
        raise ValueError('non-A100 typed GPU conflicts with gpua100 partition')
    return total, typed


def summarize(ledger: dict, raw: str) -> dict:
    expected = expected_slots(ledger)
    all_expected = set().union(*expected.values())
    candidates = {}; excluded = []; issues = []
    def problem(identity, message):
        issues.append({'job_id': identity, 'issue': message})
    parsed = parse_rows(raw)
    for row in parsed:
        identity = row['JobID']
        base_match = re.match(r'^(\d+)(?:_|\.|$)', identity)
        if not base_match or int(base_match[1]) not in expected:
            problem(identity, 'row outside ledger job scope')
            excluded.append({'raw': row, 'reason': 'outside_scope'})
            continue
        base = int(base_match[1])
        if '.' in identity:
            excluded.append({'raw': row, 'reason': 'job_step'})
            continue
        if identity == str(base) and expected[base] != {identity}:
            excluded.append({'raw': row, 'reason': 'aggregate_array_parent'})
            continue
        compressed = re.fullmatch(r'(\d+)_\[([^]]+)\]', identity)
        identities = [identity]
        if compressed:
            try:
                state = row['State'].split()[0]
                if not (state in {'PENDING', 'CANCELLED'} and row['ElapsedRaw'] == '0'
                        and not row['AllocTRES'] and row['Start'] in UNSTARTED):
                    raise ValueError('compressed row is not proven unstarted with zero allocation')
                identities = [f'{base}_{i}' for i in expand_indices(compressed[2])]
            except ValueError as exc:
                problem(identity, str(exc)); excluded.append({'raw': row, 'reason': 'unresolved_compressed_row'}); continue
        for logical_id in identities:
            if logical_id not in all_expected:
                problem(logical_id, 'unexpected allocation/task ID')
                excluded.append({'raw': row, 'logical_job_id': logical_id, 'reason': 'unexpected_task'})
                continue
            candidates.setdefault(logical_id, []).append((base, row, bool(compressed)))
    allocations = []
    for logical_id, entries in sorted(candidates.items()):
        if len(entries) != 1:
            problem(logical_id, 'duplicate allocation records; no arbitrary deduplication or double counting')
            excluded.extend({'raw': row, 'logical_job_id': logical_id, 'reason': 'duplicate_primary_record'} for _, row, _ in entries)
            continue
        base, row, compressed = entries[0]
        item = {'job_id': logical_id, 'ledger_job_id': base, 'raw': row,
                'expanded_unstarted_array': compressed, 'terminal': False,
                'allocated_gpus': None, 'elapsed_seconds': None, 'gpu_seconds': None,
                'a100_gpu_seconds': None, 'accounting_valid': False}
        try:
            state = row['State'].split()[0]
            if state not in TERMINAL | ACTIVE:
                raise ValueError(f'unknown/truncated state {state!r}')
            item['state'] = state; item['terminal'] = state in TERMINAL
            elapsed = nonnegative(row['ElapsedRaw']); gpus, typed = gpu_count(row)
            if state == 'PENDING' and (elapsed or gpus or row['Start'] not in UNSTARTED):
                raise ValueError('pending allocation has contradictory nonzero/start data')
            item.update({'allocated_gpus': gpus, 'elapsed_seconds': elapsed,
                         'gpu_seconds': elapsed*gpus})
            if not gpus:
                a100_gpus = 0; basis = 'no allocated GPUs'
            elif row['Partition'] == 'gpua100':
                a100_gpus = gpus; basis = 'Ruche gpua100 partition'
            elif typed:
                a100_gpus = sum(v for k,v in typed.items() if 'a100' in k.lower()); basis = 'typed allocated GPU TRES'
            else:
                raise ValueError('GPU type unresolved outside gpua100')
            item.update({'a100_gpu_seconds': elapsed*a100_gpus, 'gpu_type_basis': basis,
                         'accounting_valid': True})
        except (ValueError, IndexError) as exc:
            problem(logical_id, str(exc)); item['issue'] = str(exc)
        allocations.append(item)
    represented = {r['job_id'] for r in allocations}
    missing = sorted(all_expected-represented)
    by_job = []
    for job in ledger['jobs']:
        base = job['job_id']; rows = [r for r in allocations if r['ledger_job_id']==base]
        valid = [r for r in rows if r['accounting_valid']]
        known = sum(r['a100_gpu_seconds'] for r in valid)
        terminal = sum(r['a100_gpu_seconds'] for r in valid if r['terminal'])
        unresolved = sorted(expected[base]-{r['job_id'] for r in valid})
        by_job.append({'job_id':base, 'purpose':job['purpose'],
                       'cost_group':'momentum_diagnostic_reconstruction_and_validation' if base in REPLAY_JOBS else 'original_study_and_validation',
                       'expected_primary_rows':len(expected[base]), 'accounted_primary_rows':len(valid),
                       'unresolved_primary_rows':unresolved,
                       'states':dict(sorted(Counter(r.get('state','UNRESOLVED') for r in rows).items())),
                       'all_terminal':not unresolved and all(r['terminal'] for r in rows),
                       'known_a100_gpu_seconds':known, 'known_a100_gpu_hours':known/3600,
                       'terminal_a100_gpu_seconds':terminal,
                       'nonterminal_accrued_a100_gpu_seconds':known-terminal,
                       'prescribed_solver_budget_seconds':job.get('nominal_solver_seconds'),
                       'prescribed_solver_budget_is_measured_allocation':False})
    valid = [r for r in allocations if r['accounting_valid']]
    terminal_seconds = sum(r['a100_gpu_seconds'] for r in valid if r['terminal'])
    active_seconds = sum(r['a100_gpu_seconds'] for r in valid if not r['terminal'])
    coverage = not missing and not issues and len(valid)==len(all_expected)
    complete = coverage and all(r['terminal'] for r in valid)
    groups = {}
    for group in sorted({j['cost_group'] for j in by_job}):
        subset = [j for j in by_job if j['cost_group']==group]
        seconds = sum(j['known_a100_gpu_seconds'] for j in subset)
        groups[group] = {'job_ids':[j['job_id'] for j in subset], 'known_a100_gpu_seconds':seconds,
                         'known_a100_gpu_hours':seconds/3600,
                         'all_terminal_and_accounted':all(j['all_terminal'] for j in subset)}
    return {'schema_version':1, 'status':'complete' if complete else 'preliminary_incomplete',
            'generated_utc':utc(), 'accounting_complete':complete, 'snapshot_coverage_complete':coverage,
            'requested_job_ids':list(expected), 'expected_primary_rows':len(all_expected),
            'raw_sacct_rows':len(parsed),
            'raw_primary_rows_before_expansion':sum('.' not in r['JobID'] for r in parsed),
            'accounted_primary_rows':len(valid), 'missing_primary_rows':missing,
            'issues':issues, 'jobs':by_job, 'allocations':allocations, 'excluded_rows':excluded,
            'excluded_row_counts':dict(Counter(r['reason'] for r in excluded)),
            'terminal_a100_gpu_seconds':terminal_seconds, 'terminal_a100_gpu_hours':terminal_seconds/3600,
            'nonterminal_accrued_a100_gpu_seconds':active_seconds,
            'known_snapshot_a100_gpu_seconds':terminal_seconds+active_seconds,
            'known_snapshot_a100_gpu_hours':(terminal_seconds+active_seconds)/3600,
            'allocation_states':dict(Counter(r.get('state','UNRESOLVED') for r in allocations)),
            'cpu_only_primary_rows':sum(r['allocated_gpus']==0 and r['elapsed_seconds']>0 for r in valid),
            'cpu_only_elapsed_seconds':sum(r['elapsed_seconds'] for r in valid if r['allocated_gpus']==0 and r['elapsed_seconds']>0),
            'cost_groups':groups,
            'policy':'Sum primary allocation ElapsedRaw * allocated GPU count. Exclude every job step and aggregate array parent. Generic and typed GPU TRES are not added twice. Failed started allocations count; proven unstarted cancellations/pending tasks have zero accrued allocation. Active/missing/invalid accounting prevents final totals.',
            'limitations':['Scope is exactly the bound ledger job IDs, not unlisted local CPU work or unrelated cluster jobs.',
                          'Allocated time is not measured GPU utilization, solver work, campaign makespan, or billed project credits.',
                          'Original solver budgets and replay-controller times remain separate and are not added to allocation time.',
                          'Scheduler Submit/Start/End strings retain their returned timezone convention; totals use ElapsedRaw, not timestamp subtraction.',
                          'A100 identity follows documented Ruche gpua100 partition or explicit typed GPU TRES.',
                          'Snapshot totals remain provisional until all prescribed allocations are terminal and accounting is complete.']}


def capture_query(ledger_path: Path, snapshot_root: Path) -> Path:
    rawledger = ledger_path.read_bytes(); ledger = json.loads(rawledger)
    ids = list(expected_slots(ledger))
    remote = ['sacct','--parsable2','--jobs='+','.join(map(str,ids)),
              '--starttime=2026-09-26T00:00:00','--format='+','.join(FORMAT)]
    argv = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=15',
            '-o','ServerAliveCountMax=2','ruche',shlex.join(remote)]
    now=dt.datetime.now(dt.timezone.utc)
    directory=snapshot_root/now.strftime('%Y%m%dT%H%M%S%fZ');directory.mkdir(parents=True,exist_ok=False)
    (directory/'job_ledger.json').write_bytes(rawledger)
    record={'schema_version':1,'scope':'Exactly the job IDs in the bound ledger; one read-only sacct query, including steps for auditable exclusion.',
            'requested_job_ids':ids,'started_utc':now.isoformat(),'command':argv,'remote_argv':remote,'ledger_sha256':sha(rawledger)}
    try:
        process=subprocess.run(argv,capture_output=True,timeout=55)
        stdout,stderr=process.stdout,process.stderr; record['returncode']=process.returncode
    except subprocess.TimeoutExpired as exc:
        stdout,stderr=exc.stdout or b'',exc.stderr or b''
        record.update(returncode=None,error='query_timeout_no_retry')
    (directory/'sacct.psv').write_bytes(stdout);(directory/'sacct.stderr').write_bytes(stderr)
    record.update(returned_utc=utc(),stdout_sha256=sha(stdout),stderr_sha256=sha(stderr))
    (directory/'query.json').write_text(json.dumps(record,indent=2)+'\n')
    return directory


def load_snapshot(directory: Path) -> tuple[dict,dict]:
    query_bytes=(directory/'query.json').read_bytes();query=json.loads(query_bytes)
    ledger_bytes=(directory/'job_ledger.json').read_bytes();raw=(directory/'sacct.psv').read_bytes()
    stderr=(directory/'sacct.stderr').read_bytes()
    for key,data in [('ledger_sha256',ledger_bytes),('stdout_sha256',raw),('stderr_sha256',stderr)]:
        if sha(data)!=query[key]: raise ValueError('snapshot hash mismatch: '+key)
    if query['returncode']!=0: raise ValueError('sacct query unsuccessful; evidence retained, no retry')
    ledger=json.loads(ledger_bytes)
    if list(expected_slots(ledger))!=query['requested_job_ids']:
        raise ValueError('query scope differs from bound ledger')
    expected_scope='--jobs='+','.join(map(str,query['requested_job_ids']))
    if query['remote_argv'].count(expected_scope)!=1:
        raise ValueError('query command not bound to exact requested IDs')
    report=summarize(ledger,raw.decode())
    report['input_provenance']={'snapshot_directory':str(directory.resolve()),'query_sha256':sha(query_bytes),
        'ledger_sha256':sha(ledger_bytes),'raw_sacct_sha256':sha(raw),'stderr_sha256':sha(stderr),
        'queried_from_utc':query['started_utc'],'queried_until_utc':query['returned_utc'],
        'remote_argv':query['remote_argv'],'query_stderr':stderr.decode(errors='replace'),
        'helper_sha256':sha(Path(__file__).read_bytes())}
    return report,query


def markdown(report:dict)->str:
    p=report['input_provenance']
    lines=['# Compute allocation accounting','',
           '**'+('Complete accounting for the bound ledger.' if report['accounting_complete'] else 'Preliminary: final allocation accounting remains incomplete.')+'**', '',
           f"The one scoped scheduler snapshot was captured {p['queried_from_utc']} to {p['queried_until_utc']}. "
           f"It accounts for {report['accounted_primary_rows']}/{report['expected_primary_rows']} expected primary allocation/task records across {len(report['requested_job_ids'])} job IDs. "
           f"Known A100 allocation to this snapshot: **{report['known_snapshot_a100_gpu_seconds']:,} GPU-seconds ({report['known_snapshot_a100_gpu_hours']:.6f} GPU-hours)**, "
           f"of which {report['terminal_a100_gpu_seconds']:,} GPU-seconds are terminal and {report['nonterminal_accrued_a100_gpu_seconds']:,} have accrued in nonterminal allocations. "
           + ('All scoped allocations are terminal and accounted for.' if report['accounting_complete'] else
            'Active allocations can continue accruing time; this is not the final experiment total.'), '',
           '| Job / array | Purpose | States (primary tasks) | Accounted / expected | A100 GPU-seconds so far | A100 GPU-hours so far |',
           '|---|---|---|---:|---:|---:|']
    for j in report['jobs']:
        states=', '.join(f'{k}: {v}' for k,v in j['states'].items()) or 'missing'
        lines.append(f"| {j['job_id']} | {j['purpose']} | {states} | {j['accounted_primary_rows']}/{j['expected_primary_rows']} | {j['known_a100_gpu_seconds']:,} | {j['known_a100_gpu_hours']:.6f} |")
    lines+=['','Allocation categories (never added to solver or replay-controller times):','']
    for name,g in report['cost_groups'].items():
        lines.append(f"- {name}: {g['known_a100_gpu_seconds']:,} A100 GPU-seconds ({g['known_a100_gpu_hours']:.6f} hours); all terminal/accounted: {g['all_terminal_and_accounted']}.")
    lines += ['',f"Raw scheduler rows: {report['raw_sacct_rows']}; primary rows before compressed-array expansion: {report['raw_primary_rows_before_expansion']}. "
              f"Excluded rows: {json.dumps(report['excluded_row_counts'],sort_keys=True)}. "
              f"CPU-only started allocations: {report['cpu_only_primary_rows']}, summed allocation elapsed {report['cpu_only_elapsed_seconds']} seconds, contributing zero GPU time.",
              '', 'Missing allocation records: '+(', '.join(report['missing_primary_rows']) or 'none')+'.',
              'Accounting issues: '+(json.dumps(report['issues']) if report['issues'] else 'none')+'.',
              '', report['policy'], '', 'Limits:', '']
    lines += ['- '+x for x in report['limitations']]
    lines += ['',f"Raw evidence: `{p['snapshot_directory']}/sacct.psv`, SHA-256 `{p['raw_sacct_sha256']}`. "
              f"The same snapshot directory contains stderr, the exact query command/timestamps, and the bound ledger. "
              f"Helper SHA-256: `{p['helper_sha256']}`.",
              '', 'Offline regeneration:', '', '```bash',
              f".venv/bin/python account_compute_resources.py --snapshot {p['snapshot_directory']} --out reports/COMPUTE_ACCOUNTING",'```','',
              'After final replay completion, an explicitly authorized refresh captures one new immutable snapshot:', '', '```bash',
              '.venv/bin/python account_compute_resources.py --query --ledger reports/job_ledger.json --out reports/COMPUTE_ACCOUNTING','```','']
    return '\n'.join(lines)


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--snapshot',type=Path);mode.add_argument('--query',action='store_true')
    parser.add_argument('--ledger',type=Path,default=Path('reports/job_ledger.json'))
    parser.add_argument('--snapshot-root',type=Path,default=Path('reports/compute_accounting_snapshots'))
    parser.add_argument('--out',type=Path,default=Path('reports/COMPUTE_ACCOUNTING'))
    args=parser.parse_args()
    directory=capture_query(args.ledger,args.snapshot_root) if args.query else args.snapshot
    report,_=load_snapshot(directory)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.with_suffix('.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    args.out.with_suffix('.md').write_text(markdown(report))
    print(json.dumps({k:report[k] for k in ['status','accounted_primary_rows','expected_primary_rows','known_snapshot_a100_gpu_seconds','terminal_a100_gpu_seconds','nonterminal_accrued_a100_gpu_seconds','missing_primary_rows','issues']}))
    return 0 if report['snapshot_coverage_complete'] else 1


if __name__=='__main__':
    raise SystemExit(main())
