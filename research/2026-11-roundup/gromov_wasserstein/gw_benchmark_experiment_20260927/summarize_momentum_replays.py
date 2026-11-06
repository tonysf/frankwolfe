"""Summarize separately verified momentum replays; never read coupling arrays.

Pair repeated --manifest and --root arguments in order. Roots are downloaded
companion roots, not original optimizer roots. Original clocks remain plot labels;
controller elapsed time measures replay cost and is not added to worker subtotals.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
EXPECTED_COUNTS = {'tuning': 504, 'pilot': 6, 'finals': 286}
from method_names import method_name

METHODS = {'smoothing-fw', 'smoothing-fw-exact', 'smoothing-row-projection'}
PROOF_CHECKS = ('source_and_instance_bindings', 'environment_match', 'P_raw_bitwise',
                'P_rounded_bitwise', 'solver_counters', 'diagnostics_preserved_oracle_rng',
                'original_files_unchanged')
COUNTERS = ('oracle_evaluations', 'exact_gradient_calls', 'stochastic_gradient_calls',
 'sketch_probes', 'anchor_samples', 'initialization_oracle_calls', 'row_lmo_calls',
 'row_projection_calls', 'transport_lmo_calls', 'kl_projection_calls', 'euclidean_projection_calls',
 'inner_iterations_total', 'inner_stopping_failures', 'underflow_steps', 'underflow_entries_total',
 'underflow_entries_max', 'attempted_updates', 'completed_updates')
RESIDUALS = ('row_l1', 'col_l1', 'col_l2', 'both_l1', 'mass_error', 'min_entry')


def require(condition, message):
    if not condition: raise ValueError(message)


def read_json(path): return json.loads(Path(path).read_text())
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def valid_hash(value): return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)

def number(value):
    result = float(value)
    require(math.isfinite(result), 'Nonfinite scalar.')
    return result

def integer(value):
    result = int(value)
    require(not isinstance(value, bool) and number(value) == result and result >= 0, 'Invalid nonnegative integer.')
    return result

def safe_id(value):
    path = Path(value)
    require(not path.is_absolute() and str(path) == value and '..' not in path.parts and path.parts and not path.parts[0].startswith('_'), 'Unsafe run ID.')
    return path


@lru_cache(maxsize=None)
def utility_hash(revision):
    require(isinstance(revision, str) and len(revision) == 40 and all(c in '0123456789abcdef' for c in revision), 'Invalid controller revision.')
    result = subprocess.run(['git', 'show', revision + ':replay_momentum_diagnostics.py'], cwd=ROOT, capture_output=True)
    require(result.returncode == 0, 'Controller utility source unavailable in local Git: ' + revision)
    return hashlib.sha256(result.stdout).hexdigest()


def expected_bindings(trial, binding):
    expected_keys = {'metadata_sha256', 'trace_sha256', 'final_sha256', 'instance_sha256', 'source_file_sha256', 'theory_config_sha256'}
    require(set(binding) == expected_keys, 'Incomplete original binding schema.')
    require(valid_hash(binding['metadata_sha256']), 'Missing original metadata binding.')
    require(valid_hash(binding['instance_sha256']) and all(valid_hash(x) for x in binding['source_file_sha256'].values()) and valid_hash(binding['theory_config_sha256']), 'Malformed input/source binding hash.')
    require(all(value is None or valid_hash(value) for value in (binding['trace_sha256'], binding['final_sha256'])), 'Malformed trace/final binding.')
    require(binding['instance_sha256'] == trial['instance_sha256'], 'Original instance binding differs from manifest.')
    sources = trial['source_snapshot_sha256']
    require(binding['source_file_sha256'] == {name: sources[name] for name in ('run_benchmark.py', 'smoothing_fw.py', 'gw_core.py')}, 'Original source binding differs from manifest.')
    require(binding['theory_config_sha256'] == sources['theory_config.json'], 'Original theory binding differs from manifest.')
    if trial['artifact_binding'] == 'frozen_at_preparation':
        require({name: binding[key] for name, key in [('metadata.json', 'metadata_sha256'), ('trace.csv', 'trace_sha256'), ('final.npz', 'final_sha256')]} == trial['original_artifact_sha256'], 'Frozen original artifacts differ from manifest.')
    else:
        require(trial['artifact_binding'] == 'freeze_terminal_before_replay', 'Unknown artifact-binding policy.')


def check_config(trial, actual):
    if trial.get('original_config') is not None:
        require(actual == trial['original_config'], 'Original config differs from frozen pilot/tuning config.')
    spec = trial.get('original_run_spec')
    if spec:
        argv = spec['argv']
        require(argv[0] == 'run_benchmark.py' and len(argv) % 2 == 1, 'Malformed frozen original command.')
        for key, value in zip(argv[1::2], argv[2::2]):
            found = actual.get(key[2:].replace('-', '_'))
            require((number(found) == number(value) if isinstance(found, (int, float)) and not isinstance(found, bool) else found == value), 'Original config disagrees with ' + key)
        require(actual.get('initialization', 'product') == spec['initialization'], 'Original initialization disagrees with manifest.')
    if trial['method'] == 'smoothing-fw-exact': require(actual['oracle'] == 'exact', 'Exact method label has nonexact oracle.')


def validate_verified(proof, rows, trial):
    checks = proof['checks']
    require(all(checks.get(key) is True for key in PROOF_CHECKS), 'Missing successful verification proof.')
    count = proof['checkpoint_count']
    require(type(count) is int and count > 0 and len(rows) == count == len(checks['checkpoint_scalars']), 'Checkpoint proof/CSV coverage differs.')
    require(proof['candidate_checkpoint_count'] == count, 'Candidate/verified checkpoint counts differ.')
    require(proof['original_status'] == 'completed' and proof['error'] is None and proof['failure_kind'] is None, 'Verified replay carries original/replay failure.')
    require(proof['original_counters'] == proof['replay_counters'], 'Original/replay counters differ.')
    require(all(valid_hash(proof['bindings'][key]) for key in ('trace_sha256','final_sha256')), 'Verified replay lacks original artifact bindings.')
    config = proof['original_config']; check_config(trial, config)
    N = integer(proof['completed_updates'])
    require(N == proof['original_counters']['completed_updates'], 'Final update count differs.')
    require(checks['initialization_hash'] == 'matched' or (trial['original_phase'] == 'pilot' and checks['initialization_hash'] == 'unavailable_in_legacy_original; initial_metrics_and_terminal_arrays_required'), 'Initialization proof unavailable outside historical pilot.')
    for key in ('initialization_plan_sha256', 'initial_rng_state_sha256', 'terminal_rng_state_sha256'):
        require(valid_hash(proof[key]), 'Missing initialization/RNG hash.')
    for name in ('P_raw', 'P_rounded'):
        endpoint = proof['endpoint_comparisons'][name]
        require(endpoint['bitwise_match'] is True and endpoint['difference_all_finite'] is True and endpoint['max_abs_difference'] == 0, 'Endpoint comparison does not prove identity.')
        require(valid_hash(endpoint['expected_array_sha256']) and endpoint['expected_array_sha256'] == endpoint['replay_array_sha256'] == proof[name + '_array_sha256'], 'Endpoint array hashes differ.')
        require(endpoint['expected_shape'] == endpoint['replay_shape'] and endpoint['expected_dtype'] == endpoint['replay_dtype'] == config['dtype'], 'Endpoint shape/dtype differs.')
        spec = trial.get('original_run_spec')
        if spec: require(endpoint['expected_shape'] == [spec['n'], spec['m']], 'Endpoint differs from frozen dimensions.')
    calls = proof['replay_diagnostic_counts']
    require(calls['oracle_draws'] == 0 and all(integer(v) == v for v in calls.values()), 'Invalid replay diagnostic accounting.')
    require(calls['rounding_calls'] == count and calls['loss_calls'] == 3 * count, 'Replay diagnostic call coverage differs.')
    expected_gap = proof['endpoint_comparisons']['P_raw']['expected_shape']
    gap_required = expected_gap[0] == expected_gap[1] and max(expected_gap) <= config['gap_max_n']
    require(calls['assignment_lmo_calls'] == (count if gap_required else 0), 'Assignment diagnostic count differs.')
    require(calls['full_gradient_calls'] == 2 * count - 1 + (count if gap_required else 0), 'Gradient diagnostic count differs.')
    required_fields = {'raw_loss', 'rounded_loss', 'repair_l1', 'mu', 'smoothing_point_iteration', 'true_smoothed_gap'} | set(RESIDUALS) | {'rounded_' + x for x in RESIDUALS} | {'cumulative_' + x for x in COUNTERS}
    if gap_required: required_fields.add('original_fw_gap')
    seen_rows = set(); seen_iterations = []; clean = []
    previous_time = -1.
    for row, item in zip(rows, checks['checkpoint_scalars']):
        K, trace_index = integer(row['output_iteration']), integer(row['original_trace_row'])
        require(K <= N and item['iteration'] == K and item['original_trace_row'] == trace_index and item['exact_match'] is True, 'Checkpoint index/proof mismatch.')
        point_fields = required_fields | ({'last_update_gamma','last_update_rho','last_update_mu'} if K else set())
        if K and config['method']=='smoothing-row-projection': point_fields.add('last_update_eta')
        require(trace_index not in seen_rows and point_fields.issubset(item['compared_fields']), 'Duplicate or incomplete checkpoint scalar proof.')
        seen_rows.add(trace_index); seen_iterations.append(K)
        require(row['kind'] in ('initial', 'checkpoint', 'final'), 'Unobserved/missed row published as diagnostic.')
        require(integer(row['momentum_point_iteration']) == integer(row['stored_momentum_iteration']) == max(0, K-1), 'Momentum point is not the last original input.')
        require(integer(row['tracking_error_point_iteration']) == K and row['momentum_error_matches_current_output'] == str(K == 0), 'Current-output alignment mislabeled.')
        require(integer(row['replay_diagnostic_oracle_draws']) == 0, 'Diagnostic consumed oracle randomness.')
        require(integer(row['original_solver_oracle_evaluations']) == max(1, K), 'Checkpoint oracle count differs.')
        for field in point_fields:
            number(row['recomputed_' + field])  # Required finite scalar, including signed loss/gap.
        solver = number(row['original_actual_solver_seconds']); available = number(row['original_time_to_this_feasible_output'])
        require(solver >= previous_time and solver >= 0 and available >= solver, 'Invalid original observation clock.')
        previous_time = solver
        target = None if row.get('original_target_solver_seconds', '') == '' else number(row['original_target_solver_seconds'])
        if row['kind'] == 'checkpoint': require(target is not None and 0 < target <= solver, 'Checkpoint precedes original target.')
        else: require(target is None, 'Initial/final row has a fabricated target.')
        primary = number(row['momentum_error_frobenius']); stale = number(row['stored_momentum_current_output_tracking_error_frobenius'])
        require(primary >= 0 and stale >= 0, 'Negative norm.')
        clean.append({'original_trace_row': trace_index, 'kind': row['kind'], 'output_iteration': K, 'momentum_point_iteration': max(0,K-1), 'target_seconds': target, 'original_solver_seconds': solver, 'original_output_seconds': available, 'momentum_error_frobenius': primary, 'stale_current_output_error_frobenius': stale, 'original_oracle_evaluations': integer(row['original_solver_oracle_evaluations']), 'original_sketch_probes': integer(row['recomputed_cumulative_sketch_probes'])})
    require(seen_iterations == sorted(set(seen_iterations)) and seen_iterations[0] == 0 and seen_iterations[-1] == N, 'Missing/repeated/out-of-order original endpoints.')
    require(rows[0]['kind'] == 'initial', 'Initial diagnostic absent.')
    require(clean[-1]['original_solver_seconds'] == number(proof['original_timing']['solver_seconds']), 'Final original solver clock differs.')
    missed = proof['missed_original_checkpoints']
    require(all(integer(x['original_trace_row']) not in seen_rows for x in missed), 'Missed window also published as observed.')
    require(len({integer(x['original_trace_row']) for x in missed}) == len(missed), 'Duplicate missed original windows.')
    require(seen_rows | {integer(x['original_trace_row']) for x in missed} == set(range(count + len(missed))), 'Original trace row coverage has holes.')
    budget = number(config['budget'])
    expected_targets = ([number(x) for x in config['checkpoints'].split(',')] if config.get('checkpoints') is not None else [x for x in (1.,3.,10.,30.,100.,300.) if x <= budget])
    expected_targets = set(expected_targets + [budget])
    observed_targets = [x['target_seconds'] for x in clean if x['kind']=='checkpoint']
    missed_targets = [number(x['target_solver_seconds']) for x in missed]
    require(len(set(observed_targets + missed_targets)) == len(observed_targets + missed_targets) and set(observed_targets + missed_targets) == expected_targets, 'Original checkpoint-window coverage differs from frozen grid.')
    return clean


def inspect_trial(trial, path, manifest_hash, controller, ledger_record):
    spec = trial.get('original_run_spec') or {}
    result = dict(run_id=trial['run_id'], original_phase=trial['original_phase'], method=trial['method'], instance=trial['instance'], family=spec.get('family'), noise=spec.get('noise'), problem_seed=spec.get('problem_seed'), seed=spec.get('seed'), init_seed=spec.get('init_seed'), initialization=spec.get('initialization', 'product'), status='missing', terminal=False, provenance_valid=True, verified=False, issues=[], checkpoints=[])
    config = spec or trial['original_config']
    targets = config.get('checkpoints')
    if isinstance(targets, str): targets = targets.split(',')
    if targets is None: targets = [x for x in (1.,3.,10.,30.,100.,300.) if x <= number(config['budget'])]
    result['prescribed_checkpoint_seconds'] = sorted({number(x) for x in targets} | {number(config['budget'])})
    if not (path / 'replay_receipt.json').exists(): return result
    if ledger_record is None:
        result['status'] = 'pending_ledger'; return result
    try:
        receipt = read_json(path / 'replay_receipt.json')
        require({k:v for k,v in ledger_record.items() if k != 'resumed_terminal_record'} == receipt, 'Ledger record differs from replay receipt.')
        require(receipt['run_spec'] == trial and receipt['manifest_sha256'] == manifest_hash and receipt['controller_source_commit'] == controller, 'Receipt differs from manifest/controller.')
        binding = receipt['original_bindings']; expected_bindings(trial, binding)
        require(valid_hash(receipt['original_campaign_receipt_sha256']) if spec else receipt['original_campaign_receipt_sha256'] is None, 'Original campaign receipt binding missing or unexpected.')
        verification_path = path / 'verification.json'; proof = read_json(verification_path)
        require(digest(verification_path) == receipt['output_sha256']['verification_sha256'], 'Verification hash differs from receipt.')
        require(proof['schema_version'] == 1 and proof['status'] in ('verified', 'failed') and proof['status'] == receipt['status'], 'Nonterminal/inconsistent verification status.')
        require(proof['verified'] is (proof['status'] == 'verified') and proof['bindings'] == binding, 'Verification original bindings/status differ.')
        require(proof.get('utility_sha256') == utility_hash(controller), 'Utility hash differs from controller Git revision.')
        require(proof.get('original_source_commit') == trial['original_source_commit'], 'Original source revision differs.')
        check_config(trial, proof['original_config'])
        require(proof.get('run_dir') == trial['original_run_dir'], 'Original run path differs.')
        require(proof.get('checks',{}).get('original_files_unchanged') is True, 'Original immutability was not verified.')
        require(receipt['returncode'] == (0 if proof['verified'] else 1), 'Worker/verification exit status differs.')
        diagnostics_path = path / 'diagnostics.csv'
        require(diagnostics_path.is_file() is proof['verified'], 'Only verified replay may publish diagnostics.csv.')
        if proof['verified']:
            require(digest(diagnostics_path) == proof['diagnostics_sha256'] == receipt['output_sha256']['diagnostics_sha256'], 'Diagnostics hash differs from proof/receipt.')
            with diagnostics_path.open(newline='') as stream: rows = list(csv.DictReader(stream))
            result['checkpoints'] = validate_verified(proof, rows, trial)
        else:
            require(receipt['output_sha256']['diagnostics_sha256'] is None, 'Failed replay has published diagnostic hash.')
            require(proof.get('failure_kind') and proof.get('error'), 'Unverified replay has no failure reason.')
        timing = proof['timing']; elapsed = number(timing['controller_elapsed_seconds'])
        require(elapsed >= 0, 'Negative controller elapsed time.')
        require(all(number(value) >= 0 for value in timing.values()), 'Invalid replay timing component.')
        result.update(status=proof['status'], terminal=True, verified=proof['verified'], original_status=proof.get('original_status'), controller_source_commit=controller, utility_sha256=proof['utility_sha256'], original_source_commit=proof['original_source_commit'], original_bindings=binding, original_counters=proof.get('original_counters'), replay_counters=proof.get('replay_counters'), replay_diagnostic_counts=proof.get('replay_diagnostic_counts'), original_timing=proof.get('original_timing'), replay_timing=timing, controller_elapsed_seconds=elapsed, failure_kind=proof.get('failure_kind'), error=proof.get('error'), observed_checkpoint_count=len(result['checkpoints']), missed_original_checkpoints=proof.get('missed_original_checkpoints', []), verification_sha256=digest(verification_path), receipt_sha256=digest(path / 'replay_receipt.json'), diagnostics_sha256=receipt['output_sha256']['diagnostics_sha256'])
    except (ValueError, KeyError, TypeError, OSError, csv.Error) as error:
        result.update(status='invalid', provenance_valid=False, issues=[str(error)], checkpoints=[])
    return result


def inspect_phase(manifest_path, local_root):
    plan = read_json(manifest_path); manifest_hash = digest(manifest_path); root = Path(local_root).resolve()
    require(plan['schema_version'] == 1 and plan['purpose'] == 'missing_momentum_diagnostics_only', 'Unsupported replay manifest.')
    trials = plan['runs']; ids = [r['run_id'] for r in trials]
    require(len(set(ids)) == len(ids) == plan['run_count'], 'Duplicate/missing declared run IDs.')
    require(len({r['original_run_dir'] for r in trials}) == len(trials), 'Duplicate original trajectories.')
    for r in trials:
        safe_id(r['run_id']); require(r['method'] in METHODS, 'Unrelated method in replay manifest.')
    errors = []; records = {}; controller = None
    binding_path = root / '_replay/binding.json'
    if binding_path.exists():
        try:
            binding = read_json(binding_path); controller = binding['source_commit']
            require(binding['manifest_sha256'] == manifest_hash and binding['out_root'] == plan['out_root'], 'Root binding differs from manifest.')
            count = integer(binding['task_count']); require(count > 0, 'Empty sharding.')
            utility_hash(controller)
            for ledger_path in sorted((root / '_replay').glob('shard_*.json')):
                ledger = read_json(ledger_path)
                require(all(ledger.get(k) == v for k,v in binding.items()), 'Shard/root binding differs.')
                i = integer(ledger['task_index']); require(i < count and ledger_path.name == f'shard_{i:02d}.json', 'Shard identity/path differs.')
                assigned = trials[i::count]; assigned_ids = [r['run_id'] for r in assigned]
                require(ledger['assigned_count'] == len(assigned), 'Shard assigned coverage differs.')
                entries = ledger['records']; observed = [r['run_spec']['run_id'] for r in entries]
                require(observed == assigned_ids[:len(observed)], 'Shard records are not the prescribed prefix.')
                for record in entries:
                    rid = record['run_spec']['run_id']; require(rid not in records, 'Duplicate ledger trial.')
                    records[rid] = record
                if ledger['status'] in ('completed', 'completed_with_unverified_replays'):
                    require(len(entries) == len(assigned), 'Completed shard lacks prescribed records.')
                elif ledger['status'] != 'running': errors.append('Shard ' + str(i) + ': ' + str(ledger.get('status')) + ': ' + str(ledger.get('error')))
        except (ValueError, KeyError, TypeError, OSError) as error: errors.append(str(error))
    extra = {str(p.parent.relative_to(root)) for p in root.rglob('replay_receipt.json')} - set(ids) if root.exists() else set()
    if extra: errors.append('Unprescribed replay receipts: ' + ', '.join(sorted(extra)))
    results = [inspect_trial(trial, root / safe_id(trial['run_id']), manifest_hash, controller, records.get(trial['run_id'])) for trial in trials]
    if not binding_path.exists() and any((root / safe_id(rid) / 'replay_receipt.json').exists() for rid in ids): errors.append('Trial receipts exist without a root binding.')
    return dict(phase=plan['phase'], path=str(Path(manifest_path).resolve()), sha256=manifest_hash, local_root=str(root), remote_root=plan['out_root'], controller_source_commit=controller, prescribed_runs=len(trials), errors=errors), results, plan


def interval(values):
    ordered = sorted(values)
    if not ordered: return None
    if len(ordered) == 1: q1 = q3 = ordered[0]
    else: q1, _, q3 = statistics.quantiles(ordered, n=4, method='inclusive')
    return {'count':len(ordered), 'median':statistics.median(ordered), 'q25':q1, 'q75':q3}


def curve_summaries(runs):
    """Keep the prescribed denominator even when a replay/window is absent."""
    selected = [r for r in runs if r['original_phase'] in ('final_product', 'final_randomized')]
    prescribed=defaultdict(list)
    grouped=defaultdict(list)
    for r in selected:
        family='bmmc' if r['family']=='bmmc' else 'synthetic_noisy' if r['noise'] else 'synthetic_clean'
        for target in r['prescribed_checkpoint_seconds']:
            prescribed[(r['original_phase'],family,r['method'],target)].append(r)
        if not r['verified']: continue
        for point in r['checkpoints']:
            if point['target_seconds'] is not None:
                grouped[(r['original_phase'],family,r['method'],point['target_seconds'],r['instance'])].append(point)
    across=defaultdict(list)
    for key,points in grouped.items():
        across[key[:-1]].append(dict(instance=key[-1],population=len(points),x=statistics.median(p['original_solver_seconds'] for p in points),primary=statistics.median(p['momentum_error_frobenius'] for p in points),stale=statistics.median(p['stale_current_output_error_frobenius'] for p in points),points=points))
    summaries=[]
    for key,expected in sorted(prescribed.items()):
        instances=across.get(key,[])
        phase,family,method,target=key
        # Fixed BMMC has stochastic/initialization dispersion, not dataset IQR.
        samples=([p for instance in instances for p in instance['points']] if family=='bmmc' else instances)
        observed_runs=sum(x['population'] for x in instances)
        summaries.append(dict(phase=phase,family=family,method=method,target_seconds=target,prescribed_run_count=len(expected),prescribed_instance_count=len({r['instance'] for r in expected}),observed_run_count=observed_runs,observed_instance_count=len(instances),coverage_complete=observed_runs==len(expected),x=interval([x['original_solver_seconds'] if family=='bmmc' else x['x'] for x in samples]),primary=interval([x['momentum_error_frobenius'] if family=='bmmc' else x['primary'] for x in samples]),stale=interval([x['stale_current_output_error_frobenius'] if family=='bmmc' else x['stale'] for x in samples])))
    return summaries


def make_plots(runs, out):
    summaries=curve_summaries(runs)
    if not any(r['observed_run_count'] for r in summaries): return [], summaries
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'smoothing-fw':'#0072B2','smoothing-fw-exact':'#56B4E9','smoothing-row-projection':'#882255'}
    figures=[]
    for phase in sorted({r['phase'] for r in summaries}):
        fig,axes=plt.subplots(2,3,figsize=(13,7),squeeze=False)
        for column,family in enumerate(('synthetic_clean','synthetic_noisy','bmmc')):
            for method in sorted(METHODS):
                points=sorted((r for r in summaries if r['phase']==phase and r['family']==family and r['method']==method and r['observed_run_count']),key=lambda r:r['target_seconds'])
                if not points: continue
                x=[p['x']['median'] for p in points]
                for row,metric in enumerate(('primary','stale')):
                    ax=axes[row,column]; ax.plot(x,[p[metric]['median'] for p in points],label=method_name(method),color=colors[method])
                    for point in points:
                        ax.plot(point['x']['median'],point[metric]['median'],marker='o',linestyle='none',color=colors[method],markerfacecolor=colors[method] if point['coverage_complete'] else 'white')
                    if all(p[metric]['count']>1 for p in points): ax.fill_between(x,[p[metric]['q25'] for p in points],[p[metric]['q75'] for p in points],alpha=.15,color=colors[method])
            for row in range(2):
                axes[row,column].set_title(family.replace('_',' '));axes[row,column].set_xlabel('Original observed solver time (s)');axes[row,column].set_xscale('log');axes[row,column].grid(alpha=.25)
                if not axes[row,column].lines: axes[row,column].text(.5,.5,'No verified observations',ha='center',va='center',transform=axes[row,column].transAxes,color='0.4')
        axes[0,0].set_ylabel('Last-input momentum error\n(Frobenius)');axes[1,0].set_ylabel('Stale current-output error\n(Frobenius)')
        legend={}
        for ax in axes.flat:
            handles,labels=ax.get_legend_handles_labels();legend.update(zip(labels,handles))
        fig.legend(legend.values(),legend.keys(),loc='lower center',ncol=1,fontsize=9)
        fig.suptitle(phase+' — supplementary replay diagnostics\nPrimary error is at P_(K−1), output clock labels P_K. Open markers: incomplete prescribed population.',fontsize=11)
        fig.text(.5,.13,'Synthetic: within-instance medians, then across-problem IQR. BMMC: fixed-instance replicate IQR.\nAbsent windows are omitted; observed/prescribed run and instance counts are in curve_summaries.csv.',ha='center',fontsize=9)
        fig.tight_layout(rect=(0,.18,1,.91))
        for suffix in ('png','pdf'):
            name=f'{phase}_momentum_errors.{suffix}';fig.savefig(out/name,dpi=160);figures.append(name)
        plt.close(fig)
    return figures,summaries


def check_scientific_coverage(plans):
    """Bind every scientific cell to the original frozen manifest, not just totals."""
    original_paths = {'tuning': 'configs/stage_c_manifests/tuning.json',
                      'final_product': 'configs/stage_c_final_manifests/final_product.json',
                      'final_randomized': 'configs/stage_c_final_manifests/final_randomized.json'}
    originals = {phase: {r['run_id']:r for r in read_json(ROOT/path)['runs'] if r['method'] in METHODS}
                 for phase,path in original_paths.items()}
    for plan in plans:
        if plan['phase'] == 'pilot':
            cfg=read_json(ROOT/'configs/stage_c_pilot.json')
            expected={f"pilot/{Path(name).stem}/{method}" for name in cfg['instances'] for method in ('smoothing-fw','smoothing-fw-exact')}
            require({r['run_id'] for r in plan['runs']}==expected,'Pilot scientific cells differ.')
            for trial in plan['runs']:
                require(trial['original_phase']=='pilot' and trial['original_run_spec'] is None,'Pilot spec misclassified.')
                require(trial['run_id']==f"pilot/{Path(trial['instance']).stem}/{trial['method']}",'Pilot identity differs.')
            continue
        phases=('tuning',) if plan['phase']=='tuning' else ('final_product','final_randomized')
        expected={rid:spec for phase in phases for rid,spec in originals[phase].items()}
        require({r['run_id'] for r in plan['runs']}==set(expected),'Prescribed scientific cell coverage differs.')
        for trial in plan['runs']:
            spec=expected[trial['run_id']]; phase=trial['run_id'].split('/')[0]
            require(trial['original_phase']==phase and trial['original_run_spec']==spec,'Original frozen scientific spec differs.')
            require(trial['original_manifest_sha256']==digest(ROOT/original_paths[phase]),'Original scientific manifest binding differs.')
            require(trial['method']==spec['method'] and trial['instance']==spec['instance'] and trial['instance_sha256']==spec['instance_sha256'],'Scientific method/input identity differs.')


def summarize(pairs, out, plots=True):
    summaries=[];runs=[];plans=[]
    for manifest,root in pairs:
        summary,records,plan=inspect_phase(manifest,root);summaries.append(summary);runs.extend(records);plans.append(plan)
    require(Counter(p['phase'] for p in plans)==Counter(EXPECTED_COUNTS.keys()), 'Supply exactly one tuning, pilot, and finals manifest/root pair.')
    require(all(p['run_count']==EXPECTED_COUNTS[p['phase']] for p in plans), 'Prescribed replay count differs from 504+6+286.')
    all_ids=[r['run_id'] for r in runs]; require(len(all_ids)==len(set(all_ids)), 'Duplicate scientific run IDs across manifests.')
    originals=[t['original_run_dir'] for p in plans for t in p['runs']];require(len(originals)==len(set(originals)), 'Original trajectory replicated across manifests.')
    for plan in plans:
        for name,hash_value in plan['scientific_inputs'].items():
            require(digest(ROOT/safe_id(name))==hash_value, 'Frozen scientific manifest/config hash differs: '+name)
    check_scientific_coverage(plans)
    errors=[error for p in summaries for error in p['errors']]
    provenance=not errors and all(r['provenance_valid'] for r in runs)
    execution_complete=all(r['terminal'] for r in runs)
    complete=execution_complete and provenance
    counts=Counter(r['status'] for r in runs)
    out=Path(out).resolve()
    require(all(out!=Path(root).resolve() and out not in Path(root).resolve().parents and Path(root).resolve() not in out.parents for _,root in pairs), 'Report output must be disjoint from companion roots.')
    out.mkdir(parents=True,exist_ok=False)
    figures,curves=make_plots(runs,out) if plots and provenance else ([],curve_summaries(runs) if provenance else [])
    elapsed=[r['controller_elapsed_seconds'] for r in runs if r['terminal']]
    report=dict(schema_version=1,generated_utc=datetime.now(timezone.utc).isoformat(),status=('complete_verified' if complete and counts['verified']==len(runs) else 'complete_with_unverified_replays' if complete else 'partial_or_invalid'),execution_complete=execution_complete,provenance_valid=provenance,all_diagnostics_verified=complete and counts['verified']==len(runs),prescribed_runs=len(runs),terminal_runs=sum(r['terminal'] for r in runs),verified_runs=counts['verified'],unverified_terminal_runs=counts['failed'],status_counts=dict(counts),manifest_pairs=summaries,issues=errors,runs=runs,observed_verified_checkpoints=sum(len(r['checkpoints']) for r in runs),missed_original_windows_in_verified_runs=sum(len(r.get('missed_original_checkpoints',[])) for r in runs if r['verified']),controller_elapsed_seconds_sum=sum(elapsed),controller_elapsed_seconds_distribution=interval(elapsed),replay_cost_policy='Sum controller_elapsed_seconds once per terminal replay. Worker total and initialization/update/diagnostic subtotals are nested and are not added again. This is summed process elapsed, not GPU kernel time or campaign wall time.',point_policy='At original output K>=1 primary error pairs actual d_(K-1) with grad f(P_(K-1)); stale tracking uses grad f(P_K). K=0 uses d0/P0. No terminal oracle extension.',uncertainty_policy='Synthetic plots: within-instance median over prescribed stochastic/initialization runs, then across-instance median/IQR. BMMC: fixed-instance stochastic/initialization median/IQR. Deterministic duplicate starts are not new replicates. Observed and prescribed run/instance counts and coverage_complete are saved per target, including empty windows. Open markers show incomplete populations; partial populations may differ.',comparison_policy='Supplementary diagnostics only; no performance ranking, selection changes, retuning or theorem-rate claim. Original timing and optimization outcomes stay authoritative.',figures=figures,curve_summaries=curves,script_sha256=digest(__file__),command=sys.argv)
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    curve_fields=['phase','family','method','target_seconds','prescribed_run_count','prescribed_instance_count','observed_run_count','observed_instance_count','coverage_complete','x','primary','stale']
    with (out/'curve_summaries.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=curve_fields);writer.writeheader()
        for curve in curves: writer.writerow({key:json.dumps(curve[key]) if isinstance(curve[key],dict) else curve[key] for key in curve_fields})
    fields=['run_id','original_phase','method','instance','seed','init_seed','status','original_status','terminal','verified','provenance_valid','observed_checkpoint_count','controller_elapsed_seconds','failure_kind','error','issues']
    with (out/'runs.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for run in runs: writer.writerow({key:json.dumps(run.get(key)) if isinstance(run.get(key),(list,dict)) else run.get(key) for key in fields})
    lines=['# Supplementary momentum replay report','',f"**{report['status']}**: {report['terminal_runs']}/{len(runs)} prescribed replays terminal; {counts['verified']} verified, {counts['failed']} terminal unverified. Provenance valid: {provenance}. This does not change original optimization status or establish full-study completion.",'',report['point_policy'],'',f"Verified observed checkpoints: {report['observed_verified_checkpoints']}; explicitly missed original windows in verified runs: {report['missed_original_windows_in_verified_runs']}. Missing or rejected replays contribute no error curves.",'',f"Summed controller elapsed cost: {sum(elapsed):.6g} seconds across {len(elapsed)} terminal replays. Worker totals and their nested subtotals are retained in JSON and are not added again. Original solver/output clocks remain the diagnostic observation labels.",'','| Replay phase | Prescribed | Terminal | Verified | Unverified |','|---|---:|---:|---:|---:|']
    for p in plans:
        ids={x['run_id'] for x in p['runs']};selected=[r for r in runs if r['run_id'] in ids]
        lines.append(f"| {p['phase']} | {len(selected)} | {sum(r['terminal'] for r in selected)} | {sum(r['verified'] for r in selected)} | {sum(r['status']=='failed' for r in selected)} |")
    lines += ['',report['uncertainty_policy'],'',report['comparison_policy'],'','All prescribed records, failures and integrity issues are retained in [runs.csv](runs.csv) and [report.json](report.json).']
    if errors or any(r['issues'] for r in runs): lines += ['','Integrity issues:']+['- '+e for e in errors]+['- '+r['run_id']+': '+'; '.join(r['issues']) for r in runs if r['issues']]
    if figures: lines += ['','Figures:'] + ['- ['+name+']('+name+')' for name in figures] + ['','Window populations and statistics: [curve_summaries.csv](curve_summaries.csv).']
    if not figures:lines += ['','No final diagnostic figures were generated: verified final observations are absent, plotting was disabled, or provenance is invalid.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',action='append',required=True,type=Path);parser.add_argument('--root',action='append',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path);parser.add_argument('--require-complete',action='store_true');parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args();require(len(args.manifest)==len(args.root),'Pair each manifest with exactly one root in order.')
    report=summarize(list(zip(args.manifest,args.root)),args.out,plots=not args.no_plots)
    print(json.dumps({k:report[k] for k in ('status','prescribed_runs','terminal_runs','verified_runs','provenance_valid')}))
    return 1 if args.require_complete and not (report['execution_complete'] and report['provenance_valid']) else 0


if __name__=='__main__': raise SystemExit(main())
