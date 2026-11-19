"""Fixed joint loss/feasibility retune and paired CT evaluation on Elite."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback

from run import run, read_json, sha256, write_json

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'convincing_4x_recovery'
CPUS = (0, 2, 4, 6)
DEV_CAP, EVAL_CAP = 34_560, 69_120
DEV_SLICES = ('slice04300', 'slice03150')
VIEW_COUNTS = (360, 1080)
DEV_SEEDS = (110, 111, 112)
ANCHOR = (16., .1)
PAIRS = tuple(itertools.product((.25, 1., 2., 4.), (.05, .075, .1))) + (ANCHOR,)
TARGETS = {'primary': (.3, .01), 'strict': (.3, .001),
           'loose': (.4, .01), 'loose_strict': (.4, .001)}


def now():
    return datetime.now(timezone.utc).isoformat()


def frozen(name, value):
    path = ROOT / name
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f'Frozen input changed: {path}')
    else:
        write_json(path, value)


def config(phase, slice_id, n_train, seed, pair, budget, dense):
    beta, gamma = pair
    trial = f'b{beta:g}_g{gamma:g}'.replace('.', 'p')
    return dict(run_id=f'{phase}_{slice_id}_{n_train}_sframes_seed{seed}_{trial}',
                phase=phase, slice_id=slice_id, n_train=n_train,
                method='sframes', variant='sframes', trial=trial, seed=seed,
                calibration_seed=110 if phase == 'development' else 201,
                grid=512, rank=64, batch=64, view_budget=budget,
                record_every_iteration=dense, beta_scale=beta, rate=gamma)


def freeze_protocol():
    numerical = read_json(ROOT / 'compatibility_manifest.json')['current_source_sha256']
    if any(sha256(ROOT / name) != digest for name, digest in numerical.items()):
        raise ValueError('Frozen numerical sources have changed')
    recovered = read_json(OLD / 'manifest.json')
    if any(numerical[name] != recovered['sources'][name] for name in numerical if name != 'run.py'):
        raise ValueError('Solver or projector differs from the matched comparison')
    old_report = read_json(OLD / 'reports' / 'report_update.json')
    if sha256(Path(old_report['report_pdf'])) != old_report['report_pdf_sha256']:
        if not (ROOT / 'report_update.json').is_file():
            raise ValueError('Live report differs from the completed reference paper')
    files = {}
    for slice_id, expected in recovered['data'].items():
        npz = ROOT / 'data' / f'{slice_id}.npz'
        if sha256(npz) != expected:
            raise ValueError(f'Data differs from the matched comparison: {npz}')
        files[str(npz.relative_to(ROOT))] = expected
        seed = 110 if slice_id in DEV_SLICES else 201
        calibration = ROOT / 'calibration' / f'{slice_id}_seed{seed}.npz'
        files[str(calibration.relative_to(ROOT))] = sha256(calibration)
        if seed == 201:
            expected = recovered['calibration'][slice_id]
            if isinstance(expected, dict):
                expected = expected['sha256']
            if files[str(calibration.relative_to(ROOT))] != expected:
                raise ValueError(f'Calibration differs from the matched comparison: {calibration}')
        else:
            former = ROOT.parent / 'calibration' / calibration.name
            if files[str(calibration.relative_to(ROOT))] != sha256(former):
                raise ValueError(f'Calibration differs from baseline tuning: {calibration}')
    files = dict(sorted(files.items()))
    frozen('protocol.json', dict(
        design='joint beta_0/gamma_0 development sweep; single locked minibatch setting evaluated on existing cohort',
        host=os.uname().nodename, python=sys.executable, cpus=list(CPUS), blas_threads=1,
        numerical_sources=numerical, campaign_sha256=sha256(Path(__file__)),
        compatibility_manifest_sha256=sha256(ROOT / 'compatibility_manifest.json'),
        matched_recovery={name: sha256(OLD / name) for name in (
            'manifest.json', 'evaluation_plan.json', 'cohort.json', 'targets.json',
            'evaluation_provenance_audit.json', 'reports/summary.json', 'reports/report_update.json')},
        inputs=files,
        development=dict(slices=list(DEV_SLICES), training_views=list(VIEW_COUNTS),
                         seeds=list(DEV_SEEDS), calibration_seed=110, view_budget=DEV_CAP,
                         batch=64, pairs=[list(x) for x in PAIRS], observations='initial and endpoint'),
        evaluation=dict(view_budget=EVAL_CAP, observations='every native iteration including zero',
                        selection_uses_evaluation_outcomes=False,
                        comparison='saved minibatch SFRAMES and ProxSARAH at equal per-direction views',
                        cohort='existing 14 evaluation slices, not used in parameter selection'),
        targets={name: list(vals) for name, vals in TARGETS.items()},
        selection_rule=[
            'Maximize number of final joint (30% repaired-to-initial loss, 1% normalized raw orthant distance) hits across 12 development cases.',
            'If all candidates have zero hits, minimize median joint normalized threshold ratio (max of loss ratio / 0.3 and distance ratio / 0.01).',
            'Break remaining ties by lower median repaired-to-initial loss, then lower median normalized raw orthant distance, then lexicographic (c_beta, gamma0).',
            'Each median takes seed medians within slice/training size, then a median over the four slice/size cells. 0.1% and 40% targets are sensitivity checks.',
        ]))


def smoke():
    notes = []
    parent_plan = read_json(OLD / 'evaluation_plan.json')['runs']
    for n in VIEW_COUNTS:
        previous = next(c for c in parent_plan if c['n_train'] == n and
                        c['variant'] == 'sframes' and c['seed'] == 202)
        cfg = config('smoke', previous['slice_id'], n, 202, ANCHOR, 256, True)
        run(cfg, cpu=CPUS[0], runs_dir=ROOT / 'smoke_runs')
        sample = read_json(ROOT / 'smoke_runs' / cfg['run_id'] / 'trace.json')
        reference = read_json(OLD / 'runs' / previous['run_id'] / 'trace.json')[:5]
        if len(sample) != 5 or sample != reference:
            raise ValueError(f'Fresh anchor smoke differs from the saved native prefix for N={n}')
        notes.append(dict(run_id=cfg['run_id'], matched_reference=previous['run_id'],
                          native_observations=len(sample), same=True))
    cfg = config('development', DEV_SLICES[0], 1080, 110, PAIRS[0], 13 * 64, False)
    run(cfg, cpu=CPUS[0], runs_dir=ROOT / 'smoke_runs')
    sparse = read_json(ROOT / 'smoke_runs' / cfg['run_id'] / 'trace.json')
    if [e['iteration'] for e in sparse] != [0, 13]:
        raise ValueError('Sparse development smoke did not save exactly both endpoints')
    notes.append(dict(run_id=cfg['run_id'], native_observations=len(sparse), same=True))
    frozen('smoke_validation.json', notes)


def validate_result(cfg, cpu):
    folder = ROOT / 'runs' / cfg['run_id']
    result = read_json(folder / 'result.json')
    trace = read_json(folder / 'trace.json')
    protocol = read_json(ROOT / 'protocol.json')
    if (result['status'] != 'completed' or result['config'] != cfg or result['cpu'] != cpu
            or result['source_sha256'] != protocol['numerical_sources']
            or result['compatibility_manifest_sha256'] != protocol['compatibility_manifest_sha256']
            or result['data_sha256'] != protocol['inputs'][f'data/{cfg["slice_id"]}.npz']
            or result['calibration_sha256'] != protocol['inputs'][f'calibration/{cfg["slice_id"]}_seed{cfg["calibration_seed"]}.npz']
            or result['optimizer'] != {'batch': 64, 'gamma0': cfg['rate'], 'rho0': 1.,
                                        'mu0': cfg['beta_scale'] / max(result['curvature'])}
            or result['recovery_origin'] != 'fresh' or result['final'] != trace[-1]):
        raise ValueError(f'Invalid provenance or endpoint for {cfg["run_id"]}')
    steps = cfg['view_budget'] // cfg['batch']
    expected = list(range(steps + 1)) if cfg['record_every_iteration'] else [0, steps]
    if result['iteration'] != steps or [e['iteration'] for e in trace] != expected:
        raise ValueError(f'Missing or unexpected observations for {cfg["run_id"]}')
    for t in trace:
        if (t['forward_views'] != t['backward_views']
                or t['forward_views'] != t['iteration'] * cfg['batch']
                or t['gradient_calls'] != t['iteration']
                or not all(math.isfinite(t[k]) and t[k] >= 0 for k in
                           ('raw_loss', 'feasible_loss', 'd_distance', 'validation_loss', 'reference_relative_error'))
                or t['c_violation'] > 1e-10):
            raise ValueError(f'Invalid native observation in {cfg["run_id"]} at {t["iteration"]}')
    init, end = trace[0], trace[-1]
    return dict(run_id=cfg['run_id'], slice_id=cfg['slice_id'], n_train=cfg['n_train'],
                seed=cfg['seed'], forward_views=end['forward_views'], backward_views=end['backward_views'],
                observations=len(trace), iteration=end['iteration'],
                initial_loss=init['feasible_loss'], raw_loss=end['raw_loss'], repaired_loss=end['feasible_loss'],
                initial_factor_frobenius_norm=result['initial_factor_frobenius_norm'],
                d_distance=end['d_distance'],
                relative_loss=end['feasible_loss'] / init['feasible_loss'],
                relative_d_distance=end['d_distance'] / result['initial_factor_frobenius_norm'])


def update(status, phase, **extras):
    record = dict(status=status, phase=phase, updated_utc=now(), pid=os.getpid(), **extras)
    write_json(ROOT / 'progress.json', record)
    print(json.dumps(record, separators=(',', ':')), flush=True)


def run_phase(phase, configs):
    if not configs:
        raise ValueError('Empty campaign phase')
    loads = dict.fromkeys(CPUS, 0)
    jobs, done = [], {}
    for cfg in configs:
        cpu = min(loads, key=loads.get)
        steps = cfg['view_budget'] // cfg['batch']
        probes = steps + 1 if cfg['record_every_iteration'] else 2
        loads[cpu] += 2 * cfg['view_budget'] + probes * (2 * cfg['n_train'] + 360)
        folder = ROOT / 'runs' / cfg['run_id']
        if (folder / 'result.json').is_file():
            done[cfg['run_id']] = validate_result(cfg, cpu)
        else:
            jobs.append((cfg, cpu))
    total = len(configs)
    update('running', phase, completed=len(done), total=total)
    if jobs:
        ctx = multiprocessing.get_context('spawn')
        with ExitStack() as stack:
            pools = {cpu: stack.enter_context(ProcessPoolExecutor(max_workers=1, mp_context=ctx))
                     for cpu in CPUS}
            futures = {pools[cpu].submit(run, cfg, cpu): (cfg, cpu) for cfg, cpu in jobs}
            for future in as_completed(futures):
                cfg, cpu = futures[future]
                try:
                    future.result()
                    item = validate_result(cfg, cpu)
                except Exception as exc:
                    write_json(ROOT / 'execution_abort.json', dict(active=True, phase=phase,
                               run_id=cfg['run_id'], error=repr(exc), utc=now()))
                    for other in futures:
                        other.cancel()
                    raise
                done[cfg['run_id']] = item
                update('running', phase, completed=len(done), total=total, last_completed=item)
    return [done[cfg['run_id']] for cfg in configs]


def aggregate(cases, value):
    return statistics.median(statistics.median(value(c) for c in cases
                             if c['slice_id'] == sid and c['n_train'] == n)
                             for sid, n in itertools.product(DEV_SLICES, VIEW_COUNTS))


def select(configs, cases):
    by_run = {c['run_id']: c for c in cases}
    rows = []
    for beta, gamma in PAIRS:
        subset = [by_run[c['run_id']] for c in configs if (c['beta_scale'], c['rate']) == (beta, gamma)]
        if len(subset) != 12:
            raise ValueError('Missing development endpoint')
        row = dict(beta_scale=beta, gamma0=gamma, n_cases=len(subset),
                   median_relative_loss=aggregate(subset, lambda c: c['relative_loss']),
                   median_relative_d_distance=aggregate(subset, lambda c: c['relative_d_distance']),
                   median_joint_threshold_ratio=aggregate(subset, lambda c: max(
                       c['relative_loss'] / TARGETS['primary'][0], c['relative_d_distance'] / TARGETS['primary'][1])),
                   cases=subset)
        for name, (rho, epsilon) in TARGETS.items():
            row[f'{name}_hits'] = sum(c['relative_loss'] <= rho and c['relative_d_distance'] <= epsilon for c in subset)
        rows.append(row)
    zero_hit_fallback = all(r['primary_hits'] == 0 for r in rows)
    selected = min(rows, key=lambda r: (-r['primary_hits'],
                   r['median_joint_threshold_ratio'] if zero_hit_fallback else 0.,
                   r['median_relative_loss'], r['median_relative_d_distance'], r['beta_scale'], r['gamma0']))
    selection = dict(status='selected', protocol_sha256=sha256(ROOT / 'protocol.json'),
                     development_view_budget=DEV_CAP, zero_hit_fallback=zero_hit_fallback,
                     selected={k: v for k, v in selected.items() if k != 'cases'}, scores=rows)
    frozen('selection.json', selection)
    return selection['selected']


def evaluate(selected):
    old_plan = read_json(OLD / 'evaluation_plan.json')
    pair = (selected['beta_scale'], selected['gamma0'])
    current = [c for c in old_plan['runs'] if c['variant'] == 'sframes']
    prox = [c for c in old_plan['runs'] if c['variant'] == 'prox_sarah']
    keys = lambda cs: {(c['slice_id'], c['n_train'], c['seed']) for c in cs}
    if len(current) != len(prox) or len(current) != 84 or keys(current) != keys(prox) or len(keys(current)) != 84:
        raise ValueError('Historical paired evaluation cohort changed')
    cfgs = [config('evaluation', c['slice_id'], c['n_train'], c['seed'], pair, EVAL_CAP, True) for c in current]
    frozen('evaluation_plan.json', dict(selection_sha256=sha256(ROOT / 'selection.json'),
                                       comparison_root=str(OLD), expected_runs=len(cfgs), runs=cfgs,
                                       current_sframes_run_ids=[c['run_id'] for c in current],
                                       prox_sarah_run_ids=[c['run_id'] for c in prox]))
    run_phase('evaluation', cfgs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true', help='Freeze protocol and validate small fresh runs only')
    args = p.parse_args()
    with (ROOT / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            freeze_protocol()
            write_json(ROOT / 'execution_abort.json', dict(active=False))
            if args.prepare:
                smoke()
                update('ready', 'smoke_validated')
                return
            if not (ROOT / 'smoke_validation.json').is_file():
                raise ValueError('Run --prepare before starting the full sweep')
            configs = [config('development', sid, n, seed, pair, DEV_CAP, False)
                       for pair, sid, n, seed in itertools.product(PAIRS, DEV_SLICES, VIEW_COUNTS, DEV_SEEDS)]
            frozen('development_plan.json', dict(protocol_sha256=sha256(ROOT / 'protocol.json'),
                                                expected_runs=len(configs), runs=configs))
            cases = run_phase('development', configs)
            chosen = select(configs, cases)
            update('running', 'setting_locked', selection=chosen)
            evaluate(chosen)
            update('running', 'analysis_and_report')
            subprocess.run([sys.executable, str(ROOT / 'analysis.py'), '--root', str(ROOT)], check=True)
            subprocess.run([sys.executable, str(ROOT / 'report.py'), '--root', str(ROOT), '--publish'], check=True)
            audit = read_json(ROOT / 'report_update.json')
            update('completed', 'report_published', selection=chosen, report=audit)
        except Exception as exc:
            update('failed', 'campaign_halted', error=repr(exc))
            traceback.print_exc()
            raise SystemExit(1)


if __name__ == '__main__':
    main()
