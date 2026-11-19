"""Prepare or run the development-screened SPRING correction and update its report.

--prepare is stdlib-only and never starts an optimizer. The default action waits
for feasibility_retune to finish, runs the frozen 84-case evaluation, validates
all native traces, then invokes the separately audited report publication helper.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
import math
import multiprocessing
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'convincing_4x_recovery'
DEPENDENCY = ROOT.parent / 'feasibility_retune' / 'progress.json'
PYTHON = '/home/tony/zofw_experiments/.venv/bin/python'
DEPS = '/home/tony/frames_ct_2detect_20261104/deps'
CPUS = (0, 2, 4, 6)
CAP = 69_120
THREAD_VARS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
METHOD_PARAMS = {'spring_sarah': ('rate', 'refresh_period'),
                 'prox_sarah': ('rate', 'inner_steps', 'snapshot_policy'),
                 'palm': ('rate',)}
COUNTERS = ('forward_views', 'backward_views', 'gradient_calls', 'projection_calls',
            'projection_inner_iterations', 'projection_dykstra_calls', 'projection_floor_count')
PROBLEM_FIELDS = ('data_sha256', 'calibration_sha256', 'caps', 'curvature',
                  'train_view_ids', 'validation_view_ids', 'initialization',
                  'constraint_geometry', 'initial_factor_frobenius_norm',
                  'numpy', 'astra', 'factor_dtype', 'projector')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, allow_nan=False, indent=2) + '\n')
    temp.replace(path)


def frozen(name, value):
    path = ROOT / name
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Frozen input differs: {path}')
    else:
        write(path, value)


def select_baselines():
    """Screen saved development endpoints; do not inspect evaluation outcomes."""
    source = read(OLD / 'baseline_selection.json')
    scores, selected, runner_up, counts = {}, dict(source['selected']), {}, {}
    for method, params in METHOD_PARAMS.items():
        rows = []
        for candidate in source['scores'][method]:
            row = dict(candidate)
            endpoints = [r for cell in row['cases'] for r in cell.get('seed_cases', [cell])]
            expected = 4 if method == 'palm' else 12
            if len(endpoints) != expected or len(row['cases']) != 4:
                raise ValueError(f'Incomplete development candidate for {method}')
            cells = {(c['slice_id'], c['n_train']) for c in row['cases']}
            if cells != set(itertools.product(('slice04300', 'slice03150'), (360, 1080))):
                raise ValueError('Development cells changed')
            failures = []
            for endpoint in endpoints:
                ratio = endpoint.get('relative_loss')
                valid = (endpoint.get('status') == 'completed' and
                         isinstance(ratio, (int, float)) and math.isfinite(ratio) and 0 <= ratio <= 1)
                if not valid:
                    failures.append({k: endpoint.get(k) for k in ('run_id', 'status', 'relative_loss')})
            recomputed = statistics.median(statistics.median(e['relative_loss']
                for e in c.get('seed_cases', [c])) for c in row['cases'])
            if not math.isclose(recomputed, row['median_relative_loss'], rel_tol=1e-14):
                raise ValueError('Saved median score is inconsistent')
            row.update(original_selection_eligible=row['selection_eligible'],
                       selection_eligible=bool(row['selection_eligible'] and not failures),
                       endpoint_count=len(endpoints), failed_endpoint_count=len(failures),
                       failed_endpoints=failures,
                       max_relative_loss=max(e['relative_loss'] for e in endpoints))
            rows.append(row)
        eligible = sorted((r for r in rows if r['selection_eligible']),
                          key=lambda r: (r['median_relative_loss'], *(r[k] for k in params)))
        if not eligible:
            raise ValueError(f'No eligible development candidate: {method}')
        def compact(row):
            return {k: row[k] for k in (*params, 'median_relative_loss', 'max_relative_loss', 'endpoint_count')}
        selected[method] = compact(eligible[0])
        runner_up[method] = compact(eligible[1]) if len(eligible) > 1 else None
        counts[method] = dict(total=len(rows), eligible=len(eligible), rejected=len(rows)-len(eligible))
        scores[method] = rows
    if (selected['spring_sarah']['rate'], selected['spring_sarah']['refresh_period']) != (1.6, 32):
        raise ValueError('Unexpected development winner; this correction plan must be reviewed')
    unchanged = {m: all(selected[m][k] == source['selected'][m][k] for k in METHOD_PARAMS[m])
                 for m in ('prox_sarah', 'palm')}
    if not all(unchanged.values()):
        raise ValueError('The frozen plan expects ProxSARAH and PALM to remain unchanged')
    return dict(status='selected', source_selection=str(OLD / 'baseline_selection.json'),
                source_selection_sha256=digest(OLD / 'baseline_selection.json'),
                selection_uses_evaluation_outcomes=False,
                development_view_budget=source['development_view_budget'],
                rule=['Require every available development endpoint to be completed, finite and f/f0 <= 1.',
                      'Use all 12 stochastic endpoints; PALM has four deterministic endpoints.',
                      'Among eligible candidates retain the original median of four within-cell seed medians.',
                      'Break ties lexicographically by the method parameters.',
                      'Saved repaired loss is the objective used in the original baseline selection; these projected baselines have feasible iterates.'],
                selected=selected, runner_up=runner_up, original_selected=source['selected'],
                unchanged=unchanged, counts=counts, scores=scores)


def prepare():
    numerical = read(OLD / 'compatibility_manifest.json')['current_source_sha256']
    for name, expected in numerical.items():
        if digest(OLD / name) != expected:
            raise ValueError(f'Recovery source changed: {name}')
    for name in (*numerical, 'compatibility_manifest.json'):
        destination = ROOT / name
        if destination.exists():
            if digest(destination) != digest(OLD / name):
                raise ValueError(f'Existing copy differs: {destination}')
        else:
            shutil.copy2(OLD / name, destination)
    for name in ('data', 'calibration'):
        destination = ROOT / name
        if destination.is_symlink():
            if destination.resolve() != (OLD / name).resolve():
                raise ValueError(f'Input symlink differs: {destination}')
        elif destination.exists():
            raise ValueError(f'Refusing to replace existing input directory: {destination}')
        else:
            destination.symlink_to(OLD / name, target_is_directory=True)
    frozen('selection.json', select_baselines())
    original = [c for c in read(OLD / 'evaluation_plan.json')['runs'] if c['variant'] == 'spring_sarah']
    cohort = read(OLD / 'cohort.json')
    expected = set(itertools.product(cohort['evaluation_slices'], (360, 1080), (202, 203, 204)))
    if len(original) != 84 or {(c['slice_id'], c['n_train'], c['seed']) for c in original} != expected:
        raise ValueError('Original paired evaluation cohort changed')
    configs, assignments, matched, inputs, anchors = [], {}, {}, {}, {}
    recovered = read(OLD / 'manifest.json')
    for index, old in enumerate(original):
        cfg = {k: v for k, v in old.items() if 'continuation' not in k and 'checkpoint_import' not in k}
        cfg.update(rate=1.6, refresh_period=32, trial='a1p6_q32',
                   run_id=f'evaluation_{old["slice_id"]}_{old["n_train"]}_spring_sarah_seed{old["seed"]}_a1p6_q32',
                   view_budget=CAP, record_every_iteration=True)
        if cfg['phase'] != 'evaluation' or cfg['batch'] != 64 or cfg['calibration_seed'] != 201:
            raise ValueError('Unexpected original numerical configuration')
        configs.append(cfg)
        assignments[cfg['run_id']] = CPUS[index % len(CPUS)]
        matched[cfg['run_id']] = old['run_id']
        for rel, expected_hash in (
                (f'data/{cfg["slice_id"]}.npz', recovered['data'][cfg['slice_id']]),
                (f'calibration/{cfg["slice_id"]}_seed201.npz', recovered['calibration'][cfg['slice_id']])):
            expected_hash = expected_hash['sha256'] if isinstance(expected_hash, dict) else expected_hash
            if rel not in inputs:
                if digest(ROOT / rel) != expected_hash:
                    raise ValueError(f'Input differs from original: {rel}')
                inputs[rel] = expected_hash
        for filename in ('metadata.json', 'trace.json'):
            rel = f'runs/{old["run_id"]}/{filename}'
            anchors[rel] = digest(OLD / rel)
    plan = dict(expected_runs=84, fresh=True, view_budget_per_direction=CAP,
                endpoint_rule='Last native observation with forward_views and backward_views <= 69120; retain the final crossing observation separately.',
                selection_sha256=digest(ROOT / 'selection.json'), runs=configs,
                assignments=assignments, matched_run_ids=matched,
                groups=cohort['evaluation_slice_groups'])
    frozen('evaluation_plan.json', plan)
    protocol = dict(design='Fresh matched SPRING evaluation after development endpoint eligibility screening',
        host=os.uname().nodename, python=PYTHON, dependency_paths=[DEPS], cpus=list(CPUS), blas_threads=1,
        campaign_sha256=digest(Path(__file__)), numerical_sources=numerical,
        compatibility_manifest_sha256=digest(ROOT / 'compatibility_manifest.json'),
        selection_sha256=digest(ROOT / 'selection.json'), evaluation_plan_sha256=digest(ROOT / 'evaluation_plan.json'),
        matched_root=str(OLD), matched_files={name:digest(OLD / name) for name in
            ('manifest.json', 'baseline_selection.json', 'evaluation_plan.json', 'cohort.json')},
        inputs=dict(sorted(inputs.items())), matched_run_files=dict(sorted(anchors.items())),
        selection_uses_evaluation_outcomes=False, fresh_runs=True, no_timing_measurements=True,
        same_factors_data_views_and_seeds=True,
        random_stream_note='Changing refresh probability changes random-number consumption; seed pairing does not imply identical minibatches.',
        resource_dependency=dict(path=str(DEPENDENCY), required_status='completed', polling_seconds=60),
        failure_policy='Retain all evidence and halt queued work after a run/validation error. No automatic parameter changes.',
        report_publication=dict(helper='update_report.py', arguments=['--publish'],
            after='All 84 dense runs completed and validation_summary.json frozen',
            helper_hash='Frozen in publication_helper.json at launch and checked before publication'))
    frozen('protocol.json', protocol)
    verify_prepared()
    if not (ROOT / 'progress.json').exists():
        update('prepared', 'awaiting_launch', completed=0, total=84)
    return dict(status='prepared', expected_runs=84, selected=read(ROOT / 'selection.json')['selected'],
                protocol_sha256=digest(ROOT / 'protocol.json'))


def verify_prepared():
    protocol = read(ROOT / 'protocol.json')
    local = dict(protocol['numerical_sources'])
    local.update({'campaign.py':protocol['campaign_sha256'],
                  'compatibility_manifest.json':protocol['compatibility_manifest_sha256'],
                  'selection.json':protocol['selection_sha256'],
                  'evaluation_plan.json':protocol['evaluation_plan_sha256']})
    local.update(protocol['inputs'])
    for rel, expected in local.items():
        if digest(ROOT / rel) != expected:
            raise ValueError(f'Frozen campaign input changed: {rel}')
    for rel, expected in {**protocol['matched_files'], **protocol['matched_run_files']}.items():
        if digest(OLD / rel) != expected:
            raise ValueError(f'Matched reference changed: {rel}')
    if read(ROOT / 'selection.json') != select_baselines():
        raise ValueError('Development selection does not recompute')
    return protocol


def validate_result(cfg, cpu, plan, protocol):
    folder = ROOT / 'runs' / cfg['run_id']
    metadata, result, trace = (read(folder / name) for name in ('metadata.json', 'result.json', 'trace.json'))
    reference = OLD / 'runs' / plan['matched_run_ids'][cfg['run_id']]
    previous = read(reference / 'metadata.json')
    if (result['status'] != 'completed' or result['config'] != cfg or metadata['cpu'] != cpu
            or any(result.get(k) != v for k, v in metadata.items())
            or metadata['source_sha256'] != protocol['numerical_sources']
            or metadata['compatibility_manifest_sha256'] != protocol['compatibility_manifest_sha256']
            or metadata['recovery_origin'] != 'fresh' or metadata['continuation'] is not None
            or metadata['imported_checkpoint'] is not None
            or any(metadata[k] != previous[k] for k in PROBLEM_FIELDS)):
        raise ValueError(f'Run provenance differs: {cfg["run_id"]}')
    expected_optimizer = dict(batch=64, eta_u=1.6/metadata['curvature'][0],
                              eta_v=1.6/metadata['curvature'][1], refresh_period=32)
    if metadata['optimizer'] != expected_optimizer:
        raise ValueError('Optimizer parameters differ from selected development setting')
    if (not trace or result['final'] != trace[-1] or result['iteration'] != len(trace)-1
            or [t['iteration'] for t in trace] != list(range(len(trace)))
            or trace[0] != read(reference / 'trace.json')[0]):
        raise ValueError(f'Native trace or matched initialization differs: {cfg["run_id"]}')
    for i, row in enumerate(trace):
        metrics = ('raw_loss', 'feasible_loss', 'validation_loss', 'reference_relative_error',
                   'd_distance', 'c_violation', 'repair_distance', 'max_projection_distance_bound')
        if (not all(isinstance(row[k], (int,float)) and math.isfinite(row[k]) and row[k] >= 0 for k in metrics)
                or row['forward_views'] != row['backward_views'] or row['projection_calls'] != 2*i
                or row['d_distance'] != 0 or row['c_violation'] > 1e-10
                or not math.isclose(row['raw_loss'], row['feasible_loss'], abs_tol=1e-12, rel_tol=1e-10)
                or any('seconds' in k for k in row)):
            raise ValueError(f'Invalid observation: {cfg["run_id"]}, iteration {i}')
        if i:
            delta = row['forward_views'] - trace[i-1]['forward_views']
            gradients = row['gradient_calls'] - trace[i-1]['gradient_calls']
            allowed = {(2*cfg['n_train'],2)} if i == 1 else {
                (2*cfg['n_train'],2), (cfg['n_train']+128,3), (256,4)}
            if (delta,gradients) not in allowed or any(row[k] < trace[i-1][k] for k in COUNTERS):
                raise ValueError('Native work counters are inconsistent')
    within = [t for t in trace if t['forward_views'] <= CAP and t['backward_views'] <= CAP]
    if trace[-1]['forward_views'] < CAP or len(trace)-len(within) > 1:
        raise ValueError('Trace did not finish exactly at the first cap crossing')
    initial, endpoint = trace[0], within[-1]
    return dict(run_id=cfg['run_id'], matched_run_id=plan['matched_run_ids'][cfg['run_id']],
        source_directory=str(folder), slice_id=cfg['slice_id'], group=plan['groups'][cfg['slice_id']],
        n_train=cfg['n_train'], seed=cfg['seed'], cpu=cpu,
        observations=len(trace), within_budget_observations=len(within),
        initial=initial, endpoint=endpoint, final_observation=trace[-1],
        cap_slack=CAP-endpoint['forward_views'], final_overshoot=max(0,trace[-1]['forward_views']-CAP),
        initial_loss=initial['feasible_loss'], raw_loss=endpoint['raw_loss'],
        repaired_loss=endpoint['feasible_loss'], relative_loss=endpoint['feasible_loss']/initial['feasible_loss'],
        relative_raw_loss=endpoint['raw_loss']/initial['raw_loss'], d_distance=endpoint['d_distance'],
        initial_factor_frobenius_norm=metadata['initial_factor_frobenius_norm'],
        max_observed_relative_loss=max(t['raw_loss']/initial['raw_loss'] for t in within),
        endpoint_exceeds_initial=endpoint['raw_loss'] > initial['raw_loss'],
        metadata_sha256=digest(folder/'metadata.json'), trace_sha256=digest(folder/'trace.json'),
        result_sha256=digest(folder/'result.json'), matched_initial_observation=True, provenance_valid=True)


def update(status, phase, **extras):
    value = dict(status=status, phase=phase, updated_utc=now(), pid=os.getpid(), **extras)
    write(ROOT / 'progress.json', value)
    print(json.dumps(value, separators=(',', ':')), flush=True)


def worker(cfg, cpu):
    # Numerical dependencies are deliberately imported only after resource gating.
    for name in THREAD_VARS:
        os.environ[name] = '1'
    sys.path.insert(0, DEPS)
    os.sched_setaffinity(0, {cpu})
    from run import run
    return run(cfg, cpu=cpu)


def execute():
    protocol = verify_prepared()
    plan = read(ROOT / 'evaluation_plan.json')
    if os.uname().nodename != protocol['host'] or Path(sys.executable).absolute() != Path(PYTHON):
        raise ValueError(f'Launch on the prepared host with {PYTHON}')
    if not set(CPUS).issubset(os.sched_getaffinity(0)):
        raise ValueError('The launch CPU affinity excludes a frozen worker CPU')
    abort = ROOT / 'execution_abort.json'
    if abort.exists() and read(abort).get('active'):
        raise RuntimeError('Prior failure must be reviewed and execution_abort.json cleared explicitly before resuming')
    helper = ROOT / 'update_report.py'
    if not helper.is_file():
        raise RuntimeError('Report publication helper must be ready before launching the queue')
    frozen('publication_helper.json',dict(path=str(helper),sha256=digest(helper)))
    state = None
    while True:
        dependency = read(DEPENDENCY) if DEPENDENCY.exists() else {'status':'missing'}
        if dependency.get('status') == 'completed':
            break
        current = (dependency.get('status'), dependency.get('phase'))
        if current != state:
            update('waiting_for_dependency', 'resource_wait', dependency=str(DEPENDENCY),
                   dependency_status=dependency.get('status'), dependency_phase=dependency.get('phase'), total=84)
            state = current
        if abort.exists() and read(abort).get('active'):
            raise RuntimeError('Campaign abort requested while waiting')
        time.sleep(60)
    frozen('dependency_completion.json', dict(path=str(DEPENDENCY), status='completed',
           progress_sha256=digest(DEPENDENCY), progress=dependency))
    verify_prepared()
    done, jobs = {}, []
    for cfg in plan['runs']:
        cpu = plan['assignments'][cfg['run_id']]
        if (ROOT/'runs'/cfg['run_id']/'result.json').exists():
            done[cfg['run_id']] = validate_result(cfg, cpu, plan, protocol)
        else:
            jobs.append((cfg,cpu))
    update('running', 'evaluation', completed=len(done), total=84)
    if jobs:
        for name in THREAD_VARS:
            os.environ[name] = '1'
        context = multiprocessing.get_context('spawn')
        with ExitStack() as stack:
            pools = {cpu:stack.enter_context(ProcessPoolExecutor(max_workers=1, mp_context=context)) for cpu in CPUS}
            futures = {pools[cpu].submit(worker,cfg,cpu):(cfg,cpu) for cfg,cpu in jobs}
            for future in as_completed(futures):
                cfg,cpu = futures[future]
                try:
                    future.result()
                    done[cfg['run_id']] = validate_result(cfg,cpu,plan,protocol)
                except Exception as exc:
                    write(abort, dict(active=True,run_id=cfg['run_id'],error=repr(exc),utc=now()))
                    for other in futures:
                        other.cancel()
                    raise
                update('running','evaluation',completed=len(done),total=84,last_completed=cfg['run_id'])
    cases = [done[cfg['run_id']] for cfg in plan['runs']]
    summary = dict(status='completed_validated', expected_runs=84, completed_runs=len(cases),
        protocol_sha256=digest(ROOT/'protocol.json'), selection_sha256=digest(ROOT/'selection.json'),
        evaluation_plan_sha256=digest(ROOT/'evaluation_plan.json'),
        dependency_completion_sha256=digest(ROOT/'dependency_completion.json'),
        endpoint_rule=plan['endpoint_rule'], view_budget_per_direction=CAP,
        endpoint_exceeds_initial_count=sum(c['endpoint_exceeds_initial'] for c in cases),
        max_endpoint_relative_loss=max(c['relative_raw_loss'] for c in cases),
        no_scientific_claim_of_monotonicity=True, cases=cases)
    frozen('validation_summary.json',summary)
    update('evaluated','evaluation_validated',completed=84,total=84,
           validation_summary=str(ROOT/'validation_summary.json'),
           validation_summary_sha256=digest(ROOT/'validation_summary.json'))
    if digest(helper) != read(ROOT/'publication_helper.json')['sha256']:
        raise ValueError('Report helper changed after queue launch')
    subprocess.run([sys.executable,str(helper),'--publish'],check=True,cwd=ROOT)
    update('completed','report_published',completed=84,total=84,
           validation_summary=str(ROOT/'validation_summary.json'),
           validation_summary_sha256=digest(ROOT/'validation_summary.json'),
           publication_helper_sha256=digest(helper))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--prepare',action='store_true',help='freeze inputs and selection; never run optimizers')
    modes.add_argument('--verify',action='store_true',help='verify frozen inputs only; never run optimizers')
    args = parser.parse_args()
    with (ROOT/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.prepare:
            print(json.dumps(prepare(),indent=2))
        elif args.verify:
            verify_prepared()
            print(json.dumps(dict(status='verified',protocol_sha256=digest(ROOT/'protocol.json'))))
        else:
            try:
                execute()
            except Exception as exc:
                update('failed','execution',error=repr(exc),traceback=traceback.format_exc())
                raise


if __name__ == '__main__':
    main()
