"""Prepare and queue 84 fresh SPRING runs with the paper's practical local steps.

--prepare and --verify use only the standard library and never run optimizers.
Default execution waits for feasibility_retune before starting four workers,
validates inclusive work accounting, and invokes the audited report helper.
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
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent/'convincing_4x_recovery'
DEPENDENCY = ROOT.parent/'feasibility_retune'/'progress.json'
PYTHON = '/home/tony/zofw_experiments/.venv/bin/python'
DEPS = '/home/tony/frames_ct_2detect_20261104/deps'
CPUS = (0, 2, 4, 6)
CAP = 69_120
THREAD_VARS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')
UNCHANGED_SOURCES = ('ct_problem.py', 'methods.py', 'spectral_reference_core.py',
                     'spectral_reference_baseline.py')
SOURCES = ('run.py', 'local_curvature.py', *UNCHANGED_SOURCES)
COUNTERS = ('forward_views', 'backward_views', 'gradient_calls', 'oracle_calls',
            'curvature_operator_calls', 'curvature_forward_views', 'curvature_backward_views',
            'projection_calls', 'projection_inner_iterations', 'projection_dykstra_calls',
            'projection_floor_count')
PROBLEM_FIELDS = ('data_sha256', 'calibration_sha256', 'caps', 'curvature',
                  'train_view_ids', 'validation_view_ids', 'initialization',
                  'constraint_geometry', 'initial_factor_frobenius_norm',
                  'numpy', 'astra', 'factor_dtype', 'projector')
PARAMS = dict(batch=64, refresh_period=16, power_iterations=5, step_coefficient=0.5)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,allow_nan=False,indent=2)+'\n')
    temp.replace(path)


def frozen(name,value):
    path = ROOT/name
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Frozen input differs: {path}')
    else:
        write(path,value)


def specification():
    return dict(status='frozen',algorithm='SPRING-SARAH with current minibatch block-curvature steps',
        parameter_source='Published practical step rule; inherited loopless refresh setting',
        optimizer=PARAMS,
        rule='eta_U=0.5/Lhat_U(V_k); eta_V=0.5/Lhat_V(U_{k+1})',
        curvature_estimator='Fresh normalized Gaussian factor-space direction per block; five normalized Hessian applications, then the norm of a sixth application.',
        curvature_sample='64 views without replacement per block. Reuse the SARAH difference batch on recursive updates; draw a separate 64-view curvature batch on full-gradient refreshes.',
        recursion='Independent loopless full-gradient refresh with probability 1/16 per block; both first estimates are full. No initial SGD epoch or deterministic epoch refresh.',
        random_stream='Gradient RNG retains the historical seed+100000 stream. A separately checkpointed curvature RNG draws refresh minibatches and fresh power directions.',
        curvature_normalization='Hessian of the mean squared data-fit objective with the same view-and-detector normalization as problem.grad.',
        zero_curvature='Raise FloatingPointError for a zero or nonfinite estimate; do not silently floor, cap, damp, or change parameters.',
        theoretical_scope='Published practical numerical step heuristic adapted to CT, not a certificate of the theorem step-size conditions or a literal reproduction of the separate epoch-based SNMF driver.',
        tuning_performed=False,endpoint_exclusion_screen=False,evaluation_outcomes_used_for_parameters=False,
        work=dict(forward_views='All estimator and curvature forward views',
                  backward_views='All estimator and curvature adjoint views',
                  gradient_calls='Literal gradient evaluations, including both recursive difference terms',
                  curvature_operator_calls='One Hessian-vector application using one forward and one adjoint minibatch operation',
                  oracle_calls='gradient_calls + curvature_operator_calls',
                  curvature_applications_per_iteration=12,curvature_views_per_direction_per_iteration=768),
        baseline_comparison='Other saved methods have no additional curvature work during optimization; their oracle count remains the saved gradient_calls.')


def prepare():
    reference = read(OLD/'compatibility_manifest.json')['current_source_sha256']
    for name in UNCHANGED_SOURCES:
        if digest(OLD/name) != reference[name]:
            raise ValueError(f'Reference numerical source changed: {name}')
        destination = ROOT/name
        if not destination.exists():
            shutil.copy2(OLD/name,destination)
        if digest(destination) != reference[name]:
            raise ValueError(f'Problem/projection/reference solver changed: {name}')
    for name in ('data','calibration'):
        destination = ROOT/name
        if not destination.exists() and not destination.is_symlink():
            destination.symlink_to(OLD/name,target_is_directory=True)
        if not destination.is_symlink() or destination.resolve() != (OLD/name).resolve():
            raise ValueError(f'Input must be the matched reference symlink: {destination}')
    frozen('specification.json',specification())
    numerical = {name:digest(ROOT/name) for name in SOURCES}
    tests = read(ROOT/'implementation_validation.json')
    if tests.get('status') != 'passed' or tests.get('source_sha256') != numerical:
        raise ValueError('Current local-curvature implementation lacks its bounded validation')
    implementation = dict(current_source_sha256=numerical,
        reference_root=str(OLD),reference_source_sha256=reference,
        unchanged_source_sha256={name:reference[name] for name in UNCHANGED_SOURCES},
        authorized_changes={
            'local_curvature.py':'New SPRING factory: current alternating block curvature, fixed paper coefficient, separate curvature RNG, charged Hessian applications.',
            'run.py':'New fresh-only recording engine: local factory dispatch, dense step details, inclusive work accounting, source and checkpoint validation.'},
        specification_sha256=digest(ROOT/'specification.json'),
        implementation_validation_sha256=digest(ROOT/'implementation_validation.json'),
        matching_rule='Problem, initialization and unchanged projection code must match. The new SPRING algorithm source is explicitly different.')
    frozen('implementation_manifest.json',implementation)
    original = [c for c in read(OLD/'evaluation_plan.json')['runs'] if c['variant']=='spring_sarah']
    cohort = read(OLD/'cohort.json')
    expected = set(itertools.product(cohort['evaluation_slices'],(360,1080),(202,203,204)))
    if len(original)!=84 or {(c['slice_id'],c['n_train'],c['seed']) for c in original} != expected:
        raise ValueError('Original paired evaluation cohort changed')
    configs,assignments,matched,inputs,anchors = [],{},{},{},{}
    recovered = read(OLD/'manifest.json')
    for index,old in enumerate(original):
        cfg = {k:v for k,v in old.items() if k not in ('rate','refresh_period')
               and 'continuation' not in k and 'checkpoint_import' not in k}
        cfg.update(optimizer=dict(PARAMS),algorithm='spring_sarah_local_curvature',trial='localcurvature_c0p5_q16',
            run_id=f'evaluation_{old["slice_id"]}_{old["n_train"]}_spring_sarah_seed{old["seed"]}_localcurvature_c0p5_q16',
            view_budget=CAP,record_every_iteration=True)
        if cfg['phase']!='evaluation' or cfg['batch']!=64 or cfg['calibration_seed']!=201:
            raise ValueError('Unexpected original numerical configuration')
        configs.append(cfg)
        assignments[cfg['run_id']] = CPUS[index%len(CPUS)]
        matched[cfg['run_id']] = old['run_id']
        for rel,expected_hash in (
            (f'data/{cfg["slice_id"]}.npz',recovered['data'][cfg['slice_id']]),
            (f'calibration/{cfg["slice_id"]}_seed201.npz',recovered['calibration'][cfg['slice_id']])):
            expected_hash = expected_hash['sha256'] if isinstance(expected_hash,dict) else expected_hash
            if rel not in inputs:
                if digest(ROOT/rel) != expected_hash:
                    raise ValueError(f'Input differs from original: {rel}')
                inputs[rel] = expected_hash
        for name in ('metadata.json','trace.json'):
            rel = f'runs/{old["run_id"]}/{name}'
            anchors[rel] = digest(OLD/rel)
    plan = dict(expected_runs=84,fresh=True,view_budget_per_direction=CAP,
        endpoint_rule='Last native observation whose inclusive forward and backward views are <=69120; retain the final crossing observation separately.',
        specification_sha256=digest(ROOT/'specification.json'),runs=configs,assignments=assignments,
        matched_run_ids=matched,groups=cohort['evaluation_slice_groups'])
    frozen('evaluation_plan.json',plan)
    protocol = dict(design='Fresh paired SPRING evaluation with the published practical current-curvature step rule',
        host=os.uname().nodename,python=PYTHON,dependency_paths=[DEPS],cpus=list(CPUS),blas_threads=1,
        campaign_sha256=digest(Path(__file__)),numerical_sources=numerical,
        specification_sha256=digest(ROOT/'specification.json'),
        implementation_manifest_sha256=digest(ROOT/'implementation_manifest.json'),
        implementation_validation_sha256=digest(ROOT/'implementation_validation.json'),
        evaluation_plan_sha256=digest(ROOT/'evaluation_plan.json'),
        matched_root=str(OLD),matched_files={name:digest(OLD/name) for name in
            ('manifest.json','evaluation_plan.json','cohort.json','compatibility_manifest.json')},
        inputs=dict(sorted(inputs.items())),matched_run_files=dict(sorted(anchors.items())),
        tuning_performed=False,endpoint_exclusion_screen=False,evaluation_outcomes_used_for_parameters=False,
        fresh_runs=True,no_timing_measurements=True,same_factors_data_views_and_seeds=True,
        resource_dependency=dict(path=str(DEPENDENCY),required_status='completed',polling_seconds=60),
        failure_policy='Retain evidence and halt queued work after a run or validation error; no automatic step modifications.',
        report_publication=dict(helper='update_report.py',arguments=['--publish'],
            after='All 84 native traces validated, including curvature work',
            helper_hash='Frozen at queue launch in publication_helper.json'))
    frozen('protocol.json',protocol)
    verify_prepared()
    if not (ROOT/'progress.json').exists():
        update('prepared','awaiting_launch',completed=0,total=84)
    return dict(status='prepared',expected_runs=84,optimizer=PARAMS,protocol_sha256=digest(ROOT/'protocol.json'))


def verify_prepared():
    protocol = read(ROOT/'protocol.json')
    local = dict(protocol['numerical_sources'])
    local.update({'campaign.py':protocol['campaign_sha256'],
        'specification.json':protocol['specification_sha256'],
        'implementation_manifest.json':protocol['implementation_manifest_sha256'],
        'implementation_validation.json':protocol['implementation_validation_sha256'],
        'evaluation_plan.json':protocol['evaluation_plan_sha256']})
    local.update(protocol['inputs'])
    for rel,expected in local.items():
        if digest(ROOT/rel)!=expected:
            raise ValueError(f'Frozen campaign input changed: {rel}')
    for rel,expected in {**protocol['matched_files'],**protocol['matched_run_files']}.items():
        if digest(OLD/rel)!=expected:
            raise ValueError(f'Matched reference changed: {rel}')
    if read(ROOT/'specification.json') != specification():
        raise ValueError('Frozen algorithm specification differs')
    return protocol


def validate_result(cfg,cpu,plan,protocol):
    folder = ROOT/'runs'/cfg['run_id']
    metadata,result,trace = (read(folder/name) for name in ('metadata.json','result.json','trace.json'))
    reference = OLD/'runs'/plan['matched_run_ids'][cfg['run_id']]
    previous = read(reference/'metadata.json')
    implementation = read(ROOT/'implementation_manifest.json')
    if (result['status']!='completed' or result['config']!=cfg or metadata['cpu']!=cpu
        or any(result.get(k)!=v for k,v in metadata.items())
        or metadata['source_sha256']!=protocol['numerical_sources']
        or metadata['implementation_manifest_sha256']!=protocol['implementation_manifest_sha256']
        or metadata['specification_sha256']!=protocol['specification_sha256']
        or metadata['authorized_implementation_changes']!=implementation['authorized_changes']
        or metadata['recovery_origin']!='fresh' or metadata['continuation'] is not None
        or metadata['imported_checkpoint'] is not None
        or any(metadata[k]!=previous[k] for k in PROBLEM_FIELDS)
        or metadata['optimizer']!=PARAMS or cfg['optimizer']!=PARAMS):
        raise ValueError(f'Run provenance differs: {cfg["run_id"]}')
    for name in UNCHANGED_SOURCES:
        if metadata['source_sha256'][name]!=implementation['unchanged_source_sha256'][name]:
            raise ValueError('An unchanged problem/projection source differs')
    old_initial = read(reference/'trace.json')[0]
    if (not trace or result['final']!=trace[-1] or result['iteration']!=len(trace)-1
        or [t['iteration'] for t in trace]!=list(range(len(trace)))
        or any(trace[0][k]!=v for k,v in old_initial.items())
        or any(trace[0][k]!=0 for k in COUNTERS) or trace[0]['last_step'] is not None):
        raise ValueError(f'Native trace or matched initialization differs: {cfg["run_id"]}')
    for i,row in enumerate(trace):
        metrics = ('raw_loss','feasible_loss','validation_loss','reference_relative_error',
                   'd_distance','c_violation','repair_distance','max_projection_distance_bound')
        if (not all(isinstance(row[k],(int,float)) and math.isfinite(row[k]) and row[k]>=0 for k in metrics)
            or row['forward_views']!=row['backward_views'] or row['projection_calls']!=2*i
            or row['oracle_calls']!=row['gradient_calls']+row['curvature_operator_calls']
            or row['curvature_operator_calls']!=12*i
            or row['curvature_forward_views']!=768*i or row['curvature_backward_views']!=768*i
            or row['d_distance']!=0 or row['c_violation']>1e-10
            or not math.isclose(row['raw_loss'],row['feasible_loss'],abs_tol=1e-12,rel_tol=1e-10)
            or any('seconds' in k for k in row)):
            raise ValueError(f'Invalid native observation: {cfg["run_id"]}, iteration {i}')
        if not i:
            continue
        step = row['last_step']
        if any(row[k]-trace[i-1][k]!=step[k] for k in COUNTERS):
            raise ValueError('Recorded per-step and cumulative work disagree')
        refreshes = int(step['refresh_u'])+int(step['refresh_v'])
        expected_gradients = 4-refreshes
        expected_views = refreshes*cfg['n_train']+(2-refreshes)*128+768
        if (i==1 and refreshes!=2 or step['forward_views']!=expected_views
            or step['gradient_calls']!=expected_gradients
            or any(row[k]<trace[i-1][k] for k in COUNTERS)):
            raise ValueError('Inclusive estimator/curvature work counters are inconsistent')
        for block in ('u','v'):
            curvature,eta = step[f'curvature_{block}'],step[f'eta_{block}']
            if (not math.isfinite(curvature) or curvature<=0 or not math.isfinite(eta) or eta<=0
                or step[f'curvature_sample_views_{block}']!=64
                or not math.isclose(eta*curvature,.5,rel_tol=1e-13)):
                raise ValueError('Recorded step does not implement the paper curvature rule')
    within = [row for row in trace if max(row['forward_views'],row['backward_views'])<=CAP]
    if trace[-1]['forward_views']<CAP or len(trace)-len(within)>1:
        raise ValueError('Trace did not finish at the first inclusive cap crossing')
    initial,endpoint = trace[0],within[-1]
    return dict(run_id=cfg['run_id'],matched_run_id=plan['matched_run_ids'][cfg['run_id']],
        source_directory=str(folder),slice_id=cfg['slice_id'],group=plan['groups'][cfg['slice_id']],
        n_train=cfg['n_train'],seed=cfg['seed'],cpu=cpu,observations=len(trace),within_budget_observations=len(within),
        initial=initial,endpoint=endpoint,final_observation=trace[-1],cap_slack=CAP-endpoint['forward_views'],
        final_overshoot=max(0,trace[-1]['forward_views']-CAP),initial_loss=initial['feasible_loss'],
        raw_loss=endpoint['raw_loss'],repaired_loss=endpoint['feasible_loss'],
        relative_loss=endpoint['feasible_loss']/initial['feasible_loss'],
        relative_raw_loss=endpoint['raw_loss']/initial['raw_loss'],d_distance=endpoint['d_distance'],
        initial_factor_frobenius_norm=metadata['initial_factor_frobenius_norm'],
        max_observed_relative_loss=max(row['raw_loss']/initial['raw_loss'] for row in within),
        endpoint_exceeds_initial=endpoint['raw_loss']>initial['raw_loss'],
        metadata_sha256=digest(folder/'metadata.json'),trace_sha256=digest(folder/'trace.json'),
        result_sha256=digest(folder/'result.json'),matched_initial_observation=True,provenance_valid=True)


def update(status,phase,**extras):
    record = dict(status=status,phase=phase,updated_utc=now(),pid=os.getpid(),**extras)
    write(ROOT/'progress.json',record)
    print(json.dumps(record,separators=(',',':')),flush=True)


def worker(cfg,cpu):
    for name in THREAD_VARS:
        os.environ[name]='1'
    sys.path.insert(0,DEPS)
    os.sched_setaffinity(0,{cpu})
    from run import run
    return run(cfg,cpu=cpu)


def execute():
    protocol = verify_prepared()
    plan = read(ROOT/'evaluation_plan.json')
    if os.uname().nodename!=protocol['host'] or Path(sys.executable).absolute()!=Path(PYTHON):
        raise ValueError(f'Launch on the prepared host with {PYTHON}')
    if not set(CPUS).issubset(os.sched_getaffinity(0)):
        raise ValueError('Launch affinity excludes a frozen worker CPU')
    abort = ROOT/'execution_abort.json'
    if abort.exists() and read(abort).get('active'):
        raise RuntimeError('Prior failure must be reviewed and abort flag cleared before resuming')
    helper = ROOT/'update_report.py'
    if not helper.is_file():
        raise RuntimeError('Report helper must be ready before launching the queue')
    frozen('publication_helper.json',dict(path=str(helper),sha256=digest(helper)))
    state = None
    while True:
        dependency = read(DEPENDENCY) if DEPENDENCY.exists() else {'status':'missing'}
        if dependency.get('status')=='completed':
            break
        current = (dependency.get('status'),dependency.get('phase'))
        if current!=state:
            update('waiting_for_dependency','resource_wait',dependency=str(DEPENDENCY),
                   dependency_status=dependency.get('status'),dependency_phase=dependency.get('phase'),total=84)
            state=current
        if abort.exists() and read(abort).get('active'):
            raise RuntimeError('Campaign abort requested while waiting')
        time.sleep(60)
    frozen('dependency_completion.json',dict(path=str(DEPENDENCY),status='completed',
           progress_sha256=digest(DEPENDENCY),progress=dependency))
    verify_prepared()
    done,jobs = {},[]
    for cfg in plan['runs']:
        cpu=plan['assignments'][cfg['run_id']]
        if (ROOT/'runs'/cfg['run_id']/'result.json').exists():
            done[cfg['run_id']]=validate_result(cfg,cpu,plan,protocol)
        else:
            jobs.append((cfg,cpu))
    update('running','evaluation',completed=len(done),total=84)
    if jobs:
        for name in THREAD_VARS:
            os.environ[name]='1'
        context=multiprocessing.get_context('spawn')
        with ExitStack() as stack:
            pools={cpu:stack.enter_context(ProcessPoolExecutor(max_workers=1,mp_context=context)) for cpu in CPUS}
            futures={pools[cpu].submit(worker,cfg,cpu):(cfg,cpu) for cfg,cpu in jobs}
            for future in as_completed(futures):
                cfg,cpu=futures[future]
                try:
                    future.result()
                    done[cfg['run_id']]=validate_result(cfg,cpu,plan,protocol)
                except Exception as exc:
                    write(abort,dict(active=True,run_id=cfg['run_id'],error=repr(exc),utc=now()))
                    for other in futures:
                        other.cancel()
                    raise
                update('running','evaluation',completed=len(done),total=84,last_completed=cfg['run_id'])
    cases=[done[cfg['run_id']] for cfg in plan['runs']]
    summary=dict(status='completed_validated',expected_runs=84,completed_runs=len(cases),
        protocol_sha256=digest(ROOT/'protocol.json'),specification_sha256=digest(ROOT/'specification.json'),
        implementation_manifest_sha256=digest(ROOT/'implementation_manifest.json'),
        evaluation_plan_sha256=digest(ROOT/'evaluation_plan.json'),
        dependency_completion_sha256=digest(ROOT/'dependency_completion.json'),
        endpoint_rule=plan['endpoint_rule'],view_budget_per_direction=CAP,curvature_work_included=True,
        endpoint_exceeds_initial_count=sum(c['endpoint_exceeds_initial'] for c in cases),
        max_endpoint_relative_loss=max(c['relative_raw_loss'] for c in cases),
        no_endpoint_exclusion=True,no_scientific_claim_of_monotonicity=True,cases=cases)
    frozen('validation_summary.json',summary)
    update('evaluated','evaluation_validated',completed=84,total=84,
        validation_summary=str(ROOT/'validation_summary.json'),
        validation_summary_sha256=digest(ROOT/'validation_summary.json'))
    if digest(helper)!=read(ROOT/'publication_helper.json')['sha256']:
        raise ValueError('Report helper changed after queue launch')
    subprocess.run([sys.executable,str(helper),'--publish'],check=True,cwd=ROOT)
    update('completed','report_published',completed=84,total=84,
        validation_summary=str(ROOT/'validation_summary.json'),
        validation_summary_sha256=digest(ROOT/'validation_summary.json'),publication_helper_sha256=digest(helper))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare',action='store_true')
    mode.add_argument('--verify',action='store_true')
    args=parser.parse_args()
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


if __name__=='__main__':
    main()
