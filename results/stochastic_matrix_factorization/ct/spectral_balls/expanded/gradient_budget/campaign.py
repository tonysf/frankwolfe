"""Continue PALM and SPRING to the same gradient-only view cap as ProxSARAH.

The optimizer equations, parameters, complete saved prefix and random streams
are retained. Only the stopping budget changes; curvature remains charged in
its own counters and in the existing inclusive counters.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'baseline_step_rules'
SPRING=ROOT.parent/'spring_localcurvature'
PRESENTATION=ROOT.parent/'trajectory_mean_sd'
REFERENCE=ROOT.parent/'convincing_4x_recovery'
PYTHON='/home/tony/zofw_experiments/.venv/bin/python'
DEPS='/home/tony/frames_ct_2detect_20261104/deps'
CPUS=(0,2,4,6)
CAP=69120
SOURCES=('run.py','palm_paper.py','local_curvature.py','curvature_reference.py','ct_problem.py',
         'methods.py','spectral_reference_core.py','spectral_reference_baseline.py')
THREAD_VARS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS')
COUNTERS=('forward_views','backward_views','gradient_calls','oracle_calls','curvature_operator_calls',
          'curvature_forward_views','curvature_backward_views','projection_calls',
          'projection_inner_iterations','projection_dykstra_calls','projection_floor_count')
PROBLEM_FIELDS=('data_sha256','calibration_sha256','caps','curvature','train_view_ids','validation_view_ids',
                'initialization','constraint_geometry','initial_factor_frobenius_norm','numpy','astra','factor_dtype','projector')


def now():return datetime.now(timezone.utc).isoformat()
def read(path):return json.loads(Path(path).read_text())
def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)
def frozen(name,value):
    path=ROOT/name
    if path.exists():
        if read(path)!=value:raise ValueError(f'Frozen input changed: {path}')
    else:write(path,value)
def gradient_views(row,direction='forward'):
    return row[direction+'_views']-row['curvature_'+direction+'_views']


def specification():
    return dict(status='frozen',physical_runs=168,methods=['palm','spring_sarah'],training_views=[360,1080],seeds=[202,203,204],
        gradient_view_budget_per_direction=CAP,original_inclusive_view_budget_per_direction=CAP,
        stopping_rule='Continue until cumulative forward views minus curvature forward views reaches 69120; forward and adjoint counts must agree.',
        endpoint_rule='Last native iterate with both gradient-only directional view counts <=69120; exclude an overshooting final update.',
        continuation='Restore each completed source checkpoint, including factor state, gradient-estimator buffers, both RNG states and its complete dense prefix. The previously excluded inclusive-cap overshoot remains a valid prefix iterate under the new gradient cap.',
        recording='Initialization and every native update; exact original prefix preserved; no interpolation.',
        optimizer_changes=False,parameter_retuning=False,prox_sarah_or_sframes_reruns=False,
        palm=dict(step='1/Lhat',curvature='Current full-data block curvature; five normalized power applications and sixth norm evaluation',
                  gradient_views_per_update='2N',curvature_views_per_update='12N',iterations_at_gradient_cap={'360':96,'1080':32}),
        spring_sarah=dict(step='0.5/Lhat',batch=64,refresh_period=16,
                          curvature='Current sampled block curvature; five normalized power applications and sixth norm evaluation',
                          curvature_views_per_update=768,gradient_views_per_update='Sum across two blocks: N for full refresh, otherwise 2*64 for the recursive difference'),
        work=dict(gradient_views='forward_views-curvature_forward_views (and separately backward)',
                  curvature_views='Unchanged cumulative curvature_forward_views/curvature_backward_views',
                  inclusive_views='Unchanged forward_views/backward_views, including all gradient and curvature work',
                  gradient_calls='Literal batch-gradient evaluations',oracle_calls='gradient_calls+curvature_operator_calls'),
        publication='Update the existing two mean-plus/minus-SD figures and comparison table; preserve corrected ProxSARAH and SFRAMES results; no new figures.',
        failure_policy='Retain evidence and halt; no failed-run filtering or evaluation-dependent parameter changes.',optimizer_timing=False)


def source_records():
    records=[]
    for folder,method in ((BASE,'palm'),(SPRING,'spring_sarah')):
        validation=read(folder/'validation_summary.json')
        if validation['status']!='completed_validated':raise ValueError('Predecessor validation incomplete')
        cases=[c for c in validation['cases'] if c.get('method','spring_sarah')==method]
        if len(cases)!=84:raise ValueError('Wrong predecessor cohort size')
        for case in cases:
            origin=Path(case['source_directory']);meta=read(origin/'metadata.json');trace=read(origin/'trace.json')
            result=read(origin/'result.json')
            for filename,key in (('metadata.json','metadata_sha256'),('trace.json','trace_sha256'),('result.json','result_sha256')):
                if digest(origin/filename)!=case[key]:raise ValueError(f'Validated origin changed: {origin/filename}')
            if (meta['config']['method']!=method or result['status']!='completed' or result['final']!=trace[-1]
                or not (origin/'checkpoint.pkl').is_file() or gradient_views(trace[-1])>=CAP):
                raise ValueError('Origin is not an eligible completed checkpoint')
            if [row['iteration'] for row in trace]!=list(range(len(trace))):raise ValueError('Origin trace is not dense')
            if max(trace[-1]['forward_views'],trace[-1]['backward_views'])<CAP:raise ValueError('Origin stopped before inclusive cap')
            for name,expected in meta['source_sha256'].items():
                if digest(folder/name)!=expected:raise ValueError('Original numerical implementation changed')
            records.append((method,case,meta,trace))
    if len({(m,c['slice_id'],c['n_train'],c['seed']) for m,c,_,_ in records})!=168:raise ValueError('Duplicated source case')
    return records


def make_plan(records,groups):
    configs=[];estimated={};origins={}
    for method,case,meta,trace in records:
        original=meta['config'];folder=Path(case['source_directory'])
        cfg={k:original[k] for k in ('phase','method','slice_id','n_train','seed','grid','rank','calibration_seed')}
        cfg.update(run_id='gradient_extension_'+original['run_id'],record_every_iteration=True,gradient_view_budget=CAP,
            origin_directory=str(folder),origin_run_id=original['run_id'],origin_config=original,
            **{f'origin_{key}_sha256':digest(folder/filename) for key,filename in
               (('checkpoint','checkpoint.pkl'),('metadata','metadata.json'),('trace','trace.json'),('result','result.json'))})
        configs.append(cfg);last=trace[-1];remaining=CAP-gradient_views(last)
        if method=='palm':steps=math.ceil(remaining/(2*cfg['n_train']));per_iteration=14*cfg['n_train']
        else:
            average_gradient=gradient_views(last)/last['iteration'];steps=math.ceil(remaining/average_gradient)
            per_iteration=average_gradient+768
        estimated[cfg['run_id']]=2*steps*per_iteration+(steps+1)*(cfg['n_train']+360)
        origins[cfg['run_id']]=dict(group=groups[cfg['slice_id']],origin_directory=str(folder),
            origin_iteration=last['iteration'],origin_observations=len(trace),origin_gradient_views=gradient_views(last),
            origin_case=case)
    configs.sort(key=lambda c:(c['slice_id'],c['n_train'],c['seed'],c['method']))
    loads=dict.fromkeys(CPUS,0);assignments={}
    for cfg in sorted(configs,key=lambda c:(-estimated[c['run_id']],c['run_id'])):
        cpu=min(loads,key=loads.get);assignments[cfg['run_id']]=cpu;loads[cpu]+=estimated[cfg['run_id']]
    return dict(phase='evaluation',expected_runs=168,runs=configs,assignments=assignments,groups=groups,origins=origins,
        estimated_worker_loads=loads,gradient_view_budget_per_direction=CAP,original_inclusive_view_budget_per_direction=CAP,
        specification_sha256=digest(ROOT/'specification.json'),endpoint_rule=specification()['endpoint_rule'])


def audit_origins(plan):
    """Check all prepared checkpoint states without invoking any CT operator."""
    for name in THREAD_VARS:os.environ[name]='1'
    sys.path.insert(0,DEPS)
    from run import origin
    import numpy as np
    cases=[]
    for cfg in plan['runs']:
        meta,state=origin(cfg)
        solver=state['solver']
        if (not np.isfinite(solver['U']).all() or not np.isfinite(solver['V']).all()
            or 'rng' not in solver or 'curvature_rng' not in solver):
            raise ValueError('Checkpoint factors or random-stream state are invalid')
        if cfg['method']=='spring_sarah' and any(solver.get(field) is None for field in ('est_u','est_v','previous_u','previous_v')):
            raise ValueError('SPRING checkpoint lacks recursive-estimator state')
        cases.append(dict(run_id=cfg['run_id'],method=cfg['method'],origin_checkpoint_sha256=cfg['origin_checkpoint_sha256'],
            iteration=solver['k'],observations=len(state['trace']),gradient_forward_views=gradient_views(state['counters']),
            metadata_trace_result_checkpoint_consistent=True,factors_finite=True,optimizer_and_rng_state_present=True))
    return dict(status='passed',ct_operator_calls=0,origin_checkpoints=len(cases),cases=cases,
        recording_engine_sha256=digest(ROOT/'run.py'),evaluation_plan_sha256=digest(ROOT/'evaluation_plan.json'))


def prepare():
    ROOT.mkdir(exist_ok=True)
    frozen('specification.json',specification())
    source={name:digest(ROOT/name) for name in SOURCES}
    validation=read(ROOT/'implementation_validation.json')
    if validation.get('status')!='passed' or validation.get('source_sha256')!=source:
        raise ValueError('Current continuation engine lacks validated bounded evidence')
    copied={name:SPRING/name if name=='local_curvature.py' else BASE/name for name in SOURCES if name!='run.py'}
    if any(source[name]!=digest(path) for name,path in copied.items()):raise ValueError('An optimizer equation or numerical primitive changed')
    implementation=dict(current_source_sha256=source,
        unchanged_source_sha256={name:source[name] for name in copied},
        reference_sources={name:dict(path=str(path),sha256=source[name]) for name,path in copied.items()},
        authorized_changes={'run.py':'Continue immutable completed checkpoints to a gradient-only cap; preserve native prefixes and all inclusive/curvature counters.'},
        specification_sha256=digest(ROOT/'specification.json'),implementation_validation_sha256=digest(ROOT/'implementation_validation.json'))
    frozen('implementation_manifest.json',implementation)
    predecessor=read(PRESENTATION/'report_update.json')
    if predecessor['status']!='completed_published':raise ValueError('Current mean/SD report is not published')
    frozen('predecessor_publication.json',predecessor)
    baseline_report=read(BASE/'report_update.json')
    frozen('predecessor_baseline_publication.json',baseline_report)
    summary=read(baseline_report['corrected_summary']);groups=summary['design']['evaluation_slice_groups']
    if digest(baseline_report['corrected_summary'])!=baseline_report['corrected_summary_sha256']:raise ValueError('Current comparison summary changed')
    records=source_records();plan=make_plan(records,groups);frozen('evaluation_plan.json',plan)
    frozen('origin_audit.json',audit_origins(plan))
    inputs={};origins={}
    for cfg in plan['runs']:
        origin=Path(cfg['origin_directory'])
        for name,key in (('checkpoint.pkl','checkpoint'),('metadata.json','metadata'),('trace.json','trace'),('result.json','result')):
            origins[str(origin/name)]=cfg[f'origin_{key}_sha256']
        meta=read(origin/'metadata.json')
        for category,name,field in (('data',f'{cfg["slice_id"]}.npz','data_sha256'),
                                    ('calibration',f'{cfg["slice_id"]}_seed{cfg["calibration_seed"]}.npz','calibration_sha256')):
            p=ROOT/category/name
            if digest(p)!=meta[field]:raise ValueError('Input data differs from continuation origin')
            inputs[str(p)]=meta[field]
    protocol=dict(host=os.uname().nodename,python=PYTHON,cpus=list(CPUS),blas_threads=1,campaign_sha256=digest(Path(__file__)),
        numerical_sources=source,specification_sha256=digest(ROOT/'specification.json'),
        implementation_manifest_sha256=digest(ROOT/'implementation_manifest.json'),
        implementation_validation_sha256=digest(ROOT/'implementation_validation.json'),
        evaluation_plan_sha256=digest(ROOT/'evaluation_plan.json'),
        origin_audit_sha256=digest(ROOT/'origin_audit.json'),
        predecessor_publication_sha256=digest(ROOT/'predecessor_publication.json'),
        predecessor_baseline_publication_sha256=digest(ROOT/'predecessor_baseline_publication.json'),
        predecessor_validation_files={str(p):digest(p) for p in (BASE/'validation_summary.json',SPRING/'validation_summary.json')},
        origin_files=origins,inputs=inputs,gradient_view_budget_per_direction=CAP,
        publication=dict(helper='update_report.py',arguments=['--publish'],helper_hash_frozen_at_launch=True),
        continuation=True,optimizer_parameters_unchanged=True,curvature_costs_recorded_separately=True,no_optimizer_timing=True)
    frozen('protocol.json',protocol);verify_prepared()
    if not (ROOT/'progress.json').exists():update('prepared','awaiting_launch',completed=0,total=168)
    return dict(status='prepared',evaluation_runs=168,protocol_sha256=digest(ROOT/'protocol.json'))


def verify_prepared():
    protocol=read(ROOT/'protocol.json');files=dict(protocol['numerical_sources'])
    for name,field in (('campaign.py','campaign_sha256'),('specification.json','specification_sha256'),
        ('implementation_manifest.json','implementation_manifest_sha256'),('implementation_validation.json','implementation_validation_sha256'),
        ('evaluation_plan.json','evaluation_plan_sha256'),('origin_audit.json','origin_audit_sha256'),('predecessor_publication.json','predecessor_publication_sha256'),
        ('predecessor_baseline_publication.json','predecessor_baseline_publication_sha256')):files[name]=protocol[field]
    for name,expected in files.items():
        if digest(ROOT/name)!=expected:raise ValueError(f'Frozen continuation input changed: {name}')
    for name,expected in {**protocol['inputs'],**protocol['origin_files'],**protocol['predecessor_validation_files']}.items():
        if digest(name)!=expected:raise ValueError(f'Original input changed: {name}')
    if read(ROOT/'specification.json')!=specification():raise ValueError('Specification changed')
    return protocol


def validate_trace(trace,method,n,batch=64):
    if not trace or [row['iteration'] for row in trace]!=list(range(len(trace))):raise ValueError('Missing native iteration')
    cumulative={name:0 for name in COUNTERS}
    for k,row in enumerate(trace):
        if any(not isinstance(row[name],int) or row[name]<0 for name in COUNTERS):raise ValueError('Noninteger work counter')
        metrics=('raw_loss','feasible_loss','validation_loss','reference_relative_error','d_distance','c_violation','repair_distance',
                 'max_projection_distance_bound')
        if (any(not isinstance(row[name],(int,float)) or not math.isfinite(row[name]) or row[name]<0 for name in metrics)
            or row['d_distance']!=0 or row['c_violation']>1e-10
            or not math.isclose(row['raw_loss'],row['feasible_loss'],rel_tol=1e-10,abs_tol=1e-12)
            or any('seconds' in name for name in row)):raise ValueError('Invalid recorded metric')
        if k==0:
            if any(row[name]!=0 for name in COUNTERS) or row['last_step'] is not None:raise ValueError('Invalid initial counters')
            continue
        step=row['last_step'];curvature_batch=n if method=='palm' else batch
        gradient_calls=2 if method=='palm' else sum(1 if step[f'refresh_{block}'] else 2 for block in ('u','v'))
        gradient_cost=2*n if method=='palm' else sum(n if step[f'refresh_{block}'] else 2*batch for block in ('u','v'))
        if method=='spring_sarah' and k==1 and not (step['refresh_u'] and step['refresh_v']):raise ValueError('First SPRING update lacks full snapshots')
        expected=dict(forward_views=gradient_cost+12*curvature_batch,backward_views=gradient_cost+12*curvature_batch,
            gradient_calls=gradient_calls,curvature_operator_calls=12,curvature_forward_views=12*curvature_batch,
            curvature_backward_views=12*curvature_batch,oracle_calls=gradient_calls+12,projection_calls=2)
        if any(step[name]!=value for name,value in expected.items()) or step['component_gradient_calls']!=gradient_cost:
            raise ValueError('Native step work differs from unchanged algorithm')
        for block in ('u','v'):
            L,eta=step[f'curvature_{block}'],step[f'eta_{block}']
            if (not math.isfinite(L) or L<=0 or not math.isfinite(eta) or eta<=0
                or not math.isclose(L*eta,1. if method=='palm' else .5,rel_tol=1e-13)
                or step[f'curvature_sample_views_{block}']!=curvature_batch):raise ValueError('Paper step rule changed')
        for name in COUNTERS:
            cumulative[name]+=step[name]
            if row[name]!=cumulative[name]:raise ValueError('Step/cumulative work mismatch')
        for direction in ('forward','backward'):
            field='gradient_'+direction+'_views'
            if field in row and row[field]!=gradient_views(row,direction):raise ValueError('Gradient-only aliases disagree')
        if gradient_views(row)!=gradient_views(row,'backward'):raise ValueError('Directional gradient costs differ')
    return True


def validate_result(cfg,cpu,plan,protocol):
    folder=ROOT/'runs'/cfg['run_id'];origin=Path(cfg['origin_directory'])
    meta,result,trace=(read(folder/name) for name in ('metadata.json','result.json','trace.json'))
    original=read(origin/'metadata.json');prefix=read(origin/'trace.json');implementation=read(ROOT/'implementation_manifest.json')
    if (meta['config']!=cfg or meta['cpu']!=cpu or meta['optimizer']!=original['optimizer']
        or any(meta[k]!=original[k] for k in PROBLEM_FIELDS)
        or meta['source_sha256']!=protocol['numerical_sources']
        or meta['implementation_manifest_sha256']!=protocol['implementation_manifest_sha256']
        or meta['specification_sha256']!=protocol['specification_sha256']
        or meta['authorized_implementation_changes']!=implementation['authorized_changes']
        or meta.get('factory_config')!=original.get('factory_config')
        or any(result.get(k)!=v for k,v in meta.items())
        or result['status']!='completed' or result['final']!=trace[-1] or result['iteration']!=trace[-1]['iteration']):
        raise ValueError('Extension provenance or result metadata disagree')
    continuation=dict(source_directory=str(origin.resolve()),run_id=cfg['origin_run_id'],iteration=prefix[-1]['iteration'],
        gradient_forward_views=gradient_views(prefix[-1]),gradient_backward_views=gradient_views(prefix[-1],'backward'),
        source_metadata_sha256=cfg['origin_metadata_sha256'],source_trace_sha256=cfg['origin_trace_sha256'],
        source_checkpoint_sha256=cfg['origin_checkpoint_sha256'],source_result_sha256=cfg['origin_result_sha256'],
        source_code_sha256=original['source_sha256'],prefix_observations=len(prefix),optimizer_and_rng_state_restored=True,
        source_trace_prefix_preserved=True,includes_source_final_update_past_inclusive_view_cap=True)
    if (meta.get('continuation')!=continuation or meta.get('imported_checkpoint')!=str(origin.resolve()/'checkpoint.pkl')
        or meta.get('recovery_origin')!='continued_from_completed_checkpoint'
        or meta.get('budget')!=dict(metric='gradient_views',per_direction=cfg['gradient_view_budget'],
                                   definition='total_views - curvature_views',cumulative_from_initialization=True)):
        raise ValueError('Checkpoint continuation or stopping-budget provenance differs')
    if trace[:len(prefix)]!=prefix or len(trace)<=len(prefix):raise ValueError('Original dense prefix changed or was not extended')
    if any('gradient_forward_views' not in row or 'gradient_backward_views' not in row for row in trace[len(prefix):]):
        raise ValueError('New native observations omit explicit gradient-only work counters')
    validate_trace(trace,cfg['method'],cfg['n_train'])
    cap=cfg['gradient_view_budget'];within=[row for row in trace if max(gradient_views(row),gradient_views(row,'backward'))<=cap]
    if gradient_views(trace[-1])<cap or len(trace)-len(within)>1:raise ValueError('Invalid gradient budget stopping point')
    if len(within)<len(trace) and within[-1]['iteration']!=trace[-1]['iteration']-1:raise ValueError('Missing within-budget final iterate')
    endpoint=within[-1];inclusive=[row for row in trace if max(row['forward_views'],row['backward_views'])<=CAP][-1]
    original_inclusive=[row for row in prefix if max(row['forward_views'],row['backward_views'])<=CAP][-1]
    if inclusive!=original_inclusive:raise ValueError('Historical inclusive-cap endpoint changed')
    if cfg['method']=='palm' and (endpoint['iteration']!=cap//(2*cfg['n_train']) or gradient_views(endpoint)!=cap):
        raise ValueError('PALM did not attain the exact common gradient cap')
    return dict(status='completed',run_id=cfg['run_id'],method=cfg['method'],slice_id=cfg['slice_id'],seed=cfg['seed'],n_train=cfg['n_train'],
        group=plan['groups'][cfg['slice_id']],cpu=cpu,source_directory=str(folder),origin_directory=str(origin),origin_run_id=cfg['origin_run_id'],
        origin_file_sha256={name:cfg[f'origin_{name}_sha256'] for name in ('checkpoint','metadata','trace','result')},
        optimizer=meta['optimizer'],metadata_sha256=digest(folder/'metadata.json'),trace_sha256=digest(folder/'trace.json'),result_sha256=digest(folder/'result.json'),
        observations=len(trace),prefix_observations=len(prefix),prefix_preserved=True,initial=trace[0],endpoint=endpoint,inclusive_endpoint=inclusive,
        final_observation=trace[-1],within_budget_observations=len(within),gradient_forward_views=gradient_views(endpoint),
        gradient_backward_views=gradient_views(endpoint,'backward'),gradient_cap_slack=cap-gradient_views(endpoint),
        final_gradient_overshoot=max(0,gradient_views(trace[-1])-cap),added_iterations=trace[-1]['iteration']-prefix[-1]['iteration'],
        raw_loss=endpoint['raw_loss'],repaired_loss=endpoint['feasible_loss'],relative_loss=endpoint['feasible_loss']/trace[0]['feasible_loss'],
        d_distance=endpoint['d_distance'],initial_factor_frobenius_norm=meta['initial_factor_frobenius_norm'],
        endpoint_exceeds_initial=endpoint['raw_loss']>trace[0]['raw_loss'])


def update(status,phase,**values):
    value=dict(status=status,phase=phase,updated_utc=now(),pid=os.getpid(),**values)
    write(ROOT/'progress.json',value);print(json.dumps(value,separators=(',',':')),flush=True)


def worker(cfg,cpu):
    for name in THREAD_VARS:os.environ[name]='1'
    sys.path.insert(0,DEPS);os.sched_setaffinity(0,{cpu})
    from run import run
    return run(cfg,cpu=cpu)


def run_plan(plan,protocol):
    completed={};jobs=[]
    for cfg in plan['runs']:
        cpu=plan['assignments'][cfg['run_id']]
        if (ROOT/'runs'/cfg['run_id']/'result.json').exists():completed[cfg['run_id']]=validate_result(cfg,cpu,plan,protocol)
        else:jobs.append((cfg,cpu))
    update('running','evaluation',completed=len(completed),total=len(plan['runs']))
    if jobs:
        for name in THREAD_VARS:os.environ[name]='1'
        context=multiprocessing.get_context('spawn')
        with ExitStack() as stack:
            pools={cpu:stack.enter_context(ProcessPoolExecutor(max_workers=1,mp_context=context)) for cpu in CPUS}
            futures={pools[cpu].submit(worker,cfg,cpu):(cfg,cpu) for cfg,cpu in jobs}
            for future in as_completed(futures):
                cfg,cpu=futures[future]
                try:
                    future.result();completed[cfg['run_id']]=validate_result(cfg,cpu,plan,protocol)
                except Exception as exc:
                    write(ROOT/'execution_abort.json',dict(active=True,run_id=cfg['run_id'],error=repr(exc),utc=now()))
                    for other in futures:other.cancel()
                    raise
                update('running','evaluation',completed=len(completed),total=len(plan['runs']),last_completed=cfg['run_id'])
    return [completed[cfg['run_id']] for cfg in plan['runs']]


def execute():
    protocol=verify_prepared()
    if os.uname().nodename!=protocol['host'] or Path(sys.executable).absolute()!=Path(PYTHON):raise ValueError('Wrong host or Python')
    if not set(CPUS).issubset(os.sched_getaffinity(0)):raise ValueError('Worker CPUs unavailable')
    abort=ROOT/'execution_abort.json'
    if abort.exists() and read(abort).get('active'):raise RuntimeError('Review retained operational failure before resuming')
    helper=ROOT/'update_report.py'
    if not helper.is_file():raise ValueError('Report updater must be ready before launch')
    publication_files=('update_report.py','plot_trajectories.py','validate_report.py','report_fixture_validation.json','README.md')
    frozen('publication_helper.json',dict(path=str(helper),sha256=digest(helper),
        dependencies={name:digest(ROOT/name) for name in publication_files}))
    plan=read(ROOT/'evaluation_plan.json');cases=run_plan(plan,protocol)
    verify_prepared()
    summary=dict(status='completed_validated',expected_runs=168,completed_runs=len(cases),cases=cases,
        protocol_sha256=digest(ROOT/'protocol.json'),specification_sha256=digest(ROOT/'specification.json'),
        implementation_manifest_sha256=digest(ROOT/'implementation_manifest.json'),evaluation_plan_sha256=digest(ROOT/'evaluation_plan.json'),
        gradient_view_budget_per_direction=CAP,original_inclusive_view_budget_per_direction=CAP,
        endpoint_rule=plan['endpoint_rule'],curvature_costs_recorded_separately=True,original_prefixes_unchanged=True,
        endpoint_exceeds_initial_count=sum(c['endpoint_exceeds_initial'] for c in cases))
    frozen('validation_summary.json',summary)
    update('evaluated','evaluation_validated',completed=168,total=168,validation_summary_sha256=digest(ROOT/'validation_summary.json'))
    publication=read(ROOT/'publication_helper.json')
    if any(digest(ROOT/name)!=sha for name,sha in publication['dependencies'].items()):
        raise ValueError('A publication dependency changed after launch')
    subprocess.run([sys.executable,str(helper),'--publish'],check=True,cwd=ROOT)
    update('completed','report_published',completed=168,total=168,validation_summary_sha256=digest(ROOT/'validation_summary.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--verify',action='store_true');args=parser.parse_args()
    with (ROOT/'campaign.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.prepare:print(json.dumps(prepare(),indent=2))
        elif args.verify:verify_prepared();print(json.dumps(dict(status='verified',protocol_sha256=digest(ROOT/'protocol.json'))))
        else:
            try:execute()
            except Exception as exc:update('failed','execution',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
