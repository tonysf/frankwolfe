"""Tune the fixed ProxSARAH-v3 scale, then evaluate both paper baselines.

Preparation is stdlib-only. Numerical work waits for SPRING publication.
All development outcomes are retained; only operational errors stop the queue.
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

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'convincing_4x_recovery'
DEVELOPMENT_REFERENCE=ROOT.parent/'convincing_4x'/'runs'
PREDECESSOR=ROOT.parent/'spring_localcurvature'
DEPENDENCY=PREDECESSOR/'progress.json'
PYTHON='/home/tony/zofw_experiments/.venv/bin/python'
DEPS='/home/tony/frames_ct_2detect_20261104/deps'
CPUS=(0,2,4,6)
DEV_CAP,EVAL_CAP=34560,69120
DEV_SLICES=('slice04300','slice03150')
DEV_SEEDS=(110,111,112)
SIZES=(360,1080)
INITIAL_EXPONENTS=tuple(range(-5,3))
MIN_EXPONENT,MAX_EXPONENT=-8,5
UNCHANGED_SOURCES=('ct_problem.py','methods.py','spectral_reference_core.py','spectral_reference_baseline.py')
SOURCES=('run.py','palm_paper.py','prox_paper.py','curvature_reference.py',*UNCHANGED_SOURCES)
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
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,allow_nan=False,indent=2)+'\n');temp.replace(path)
def frozen(name,value):
    path=ROOT/name
    if path.exists():
        if read(path)!=value:raise ValueError(f'Frozen input differs: {path}')
    else:write(path,value)


def prox_parameters(n,loss_scale):
    gamma=.99;C=2./(3.*gamma**2);eta=2./(4.+gamma)
    m=math.isqrt(n)
    return dict(paper_variant='ProxSARAH-v3',normalized_lipschitz=1.,loss_scale=loss_scale,
        gamma=gamma,relaxation=gamma,C=C,eta_normalized=eta,eta=eta/loss_scale,
        batch=math.floor(math.sqrt(n)/C),inner_steps=m,snapshot_batch=n,updates_per_epoch=m+1)


def specification():
    return dict(status='frozen',
        palm=dict(rule='Current full-data alternating block step eta=1/Lhat, using five normalized Hessian applications and a sixth norm evaluation.',
                  power_iterations=5,step_coefficient=1.,curvature_batch='all training views',
                  stochasticity='Fresh Gaussian power directions; evaluate all three oracle/power seed labels.'),
        prox_sarah=dict(paper_variant='ProxSARAH-v3',source='Section 5 numerical variant with Section 3.4 constant-step parameter coupling',
            normalized_lipschitz=1.,gamma=.99,C='2/(3*gamma^2)',inner_steps='floor(sqrt(N))',
            batch='floor(sqrt(N)/C)',eta_original_objective='[2/(4+gamma)]/Lhat',
            full_snapshot=True,updates_per_epoch='one snapshot update plus m recursive updates',
            fixed_scale='Lhat=c_L*(L_U^0+L_V^0); c_L selected on original development cases only',
            scale_status='Practical development-tuned smoothness estimate, not a certified global Lipschitz upper bound'),
        development=dict(slices=list(DEV_SLICES),training_views=list(SIZES),seeds=list(DEV_SEEDS),
            calibration_seed=110,view_budget_per_direction=DEV_CAP,recording='initial, final, and last native within-cap observation',
            initial_scale_exponents=list(INITIAL_EXPONENTS),scale_base=2,guard_exponents=[MIN_EXPONENT,MAX_EXPONENT],
            eligibility='All 12 runs completed with finite within-budget repaired/initial loss <=1.',
            ranking='Median over the four slice/view cells of their three-seed median loss ratios; ties favor larger c_L.',
            expansion='If the eligible winner is at a tested edge, evaluate all 12 cases at the next factor-two value on that edge, then reselect. With no eligible candidate, expand only the safer upper edge. Stop for review if the winner is at a guard edge or no eligible candidate exists at the upper guard.',
            failures='Keep scientific nonfinite failures as ineligible outcomes and continue all scheduled cases; operational or projection failures halt for review.'),
        evaluation=dict(physical_runs=168,methods=['palm','prox_sarah'],seeds=[202,203,204],
            view_budget_per_direction=EVAL_CAP,dense=True,shared_palm_seeds=False),
        work=dict(gradient_calls='Literal gradient evaluations',curvature_operator_calls='Hessian applications',
            oracle_calls='gradient_calls+curvature_operator_calls',directional_views='Include estimator and curvature work',
            palm_per_iteration='2 gradients +12 full-data Hessian applications =14N views per direction',
            prox_per_iteration='Full snapshot: one gradient on N views; recursive update: two gradients on b views'),
        parameter_selection_uses_evaluation_outcomes=False,
        implementation_scope='Published practical numerical variants adapted to CT; neither power estimates nor tuned Lhat are certified theorem constants.',
        publication='Follow the completed SPRING paper update, preserve all other arms, and add no figures.')


def label(exponent):return f'l2{"m" if exponent<0 else "p"}{abs(exponent)}'


def config(phase,sid,n,seed,method,exponent=None):
    trial='palm_paper_power5' if method=='palm' else 'v3_'+label(exponent)
    cfg=dict(run_id=f'{phase}_{sid}_{n}_{method}_seed{seed}_{trial}',phase=phase,slice_id=sid,
        n_train=n,method=method,variant=method,trial=trial,seed=seed,calibration_seed=110 if phase=='development' else 201,
        grid=512,rank=64,batch=n if method=='palm' else prox_parameters(n,1.)['batch'],
        view_budget=DEV_CAP if phase=='development' else EVAL_CAP,record_every_iteration=phase=='evaluation')
    if method=='prox_sarah':cfg.update(scale_exponent=exponent,scale_multiplier=2.**exponent)
    return cfg


def dev_plan(exponents,round_number,prior_round_sha256=None):
    configs=[];assignments={};matched={};directories={}
    for exponent in exponents:
        for index,(sid,n,seed) in enumerate(itertools.product(DEV_SLICES,SIZES,DEV_SEEDS)):
            cfg=config('development',sid,n,seed,'prox_sarah',exponent);configs.append(cfg)
            assignments[cfg['run_id']]=CPUS[index%len(CPUS)]
            old=f'development_{sid}_{n}_prox_sarah_seed{seed}_a2p3_m96_sminibatch'
            matched[cfg['run_id']]=old;directories[cfg['run_id']]=str(DEVELOPMENT_REFERENCE/old)
    return dict(phase='development',round=round_number,scale_exponents=list(exponents),runs=configs,
        assignments=assignments,matched_run_ids=matched,matched_source_directories=directories,
        specification_sha256=digest(ROOT/'specification.json'),prior_round_sha256=prior_round_sha256)


def prepare():
    reference=read(OLD/'compatibility_manifest.json')['current_source_sha256']
    for name in UNCHANGED_SOURCES:
        if digest(OLD/name)!=reference[name]:raise ValueError(f'Reference changed: {name}')
        if not (ROOT/name).exists():shutil.copy2(OLD/name,ROOT/name)
        if digest(ROOT/name)!=reference[name]:raise ValueError(f'Unchanged module differs: {name}')
    if digest(ROOT/'curvature_reference.py')!=digest(PREDECESSOR/'local_curvature.py'):
        raise ValueError('PALM Hessian reference is not the exact validated SPRING implementation')
    for name in ('data','calibration'):
        path=ROOT/name
        if not path.exists() and not path.is_symlink():path.symlink_to(OLD/name,target_is_directory=True)
        if not path.is_symlink() or path.resolve()!=(OLD/name).resolve():raise ValueError('Input link changed')
    frozen('specification.json',specification())
    sources={name:digest(ROOT/name) for name in SOURCES}
    tests=read(ROOT/'implementation_validation.json')
    if tests.get('status')!='passed' or tests.get('source_sha256')!=sources:
        raise ValueError('Current recording engine and factories lack bounded validation')
    implementation=dict(current_source_sha256=sources,reference_root=str(OLD),reference_source_sha256=reference,
        unchanged_source_sha256={name:reference[name] for name in UNCHANGED_SOURCES},
        copied_curvature_source=dict(path=str(PREDECESSOR/'local_curvature.py'),sha256=digest(ROOT/'curvature_reference.py')),
        authorized_changes={
            'palm_paper.py':'Current full-data alternating curvature, paper step coefficient1, checkpointed fresh power directions.',
            'prox_paper.py':'Named ProxSARAH-v3 parameter coupling, full snapshots, fixed calibrated loss scale.',
            'curvature_reference.py':'Exact copy of the previously validated CT block Hessian/power implementation.',
            'run.py':'Fresh development/evaluation recording, sparse within-budget endpoints, dense evaluation, inclusive counters and retained scientific failures.'},
        specification_sha256=digest(ROOT/'specification.json'),
        implementation_validation_sha256=digest(ROOT/'implementation_validation.json'))
    frozen('implementation_manifest.json',implementation)
    cohort=read(OLD/'cohort.json');manifest=read(OLD/'manifest.json')
    inputs={}
    for sid in (*cohort['evaluation_slices'],*DEV_SLICES):
        rel=f'data/{sid}.npz';expected=manifest['data'][sid]
        if digest(ROOT/rel)!=expected:raise ValueError(f'Data changed: {sid}')
        inputs[rel]=expected
        seed=110 if sid in DEV_SLICES else 201
        rel=f'calibration/{sid}_seed{seed}.npz';inputs[rel]=digest(ROOT/rel)
    anchors={}
    for sid,n,seed in itertools.product(DEV_SLICES,SIZES,DEV_SEEDS):
        folder=DEVELOPMENT_REFERENCE/f'development_{sid}_{n}_prox_sarah_seed{seed}_a2p3_m96_sminibatch'
        meta=read(folder/'metadata.json')
        if meta['calibration_sha256']!=inputs[f'calibration/{sid}_seed110.npz']:
            raise ValueError('Development calibration differs from historical selection')
        for name in ('metadata.json','trace.json'):anchors[str(folder/name)]=digest(folder/name)
    original=read(OLD/'evaluation_plan.json')['runs']
    selected=[c for c in original if c['variant'] in ('palm','prox_sarah')]
    if len(selected)!=112:raise ValueError('Original baseline reference cohort changed')
    for cfg in selected:
        folder=OLD/'runs'/cfg['run_id']
        for name in ('metadata.json','trace.json'):anchors[str(folder/name)]=digest(folder/name)
        meta=read(folder/'metadata.json')
        if meta['calibration_sha256']!=inputs[f'calibration/{cfg["slice_id"]}_seed201.npz']:
            raise ValueError('Evaluation calibration differs from the original comparison')
    template=dict(slices=cohort['evaluation_slices'],groups=cohort['evaluation_slice_groups'],
        training_views=list(SIZES),seeds=[202,203,204],methods=['palm','prox_sarah'],physical_runs=168,
        original_references=selected,view_budget_per_direction=EVAL_CAP)
    frozen('evaluation_template.json',template)
    frozen('development_plan_r00.json',dev_plan(INITIAL_EXPONENTS,0))
    protocol=dict(host=os.uname().nodename,python=PYTHON,cpus=list(CPUS),blas_threads=1,
        campaign_sha256=digest(Path(__file__)),numerical_sources=sources,
        specification_sha256=digest(ROOT/'specification.json'),
        implementation_manifest_sha256=digest(ROOT/'implementation_manifest.json'),
        implementation_validation_sha256=digest(ROOT/'implementation_validation.json'),
        evaluation_template_sha256=digest(ROOT/'evaluation_template.json'),
        initial_development_plan_sha256=digest(ROOT/'development_plan_r00.json'),
        inputs=inputs,matched_run_files=anchors,
        matched_files={str(OLD/name):digest(OLD/name) for name in ('manifest.json','cohort.json','evaluation_plan.json')},
        resource_dependency=dict(path=str(DEPENDENCY),status='completed',phase='report_published'),
        publication=dict(helper='update_report.py',arguments=['--publish'],helper_hash_frozen_at_launch=True),
        fresh_runs=True,curvature_work_included=True,no_optimizer_timing=True,
        evaluation_outcomes_excluded_from_selection=True)
    frozen('protocol.json',protocol);verify_prepared()
    if not (ROOT/'progress.json').exists():update('prepared','awaiting_launch',completed=0)
    return dict(status='prepared',initial_development_runs=96,evaluation_runs=168,
                protocol_sha256=digest(ROOT/'protocol.json'))


def verify_prepared():
    protocol=read(ROOT/'protocol.json')
    files=dict(protocol['numerical_sources']);files.update(protocol['inputs'])
    for name,field in (('campaign.py','campaign_sha256'),('specification.json','specification_sha256'),
        ('implementation_manifest.json','implementation_manifest_sha256'),
        ('implementation_validation.json','implementation_validation_sha256'),
        ('evaluation_template.json','evaluation_template_sha256'),
        ('development_plan_r00.json','initial_development_plan_sha256')):files[name]=protocol[field]
    for name,expected in files.items():
        if digest(ROOT/name)!=expected:raise ValueError(f'Frozen input changed: {name}')
    for name,expected in {**protocol['matched_files'],**protocol['matched_run_files']}.items():
        if digest(Path(name))!=expected:raise ValueError(f'Matched reference changed: {name}')
    if read(ROOT/'specification.json')!=specification():raise ValueError('Specification differs')
    return protocol


def expected_counts(method,n,k,params):
    if method=='palm':
        return dict(forward_views=14*n*k,backward_views=14*n*k,gradient_calls=2*k,oracle_calls=14*k,
            curvature_operator_calls=12*k,curvature_forward_views=12*n*k,curvature_backward_views=12*n*k,
            projection_calls=2*k)
    snapshots=(k+params['inner_steps'])//params['updates_per_epoch'] if k else 0
    recursive=k-snapshots;views=snapshots*n+recursive*2*params['batch'];calls=snapshots+2*recursive
    return dict(forward_views=views,backward_views=views,gradient_calls=calls,oracle_calls=calls,
        curvature_operator_calls=0,curvature_forward_views=0,curvature_backward_views=0,projection_calls=2*k)


def validate_result(cfg,cpu,plan,protocol):
    folder=ROOT/'runs'/cfg['run_id']
    meta,result,trace=(read(folder/name) for name in ('metadata.json','result.json','trace.json'))
    ref=Path(plan['matched_source_directories'][cfg['run_id']]);old=read(ref/'metadata.json')
    implementation=read(ROOT/'implementation_manifest.json')
    if cfg['method']=='palm':
        factory=dict(power_iterations=5,step_coefficient=1.)
        params=dict(factory,curvature_batch_size=cfg['n_train'])
    else:
        factory=dict(loss_scale=cfg['scale_multiplier']*sum(meta['curvature']))
        params=prox_parameters(cfg['n_train'],factory['loss_scale'])
    if (meta['config']!=cfg or meta['cpu']!=cpu or meta['optimizer']!=params or meta['factory_config']!=factory
        or any(result.get(k)!=v for k,v in meta.items())
        or meta['source_sha256']!=protocol['numerical_sources']
        or meta['implementation_manifest_sha256']!=protocol['implementation_manifest_sha256']
        or meta['specification_sha256']!=protocol['specification_sha256']
        or meta['authorized_implementation_changes']!=implementation['authorized_changes']
        or meta['recovery_origin']!='fresh' or meta['continuation'] is not None or meta['imported_checkpoint'] is not None
        or any(meta[k]!=old[k] for k in PROBLEM_FIELDS)):
        raise ValueError(f'Invalid provenance: {cfg["run_id"]}')
    old_initial=read(ref/'trace.json')[0]
    if (not trace or result['final']!=trace[-1] or result['iteration']!=trace[-1]['iteration']
        or any(trace[0][k]!=v for k,v in old_initial.items())
        or trace[0]['last_step'] is not None):raise ValueError('Initial/native result disagrees')
    indices=[row['iteration'] for row in trace]
    if indices!=sorted(set(indices)) or indices[0]!=0:raise ValueError('Native observation order changed')
    if cfg['record_every_iteration'] and indices!=list(range(indices[-1]+1)):
        raise ValueError('Dense evaluation has missing iterations')
    for row in trace:
        k=row['iteration'];expected=expected_counts(cfg['method'],cfg['n_train'],k,params)
        if any(row[name]!=v for name,v in expected.items()):raise ValueError('Inclusive work counts disagree')
        metrics=('raw_loss','feasible_loss','validation_loss','reference_relative_error','d_distance',
                 'c_violation','repair_distance','max_projection_distance_bound')
        if (not all(isinstance(row[name],(int,float)) and math.isfinite(row[name]) and row[name]>=0 for name in metrics)
            or row['d_distance']!=0 or row['c_violation']>1e-10
            or not math.isclose(row['raw_loss'],row['feasible_loss'],abs_tol=1e-12,rel_tol=1e-10)
            or any('seconds' in name for name in row)):
            raise ValueError('Invalid recorded metric')
        if not k:continue
        step=row['last_step'];before=expected_counts(cfg['method'],cfg['n_train'],k-1,params)
        if any(step[name]!=v-before[name] for name,v in expected.items()):raise ValueError('Step and cumulative costs disagree')
        if step['component_gradient_calls']!=step['forward_views']-step['curvature_forward_views']:
            raise ValueError('Component gradient count includes curvature')
        if cfg['method']=='palm':
            for block in ('u','v'):
                L,eta=step[f'curvature_{block}'],step[f'eta_{block}']
                if (not math.isfinite(L) or L<=0 or not math.isfinite(eta) or eta<=0
                    or not math.isclose(L*eta,1.,rel_tol=1e-13)
                    or step[f'curvature_sample_views_{block}']!=cfg['n_train']):raise ValueError('PALM step rule differs')
        else:
            position=(k-1)%params['updates_per_epoch']
            if (step['epoch_position']!=position or step['refresh']!=(position==0)
                or step['eta']!=params['eta'] or step['gamma']!=params['gamma']):raise ValueError('ProxSARAH-v3 recipe differs')
    for before,after in zip(trace,trace[1:]):
        if any(after[name]<before[name] for name in COUNTERS):raise ValueError('Counters decrease')
    common=dict(run_id=cfg['run_id'],method=cfg['method'],slice_id=cfg['slice_id'],n_train=cfg['n_train'],seed=cfg['seed'],cpu=cpu,
        source_directory=str(folder),matched_run_id=plan['matched_run_ids'][cfg['run_id']],
        optimizer=params,observations=len(trace),initial=trace[0],final_observation=trace[-1],
        metadata_sha256=digest(folder/'metadata.json'),trace_sha256=digest(folder/'trace.json'),result_sha256=digest(folder/'result.json'),
        initial_factor_frobenius_norm=meta['initial_factor_frobenius_norm'],matched_initial_observation=True,provenance_valid=True)
    if 'groups' in plan:common['group']=plan['groups'][cfg['slice_id']]
    if cfg['method']=='prox_sarah':common.update(scale_multiplier=cfg['scale_multiplier'],scale_exponent=cfg['scale_exponent'])
    if result['status']=='failed_scientific' and cfg['phase']=='development':
        return dict(common,status='failed_scientific',failure=result['failure'],endpoint=None,relative_loss=None)
    if result['status']!='completed':raise ValueError('Evaluation or operational run did not complete')
    cap=cfg['view_budget'];within=[row for row in trace if max(row['forward_views'],row['backward_views'])<=cap]
    if trace[-1]['forward_views']<cap or len(trace)-len(within)>1:raise ValueError('Invalid cap crossing')
    if trace[-1]['forward_views']>cap and trace[-2]['iteration']!=trace[-1]['iteration']-1:
        raise ValueError('Missing last native within-budget iterate')
    if not cfg['record_every_iteration'] and len(trace)>3:raise ValueError('Unexpected sparse recording')
    endpoint=within[-1];initial=trace[0]
    return dict(common,status='completed',endpoint=endpoint,within_budget_observations=len(within),
        cap_slack=cap-endpoint['forward_views'],final_overshoot=max(0,trace[-1]['forward_views']-cap),
        initial_loss=initial['feasible_loss'],raw_loss=endpoint['raw_loss'],repaired_loss=endpoint['feasible_loss'],
        relative_loss=endpoint['feasible_loss']/initial['feasible_loss'],relative_raw_loss=endpoint['raw_loss']/initial['raw_loss'],
        d_distance=endpoint['d_distance'],endpoint_exceeds_initial=endpoint['raw_loss']>initial['raw_loss'])


def update(status,phase,**extras):
    value=dict(status=status,phase=phase,updated_utc=now(),pid=os.getpid(),**extras)
    write(ROOT/'progress.json',value);print(json.dumps(value,separators=(',',':')),flush=True)


def worker(cfg,cpu):
    for name in THREAD_VARS:os.environ[name]='1'
    sys.path.insert(0,DEPS);os.sched_setaffinity(0,{cpu})
    from run import run
    return run(cfg,cpu=cpu)


def run_plan(plan,protocol):
    done={};jobs=[]
    for cfg in plan['runs']:
        cpu=plan['assignments'][cfg['run_id']]
        if (ROOT/'runs'/cfg['run_id']/'result.json').exists():done[cfg['run_id']]=validate_result(cfg,cpu,plan,protocol)
        else:jobs.append((cfg,cpu))
    phase=plan['phase'];total=len(plan['runs']);update('running',phase,completed=len(done),total=total,round=plan.get('round'))
    if jobs:
        for name in THREAD_VARS:os.environ[name]='1'
        context=multiprocessing.get_context('spawn')
        with ExitStack() as stack:
            pools={cpu:stack.enter_context(ProcessPoolExecutor(max_workers=1,mp_context=context)) for cpu in CPUS}
            futures={pools[cpu].submit(worker,cfg,cpu):(cfg,cpu) for cfg,cpu in jobs}
            for future in as_completed(futures):
                cfg,cpu=futures[future]
                try:
                    future.result();done[cfg['run_id']]=validate_result(cfg,cpu,plan,protocol)
                except Exception as exc:
                    write(ROOT/'execution_abort.json',dict(active=True,run_id=cfg['run_id'],error=repr(exc),utc=now()))
                    for other in futures:other.cancel()
                    raise
                update('running',phase,completed=len(done),total=total,round=plan.get('round'),last_completed=cfg['run_id'],last_status=done[cfg['run_id']]['status'])
    return [done[cfg['run_id']] for cfg in plan['runs']]


def rank_candidates(cases,exponents):
    rows=[]
    for exponent in sorted(exponents):
        subset=[case for case in cases if case['scale_exponent']==exponent]
        if len(subset)!=12 or len({(c['slice_id'],c['n_train'],c['seed']) for c in subset})!=12:
            raise ValueError('Incomplete development candidate')
        complete=all(c['status']=='completed' and c['relative_loss'] is not None and math.isfinite(c['relative_loss']) for c in subset)
        median=statistics.median(statistics.median(c['relative_loss'] for c in subset if c['slice_id']==sid and c['n_train']==n)
            for sid,n in itertools.product(DEV_SLICES,SIZES)) if complete else None
        failures=[c['run_id'] for c in subset if c['status']!='completed' or c['relative_loss'] is None or c['relative_loss']>1]
        rows.append(dict(scale_exponent=exponent,scale_multiplier=2.**exponent,c_L=2.**exponent,n_cases=12,
            selection_eligible=not failures,median_relative_loss=median,
            max_relative_loss=max(c['relative_loss'] for c in subset) if complete else None,
            failed_endpoint_count=len(failures),failed_run_ids=failures,cases=subset))
    eligible=[row for row in rows if row['selection_eligible']]
    winner=min(eligible,key=lambda row:(row['median_relative_loss'],-row['c_L'])) if eligible else None
    return rows,winner


def develop(protocol):
    all_cases=[];tested=[];history=[];exponents=list(INITIAL_EXPONENTS);round_number=0;prior=None
    while True:
        plan=dev_plan(exponents,round_number,prior);name=f'development_plan_r{round_number:02d}.json'
        frozen(name,plan);cases=run_plan(plan,protocol);all_cases.extend(cases);tested.extend(exponents)
        rows,winner=rank_candidates(all_cases,tested)
        next_exponent=None;review=None
        if winner is None:
            if max(tested)<MAX_EXPONENT:next_exponent=max(tested)+1
            else:review='No eligible setting at the upper scale guard'
        elif winner['scale_exponent']==min(tested):
            if min(tested)>MIN_EXPONENT:next_exponent=min(tested)-1
            else:review='Selected setting remains at the lower scale guard'
        elif winner['scale_exponent']==max(tested):
            if max(tested)<MAX_EXPONENT:next_exponent=max(tested)+1
            else:review='Selected setting remains at the upper scale guard'
        status='needs_review' if review else ('extend_grid' if next_exponent is not None else 'selected')
        record=dict(status=status,round=round_number,plan_sha256=digest(ROOT/name),tested_exponents=sorted(tested),
            scores=rows,selected=None if winner is None else {k:v for k,v in winner.items() if k!='cases'},
            next_exponent=next_exponent,review_reason=review)
        result_name=f'selection_round_r{round_number:02d}.json';frozen(result_name,record)
        history.append(dict(plan=name,plan_sha256=digest(ROOT/name),selection=result_name,selection_sha256=digest(ROOT/result_name)))
        if review:
            frozen('selection_review.json',dict(record,history=history))
            update('needs_review','development_selection',reason=review,history=history)
            return None
        if status=='selected':
            selection=dict(status='selected',selected=record['selected'],scores=rows,history=history,
                protocol_sha256=digest(ROOT/'protocol.json'),specification_sha256=digest(ROOT/'specification.json'),
                development_view_budget_per_direction=DEV_CAP,development_cases_per_setting=12,
                evaluation_outcomes_used=False,source='Named ProxSARAH-v3; fixed scale calibrated on original development cohort',
                fixed_palm=dict(power_iterations=5,step_coefficient=1.,curvature_batch='full'))
            frozen('selection.json',selection);return selection
        exponents=[next_exponent];prior=digest(ROOT/result_name);round_number+=1


def evaluation_plan(selection):
    template=read(ROOT/'evaluation_template.json');old=template['original_references']
    refs={(c['slice_id'],c['n_train'],c['seed'],c['method']):c for c in old}
    configs=[];assignments={};matched={};directories={};logical=[];loads=dict.fromkeys(CPUS,0)
    for index,(sid,n,seed,method) in enumerate(itertools.product(template['slices'],SIZES,(202,203,204),('palm','prox_sarah'))):
        exponent=None if method=='palm' else selection['selected']['scale_exponent']
        cfg=config('evaluation',sid,n,seed,method,exponent);configs.append(cfg)
        # Dense diagnostics dominate the short-batch arm's total work. Balance
        # estimated operations while keeping each run's resume CPU immutable.
        params={} if method=='palm' else prox_parameters(n,1.)
        steps=1
        while expected_counts(method,n,steps,params)['forward_views']<EVAL_CAP:steps+=1
        cpu=min(loads,key=loads.get);assignments[cfg['run_id']]=cpu
        loads[cpu]+=2*EVAL_CAP+(steps+1)*(n+360)
        ref=refs[sid,n,202 if method=='palm' else seed,method]
        matched[cfg['run_id']]=ref['run_id'];directories[cfg['run_id']]=str(OLD/'runs'/ref['run_id'])
        logical.append(dict(method=method,slice_id=sid,n_train=n,seed=seed,physical_run_id=cfg['run_id']))
    if len(configs)!=168:raise ValueError('Wrong physical evaluation count')
    return dict(phase='evaluation',expected_runs=168,runs=configs,assignments=assignments,matched_run_ids=matched,
        matched_source_directories=directories,logical_cases=logical,groups=template['groups'],
        selection_sha256=digest(ROOT/'selection.json'),specification_sha256=digest(ROOT/'specification.json'),
        view_budget_per_direction=EVAL_CAP,endpoint_rule='Last native observation at or below the inclusive forward/adjoint view cap.')


def execute():
    protocol=verify_prepared()
    if os.uname().nodename!=protocol['host'] or Path(sys.executable).absolute()!=Path(PYTHON):raise ValueError('Wrong host or Python')
    if not set(CPUS).issubset(os.sched_getaffinity(0)):raise ValueError('Launch affinity excludes a worker CPU')
    abort=ROOT/'execution_abort.json'
    if abort.exists() and read(abort).get('active'):raise RuntimeError('Review prior operational failure and abort flag before resume')
    helper=ROOT/'update_report.py'
    if not helper.is_file():raise RuntimeError('Publication helper must exist before queue launch')
    frozen('publication_helper.json',dict(path=str(helper),sha256=digest(helper)))
    state=None
    while True:
        dependency=read(DEPENDENCY) if DEPENDENCY.exists() else dict(status='missing')
        if (dependency.get('status'),dependency.get('phase'))==('completed','report_published'):break
        current=(dependency.get('status'),dependency.get('phase'))
        if current!=state:
            update('waiting_for_dependency','resource_wait',dependency=str(DEPENDENCY),dependency_status=current[0],dependency_phase=current[1]);state=current
        if abort.exists() and read(abort).get('active'):raise RuntimeError('Abort requested during dependency wait')
        time.sleep(60)
    frozen('dependency_completion.json',dict(path=str(DEPENDENCY),progress_sha256=digest(DEPENDENCY),progress=dependency))
    verify_prepared();selection=develop(protocol)
    if selection is None:return
    plan=evaluation_plan(selection);frozen('evaluation_plan.json',plan)
    cases=run_plan(plan,protocol)
    summary=dict(status='completed_validated',expected_runs=168,completed_runs=len(cases),cases=cases,logical_cases=plan['logical_cases'],
        protocol_sha256=digest(ROOT/'protocol.json'),specification_sha256=digest(ROOT/'specification.json'),
        implementation_manifest_sha256=digest(ROOT/'implementation_manifest.json'),selection_sha256=digest(ROOT/'selection.json'),
        evaluation_plan_sha256=digest(ROOT/'evaluation_plan.json'),dependency_completion_sha256=digest(ROOT/'dependency_completion.json'),
        view_budget_per_direction=EVAL_CAP,endpoint_rule=plan['endpoint_rule'],curvature_work_included=True,
        endpoint_exceeds_initial_count=sum(c['endpoint_exceeds_initial'] for c in cases))
    frozen('validation_summary.json',summary)
    update('evaluated','evaluation_validated',completed=168,total=168,validation_summary_sha256=digest(ROOT/'validation_summary.json'))
    if digest(helper)!=read(ROOT/'publication_helper.json')['sha256']:raise ValueError('Publication helper changed after launch')
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
