"""Fixed-horizon pair-sampling comparison with source-bound launch gates.

This controller never tunes parameters or consults losses to decide whether a
run may continue. Numerical tests, resource measurements, and provenance form
the preflight gate. All GPU computations run inside scheduled allocations.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
METHODS=('smoothing-fw','published-smd','published-msbpg')
PLANS=('selected_raw','selected_rounded','last_raw','last_rounded')
ARRAYS=('A.npy','B.npy','a.npy','b.npy')


def _legacy():
    spec=importlib.util.spec_from_file_location('_pair_execution',ROOT.parent/'published_stochastic_20261002/campaign.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

execute=_legacy().execute


def _msbpg_audit(metadata,expected):
    spec=importlib.util.spec_from_file_location('_pair_msbpg_theory',ROOT.parent/'msbpg_20261005/campaign.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.validate_theory_metadata(metadata,expected)


def require(condition,message):
    if not condition:raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(),parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON '+value)))


def write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n');temp.replace(path)


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def finite(value,label,nonnegative=False):
    require(isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value),'Invalid '+label)
    require(not nonnegative or value>=0,'Negative '+label)
    return float(value)


def load_config(path):
    path=Path(path).resolve();c=read(path)
    require(c['schema_version']==1 and tuple(c['methods'])==METHODS,'Unexpected method specification.')
    require(c['oracle']=='pair-sampling' and c['pairs_per_call']==1024,'Frozen pair-sampling budget differs.')
    require(c['seeds']==[710,711,712] and c['initializations']==['product'],'Frozen seed/start cells differ.')
    require(c['plan_storage']=='none' and c['checkpoint_seconds']==[],'Frozen storage/checkpoint policy differs.')
    require(c['runner']['dtype']=='float32' and c['runner']['inner_tol']==c['runner']['inner_gap_tol']==1e-7,
            'Production precision or numerical tolerance differs.')
    require(0<c['run_timeout_seconds']<=23*3600,'Production timeout exceeds Ruche margin.')
    require(c['preflight']['runtime_safety_factor']>=2 and 0<c['preflight']['maximum_gpu_reserved_fraction']<1,
            'Preflight margin is missing.')
    for name,v in c['instances'].items():
        require(v['world_size'] in (1,4) and v['n']>0,'Invalid production topology.')
        require(v['final_iterations']==(200000 if v['n']==512 else 20000),'Frozen horizon changed.')
        require(0<v['preflight_iterations']<v['final_iterations'],'Invalid preflight horizon.')
        require(set(v['array_files_sha256'])==set(ARRAYS) and len(v['instance_manifest_sha256'])==64,'Missing input binding.')
        require(set(v['prefactors'])==set(METHODS),'Missing method prefactors.')
        checkpoints=v['final_checkpoint_iterations']
        require(checkpoints==sorted(set(checkpoints)) and checkpoints[-1]==v['final_iterations'],'Invalid checkpoint list.')
    for label,p in c['prefactor_provenance'].items():
        provenance=path.parents[2]/p['file']
        require(sha256(provenance)==p['sha256'],'Prefactor evidence changed: '+label)
    real=read(path.parents[2]/c['prefactor_provenance']['real']['file'])['selections']
    synthetic=read(path.parents[2]/c['prefactor_provenance']['synthetic']['file'])['instances']['metric512_s1000']['prefactors']
    for dataset,instance in c['instances'].items():
        if dataset=='metric512_s1000':
            require({k:v for k,v in instance['prefactors']['smoothing-fw'].items() if k!='id'}==synthetic,
                    'Synthetic FraMES prefactors differ from frozen prior configuration.')
            require(instance['prefactors']['published-smd']['prefactor_fraction']==1.
                    and instance['prefactors']['published-msbpg']['coupling_fraction']==1.
                    and instance['prefactors']['published-msbpg']['momentum_prefactor']==1.,'Prespecified synthetic baseline defaults differ.')
        else:
            for method in METHODS:
                require(instance['prefactors'][method]==real[dataset][method]['candidate'],'Real-data prefactor differs from prior selection.')
                require(instance['array_files_sha256']==real[dataset][method]['array_files_sha256']
                        and instance['instance_manifest_sha256']==real[dataset][method]['instance_manifest_sha256'],
                        'Real-data input differs from frozen selected-prefactor evidence.')
    audit=path.parent/c['theory_audit']['file']
    require(sha256(audit)==c['theory_audit']['sha256'],'Mathematical oracle audit changed.')
    audit_record=read(audit)
    require(audit_record['status']=='valid_noise_conditions' and audit_record['valid_noise_conditions'] is True,
            'Conditional unbiasedness/variance audit is not valid.')
    c['_config_path']=str(path)
    return c


def source_hashes(source):
    """Use the pair runner's complete numerical dependency manifest."""
    source=Path(source).resolve()
    spec=importlib.util.spec_from_file_location('_pair_runner_hashes',source/'extensions/pair_sampling_20261011/run_pair.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.source_hashes()


def identity(c,source,prepared,dataset):
    manifest=Path(prepared)/dataset/'manifest.json';instance=c['instances'][dataset]
    require(sha256(manifest)==instance['instance_manifest_sha256'],'Frozen input manifest differs.')
    require(read(manifest)['array_files_sha256']==instance['array_files_sha256'],'Frozen input arrays differ.')
    source=Path(source).resolve()
    return dict(config_sha256=sha256(c['_config_path']),source_file_sha256=source_hashes(source),
                controller_source_sha256={name:sha256(source/'extensions'/name)
                    for name in ('pair_sampling_20261011/campaign.py','pair_sampling_20261011/report_campaign.py',
                                 'pair_sampling_20261011/verify_checks.py','pair_sampling_20261011/validate_distributed.py',
                                 'published_stochastic_20261002/campaign.py','msbpg_20261005/campaign.py')},
                instance_manifest_sha256=instance['instance_manifest_sha256'],array_files_sha256=instance['array_files_sha256'])


def arguments(c,prepared,out,dataset,method,seed,phase):
    instance=c['instances'][dataset]
    values=dict(instance=str((Path(prepared)/dataset).resolve()),out=str(Path(out).resolve()),method=method,device='cuda',
                iterations=instance['preflight_iterations'] if phase=='preflight' else instance['final_iterations'],
                seed=seed,output_seed=seed+20000,init_seed=seed+10000,initialization='product',
                oracle='pair-sampling',pairs_per_call=c['pairs_per_call'],save_plans='none',
                checkpoints='',checkpoint_iterations=','.join(map(str,([1,instance['preflight_iterations']//2,instance['preflight_iterations']]
                                               if phase=='preflight' else instance['final_checkpoint_iterations']))))
    values.update(c['runner']);values.update({k:v for k,v in instance['prefactors'][method].items() if k!='id'})
    return values


def command(source,world,values):
    result=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc-per-node={world}',
            str(Path(source).resolve()/'extensions/pair_sampling_20261011/run_pair.py')]
    for key,value in values.items():result+=['--'+key.replace('_','-'),str(value)]
    return result


def expected_config(source,values):
    """Bind parser defaults as well as every explicit numerical argument."""
    spec=importlib.util.spec_from_file_location('_pair_parser',Path(source)/'extensions/pair_sampling_20261011/run_pair.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    words=[]
    for key,value in values.items():words+=['--'+key.replace('_','-'),str(value)]
    return vars(module.parser().parse_args(words))


def validate_metadata(metadata,expected,provenance,instance,tolerance):
    require(metadata['status']=='completed' and metadata['method_key']==expected['method'],'Runner did not complete requested method.')
    require(metadata['config']==expected,'Runner differs from frozen full argument namespace.')
    for key in ('source_file_sha256','instance_manifest_sha256','array_files_sha256'):
        require(metadata[key]==provenance[key],'Numerical provenance differs: '+key)
    require(metadata['dimensions']==dict(source=instance['n'],target=instance['n']),'Wrong full dataset dimensions.')
    require(metadata['environment']['world_size']==instance['world_size'],'Wrong production topology.')
    T=expected['iterations'];counter=metadata['counters']
    require(counter['completed_updates']==counter['stochastic_gradient_calls']==T,'Incomplete horizon or wrong gradient count.')
    require(counter['pair_samples']==T*expected['pairs_per_call'],'Pair-sampling work counter differs.')
    require(counter['sketch_probes']==0,'Pair sampling must not be reported as random-projection probes.')
    require(counter['inner_stopping_failures']==0,'An inner solve failed.')
    require(counter['initialization_oracle_calls']==int(expected['method']!='published-smd'),'Unexpected initialization gradient reuse.')
    oracle=metadata['gradient_oracle']
    require(oracle['name']=='pair-sampling' and oracle['pair_samples_per_call']==expected['pairs_per_call']
            and oracle['probes_per_call']==0 and oracle['target_marginals_substituted'] is False,'Incorrect pair oracle identity.')
    for key in ('mass_min','mass_max','mass_last'):require(finite(oracle[key],key)>0,'Invalid sampled coupling mass.')
    streams=metadata['random_streams']
    require(streams['pairs']['seed']==expected['seed'] and streams['pairs']['rank_zero_generator'] is True
            and streams['pairs']['with_replacement'] is True,'Wrong paired sampling random stream.')
    require(streams['output']['seed']==expected['output_seed'] and streams['output']['independent_generator'] is True
            and streams['initialization']['seed']==expected['init_seed'],'Output/initialization RNG differs.')
    ranks=metadata['ranks'];world=instance['world_size'];n=instance['n']
    require(len(ranks)==world and {r['rank'] for r in ranks}==set(range(world)),'Missing rank receipts.')
    for rank in ranks:
        require('A100' in (rank.get('gpu') or ''),'Production/preflight must use A100 GPUs.')
        require(set(rank['saved_plans'])==set(PLANS),'Missing saved output hashes.')
        for record in rank['saved_plans'].values():
            require(record['shape']==[n//world+int(rank['rank']<n%world),n] and record['dtype']=='float32','Wrong output shard shape/dtype.')
            require(record['persisted'] is False and record['file'] is None and record['file_sha256'] is None
                    and record['bytes']==0 and len(record['array_sha256'])==64,'Output storage/hash policy differs.')
    val=metadata['final_float64_validation']
    require(val['dtype']=='float64' and val['exact_objective'] is True and set(val['plans'])==set(PLANS),'Missing exact FP64 final audits.')
    for name,record in val['plans'].items():
        finite(record['loss'],name+' loss')
        residual=finite(record['marginal_residuals']['both_l1'],name+' residual',True)
        require(finite(record['marginal_residuals']['min_entry'],name+' minimum',True)>=0,'Negative output.')
        if name.endswith('_rounded'):require(residual<=tolerance,'Rounded output violates final feasibility.')
    inner=metadata['inner_trace']
    if expected['method']=='smoothing-fw':
        require(not inner and counter['inner_iterations_total']==counter['certificate_checks']==0,'Unexpected FW inner work.')
    else:
        require(len(inner)==T,'Missing per-step certificates.')
        for index,info in enumerate(inner):
            require(info['iteration']==index and info['converged'] is True and info['proximal_optimality_certified'] is True
                    and info['certificate_valid'] is True,'Rejected inner step.')
            require(finite(info['marginal_l1'],'stored-plan marginal error',True)<=1e-7,'Actual inner marginal tolerance failed.')
            require(finite(info['normalized_gap_upper_bound'],'guarded primal-dual gap',True)<=1e-7,'Inner optimality tolerance failed.')
            require(finite(info['normalized_primal_dual_gap'],'raw primal-dual gap')<=1e-7,'Raw inner gap failed.')
            require(finite(info['primal_witness_marginal_l1'],'FP64 primal witness',True)<=1e-12,'Infeasible primal witness.')
        require(sum(x['inner_iterations'] for x in inner)==counter['inner_iterations_total']
                and sum(x['certificate_checks'] for x in inner)==counter['certificate_checks'],'Inner work counter differs.')
    output=metadata['output_selection']
    require(0<=output['index']<T and output['horizon']==T and output['last_iterate_index']==T
            and output['independent_rng'] is True,'Invalid sampled theorem/diagnostic output.')
    require(output['output_seed']==expected['output_seed'],'Selected-output seed differs.')
    probability=(output['index']+1)**-.5/sum((k+1)**-.5 for k in range(T))
    require(math.isclose(output['probability'],probability,rel_tol=1e-12),'Incorrect independently weighted output probability.')
    if expected['method']=='published-msbpg':
        _msbpg_audit(metadata,expected)
    else:
        schedule=metadata['schedule']
        require((schedule['gamma_exponent'],schedule['rho_exponent'],schedule['mu_exponent'])==(.75,.5,.25),'FraMES schedule exponents differ.')
        if expected['method']=='published-smd':
            bound=finite(metadata['cmax'],'GW cost bound',True)
            for k,info in enumerate(inner):
                eta=expected['prefactor_fraction']*(.25/bound if bound>0 else 1.)/math.sqrt(k+1)
                require(math.isclose(info['eta'],eta,rel_tol=1e-12) and info['prefactor_fraction']==expected['prefactor_fraction'],
                        'SMD step does not match its published schedule.')
    for key,value in metadata['timing'].items():
        if isinstance(value,(int,float)):finite(value,'timing '+key,True)
    return val['plans']['last_rounded']['loss']


def directory(out,dataset,method,seed):
    return Path(out)/dataset/method/f'seed{seed}'


def run_cell(c,source,prepared,out,dataset,method,seed,phase,executor=execute,gate_receipt=None):
    target=directory(out,dataset,method,seed);target.mkdir(parents=True,exist_ok=False)
    provenance=identity(c,source,prepared,dataset)
    values=arguments(c,prepared,target/'output',dataset,method,seed,phase)
    expected=expected_config(source,values);cmd=command(source,c['instances'][dataset]['world_size'],values)
    request=dict(**provenance,stage=phase,instance=dataset,method=method,seed=seed,initialization='product',
                 expected_runner_config=expected,command=cmd,slurm_job_id=os.environ.get('SLURM_JOB_ID'))
    if gate_receipt is not None:request['gate']=gate_receipt
    write(target/'request.json',request);start=time.monotonic()
    try:
        timeout=c['preflight']['run_timeout_seconds'] if phase=='preflight' else c['run_timeout_seconds']
        code=executor(cmd,target/'runner.log',timeout)
        require(code==0,f'Runner failed with exit status {code}.')
        path=target/'output/metadata.json';metadata=read(path)
        loss=validate_metadata(metadata,expected,provenance,c['instances'][dataset],c['final_marginal_l1_tolerance'])
        status=dict(status='completed',returncode=code,metadata_file=str(path),metadata_sha256=sha256(path),
                    request_sha256=sha256(target/'request.json'),process_elapsed_seconds=time.monotonic()-start,last_rounded_loss=loss)
        write(target/'status.json',status)
        return metadata,status,provenance
    except Exception as exc:
        write(target/'status.json',dict(status='failed',returncode=locals().get('code'),error=repr(exc),traceback=traceback.format_exc(),
                                       process_elapsed_seconds=time.monotonic()-start))
        raise


def resource_checks(c,instance,metadata,status):
    timing=metadata['timing'];pilot=metadata['config']['iterations'];T=instance['final_iterations']
    solver=finite(timing['solver_seconds'],'pilot solver seconds',True)
    require(solver>0,'Pilot solver time must be positive.')
    diagnostic=sum(finite(timing.get(k,0.),k,True) for k in ('exact_diagnostic_seconds','repair_seconds'))
    logs=finite(timing.get('progress_logging_seconds',0.),'logging time',True)
    total=max(finite(timing['total_elapsed_seconds'],'pilot total',True),status['process_elapsed_seconds'])
    fixed=max(0.,total-solver-diagnostic-logs)
    count=len(metadata['trace']);require(count>0,'Missing pilot diagnostics.')
    pilotlogs=len(metadata['progress_reporting']['completed_updates_recorded']);require(pilotlogs>0,'Missing progress accounting.')
    finalcount=len(instance['final_checkpoint_iterations'])+3
    estimate=solver/pilot*T+diagnostic/count*finalcount+logs/pilotlogs*(T//c['runner']['progress_every']+finalcount)+fixed
    fractions=[phase['cuda_peak_reserved_bytes']/phase['cuda_total_memory_bytes']
               for rank in metadata['ranks'] for phase in rank['memory'].values()]
    reserved=max(fractions);bound=estimate*c['preflight']['runtime_safety_factor']
    require(reserved<=c['preflight']['maximum_gpu_reserved_fraction'],'Insufficient measured GPU memory headroom.')
    require(bound<=c['run_timeout_seconds'],'Full fixed-horizon forecast exceeds timeout with safety margin; production remains blocked.')
    return dict(passed=True,maximum_gpu_reserved_fraction=reserved,estimated_total_seconds=estimate,
                forecast_with_margin_seconds=bound,allowed_seconds=c['run_timeout_seconds'],
                includes='Solver, exact diagnostics, repairs, logging, initialization, validation, subprocess startup and serialization; no loss test.')


def check_receipt(path,source,world=None):
    record=read(path)
    require(record['status']=='passed','Correctness check failed.')
    require(record['source_file_sha256']==source_hashes(source),'Correctness checks used different source.')
    if world is not None:require(record['world_size']==world,'Correctness check used another topology.')
    return dict(file=str(Path(path).resolve()),sha256=sha256(path))


def preflight(args):
    c=load_config(args.config);out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    instance=c['instances'][args.instance];gate=dict(status='running',config_sha256=sha256(args.config),instance=args.instance,method=args.method)
    try:
        gate['cpu_checks']=check_receipt(args.cpu_checks,args.source)
        gate['gpu_checks']=check_receipt(args.gpu_checks,args.source,instance['world_size'])
        metadata,status,provenance=run_cell(c,args.source,args.prepared,out/'runs',args.instance,args.method,c['preflight']['seed'],'preflight')
        gate.update(provenance=provenance,calibration=status,resource_checks=resource_checks(c,instance,metadata,status),status='passed')
        write(out/'gate.json',gate)
    except Exception as exc:
        gate.update(status='failed',error=repr(exc),traceback=traceback.format_exc());write(out/'gate.json',gate);raise


def validate_gate(c,source,prepared,path,dataset,method):
    gate=read(path);provenance=identity(c,source,prepared,dataset)
    require(gate['status']=='passed' and gate['instance']==dataset and gate['method']==method,'Production gate missing or wrong method.')
    require(gate['provenance']==provenance and gate['config_sha256']==sha256(c['_config_path']),'Production gate provenance changed.')
    require(gate['resource_checks']['passed'] is True,'Resource gate failed.')
    for name in ('cpu_checks','gpu_checks'):
        receipt=gate[name]
        require(check_receipt(receipt['file'],source,c['instances'][dataset]['world_size'] if name=='gpu_checks' else None)==receipt,
                'Check receipt changed.')
    calibration=gate['calibration'];require(sha256(calibration['metadata_file'])==calibration['metadata_sha256'],'Preflight metadata changed.')
    expected=expected_config(source,arguments(c,prepared,Path(path).parent/'runs'/dataset/method/f"seed{c['preflight']['seed']}"/'output',
                                             dataset,method,c['preflight']['seed'],'preflight'))
    metadata=read(calibration['metadata_file'])
    validate_metadata(metadata,expected,provenance,c['instances'][dataset],c['final_marginal_l1_tolerance'])
    require(resource_checks(c,c['instances'][dataset],metadata,calibration)==gate['resource_checks'],'Resource forecast changed.')
    return dict(file=str(Path(path).resolve()),sha256=sha256(path))


def run_task(args):
    c=load_config(args.config);require(0<=args.task_index<len(c['seeds']),'Invalid seed task.')
    gate=validate_gate(c,args.source,args.prepared,args.gate,args.instance,args.method)
    run_cell(c,args.source,args.prepared,args.out,args.instance,args.method,c['seeds'][args.task_index],'production',gate_receipt=gate)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    for mode in ('preflight','run'):
        p=sub.add_parser(mode)
        for name in ('config','source','prepared','out','instance','method'):p.add_argument('--'+name,required=True)
        if mode=='preflight':
            p.add_argument('--cpu-checks',required=True);p.add_argument('--gpu-checks',required=True)
        else:p.add_argument('--gate',required=True);p.add_argument('--task-index',type=int,required=True)
    args=parser.parse_args();require(bool(os.environ.get('SLURM_JOB_ID')),'GPU campaign work requires a Slurm allocation.')
    (preflight if args.mode=='preflight' else run_task)(args)


if __name__=='__main__':main()
