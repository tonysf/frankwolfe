"""Bounded four-GPU exploratory test; pilot gates scientific comparisons."""
import argparse
import hashlib
import json
import os
import signal
import math
from pathlib import Path
import subprocess
import sys
import time


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def gate(result, world=4, headroom=.8):
    """Reject incomplete, malformed or nonfinite pilot evidence before scaling."""
    def finite(value, label, *, minimum=None):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise RuntimeError('Nonfinite or invalid pilot field: ' + label)
        if minimum is not None and value < minimum:
            raise RuntimeError('Out-of-range pilot field: ' + label)
        return value

    if not isinstance(world, int) or isinstance(world, bool) or world < 1:
        raise ValueError('A positive integer expected world size is required.')
    finite(headroom, 'headroom')
    if not 0 < headroom < 1:
        raise ValueError('Headroom fraction must be strictly between zero and one.')
    if result.get('status') not in ('completed', 'completed_with_inner_failures'):
        raise RuntimeError('Pilot did not complete scientific validation.')
    ranks = result.get('ranks', [])
    if result.get('environment', {}).get('world_size') != world or len(ranks) != world:
        raise RuntimeError('Unexpected distributed world size or number of rank records.')
    rank_ids = [rank.get('rank') for rank in ranks]
    if any(isinstance(rank, bool) or not isinstance(rank, int) for rank in rank_ids) or sorted(rank_ids) != list(range(world)):
        raise RuntimeError('Pilot must contain exactly one record for each distributed rank.')
    host_peaks = []
    for rank in ranks:
        if not isinstance(rank.get('gpu'), str) or 'A100' not in rank['gpu']:
            raise RuntimeError('Pilot must run on the requested A100 hardware.')
        phases = rank.get('memory', {})
        if not phases:
            raise RuntimeError('Pilot rank is missing memory measurements.')
        rank_host_peaks = []
        for phase_name, phase in phases.items():
            prefix = 'rank%d/%s/' % (rank['rank'], phase_name)
            reserved = finite(phase.get('cuda_peak_reserved_bytes'), prefix+'reserved_memory', minimum=0)
            capacity = finite(phase.get('cuda_total_memory_bytes'), prefix+'device_capacity', minimum=1)
            rss = finite(phase.get('cpu_process_peak_rss_kib'), prefix+'host_rss', minimum=0)
            rank_host_peaks.append(rss)
            if 'cuda_peak_allocated_bytes' in phase:
                allocated = finite(phase['cuda_peak_allocated_bytes'], prefix+'allocated_memory', minimum=0)
                if allocated > reserved:
                    raise RuntimeError('Allocated GPU memory cannot exceed reserved GPU memory.')
            if reserved >= headroom*capacity:
                raise RuntimeError('Insufficient measured GPU memory headroom.')
        host_peaks.append(max(rank_host_peaks))
    if sum(host_peaks) >= headroom*128*1024**2:
        raise RuntimeError('Insufficient host memory headroom.')
    updates = result.get('counters', {}).get('completed_updates')
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 1:
        raise RuntimeError('Pilot completed no valid number of updates.')
    validation = result.get('final_float64_validation', {})
    if validation.get('dtype') != 'float64' or validation.get('exact_objective') is not True:
        raise RuntimeError('Pilot lacks exact float64 final validation.')
    for field in ('raw_loss', 'rounded_loss'):
        finite(validation.get(field), field)
    residual = finite(validation.get('rounded', {}).get('both_l1'), 'rounded_both_l1', minimum=0)
    if residual > 100*1.1920928955078125e-7:
        raise RuntimeError('Pilot rounded output failed feasibility validation.')
    for kind in ('raw', 'rounded'):
        for field, value in validation.get(kind, {}).items():
            finite(value, kind+'_'+field, minimum=0)


def _source_hashes(source, names):
    source = Path(source)
    return {Path(name).name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names}


def _rank_records(result, world=4):
    ranks = result.get('ranks', [])
    identifiers = [item.get('rank') for item in ranks]
    if (len(ranks) != world or any(type(item) is not int for item in identifiers)
            or sorted(identifiers) != list(range(world))):
        raise RuntimeError('Expected exactly one result per distributed rank.')
    return {item['rank']: item for item in ranks}


def validate_result(result, expected_config, manifest, source):
    """Bind a successful launcher result to its actual requested experiment."""
    required = {'instance', 'out', 'method', 'device', 'budget', 'checkpoints', 'seed',
                'init_seed', 'initialization', 'batch', 'rho0', 'gamma0', 'mu_scale',
                'eta', 'inner_tol', 'inner_max', 'max_updates', 'exact_block_rows',
                'correspondence', 'correspondence_sha256', 'save_plans'}
    if set(expected_config) != required:
        raise ValueError('Expected configuration must cover every runner argument.')
    if result.get('status') not in ('completed', 'completed_with_inner_failures'):
        raise RuntimeError('Distributed run did not complete.')
    actual = result.get('config', {})
    if set(actual) != required or any(actual.get(key) != value for key, value in expected_config.items()):
        raise RuntimeError('Recorded runner configuration differs from the launched experiment.')
    hashes = manifest.get('array_files_sha256')
    if not isinstance(hashes, dict) or set(hashes) != {'A.npy','B.npy','a.npy','b.npy'}:
        raise RuntimeError('Frozen instance manifest is missing its exact array hashes.')
    if result.get('array_files_sha256') != hashes or result.get('instance_manifest') != manifest:
        raise RuntimeError('Result is not bound to the frozen mathematical instance.')
    if manifest.get('correspondence_sha256') != expected_config['correspondence_sha256']:
        raise RuntimeError('Correspondence differs from the frozen instance.')
    expected_sources = _source_hashes(source, ('run_sharded.py','sharded_gw.py','theory_config.json'))
    if result.get('source_file_sha256') != expected_sources:
        raise RuntimeError('Result source hashes differ from the staged implementation.')
    environment = result.get('environment', {})
    if environment.get('world_size') != 4 or environment.get('backend') != 'nccl':
        raise RuntimeError('Scientific comparison requires four NCCL ranks.')
    ranks = _rank_records(result)
    counters = result.get('counters', {})
    updates = counters.get('completed_updates')
    if type(updates) is not int or updates < 1:
        raise RuntimeError('Run must complete at least one update.')
    if (counters.get('stochastic_gradient_calls') != updates
            or counters.get('initialization_oracle_calls') != 1
            or counters.get('sketch_probes') != expected_config['batch'] * updates):
        raise RuntimeError('Oracle counters do not preserve first-estimate reuse.')
    validation = result.get('final_float64_validation', {})
    if validation.get('dtype') != 'float64' or validation.get('exact_objective') is not True:
        raise RuntimeError('Exact float64 validation is required.')
    for field in ('raw_loss','rounded_loss'):
        value = validation.get(field)
        if type(value) not in (int,float) or not math.isfinite(value):
            raise RuntimeError('Final validated loss must be finite.')
    residual = validation.get('rounded', {}).get('both_l1')
    if type(residual) not in (int,float) or not math.isfinite(residual) or not 0 <= residual <= 100*1.1920928955078125e-7:
        raise RuntimeError('Final rounded output must pass finite feasibility validation.')
    for value in result.get('timing', {}).values():
        if type(value) not in (int,float) or not math.isfinite(value) or value < 0:
            raise RuntimeError('Reported timing must be finite and nonnegative.')
    n, m = manifest['n'], manifest['m']
    for rank, record in ranks.items():
        if 'A100' not in str(record.get('gpu')):
            raise RuntimeError('Scientific comparison requires A100 GPUs.')
        initial_hash = record.get('initial_shard_sha256')
        if not isinstance(initial_hash, str) or len(initial_hash) != 64:
            raise RuntimeError('Initial shard must have a recorded SHA256.')
        plans = record.get('saved_plans', {})
        if set(plans) != {'P_raw','P_rounded'}:
            raise RuntimeError('Both current raw and rounded plans must be validated and hashed.')
        for name, plan in plans.items():
            if plan.get('shape') != [n//4 + int(rank < n%4), m]:
                raise RuntimeError('Saved plan shape disagrees with the global partition.')
            digest = plan.get('array_sha256')
            if not isinstance(digest, str) or len(digest) != 64:
                raise RuntimeError('Plan must have a recorded value hash.')
            if expected_config['save_plans']:
                expected_path = Path(expected_config['out'])/('rank%d' % rank)/(name+'.npy')
                if plan.get('file') != str(expected_path) or not expected_path.is_file():
                    raise RuntimeError('Requested final plan shard is missing.')
                if plan.get('bytes') != expected_path.stat().st_size:
                    raise RuntimeError('Saved plan file size does not match its record.')
                if not isinstance(plan.get('file_sha256'), str) or len(plan['file_sha256']) != 64:
                    raise RuntimeError('Saved plan is missing its file hash.')


def verify_pair(first, second):
    """Matched seeds share arrays, initial states, budgets and one GPU allocation."""
    methods = {first['config']['method'], second['config']['method']}
    if methods != {'smoothing-fw','stochastic-kl-momentum'}:
        raise RuntimeError('A pair must contain each stochastic method exactly once.')
    for key in ('seed','init_seed','initialization','batch','rho0','budget','checkpoints',
                'instance','correspondence','correspondence_sha256','max_updates'):
        if first['config'][key] != second['config'][key]:
            raise RuntimeError('Paired runs disagree on ' + key)
    if first['array_files_sha256'] != second['array_files_sha256']:
        raise RuntimeError('Paired runs used different mathematical arrays.')
    if first['source_file_sha256'] != second['source_file_sha256']:
        raise RuntimeError('Paired runs used different source implementations.')
    for key in ('world_size','backend','hostname','slurm_job_id'):
        if first['environment'].get(key) != second['environment'].get(key):
            raise RuntimeError('Paired runs used different compute allocations.')
    first_ranks, second_ranks = _rank_records(first), _rank_records(second)
    for rank in first_ranks:
        if first_ranks[rank]['initial_shard_sha256'] != second_ranks[rank]['initial_shard_sha256']:
            raise RuntimeError('Paired runs started from different coupling shards.')
        if first_ranks[rank]['gpu'] != second_ranks[rank]['gpu']:
            raise RuntimeError('Paired rank hardware differs.')


def correctness_gate(result, source):
    if result.get('status') != 'passed' or result.get('world_size') != 4:
        raise RuntimeError('Four-GPU correctness validation did not pass.')
    for key, expected in dict(n=17,m=17,batch=128,momentum_steps=5,
                              precision=['float64','float32'],tf32=False,
                              deterministic_algorithms=True).items():
        if result.get(key) != expected:
            raise RuntimeError('Correctness validation coverage differs from the frozen test.')
    checks = result.get('checks', [])
    if len(checks) != 352 or any(item.get('passed') is not True for item in checks) or result.get('failed_checks') != []:
        raise RuntimeError('All 352 independent distributed correctness checks must pass.')
    expected_sources = _source_hashes(source, ('gpu_correctness.py','sharded_gw.py','reference/gw_core.py'))
    if result.get('source_sha256') != expected_sources:
        raise RuntimeError('Correctness evidence does not match the staged source.')
    ranks = _rank_records(result)
    hosts = {record.get('hostname') for record in ranks.values()}
    if len(hosts) != 1 or None in hosts:
        raise RuntimeError('Correctness validation must use one node.')
    for rank, record in ranks.items():
        if ('A100' not in str(record.get('gpu_name')) or record.get('local_rank') != rank
                or record.get('device') != 'cuda:%d' % rank):
            raise RuntimeError('Correctness validation must use four distinct local A100 ranks.')



def validate_preparation(config, preparation, manifest, manifest_path):
    """Full-population replacement must use every cell of the frozen QC output."""
    n = config.get('expected_n')
    instance_id = config.get('instance_id')
    if type(n) is not int or n < 2 or instance_id != 'shareseq_full':
        raise ValueError('Expected the full SHARE-seq instance and an explicit dimension.')
    if preparation.get('status') != 'completed' or preparation.get('qc', {}).get('retained_paired_cells') != n:
        raise ValueError('Full instance must contain every previously QC-passing paired cell.')
    if manifest.get('id') != instance_id or manifest.get('n') != n or manifest.get('m') != n:
        raise ValueError('Instance identity/dimensions disagree with the frozen campaign.')
    instance_record = preparation.get('instances', {}).get(instance_id, {})
    if instance_record.get('metadata_sha256') != hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest():
        raise ValueError('Instance manifest does not match completed preparation.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True)
    p.add_argument('--prepared',required=True)
    p.add_argument('--out',required=True)
    args = p.parse_args()
    source, prepared, out = Path(args.source), Path(args.prepared), Path(args.out)
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Campaign must run inside Slurm.')
    config = json.loads((source/'campaign.json').read_text())
    if (len(config['methods']) != 2 or set(config['methods']) != {'smoothing-fw','stochastic-kl-momentum'}
        or config['batch'] != 128 or config['momentum_exponent'] != .5
        or config['fw']['step_exponent'] != .75 or config['fw']['smoothing_exponent'] != .25
        or config['kl']['check_every'] != 10 or config['initialization'] != 'product a*b^T'):
        raise ValueError('Campaign must preserve the frozen oracle, exponents, stopping checks and initialization.')
    instance = prepared/config['instance_id']
    manifest = json.loads((instance/'manifest.json').read_text())
    preparation = json.loads((prepared/'preparation.json').read_text())
    validate_preparation(config, preparation, manifest, instance/'manifest.json')
    labels = instance/'correspondence.npz'
    label_sha = hashlib.sha256(labels.read_bytes()).hexdigest()
    write(out/'frozen_campaign.json', dict(config=config,instance_manifest=manifest,
          prepared=str(prepared),source=str(source),correspondence_sha256=label_sha,
          source_manifest_sha256=hashlib.sha256((source/'SHA256SUMS').read_bytes()).hexdigest(),
          slurm_job_id=os.environ['SLURM_JOB_ID']))
    launcher = [sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc-per-node=4']
    started = time.perf_counter()
    commands = []

    def execute(arguments, name):
        command = launcher + arguments
        record = dict(name=name,command=command,started_unix=time.time())
        commands.append(record)
        write(out/'commands.json',commands)
        with (out/(name+'.log')).open('w') as log:
            process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                returncode = process.wait(timeout=2400)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                returncode = 124
        record.update(returncode=returncode,finished_unix=time.time())
        write(out/'commands.json',commands)
        if returncode:
            write(out/'failure.json',dict(status='failed',stage=name,returncode=returncode,log=str(out/(name+'.log'))))
            raise RuntimeError('Distributed stage failed: '+name)

    execute([str(source/'gpu_correctness.py'),'--out',str(out/'correctness.json'),'--device','cuda'], 'correctness')
    correctness = json.loads((out/'correctness.json').read_text())
    try:
        correctness_gate(correctness, source)
    except Exception as exc:
        write(out/'failure.json',dict(status='failed',stage='correctness_gate',error=repr(exc)))
        raise
    results = []
    for phase in ('pilot','final'):
        seeds = config[phase]['seeds']
        for index,seed in enumerate(seeds):
            pair_results = []
            methods = list(config['methods'])
            if index % 2:
                methods.reverse()
            for method in methods:
                name = f'{phase}_{seed}_{method}'
                destination = out/name
                arguments = [str(source/'run_sharded.py'),'--instance',str(instance),'--out',str(destination),
                    '--method',method,'--device','cuda','--budget',str(config[phase]['budget_seconds']),
                    '--checkpoints',','.join(map(str,config[phase]['checkpoints'])),
                    '--seed',str(seed),'--init-seed',str(config['initialization_seed']),
                    '--initialization','product','--batch',str(config['batch']),'--rho0',str(config['rho0']),
                    '--gamma0',str(config['fw']['gamma0']),'--mu-scale',str(config['fw']['mu_scale']),
                    '--eta',str(config['kl']['eta']),'--inner-tol',str(config['kl']['inner_tol']),
                    '--inner-max',str(config['kl']['inner_max']),
                    '--correspondence',str(labels),'--correspondence-sha256',label_sha]
                if phase == 'final':
                    arguments.append('--save-plans')
                execute(arguments,name)
                result = json.loads((destination/'metadata.json').read_text())
                expected_config = dict(instance=str(instance),out=str(destination),method=method,
                    device='cuda',budget=config[phase]['budget_seconds'],
                    checkpoints=','.join(map(str,config[phase]['checkpoints'])),seed=seed,
                    init_seed=config['initialization_seed'],initialization='product',batch=config['batch'],
                    rho0=config['rho0'],gamma0=config['fw']['gamma0'],mu_scale=config['fw']['mu_scale'],
                    eta=config['kl']['eta'],inner_tol=config['kl']['inner_tol'],inner_max=config['kl']['inner_max'],
                    max_updates=0,exact_block_rows=1024,correspondence=str(labels),
                    correspondence_sha256=label_sha,save_plans=phase=='final')
                try:
                    validate_result(result,expected_config,manifest,source)
                    if phase == 'pilot':
                        gate(result)
                    pair_results.append(result)
                    if len(pair_results) == 2:
                        verify_pair(*pair_results)
                except Exception as exc:
                    write(out/'failure.json',dict(status='failed',stage=name+'_validation',error=repr(exc)))
                    raise
                results.append(dict(phase=phase,seed=seed,method=method,path=str(destination),
                    status=result['status'],loss=result['final_float64_validation']['rounded_loss'],
                    updates=result['counters']['completed_updates'],timing=result['timing']))
                write(out/'progress.json',dict(status='running',results=results))
        if phase == 'pilot':
            write(out/'pilot_gate.json',dict(status='passed',maximum_reserved_fraction=.8,
                                            results=results.copy()))
    write(out/'completion.json',dict(status='completed',results=results,
          allocated_workload_seconds=time.perf_counter()-started,
          scope='six final runs; fixed transferred constants, no SHARE-seq tuning'))


if __name__ == '__main__':
    main()
