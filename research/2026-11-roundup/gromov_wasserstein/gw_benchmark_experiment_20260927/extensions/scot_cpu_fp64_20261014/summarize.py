"""Validate and report the completed, fixed 2,000-update float64 CPU pilot.

No optimization is run.  After provenance checks, saved plans are evaluated
using the original float64 geometry and an independent POT contraction.  A
fresh float64 repair is reported separately, never used to change selection.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HORIZON = 2000
CHECKPOINTS = [1, 10, 50, 100, 200, 500, 1000, 1500, 2000]
CANDIDATES = [dict(id='fw04', gamma0=.3, rho0=.3, mu_scale=1.),
              dict(id='fw05', gamma0=.3, rho0=.3, mu_scale=10.),
              dict(id='fw07', gamma0=1., rho0=.3, mu_scale=1.),
              dict(id='fw08', gamma0=1., rho0=.3, mu_scale=10.)]
SOURCES = {
    'published_stochastic_20261002/run_published.py',
    'published_stochastic_20261002/smd.py',
    'published_stochastic_20261002/certified_prox.py',
    'published_stochastic_20261002/inner_solver.json',
    'shareseq_25000_20260930/run_sharded.py',
    'shareseq_25000_20260930/sharded_gw.py',
    'msbpg_20261005/run_msbpg.py', 'msbpg_20261005/msbpg.py',
    'msbpg_20261005/stable_prox.py',
    'pair_sampling_20261011/run_pair.py', 'pair_sampling_20261011/oracle.py',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(left, right, label, tolerance=1e-10):
    require(finite(left) and finite(right) and abs(left-right) <= tolerance,
            f'{label}: independently evaluated value differs ({left!r}, {right!r}).')


def summary(values):
    require(bool(values) and all(finite(x) for x in values), 'Nonfinite/empty aggregate.')
    return dict(median=statistics.median(values), minimum=min(values), maximum=max(values), n=len(values))


def archived_sources(directory, hashes):
    require(isinstance(hashes, dict) and hashes, 'Missing source hashes.')
    for name, expected in hashes.items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts,
                'Invalid archived source path.')
        require(sha256(directory/relative) == expected, f'Archived source changed: {name}')


def execution_provenance(root, config, completed):
    """Accept overlap only with the explicit, source-bound user amendment."""
    intervals = {}
    confirmations = {r['seed']:r for r in completed['confirmation']}
    for seed, run in confirmations.items():
        start, finish = (dt.datetime.fromisoformat(run['receipt'][key]) for key in ('started_utc','finished_utc'))
        require(finish > start, 'Invalid confirmation execution interval.')
        intervals[seed] = (start, finish)
    path = root/'execution_amendment.json'
    if path.exists():
        amendment = read(path)
        amendment_hash = sha256(path)
        require(amendment['config_sha256'] == sha256(root/'local_config.json')
                and amendment['frozen_controller_sha256'] == config['controller_sha256']
                and amendment['parallel_controller_sha256'] == sha256(ROOT/'extensions/scot_cpu_fp64_20261014/parallel_confirm.py'),
                'Parallel execution amendment source/config binding changed.')
        require(amendment['remaining_seeds'] == [812,813] and amendment['existing_seed'] == 811
                and amendment['additional_workers'] == 2 and amendment['maximum_total_workers'] == 3
                and amendment['cpu_threads_per_process'] == 1
                and amendment['scientific_prescription_unchanged'] is True
                and amendment['comparison_axis'] == 'completed_updates'
                and amendment['confirmation_timings_comparable'] is False,
                'Parallel amendment changed scientific controls or timing policy.')
        require(isinstance(amendment['reason'],str) and amendment['reason']
                and isinstance(amendment['user_instruction'],str) and amendment['user_instruction'],
                'Missing explicit reason/user instruction for the execution amendment.')
        require(completed['execution_amendment_sha256'] == amendment_hash
                and completed['confirmation_timings_comparable'] is False,
                'Completion does not bind the timing waiver.')
        created = dt.datetime.fromisoformat(amendment['created_utc'])
        require(intervals[811][0] <= created and all(created <= intervals[s][0] for s in (812,813)),
                'The amendment must precede the two newly launched confirmations.')
        for seed in (812,813):
            require(read(root/f'confirmation_seed{seed}.json')['execution_amendment_sha256'] == amendment_hash,
                    'Parallel confirmation does not bind the amendment.')
        parallel = read(root/'stage_parallel_confirm.json')
        final = read(root/'stage_finalize_parallel.json')
        require(parallel['status'] == final['status'] == 'completed'
                and parallel['execution_amendment_sha256'] == final['execution_amendment_sha256'] == amendment_hash
                and parallel['confirmation'] == [confirmations[812],confirmations[813]]
                and final['completed_sha256'] == sha256(root/'completed.json'),
                'Parallel stage/final completion binding changed.')
        require(all(not (root/name).exists() for name in
                    ('stage_confirm_seed812.json','stage_confirm_seed813.json','stage_finalize.json')),
                'Ambiguous simultaneous sequential/parallel completion routes.')
        original_phases = ('tune','confirm_seed811')
        files = ['execution_amendment.json','stage_parallel_confirm.json','stage_finalize_parallel.json']
        policy = dict(mode='user_amended_parallel_confirmation',amendment=amendment,
                      execution_amendment_sha256=amendment_hash,maximum_confirmation_processes=3,
                      confirmation_timings_comparable=False,
                      analysis_axis='completed updates; no runtime comparison',
                      note='Seed811 continued its original stage; seeds812/813 ran as two additional one-thread workers. Scientific controls and selected prefactors stayed fixed.')
    else:
        require('execution_amendment_sha256' not in completed, 'Completion references a missing amendment.')
        ordered = sorted(intervals.values())
        require(all(a[1] <= b[0] for a,b in zip(ordered,ordered[1:])),
                'Overlapping confirmations require an explicit validated amendment.')
        original_phases = ('tune','confirm_seed811','confirm_seed812','confirm_seed813','finalize')
        files = []
        policy = dict(mode='original_sequential_confirmation',confirmation_timings_comparable=True,
                      maximum_confirmation_processes=1,analysis_axis='completed updates; no runtime comparison')
    for phase in original_phases:
        stage = read(root/f'stage_{phase}.json')
        require(stage['status'] == 'completed' and stage['stage'] == phase
                and stage['controller_sha256'] == config['controller_sha256'],
                'Missing or changed completed controller stage: '+phase)
        files.append(f'stage_{phase}.json')
    policy['input_sha256'] = {name:sha256(root/name) for name in files}
    policy['timing_comparison_performed'] = False
    return policy


def validate_design(root):
    config, selected, completed = (read(root/name) for name in
                                    ('local_config.json', 'selection.json', 'completed.json'))
    require(config['fixed_horizon'] == HORIZON and config['candidates'] == CANDIDATES,
            'The four-candidate, 2,000-update design changed.')
    require(config['tuning_seed'] == 810 and config['confirmation_seeds'] == [811, 812, 813]
            and config['checkpoint_iterations'] == CHECKPOINTS, 'Prescribed seeds/checkpoints changed.')
    require(config['threads_per_process'] == 1 and config['tuning_workers'] == 2
            and config['confirmation_workers'] == 1 and config['pairs_per_call'] == 1024,
            'Thread/oracle/parallelism design changed.')
    require(config['optimizer_dtype'] == 'float64' and config['final_evaluation_dtype'] == 'float64'
            and config['initialization'] == 'uniform product', 'Precision/start design changed.')
    require(config['controller_sha256'] == sha256(ROOT/'extensions/scot_cpu_fp64_20261014/compare_frames.py'),
            'Comparison controller changed since launch.')
    require(config['reference_sha256'] == sha256(root/'reference/reference.json')
            and config['baseline_sha256'] == sha256(root/'baseline_local/replay.json')
            and config['instance_manifest_sha256'] == sha256(root/'reference/prepared/scot1047/manifest.json'),
            'Frozen reference/baseline/manifest binding changed.')
    require(set(config['source_file_sha256']) == SOURCES
            and all(sha256(ROOT/'extensions'/name) == value for name,value in config['source_file_sha256'].items()),
            'Frozen numerical source identity differs.')
    require(config['baseline_dtype'] == 'float64'
            and config['checkpoint_seconds'] == '30,60,120,180,240,300,600', 'Precision/checkpoint policy changed.')
    metric = 'independent_float64_last_rounded_loss_at_fixed_horizon'
    require(config['selected_metric'] == selected['selection_metric'] == metric,
            'Selection must use the final saved rounded plan, not the best checkpoint.')
    require(selected['status'] == 'completed' and selected['config_sha256'] == sha256(root/'local_config.json'),
            'Selection is not bound to the completed frozen configuration.')
    require(completed['status'] == 'completed' and completed['config'] == config
            and completed['selected'] == selected['selected'] and completed['tuning'] == selected['tuning'],
            'Completion and selection records disagree.')
    require(len(completed['tuning']) == 4 and len(completed['confirmation']) == 3,
            'Expected four tuning and three confirmation runs.')
    require([r['candidate'] for r in completed['tuning']] == CANDIDATES
            and all(r['seed'] == 810 for r in completed['tuning']), 'Tuning cells differ.')
    require([r['seed'] for r in completed['confirmation']] == [811, 812, 813]
            and all(r['candidate'] == selected['selected'] for r in completed['confirmation']),
            'Fresh confirmation cells differ.')
    require(all(finite(r['last_rounded_loss_float64']) for r in completed['tuning']),
            'Invalid tuning endpoint cannot enter selection.')
    winner = min(completed['tuning'], key=lambda r: r['last_rounded_loss_float64'])['candidate']
    require(winner == selected['selected'], 'Selection is not the prescribed best final tuning value.')
    for run in completed['confirmation']:
        stage = read(root/f'confirmation_seed{run["seed"]}.json')
        require(stage['status'] == 'completed' and stage['run'] == run
                and stage['config_sha256'] == sha256(root/'local_config.json')
                and stage['selection_sha256'] == sha256(root/'selection.json'), 'Confirmation selection binding changed.')
    execution = execution_provenance(root,config,completed)
    return config, selected, completed, execution


def trajectory(rows, solver_seconds):
    require(isinstance(rows, list) and bool(rows), 'Missing observations.')
    result, previous_iteration, previous_solver = {}, -1, -1.
    for row in rows:
        k, clock, output = row['iteration'], row['actual_solver_seconds'], row['time_to_this_feasible_output']
        require(type(k) is int and 0 <= k <= HORIZON and k >= previous_iteration,
                'Invalid/nonmonotone observation index.')
        require(all(finite(x) for x in (clock, output, row['rounded_loss'], row['raw_loss'], row['repair_seconds_current'])),
                'Nonfinite observation.')
        require(clock >= previous_solver and 0 <= clock <= solver_seconds+1e-8 and output >= clock,
                'Invalid observation timing.')
        close(output, clock+row['repair_seconds_current'], 'Output clock', 1e-8)
        if k in result:
            close(row['rounded_loss'], result[k]['rounded_loss'], 'Repeated-iteration rounded loss', 1e-12)
            close(row['raw_loss'], result[k]['raw_loss'], 'Repeated-iteration raw loss', 1e-12)
        # Selected/current/last can coincide; retain the final occurrence once.
        result[k] = dict(iteration=k, rounded_loss=row['rounded_loss'], raw_loss=row['raw_loss'],
                         actual_solver_seconds=clock, time_to_this_feasible_output=output,
                         repair_seconds_current=row['repair_seconds_current'],
                         raw_residuals=row['raw_residuals'], rounded_residuals=row['rounded_residuals'])
        previous_iteration, previous_solver = k, clock
    require(set([0]+CHECKPOINTS).issubset(result) and rows[-1]['kind'] == 'last'
            and rows[-1]['iteration'] == HORIZON, 'Missing prescribed checkpoints or final observation.')
    return [result[k] for k in sorted(result)]


def validate_run(root, phase, cell, reference, baseline_hostname):
    candidate, seed = cell['candidate'], cell['seed']
    name = f'{phase}_{candidate["id"]}_seed{seed}'
    directory = root/'runs'/name
    require(Path(cell['directory']).resolve() == directory.resolve(), 'Run directory differs from prescribed cell.')
    metadata_path, receipt_path = directory/'metadata.json', root/f'{name}_receipt.json'
    metadata, receipt = read(metadata_path), read(receipt_path)
    require(receipt == cell['receipt'] and receipt['name'] == name and receipt['exit_code'] == 0
            and receipt['cpu_threads_per_process'] == 1 and receipt['concurrent_tuning'] == (phase == 'tune'),
            'Run receipt/configuration mismatch.')
    frozen = read(root/'local_config.json')
    binding = {key:frozen[key] for key in ('reference_sha256','baseline_sha256','instance_manifest_sha256','array_files_sha256')}
    identity = {key:frozen[key] for key in ('source_commit','controller_sha256','source_file_sha256')}
    require(receipt['status'] == 'completed' and receipt['optimizer_dtype'] == 'float64'
            and receipt['config_sha256'] == sha256(root/'local_config.json')
            and receipt['input_binding'] == binding and receipt['source_identity'] == identity
            and receipt['metadata_sha256'] == cell['metadata_sha256'] == sha256(metadata_path),
            'FP64 run/config/input/source/metadata binding changed.')
    require(receipt['thread_environment'] == dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                                                 MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1'),
            'FP64 worker thread environment differs.')
    require(finite(receipt['process_elapsed_seconds']) and receipt['process_elapsed_seconds'] > 0,
            'Invalid process time.')
    require(metadata['status'] == 'completed' and metadata['method_key'] == 'smoothing-fw',
            'Incomplete or wrong method.')
    require(set(metadata['source_file_sha256']) == SOURCES, 'Unexpected numerical source inventory.')
    for name_, expected in metadata['source_file_sha256'].items():
        require(sha256(ROOT/'extensions'/name_) == expected, f'Numerical source changed: {name_}')
    config = metadata['config']
    expected_config = dict(method='smoothing-fw', device='cpu', dtype='float64', iterations=HORIZON,
                           seed=seed, output_seed=seed+20000, init_seed=seed+10000,
                           initialization='product', oracle='pair-sampling', pairs_per_call=1024,
                           save_plans='all', progress_every=1000,
                           checkpoint_iterations=','.join(map(str, CHECKPOINTS)),
                           checkpoints='30,60,120,180,240,300,600',
                           **{k:v for k,v in candidate.items() if k != 'id'})
    require(all(config.get(k) == v for k,v in expected_config.items()), 'Numerical settings differ from the frozen cell.')
    require(Path(config['out']).resolve() == directory.resolve()
            and Path(config['instance']).resolve() == (root/'reference/prepared/scot1047').resolve(),
            'Run input/output location mismatch.')
    command = receipt['command']
    require(Path(command[1]).resolve() == (ROOT/'extensions/pair_sampling_20261011/run_pair.py').resolve()
            and len(command[2:]) % 2 == 0, 'Unexpected runner command.')
    options = dict(zip(command[2::2], command[3::2]))
    require(len(options) == len(command[2:])//2 and all(key.startswith('--') for key in options),
            'Duplicate or malformed runner arguments.')
    for key, value in options.items():
        field = key[2:].replace('-', '_')
        require(field in config and str(config[field]) == value, f'Command/metadata differ: {key}')
    c = metadata['counters']
    require(c['completed_updates'] == c['stochastic_gradient_calls'] == HORIZON
            and c['pair_samples'] == HORIZON*1024 and c['sketch_probes'] == 0
            and c['initialization_oracle_calls'] == 1
            and c['inner_iterations_total'] == c['inner_stopping_failures'] == 0,
            'Invalid update/oracle/initialization counts.')
    oracle = metadata['gradient_oracle']
    require(oracle['name'] == 'pair-sampling' and oracle['pair_samples_per_call'] == 1024
            and oracle['probes_per_call'] == 0 and oracle['target_marginals_substituted'] is False
            and oracle['gradient_arithmetic'] == 'float64',
            'Wrong pair-gradient oracle.')
    require(all(finite(oracle[k]) and oracle[k] > 0 for k in ('mass_min','mass_max','mass_last')),
            'Invalid actual sampling mass.')
    environment = metadata['environment']
    require(environment['hostname'] == baseline_hostname and environment['world_size'] == 1
            and environment['backend'] == 'gloo' and environment['deterministic_algorithms'] is True,
            'CPU topology/host/determinism mismatch.')
    require(len(metadata['ranks']) == 1 and metadata['ranks'][0]['rank'] == 0
            and metadata['ranks'][0]['device'] == 'cpu' and metadata['ranks'][0]['gpu'] is None,
            'Expected exactly one CPU rank.')
    prepared = reference['prepared_instance']
    require(metadata['array_files_sha256'] == prepared['array_files_sha256']
            and metadata['instance_manifest_sha256'] == prepared['manifest_sha256']
            and frozen['array_files_sha256'] == prepared['array_files_sha256'],
            'Reference/run input hashes differ.')
    validation = metadata['final_float64_validation']
    require(validation['dtype'] == 'float64' and validation['exact_objective'] is True,
            'Missing full float64 endpoint validation.')
    require(cell['last_rounded_loss_float64'] == validation['plans']['last_rounded']['loss'],
            'Selection receipt does not match final endpoint.')
    rank = metadata['ranks'][0]
    require(read(directory/'rank0/metadata.json') == rank, 'Rank metadata differs from top-level record.')
    files = {}
    for key in ('last_raw', 'last_rounded'):
        saved = rank['saved_plans'][key]
        path = directory/'rank0'/f'{key}.npy'
        require(saved['persisted'] is True and saved['shape'] == [1047,1047]
                and saved['dtype'] == 'float64' and Path(saved['file']).resolve() == path.resolve()
                and sha256(path) == saved['file_sha256'] and path.stat().st_size == saved['bytes'],
                f'Saved plan binding failed: {key}')
        files[key] = dict(path=str(path), sha256=sha256(path), array_sha256=saved['array_sha256'])
    timing = metadata['timing']
    require(all(finite(v) and v >= 0 for v in timing.values()), 'Invalid timing component.')
    require(timing['total_elapsed_seconds'] >= timing['solver_seconds']
            and receipt['process_elapsed_seconds'] >= timing['total_elapsed_seconds'], 'Elapsed time smaller than solver work.')
    trace = trajectory(metadata['trace'], timing['solver_seconds'])
    return dict(name=name, phase=phase, seed=seed, candidate=candidate, metadata_sha256=sha256(metadata_path),
                receipt_sha256=sha256(receipt_path), config=config, counters=c, timing=timing,
                process_elapsed_seconds=receipt['process_elapsed_seconds'], environment=environment,
                source_file_sha256=metadata['source_file_sha256'], saved_files=files,
                initial_plan_sha256=rank['initial_shard_sha256'], memory=rank['memory'],
                recorded_float64=validation['plans'], observations=trace,
                saved_rounded_time_to_output_seconds=trace[-1]['time_to_this_feasible_output'],
                timing_eligible_for_comparison=False,
                timing_scope='Raw archived measurements only; iteration-based assessment requested.')


def validate_baseline(root, reference):
    directory = root/'baseline_local'
    path = directory/'replay.json'
    baseline = read(path)
    require(baseline['status'] == 'passed' and baseline['gate']['passed'], 'Local reference did not pass.')
    require(baseline['reference_sha256'] == sha256(root/'reference/reference.json'), 'Local reference binds different source result.')
    require(baseline['prepared_instance']['array_files_sha256'] == reference['prepared_instance']['array_files_sha256']
            and baseline['prepared_instance']['manifest_sha256'] == reference['prepared_instance']['manifest_sha256'],
            'Local reference geometry differs.')
    archived_sources(directory, baseline['source_file_sha256'])
    require(baseline['source_file_sha256']['provenance/reference.py'] == sha256(ROOT/'extensions/scot_reproduction_20261014/reference.py')
            and baseline['source_file_sha256']['provenance/replay_reference.py'] == sha256(ROOT/'extensions/scot_cpu_20261014/replay_reference.py'),
            'Reference evaluation/replay source changed.')
    for name, key in [('P_raw.npy','raw_plan_sha256'), ('P_rounded.npy','rounded_plan_sha256'),
                      ('pot_log.json','pot_log_sha256'), ('numpy_configuration.txt','numpy_configuration_sha256')]:
        require(sha256(directory/name) == baseline[key], f'Local reference artifact changed: {name}')
    require(baseline['solver']['effective_overrides'] == dict(epsilon=.001,loss_fun='square_loss',log=True,verbose=False),
            'Original reference solver settings changed.')
    threads = baseline['environment']['thread_environment']
    require(all(threads[k] == '1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')),
            'Reference thread configuration differs.')
    require(all(finite(v) and v >= 0 for v in baseline['timing'].values()), 'Invalid reference timing.')
    return baseline, sha256(path)


def evaluate_saved(ref, np, ot, arrays, path, recorded, expected_array_hash=None):
    values = np.load(path, allow_pickle=False)
    require(values.shape == (1047,1047) and values.dtype == np.dtype('float64'),
            'Invalid saved coupling shape/dtype.')
    if expected_array_hash is not None:
        require(hashlib.sha256(values.tobytes()).hexdigest() == expected_array_hash, 'Saved coupling content hash differs.')
    A,B,a,b = arrays
    values64 = values.astype(np.float64)
    metrics = ref.evaluate(A,B,values64,a,b,ot)
    require(metrics['independent_loss_agrees'], 'Independent full-objective contractions disagree.')
    close(metrics['loss'], recorded['loss'], 'Saved-plan full loss')
    expected_residuals = recorded.get('marginal_residuals', recorded.get('residuals'))
    for name, value in expected_residuals.items():
        close(metrics['residuals'][name], value, 'Saved-plan residual '+name, 1e-12)
    return values64, metrics


def numerical_validation(ref, root, baseline, records):
    import numpy as np
    import ot
    arrays = tuple(np.load(root/'reference/prepared/scot1047'/name, allow_pickle=False) for name in ref.ARRAY_NAMES)
    require(all(x.dtype == np.float64 and np.isfinite(x).all() for x in arrays), 'Original geometry is not finite float64.')
    A,B,a,b = arrays
    initial = ref.evaluate(A,B,np.outer(a,b),a,b,ot)
    close(initial['loss'], baseline['initial']['loss'], 'Initial objective')
    expected_initial = hashlib.sha256(np.outer(a,b).tobytes()).hexdigest()
    for key in ('raw','rounded'):
        _, metric = evaluate_saved(ref,np,ot,arrays,root/'baseline_local'/f'P_{key}.npy',baseline[key])
        baseline['independent_'+key] = metric
    require(baseline['independent_rounded']['residuals']['both_l1'] <= 100*np.finfo(np.float64).eps,
            'Reference repaired plan exceeds the strict float64 feasibility tolerance.')
    for row in records:
        require(row['initial_plan_sha256'] == expected_initial, 'Uniform product initialization hash differs.')
        for key in ('last_raw','last_rounded'):
            f = row['saved_files'][key]
            values, metric = evaluate_saved(ref,np,ot,arrays,Path(f['path']),row['recorded_float64'][key],f['array_sha256'])
            row[key+'_independent'] = metric
            if key == 'last_raw':
                before = time.perf_counter()
                repaired = ref.round_transport(values,a,b)
                repair_seconds = time.perf_counter()-before
                repaired_metrics = ref.evaluate(A,B,repaired,a,b,ot)
                require(repaired_metrics['independent_loss_agrees']
                        and repaired_metrics['residuals']['min_entry'] >= 0
                        and repaired_metrics['residuals']['both_l1'] <= ref.FEASIBILITY_TOLERANCE,
                        'Auxiliary NumPy repair fails the reference helper feasibility criterion.')
                row['additional_float64_repair'] = dict(
                    **repaired_metrics, repair_entrywise_l1=float(np.abs(repaired-values).sum()),
                    repair_seconds=repair_seconds, array_sha256=hashlib.sha256(repaired.tobytes()).hexdigest(),
                    feasibility_tolerance_both_l1=ref.FEASIBILITY_TOLERANCE,
                    passes_saved_output_strict_tolerance=bool(
                        repaired_metrics['residuals']['both_l1'] <= 100*np.finfo(np.float64).eps),
                    role='Auxiliary post-run diagnostic; the actual saved rounded plan is the primary endpoint and separately must pass 100*eps64.',
                    source='Saved float64 raw plan; independently repaired once after original runs, without optimization.',
                    timing_scope='Postprocessing repair measurement; not inserted into the original trajectory clock.')
        require(row['last_rounded_independent']['residuals']['both_l1'] <= 100*np.finfo(np.float64).eps,
                'Actual saved rounded plan fails the strict float64 feasibility tolerance.')
        row['saved_rounding_loss_minus_additional_float64_repair'] = (
            row['last_rounded_independent']['loss']-row['additional_float64_repair']['loss'])
    return initial, dict(numpy=np.__version__,POT=ot.__version__)


def aggregate(records, baseline_loss, initial_loss):
    result = {}
    for field, key in [('saved_rounded','last_rounded_independent'), ('auxiliary_float64_repair','additional_float64_repair')]:
        losses = [r[key]['loss'] for r in records]
        result[field] = dict(loss=summary(losses),
            reduction_percent=summary([100*(1-x/initial_loss) for x in losses]),
            loss_minus_local_reference=summary([x-baseline_loss for x in losses]),
            relative_excess_over_local_reference_percent=summary([100*(x/baseline_loss-1) for x in losses]),
            rounded_both_l1=summary([r[key]['residuals']['both_l1'] for r in records]))
    return result


def precision_comparison(path, report):
    """Compare prior verified FP32 results without rerunning old numerics."""
    path = Path(path).resolve()
    old = read(path)
    old_root = Path(old['input_root']).resolve()
    require(old['status'] == 'complete_validated' and old['fixed_horizon'] == HORIZON
            and old['expected_confirmation_count'] == old['observed_confirmation_count'] == 3,
            'Prior float32 report is incomplete or describes a different horizon.')
    require(old['script_sha256'] == sha256(ROOT/'extensions/scot_cpu_20261014/summarize.py'),
            'Prior float32 reporting source changed.')
    for name, expected in old['input_sha256'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts
                and sha256(old_root/relative) == expected, 'Prior float32 input binding changed.')
    require(old['reference_array_files_sha256'] == report['reference_array_files_sha256']
            and old['baseline_replay_sha256'] == report['baseline_replay_sha256'],
            'Float32/float64 reports use different geometry or local reference replay.')
    old_config = old['configuration']
    require(old_config['optimizer_dtype'] == 'float32' and old_config['final_evaluation_dtype'] == 'float64',
            'Prior report is not the float32 optimizer experiment.')
    for key in ('candidates','fixed_horizon','tuning_seed','confirmation_seeds','pairs_per_call',
                'initialization','checkpoint_iterations','selected_metric','threads_per_process',
                'tuning_workers','confirmation_workers'):
        require(old_config[key] == report['configuration'][key], 'Precision comparison design differs: '+key)
    old_selection = read(old_root/'selection.json')
    require(old_selection['selected'] == old['selected_candidate'], 'Prior selected candidate changed.')
    old_runs = {r['seed']:r for r in old['runs'] if r['phase'] == 'confirm'}
    new_runs = {r['seed']:r for r in report['runs'] if r['phase'] == 'confirm'}
    require(len([r for r in old['runs'] if r['phase']=='confirm']) == 3
            and set(old_runs) == set(new_runs) == {811,812,813}, 'Precision comparison seed coverage differs.')
    rows = []
    for seed in sorted(old_runs):
        previous, current = old_runs[seed], new_runs[seed]
        directory = old_root/'runs'/previous['name']
        require(sha256(directory/'metadata.json') == previous['metadata_sha256']
                and sha256(old_root/f'{previous["name"]}_receipt.json') == previous['receipt_sha256'],
                'Prior confirmation metadata/receipt changed.')
        metadata = read(directory/'metadata.json')
        require(previous['config']['dtype'] == 'float32'
                and previous['candidate'] == old['selected_candidate']
                and metadata['status'] == 'completed'
                and previous['source_file_sha256'] == current['source_file_sha256'] == metadata['source_file_sha256'],
                'Precision comparison numerical sources or old selected setting differ.')
        for key in ('last_raw','last_rounded'):
            saved = previous['saved_files'][key]
            require(sha256(directory/'rank0'/f'{key}.npy') == saved['sha256'],
                    'Prior saved plan binding changed.')
            close(previous[key+'_independent']['loss'],
                  metadata['final_float64_validation']['plans'][key]['loss'],
                  'Prior independently evaluated endpoint')
        old_loss, new_loss = previous['last_rounded_independent']['loss'], current['last_rounded_independent']['loss']
        old_repaired, new_repaired = previous['additional_float64_repair']['loss'], current['additional_float64_repair']['loss']
        rows.append(dict(seed=seed, float32_candidate=previous['candidate']['id'],
            float64_candidate=current['candidate']['id'], same_selected_candidate=previous['candidate']==current['candidate'],
            float32_saved_rounded_loss_evaluated_float64=old_loss,
            float64_saved_rounded_loss=new_loss, saved_loss_float64_minus_float32=new_loss-old_loss,
            float32_additional_float64_repair_loss=old_repaired,
            float64_additional_float64_repair_loss=new_repaired,
            auxiliary_repaired_loss_float64_minus_float32=new_repaired-old_repaired,
            float32_run_metadata_sha256=previous['metadata_sha256'],
            float64_run_metadata_sha256=current['metadata_sha256']))
    return dict(status='descriptive_verified_artifact_comparison', float32_report_path=str(path),
        float32_report_sha256=sha256(path), expected_seed_count=3, observed_seed_count=3,
        float32_selected_candidate=old['selected_candidate'],float64_selected_candidate=report['selected_candidate'],
        selected_candidate_changed=old['selected_candidate'] != report['selected_candidate'],
        selection_policy='Each precision independently selects the best final saved-rounded objective from the same four candidates at development seed810; no confirmation-based choice.',
        float32_confirmation=aggregate(list(old_runs.values()),report['baseline']['independent_rounded']['loss'],report['initial']['loss']),
        float64_confirmation=report['confirmation_summary'], seed_matched_endpoints=rows,
        caveat='Matching seed numbers are descriptive bookkeeping, not identical sampled-pair paths. The sampling law depends on each evolving coupling, so precision and independently selected prefactors can change later samples. No precision-isolated causal or paired-path equivalence claim.')


def common_curve(records):
    maps = [{r['iteration']:r for r in run['observations']} for run in records]
    common = sorted(set.intersection(*(set(m) for m in maps)))
    require(set([0]+CHECKPOINTS).issubset(common), 'Confirmation common-grid coverage is incomplete.')
    return [dict(iteration=k, expected_seed_count=3, observed_seed_count=len(maps),
                 rounded_loss_median=statistics.median(m[k]['rounded_loss'] for m in maps),
                 rounded_loss_minimum=min(m[k]['rounded_loss'] for m in maps),
                 rounded_loss_maximum=max(m[k]['rounded_loss'] for m in maps)) for k in common]


def plot(report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(figsize=(9.4,5.8))
    colors = ['#0072B2','#009E73','#D55E00']
    confirmations = [r for r in report['runs'] if r['phase']=='confirm']
    curve = report['common_iteration_curve']
    baseline_loss = report['baseline']['independent_rounded']['loss']
    for color, record in zip(colors,confirmations):
        rows = record['observations']
        x = [r['iteration'] for r in rows]
        ax.plot(x,[r['rounded_loss'] for r in rows],color=color,lw=1.25,alpha=.75)
        ax.scatter(x[-1],record['last_rounded_independent']['loss'],s=34,facecolors='white',edgecolors=color,zorder=6)
    x = [r['iteration'] for r in curve]
    ax.fill_between(x,[r['rounded_loss_minimum'] for r in curve],
                    [r['rounded_loss_maximum'] for r in curve],color='#666666',alpha=.12)
    ax.plot(x,[r['rounded_loss_median'] for r in curve],color='#222222',lw=2,marker='.',ms=5)
    ax.axhline(baseline_loss,color='#8E44AD',ls='--',lw=1.5)
    ax.set_xlabel('Completed FraMES updates')
    ax.grid(alpha=.2)
    ax.margins(x=.035,y=.09)
    ax.set_ylabel('Full squared GW objective')
    ax.set_xlim(left=0,right=HORIZON*1.035)
    handles = [Line2D([0],[0],color=c,lw=1.5,label=f'FraMES seed {r["seed"]}') for c,r in zip(colors,confirmations)]
    handles += [Line2D([0],[0],color='#222222',lw=2,label='Median; band = range of 3 seeds'),
                Line2D([0],[0],color='#8E44AD',ls='--',label='SCOT/POT feasible reference (float64)'),
                Line2D([0],[0],color='#555555',marker='o',markerfacecolor='white',ls='',label='Saved endpoint evaluated in float64')]
    fig.suptitle('FraMES on SCOT1047 — float64, 2,000 updates',fontsize=14,y=.99)
    fig.text(.5,.935,'1,024 iid coupling pairs per update + gradient momentum; three fresh confirmation seeds',ha='center',fontsize=9.5)
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.065),ncol=2,frameon=False,fontsize=9)
    fig.text(.5,.018,'Current float64 repaired outputs; median on common logged iterations. Band = seed range, not a confidence interval.\n'
             'Iteration-based assessment only; archived runtimes are not compared.',
             ha='center',fontsize=8.3)
    fig.subplots_adjust(top=.87,bottom=.27,left=.105,right=.98)
    for suffix in ('png','pdf'):
        fig.savefig(out/f'scientific_plot.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def report_text(report):
    baseline, a = report['baseline'], report['confirmation_summary']
    saved = a['saved_rounded']
    auxiliary = a['auxiliary_float64_repair']
    initial = report['initial']['loss']
    comparison = report['precision_comparison']
    return '\n'.join([
        'Local SCOT1047 float64 CPU comparison: complete, validated; 2,000 updates per FraMES run.',
        f'Selected prefactors (best final tuning loss, seed810): {report["selected_candidate"]}',
        'FraMES = smoothed row Frank-Wolfe + momentum, with 1,024 iid coupling-pair samples per update.',
        f'Initial full GW loss: {initial:.12g}',
        f'Local SCOT/POT repaired feasible loss: {baseline["independent_rounded"]["loss"]:.12g}',
        f'Confirmation saved-rounded float64 loss median/range: {a["saved_rounded"]["loss"]}',
        f'Saved feasible percentage reduction median/range: {saved["reduction_percent"]}',
        f'Saved feasible loss minus local reference median/range: {saved["loss_minus_local_reference"]}',
        f'Auxiliary NumPy float64 repair loss median/range: {auxiliary["loss"]}',
        f'Local reference warnings: {len(baseline["warnings"])}; raw marginal L1: {baseline["raw"]["residuals"]["both_l1"]:.6g}',
        f'Saved-output combined marginal L1 tolerance: {report["strict_feasibility_tolerance_both_l1"]:.8g} (100 times float64 machine epsilon)',
        f'Auxiliary NumPy repair tolerance: {report["auxiliary_repair_feasibility_tolerance_both_l1"]:.8g}, the unchanged reference helper criterion; auxiliary repairs exceeding the saved-output bound: {report["auxiliary_repair_strict_bound_exceedance_count"]}/7.',
        '',
        f'Previous float32 selected candidate: {comparison["float32_selected_candidate"]}',
        f'Float64 independently retuned selected candidate: {comparison["float64_selected_candidate"]}',
        f'Selected candidate changed: {comparison["selected_candidate_changed"]}',
        f'Previous float32 confirmation saved-rounded loss median/range (float64 evaluation): {comparison["float32_confirmation"]["saved_rounded"]["loss"]}',
        f'New float64 confirmation saved-rounded loss median/range: {comparison["float64_confirmation"]["saved_rounded"]["loss"]}',
        comparison['caveat'],
        '',
        f'Execution: {report["execution"]["mode"]}. Assessment uses completed iterations only; no CPU timing comparison is performed. Raw timing observations remain archived in JSON.',
        'Both the NumPy/POT reference and PyTorch/FraMES optimizer, gradients, momentum, repairs and diagnostics use float64 on the original float64 distances. Precision is matched; numerical libraries and algorithms still differ.',
        'Online curves show actual current repaired outputs, not best-so-far values. Medians use common logged iterations. The SCOT/POT endpoint is an attainable-loss reference, not a matched-iteration competitor.',
        'The additional float64 repair uses the saved raw endpoint after the run. Its loss/residual/time are reported separately, never substituted into the original trajectory, selection or timing. The saved-output tolerance remains 100*eps64; the independent NumPy repair uses its original 1e-12 criterion because its reduction and clipping roundoff differs.',
        'Three confirmation seeds on one fixed dataset; range is not a confidence interval. This pilot does not establish a general performance ranking or a last-iterate theorem.',
        '',
        'Artifacts: report.json, scientific_plot.png, scientific_plot.pdf, curve_data.csv, endpoints.csv, precision_comparison.json, precision_comparison.csv.',
        '',
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'experiments/scot_cpu_fp64_20261014')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--float32-report',type=Path,default=ROOT/'experiments/scot_cpu_20261014/report/report.json')
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.resolve() if args.out else root/'report'
    # This process has no optimizer and must not launch threaded diagnostics.
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    started = time.perf_counter()
    config, selection, completed, execution = validate_design(root)
    path = ROOT/'extensions/scot_reproduction_20261014/reference.py'
    spec = importlib.util.spec_from_file_location('_scot_cpu_report_reference',path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    reference = ref.verify_reference(root/'reference')
    baseline, baseline_hash = validate_baseline(root,reference)
    records = [validate_run(root,phase,cell,reference,baseline['environment']['hostname'])
               for phase,key in [('tune','tuning'),('confirm','confirmation')] for cell in completed[key]]
    expected_names = {r['name'] for r in records}
    require({p.name for p in (root/'runs').iterdir() if p.is_dir()} == expected_names,
            'Unexpected/missing scientific run directories.')
    require(len({r['initial_plan_sha256'] for r in records}) == 1, 'Initial plans differ across runs.')
    require(all(r['source_file_sha256'] == records[0]['source_file_sha256'] for r in records),
            'Numerical source versions differ across runs.')
    # Matrix work starts only after every source, input, receipt and plan-file hash check.
    initial, evaluation_environment = numerical_validation(ref,root,baseline,records)
    confirmations = [r for r in records if r['phase']=='confirm']
    report = dict(schema_version=1,status='complete_validated',generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        script_sha256=sha256(__file__),input_root=str(root),fixed_horizon=HORIZON,method='FraMES: smoothed row Frank-Wolfe + momentum (pair-sampled gradient)',
        optimizer_dtype='float64',reference_optimizer_dtype='float64',strict_feasibility_tolerance_both_l1=100*sys.float_info.epsilon,
        auxiliary_repair_feasibility_tolerance_both_l1=ref.FEASIBILITY_TOLERANCE,
        auxiliary_repair_strict_bound_exceedance_count=sum(
            not r['additional_float64_repair']['passes_saved_output_strict_tolerance'] for r in records),
        input_sha256={name:sha256(root/name) for name in ('local_config.json','selection.json','completed.json','reference/reference.json',
            'confirmation_seed811.json','confirmation_seed812.json','confirmation_seed813.json')},
        execution=execution,comparison_axis='completed_updates',timing_comparison_performed=False,
        confirmation_timings_comparable=execution['confirmation_timings_comparable'],
        baseline_replay_sha256=baseline_hash,configuration=config,selected_candidate=selection['selected'],
        selection_verified_against_final_tuning_values=True,expected_tuning_count=4,observed_tuning_count=4,
        expected_confirmation_count=3,observed_confirmation_count=3,
        reference_array_files_sha256=reference['prepared_instance']['array_files_sha256'],
        initial=initial,baseline=baseline,runs=records,
        confirmation_summary=aggregate(confirmations,baseline['independent_rounded']['loss'],initial['loss']),
        common_iteration_curve=common_curve(confirmations),evaluation_environment=evaluation_environment,
        validation_seconds=time.perf_counter()-started,
        caveats=[
            'Four-candidate development selection at seed810; confirmation seeds811,812,813 only.',
            'Final saved-rounded criterion retained; no best-checkpoint or retrospective repair-based reselection.',
            execution.get('note','Original sequential confirmation execution preserved; no runtime comparison requested.'),
            'Iteration counts are the assessment axis. No method or precision timing comparison is performed.',
            'Both reference and FraMES use float64 throughout; NumPy/POT versus PyTorch implementations and algorithms differ.',
            'Actual saved float64-rounded plan must satisfy combined marginal L1 <=100*eps64; independent repeat repair is reported separately.',
            'Auxiliary NumPy repair uses the frozen reference helper tolerance 1e-12, with its measured residual and 100*eps64 pass/fail retained; it does not control selection or replace saved endpoints.',
            'Solver+current repair excludes instrumentation and previous repairs. Total/process elapsed times are separate.',
            'No original-run timestamp is assigned to the additional post-run float64 repair.',
            'One dataset and three confirmation seeds; ranges are descriptive, not confidence intervals or theorem tests.'])
    report['precision_comparison'] = precision_comparison(args.float32_report,report)
    out.mkdir(parents=True,exist_ok=True)
    write(out/'report.json',report)
    write(out/'precision_comparison.json',report['precision_comparison'])
    (out/'summary.txt').write_text(report_text(report))
    with (out/'curve_data.csv').open('w',newline='') as stream:
        rows = report['common_iteration_curve']
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    with (out/'precision_comparison.csv').open('w',newline='') as stream:
        rows = report['precision_comparison']['seed_matched_endpoints']
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
    with (out/'endpoints.csv').open('w',newline='') as stream:
        fields = ['name','phase','seed','candidate_id','raw_loss','saved_rounded_loss','additional_float64_repaired_loss',
                  'raw_column_l1','saved_rounded_both_l1','additional_float64_repaired_both_l1']
        writer = csv.DictWriter(stream,fieldnames=fields,lineterminator='\n'); writer.writeheader()
        for r in records:
            writer.writerow(dict(name=r['name'],phase=r['phase'],seed=r['seed'],candidate_id=r['candidate']['id'],
                raw_loss=r['last_raw_independent']['loss'],saved_rounded_loss=r['last_rounded_independent']['loss'],
                additional_float64_repaired_loss=r['additional_float64_repair']['loss'],
                raw_column_l1=r['last_raw_independent']['residuals']['col_l1'],
                saved_rounded_both_l1=r['last_rounded_independent']['residuals']['both_l1'],
                additional_float64_repaired_both_l1=r['additional_float64_repair']['residuals']['both_l1']))
    plot(report,out)
    print(json.dumps(dict(status=report['status'],out=str(out),confirmation_summary=report['confirmation_summary'])))


if __name__ == '__main__':
    main()
