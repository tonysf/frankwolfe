"""Report the fixed-horizon FP64 SCOT comparison without running an optimizer.

Current and final rounded objectives are practical diagnostics.  Independently
sampled SMD/MSBPG outputs and their finite-accuracy proximal certificates remain
separate; this script makes no exact-proximal or last-iterate theorem claim.
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

ROOT = Path(__file__).resolve().parents[2]
HORIZON = 2000
SEEDS = [811, 812, 813]
CHECKPOINTS = [0, 1, 10, 50, 100, 200, 500, 1000, 1500, 2000]
METHODS = ('smoothing-fw', 'published-smd', 'published-msbpg')
LABELS = {
    'smoothing-fw': 'FraMES (pair gradient + momentum)',
    'published-smd': 'Entropic SMD (fresh pair gradient)',
    'published-msbpg': 'MSBPG (paper momentum and schedules)',
}
THREAD_ENV = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
STRICT_FEASIBILITY = 100 * sys.float_info.epsilon
INNER_TOLERANCE = 1e-7
PLAN_KEYS = ('selected_raw', 'selected_rounded', 'last_raw', 'last_rounded')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(left, right, label, tolerance=1e-12):
    require(finite(left) and finite(right) and abs(left-right) <= tolerance,
            f'{label}: inconsistent values {left!r}, {right!r}.')


def summary(values):
    require(values and all(finite(v) for v in values), 'Nonfinite or empty summary.')
    return dict(n=len(values), median=statistics.median(values), minimum=min(values), maximum=max(values))


def source_files(hashes):
    require(isinstance(hashes, dict) and hashes, 'Missing numerical source inventory.')
    for name, expected in hashes.items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts,
                'Invalid numerical source path.')
        require(sha256(ROOT/'extensions'/relative) == expected, 'Changed numerical source: ' + name)


def inner_certificates(metadata):
    """Audit every recorded accepted solve, not just its self-reported maxima."""
    method, config, counters = metadata['method_key'], metadata['config'], metadata['counters']
    rows = metadata['inner_trace']
    if method == 'smoothing-fw':
        require(not rows and counters['inner_iterations_total'] == counters['inner_stopping_failures'] == 0,
                'FraMES unexpectedly contains an inner solver.')
        return dict(accepted_solves=0, numerical_criteria=None, inner_iterations=0)
    require(config['inner_tol'] == config['inner_gap_tol'] == INNER_TOLERANCE
            and counters['inner_stopping_failures'] == 0 and len(rows) == config['iterations'],
            'Missing inner solves or changed numerical stopping criteria.')
    maxima = dict(marginal_l1=0., normalized_primal_dual_gap=0.,
                  normalized_gap_upper_bound=0., primal_witness_marginal_l1=0.)
    iterations = checks = underflows = underflow_steps = 0
    for k, row in enumerate(rows):
        require(row['iteration'] == k and row['converged'] is True
                and row['proximal_optimality_certified'] is True and row['certificate_valid'] is True
                and 0 < row['inner_iterations'] <= config['inner_max'],
                'Missing or unaccepted proximal step.')
        fields = ('primal_objective', 'dual_objective', 'primal_dual_gap',
                  'stable_primal_dual_gap', 'gap_roundoff_tolerance',
                  'normalized_primal_dual_gap', 'normalized_gap_upper_bound',
                  'marginal_l1', 'primal_witness_marginal_l1')
        require(all(finite(row[x]) for x in fields), 'Nonfinite proximal certificate.')
        primal, dual = row['primal_objective'], row['dual_objective']
        gap, stable, guard = (row[x] for x in ('primal_dual_gap', 'stable_primal_dual_gap', 'gap_roundoff_tolerance'))
        scale = max(1., abs(primal), abs(dual))
        upper = (max(gap, stable) + guard)/scale
        require(0 <= row['marginal_l1'] <= INNER_TOLERANCE
                and 0 <= row['primal_witness_marginal_l1'] <= 1e-12
                and 0 <= guard/scale <= INNER_TOLERANCE
                and min(gap, stable) >= -guard and abs(gap-stable) <= guard
                and upper <= INNER_TOLERANCE,
                'A recorded proximal solve fails its guarded numerical criteria.')
        close(primal-dual, gap, 'Primal-dual identity', 1e-12*scale)
        close(gap/scale, row['normalized_primal_dual_gap'], 'Normalized gap')
        close(upper, row['normalized_gap_upper_bound'], 'Guarded normalized gap')
        require(row['inner_tolerance'] == row['normalized_gap_tolerance'] == INNER_TOLERANCE,
                'Per-step inner tolerance changed.')
        for field in maxima:
            maxima[field] = max(maxima[field], row[field])
        iterations += row['inner_iterations']
        checks += row['certificate_checks']
        underflows += row['underflow_entries']
        underflow_steps += int(row['underflow_entries'] > 0)
    for field, value in maxima.items():
        require(value == metadata['inner_acceptance_maxima'][field], 'Inner maximum differs: ' + field)
    require(iterations == counters['inner_iterations_total'] and checks == counters['certificate_checks']
            and underflows == counters['underflow_entries_total'] and underflow_steps == counters['underflow_steps'],
            'Inner solve work/underflow counters disagree with full recorded trace.')
    return dict(accepted_solves=len(rows), maxima=maxima, inner_iterations=iterations,
                certificate_checks=checks, underflow_entries=underflows, underflow_steps=underflow_steps,
                numerical_criteria=dict(actual_plan_summed_marginal_l1=INNER_TOLERANCE,
                    guarded_normalized_primal_dual_gap=INNER_TOLERANCE,
                    feasible_witness_summed_marginal_l1=1e-12),
                scope='Finite-accuracy numerical certificates, not an exact-proximal convergence theorem.')


def paper_parameters(metadata):
    """Check schedule arithmetic for every accepted step using recorded constants."""
    method, config = metadata['method_key'], metadata['config']
    if method == 'smoothing-fw':
        schedule = metadata['schedule']
        require((schedule['gamma_exponent'], schedule['rho_exponent'], schedule['mu_exponent']) == (.75,.5,.25),
                'FraMES decay exponents changed.')
        return dict(gamma0=config['gamma0'], rho0=config['rho0'], mu_scale=config['mu_scale'],
                    exponents=dict(gamma=.75, rho=.5, mu=.25),
                    scope='Previously selected empirical prefactors; no final-iterate theorem claim.')
    cmax = metadata['cmax']
    require(finite(cmax) and cmax > 0, 'Invalid squared GW cost bound.')
    if method == 'published-smd':
        fraction = config['prefactor_fraction']
        require(0 < fraction <= 1, 'SMD prefactor outside the frozen analytic cap.')
        for k, row in enumerate(metadata['inner_trace']):
            close(row['eta'], fraction/(4*cmax*math.sqrt(k+1)), 'SMD step', 1e-14)
        return dict(cmax=cmax, prefactor_fraction=fraction, eta0=fraction/(4*cmax),
                    schedule='eta_k = fraction/(4*Cmax*sqrt(k+1))', momentum=False,
                    paper='Fatkhullin–He, AISTATS 2024, equation (2), Theorem 4.3',
                    scope='Fresh-gradient entropic SMD; this decay is an admissible specialization, not a uniquely prescribed schedule. Proximal solves use finite numerical tolerances.')
    constants = metadata['theorem_constants']
    L, M = constants['hessian_bound'], constants['maximum_feasible_entry']
    require(all(finite(x) and x > 0 for x in (L,M)) and constants['mu'] == 1.
            and constants['relative_smoothness'] == L and constants['kernel_scale'] == M
            and constants['kappa'] == 2*L*L and constants['exact_loaded_distance_symmetry_verified'] is True,
            'MSBPG theorem constants or symmetry contract changed.')
    close(constants['coupling_c_max'], 1/(2*math.sqrt(2*L*L)), 'MSBPG coupling cap', 1e-18)
    fraction, theta = config['coupling_fraction'], config['momentum_prefactor']
    require(0 < fraction <= 1 and 0 < theta <= 1, 'MSBPG prefactors exceed admissible interval.')
    c = fraction*constants['coupling_c_max']
    for k, row in enumerate(metadata['inner_trace']):
        alpha = c*theta/math.sqrt(k+1)
        close(row['alpha'], alpha, 'MSBPG alpha', 1e-18)
        close(row['eta'], alpha/M, 'MSBPG effective KL step', 1e-16)
        close(row['beta_next'], theta/math.sqrt(k+1), 'MSBPG next momentum weight', 1e-14)
        if k == 0:
            require(row['beta'] is None, 'MSBPG must initialize momentum from its first gradient.')
        else:
            close(row['beta'], theta/math.sqrt(k), 'MSBPG momentum weight', 1e-14)
        require(row['theorem_c_bound_satisfied'] is True and row['alpha_small_step_bound_satisfied'] is True
                and alpha < 1/L, 'MSBPG schedule violates its recorded parameter conditions.')
    return dict(constants=constants, coupling_fraction=fraction, momentum_prefactor=theta,
                alpha0=c*theta, effective_kl_eta0=c*theta/M,
                paper='Ding–Li–Toh, JMLR 26(39), 2025, equations (8)–(9), Theorem 24 and Remark 25',
                scope='Published coupled MSBPG structure; conservative problem constants and finite proximal tolerances remain explicit.')


def output_selection(metadata):
    selected, config = metadata['output_selection'], metadata['config']
    horizon = config['iterations']
    index = selected['index']
    require(type(index) is int and 0 <= index < horizon and selected['horizon'] == horizon
            and selected['last_iterate_index'] == horizon and selected['independent_rng'] is True
            and selected['output_seed'] == config['output_seed']
            and selected['selected_raw_is_the_algorithm_output'] is True
            and selected['rounded_selected_is_diagnostic'] is True,
            'Independent weighted output is conflated with the final/current diagnostic.')
    probability = (index+1)**-.5 / sum((k+1)**-.5 for k in range(horizon))
    close(selected['probability'], probability, 'Independent output probability', 1e-14)
    result = dict(selected)
    if metadata['method_key'] == 'published-msbpg':
        statistic = metadata['selected_update_statistic']
        require(statistic['index'] == index
                and finite(statistic['normalized_displacement_squared'])
                and statistic['normalized_displacement_squared'] >= 0,
                'Invalid recorded MSBPG independently selected update statistic.')
        close(statistic['alpha'], metadata['inner_trace'][index]['alpha'], 'Selected MSBPG alpha', 1e-18)
        result['recorded_update_statistic'] = statistic
        result['scope_caveat'] = 'Numerical squared displacement divided by alpha squared at the independently sampled update; finite proximal error precludes an exact-theorem certificate.'
    elif metadata['method_key'] == 'published-smd':
        result['scope_caveat'] = 'Independent numerical random output. The BFBE stationarity quantity controlled by the paper is not evaluated; its objective is descriptive.'
    else:
        result['scope_caveat'] = 'This independently sampled snapshot is a diagnostic only for FraMES, not its manuscript output statistic.'
    return result


def recorded_curve_feasibility(observations):
    """Check logged FP64 current outputs; unsaved checkpoints are not recomputed."""
    require(observations, 'Missing current-output observations.')
    maxima = []
    for observation in observations:
        residuals = observation['rounded_residuals']
        required = ('row_l1','col_l1','col_l2','both_l1','mass_error','min_entry')
        require(all(name in residuals for name in required)
                and all(finite(value) and value >= 0 for value in residuals.values())
                and residuals['both_l1'] <= STRICT_FEASIBILITY,
                'A recorded current-rounded checkpoint fails FP64 feasibility/nonnegativity.')
        maxima.append(residuals['both_l1'])
    return dict(observed_count=len(observations),maximum_recorded_marginal_l1=max(maxima),
                marginal_l1_tolerance=STRICT_FEASIBILITY,
                validation_source='Recorded FP64 residuals of every current rounded observation, including all plotted common checkpoints. Intermediate plans are unsaved; this is a log audit, not an independent recomputation.')


def frozen_config(root):
    config = read(root/'local_config.json')
    require(config['fixed_horizon'] == HORIZON and config['tuning_seed'] == 810
            and config['confirmation_seeds'] == SEEDS and config['pairs_per_call'] == 1024
            and config['optimizer_dtype'] == config['final_evaluation_dtype'] == 'float64'
            and config['initialization'] == 'uniform product'
            and config['inner_tol'] == config['inner_gap_tol'] == INNER_TOLERANCE
            and config['inner_max'] == 10000 and config['check_every'] == 10,
            'Frozen scientific design changed.')
    require(config['checkpoint_iterations'] == CHECKPOINTS[1:]
            and config['checkpoint_seconds'] == ''
            and config['threads_per_process'] == 1 and config['thread_environment'] == THREAD_ENV
            and config['workers'] == 4 and len(config['physical_cores']) == 4,
            'Frozen logging or CPU execution policy changed.')
    require(len({(x['package_id'],x['core_id']) for x in config['physical_cores']}) == 4
            and len({x['cpu_id'] for x in config['physical_cores']}) == 4,
            'CPU affinity does not identify four different physical cores.')
    methods = config['methods']
    require(set(methods) == {'published-smd','published-msbpg'}
            and [x['prefactor_fraction'] for x in methods['published-smd']] == [.1,.3,.6,1.]
            and {(x['coupling_fraction'],x['momentum_prefactor']) for x in methods['published-msbpg']} ==
                {(.3,.3),(.3,1.),(1.,.3),(1.,1.)}
            and len(methods['published-msbpg']) == 4,
            'The four-candidate method grids differ from the authorized design.')
    require(config['specification']['methods'] == methods
            and config['controller_sha256'] == sha256(ROOT/'extensions/scot_cpu_methods_fp64_20261014/run_comparison.py'),
            'Frozen controller or embedded specification changed.')
    source_files(config['source_file_sha256'])
    source_files(config['helper_source_sha256'])
    bindings = dict(reference_sha256=sha256(root/'reference/reference.json'),
                    baseline_sha256=sha256(root/'baseline_local/replay.json'),
                    instance_manifest_sha256=sha256(root/'reference/prepared/scot1047/manifest.json'),
                    array_files_sha256={name:sha256(root/'reference/prepared/scot1047'/name)
                                        for name in ('A.npy','B.npy','a.npy','b.npy')},
                    frames_completed_sha256=sha256(root/'frames_result/completed.json'),
                    frames_config_sha256=sha256(root/'frames_result/local_config.json'))
    require(all(config[k] == value for k,value in bindings.items()), 'Frozen input or prior FraMES binding changed.')
    return config, bindings


def design(root):
    config, bindings = frozen_config(root)
    methods = config['methods']
    selected, completed = (read(root/name) for name in ('selection.json','completed.json'))
    metric = 'runner_recomputed_float64_last_rounded_loss_at_fixed_horizon'
    require(selected['status'] == completed['status'] == 'completed'
            and selected['config_sha256'] == sha256(root/'local_config.json')
            and selected['selection_metric'] == config['selected_metric'] == metric
            and completed['selection_sha256'] == sha256(root/'selection.json')
            and completed['config'] == config and completed['selected'] == selected['selected']
            and completed['tuning'] == selected['tuning'], 'Selection/completion binding changed.')
    expected_tuning = [(method,candidate,810) for method in METHODS[1:] for candidate in methods[method]]
    require([(x['method'],x['candidate'],x['seed']) for x in completed['tuning']] == expected_tuning,
            'Tuning population differs from all eight prescribed cells.')
    for method in METHODS[1:]:
        trials = [x for x in completed['tuning'] if x['method'] == method]
        require(all(finite(x['last_rounded_loss_float64']) for x in trials), 'Invalid tuning endpoint.')
        winner = min(trials,key=lambda x:x['last_rounded_loss_float64'])['candidate']
        require(winner == selected['selected'][method], 'Selection is not the best final tuning output.')
    expected_confirmation = [(method,selected['selected'][method],seed) for method in METHODS[1:] for seed in SEEDS]
    require([(x['method'],x['candidate'],x['seed']) for x in completed['confirmation']] == expected_confirmation,
            'Fresh confirmation population differs from the six prescribed cells.')
    return config, selected, completed, bindings


def new_run(root, phase, cell, frozen, binding, prior):
    method, candidate, seed = cell['method'], cell['candidate'], cell['seed']
    name = f'{phase}_{candidate["id"]}_seed{seed}'
    directory = root/'runs'/name
    receipt_path, metadata_path = root/f'{name}_receipt.json', directory/'metadata.json'
    receipt, metadata = read(receipt_path), read(metadata_path)
    require(Path(cell['directory']).resolve() == directory.resolve() and receipt == cell['receipt']
            and receipt['status'] == 'completed' and receipt['exit_code'] == 0
            and receipt['name'] == name and receipt['method'] == method
            and receipt['candidate'] == candidate and receipt['seed'] == seed,
            'Run identity or completion receipt differs.')
    identity = {k:frozen[k] for k in ('source_commit','controller_sha256','source_file_sha256','helper_source_sha256')}
    require(receipt['config_sha256'] == sha256(root/'local_config.json')
            and receipt['input_binding'] == binding and receipt['source_identity'] == identity
            and receipt['thread_environment'] == THREAD_ENV and receipt['cpu_core'] in frozen['physical_cores']
            and receipt['metadata_sha256'] == cell['metadata_sha256'] == sha256(metadata_path),
            'Run/source/input/CPU receipt binding changed.')
    config = metadata['config']
    expected = dict(method=method,device='cpu',dtype='float64',iterations=HORIZON,seed=seed,
        output_seed=seed+20000,init_seed=seed+10000,initialization='product',oracle='pair-sampling',
        pairs_per_call=1024,save_plans='all',progress_every=1000,
        checkpoint_iterations=','.join(map(str,CHECKPOINTS[1:])),checkpoints=frozen['checkpoint_seconds'],
        inner_tol=INNER_TOLERANCE,inner_gap_tol=INNER_TOLERANCE,inner_max=10000,check_every=10,
        **{k:v for k,v in candidate.items() if k != 'id'})
    require(metadata['status'] == 'completed' and metadata['method_key'] == method
            and all(config.get(k) == value for k,value in expected.items()), 'Executed numerical settings differ.')
    require(Path(config['out']).resolve() == directory.resolve()
            and Path(config['instance']).resolve() == (root/'reference/prepared/scot1047').resolve(),
            'Executed array/output directory differs.')
    command = receipt['command']
    require(command[:3] == ['taskset','-c',str(receipt['cpu_core']['cpu_id'])]
            and Path(command[4]).resolve() == (ROOT/'extensions/pair_sampling_20261011/run_pair.py').resolve()
            and len(command[5:]) % 2 == 0, 'Unexpected CPU runner command.')
    options = dict(zip(command[5::2],command[6::2]))
    require(len(options) == len(command[5:])//2
            and all(k.startswith('--') and k[2:].replace('-','_') in config
                    and str(config[k[2:].replace('-','_')]) == value for k,value in options.items()),
            'Runner command/configuration mismatch or duplicate argument.')
    require(metadata['source_file_sha256'] == frozen['source_file_sha256']
            and metadata['array_files_sha256'] == binding['array_files_sha256']
            and metadata['instance_manifest_sha256'] == binding['instance_manifest_sha256']
            and metadata['dimensions'] == dict(source=1047,target=1047), 'Numerical source or geometry differs.')
    counters, env, oracle = metadata['counters'], metadata['environment'], metadata['gradient_oracle']
    require(counters['completed_updates'] == counters['stochastic_gradient_calls'] == HORIZON
            and counters['pair_samples'] == HORIZON*1024 and counters['sketch_probes'] == 0
            and counters['initialization_oracle_calls'] == int(method == 'published-msbpg'),
            'Incorrect update, sampling, or initialization counts.')
    require(oracle['name'] == 'pair-sampling' and oracle['pair_samples_per_call'] == 1024
            and oracle['probes_per_call'] == 0 and oracle['target_marginals_substituted'] is False
            and oracle['gradient_arithmetic'] == 'float64'
            and all(finite(oracle[k]) and oracle[k] > 0 for k in ('mass_min','mass_max','mass_last')),
            'Different gradient precision or pair-sampling law.')
    streams = metadata['random_streams']
    require(streams['pairs']['seed'] == seed and streams['pairs']['with_replacement'] is True
            and streams['pairs']['rank_zero_generator'] is True
            and streams['output']['seed'] == seed+20000 and streams['output']['independent_generator'] is True
            and streams['initialization']['seed'] == seed+10000, 'Stochastic/output/initialization streams differ.')
    require(env['world_size'] == 1 and env['backend'] == 'gloo' and env['deterministic_algorithms'] is True
            and len(metadata['ranks']) == 1 and metadata['ranks'][0]['device'] == 'cpu'
            and metadata['ranks'][0]['gpu'] is None, 'Wrong CPU topology or determinism.')
    rank = metadata['ranks'][0]
    require(rank == read(directory/'rank0/metadata.json'), 'Rank-level metadata differs.')
    saved, final = rank['saved_plans'], metadata['final_float64_validation']
    require(set(saved) == set(PLAN_KEYS) and final['dtype'] == 'float64' and final['exact_objective'] is True
            and set(final['plans']) == set(PLAN_KEYS)
            and cell['last_rounded_loss_float64'] == final['plans']['last_rounded']['loss'],
            'Missing or changed final saved-plan validation/selection value.')
    files = {}
    for key in PLAN_KEYS:
        path, entry = directory/'rank0'/f'{key}.npy', saved[key]
        require(entry['persisted'] is True and entry['dtype'] == 'float64' and entry['shape'] == [1047,1047]
                and entry['array_bytes'] == 1047*1047*8 and Path(entry['file']).resolve() == path.resolve()
                and entry['file_sha256'] == sha256(path) and entry['bytes'] == path.stat().st_size,
                'Saved plan file/dtype/shape/hash changed: ' + key)
        files[key] = dict(path=str(path),sha256=sha256(path),array_sha256=entry['array_sha256'])
    require(all(finite(v) and v >= 0 for v in metadata['timing'].values()), 'Invalid archived timing.')
    return dict(name=name,method_key=method,phase=phase,candidate=candidate,seed=seed,
                metadata_sha256=sha256(metadata_path),receipt_sha256=sha256(receipt_path),
                config=config,environment=env,counters=counters,source_file_sha256=metadata['source_file_sha256'],
                initial_plan_sha256=rank['initial_shard_sha256'],saved_files=files,recorded_float64=final['plans'],
                inner_audit=inner_certificates(metadata),parameters=paper_parameters(metadata),
                output_selection=output_selection(metadata),
                observations=prior.trajectory(metadata['trace'],metadata['timing']['solver_seconds']),
                timing=metadata['timing'],process_elapsed_seconds=receipt['process_elapsed_seconds'],
                timing_eligible_for_comparison=False,memory=rank['memory'])


def stage_bindings(root, config, selection, completed):
    names = ('init','tune_both_0_end','select','confirm_both_0_end','finalize')
    stages = {}
    for name in names:
        path = root/f'stage_{name}.json'
        value = read(path)
        require(value['status'] == 'completed' and value['stage'] == name
                and value['controller_sha256'] == config['controller_sha256'],
                'Incomplete or changed prerequisite stage: ' + name)
        stages[name] = value
    require(stages['init']['result']['config_sha256'] == sha256(root/'local_config.json')
            and stages['tune_both_0_end']['result']['runs'] == completed['tuning']
            and stages['select']['result'] == selection
            and stages['confirm_both_0_end']['result']['runs'] == completed['confirmation']
            and stages['confirm_both_0_end']['result']['selection_sha256'] == sha256(root/'selection.json')
            and Path(stages['finalize']['result']['completed_file']).resolve() == (root/'completed.json').resolve(),
            'Stage output/config/selection binding changed.')
    return {f'stage_{name}.json':sha256(root/f'stage_{name}.json') for name in names}


def reused_frames(root, config, reference, baseline, prior):
    directory = (root/'frames_result').resolve()
    report_path = directory/'report/report.json'
    previous = read(report_path)
    prior_source = ROOT/'extensions/scot_cpu_fp64_20261014/summarize.py'
    require(previous['status'] == 'complete_validated' and previous['script_sha256'] == sha256(prior_source)
            and previous['optimizer_dtype'] == 'float64' and previous['fixed_horizon'] == HORIZON
            and previous['reference_array_files_sha256'] == config['array_files_sha256']
            and previous['baseline_replay_sha256'] == config['baseline_sha256'],
            'Prior FraMES report/source/geometry is not the verified matching FP64 experiment.')
    for name, expected in previous['input_sha256'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts
                and sha256(directory/relative) == expected, 'Prior FraMES report input changed.')
    _, selected, completed, execution = prior.validate_design(directory)
    require(selected['selected'] == previous['selected_candidate'] ==
            dict(id='fw08',gamma0=1.,rho0=.3,mu_scale=10.), 'Different reused FraMES candidate.')
    previous_runs = {r['seed']:r for r in previous['runs'] if r['phase'] == 'confirm'}
    previous_tuning = [r for r in previous['runs'] if r['phase'] == 'tune']
    require(len(previous_tuning) == 4 and
            min(previous_tuning,key=lambda r:r['last_rounded_independent']['loss'])['candidate'] == selected['selected'],
            'Prior independently evaluated FraMES tuning losses do not confirm its frozen selection.')
    result = []
    for cell in completed['confirmation']:
        row = prior.validate_run(directory,'confirm',cell,reference,baseline['environment']['hostname'])
        require(row['metadata_sha256'] == previous_runs[row['seed']]['metadata_sha256']
                and row['receipt_sha256'] == previous_runs[row['seed']]['receipt_sha256']
                and row['source_file_sha256'] == config['source_file_sha256'],
                'Reused FraMES metadata/receipt or shared numerical source differs.')
        metadata = read(Path(cell['directory'])/'metadata.json')
        row.update(method_key='smoothing-fw',reused_from_previous_experiment=True,
                   inner_audit=inner_certificates(metadata),parameters=paper_parameters(metadata),
                   output_selection=output_selection(metadata))
        result.append(row)
    return result, dict(report_path=str(report_path),report_sha256=sha256(report_path),
                        reporting_source_sha256=sha256(prior_source),execution=execution,
                        selected_candidate=selected['selected'],
                        independent_tuning_winner_confirmed=True,
                        scope='The three existing fresh FP64 confirmation runs are reused once; no new FraMES trials or reselection.')


def numerical_validation(root, reference_module, baseline, records):
    import numpy as np
    import ot
    ref = reference_module
    arrays = tuple(np.load(root/'reference/prepared/scot1047'/name,allow_pickle=False) for name in ref.ARRAY_NAMES)
    A,B,a,b = arrays
    require(all(x.dtype == np.float64 and np.isfinite(x).all() for x in arrays)
            and A.shape == B.shape == (1047,1047) and a.shape == b.shape == (1047,)
            and np.array_equal(A,A.T) and np.array_equal(B,B.T), 'Invalid original FP64 geometry.')
    initial_plan = np.outer(a,b)
    initial_hash = hashlib.sha256(initial_plan.tobytes()).hexdigest()
    initial = ref.evaluate(A,B,initial_plan,a,b,ot)
    require(initial['independent_loss_agrees'], 'Initial independent objective contractions disagree.')
    close(initial['loss'],baseline['initial']['loss'],'Original initial loss')
    for key in ('raw','rounded'):
        values = np.load(root/'baseline_local'/f'P_{key}.npy',allow_pickle=False)
        require(values.dtype == np.float64, 'Reference plan precision changed.')
        metric = ref.evaluate(A,B,values,a,b,ot)
        require(metric['independent_loss_agrees'], 'Independent reference objective contractions disagree.')
        close(metric['loss'],baseline[key]['loss'],'Local reference objective')
        baseline['independent_'+key] = metric
    require(baseline['independent_rounded']['residuals']['both_l1'] <= STRICT_FEASIBILITY,
            'Actual reference saved rounded output fails strict FP64 feasibility.')
    cmax = max(abs(float(A.min())-float(B.max())),abs(float(A.max())-float(B.min())))**2
    M = min(float(a.max()),float(b.max()))
    row_bound = float(2*np.max(len(b)*(A*A).sum(1)[:,None] + len(a)*(B*B).sum(1)[None,:]
                              - 2*A.sum(1)[:,None]*B.sum(1)[None,:]))
    for row in records:
        require(row['initial_plan_sha256'] == initial_hash, 'Saved initial plan is not the common FP64 product.')
        if row['method_key'] != 'smoothing-fw':
            close(row['parameters'].get('cmax',cmax),cmax,'SMD cost bound',0.)
        if row['method_key'] == 'published-msbpg':
            constants = row['parameters']['constants']
            require(constants['maximum_feasible_entry'] == M
                    and constants['hessian_bound_input_dtype'] == 'torch.float64'
                    and constants['hessian_statistics_dtype'] == 'torch.float64'
                    and constants['hessian_bound'] >= row_bound
                    and abs(constants['raw_hessian_bound']-row_bound) <= constants['hessian_fp64_absolute_guard'],
                    'MSBPG constants do not agree with the independent original-geometry audit.')
        independent = {}
        evaluated_keys = set(PLAN_KEYS if row['phase'] == 'confirm' and row['method_key'] != 'smoothing-fw'
                             else ('last_raw','last_rounded'))
        for key, saved in row['saved_files'].items():
            plan = np.load(saved['path'],allow_pickle=False)
            require(plan.shape == (1047,1047) and plan.dtype == np.float64
                    and np.isfinite(plan).all() and (plan >= 0).all()
                    and hashlib.sha256(plan.tobytes()).hexdigest() == saved['array_sha256'],
                    'Saved plan content/dtype/shape/nonnegativity changed.')
            residuals = ref.residuals(plan,a,b)
            if key.endswith('_rounded'):
                require(residuals['both_l1'] <= STRICT_FEASIBILITY,
                        'Actual saved rounded output fails independent strict FP64 feasibility.')
            elif row['method_key'] != 'smoothing-fw':
                require(residuals['both_l1'] <= INNER_TOLERANCE,
                        'Actual raw proximal output exceeds its prescribed marginal tolerance.')
            if key in evaluated_keys:
                metric = ref.evaluate(A,B,plan,a,b,ot)
                require(metric['independent_loss_agrees'], 'Independent actual-marginal GW contractions disagree.')
                close(metric['loss'],row['recorded_float64'][key]['loss'],'Saved full objective',1e-10)
                for field, value in row['recorded_float64'][key]['marginal_residuals'].items():
                    close(metric['residuals'][field],value,'Saved marginal residual',1e-12)
                independent[key] = metric
        require(set(independent) == evaluated_keys, 'Missing independently evaluated output.')
        row['independent'] = independent
        row['independently_evaluated_plan_keys'] = sorted(independent)
    return initial, dict(numpy=np.__version__,POT=ot.__version__,threads=THREAD_ENV,
                        independently_computed_cmax=cmax,independently_computed_maximum_entry=M,
                        independently_computed_hessian_row_bound=row_bound,
                        no_post_run_repair=True)


def independent_selection(records, selected):
    result = {}
    for method in METHODS[1:]:
        trials = [r for r in records if r['method_key'] == method and r['phase'] == 'tune']
        require(len(trials) == 4, 'Independent winner audit requires all four tuning candidates.')
        winner = min(trials,key=lambda r:r['independent']['last_rounded']['loss'])['candidate']
        require(winner == selected[method],
                'Independent NumPy/POT endpoint validation changes the ordering of the selected tuning candidate; review required without automatic reselection.')
        result[method] = dict(status='same_winner_confirmed',runner_selected=selected[method],
            independently_selected=winner,
            candidates=[dict(candidate=r['candidate'],
                runner_float64_loss=r['recorded_float64']['last_rounded']['loss'],
                numpy_actual_marginal_loss=r['independent']['last_rounded']['loss'],
                pot_actual_marginal_loss=r['independent']['last_rounded']['pot_actual_marginals_loss']) for r in trials],
            scope='The controller selected the final saved-rounded loss from its separate Torch FP64 evaluation. This later independent NumPy/POT audit confirms that choice; it does not reselect from confirmation runs.')
    return result


def curves(records):
    result = []
    for method in METHODS:
        runs = [r for r in records if r['method_key'] == method and r['phase'] == 'confirm']
        require(len(runs) == 3 and {r['seed'] for r in runs} == set(SEEDS), 'Missing fresh seed population.')
        maps = [{p['iteration']:p for p in r['observations']} for r in runs]
        common = sorted(set.intersection(*(set(x) for x in maps)))
        require(set(CHECKPOINTS).issubset(common), 'Missing common logged iteration.')
        for k in common:
            values = [p[k]['rounded_loss'] for p in maps]
            result.append(dict(method=method, iteration=k, expected_seed_count=3, observed_seed_count=3,
                               loss_median=statistics.median(values), loss_minimum=min(values), loss_maximum=max(values)))
    return result


def endpoint_summary(records, initial_loss, reference_loss):
    result = {}
    for method in METHODS:
        runs = [r for r in records if r['method_key'] == method and r['phase'] == 'confirm']
        losses = [r['independent']['last_rounded']['loss'] for r in runs]
        result[method] = dict(expected_seed_count=3, observed_seed_count=len(runs), loss=summary(losses),
            reduction_percent=summary([100*(1-x/initial_loss) for x in losses]),
            loss_minus_reference=summary([x-reference_loss for x in losses]),
            raw_marginal_l1=summary([r['independent']['last_raw']['residuals']['both_l1'] for r in runs]),
            rounded_marginal_l1=summary([r['independent']['last_rounded']['residuals']['both_l1'] for r in runs]))
    return result


def plot(report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(METHODS, ('#0072B2','#D55E00','#009E73')))
    fig, ax = plt.subplots(figsize=(10,6.7))
    for method in METHODS:
        points = [p for p in report['curves'] if p['method'] == method]
        x = [p['iteration'] for p in points]
        ax.fill_between(x,[p['loss_minimum'] for p in points],[p['loss_maximum'] for p in points],
                        color=colors[method],alpha=.15)
        ax.plot(x,[p['loss_median'] for p in points],color=colors[method],lw=2,marker='.',ms=5,label=LABELS[method])
        ax.scatter(HORIZON,report['confirmation_summary'][method]['loss']['median'],
                   facecolor='white',edgecolor=colors[method],s=40,zorder=5)
    ax.axhline(report['reference_loss'],color='#8E44AD',ls='--',lw=1.5,label='SCOT/POT feasible reference')
    ax.set(xlabel='Completed updates',ylabel='Full squared GW objective',xlim=(0,HORIZON*1.035))
    ax.grid(alpha=.2)
    ax.margins(y=.08)
    fig.suptitle('SCOT1047 — matched float64 stochastic comparison',fontsize=14,y=.98)
    fig.text(.5,.925,'2,000 updates · 1,024 iid coupling pairs/update · three fresh seeds (811–813)',ha='center',fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.09),ncol=2,frameon=False,fontsize=9)
    fig.text(.5,.025,'Medians at common logged iterations; shaded bands are seed ranges, not confidence intervals.\n'
             'Current rounded diagnostics; circles = independently evaluated saved endpoints. No runtime comparison.',
             ha='center',fontsize=8.6)
    fig.subplots_adjust(top=.86,bottom=.27,left=.10,right=.98)
    for suffix in ('png','pdf'):
        fig.savefig(out/f'scientific_plot.{suffix}',dpi=180,bbox_inches='tight')
    plt.close(fig)

    # A separate, explicitly magnified view makes small changes visible without
    # distorting the shared objective scale of the primary comparison.
    points = [p for p in report['curves'] if p['method'] == 'published-msbpg']
    if max(p['loss_maximum'] for p in points)-min(p['loss_minimum'] for p in points) < .01*report['initial']['loss']:
        fig, ax = plt.subplots(figsize=(8.3,4.8))
        x = [p['iteration'] for p in points]
        ax.fill_between(x,[p['loss_minimum'] for p in points],[p['loss_maximum'] for p in points],color=colors['published-msbpg'],alpha=.18)
        ax.plot(x,[p['loss_median'] for p in points],color=colors['published-msbpg'],lw=2,marker='.')
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        ax.set(xlabel='Completed updates',ylabel='Full squared GW objective',title='MSBPG detail — magnified objective axis')
        ax.grid(alpha=.2)
        ax.margins(y=.15)
        fig.text(.5,.025,'Median and range of the same three fresh seeds. This separate y-scale magnifies small changes;\n'
                 'use scientific_plot for the three-method comparison on a common scale.',ha='center',fontsize=8.5)
        fig.subplots_adjust(bottom=.24,left=.16,right=.98)
        for suffix in ('png','pdf'):
            fig.savefig(out/f'msbpg_detail.{suffix}',dpi=180,bbox_inches='tight')
        plt.close(fig)


def text_summary(report):
    lines = [
        'SCOT1047: complete FP64 stochastic comparison, 2,000 updates per run.',
        'All three methods use 1,024 iid coupling-pair samples per update and the identical FP64 product start and geometry.',
        'Primary assessment: current and final saved-rounded full GW objective versus completed updates. No timing comparison.',
        f'Initial full GW loss: {report["initial"]["loss"]:.12g}',
        f'Attainable local SCOT/POT feasible reference: {report["reference_loss"]:.12g}',
        '',
    ]
    for method in METHODS:
        values = report['confirmation_summary'][method]
        loss, reduction = values['loss'],values['reduction_percent']
        lines += [LABELS[method],
            f'  Fresh seeds 811–813: median final loss {loss["median"]:.12g}; range [{loss["minimum"]:.12g}, {loss["maximum"]:.12g}].',
            f'  Reduction from product start: median {reduction["median"]:.6g}%; range [{reduction["minimum"]:.6g}%, {reduction["maximum"]:.6g}%].',
            f'  Selected prefactors: {report["selected_candidates"][method]}',
            f'  Maximum independent saved-rounded marginal L1: {values["rounded_marginal_l1"]["maximum"]:.6g}.',
            '']
    lines += [
        'Each method used four development candidates at seed810; fresh confirmation seeds did not select parameters. FraMES fw08 and its three earlier FP64 confirmations are reused once.',
        'The SMD/MSBPG controller selects the final saved-rounded objective from the runner’s separate Torch FP64 reevaluation. This report independently recomputes actual-marginal objectives with NumPy and POT and confirms each tuning winner.',
        'SMD and MSBPG use published update equations with theorem-admissible parameters, not a reproduction of the authors’ empirical presets. SMD uses a fresh gradient; MSBPG uses its published coupled momentum and step schedule.',
        'Every accepted transport proximal update has recorded actual-plan marginal L1 and guarded normalized primal-dual gap at most 1e-7. These numerical certificates do not supply an exact-proximal or inexact outer convergence theorem.',
        'Raw proximal plans are only approximately feasible; plotted rounded diagnostics are separately checked at 100*eps64. FraMES raw plans obey row constraints and may have column residuals. All losses use actual marginals, including raw plans.',
        'The independently weighted SMD/MSBPG output is a raw P_J with J in 0,...,T−1. It is kept separate from P_T, rounded diagnostics, and prefactor selection. The SMD BFBE quantity is not evaluated. The MSBPG selected squared-displacement statistic is numerical and is preserved separately.',
        'Matching seed labels do not imply identical sampled pairs: each method samples conditionally from its own evolving coupling. Equal iterations and pairs do not imply equal arithmetic work.',
        'Bands are ranges over three seeds on one fixed dataset, not confidence intervals. The SCOT/POT reference is an attained objective, not a certified optimum or a matched-iteration competitor.',
        'The primary figure uses a common objective axis. If produced, msbpg_detail explicitly magnifies a separate y-axis and does not establish a larger comparative improvement.',
        '',
        'Artifacts: report.json; scientific_plot.png/pdf; optional msbpg_detail.png/pdf; curve_data.csv; endpoints.csv; theory_outputs.csv; selection_audit.json.',
        '',
    ]
    return '\n'.join(lines)


def save_csv(path, rows, fields=None):
    require(rows or fields, 'CSV schema is missing.')
    with Path(path).open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields or list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'experiments/scot_cpu_methods_fp64_20261014')
    parser.add_argument('--out',type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.resolve() if args.out else root/'report'
    for name, value in THREAD_ENV.items():
        os.environ[name] = value
    config, selection, completed, binding = design(root)
    stages = stage_bindings(root,config,selection,completed)
    prior_path = ROOT/'extensions/scot_cpu_fp64_20261014/summarize.py'
    prior = load_module(prior_path,'_scot_methods_prior_report')
    reference_path = ROOT/'extensions/scot_reproduction_20261014/reference.py'
    ref = load_module(reference_path,'_scot_methods_reference')
    reference = ref.verify_reference(root/'reference')
    baseline,_ = prior.validate_baseline(root,reference)
    records = [new_run(root,phase,cell,config,binding,prior)
               for phase,key in [('tune','tuning'),('confirm','confirmation')] for cell in completed[key]]
    require({p.name for p in (root/'runs').iterdir() if p.is_dir()} == {r['name'] for r in records},
            'Extra or missing numerical runs in the prescribed experiment.')
    frames, reuse = reused_frames(root,config,reference,baseline,prior)
    records += frames
    for record in records:
        record['recorded_curve_feasibility_audit'] = recorded_curve_feasibility(record['observations'])
    require(len({r['initial_plan_sha256'] for r in records}) == 1
            and all(r['environment']['hostname'] == baseline['environment']['hostname'] for r in records),
            'Different product starts or CPU hosts.')
    # Every source, receipt and saved file is bound before matrix evaluation.
    initial,evaluation = numerical_validation(root,ref,baseline,records)
    independent_winners = independent_selection(records,selection['selected'])
    reference_loss = baseline['independent_rounded']['loss']
    selected = dict(selection['selected'],**{'smoothing-fw':reuse['selected_candidate']})
    report = dict(schema_version=1,status='complete_validated',
        generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),script_sha256=sha256(__file__),
        reporting_dependency_sha256={str(prior_path.relative_to(ROOT)):sha256(prior_path),
                                     str(reference_path.relative_to(ROOT)):sha256(reference_path)},
        input_root=str(root),input_sha256={name:sha256(root/name) for name in
            ('local_config.json','selection.json','completed.json')},stage_sha256=stages,input_binding=binding,
        configuration=config,fixed_horizon=HORIZON,pairs_per_update=1024,optimizer_dtype='float64',
        expected_new_runs=14,observed_new_runs=14,expected_reused_runs=3,observed_reused_runs=3,
        expected_primary_confirmation_runs=9,observed_primary_confirmation_runs=9,
        comparison_axis='completed_updates',timing_comparison_performed=False,
        raw_timing_scope='Archived only; up to four one-thread processes overlap. No runtime aggregates or speed ranking.',
        primary_endpoint='Independently evaluated actual saved final rounded coupling; no further repair.',
        strict_saved_rounded_marginal_l1_tolerance=STRICT_FEASIBILITY,
        inner_actual_plan_marginal_l1_tolerance=INNER_TOLERANCE,inner_guarded_normalized_gap_tolerance=INNER_TOLERANCE,
        selected_candidates=selected,independent_tuning_selection_audit=independent_winners,
        frames_reuse=reuse,initial=initial,baseline=baseline,reference_loss=reference_loss,
        evaluation_environment=evaluation,runs=records,curves=curves(records),
        confirmation_summary=endpoint_summary(records,initial['loss'],reference_loss),
        caveats=[
            'Final and current rounded objectives are empirical diagnostics, not the papers’ random-output stationarity quantities.',
            'SMD/MSBPG use published updates with theorem-admissible prefactors, not authors’ empirical presets.',
            'Both numerical proximal criteria must pass at 1e-7; no outer theorem for this finite inexactness is claimed.',
            'Matching seeds and pair counts do not imply identical conditional sampled pairs or arithmetic work.',
            'Each method had four tuning candidates and three fresh seeds; FraMES outputs are reused once, without rerunning or reselection.',
            'The SCOT/POT endpoint is an attainable reference, not a certified optimum or a matched-iteration competitor.',
            'One fixed dataset and three seed replicates; ranges are descriptive, not confidence intervals.',
            'All saved rounded plans pass 100*eps64 in an independent NumPy check. No auxiliary post-run repair is used.'])
    out.mkdir(parents=True,exist_ok=True)
    write(out/'report.json',report)
    write(out/'selection_audit.json',dict(new_methods=independent_winners,frames=reuse))
    (out/'summary.txt').write_text(text_summary(report))
    save_csv(out/'curve_data.csv',report['curves'])
    rows = []
    theoretical = []
    for record in records:
        raw,rounded = record['independent']['last_raw'],record['independent']['last_rounded']
        inner = record['inner_audit']
        rows.append(dict(method=record['method_key'],phase=record['phase'],candidate=record['candidate']['id'],seed=record['seed'],
            updates=HORIZON,pair_samples=record['counters']['pair_samples'],raw_full_loss=raw['loss'],
            saved_rounded_full_loss=rounded['loss'],raw_marginal_l1=raw['residuals']['both_l1'],
            raw_column_l1=raw['residuals']['col_l1'],raw_column_l2=raw['residuals']['col_l2'],
            rounded_marginal_l1=rounded['residuals']['both_l1'],inner_iterations=inner['inner_iterations'],
            maximum_guarded_normalized_gap=inner.get('maxima',{}).get('normalized_gap_upper_bound'),
            underflow_entries=inner.get('underflow_entries',0),metadata_sha256=record['metadata_sha256']))
        if record['phase'] == 'confirm' and record['method_key'] != 'smoothing-fw':
            output = record['output_selection']
            theoretical.append(dict(method=record['method_key'],seed=record['seed'],
                sampled_iteration=output['index'],sampling_probability=output['probability'],
                sampled_raw_full_loss=record['independent']['selected_raw']['loss'],
                sampled_raw_marginal_l1=record['independent']['selected_raw']['residuals']['both_l1'],
                sampled_rounded_diagnostic_full_loss=record['independent']['selected_rounded']['loss'],
                msbpg_normalized_displacement_squared=output.get('recorded_update_statistic',{}).get('normalized_displacement_squared'),
                scope_caveat=output['scope_caveat']))
    save_csv(out/'endpoints.csv',rows)
    save_csv(out/'theory_outputs.csv',theoretical)
    plot(report,out)
    print(json.dumps(dict(status=report['status'],out=str(out),confirmation_summary=report['confirmation_summary'])))


if __name__ == '__main__':
    main()
