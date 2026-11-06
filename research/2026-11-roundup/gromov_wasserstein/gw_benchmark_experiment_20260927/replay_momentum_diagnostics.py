"""Reconstruct omitted momentum diagnostics without changing an original run.

The original run's immutable source_snapshot is imported in a fresh process.
Its recorded update count, rather than an elapsed-time stopping rule, is replayed.
At post-update output P_K, the actual stored momentum is d_(K-1): the primary
error pairs it with its own input P_(K-1), and a separate, explicitly stale
tracking error compares it with grad f(P_K). Initial diagnostics use d0/P0.
No new terminal oracle is drawn. These measurements are supplementary and do
not turn the prior-input error into a current-output d_K/P_K measurement.

diagnostics.csv is released only after exact checkpoint scalar/counter agreement,
bitwise final raw/rounded plan agreement, and provenance/environment checks.
Failed original runs are retained as unsupported, unverified records. Original
timings are copied as labels; all replay computation has its own timing account.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


SOURCE_NAMES = ('run_benchmark.py', 'smoothing_fw.py', 'gw_core.py')
COUNTER_NAMES = (
    'oracle_evaluations', 'exact_gradient_calls', 'stochastic_gradient_calls',
    'sketch_probes', 'anchor_samples', 'initialization_oracle_calls',
    'row_lmo_calls', 'row_projection_calls', 'transport_lmo_calls',
    'kl_projection_calls', 'euclidean_projection_calls', 'inner_iterations_total',
    'inner_stopping_failures', 'underflow_steps', 'underflow_entries_total',
    'underflow_entries_max', 'attempted_updates', 'completed_updates',
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def input_bindings(run_dir, instance):
    """Public stdlib helper used by companion manifests and receipts."""
    run_dir, instance = Path(run_dir), Path(instance)
    failed = json.loads((run_dir / 'metadata.json').read_text()).get('status') == 'failed'
    optional_hash = lambda path: None if failed and not path.exists() else sha256_file(path)
    return dict(
        metadata_sha256=sha256_file(run_dir / 'metadata.json'),
        trace_sha256=optional_hash(run_dir / 'trace.csv'),
        final_sha256=optional_hash(run_dir / 'final.npz'),
        instance_sha256=sha256_file(instance),
        source_file_sha256={name: sha256_file(run_dir / 'source_snapshot' / name)
                            for name in SOURCE_NAMES},
        theory_config_sha256=sha256_file(run_dir / 'source_snapshot' / 'theory_config.json'),
    )


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, label):
    result = int(value)
    require(not isinstance(value, bool) and float(value) == result and result >= 0,
            f'Invalid nonnegative integer: {label}')
    return result


def _restore_environment(original):
    # Called before numerical imports, including the original runner import.
    for key, value in original['thread_environment'].items():
        require(key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'),
                f'Unexpected original thread environment key: {key}')
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    workspace = original['cublas_workspace_config']
    if workspace is None:
        os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
    else:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = workspace


def _isolated_imports(snapshot):
    sys.dont_write_bytecode = True
    # Remove the live working checkout from module search, leaving the archived
    # implementation and installed third-party packages. Never fall back to live
    # kernels, including for an older snapshot with a different API.
    live = Path(__file__).resolve().parent
    sys.path[:] = [str(snapshot)] + [entry for entry in sys.path
                                     if entry and Path(entry).resolve() not in (live, snapshot)]
    modules = {}
    for name in ('gw_core', 'smoothing_fw', 'run_benchmark'):
        require(name not in sys.modules, f'Numerical module already imported: {name}')
        # run_benchmark imports the two modules above; they are intentionally the
        # already verified archived modules, not freshly reloaded versions.
        source = snapshot / (name + '.py')
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        # -B alone prevents writes but still permits cached-bytecode reads.
        # Compile the bound source bytes directly so stale .pyc files cannot
        # replace the implementation whose hash we checked.
        exec(compile(source.read_bytes(), str(source), 'exec'), module.__dict__)
        require(Path(module.__file__).resolve() == source,
                f'Import escaped the original source snapshot: {name}')
        modules[name] = module
    return modules['run_benchmark'], modules['smoothing_fw'], modules['gw_core']


def _check_environment(rb, torch, device, original):
    torch.set_num_threads(original['torch_num_threads'])
    torch.set_num_interop_threads(original['torch_num_interop_threads'])
    torch.use_deterministic_algorithms(original['deterministic_algorithms'])
    torch.set_float32_matmul_precision(original['float32_matmul_precision'])
    torch.backends.cuda.matmul.allow_tf32 = original['cuda_matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32 = original['cudnn_allow_tf32']
    require(original['amp'] is False and original['deterministic_algorithms'] is True,
            'Replay requires original deterministic no-AMP policy.')
    require(device.type == torch.device(original['device']).type,
            'Replay device class differs from the original run.')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    current = rb.environment(device)
    keys = ('packages', 'cuda_runtime', 'torch_num_threads', 'torch_num_interop_threads',
            'thread_environment', 'deterministic_algorithms', 'float32_matmul_precision',
            'cuda_matmul_allow_tf32', 'cudnn_allow_tf32', 'amp', 'cublas_workspace_config')
    for key in keys:
        require(current[key] == original[key], f'Replay environment mismatch: {key}')
    require(current['python'].split()[0] == original['python'].split()[0],
            'Replay Python version differs from original.')
    if device.type == 'cuda':
        for key in ('name', 'capability'):
            require(current['gpu'][key] == original['gpu'][key],
                    f'Replay GPU mismatch: {key}')
    else:
        require(current['cpu_model'] == original['cpu_model'], 'Replay CPU model differs.')
    return current


def replay_worker(args):
    """Worker entry point; use the CLI controller for genuine module isolation."""
    run_dir, instance, out = (Path(value).resolve() for value in
                              (args.run_dir, args.instance, args.out))
    require(out != run_dir and run_dir not in out.parents,
            'Replay outputs must be separate from original run.')
    out.mkdir(parents=True, exist_ok=True)
    require(not any((out / name).exists() for name in
                    ('verification.json', 'diagnostics.csv', 'candidate_diagnostics.csv')),
            'Replay output already exists; use a fresh directory.')
    started = time.perf_counter()
    report = dict(schema_version=1, status='failed', verified=False, error=None,
                  failure_kind=None, bindings={}, checks={}, checkpoint_count=0,
                  replay_diagnostic_counts=dict(loss_calls=0, full_gradient_calls=0,
                                                assignment_lmo_calls=0, rounding_calls=0,
                                                marginal_residual_calls=0, oracle_draws=0),
                  run_dir=str(run_dir), instance=str(instance),
                  utility_sha256=sha256_file(Path(__file__).resolve()),
                  timing=dict(replay_initialization_seconds=0., replay_update_seconds=0.,
                              replay_diagnostics_seconds=0.),
                  interpretation={
                      'primary': 'At output K>=1: ||d_(K-1)-grad f(P_(K-1))||_F; at K=0: ||d0-grad f(P0)||_F.',
                      'secondary': 'At output K>=1: ||d_(K-1)-grad f(P_K)||_F; stale stored momentum, not d_K.',
                      'terminal_oracle': 'No additional oracle or EMA update after the recorded terminal update.',
                      'clocks': 'Original clocks are copied labels. All reconstruction/diagnostic compute is separate.',
                      'coverage': 'Prior-input error is one update behind output, not a d_K/P_K extension of the original trajectory.'})
    diagnostics = []
    try:
        report['bindings'] = input_bindings(run_dir, instance)
        meta = json.loads((run_dir / 'metadata.json').read_text())
        config = meta['config']
        snapshot = run_dir / 'source_snapshot'
        report.update(original_source_commit=meta.get('source_commit'),
                      original_status=meta['status'], original_timing=meta.get('timing'),
                      original_config=config, original_counters=meta['counters'],
                      completed_updates=integer(meta['counters']['completed_updates'], 'completed_updates'))
        for name in SOURCE_NAMES:
            require(report['bindings']['source_file_sha256'][name] == meta['source_file_sha256'][name],
                    f'Original source hash mismatch: {name}')
        require(json.loads((snapshot / 'theory_config.json').read_text()) == meta['theory_config'],
                'Archived theory configuration differs from original metadata.')
        require(report['bindings']['instance_sha256'] == meta['instance_sha256'],
                'Canonical instance file hash mismatch.')
        report['checks']['source_and_instance_bindings'] = True
        require(config['method'] in ('smoothing-fw', 'smoothing-fw-exact', 'smoothing-row-projection'),
                'Only practical smoothing methods are supported.')
        if meta['status'] == 'failed':
            report['failure_kind'] = 'unsupported_original_failure'
            raise ValueError('Failed original: no verified completed endpoint is reconstructed; original evidence retained.')
        require(meta['status'] == 'completed', 'Original run is not a supported terminal success.')
        require(meta.get('final_float64_validation') is not None, 'Original final float64 validation is absent.')
        _restore_environment(meta['environment'])
        rb, smooth, core = _isolated_imports(snapshot)
        import numpy as np
        import torch
        from dataclasses import asdict, fields
        device = torch.device(args.device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', 0)
        report['replay_environment'] = _check_environment(rb, torch, device, meta['environment'])
        report['checks']['environment_match'] = True
        with np.load(instance, allow_pickle=False) as archive:
            canonical = {key: np.array(archive[key], copy=True) for key in ('A', 'B', 'a', 'b')}
        array_hashes = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in canonical.items()}
        require(array_hashes == meta['array_sha256'], 'Canonical array hash mismatch.')
        report['array_sha256'] = array_hashes
        dtype = getattr(torch, config['dtype'])
        problem = core.GWProblem(*(torch.as_tensor(canonical[key], dtype=dtype, device=device)
                                   for key in ('A', 'B', 'a', 'b')))
        resolved = meta['smoothing_config']
        momentum_config = smooth.MomentumConfig(**{field.name: resolved[field.name]
                                                   for field in fields(smooth.MomentumConfig)})
        require(momentum_config.oracle == config['oracle'] and momentum_config.batch == config['batch'],
                'Resolved oracle configuration is inconsistent.')
        require(momentum_config.update == ('row_projection' if config['method'] == 'smoothing-row-projection' else 'fw'),
                'Resolved update rule is inconsistent.')
        report['resolved_smoothing_config'] = asdict(momentum_config)
        with (run_dir / 'trace.csv').open(newline='') as stream:
            original_rows = list(csv.DictReader(stream))
        measured = []
        missed = []
        for index, row in enumerate(original_rows):
            if row.get('rounded_loss', '') != '':
                require(row['kind'] in ('initial', 'checkpoint', 'final'), 'Unknown measured checkpoint kind.')
                iteration = integer(row['iteration'], 'trace iteration')
                require(iteration <= report['completed_updates'], 'Checkpoint exceeds original update count.')
                measured.append((index, iteration, row))
            else:
                require(row['kind'] in ('missed', 'missed_checkpoint'), 'Non-measured trace row is not a missed checkpoint.')
                missed.append(dict(original_trace_row=index, **row))
        require(measured and measured[0][1] == 0 and measured[0][2]['kind'] == 'initial',
                'Original initial checkpoint is absent.')
        require(measured[-1][1] == report['completed_updates'], 'Original terminal checkpoint is absent.')
        require([item[1] for item in measured] == sorted({item[1] for item in measured}),
                'Original measured checkpoint iterations are repeated or out of order.')
        report['missed_original_checkpoints'] = missed
        targets = {iteration: (index, row) for index, iteration, row in measured}
        counters = dict.fromkeys(COUNTER_NAMES, 0)
        require(set(meta['counters']) == set(counters), 'Unsupported original counter schema.')
        rb.sync(device)
        init_started = time.perf_counter()
        initialization = config.get('initialization', 'product')
        init_keywords = dict(seed=config['seed'])
        if 'P0' in inspect.signature(smooth.initialize_state).parameters:
            P0, details = rb.make_initial_plan(problem, initialization, config['init_seed'])
            init_keywords['P0'] = P0
        else:
            require(initialization == 'product', 'Archived initializer supports product only.')
            details = {'mode': 'product', 'legacy_archive': True}
        state = smooth.initialize_state(problem, momentum_config, **init_keywords)
        rb.sync(device)
        report['timing']['replay_initialization_seconds'] = time.perf_counter() - init_started
        counters.update(asdict(state.counts))
        initial_hash = hashlib.sha256(state.P.detach().cpu().numpy().tobytes()).hexdigest()
        report['initialization_plan_sha256'] = initial_hash
        report['initialization_details'] = details
        original_initial_hash = meta.get('initialization_plan_sha256')
        if original_initial_hash is not None:
            require(initial_hash == original_initial_hash, 'Original initialization plan hash mismatch.')
        report['checks']['initialization_hash'] = ('matched' if original_initial_hash else
            'unavailable_in_legacy_original; initial_metrics_and_terminal_arrays_required')
        if 'initialization_details' in meta:
            require(details == meta['initialization_details'], 'Original initialization details mismatch.')
        if 'initialization_residuals' in meta:
            report['replay_diagnostic_counts']['marginal_residual_calls'] += 1
            require(core.marginal_residuals(state.P, problem.a, problem.b) == meta['initialization_residuals'],
                    'Original initialization residuals mismatch.')
        report['initial_rng_state_sha256'] = hashlib.sha256(state.generator.get_state().numpy().tobytes()).hexdigest()
        scalar_checks = []

        def evaluate(iteration, info):
            index, original = targets[iteration]
            rb.sync(device)
            diagnostic_start = time.perf_counter()
            generator_before = state.generator.get_state().clone()
            P = state.P
            R = core.round_transport(P, problem.a, problem.b)
            report['replay_diagnostic_counts']['rounding_calls'] += 1
            rb._assert_plan(P)
            rb._assert_plan(R)
            raw_residuals = core.marginal_residuals(P, problem.a, problem.b)
            rounded_residuals = core.marginal_residuals(R, problem.a, problem.b)
            report['replay_diagnostic_counts']['marginal_residual_calls'] += 2
            metrics = dict(raw_loss=problem.loss(P).item(), rounded_loss=problem.loss(R).item(),
                           repair_l1=(R - P).abs().sum().item(), **raw_residuals,
                           **{'rounded_' + key: value for key, value in rounded_residuals.items()})
            mu = momentum_config.schedule(iteration)['mu']
            report['replay_diagnostic_counts']['loss_calls'] += 2
            G = problem.gradient(P)
            report['replay_diagnostic_counts']['full_gradient_calls'] += 1
            H = G + (P.sum(0) - problem.b)[None, :] / mu
            metrics.update(mu=mu, smoothing_point_iteration=iteration,
                           true_smoothed_gap=((H * P).sum() - (problem.a * H.min(1).values).sum()).item())
            n, m = P.shape
            gap_required = n == m and max(n, m) <= config['gap_max_n']
            require((original.get('original_fw_gap', '') != '') == gap_required,
                    f'Original FW gap coverage mismatch at iteration {iteration}.')
            if gap_required:
                rounded_gradient = problem.gradient(R)
                report['replay_diagnostic_counts']['full_gradient_calls'] += 1
                report['replay_diagnostic_counts']['assignment_lmo_calls'] += 1
                S = core.assignment_lmo(rounded_gradient, problem.a, problem.b)
                metrics['original_fw_gap'] = (rounded_gradient * (R - S)).sum().item()
            if iteration:
                metrics.update({'last_update_' + key: value for key, value in momentum_config.schedule(iteration - 1).items()})
            metrics.update({'cumulative_' + key: value for key, value in counters.items()})
            point = state.P if info is None else info.P_before
            estimate = state.d if info is None else info.d_used
            point_iteration = 0 if info is None else info.update_index
            exact_at_input = G if info is None else problem.gradient(point)
            report['replay_diagnostic_counts']['full_gradient_calls'] += int(info is not None)
            matched_error = (estimate - exact_at_input).norm().item()
            stale_error = (estimate - G).norm().item()
            diag = dict(original_trace_row=index, kind=original['kind'], output_iteration=iteration,
                        momentum_point_iteration=point_iteration,
                        momentum_error_matches_current_output=point_iteration == iteration,
                        momentum_error_frobenius=matched_error,
                        stored_momentum_iteration=point_iteration,
                        tracking_error_point_iteration=iteration,
                        stored_momentum_current_output_tracking_error_frobenius=stale_error,
                        matched_input_raw_loss=problem.loss(point).item(),
                        matched_input_col_l2=(point.sum(0) - problem.b).norm().item(),
                        matched_input_gradient_frobenius=exact_at_input.norm().item(),
                        current_output_gradient_frobenius=G.norm().item(),
                        original_target_solver_seconds=original.get('target_solver_seconds'),
                        original_actual_solver_seconds=original['actual_solver_seconds'],
                        original_time_to_this_feasible_output=original.get('time_to_this_feasible_output'),
                        original_solver_oracle_evaluations=counters['oracle_evaluations'],
                        replay_diagnostic_oracle_draws=0,
                        **{'recomputed_' + key: value for key, value in metrics.items()})
            report['replay_diagnostic_counts']['loss_calls'] += 1
            diagnostics.append(diag)
            require(all(math.isfinite(value) for value in diag.values() if isinstance(value, float)),
                    f'Nonfinite diagnostic at output iteration {iteration}.')
            require(torch.equal(generator_before, state.generator.get_state()),
                    f'Diagnostic changed oracle RNG at iteration {iteration}.')
            compared = []
            for key, value in metrics.items():
                if original.get(key, '') == '':
                    require(value is None, f'Original checkpoint field missing: {key}')
                    continue
                require(value is not None and math.isfinite(float(original[key])) and float(original[key]) == value,
                        f'Checkpoint scalar mismatch at iteration {iteration}: {key}: original={original[key]}, replay={value}')
                compared.append(key)
            # Newer prospective runners already log these exact quantities.
            # Historical snapshots omit them; neither changes replay work.
            for key in ('momentum_point_iteration', 'momentum_error_matches_current_output',
                        'momentum_error_frobenius', 'stored_momentum_iteration',
                        'tracking_error_point_iteration',
                        'stored_momentum_current_output_tracking_error_frobenius'):
                if original.get(key, '') == '':
                    continue
                if isinstance(diag[key], bool):
                    require(original[key] in ('True', 'False'), f'Invalid original boolean: {key}')
                expected = (original[key] == 'True' if isinstance(diag[key], bool) else float(original[key]))
                require(expected == diag[key], f'Original momentum diagnostic mismatch at {iteration}: {key}')
                compared.append(key)
            scalar_checks.append(dict(original_trace_row=index, iteration=iteration,
                                      compared_fields=compared, exact_match=True))
            rb.sync(device)
            report['timing']['replay_diagnostics_seconds'] += time.perf_counter() - diagnostic_start
            return R

        with torch.no_grad():
            final_rounded = evaluate(0, None)
            for completed in range(1, report['completed_updates'] + 1):
                counters['attempted_updates'] += 1
                rb.sync(device)
                update_started = time.perf_counter()
                info = smooth.step(problem, state)
                rb._assert_plan(state.P)
                counters.update(asdict(state.counts))
                counters['completed_updates'] += 1
                rb.sync(device)
                report['timing']['replay_update_seconds'] += time.perf_counter() - update_started
                if completed in targets:
                    final_rounded = evaluate(completed, info)
                del info
        require(counters == meta['counters'], 'Final original solver counter mismatch.')
        report['replay_counters'] = counters
        report['checks']['checkpoint_scalars'] = scalar_checks
        report['checks']['solver_counters'] = True
        report['checks']['diagnostics_preserved_oracle_rng'] = True
        report['terminal_rng_state_sha256'] = hashlib.sha256(state.generator.get_state().numpy().tobytes()).hexdigest()
        report['endpoint_comparisons'] = {}
        mismatched_plans = []
        with np.load(run_dir / 'final.npz', allow_pickle=False) as final:
            require(set(final.files) == {'P_raw', 'P_rounded'}, 'Original final plans are incomplete.')
            for name, tensor in (('P_raw', state.P), ('P_rounded', final_rounded)):
                actual, expected = tensor.detach().cpu().numpy(), final[name]
                shape_match, dtype_match = actual.shape == expected.shape, actual.dtype == expected.dtype
                bitwise_match = shape_match and dtype_match and actual.tobytes() == expected.tobytes()
                difference = (np.abs(actual.astype(np.float64) - expected.astype(np.float64))
                              if shape_match else None)
                maximum = float(difference.max()) if difference is not None and difference.size else None
                report['endpoint_comparisons'][name] = dict(
                    expected_array_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
                    replay_array_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
                    expected_shape=list(expected.shape), replay_shape=list(actual.shape),
                    expected_dtype=str(expected.dtype), replay_dtype=str(actual.dtype),
                    bitwise_match=bitwise_match,
                    max_abs_difference=maximum if maximum is not None and math.isfinite(maximum) else None,
                    difference_all_finite=bool(np.isfinite(difference).all()) if difference is not None else None)
                if not bitwise_match:
                    mismatched_plans.append(name)
                report['checks'][name + '_bitwise'] = bitwise_match
                report[name + '_array_sha256'] = hashlib.sha256(actual.tobytes()).hexdigest()
        require(not mismatched_plans, 'Final bitwise plan mismatch: ' + ', '.join(mismatched_plans))
        require(input_bindings(run_dir, instance) == report['bindings'], 'Original input changed during replay.')
        report['checks']['original_files_unchanged'] = True
        report['checkpoint_count'] = len(diagnostics)
        report.update(status='verified', verified=True)
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        if report['failure_kind'] is None:
            report['failure_kind'] = 'reconstruction_verification_failed'
        traceback.print_exc()
    finally:
        if report['bindings']:
            try:
                unchanged = input_bindings(run_dir, instance) == report['bindings']
            except Exception:
                unchanged = False
            report['checks']['original_files_unchanged'] = unchanged
            if not unchanged:
                report.update(status='failed', verified=False,
                              failure_kind='original_inputs_changed_during_replay',
                              error=(report.get('error') or '') + '; Original input changed during replay.')
        report['timing']['replay_worker_total_seconds'] = time.perf_counter() - started
        report['timing']['replay_other_seconds'] = (report['timing']['replay_worker_total_seconds']
            - sum(report['timing'][key] for key in ('replay_initialization_seconds',
                      'replay_update_seconds', 'replay_diagnostics_seconds')))
        report['candidate_checkpoint_count'] = len(diagnostics)
        if diagnostics:
            candidate = out / 'candidate_diagnostics.csv'
            write_csv(candidate, diagnostics)
            if report['verified']:
                candidate.replace(out / 'diagnostics.csv')
                report['diagnostics_sha256'] = sha256_file(out / 'diagnostics.csv')
        write_json(out / 'verification.json', report)
    return 0 if report['verified'] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--instance', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--_worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args._worker:
        return replay_worker(args)
    run_dir, out = Path(args.run_dir).resolve(), Path(args.out).resolve()
    require(out != run_dir and run_dir not in out.parents, 'Replay outputs must be separate from original run.')
    out.mkdir(parents=True, exist_ok=True)
    require(not any((out / name).exists() for name in
                    ('verification.json', 'diagnostics.csv', 'candidate_diagnostics.csv', 'worker.log')),
            'Replay output already exists; use a fresh directory.')
    started = time.perf_counter()
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--_worker',
               '--run-dir', str(run_dir), '--instance', str(Path(args.instance).resolve()),
               '--out', str(out), '--device', args.device]
    with (out / 'worker.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False,
                                env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    report_path = out / 'verification.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        report['timing']['controller_elapsed_seconds'] = time.perf_counter() - started
        report['worker_exit_code'] = result.returncode
        if result.returncode != 0 and report.get('verified'):
            report.update(status='failed', verified=False, failure_kind='worker_exit_after_verification',
                          error='Worker exited abnormally after writing verification.')
            if (out / 'diagnostics.csv').exists():
                (out / 'diagnostics.csv').replace(out / 'candidate_diagnostics.csv')
            report.pop('diagnostics_sha256', None)
        write_json(report_path, report)
    else:
        report = dict(schema_version=1, status='failed', verified=False,
                      failure_kind='worker_failed_without_verification', worker_exit_code=result.returncode,
                      error='Worker terminated without a verification record; see worker.log.',
                      timing={'controller_elapsed_seconds': time.perf_counter() - started})
        write_json(report_path, report)
    print(json.dumps({key: report.get(key) for key in ('status', 'verified', 'failure_kind', 'error')}))
    return 0 if result.returncode == 0 and report['verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
