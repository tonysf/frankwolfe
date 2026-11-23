"""Prospective CPU precision control at fixed update counts, not a timing study.

Frozen n128 seed1000, exact/sketch128, float32/float64, seed0, 2000 updates.
No trajectory averages, fitted rates, last-iterate guarantees, or GPU claims.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from gw_core import GWProblem, assignment_lmo, rademacher, round_transport, row_lmo
from smoothing_fw import MomentumConfig, initialize_state, step


ROOT = Path(__file__).resolve().parent
CHECKPOINTS = (0, 200, 1000, 2000)
UPDATES = 2000
SEED = 0
BATCH = 128


def digest(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        value = value.tobytes()
    return hashlib.sha256(value).hexdigest()


def git(*argv):
    result = subprocess.run(['git', '-C', str(ROOT), *argv], capture_output=True, text=True)
    return {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def matched_reference_stream():
    generators = {name: torch.Generator(device='cpu').manual_seed(SEED)
                  for name in ('float32', 'float64')}
    likes = {'float32': torch.empty((), dtype=torch.float32),
             'float64': torch.empty((), dtype=torch.float64)}
    probe_hash = hashlib.sha256()
    states = []
    for _ in range(UPDATES):
        draws = {name: rademacher(128, BATCH, like=likes[name], generator=generators[name])
                 for name in generators}
        assert torch.equal(draws['float32'].double(), draws['float64'])
        assert torch.equal(generators['float32'].get_state(), generators['float64'].get_state())
        probe_hash.update(draws['float32'].to(torch.int8).numpy().tobytes())
        states.append(digest(generators['float32'].get_state()))
    return states, probe_hash.hexdigest()


def residuals(P, a, b):
    row, col = P.sum(1) - a, P.sum(0) - b
    return {'row_l1': row.abs().sum().item(), 'row_l2': row.norm().item(),
            'col_l1': col.abs().sum().item(), 'col_l2': col.norm().item(),
            'mass_error': abs(P.sum().item() - 1), 'min_entry': P.min().item()}


@torch.no_grad()
def run_one(arrays, reference, oracle, dtype_name, expected_states, out):
    out.mkdir()
    dtype = getattr(torch, dtype_name)
    problem = GWProblem(*(torch.tensor(arrays[key], dtype=dtype) for key in ('A', 'B', 'a', 'b')))
    config = MomentumConfig(alpha=.75, beta=.5, delta=.25, gamma0=1., rho0=1.,
                            mu0=1/128, oracle=oracle, batch=BATCH)
    config_record = {'oracle': oracle, 'dtype': dtype_name, 'device': 'cpu',
                     'updates': UPDATES, 'seed': SEED, 'm_mu0': 1.,
                     'schedule': config.metadata(), 'checkpoints': list(CHECKPOINTS)}
    save_json(out / 'config.json', config_record)
    checkpoints = []
    tensors = {}
    generator_states = []
    started = time.perf_counter()
    state = initialize_state(problem, config, seed=SEED)
    initial_state_hash = digest(torch.Generator(device='cpu').manual_seed(SEED).get_state())

    def record_draw():
        state_hash = digest(state.generator.get_state())
        if oracle == 'sketch':
            index = len(generator_states)
            assert state_hash == expected_states[index], (oracle, dtype_name, index)
        else:
            assert state_hash == initial_state_hash
        generator_states.append(state_hash)

    def observe(k, P, momentum, fresh, completed_updates):
        assert torch.isfinite(P).all() and P.min().item() >= 0
        # This is the actual repair in the solver precision, then promoted intact.
        rounded = round_transport(P, problem.a, problem.b)
        raw64, rounded64 = P.double(), rounded.double()
        gradient = reference.gradient(raw64)
        schedule = config.schedule(k)
        H = gradient + (raw64.sum(0) - reference.b)[None, :] / schedule['mu']
        true_gap = (H * (raw64 - row_lmo(H, reference.a))).sum().item()
        rounded_gradient = reference.gradient(rounded64)
        rounded_gap = (rounded_gradient * (rounded64 - assignment_lmo(
            rounded_gradient, reference.a, reference.b))).sum().item()
        row = {'oracle': oracle, 'solver_dtype': dtype_name,
               'point_iteration': k, 'completed_updates_at_observation': completed_updates,
               **{key: schedule[key] for key in ('gamma', 'rho', 'mu')},
               **{'raw_' + key: value for key, value in residuals(raw64, reference.a, reference.b).items()},
               **{'rounded_' + key: value for key, value in residuals(rounded64, reference.a, reference.b).items()},
               'raw_true_smoothed_gap': true_gap,
               'raw_loss': reference.loss(raw64).item(),
               'rounded_loss': reference.loss(rounded64).item(),
               'rounded_original_fw_gap': rounded_gap,
               'repair_entrywise_l1': (raw64 - rounded64).abs().sum().item(),
               'same_point_momentum_available': momentum is not None,
               'momentum_error_frobenius': None,
               'momentum_error_relative_frobenius': None,
               'fresh_estimator_error_frobenius': None,
               'evaluation_dtype': 'float64',
               'rounded_output_policy': 'round actual raw point in solver dtype, then promote unchanged'}
        if momentum is not None:
            error = (momentum.double() - gradient).norm().item()
            row['momentum_error_frobenius'] = error
            row['momentum_error_relative_frobenius'] = error / max(gradient.norm().item(), 1e-30)
            row['fresh_estimator_error_frobenius'] = (fresh.double() - gradient).norm().item()
            tensors[f'd_{k}'] = momentum.numpy()
            tensors[f'fresh_gradient_{k}'] = fresh.numpy()
        assert all(np.isfinite(value) for value in row.values() if isinstance(value, float)), row
        assert row['rounded_min_entry'] >= 0
        tolerance = 32 * torch.finfo(dtype).eps
        assert row['rounded_row_l1'] + row['rounded_col_l1'] <= tolerance, row
        tensors[f'P_{k}'] = P.numpy()
        tensors[f'P_rounded_{k}'] = rounded.numpy()
        checkpoints.append(row)

    record_draw()
    observe(0, state.P, state.d, state.d, 0)
    for _ in range(UPDATES):
        info = step(problem, state)
        if info.oracle_evaluations_this_step:
            record_draw()
        if info.update_index in (200, 1000):
            observe(info.update_index, info.P_before, info.d_used,
                    info.gradient_estimate, info.completed_updates)
    # No d_2000 exists after exactly 2000 updates. Do not compare d_1999 to P_2000.
    observe(UPDATES, state.P, None, None, state.iteration)
    elapsed = time.perf_counter() - started
    counts = asdict(state.counts)
    assert state.iteration == UPDATES
    assert len(generator_states) == counts['oracle_evaluations'] == UPDATES
    assert counts['row_lmo_calls'] == UPDATES
    assert counts['sketch_probes'] == (BATCH * UPDATES if oracle == 'sketch' else 0)
    assert counts['exact_gradient_calls'] == (UPDATES if oracle == 'exact' else 0)
    assert [row['point_iteration'] for row in checkpoints] == list(CHECKPOINTS)
    with (out / 'checkpoints.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checkpoints[0]))
        writer.writeheader()
        writer.writerows(checkpoints)
    np.savez_compressed(out / 'checkpoints.npz', **tensors)
    save_json(out / 'generator_state_sha256.json', generator_states)
    result = config_record | {'status': 'completed', 'counts': counts, 'checkpoints': checkpoints,
                              'generator_state_stream_sha256': digest(''.join(generator_states).encode()),
                              'wall_seconds_including_diagnostics': elapsed,
                              'not_a_performance_measurement': True}
    save_json(out / 'summary.json', result)
    print(f'{oracle} {dtype_name}: 2000 updates; final raw column L1 '
          f'{checkpoints[-1]["raw_col_l1"]:.6g}; elapsed including diagnostics {elapsed:.2f}s', flush=True)
    return result


def write_reports(report, report_prefix):
    report_prefix.parent.mkdir(parents=True, exist_ok=True)
    save_json(report_prefix.with_suffix('.json'), report)
    lines = ['# CPU precision control', '',
             'Fixed frozen n=128 random-metric instance, data seed1000, noise0. '
             'Four 2,000-update controls use product initialization and the same schedule: '
             'gamma0=rho0=1, exponents (3/4,1/2,1/4), mu0=1/128. '
             'Sketch runs use 128 fresh Rademacher probes per oracle call, seed0. '
             'This is one problem and one stochastic seed, not a performance comparison or an independent-replicate study.', '',
             'All metrics below are evaluated in float64 from actual solver outputs. '
             'Rounded outputs were repaired in their original solver precision and then promoted unchanged. '
             'Canonical costs were stored in float32 and promoted for the float64 solver; '
             'this isolates arithmetic precision on the same frozen problem.', '',
             '| Oracle | Solver precision | Point k | Raw row L1 | Raw column L1 | Raw column L2 | True smoothed gap | Rounded loss | Momentum error |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for result in report['runs']:
        for row in result['checkpoints']:
            momentum = row['momentum_error_frobenius']
            lines.append(f"| {result['oracle']} | {result['dtype']} | {row['point_iteration']} | "
                         f"{row['raw_row_l1']:.6g} | {row['raw_col_l1']:.6g} | {row['raw_col_l2']:.6g} | "
                         f"{row['raw_true_smoothed_gap']:.6g} | {row['rounded_loss']:.8g} | "
                         + ('unavailable |' if momentum is None else f'{momentum:.6g} |'))
    lines += ['', 'At k=200 and k=1000, the point and its same-point momentum are retained '
              'from the next update record (201/1001 updates completed when observed). '
              'At k=2000, no same-point momentum was sampled: d1999 belongs to P1999 and is not used for a P2000 error.', '',
              'The true gap is the raw row-domain smoothed FW gap using the exact gradient and mu_k. '
              'CSV/JSON also preserve raw row L2, rounded marginal errors, original FW gaps at the rounded points, '
              'fresh estimator errors, and repair distances. Small signed numerical losses/gaps are not clipped.', '',
              '## Interpretation', '']
    for comparison in report['comparisons']:
        lines.append(f"- {comparison['oracle']}: final raw column L1 is "
                     f"{comparison['float32_final_raw_col_l1']:.6g} in float32 and "
                     f"{comparison['float64_final_raw_col_l1']:.6g} in float64 "
                     f"(float64/float32={comparison['float64_over_float32_col_l1']:.4g}). "
                     f"The actual rounded column L1 is {comparison['float32_final_rounded_col_l1']:.3g} "
                     f"and {comparison['float64_final_rounded_col_l1']:.3g}, respectively.")
    lines += ['', report['interpretation'], '',
              'No all-iterate averages were collected. These four checkpoints establish neither a convergence rate '
              'nor a last-iterate theorem claim, nor a numerical asymptotic floor. '
              'Matching discrete probes does not require equal optimization paths: rounding can change argmin decisions.', '',
              f"Matched-probe check: every one of {UPDATES} independently replayed float32/float64 probe batches "
              'was exactly equal after promotion; actual solver generator states matched the replay after every draw. '
              'Exact-gradient runs never advanced their independent seed0 generator. '
              f"Probe stream SHA256: `{report['matched_probes']['discrete_int8_stream_sha256']}`.", '',
              f"Git revision: `{report['provenance']['git_revision']['stdout'].strip()}`. "
              'The run retains executed source snapshots, exact hashes, dirty status, fixed configuration, '
              'raw/rounded checkpoint arrays and per-draw RNG-state hashes.', '']
    report_prefix.with_suffix('.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7), sharey=True)
    for axis, oracle in zip(axes, ('exact', 'sketch')):
        for result in report['runs']:
            if result['oracle'] != oracle:
                continue
            points = [row for row in result['checkpoints'] if row['point_iteration'] > 0]
            axis.plot([row['point_iteration'] for row in points],
                      [row['raw_col_l1'] for row in points], marker='o', label=result['dtype'])
        axis.set(title=oracle if oracle == 'exact' else 'sketch (128 probes)',
                 xlabel='Completed-update point k', xscale='log', yscale='log')
        axis.grid(alpha=.25)
        axis.legend()
    axes[0].set_ylabel('Raw column L1 residual (evaluated in float64)')
    fig.suptitle('CPU precision control · one fixed n=128 problem · seed0')
    fig.tight_layout()
    fig.savefig(report_prefix.with_suffix('.png'), dpi=170)
    fig.savefig(report_prefix.with_suffix('.pdf'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'runs/precision_control')
    parser.add_argument('--report-prefix', type=Path, default=ROOT / 'reports/precision_control')
    args = parser.parse_args()
    for suffix in ('.json', '.md', '.png', '.pdf'):
        if args.report_prefix.with_suffix(suffix).exists():
            raise FileExistsError(args.report_prefix.with_suffix(suffix))
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot = args.out / 'source_snapshot'
    snapshot.mkdir()
    source_paths = ('precision_control.py', 'gw_core.py', 'smoothing_fw.py',
                    'configs/theory_config.json', 'PROTOCOL.md')
    hashes = {}
    for relative in source_paths:
        original = ROOT / relative
        saved = snapshot / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, saved)
        hashes[relative] = digest(saved.read_bytes())
    instance = ROOT / 'data/metric128_s1000.npz'
    manifest_path = instance.with_suffix('.json')
    manifest = json.loads(manifest_path.read_text())
    assert manifest['n'] == manifest['m'] == 128 and manifest['seed'] == 1000 and manifest['noise'] == 0
    with np.load(instance, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in ('A', 'B', 'a', 'b')}
    assert digest(arrays['A']) == manifest['A_sha256']
    assert digest(arrays['B']) == manifest['B_sha256']
    theory = json.loads((ROOT / 'configs/theory_config.json').read_text())
    for name, expected in dict(alpha=.75, beta=.5, delta=.25, gamma0=1., rho0=1.).items():
        assert theory[name] == expected
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = GWProblem(*(torch.tensor(arrays[key], dtype=torch.float64) for key in ('A', 'B', 'a', 'b')))
    config = {'scope': 'prospective CPU precision control; no GPU/performance/theorem-average claim',
              'n': 128, 'data_seed': 1000, 'stochastic_seed': SEED, 'noise': 0,
              'updates': UPDATES, 'checkpoints': CHECKPOINTS, 'oracles': ['exact', 'sketch'],
              'sketch_batch': BATCH, 'dtypes': ['float32', 'float64'],
              'schedule': {name: theory[name] for name in ('alpha', 'beta', 'delta', 'gamma0', 'rho0')},
              'mu0': 1/128, 'm_mu0': 1., 'initialization': 'product',
              'instance_sha256': digest(instance.read_bytes()),
              'instance_manifest_sha256': digest(manifest_path.read_bytes()),
              'array_sha256': {key: digest(value) for key, value in arrays.items()},
              'evaluation_dtype': 'float64; actual solver-precision rounded outputs promoted unchanged'}
    save_json(args.out / 'config.json', config)  # Persist prospective choices before any run.
    provenance = {'started_at_utc': datetime.now(timezone.utc).isoformat(),
                  'git_revision': git('rev-parse', 'HEAD'), 'git_status': git('status', '--porcelain'),
                  'executed_source_sha256': hashes, 'hostname': platform.node(),
                  'platform': platform.platform(), 'python': sys.version,
                  'torch': torch.__version__, 'numpy': np.__version__,
                  'device': 'cpu', 'threads': torch.get_num_threads(),
                  'interop_threads': torch.get_num_interop_threads(),
                  'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                  'matmul_precision': torch.get_float32_matmul_precision(),
                  'allow_tf32': False, 'amp': False}
    save_json(args.out / 'provenance.json', provenance)
    report = {'status': 'started', 'config': config, 'provenance': provenance, 'runs': []}
    try:
        expected_states, stream_hash = matched_reference_stream()
        save_json(args.out / 'reference_generator_state_sha256.json', expected_states)
        report['matched_probes'] = {'batches': UPDATES, 'probes_per_batch': BATCH,
                                   'all_discrete_probes_equal': True,
                                   'all_actual_postdraw_states_equal_to_reference': True,
                                   'discrete_int8_stream_sha256': stream_hash,
                                   'reference_state_stream_sha256': digest(''.join(expected_states).encode())}
        with torch.no_grad():
            for oracle in ('exact', 'sketch'):
                for dtype_name in ('float32', 'float64'):
                    result = run_one(arrays, reference, oracle, dtype_name, expected_states,
                                     args.out / f'{oracle}_{dtype_name}')
                    report['runs'].append(result)
        report['comparisons'] = []
        for oracle in ('exact', 'sketch'):
            matching = {run['dtype']: run['checkpoints'][-1] for run in report['runs'] if run['oracle'] == oracle}
            fp32, fp64 = matching['float32'], matching['float64']
            report['comparisons'].append({'oracle': oracle,
                'float32_final_raw_col_l1': fp32['raw_col_l1'],
                'float64_final_raw_col_l1': fp64['raw_col_l1'],
                'float64_over_float32_col_l1': fp64['raw_col_l1'] / fp32['raw_col_l1'],
                'float32_final_rounded_col_l1': fp32['rounded_col_l1'],
                'float64_final_rounded_col_l1': fp64['rounded_col_l1']})
        substantial_in_both = all(row['float64_final_raw_col_l1'] > 1e-5 for row in report['comparisons'])
        report['interpretation'] = (
            'Substantial raw column infeasibility remains with float64 in both controls. '
            'Thus these finite-update residuals cannot be explained solely by float32 roundoff. '
            'The much smaller residuals of the actual rounded outputs distinguish arithmetic-scale repair error '
            'from the raw solver residual. This supports a finite-time algorithmic explanation for this control; '
            'it does not identify a unique cause or establish an asymptotic floor.' if substantial_in_both else
            'The precision comparisons above should be interpreted per oracle. This bounded control alone '
            'does not establish whether a numerical floor explains the raw residuals or an asymptotic trend.')
        # Refuse silently mixed execution/source snapshots if another edit touched an imported kernel.
        assert all(digest((ROOT / relative).read_bytes()) == value for relative, value in hashes.items())
        report['status'] = 'completed'
        report['provenance']['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        write_reports(report, args.report_prefix)
    except Exception:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        save_json(args.out / 'summary.json', report)
    print(args.report_prefix.with_suffix('.md'))


if __name__ == '__main__':
    main()
