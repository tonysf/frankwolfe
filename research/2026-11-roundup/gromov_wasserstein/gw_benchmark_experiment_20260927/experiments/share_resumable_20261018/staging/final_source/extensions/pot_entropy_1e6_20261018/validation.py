"""CPU/A100 epsilon=1e-6 checks and bounded actual-data conditioning probes."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
BASE = HERE.parent/'pot_entropy_init_20261018'
spec = importlib.util.spec_from_file_location('entropy_addon_validation_base', BASE/'validation.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
runner, reference = base.runner, base.reference
atomic_json, sha256, CheckpointStore = base.atomic_json, base.sha256, base.CheckpointStore
EPSILON = 1e-6


def write_fixture(path):
    path.mkdir(parents=True)
    rng = np.random.default_rng(849)
    x, y = rng.normal(size=(8, 3)), rng.normal(size=(8, 3))
    # Keep cost/epsilon nontrivial but numerically suitable for an independent
    # unscaled dense Sinkhorn reference. Actual dataset probes are unmodified.
    arrays = dict(A=(.0009*np.linalg.norm(x[:, None]-x[None, :], axis=-1)).astype('float32'),
                  B=(.0009*np.linalg.norm(y[:, None]-y[None, :], axis=-1)).astype('float32'),
                  a=np.full(8, 1/8, dtype='float64'), b=np.full(8, 1/8, dtype='float64'))
    for name, value in arrays.items():
        np.save(path/(name+'.npy'), value)
    atomic_json(path/'manifest.json', dict(n=8, m=8,
        array_files_sha256={name+'.npy': sha256(path/(name+'.npy')) for name in arrays}))


@torch.no_grad()
def tiny_checks(instance, device):
    p, _ = runner.load_problem(instance, device=device, exact_block_rows=4)
    checks = []
    A, B, a, b = [v.cpu().numpy() for v in (p.A, p.B, p.a, p.b)]
    for kind in ('product', 'permutation-mixture'):
        plan = base.initial(p, kind, 901)
        expected = plan.cpu().numpy().copy()
        prox = base.DistributedCertifiedPOTProx(p, gap_tol=1e-11, certificate_block_rows=4)
        errors = []
        for _ in range(3):
            direct = reference.dense_gradient(A, B, expected)
            gradient = p.gradient(plan)
            assert np.max(np.abs(gradient.cpu().numpy()-direct)) <= 1e-12
            plan, _, info = prox(runner.product_log_reference(p), gradient,
                                 eta=1/EPSILON, tol=1e-12, check_every=1)
            assert info['converged'] and info['proximal_optimality_certified']
            expected = reference.dense_sinkhorn(direct, a, b, eta=1/EPSILON)
            errors.append(float(np.max(np.abs(plan.cpu().numpy()-expected))))
            assert errors[-1] <= 2e-9, (kind, errors[-1])
        checks.append(dict(name='independent_full_gradient_transport', epsilon=EPSILON,
                           initialization=kind, max_plan_error=max(errors)))
    # Large additive row/column potentials force underflow, while the exact
    # transport solution remains the product coupling. Never clip the kernel.
    gradient = torch.zeros((p.n, p.m), device=p.device, dtype=torch.float64)
    gradient[0, :] = 1.
    gradient[:, 0] += 1.
    prox = base.DistributedCertifiedPOTProx(p, certificate_block_rows=4)
    plan, info = runner.pot_step(p, runner.product_log_reference(p), gradient, prox, epsilon=EPSILON)
    assert info['sinkhorn_path'] == 'original-log-domain-fallback'
    assert torch.max(torch.abs(plan-torch.outer(p.a, p.b))).item() < 1e-9
    checks.append(dict(name='low_entropy_underflow_log_fallback', epsilon=EPSILON,
                       marginal_l1=info['marginal_l1'],
                       normalized_gap_upper_bound=info['normalized_gap_upper_bound']))
    return checks


def tiny_restart(instance, out, device):
    command = [sys.executable, str(BASE/'runner.py'), '--instance', str(instance),
               '--device', device, '--initialization', 'permutation-mixture', '--seed', '901',
               '--epsilon', str(EPSILON), '--iterations', '4', '--checkpoint-every', '1',
               '--diagnostic-every', '1', '--max-seconds', '600', '--retain-checkpoint']
    split, full = out/'tiny_restart', out/'tiny_uninterrupted'
    subprocess.run(command+['--out', str(split), '--chunk-updates', '1'], check=True)
    part = json.loads((split/'progress.json').read_text())
    assert part['status'] == 'checkpointed' and part['completed_updates'] == 1
    subprocess.run(command+['--out', str(split)], check=True)
    subprocess.run(command+['--out', str(full)], check=True)
    result, original = [json.loads((folder/'metadata.json').read_text()) for folder in (split, full)]
    assert result['status'] == 'completed' and result['completed_updates'] == 4
    for name in ('trace', 'outer_trace', 'inner_trace', 'counters', 'stopping_reason'):
        assert original[name] == result[name], name
    states = [CheckpointStore(folder/'checkpoints').load(identity=meta['identity'])[0]
              for folder, meta in ((split, result), (full, original))]
    assert torch.equal(states[0]['plan'], states[1]['plan'])
    assert result['max_memory_reserved_fraction'] < .9
    for folder in (split, full):
        CheckpointStore(folder/'checkpoints').remove_completed_state()
    return dict(name='exact_tiny_restart', epsilon=EPSILON, iterations=4,
                max_memory_reserved_fraction=result['max_memory_reserved_fraction'],
                final_loss=result['final_loss'])


def actual_probe(instance, out, device, initialization):
    folder = out/('actual_probe_'+initialization)
    command = [sys.executable, str(BASE/'runner.py'), '--instance', str(instance),
               '--out', str(folder), '--device', device, '--initialization', initialization,
               '--seed', '901', '--epsilon', str(EPSILON), '--iterations', '2',
               '--diagnostic-every', '1', '--checkpoint-every', '1', '--max-seconds', '900']
    executed = subprocess.run(command)
    if executed.returncode:
        failure = json.loads((folder/'failure.json').read_text())
        assert failure['failure_kind'] == 'inner_certificate_not_met', failure['error']
        record = dict(name='actual_low_entropy_probe', initialization=initialization,
                      epsilon=EPSILON, status='inner_certificate_not_met',
                      completed_updates=failure['completed_updates'], certificate=failure['certificate'])
    else:
        result = json.loads((folder/'metadata.json').read_text())
        assert result['status'] in ('completed', 'checkpointed')
        assert result['max_memory_reserved_fraction'] < .9
        record = dict(name='actual_low_entropy_probe', initialization=initialization,
                      epsilon=EPSILON,
                      status='completed' if result['status']=='completed' else 'time_budget_exhausted',
                      completed_updates=result['completed_updates'], elapsed_seconds=result['elapsed_seconds'],
                      max_memory_reserved_fraction=result['max_memory_reserved_fraction'],
                      inner_trace=result['inner_trace'], counters=result['counters'], checkpoint=result['checkpoint'])
        if result['status'] == 'checkpointed':
            CheckpointStore(folder/'checkpoints').remove_completed_state()
            record['probe_checkpoint_removed'] = True
    atomic_json(folder/'probe_receipt.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--instance', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.device == 'cuda':
        torch.cuda.set_device(0)
        assert 'A100' in torch.cuda.get_device_name(), 'Gate requires target A100 hardware'
    fixture = args.out/'tiny_input'
    write_fixture(fixture)
    checks = tiny_checks(fixture, args.device)
    checks.append(tiny_restart(fixture, args.out, args.device))
    if args.instance:
        for kind in ('product', 'permutation-mixture'):
            checks.append(actual_probe(args.instance, args.out, args.device, kind))
    atomic_json(args.out/'checks.json', dict(status='passed', world_size=1, device=args.device,
        dtype='float64', checks=checks, source_file_sha256=runner.source_hashes(),
        validation_sha256=sha256(Path(__file__)), slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None,
        interpretation='Tiny parity/restart checks are required. Actual-size conditioning probes explicitly report completion, numerical-cap failure, or time-budget exhaustion; neither timeout nor failed trial is an accepted optimization result.'))


if __name__ == '__main__':
    main()
