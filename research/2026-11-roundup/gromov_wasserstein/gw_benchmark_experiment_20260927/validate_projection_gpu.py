"""Allocated CUDA projection checks; CPU references use the same stored inputs.

Run only in a one-A100 Slurm allocation. These are numerical/determinism checks,
not timing comparisons. Both smoothing row projection and Dykstra PG include all
fixed-order scan passes in their existing solver timers.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import traceback

import torch
from capture_environment import capture
from gw_core import simplex_rows


def projection_cases(n: int):
    """Independent CPU-generated float32 inputs; every row has mass 1/n."""
    generator = torch.Generator(device='cpu').manual_seed(51000 + n)
    uniform = torch.rand((n, n), dtype=torch.float32, generator=generator)
    normal = torch.randn((n, n), dtype=torch.float32, generator=generator)
    mass = torch.full((n,), 1/n, dtype=torch.float32)
    cases = {
        'positive_dense': (0.2 + 3.0 * uniform) / n**2,
        'negative_dense': -(0.2 + 3.0 * uniform) / n**2,
        'mixed_dense': 3.0 * normal / n**2,
        'gradient_scale_sparse': (-0.1 + 0.1 * normal) / n,
        'negative_tied': torch.full((n, n), -2/n**2, dtype=torch.float32),
        'positive_tied': torch.full((n, n), 2/n**2, dtype=torch.float32),
    }
    return [(label, value, mass) for label, value in cases.items()]


def compare_projection(label, values, mass, projected, repeated):
    """Evaluate actual float32 outputs in float64, without repairing them."""
    values64, mass64 = values.double(), mass.double()
    reference = simplex_rows(values64, mass64)
    actual = projected.detach().cpu().double()
    again = repeated.detach().cpu()
    error = (actual - reference).abs()
    row_error = (actual.sum(1) - mass64).abs()
    # Absolute scale reflects rounding in both the prefix and final subtraction.
    scale = max(values64.abs().max().item(), mass64.max().item())
    tolerance = 32 * torch.finfo(torch.float32).eps * scale
    row = {
        'case': label, 'n': len(mass), 'dtype': str(projected.dtype),
        'input_sha256': hashlib.sha256(values.numpy().tobytes()).hexdigest(),
        'mass_sha256': hashlib.sha256(mass.numpy().tobytes()).hexdigest(),
        'repeated_bitwise_equal': torch.equal(projected.detach().cpu(), again),
        'finite': bool(torch.isfinite(actual).all()),
        'min_entry': actual.min().item(),
        'row_l1_float64': row_error.sum().item(),
        'max_row_absolute_error_float64': row_error.max().item(),
        'max_absolute_error_vs_cpu_float64': error.max().item(),
        'frobenius_error_vs_cpu_float64': error.norm().item(),
        'max_absolute_error_tolerance': tolerance,
        'row_l1_tolerance': 2e-6,
        'cpu_float64_row_l1': (reference.sum(1) - mass64).abs().sum().item(),
    }
    row['passed'] = (row['repeated_bitwise_equal'] and row['finite']
                     and row['min_entry'] >= 0
                     and row['row_l1_float64'] <= row['row_l1_tolerance']
                     and row['max_absolute_error_vs_cpu_float64'] <= tolerance
                     and row['cpu_float64_row_l1'] < 1e-12)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    source = out / 'source_snapshot'
    source.mkdir()
    for name in ('validate_projection_gpu.py', 'gw_core.py', 'capture_environment.py'):
        shutil.copy2(Path(__file__).resolve().parent / name, source / name)
    report = {
        'status': 'started', 'device': 'cuda', 'sizes': [16, 128, 1000],
        'solver_dtype': 'float32', 'evaluation_dtype': 'float64',
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(source.iterdir())},
        'rows': [],
        'prefix_scan': 'Hillis-Steele inclusive; offsets 1,2,4,...; out-of-place; input precision',
        'timing_note': 'No performance comparison. All O(log n) CUDA scan passes are included in the existing projector/solver timers for every method using simplex_rows.',
        'reference_note': 'CPU float64 simplex projection of the identical stored float32 numbers; actual GPU outputs promoted for evaluation; no repair.',
    }
    try:
        report['environment'] = capture('cuda', 1)
        assert torch.are_deterministic_algorithms_enabled()
        with torch.no_grad():
            for n in report['sizes']:
                for label, values, mass in projection_cases(n):
                    cuda_values, cuda_mass = values.cuda(), mass.cuda()
                    projected = simplex_rows(cuda_values, cuda_mass)
                    repeated = simplex_rows(cuda_values, cuda_mass)
                    torch.cuda.synchronize()
                    row = compare_projection(label, values, mass, projected, repeated)
                    report['rows'].append(row)
                    assert row['passed'], row
        report['status'] = 'passed'
    except Exception:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        (out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f"{len(report['rows'])} CUDA projection cases passed with deterministic algorithms enabled")


if __name__ == '__main__':
    main()
