"""CPU checks of the fixed-order scan and independent projection optimality.

These exercise the CUDA arithmetic tree on CPU; actual GPU determinism is checked
separately by validate_projection_gpu.py inside an allocated A100 job.
"""
import numpy as np
import pytest
import torch
import gw_core
from validate_projection_gpu import projection_cases, compare_projection


@pytest.mark.parametrize('width', [1, 3, 16, 128, 1000])
def test_fixed_scan_inclusive_prefix_exact_integer_inputs_and_no_mutation(width):
    generator = torch.Generator().manual_seed(width)
    values = torch.randint(-8, 9, (5, width), generator=generator).float()
    old = values.clone()
    expected = values.double().cumsum(1).float()
    actual = gw_core._fixed_order_row_cumsum(values)
    assert torch.equal(actual, expected)
    assert torch.equal(values, old)
    assert actual.dtype == values.dtype and actual.device == values.device
    assert torch.equal(actual, gw_core._fixed_order_row_cumsum(values))


def _bisection_reference(values, mass):
    """Independent scalar KKT root solve, no sorting or cumulative sums."""
    z = values.double().numpy()
    a = mass.double().numpy()
    lower = z.min(1) - a
    upper = z.max(1)
    for _ in range(100):
        middle = (lower + upper) / 2
        above = np.maximum(z - middle[:, None], 0).sum(1) > a
        lower = np.where(above, middle, lower)
        upper = np.where(above, upper, middle)
    return torch.from_numpy(np.maximum(z - ((lower + upper) / 2)[:, None], 0))


@pytest.mark.parametrize('n', [16, 128, 1000])
def test_fixed_scan_projection_matches_float64_and_independent_kkt(n, monkeypatch):
    # Force the CUDA addition tree on CPU, while retaining the complete projector.
    for label, values, mass in projection_cases(n):
        expected = gw_core.simplex_rows(values.double(), mass.double())
        independent = _bisection_reference(values[:12], mass[:12])
        torch.testing.assert_close(expected[:12], independent, rtol=0, atol=1e-14)
        with monkeypatch.context() as patch:
            patch.setattr(gw_core, '_simplex_row_cumsum', gw_core._fixed_order_row_cumsum)
            actual = gw_core.simplex_rows(values, mass)
            repeated = gw_core.simplex_rows(values, mass)
        record = compare_projection(label, values, mass, actual, repeated)
        assert record['passed'], record


def test_cpu_projector_keeps_native_cumsum(monkeypatch):
    def reject(_):
        raise AssertionError('CPU must not route through CUDA scan')
    monkeypatch.setattr(gw_core, '_fixed_order_row_cumsum', reject)
    values = torch.tensor([[.5, -.3, .8], [-2., -2., -2.]], dtype=torch.float64)
    projected = gw_core.simplex_rows(values, torch.tensor([1., .3], dtype=torch.float64))
    torch.testing.assert_close(projected, torch.tensor([[.35, 0., .65], [.1, .1, .1]], dtype=torch.float64), rtol=0, atol=1e-14)
