"""Fixed-input operator diagnostics stay independent and bounded on CPU JAX."""

import hashlib
import json

import numpy as np
import pytest

from paper.experiments import qpt_reproducibility_probes as probes
from paper.experiments.qpt_structured_data import standard_local_basis, standard_local_measurements
from paper.experiments.qpt_structured_operators import rank_one_measurement_vectors


@pytest.fixture(scope="module")
def cpu_jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    return jax, jnp, jax.devices("cpu")[0]


def inputs(n=2, rank=1):
    rng = np.random.default_rng(711)
    factor = (rng.normal(size=(4**n, rank)) + 1j * rng.normal(size=(4**n, rank))) / 4
    return {
        "factor": factor,
        "previous": (rng.normal(size=factor.shape) + 1j * rng.normal(size=factor.shape)) / 7,
        "symbols": rng.integers(0, 24, size=(5, n)),
        "observations": rng.normal(size=5),
        "local_measurements": standard_local_measurements(),
        "local_basis": standard_local_basis(),
        "rho": 0.35, "beta": 2.3, "gamma": 0.2, "tau": 10,
        "measurement_backend": "tensor", "repeats": 3,
    }


@pytest.mark.parametrize("backend", ["tensor", "rank-one"])
@pytest.mark.parametrize("n,rank", [(1, 1), (2, 1), (2, 2)])
@pytest.mark.parametrize("iteration", [0, 17])
def test_fixed_input_probes_match_independent_dense_reference(cpu_jax, backend, n, rank, iteration):
    options = inputs(n=n, rank=rank)
    options.update(measurement_backend=backend, iteration=iteration)
    snapshot = {name: value.copy() for name, value in options.items() if isinstance(value, np.ndarray)}
    result = probes.run_operator_probes(*cpu_jax, **options)
    assert result["cpu_reference"]["status"] == "validated"
    assert result["cpu_reference"]["validated"] is True
    assert result["replay_exact"] is True
    assert result["replay_allclose"] is True
    assert result["all_outputs_finite"] is True
    assert result["compiled_executables"] == 1
    assert len(result["runs"]) == 3
    assert result["iteration"] == iteration
    assert result["measurement_backend"] == backend
    for row in result["runs"]:
        assert set(row["versus_cpu"]) == {
            "measurement_loss", "measurement_gradient", "tp_loss", "tp_gradient",
            "moreau_gradient", "momentum_gradient", "combined_gradient", "lmo_atom",
            "next_factor", "estimated_gap",
        }
        assert all(item["allclose"] and item["finite"] for item in row["versus_cpu"].values())
        if iteration == 0:
            assert row["outputs"]["momentum_gradient"]["sha256"] == row["outputs"]["measurement_gradient"]["sha256"]
    for name, value in snapshot.items():
        np.testing.assert_array_equal(options[name], value)
    json.dumps(result, allow_nan=False)


def test_probe_constructs_one_jit_and_replays_it(cpu_jax, monkeypatch):
    jax, jnp, device = cpu_jax
    original_jit = jax.jit
    constructions = []

    def counted(function):
        constructions.append(function)
        return original_jit(function)

    monkeypatch.setattr(jax, "jit", counted)
    probes.run_operator_probes(jax, jnp, device, **inputs(n=1))
    assert len(constructions) == 1


def test_reference_cap_skips_all_dense_construction_without_claiming_validation(cpu_jax, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Dense construction must not be attempted over the cap")

    monkeypatch.setattr(probes, "_dense_reference", forbidden)
    result = probes.run_operator_probes(*cpu_jax, **inputs(), max_reference_bytes=1)
    assert result["cpu_reference"]["status"] == "skipped"
    assert result["cpu_reference"]["validated"] is False
    assert result["replay_exact"] is True
    assert all(row["versus_cpu"] is None for row in result["runs"])
    assert "exceeds byte cap" in result["cpu_reference"]["reason"]
    json.dumps(result, allow_nan=False)


def test_reference_estimate_allows_current_three_qubit_case_and_bounds_larger_case():
    assert probes._reference_allocation_estimate(4**3, 1, 32) < 128 * 1024**2
    assert probes._reference_allocation_estimate(4**4, 1, 32) > 128 * 1024**2


def test_actual_supplied_bank_is_used_and_hashed_without_eigen_reconstruction(cpu_jax, monkeypatch):
    options = inputs()
    vectors = rank_one_measurement_vectors(options["local_measurements"])
    # Deliberate corruption is compared against the independent original
    # matrix bank, not silently reconstructed/corrected by the diagnostic.
    supplied = vectors * 1.01
    options.update(measurement_backend="rank-one", selected_bank=supplied)

    def forbidden(*args, **kwargs):
        raise AssertionError("Do not recompute a caller-supplied bank")

    monkeypatch.setattr(probes, "rank_one_measurement_vectors", forbidden)
    result = probes.run_operator_probes(*cpu_jax, **options)
    assert result["selected_bank_sha256"] == hashlib.sha256(supplied.tobytes()).hexdigest()
    assert result["derived_rank_one_bank_sha256"] is None
    assert result["selected_bank_source"] == "supplied by caller"
    assert result["replay_exact"] is True
    assert result["cpu_reference"]["status"] == "mismatch"
    assert result["cpu_reference"]["validated"] is False
    assert result["runs"][0]["versus_cpu"]["measurement_gradient"]["allclose"] is False
    assert result["runs"][0]["versus_cpu"]["tp_gradient"]["allclose"] is True


def test_nonfinite_device_results_are_strict_json_and_never_pass(cpu_jax, monkeypatch):
    original = probes.measurement_loss_and_gradient

    def nonfinite(*args, **kwargs):
        loss, gradient = original(*args, **kwargs)
        return loss, gradient * np.nan

    monkeypatch.setattr(probes, "measurement_loss_and_gradient", nonfinite)
    result = probes.run_operator_probes(*cpu_jax, **inputs(n=1))
    assert result["cpu_reference"]["status"] == "mismatch"
    assert result["replay_exact"] is False
    assert result["replay_allclose"] is False
    assert result["all_outputs_finite"] is False
    comparison = result["runs"][0]["versus_cpu"]["measurement_gradient"]
    assert comparison["max_abs_difference"] is None
    assert comparison["relative_l2_difference"] is None
    json.dumps(result, allow_nan=False)


def test_zero_reference_relative_error_is_not_infinity_or_false_zero():
    result = probes._compare(np.array([1.0]), np.array([0.0]), rtol=1e-10, atol=1e-12)
    assert result["reference_is_zero"] is True
    assert result["relative_l2_difference"] is None
    assert result["max_abs_difference"] == 1.0
    assert result["allclose"] is False
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("key,value", [
    ("observations", None), ("observations", np.ones(4)),
    ("symbols", np.ones((5, 2)) * 24), ("previous", np.ones((4, 1))),
    ("rho", -1), ("beta", 0), ("gamma", 2), ("tau", -1),
    ("repeats", 1), ("iteration", -1), ("max_reference_bytes", -1),
    ("local_basis", np.zeros((4, 4, 4))), ("measurement_backend", "invalid"),
])
def test_invalid_inputs_fail_explicitly(cpu_jax, key, value):
    options = inputs()
    options[key] = value
    with pytest.raises((ValueError, TypeError)):
        probes.run_operator_probes(*cpu_jax, **options)


def test_single_precision_uses_actual_cast_inputs_and_explicit_tolerances(cpu_jax):
    options = inputs()
    options.update(factor=options["factor"].astype(np.complex64), measurement_backend="rank-one")
    result = probes.run_operator_probes(*cpu_jax, **options, rtol=2e-5, atol=2e-6)
    assert result["inputs"]["selected_bank"]["dtype"] == "complex64"
    assert result["inputs"]["observations"]["dtype"] == "float32"
    assert result["cpu_reference"]["status"] == "validated"
    assert result["rtol"] == 2e-5
    json.dumps(result, allow_nan=False)
