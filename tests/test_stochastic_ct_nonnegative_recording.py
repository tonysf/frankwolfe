"""Recording, constraints and restart checks for the new CT runner."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from paper.experiments.stochastic_ct import nonnegative_run as runner


class TinyProblem:
    def __init__(self, config):
        self.size = config["grid"]
        self.detectors = 3
        self.train_ids = np.arange(config["n_train"])
        self.validation_ids = np.arange(config["n_train"], config["n_train"] + 2)
        generator = np.random.default_rng(12)
        self.operator = generator.uniform(0.01, 1.0, (config["n_train"] + 2, 3, self.size**2))
        target = generator.uniform(0.0, 1.0, (self.size, self.size))
        self.y = self.operator @ target.ravel()
        self.y[0, 0] = -0.1
        self.reference = target

    def forward(self, image, views):
        return self.operator[views] @ image.ravel()

    def backward(self, data, views):
        return np.einsum("vdi,vd->i", self.operator[views], data).reshape(self.size, self.size)

    def image_loss(self, image, views):
        return float(0.5 * np.mean((self.forward(image, views) - self.y[views])**2))

    def close(self):
        pass


def setup_run(root, method):
    root.mkdir(parents=True, exist_ok=True)
    data = root / "data.npz"
    np.savez(data, description=np.array("synthetic recording fixture"))
    U = np.random.default_rng(1).uniform(0.1, 0.3, (4, 2))
    V = np.random.default_rng(2).uniform(0.1, 0.3, (4, 2))
    calibration = root / "calibration.npz"
    np.savez(calibration, U=U, V=V,
        caps=np.array([2 * np.linalg.norm(U, 2), 2 * np.linalg.norm(V, 2)]),
        data_sha256=np.array(runner.sha256(data)),
        problem_sha256=np.array(runner.source_hashes()["ct_problem.py"]))
    runner.write(root / "protocol.json", {"numerical_sources": runner.source_hashes()})
    return dict(run_id=method, method=method, phase="development", grid=4, rank=2,
        n_train=8, slice_id="synthetic", calibration_seed=110, record_every_iteration=True,
        view_budget=8 * 7, data_file=str(data), calibration_file=str(calibration),
        data_sha256=runner.sha256(data), calibration_sha256=runner.sha256(calibration),
        parameters={} if method == "mu" else dict(memory=10, relative_kkt_tolerance=1e-3,
            ftol=1e-9, maxls=20, inner_maxiter=5))


@pytest.mark.parametrize("method", ["mu", "anls"])
def test_native_recording_and_checkpoint_recovery(tmp_path, method):
    config = setup_run(tmp_path, method)
    result = runner.run(config, root=tmp_path, problem_factory=TinyProblem)
    folder = tmp_path / "runs" / method
    trace = runner.read(folder / "trace.json")
    accepted = [row for row in trace if row["observation_type"] == "accepted_update"]
    assert [row["iteration"] for row in accepted] == list(range(1, len(accepted) + 1))
    assert result["final"] == trace[-1]
    assert max(trace[-1]["forward_views"], trace[-1]["backward_views"]) <= config["view_budget"]
    assert all(row["d_distance"] == 0 and row["raw_loss"] == row["feasible_loss"] for row in trace)
    assert runner.run(config, root=tmp_path, problem_factory=TinyProblem) == result
    # A crash after terminal checkpoint publication but before result.json must
    # not append a second fictitious terminal update when resumed.
    (folder / "result.json").unlink()
    assert runner.run(config, root=tmp_path, problem_factory=TinyProblem) == result
    assert runner.read(folder / "trace.json") == trace
    # The last checkpoint remains loadable at the receipt/rename crash boundary.
    receipt = runner.read(folder / "checkpoint.sha256.json")
    receipt["previous_sha256"] = receipt["sha256"]
    receipt["sha256"] = "0" * 64
    runner.write(folder / "checkpoint.sha256.json", receipt)
    (folder / "result.json").unlink()
    assert runner.run(config, root=tmp_path, problem_factory=TinyProblem) == result


def test_diagnostics_do_not_impose_old_spectral_caps():
    class State:
        U = np.ones((4, 2)) * 2
        V = np.ones((4, 2)) * 3
        counters = {"forward_views": 0, "backward_views": 0}
        iteration = completed_blocks = outer_sweeps = 0

    problem = TinyProblem(dict(grid=4, n_train=8))
    row = runner.observe(problem, State(), (0.01, 0.01), {"observation_type": "initial"})
    assert row["reference_spectral_ratio_u"] > 1
    assert row["reference_spectral_ratio_v"] > 1
    assert row["raw_loss"] == row["feasible_loss"] == problem.image_loss(State.U @ State.V.T, problem.train_ids)
    assert row["repair_distance"] == row["max_constraint_violation"] == 0


def test_changed_data_is_rejected_before_optimization(tmp_path):
    config = setup_run(tmp_path, "mu")
    Path(config["data_file"]).write_bytes(b"changed input")
    with pytest.raises(ValueError, match="Frozen input changed"):
        runner.run(config, root=tmp_path, problem_factory=TinyProblem)


@pytest.mark.parametrize("method", ["mu", "anls"])
def test_interrupted_observation_recovers_dense_accepted_state(tmp_path, method):
    config = setup_run(tmp_path, method)
    direct = runner.run(config, root=tmp_path, output=tmp_path / "direct", problem_factory=TinyProblem)

    class DiagnosticFailure(TinyProblem):
        calls = 0

        def image_loss(self, image, views):
            self.calls += 1
            if self.calls == 3:
                raise RuntimeError("synthetic diagnostic interruption")
            return super().image_loss(image, views)

    folder = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="synthetic diagnostic interruption"):
        runner.run(config, root=tmp_path, output=folder, problem_factory=DiagnosticFailure)
    resumed = runner.run(config, root=tmp_path, output=folder, problem_factory=TinyProblem)
    assert resumed == direct
    assert runner.read(folder / "trace.json") == runner.read(tmp_path / "direct/trace.json")


@pytest.mark.parametrize("method", ["mu", "anls"])
def test_interrupted_terminal_observation_retains_original_stop_event(tmp_path, method, monkeypatch):
    config = setup_run(tmp_path, method)
    direct = runner.run(config, root=tmp_path, output=tmp_path / "direct", problem_factory=TinyProblem)
    observe = runner.observe

    def failing_observe(problem, solver, caps, event):
        if event["observation_type"] in ("budget_stop", "converged"):
            raise RuntimeError("synthetic terminal diagnostic interruption")
        return observe(problem, solver, caps, event)

    monkeypatch.setattr(runner, "observe", failing_observe)
    folder = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="synthetic terminal diagnostic interruption"):
        runner.run(config, root=tmp_path, output=folder, problem_factory=TinyProblem)
    monkeypatch.setattr(runner, "observe", observe)
    resumed = runner.run(config, root=tmp_path, output=folder, problem_factory=TinyProblem)
    assert resumed == direct
    assert runner.read(folder / "trace.json") == runner.read(tmp_path / "direct/trace.json")
