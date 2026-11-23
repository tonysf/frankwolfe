"""Faithful elapsed-budget instrumentation of the installed POT Sampled GW.

The installed native function remains the estimator: runtime source is inspected,
strictly matched to reviewed ASTs, copied, archived, and wrapped at its outer loop
boundary. No native expression, initialization, cost normalization, sample count,
RNG draw, inner Sinkhorn setting, or stopping rule is replaced. An optional elapsed
solver-time budget adds a stop only after a completed native iteration attempt.

Public API returns (native_return_value, instrumentation_report). Its callback is
callback(snapshot_T, info), where info is a dictionary. Callbacks receive a copy,
not the live native plan; snapshot creation and callback work are excluded from
solver time and explicitly reported. NumPy sampling and any device transfers made
by POT remain solver work. CUDA execution must be validated on the target host.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Callable

import numpy as np
import ot

# Reviewed exact wheel sources, excluding only the native docstring and AST
# location metadata. 0.9.7 adds sparse-row handling; dense native math is retained.
REVIEWED_AST_SHA256 = {
    "0.9.6.post1": "31347198bec93adf8db71d8199f36b3ad3405dd59777a83a3e245fee85fa8885",
    "0.9.7.post1": "2522a083804d9676f51a5ceb6fdace47ec29646becc8dfb46c3541e846134d43",
}
ARGUMENT_NAMES = ["C1", "C2", "p", "q", "loss_fun", "nb_samples_grad", "epsilon", "max_iter", "log", "verbose", "random_state"]


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _semantic_digest(module: ast.Module) -> str:
    cleaned = copy.deepcopy(module)
    function = cleaned.body[0]
    if function.body and isinstance(function.body[0], ast.Expr) and isinstance(function.body[0].value, ast.Constant) and isinstance(function.body[0].value.value, str):
        function.body.pop(0)
    return _digest(ast.dump(cleaned, include_attributes=False).encode())


def _rng_digest(generator) -> str:
    state = generator.get_state()
    return _digest(state[0].encode() + state[1].tobytes() + repr(state[2:]).encode())


def _synchronize(value) -> None:
    if getattr(value, "is_cuda", False):
        import torch
        torch.cuda.synchronize(value.device)


def _snapshot(value, backend):
    snapshot = backend.copy(value)
    if hasattr(snapshot, "detach"):
        snapshot = snapshot.detach()
    elif isinstance(snapshot, np.ndarray):
        snapshot.setflags(write=False)
    return snapshot


class _Monitor:
    def __init__(self, callback, time_budget, callback_initial):
        self.callback = callback
        self.time_budget = time_budget
        self.callback_initial = callback_initial
        self.solver_seconds = 0.0
        self.callback_seconds = 0.0
        self.callback_calls = 0
        self.completed_iterations = 0
        self.completed_updates = 0
        self.budget_reached = False
        self.previous_plan = None
        self.generator = None
        self.backend = None
        self.run_started = None
        self.segment_started = None
        self.initial_rng_digest = None

    def start(self, reference):
        _synchronize(reference)
        self.run_started = self.segment_started = time.perf_counter()

    def enter_loop(self, T, generator, backend):
        # Called after the unmodified native product initialization and symmetry
        # checks; their elapsed time stays in the first solver segment.
        self.previous_plan, self.generator, self.backend = T, generator, backend
        self.initial_rng_digest = _rng_digest(generator)
        if self.callback_initial and self.callback is not None:
            self._checkpoint(T, initial=True)

    def _checkpoint(self, T, initial=False):
        _synchronize(T)
        now = time.perf_counter()
        self.solver_seconds += now - self.segment_started
        # Do not stop before the first native iteration: the budget is checked
        # at native iteration boundaries, including no-op and native-break paths.
        if not initial and self.time_budget is not None:
            self.budget_reached = self.solver_seconds >= self.time_budget
        if self.callback is not None:
            diagnostic_started = time.perf_counter()
            before_rng = _rng_digest(self.generator)
            snapshot = _snapshot(T, self.backend)
            info = {
                "completed_iterations": self.completed_iterations,
                "native_iteration_index": self.completed_iterations - 1 if not initial else None,
                "completed_updates": self.completed_updates,
                "solver_seconds": self.solver_seconds,
                "updated_transport": self.updated_transport if not initial else False,
                "is_initial": initial,
                "time_budget_reached": self.budget_reached,
                "callback_seconds_cumulative_before": self.callback_seconds,
                "instrumented_elapsed_seconds": now - self.run_started,
            }
            self.callback(snapshot, info)
            _synchronize(snapshot)
            if _rng_digest(self.generator) != before_rng:
                raise RuntimeError("Read-only callback changed the native sampler RNG state; run invalid.")
            self.callback_calls += 1
            self.callback_seconds += time.perf_counter() - diagnostic_started
        self.segment_started = time.perf_counter()
        return self.budget_reached

    def iteration_finished(self, T, completed_iterations, exception_type):
        # A finally hook must never suppress a native exception with a budget
        # break. Native continue/break are clean control flow, not exceptions.
        if exception_type is not None:
            return False
        self.completed_iterations = completed_iterations
        self.updated_transport = T is not self.previous_plan
        self.completed_updates += int(self.updated_transport)
        self.previous_plan = T
        return self._checkpoint(T)

    def finish(self, reference):
        _synchronize(reference)
        self.solver_seconds += time.perf_counter() - self.segment_started

    def report(self, max_iter):
        return {
            "solver_seconds": self.solver_seconds,
            "callback_seconds": self.callback_seconds,
            "callback_calls": self.callback_calls,
            "completed_iterations": self.completed_iterations,
            "completed_updates": self.completed_updates,
            "time_budget_seconds": self.time_budget,
            "time_budget_reached": self.budget_reached,
            "max_iter_exhausted": self.completed_iterations >= max_iter,
            "native_early_stop": self.completed_iterations < max_iter and not self.budget_reached,
            "initial_rng_state_sha256": self.initial_rng_digest,
            "final_rng_state_sha256": _rng_digest(self.generator) if self.generator is not None else None,
            "inner_sinkhorn_iterations": None,
            "inner_sinkhorn_stopping_failures": None,
            "inner_counts_note": "Native default Sinkhorn calls are unchanged and do not expose these counts.",
            "iteration_note": "completed_iterations counts native outer-loop attempts, including zero-cost skips; completed_updates counts native T=copy(new_T) assignments.",
            "timing_note": "Synchronized native setup/initialization, sampling, CPU/GPU transfers, cost normalization, inner solves, and post-loop native work are included. Snapshot copying and callback evaluation are excluded and reported separately. Budget checks occur only after native outer-loop attempts; no restart or backdating.",
        }


def _instrument_runtime(monitor: _Monitor, archive_dir: Path):
    native = ot.gromov.sampled_gromov_wasserstein
    version = ot.__version__
    if version not in REVIEWED_AST_SHA256:
        raise RuntimeError(f"POT {version} has not been reviewed; refusing native source instrumentation.")
    source = inspect.getsource(native)
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise RuntimeError("Native source is not the reviewed single function.")
    function = tree.body[0]
    if [a.arg for a in function.args.args] != ARGUMENT_NAMES or function.decorator_list:
        raise RuntimeError("Native signature/decorators differ from the reviewed function.")
    fingerprint = _semantic_digest(tree)
    if fingerprint != REVIEWED_AST_SHA256[version]:
        raise RuntimeError(f"POT {version} native AST hash {fingerprint} differs from reviewed source; refusing instrumentation.")
    if any(isinstance(node, ast.Name) and node.id.startswith("_sgw_") for node in ast.walk(tree)):
        raise RuntimeError("Native function collides with reserved instrumentation names.")
    candidates = [(i, node) for i, node in enumerate(function.body)
                  if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "cpt"]
    if len(candidates) != 1:
        raise RuntimeError("Expected precisely one top-level native cpt loop.")
    loop_index, loop = candidates[0]
    if ast.dump(loop.iter, include_attributes=False) != ast.dump(ast.parse("range(max_iter)", mode="eval").body, include_attributes=False) or loop.orelse:
        raise RuntimeError("Native loop bound or loop-else is unsupported.")
    original_body = loop.body
    hook = ast.parse("if _sgw_monitor.iteration_finished(T, cpt + 1, _sgw_sys.exc_info()[0]):\n    break\n").body
    loop.body = [ast.Try(body=original_body, handlers=[], orelse=[], finalbody=hook)]
    function.body.insert(loop_index, ast.parse("_sgw_monitor.enter_loop(T, generator, nx)").body[0])
    ast.fix_missing_locations(tree)
    # Prove the transformation changes only the two declared instrumentation
    # insertions: reversing them must recover the reviewed full semantic AST.
    recovered = copy.deepcopy(tree)
    recovered_function = recovered.body[0]
    recovered_function.body.pop(loop_index)
    recovered_loop = recovered_function.body[loop_index]
    recovered_loop.body = recovered_loop.body[0].body
    if _semantic_digest(recovered) != fingerprint:
        raise RuntimeError("AST instrumentation changed native expressions unexpectedly.")
    generated = ("# Runtime-installed POT function; only guarded outer-loop monitoring added.\n"
                 "# Executed with the installed function's globals plus _sgw_monitor/_sgw_sys.\n" + ast.unparse(tree) + "\n")
    archive_dir = Path(archive_dir).resolve()
    if archive_dir.exists() and any(archive_dir.iterdir()):
        raise ValueError(f"Native source archive must be new or empty: {archive_dir}")
    archive_dir.mkdir(parents=True, exist_ok=True)
    module_path = Path(inspect.getsourcefile(native))
    source_files = {
        "native_function.py": source.encode(),
        "native_module.py": module_path.read_bytes(),
        "instrumented_function.py": generated.encode(),
        "instrumentation_module.py": Path(__file__).read_bytes(),
    }
    for name, content in source_files.items():
        (archive_dir / name).write_bytes(content)
    manifest = {
        "pot_version": version, "native_function": native.__module__ + "." + native.__name__,
        "native_signature": str(inspect.signature(native)), "native_module_path": str(module_path),
        "native_semantic_ast_sha256": fingerprint,
        "archive_sha256": {name: _digest(content) for name, content in source_files.items()},
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "transformation": "Insert monitor before native cpt loop; wrap the original loop body in try/finally with a read-only monitor and optional post-iteration budget break. Reversing these insertions recovers the reviewed AST exactly.",
        "unchanged": ["product initialization", "native NumPy RandomState sampling", "conditional row sampling", "native sample-count interpretation", "native cost normalization", "inner Sinkhorn defaults", "native convergence and warning breaks", "native final log estimator"],
        "backend_note": "Native POT backend dispatch is retained. Torch costs/Sinkhorn use the tensor device, but native RandomState sampling uses explicit to_numpy CPU transfers. This is a hybrid implementation, not an all-GPU port.",
    }
    (archive_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    namespace = dict(native.__globals__)
    namespace.update(_sgw_monitor=monitor, _sgw_sys=sys)
    # Compile exactly the already archived text, not another generated variant.
    exec(compile(generated, str(archive_dir / "instrumented_function.py"), "exec"), namespace)
    return namespace[native.__name__], manifest


def sampled_gromov_wasserstein(
    C1, C2, p, q, loss_fun, nb_samples_grad=100, epsilon=1, max_iter=500,
    log=False, verbose=False, random_state=None, *, callback: Callable | None = None,
    time_budget: float | None = None, archive_dir, callback_initial=False,
):
    """Run a faithfully instrumented installed native POT function exactly once.

    A callback receives (snapshot_T, info), with info['solver_seconds'] and native
    iteration/update counts. The return is (the_native_return_value, report).
    A mandatory fresh archive directory preserves installed and generated source
    before any native iteration executes. Unsupported source fails closed.
    """
    if time_budget is not None and (not math.isfinite(time_budget) or time_budget <= 0):
        raise ValueError("time_budget must be finite and strictly positive, or None.")
    if callback is not None and not callable(callback):
        raise TypeError("callback must be callable or None.")
    preparation_started = time.perf_counter()
    monitor = _Monitor(callback, time_budget, callback_initial)
    instrumented, provenance = _instrument_runtime(monitor, Path(archive_dir))
    preparation_seconds = time.perf_counter() - preparation_started
    monitor.start(C1)
    try:
        native_result = instrumented(C1, C2, p, q, loss_fun,
                                     nb_samples_grad=nb_samples_grad, epsilon=epsilon,
                                     max_iter=max_iter, log=log, verbose=verbose,
                                     random_state=random_state)
        plan = native_result[0] if log else native_result
        monitor.finish(plan)
    except BaseException:
        failure = {"status": "failed", "completed_iterations": monitor.completed_iterations,
                   "completed_updates": monitor.completed_updates, "traceback": traceback.format_exc()}
        (Path(archive_dir) / "execution_failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
    report = monitor.report(max_iter)
    report.update(status="completed", provenance=provenance,
                  archive_dir=str(Path(archive_dir).resolve()),
                  instrumentation_preparation_seconds=preparation_seconds,
                  total_native_call_instrumented_seconds=time.perf_counter()-monitor.run_started,
                  cuda_execution=bool(getattr(plan, "is_cuda", False)))
    (Path(archive_dir) / "execution_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return native_result, report


def _self_test(out: Path):
    """Bounded CPU native-vs-instrumented equality and random-state checks."""
    import torch
    torch.set_num_threads(1)
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(867)
    A = rng.uniform(0.5, 1.0, (8, 8))
    A = (A + A.T) / 2
    np.fill_diagonal(A, 0)
    B = rng.uniform(0.5, 1.0, (8, 8))
    B = (B + B.T) / 2
    np.fill_diagonal(B, 0)
    uniform = np.full(8, 1 / 8)
    checks = []
    for backend in ("numpy", "torch_cpu"):
        for use_log in (False, True):
            case = f"{backend}_log{int(use_log)}"
            arrays = [A, B, uniform, uniform]
            if backend == "torch_cpu":
                arrays = [torch.from_numpy(x.copy()) for x in arrays]
            native_rng = np.random.RandomState(73)
            native = ot.gromov.sampled_gromov_wasserstein(*arrays, lambda x,y: (x-y)**2,
                        nb_samples_grad=4, epsilon=0.1, max_iter=7, log=use_log, random_state=native_rng)
            for callback_mode in ("disabled", "record_only"):
                observations = []
                def callback(T, info):
                    if isinstance(T, np.ndarray):
                        assert not T.flags.writeable
                    observations.append((info.copy(), float(T.sum())))
                local_rng = np.random.RandomState(73)
                result, report = sampled_gromov_wasserstein(*arrays, lambda x,y: (x-y)**2,
                    nb_samples_grad=4, epsilon=0.1, max_iter=7, log=use_log, random_state=local_rng,
                    callback=callback if callback_mode == "record_only" else None,
                    archive_dir=out / f"{case}_{callback_mode}")
                expected_plan = native[0] if use_log else native
                actual_plan = result[0] if use_log else result
                if backend == "torch_cpu":
                    assert torch.equal(actual_plan, expected_plan)
                else:
                    assert np.array_equal(actual_plan, expected_plan)
                assert _rng_digest(native_rng) == _rng_digest(local_rng)
                if use_log:
                    assert native[1] == result[1]
                assert report["completed_iterations"] == report["completed_updates"] == 7
                if callback_mode == "record_only":
                    assert [r[0]["completed_iterations"] for r in observations] == list(range(1, 8))
                    assert all(abs(r[1]-1) < 1e-12 for r in observations)
                checks.append({"case": case, "callback": callback_mode, "exact_coupling_match": True,
                               "exact_rng_state_match": True, "native_log_match": True if use_log else None})
    # Zero sampled cost takes the native continue path: preserve its RNG draws,
    # publish no-op attempts, and ensure the budget stops without restarting.
    zero = np.zeros((8, 8))
    attempts = []
    _, budget_report = sampled_gromov_wasserstein(zero, zero, uniform, uniform, lambda x,y:(x-y)**2,
        nb_samples_grad=4, epsilon=.1, max_iter=100000, random_state=np.random.RandomState(5),
        callback=lambda T, info: attempts.append(info), time_budget=0.000001,
        archive_dir=out / "zero_cost_budget")
    assert budget_report["time_budget_reached"] and budget_report["completed_iterations"] == 1
    assert budget_report["completed_updates"] == 0 and len(attempts) == 1
    # A native convergence break commits new_T before leaving; it must still be
    # exposed once, with exactly the native result and final sampling state.
    loss_constant = lambda x,y: np.ones(np.broadcast_shapes(x.shape,y.shape))
    native_rng = np.random.RandomState(17)
    native = ot.gromov.sampled_gromov_wasserstein(A,B,uniform,uniform,loss_constant,
        nb_samples_grad=4,epsilon=.1,max_iter=150,random_state=native_rng)
    local_rng = np.random.RandomState(17)
    result, stable_report = sampled_gromov_wasserstein(A,B,uniform,uniform,loss_constant,
        nb_samples_grad=4,epsilon=.1,max_iter=150,random_state=local_rng,
        archive_dir=out / "native_convergence_break")
    assert np.array_equal(native,result) and _rng_digest(native_rng)==_rng_digest(local_rng)
    assert stable_report["native_early_stop"] and stable_report["completed_iterations"]==101
    summary = {"status":"passed", "pot_version":ot.__version__, "cpu_only":True,
               "cuda_execution_validated":False, "equality_cases":checks,
               "zero_cost_continue_and_budget":"passed", "native_convergence_break":"passed",
               "reviewed_runtime_versions":list(REVIEWED_AST_SHA256),
               "note":"Exact coupling and RandomState equality checked against local installed native POT; CUDA and remote 0.9.6.post1 execution remain separate validations."}
    (out/"validation.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true",required=True)
    parser.add_argument("--out",type=Path,required=True)
    options=parser.parse_args()
    _self_test(options.out)
