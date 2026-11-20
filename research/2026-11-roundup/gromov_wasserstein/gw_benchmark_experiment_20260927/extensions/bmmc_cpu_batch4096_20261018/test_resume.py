"""Small CPU q=4096 checkpoint-continuity check using the production wrappers."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import tempfile
import time

# Set every numerical thread pool before importing either fixture or runner.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECEIPT = ROOT / "experiments/bmmc_cpu_batch4096_20261018/validation.json"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def run_check(receipt_path=DEFAULT_RECEIPT):
    started = time.monotonic()
    profile = load_module("bmmc_q4096_resume_profile", ROOT / "extensions/cpu_large_pilot_20261018/profile_runner.py")
    fixtures = load_module("bmmc_q4096_resume_fixtures", ROOT / "extensions/wallclock_replay_20261018/test_runner.py")
    runner = profile.runner
    runner.torch.set_num_threads(1)
    checks = {}

    def require(name, condition):
        checks[name] = bool(condition)
        if not condition:
            raise AssertionError(name)

    receipt = dict(
        status="running", started_at=datetime.now(timezone.utc).isoformat(),
        host=platform.node(), platform=platform.platform(),
        scope="3-by-3 fixture, CPU FP64, q4096; six uninterrupted updates versus two chunks of three",
        limitations="Checks checkpoint continuity and wrapper parity on a small fixture; does not establish full-size memory fit or final BMMC loss.",
        configuration=dict(method="frames", seed=811, pairs=4096, iterations=6,
                           gamma0=.3, rho0=1., mu_scale=10., num_threads=1),
        checks=checks,
    )
    receipt_path = Path(receipt_path).resolve()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="bmmc_q4096_resume_") as directory:
            temporary = Path(directory)
            fixtures.write_instance(temporary / "input")

            def arguments(output, chunk):
                args = fixtures.arguments(runner, temporary / "input", output, "frames", chunk=chunk)
                args.pairs, args.seed, args.rho0 = 4096, 811, 1.
                return args

            full, full_profile = profile.run_profiled(arguments(temporary / "full", 6))
            partial, partial_profile = profile.run_profiled(arguments(temporary / "resumed", 3))
            resumed, resumed_profile = profile.run_profiled(arguments(temporary / "resumed", 3))
            require("uninterrupted_completed_six", full["status"] == "completed" and full["completed_updates"] == 6)
            require("first_chunk_checkpointed_three", partial["status"] == "checkpointed" and partial["completed_updates"] == 3)
            require("resumed_completed_six", resumed["status"] == "completed" and resumed["completed_updates"] == 6)
            require("profile_launch_update_counts_6_3_3", [s["completed_updates_in_launch"] for s in (full_profile, partial_profile, resumed_profile)] == [6, 3, 3])
            require("one_thread_each_launch", all(s["num_threads"] == 1 for s in (full_profile, partial_profile, resumed_profile)))
            require("immutable_identity_equal", full["identity"] == partial["identity"] == resumed["identity"])
            require("counter_values_equal", full["counters"] == resumed["counters"])
            require("six_fresh_oracle_calls", resumed["counters"]["stochastic_gradient_calls"] == 6)
            require("pair_count_24576", resumed["counters"]["pair_samples"] == 6 * 4096)
            require("trace_equal_excluding_elapsed_seconds", canonical(fixtures.without_timing(full["trace"])) == canonical(fixtures.without_timing(resumed["trace"])))
            require("final_loss_bitwise_equal", canonical(full["final_loss"]) == canonical(resumed["final_loss"]))
            require("resume_uses_durable_post_checkpoint_clock", resumed["timing"]["resume_elapsed_source"] == "matching_post_checkpoint_progress" and resumed["timing"]["elapsed_prior_seconds"] == partial["elapsed_seconds"])

            expected, _ = runner.CheckpointStore(temporary / "full/checkpoints").load(identity=full["identity"])
            actual, _ = runner.CheckpointStore(temporary / "resumed/checkpoints").load(identity=resumed["identity"])
            state_hashes = {}
            for name in ("plan", "momentum", "generator_state", "torch_rng_state", "cuda_rng_state"):
                left, right = expected[name], actual[name]
                if left is None:
                    require(name + "_both_none", right is None)
                    state_hashes[name] = None
                else:
                    left_bytes = left.detach().cpu().contiguous().numpy().tobytes()
                    right_bytes = right.detach().cpu().contiguous().numpy().tobytes()
                    require(name + "_bitwise_equal", left.dtype == right.dtype and left.shape == right.shape and left_bytes == right_bytes)
                    state_hashes[name] = dict(sha256=digest_bytes(left_bytes), shape=list(left.shape), dtype=str(left.dtype))
            require("checkpoint_counters_equal", expected["counters"] == actual["counters"])
            require("checkpoint_iterations_equal_six", expected["iteration"] == actual["iteration"] == 6)
            # The durable state precedes receipt publication; its cached
            # completed_updates counter can refer to the previous receipt.
            # Compare equivalent artifacts and verify the actual state index.
            require("checkpoint_work_counters_match_receipt", all(value == full["counters"][name] for name, value in actual["counters"].items() if name != "completed_updates"))
            require("checkpoint_trace_equal_excluding_elapsed_seconds", canonical(fixtures.without_timing(expected["trace"])) == canonical(fixtures.without_timing(actual["trace"])))
            receipt.update(
                status="passed", final_loss=full["final_loss"], counters=full["counters"],
                checkpoint_counters=actual["counters"], checkpoint_iteration=actual["iteration"],
                trace_without_timing_sha256=digest_bytes(canonical(fixtures.without_timing(full["trace"]))),
                tensor_state_hashes=state_hashes, source_file_sha256=full["source_file_sha256"],
                wrapper_sha256=full_profile["wrapper_sha256"],
                fixture_sha256=runner.sha256(ROOT / "extensions/wallclock_replay_20261018/test_runner.py"),
                test_sha256=runner.sha256(__file__),
                torch_version=full["identity"]["torch_version"], numpy_version=full["identity"]["numpy_version"],
                fixture_array_files_sha256=full["array_files_sha256"],
            )
    except BaseException as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt.update(finished_at=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
        runner.atomic_json(receipt_path, receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    result = run_check(parser.parse_args().receipt)
    print(json.dumps(dict(status=result["status"], checks=len(result["checks"]), final_loss=result["final_loss"]), sort_keys=True))
