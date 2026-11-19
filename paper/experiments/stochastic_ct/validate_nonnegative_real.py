"""Bounded real-projector checks before freezing a nonnegative CT campaign."""

import argparse
from pathlib import Path

import numpy as np

try:
    from . import nonnegative_run as runner
except ImportError:
    import nonnegative_run as runner


def validate(reference, output, cpu=0, pair_budget=6):
    reference, output = Path(reference).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    runner.write(output / "protocol.json", dict(numerical_sources=runner.source_hashes(),
        purpose="bounded real-projector smoke only; not evaluation evidence"))
    data = reference / "data/slice03150.npz"
    calibration = reference / "calibration/slice03150_seed110.npz"
    source = runner.source_hashes()
    records = []
    for n in (360, 1080):
        for method in ("mu", "anls"):
            config = dict(run_id=f"real_smoke_{method}_{n}", phase="development", method=method,
                slice_id="slice03150", n_train=n, grid=512, rank=64, calibration_seed=110,
                record_every_iteration=True, view_budget=pair_budget*n,
                data_file=str(data), data_sha256=runner.sha256(data),
                calibration_file=str(calibration), calibration_sha256=runner.sha256(calibration),
                parameters={} if method == "mu" else dict(memory=10, relative_kkt_tolerance=1e-3,
                    ftol=1e-9, maxls=20, inner_maxiter=5))
            result = runner.run(config, cpu=cpu, root=output)
            folder = output / "runs" / config["run_id"]
            trace = runner.read(folder / "trace.json")
            accepted = [row for row in trace if row["observation_type"] == "accepted_update"]
            if not accepted:
                raise AssertionError("No accepted real-projector update")
            if (max(result["final"]["forward_views"], result["final"]["backward_views"]) > config["view_budget"]
                    or any(row["d_distance"] != 0 for row in trace)
                    or result["final"]["raw_loss"] > result["initial"]["raw_loss"]):
                raise AssertionError("Invalid real-projector recording or budget")
            # Allow a float32 projector's roundoff, not algorithmic loss growth.
            if any(b["raw_loss"] > a["raw_loss"] + 1e-7 * max(1.0, a["raw_loss"])
                   for a, b in zip(trace, trace[1:])):
                raise AssertionError("Unexpected nonmonotone NNLS/MU step")
            if runner.run(config, cpu=cpu, root=output) != result:
                raise AssertionError("Completed replay is not idempotent")
            if method == "mu":
                problem = runner.make_problem(config)
                try:
                    with np.load(calibration, allow_pickle=False) as saved:
                        U, V = saved["U"], saved["V"]
                    ids = problem.train_ids
                    scale = len(ids) * problem.detectors
                    rhs = problem.backward(problem.y[ids], ids).astype(np.float64) / scale
                    normal = problem.backward(problem.forward(U @ V.T, ids), ids).astype(np.float64) / scale
                    b, q = rhs @ V, normal @ V
                    denominator = q + np.maximum(-b, 0)
                    ratio = np.ones_like(U)
                    np.divide(np.maximum(b, 0), denominator, out=ratio, where=denominator > 0)
                    expected = U * ratio
                    if runner.array_hash(expected) != accepted[0]["factor_u_sha256"]:
                        raise AssertionError("First real MU update differs from independent quadratic formula")
                    if runner.array_hash(V) != accepted[0]["factor_v_sha256"]:
                        raise AssertionError("MU updated the wrong block")
                finally:
                    problem.close()
            records.append(dict(run_id=config["run_id"], method=method, n_train=n,
                accepted_updates=len(accepted), initial_loss=result["initial"]["raw_loss"],
                final_loss=result["final"]["raw_loss"],
                forward_views=result["final"]["forward_views"],
                backward_views=result["final"]["backward_views"],
                source_sha256=result["metadata"]["source_sha256"],
                trace_sha256=runner.sha256(folder / "trace.json"),
                independent_first_mu_update=(method == "mu")))
    if source != runner.source_hashes():
        raise AssertionError("Source changed while validating")
    result = dict(status="passed", purpose="smoke validation, not scientific evaluation",
        source_sha256=source, cases=records)
    runner.write(output / "real_engine_validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=0)
    parser.add_argument("--pair-budget", type=int, default=6)
    args = parser.parse_args()
    print(validate(args.reference_root, args.output, args.cpu, args.pair_budget))


if __name__ == "__main__":
    main()
