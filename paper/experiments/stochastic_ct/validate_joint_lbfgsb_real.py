"""Bounded real-CT validation; these runs are not evaluation evidence."""

import argparse
from pathlib import Path

try:
    from . import joint_lbfgsb_run as runner
except ImportError:
    import joint_lbfgsb_run as runner

import numpy as np
from scipy.optimize import minimize


def validate(reference, output, cpu=0, pair_budget=6):
    reference, output = Path(reference).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = runner.source_hashes()
    runner.write(output / "protocol.json", dict(numerical_sources=sources,
        purpose="Bounded real-projector validation only; not scientific evaluation."))
    data = reference / "data/slice03150.npz"
    calibration = reference / "calibration/slice03150_seed110.npz"
    parameters = dict(memory=10, relative_kkt_tolerance=1e-3, ftol=1e-9, maxls=20)
    cases = []
    for n in (360, 1080):
        cfg = dict(run_id=f"real_smoke_joint_lbfgsb_{n}", phase="development",
            method="joint_lbfgsb", slice_id="slice03150", n_train=n, grid=512,
            rank=64, calibration_seed=110, record_every_iteration=True,
            view_budget=pair_budget*n, data_file=str(data), data_sha256=runner.sha256(data),
            calibration_file=str(calibration), calibration_sha256=runner.sha256(calibration),
            parameters=parameters)
        result = runner.run(cfg, cpu=cpu, root=output)
        folder = output / "runs" / cfg["run_id"]
        trace = runner.read(folder / "trace.json")
        accepted = [row for row in trace if row["observation_type"] == "accepted_update"]
        assert accepted and result["final"]["raw_loss"] < result["initial"]["raw_loss"]
        assert all(row["d_distance"] == 0 and row["raw_loss"] == row["feasible_loss"] for row in trace)
        assert all(row["forward_views"] == row["backward_views"] <= cfg["view_budget"] for row in trace)
        assert all(b["raw_loss"] <= a["raw_loss"] + 1e-7*max(1., a["raw_loss"])
                   for a, b in zip(trace, trace[1:]))
        assert runner.run(cfg, cpu=cpu, root=output) == result

        # Independently reproduce the first accepted joint step using SciPy's
        # public interface and a direct analytic CT objective/gradient.
        problem = runner.make_problem(cfg)
        try:
            with np.load(calibration, allow_pickle=False) as saved:
                U, V = saved["U"].copy(), saved["V"].copy()
            split, shape = U.size, U.shape
            x0 = np.concatenate((U.ravel(), V.ravel()))
            ids = problem.train_ids

            def objective(x):
                u, v = x[:split].reshape(shape), x[split:].reshape(shape)
                residual = problem.forward(u @ v.T, ids) - problem.y[ids]
                z = problem.backward(residual, ids).astype(np.float64) / residual.size
                return (float(0.5*np.mean(np.square(residual, dtype=np.float64))),
                        np.concatenate(((z @ v).ravel(), (z.T @ u).ravel())))

            initial_gradient = objective(x0)[1]
            target = parameters["relative_kkt_tolerance"] * np.max(np.abs(np.minimum(x0, initial_gradient)))
            first = minimize(objective, x0, jac=True, method="L-BFGS-B",
                bounds=[(0., None)]*x0.size,
                options=dict(maxiter=1, maxcor=parameters["memory"],
                    maxls=parameters["maxls"], ftol=parameters["ftol"], gtol=target))
            assert first.nit == 1
            assert accepted[0]["factor_u_sha256"] == runner.array_hash(first.x[:split].reshape(shape))
            assert accepted[0]["factor_v_sha256"] == runner.array_hash(first.x[split:].reshape(shape))
            assert accepted[0]["factor_u_sha256"] != runner.array_hash(U)
            assert accepted[0]["factor_v_sha256"] != runner.array_hash(V)
        finally:
            problem.close()
        cases.append(dict(run_id=cfg["run_id"], n_train=n,
            accepted_updates=len(accepted), first_step_matches_public_scipy=True,
            both_factors_updated=True, initial_loss=result["initial"]["raw_loss"],
            final_loss=result["final"]["raw_loss"],
            forward_views=result["final"]["forward_views"],
            backward_views=result["final"]["backward_views"],
            trace_sha256=runner.sha256(folder / "trace.json")))
    assert runner.source_hashes() == sources, "Numerical sources changed during validation"
    evidence = dict(status="passed", purpose="smoke validation, not scientific evaluation",
                    source_sha256=sources, cases=cases)
    runner.write(output / "real_engine_validation.json", evidence)
    return evidence


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
