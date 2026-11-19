"""Small actual-projector checks for native continuation and snapshot accounting."""

from campaign import development_jobs, initial_grids
from run import ROOT, method_config, read_json, run, sha256, write_json

import argparse
import json
import pickle
import time
import numpy as np


def equal(a, b):
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal(a[key], b[key]) for key in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(left, right) for left, right in zip(a, b))
    return a == b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, default=4)
    args = parser.parse_args()
    root = ROOT / "smoke_runs"
    tag = sha256(ROOT / "run.py")[:10]
    started = time.monotonic()
    prior = ROOT.parent / "joint_tuning/runs/evaluation_slice03041_360_sframes_seed202_b16_g0p1"
    config = dict(read_json(prior / "result.json")["config"], run_id=f"smoke_{tag}_sframes",
                  view_budget=17344, continuation_directory=str(prior))
    run(config, cpu=args.cpu, runs_dir=root)
    old_trace = read_json(prior / "trace.json")
    trace = read_json(root / config["run_id"] / "trace.json")
    assert trace[:-1] == old_trace and trace[-1]["iteration"] == 271

    base = dict(run_id=f"smoke_{tag}_spring_prefix", phase="development", slice_id="slice03150",
                n_train=360, method="spring_sarah", variant="spring_sarah", trial="smoke", seed=110,
                calibration_seed=110, grid=512, rank=64, batch=64, view_budget=720,
                rate=1.6, refresh_period=16, record_every_iteration=True)
    run(base, cpu=args.cpu, runs_dir=root)
    continued = dict(base, run_id=f"smoke_{tag}_spring_continued", view_budget=721,
                     continuation_directory=str(root / base["run_id"]))
    uninterrupted = dict(base, run_id=f"smoke_{tag}_spring_uninterrupted", view_budget=721)
    run(continued, cpu=args.cpu, runs_dir=root)
    run(uninterrupted, cpu=args.cpu, runs_dir=root)
    left, right = (pickle.loads((root / config["run_id"] / "checkpoint.pkl").read_bytes())
                   for config in (continued, uninterrupted))
    assert equal(left["solver"], right["solver"])
    assert left["trace"] == right["trace"] and left["counters"] == right["counters"]

    snapshots = {}
    for policy, size in (("minibatch", 256), ("full", 1080)):
        prox = dict(base, run_id=f"smoke_{tag}_prox_{policy}", method="prox_sarah", variant="prox_sarah",
                    n_train=1080, view_budget=size + 1, snapshot_policy=policy,
                    rate=1.5, inner_steps=12, record_every_iteration=False)
        prox.pop("refresh_period")
        run(prox, cpu=args.cpu, runs_dir=root)
        result = read_json(root / prox["run_id"] / "result.json")
        assert result["iteration"] == 2 and result["final"]["forward_views"] == size + 128
        assert result["final"]["gradient_calls"] == 3 and result["optimizer"]["snapshot_batch"] == size
        snapshots[policy] = dict(iteration=2, forward_views=result["final"]["forward_views"], gradient_calls=3)
    jobs = development_jobs(initial_grids())
    assert len(jobs) == 600 and len({row["run_id"] for row in jobs}) == 600
    assert {row["seed"] for row in jobs if row["variant"] != "palm"} == {110, 111, 112}
    assert all(row["calibration_seed"] == 110 for row in jobs)
    audit = dict(sframes_prefix_observations=len(old_trace), sframes_new_iteration=trace[-1]["iteration"],
                 spring_continuation_equals_uninterrupted=True, spring_final_iteration=right["solver"]["k"],
                 prox_snapshot_accounting=snapshots, initial_development_jobs=len(jobs),
                 source_sha256={name: sha256(ROOT / name) for name in
                                ("run.py", "campaign.py", "methods.py", "ct_problem.py",
                                 "spectral_reference_core.py", "spectral_reference_baseline.py")},
                 operational_smoke_elapsed_seconds=time.monotonic() - started)
    write_json(root / "engine_validation.json", audit)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
