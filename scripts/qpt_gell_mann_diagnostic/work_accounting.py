"""Reconstruct optimizer row charges without rerunning optimization.

Retained from 2026-10-06 work_accounting.py, SHA256
53068736a1368f083359e1d8955cfd2e70861b42055ed17da1be6b4c78ce6d8a.
"""
import re

from run_diagnostic import require


def check_work(arrays, prefix, method, status, failure_reason, np):
    steps = arrays[prefix + "_checkpoint_steps"]
    rows = arrays[prefix + "_cumulative_data_row_accesses"]
    times = arrays[prefix + "_optimizer_seconds"]
    require(steps.dtype.kind == rows.dtype.kind == "i" and times.dtype.kind == "f"
            and times.shape == steps.shape == rows.shape and np.all(np.isfinite(times))
            and np.all(times >= 0) and np.all(np.diff(times) >= 0)
            and np.all(rows >= 0) and np.all(np.diff(rows) >= 0),
            "Malformed or nonmonotone time/work arrays")
    if method == "frames":
        require(np.array_equal(rows, 32 * steps), "FRAMES row charge differs from32*steps")
        return {"rule": "32*completed_steps", "verified_terminal_rows": int(rows[-1])}
    if method == "printed-adafgd":
        require(np.array_equal(rows[:-1], 128 * steps[:-1]), "Printed checkpoint row charge differs")
        extra = int(rows[-1]) - 128 * int(steps[-1])
        require(extra == 0 or (status == "failed" and extra == 128),
                "Printed terminal row charge differs")
        return {"rule": "128*completed_steps plus one attempted pass only on failed terminal update",
                "failed_terminal_attempt_rows": extra, "verified_terminal_rows": int(rows[-1])}
    completed = int(steps[-1])
    key = prefix + "_diagnostic_objective_evaluations"
    if key not in arrays:
        require(status == "failed" and completed == 0 and int(rows[-1]) == 0,
                "Armijo objective-evaluation history missing")
        return {"rule": "initial checkpoint failure before optimizer work", "verified_terminal_rows": 0}
    evaluations = arrays[key]
    require(evaluations.dtype.kind == "i" and evaluations.shape == (completed,)
            and np.all(evaluations >= 1), "Malformed Armijo objective-evaluation history")
    cumulative = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(evaluations)))
    accepted_rows = 128 * (steps + cumulative[steps])
    require(np.array_equal(rows[:-1], accepted_rows[:-1]), "Armijo intermediate row charge differs")
    extra = int(rows[-1] - accepted_rows[-1])
    if status == "converged":
        require(extra == 128, "Converged Armijo must charge exactly one terminal gradient pass")
    elif status == "complete":
        require(extra == 0, "Complete Armijo has an unexplained terminal row charge")
    else:
        line_search = re.search(r"Armijo line search failed after (\d+) trials", failure_reason or "")
        if line_search:
            require(extra == 128 * (1 + int(line_search.group(1))),
                    "Failed Armijo line-search row charge differs")
        else:
            require(extra in {0, 128}, "Failed Armijo has an unexplained terminal row charge")
    return {"rule": "128*(accepted gradients + accepted-iteration objective evaluations) + terminal attempt",
            "accepted_gradient_passes": completed, "accepted_iteration_objective_evaluations": int(evaluations.sum()),
            "terminal_attempt_rows": extra, "verified_terminal_rows": int(rows[-1])}
