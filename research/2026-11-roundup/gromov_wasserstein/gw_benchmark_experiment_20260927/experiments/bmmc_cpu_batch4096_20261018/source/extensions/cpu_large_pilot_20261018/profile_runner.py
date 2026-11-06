"""Measure CPU components of the unchanged timestamped campaign runner.

Every wrapper delegates once with the original arguments and returns the
original result. JSONL events survive a supervisor timeout; only successful
updates are reported as completed. The underlying soft limit is checked
between outer updates, so a supervisor must enforce any hard deadline.
"""
from __future__ import annotations

from contextlib import contextmanager
import functools
import importlib.util
import json
import os
from pathlib import Path
import resource
import time
import uuid


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location(
    "cpu_pilot_campaign_runner", ROOT / "extensions/wallclock_replay_20261018/runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@contextmanager
def instrument(events_path):
    """Temporarily instrument public component boundaries, without tensor work."""
    events_path = Path(events_path)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    launch_id = str(uuid.uuid4())
    started = time.monotonic()
    totals, counts, patches = {}, {}, []
    pending = {"started": None, "number": 0, "base_iteration": 0}
    with events_path.open("a", buffering=1) as handle:
        def emit(phase, duration=None, **details):
            usage = resource.getrusage(resource.RUSAGE_SELF)
            event = dict(launch_id=launch_id, phase=phase,
                         elapsed_seconds=time.monotonic() - started,
                         peak_rss_bytes=usage.ru_maxrss * 1024,
                         cpu_user_seconds=usage.ru_utime, cpu_system_seconds=usage.ru_stime,
                         rss_bytes=int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE"),
                         completed_updates_in_launch=pending["number"],
                         completed_updates=pending["base_iteration"] + pending["number"],
                         **details)
            if duration is not None:
                event["duration_seconds"] = duration
                totals[phase] = totals.get(phase, 0.) + duration
                counts[phase] = counts.get(phase, 0) + 1
                event["completed_calls"] = counts[phase]
            handle.write(json.dumps(event, sort_keys=True) + "\n")

        def patch(owner, name, phase, *, begins_update=False, ends_update=False):
            original = getattr(owner, name)

            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                tick = time.monotonic()
                if begins_update:
                    pending["started"] = tick
                try:
                    result = original(*args, **kwargs)
                except BaseException as error:
                    emit(phase + "_failed", time.monotonic() - tick,
                         error=f"{type(error).__name__}: {error}")
                    raise
                returned = time.monotonic()
                details = {}
                if phase == "diagnostic":
                    details = {"diagnostic": result, "iteration": result["iteration"]}
                elif phase == "pot_step":
                    info = result[1]
                    details = {key: info.get(key) for key in
                               ("inner_iterations", "sinkhorn_path", "marginal_l1",
                                "normalized_gap_upper_bound")}
                    details["certificate"] = info
                elif phase == "checkpoint":
                    details["iteration"] = kwargs.get("iteration")
                elif phase == "checkpoint_load":
                    pending["base_iteration"] = result[0]["iteration"] if result[0] is not None else 0
                    details["iteration"] = pending["base_iteration"]
                if pending["started"] is not None:
                    details["outer_update_index"] = pending["base_iteration"] + pending["number"] + 1
                emit(phase, returned - tick, **details)
                if ends_update:
                    if pending["started"] is None:
                        raise RuntimeError("Completed update lacks an oracle start event")
                    pending["number"] += 1
                    emit("update", returned - pending["started"],
                         update_in_launch=pending["number"], **details)
                    pending["started"] = None
                return result

            patches.append((owner, name, original))
            setattr(owner, name, wrapped)

        patch(runner, "load_problem", "load_problem")
        patch(runner, "diagnose", "diagnostic")
        patch(runner, "sample_pairs", "sample_pairs", begins_update=True)
        patch(runner, "pair_gradient", "pair_gradient")
        patch(runner.RowShardedGW, "gradient", "full_gradient", begins_update=True)
        patch(runner.RowShardedGW, "fw_step", "fw_step", ends_update=True)
        patch(runner, "pot_step", "pot_step", ends_update=True)
        patch(runner.CheckpointStore, "save", "checkpoint")
        patch(runner.CheckpointStore, "load", "checkpoint_load")
        summary = dict(launch_id=launch_id, events_path=str(events_path.resolve()),
                       component_total_seconds=totals, component_counts=counts)
        emit("profile_start", pid=os.getpid(), requested_omp_threads=os.environ.get("OMP_NUM_THREADS"))
        try:
            yield summary
        finally:
            for owner, name, original in reversed(patches):
                setattr(owner, name, original)
            summary.update(elapsed_seconds=time.monotonic() - started,
                           peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                           completed_updates_in_launch=pending["number"])
            emit("profile_end")


def run_profiled(args, profile_out=None):
    if args.device != "cpu":
        raise ValueError("This pilot wrapper supports CPU only")
    profile_out = Path(profile_out or args.out)
    events_path = profile_out / "profile.jsonl"
    with instrument(events_path) as summary:
        record = runner.run(args)
    summary.update(status=record["status"], wrapper_sha256=runner.sha256(__file__),
                   source_file_sha256=record["source_file_sha256"],
                   num_threads=record["identity"]["num_threads"],
                   note="Component timers overlap the enclosing update timer. Update durations exclude diagnostics and checkpoint writes; include the oracle, momentum if used, and optimizer update. Underlying numerical calls and stopping tolerances are unchanged. Peak RSS is the process high-water mark; imports precede profile timing.")
    runner.atomic_json(profile_out / "profile_summary.json", summary)
    return record, summary


def main():
    parser = runner.parser()
    parser.add_argument("--profile-out")
    args = parser.parse_args()
    _, summary = run_profiled(args, args.profile_out)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
