"""Launch or resume the two local-curvature arms and verify their strict pairs."""

from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys

from run import atomic_json, now, source_hashes


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_pair(strict, practical, iterations):
    """Check horizon, every logged oracle count, and terminal sampling state."""
    strict_trace = read_json(strict / "trace.json")
    practical_trace = read_json(practical / "trace.json")
    columns = ("iteration", "component_gradient_calls", "sampled_images")
    strict_counts = [tuple(row[key] for key in columns) for row in strict_trace]
    practical_counts = [tuple(row[key] for key in columns) for row in practical_trace]
    if strict_counts != practical_counts or strict_counts[-1][0] != iterations:
        raise RuntimeError(f"Oracle trace mismatch: {practical}")
    if strict_trace[0]["raw_loss"] != practical_trace[0]["raw_loss"]:
        raise RuntimeError(f"Initial objective mismatch: {practical}")
    states = []
    for directory in (strict, practical):
        with (directory / "checkpoint.pkl").open("rb") as stream:
            checkpoint = pickle.load(stream)
        if checkpoint["iteration"] != iterations:
            raise RuntimeError(f"Checkpoint horizon mismatch: {directory}")
        states.append(checkpoint["solver"])
    for key in ("rng_state", "full_gradient_calls", "sampled_gradient_calls"):
        if states[0][key] != states[1][key]:
            raise RuntimeError(f"Paired {key} mismatch: {practical}")
    return dict(matched=True, probes=len(strict_counts),
                component_gradient_calls=strict_counts[-1][1],
                full_block_refreshes=states[1]["full_gradient_calls"])


def main():
    lock = (HERE / "launch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = read_json(HERE / "manifest.json")
    campaign_path = PROJECT / "campaign_config.json"
    if sha256(campaign_path) != manifest["campaign_config_sha256"]:
        raise RuntimeError("Frozen campaign configuration changed")
    campaign = read_json(campaign_path)
    frozen_hashes = campaign["source_sha256"] | campaign["pipeline_sha256"]
    if any(sha256(PROJECT / name) != expected for name, expected in frozen_hashes.items()):
        raise RuntimeError("Frozen campaign sources changed")
    if source_hashes() != manifest["source_sha256"] or sha256(Path(__file__)) != manifest["launcher_sha256"]:
        raise RuntimeError("Ablation source differs from the manifest")

    strict = {r["dataset"]: r for r in manifest["runs"] if r["step_rule"] == "theorem"}
    practical = [r for r in manifest["runs"] if r["step_rule"] == "local_curvature"]
    labels_and_rule = {"baseline_mode", "local_curvature_seed", "step_rule", "candidate", "phase"}
    for arm in practical:
        reference = PROJECT / strict[arm["dataset"]]["path"]
        result = read_json(reference / "result.json")
        if (result["config"] != strict[arm["dataset"]]["config"]
                or result["source_sha256"] != campaign["source_sha256"]
                or result["status"] != "completed"
                or result["iteration"] != arm["config"]["iterations"]
                or result["data_sha256"] != sha256(PROJECT / arm["config"]["data_path"])):
            raise RuntimeError(f"Invalid strict reference: {reference}")
        common_config = lambda config: {k: v for k, v in config.items() if k not in labels_and_rule}
        if common_config(arm["config"]) != common_config(result["config"]):
            raise RuntimeError(f"Ablation changes a frozen setting: {arm['dataset']}")
        if arm["cpu"] not in os.sched_getaffinity(0):
            raise RuntimeError(f"Requested CPU unavailable: {arm['cpu']}")

    progress_path = HERE / "progress.json"
    status = dict(status="running", pid=os.getpid(), host=platform.node(), started_utc=now(),
                  python=sys.version, executable=sys.executable, runs={})
    atomic_json(progress_path, status)
    try:
        children = {}
        with ExitStack() as files:
            for arm in practical:
                dataset = arm["dataset"]
                output = PROJECT / arm["path"]
                output.mkdir(parents=True, exist_ok=True)
                config_path = output / "config.json"
                atomic_json(config_path, arm["config"])
                if (output / "result.json").exists():
                    continue
                log = files.enter_context((output / "stdout.log").open("a"))
                children[dataset] = subprocess.Popen(
                    [sys.executable, "-u", str(HERE / "run.py"), str(config_path),
                     str(output), "--cpu", str(arm["cpu"])],
                    cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
                status["runs"][dataset] = dict(pid=children[dataset].pid, status="running")
            atomic_json(progress_path, status)
            for arm in practical:
                dataset = arm["dataset"]
                exit_code = children[dataset].wait() if dataset in children else 0
                status["runs"][dataset] = dict(exit_code=exit_code, status="failed")
                output = PROJECT / arm["path"]
                if exit_code == 0:
                    try:
                        result = read_json(output / "result.json")
                        if (result["config"] != arm["config"]
                                or result["source_sha256"] != manifest["source_sha256"]
                                or result["status"] != "completed"
                                or result["iteration"] != arm["config"]["iterations"]
                                or result["data_sha256"] != read_json(
                                    PROJECT / strict[dataset]["path"] / "result.json")["data_sha256"]):
                            raise RuntimeError(f"Invalid practical result: {output}")
                        pairing = verify_pair(PROJECT / strict[dataset]["path"], output,
                                              arm["config"]["iterations"])
                        status["runs"][dataset].update(status="completed", pairing=pairing)
                    except (OSError, ValueError, RuntimeError) as exc:
                        status["runs"][dataset]["error"] = str(exc)
                atomic_json(progress_path, status)
        subprocess.run([sys.executable, str(HERE / "report.py")], cwd=HERE, check=True)
        status["status"] = "completed" if all(
            r["status"] == "completed" for r in status["runs"].values()) else "failed"
    except Exception as exc:
        status.update(status="failed", error=repr(exc))
        raise
    finally:
        atomic_json(progress_path, dict(status, finished_utc=now()))
    return 0 if status["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
