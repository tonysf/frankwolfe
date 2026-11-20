"""CPU/four-A100 numerical and completed-horizon continuation gates."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "extensions/large_pot_frames_20261016"))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


runner = module("share_continuation_gate_runner", HERE / "runner.py")
old_validation = module("share_continuation_base_validation", ROOT / "extensions/large_pot_frames_20261016/validation.py")
from checkpoint import CheckpointStore, atomic_json, sha256


def fixture(path):
    path.mkdir(parents=True)
    rng = np.random.default_rng(868)
    x, y = rng.normal(size=(8, 3)), rng.normal(size=(8, 3))
    arrays = dict(A=(.02*np.linalg.norm(x[:, None]-x[None, :], axis=-1)).astype("float32"),
                  B=(.02*np.linalg.norm(y[:, None]-y[None, :], axis=-1)).astype("float32"),
                  a=np.full(8, 1/8, dtype="float64"), b=np.full(8, 1/8, dtype="float64"))
    for name, value in arrays.items():
        np.save(path / (name + ".npy"), value)
    atomic_json(path / "manifest.json", dict(n=8, m=8,
        array_files_sha256={name+".npy": sha256(path/(name+".npy")) for name in arrays}))


def continuation_check(root, device, rank, world):
    instance = root / "continuation_input"
    if rank == 0:
        fixture(instance)
    if world > 1:
        dist.barrier()
    def args(name, horizon, extend=None):
        command = ["--instance", str(instance), "--out", str(root/name), "--method", "frames",
                   "--device", device, "--seed", "810", "--iterations", str(horizon),
                   "--pairs", "1024", "--gamma0", ".3", "--rho0", ".3", "--mu-scale", "10",
                   "--checkpoint-every", "2", "--diagnostic-every", "2", "--retain-checkpoint"]
        if extend:
            command += ["--extend-from", str(extend)]
        return runner.parser().parse_args(command)
    full = runner.run(args("uninterrupted_6", 6))
    parent = runner.run(args("completed_4", 4))
    parent_folder = root / "completed_4/checkpoints"
    parent_marker = (parent_folder / "COMMITTED.json").read_bytes()
    parent_shard = parent_folder / parent["checkpoint"]["generation"] / f"rank-{rank:04d}.pt"
    parent_hash = sha256(parent_shard)
    extended = runner.run(args("extended_6", 6, parent_folder))
    assert full["status"] == parent["status"] == extended["status"] == "completed"
    assert full["trace"] == extended["trace"]
    assert full["counters"] == extended["counters"]
    assert (parent_folder / "COMMITTED.json").read_bytes() == parent_marker
    assert sha256(parent_shard) == parent_hash
    a, _ = CheckpointStore(root / "uninterrupted_6/checkpoints", rank=rank, world_size=world).load(
        identity=full["identity"], map_location="cpu")
    b, _ = CheckpointStore(root / "extended_6/checkpoints", rank=rank, world_size=world).load(
        identity=extended["identity"], map_location="cpu")
    for name in ("plan", "momentum", "generator_state", "torch_rng_state"):
        assert torch.equal(a[name], b[name]), name
    if device == "cuda":
        assert torch.equal(a["cuda_rng_state"], b["cuda_rng_state"])
    return dict(name="completed_horizon_extension", parent_updates=4, final_updates=6,
                exact_plan_momentum_rng_trace_match=True, parent_checkpoint_unchanged=True,
                world_size=world)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if args.device == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        assert "A100" in torch.cuda.get_device_name(), "Validate on target hardware"
    if world > 1:
        dist.init_process_group("nccl" if args.device == "cuda" else "gloo")
    try:
        args.out.mkdir(parents=True, exist_ok=True)
        base = old_validation.run_validation(args.device)
        base["checks"].append(continuation_check(args.out, args.device, rank, world))
        base["source_file_sha256"].update(runner.source_hashes())
        base["validation_sha256"] = sha256(Path(__file__))
        base["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
        if rank == 0:
            atomic_json(args.out / "checks.json", base)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
