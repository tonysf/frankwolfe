"""One-process and four-rank Gloo gates with identical independent references."""
from datetime import timedelta
import json
from pathlib import Path
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validation import run_validation


def test_cpu_validation_has_full_pair_and_collective_fallback_checks():
    result = run_validation("cpu")
    assert result["status"] == "passed"
    assert result["world_size"] == 1
    assert len(result["checks"]) == 5
    assert len(result["source_file_sha256"]) >= 5


def _four_rank_worker(rank, rendezvous, directory):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank,
                            world_size=4, timeout=timedelta(seconds=120))
    try:
        result = run_validation("cpu")
        Path(directory, f"rank{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


def test_four_rank_gloo_matches_same_dense_reference_and_collective_fallback():
    with tempfile.TemporaryDirectory(prefix="large-pot-gloo-") as directory:
        rendezvous = "file://" + str(Path(directory) / "rendezvous")
        mp.spawn(_four_rank_worker, args=(rendezvous, directory), nprocs=4, join=True)
        records = [json.loads(Path(directory, f"rank{rank}.json").read_text())
                   for rank in range(4)]
        for record in records:
            assert record["status"] == "passed" and record["world_size"] == 4
        for record in records[1:]:
            assert record["checks"] == records[0]["checks"]
