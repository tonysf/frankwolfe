"""Algebra, actual sampling law and uneven four-rank tests; tiny CPU only."""
from __future__ import annotations

import datetime
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from validate_distributed import check_oracle, check_updates


def worker(rank, world, store):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://'+store, rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=120))
    try:
        check_oracle(torch.device('cpu'), torch.float64)
        check_oracle(torch.device('cpu'), torch.float32)
        check_oracle(torch.device('cpu'), torch.float64, n=2, m=3)
        check_updates(torch.device('cpu'))
    finally:
        dist.destroy_process_group()


class PairOracleTests(unittest.TestCase):
    def test_float64_weighted_enumeration_actual_law_and_rejections(self):
        torch.set_num_threads(1)
        proof = check_oracle(torch.device('cpu'), torch.float64)
        self.assertNotEqual(proof['nonunit_mass'], 1.)
        self.assertLess(proof['exact_enumeration_maximum_abs_error'], 3e-12)

    def test_float32_arithmetic(self):
        check_oracle(torch.device('cpu'), torch.float32)

    def test_four_shards_rectangular_sparse_empty_shards_and_all_updates(self):
        with tempfile.TemporaryDirectory(prefix='pair_shard_test_') as tmp:
            mp.spawn(worker, args=(4, str(Path(tmp)/'store')), nprocs=4, join=True)


if __name__ == '__main__':
    unittest.main()
