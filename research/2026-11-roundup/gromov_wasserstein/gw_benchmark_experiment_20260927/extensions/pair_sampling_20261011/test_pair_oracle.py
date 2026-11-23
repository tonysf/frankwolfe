"""Algebra, actual sampling law and uneven four-rank tests; tiny CPU only."""
from __future__ import annotations

import datetime
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from validate_distributed import check_oracle, check_updates, check_prefix_scan
from oracle import fixed_tree_prefix_sum


def worker(rank, world, store):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group('gloo', init_method='file://'+store, rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=120))
    try:
        check_prefix_scan(torch.device('cpu'))
        check_oracle(torch.device('cpu'), torch.float64)
        check_oracle(torch.device('cpu'), torch.float32)
        check_oracle(torch.device('cpu'), torch.float64, n=2, m=3)
        check_updates(torch.device('cpu'))
    finally:
        dist.destroy_process_group()


class PairOracleTests(unittest.TestCase):
    def test_strict_deterministic_prefix_tree_and_production_widths(self):
        previous = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            proof = check_prefix_scan(torch.device('cpu'))
            self.assertIn(31890, proof['widths'])
            # This test would reject a reintroduced floating cumsum call even
            # on a CPU host where that operation would otherwise be supported.
            values = torch.tensor([[0., .5, .25, 0., 1.]], dtype=torch.float64)
            expected = values.cumsum(-1)
            with mock.patch.object(torch.Tensor, 'cumsum', side_effect=AssertionError('Forbidden cumsum')):
                with mock.patch.object(torch, 'cumsum', side_effect=AssertionError('Forbidden cumsum')):
                    torch.testing.assert_close(fixed_tree_prefix_sum(values), expected, rtol=0, atol=0)
        finally:
            torch.use_deterministic_algorithms(previous, warn_only=warn_only)

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
