"""Controller/audit checks using existing pilot receipts; no solver is run."""
import concurrent.futures
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('scot_methods_contract', HERE / 'run_comparison.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='scot_methods_contract_')
        self.out = Path(self.tmp.name)
        self.spec = c.read(HERE / 'specification.json')

    def tearDown(self):
        self.tmp.cleanup()

    def test_four_admissible_candidates_each_and_fixed_numerical_arguments(self):
        c.validate_spec(self.spec)
        self.assertEqual([row['prefactor_fraction'] for row in self.spec['methods']['published-smd']], [.1, .3, .6, 1.])
        for method, rows in self.spec['methods'].items():
            self.assertEqual(len(rows), 4)
            for candidate in rows:
                words = c.arguments(self.out, 'tune', method, candidate, 810)
                args = dict(zip(words[::2], words[1::2]))
                self.assertEqual(args['--dtype'], 'float64')
                self.assertEqual(args['--iterations'], '2000')
                self.assertEqual(args['--pairs-per-call'], '1024')
                self.assertEqual(args['--inner-tol'], '1e-07')
                self.assertEqual(args['--inner-gap-tol'], '1e-07')
                self.assertEqual(args['--checkpoints'], '')
                self.assertEqual(args['--initialization'], 'product')
        for change in (lambda x: x['methods']['published-smd'][0].update(exponent=.1),
                       lambda x: x['methods']['published-msbpg'][0].update(coupling_fraction=2.),
                       lambda x: x['methods']['published-smd'].pop()):
            value = copy.deepcopy(self.spec)
            change(value)
            with self.assertRaises(ValueError):
                c.validate_spec(value)

    def test_four_core_leases_allow_four_simultaneous_workers(self):
        (self.out / 'core_leases').mkdir()
        config = dict(physical_cores=[dict(cpu_id=i) for i in range(4)], run_timeout_seconds=5)
        barrier = threading.Barrier(4)
        def task(index):
            with c.core_lease(self.out, config) as core:
                barrier.wait(timeout=3)
                return core['cpu_id']
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            values = list(pool.map(task, range(4)))
        self.assertEqual(set(values), {0, 1, 2, 3})

    def test_frozen_source_revision_is_not_replaced_by_current_git_head(self):
        fake_runner = SimpleNamespace(source_hashes=lambda: {'runner': 'numeric-hash'})
        with mock.patch.object(c, 'runner', return_value=fake_runner), \
                mock.patch.object(c, 'sha256', return_value='source-hash'), \
                mock.patch.object(c.subprocess, 'check_output', side_effect=AssertionError('must not query current HEAD')):
            value = c.source_identity('frozen-revision')
        self.assertEqual(value['source_commit'], 'frozen-revision')
        self.assertEqual(value['source_file_sha256'], {'runner': 'numeric-hash'})
        self.assertNotIn('scot_cpu_methods_fp64_20261014/summarize.py', value['helper_source_sha256'])

    def test_durable_failure_and_no_overwrite(self):
        def fail():
            raise RuntimeError('deliberate failure')
        with self.assertRaises(RuntimeError):
            c.durable_stage(self.out, 'tune_test', fail)
        path = self.out / 'stage_tune_test.json'
        before = path.read_bytes()
        self.assertEqual(c.read(path)['status'], 'failed')
        with self.assertRaises(FileExistsError):
            c.durable_stage(self.out, 'tune_test', lambda: {})
        self.assertEqual(path.read_bytes(), before)

    def test_selection_is_methodwise_and_preserves_declared_tie_order(self):
        rows = [dict(method=method, candidate=candidate, last_rounded_loss_float64=.07)
                for method in c.METHODS for candidate in self.spec['methods'][method]]
        self.assertEqual(c.best(rows)['published-smd']['id'], 'smd00')
        rows[3]['last_rounded_loss_float64'] = .06
        self.assertEqual(c.best(rows)['published-smd']['id'], 'smd03')
        self.assertEqual(c.best(rows)['published-msbpg']['id'], 'msbpg00')

    def test_real_200_step_pilot_audits_and_rejected_inner_certificate(self):
        root = c.ROOT / 'experiments/scot_cpu_methods_fp64_20261014'
        for name in ('pilot_smd', 'pilot_msbpg'):
            path = root / name / 'metadata.json'
            if not path.exists():
                self.skipTest('Local numerical pilot receipt unavailable.')
            metadata = c.read(path)
            self.assertEqual(metadata['config']['iterations'], 200)
            binding = {key: metadata[key] for key in ('array_files_sha256', 'instance_manifest_sha256')}
            loss = c.audit_numerics(metadata, metadata['config'], binding)
            self.assertEqual(loss, metadata['final_float64_validation']['plans']['last_rounded']['loss'])
            modified = copy.deepcopy(metadata)
            modified['inner_trace'][0]['normalized_gap_upper_bound'] = 1e-6
            with self.assertRaisesRegex(ValueError, 'Inner acceptance'):
                c.audit_numerics(modified, modified['config'], binding)


if __name__ == '__main__':
    unittest.main()
