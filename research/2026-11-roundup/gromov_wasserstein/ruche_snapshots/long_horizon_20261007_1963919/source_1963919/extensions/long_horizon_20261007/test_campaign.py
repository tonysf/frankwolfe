"""Dispatch, frozen selection, failure preservation, and metadata audit tests."""
from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c = module_at('_long_horizon_campaign_under_test', ROOT / 'campaign.py')
prior_fixture = module_at('_long_horizon_prior_fixture', ROOT.parent / 'msbpg_20261005/test_campaign.py')


def metadata(expected, provenance):
    # Reuse the already independent theorem/certificate fixture, then specialize
    # the non-MSBPG counters. Production validators are never mocked here.
    expanded = dict(expected, momentum_prefactor=expected.get('momentum_prefactor', 1.),
                    coupling_fraction=expected.get('coupling_fraction', 1.))
    result = prior_fixture.metadata(expanded, provenance)
    result.update(config=expected, method_key=expected['method'])
    if expected['method'] != 'published-msbpg':
        result['counters']['initialization_oracle_calls'] = int(expected['method'] == 'smoothing-fw')
    if expected['method'] == 'smoothing-fw':
        result['inner_trace'] = []
        result['counters'].update(inner_iterations_total=0, certificate_checks=0)
    return result


class Fixture:
    def __init__(self, base):
        self.base = Path(base)
        self.source, self.prepared, self.out = (self.base / name for name in ('source', 'prepared', 'results'))
        self.path = self.base / 'campaign.json'
        self.config = c.read_json(ROOT / 'campaign.json')
        self.config.update(final_iterations=4, final_checkpoint_iterations=[1, 4])
        for name in (*c.SOURCE_FILES, *c.CONTROLLER_FILES):
            path = self.source / 'extensions' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('immutable source fixture: '+name)
        data = self.prepared / 'tiny'
        data.mkdir(parents=True)
        hashes = {name: 'b'*64 for name in c.ARRAYS}
        c.write_json(data / 'manifest.json', dict(array_files_sha256=hashes))
        manifest_hash = c.sha256(data / 'manifest.json')
        self.config['instances'] = {'tiny': dict(n=8, world_size=1, array_files_sha256=hashes,
                                                instance_manifest_sha256=manifest_hash)}
        self.config['selected_candidates'] = {'tiny': {method: candidates[3]
                                                       for method, candidates in self.config['methods'].items()}}
        self.artifact = dict(schema_version=1, selections={'tiny': {
            method: dict(candidate=candidate, selection_file_sha256='c'*64,
                         array_files_sha256=hashes, instance_manifest_sha256=manifest_hash)
            for method, candidate in self.config['selected_candidates']['tiny'].items()}})
        self.calls = []
        self.persist()

    def persist(self):
        path = self.base / self.config['selections_provenance']['file']
        c.write_json(path, self.artifact)
        self.config['selections_provenance']['sha256'] = c.sha256(path)
        c.write_json(self.path, self.config)

    def args(self, method='published-msbpg', index=0):
        return argparse.Namespace(config=str(self.path), source=str(self.source), prepared=str(self.prepared),
                                  out=str(self.out), instance='tiny', method=method, task_index=index)

    def execute(self, command, path, timeout):
        request = c.read_json(path.parent / 'request.json')
        expected = request['expected_runner_config']
        self.calls.append((expected, timeout, command))
        Path(expected['out']).mkdir(parents=True)
        c.write_json(Path(expected['out']) / 'metadata.json', metadata(expected, request))
        path.write_text('numerical worker fixture completed\n')
        return 0


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='long_horizon_controller_')
        self.fixture = Fixture(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_six_seed_start_tasks_and_exact_legacy_dispatch_for_all_methods(self):
        f = self.fixture
        config = c.load_config(f.path)
        self.assertEqual([c.cell_for_task(config, i) for i in range(6)],
                         [('product', 810), ('product', 811), ('product', 812),
                          ('randomized', 810), ('randomized', 811), ('randomized', 812)])
        for method in c.METHODS:
            for i in range(6):
                result = c.run_task(f.args(method, i), executor=f.execute)
                self.assertEqual(result['status'], 'completed')
                start, seed = c.cell_for_task(config, i)
                audit = c.validate_run(config, f.source, f.prepared, f.out, 'tiny', method, start, seed)
                self.assertEqual(audit['last_rounded_loss'], .2)
                arguments = audit['request']['expected_runner_config']
                original = c.legacy(method).runner_config(config, f.prepared, arguments['out'], 'tiny', 'final',
                    method, config['selected_candidates']['tiny'][method], start, seed)
                original['save_plans'] = 'none'
                self.assertEqual(arguments, original)
                self.assertEqual(arguments['iterations'], 4)
                self.assertEqual((arguments['init_seed'], arguments['output_seed']), (seed+10000, seed+20000))
                self.assertEqual(set(audit['request']['source_file_sha256']), set(c.legacy(method).SOURCE_FILES))
        self.assertEqual(len(f.calls), 18)
        self.assertEqual({call[1] for call in f.calls}, set(config['run_timeout_seconds_by_method'].values()))

    def test_selected_parameters_are_bound_to_artifact_and_catalog(self):
        f = self.fixture
        changed = copy.deepcopy(f.config)
        changed['selected_candidates']['tiny']['published-msbpg'] = changed['methods']['published-msbpg'][4]
        c.write_json(f.path, changed)
        with self.assertRaisesRegex(ValueError, 'frozen selection evidence'):
            c.load_config(f.path)
        changed['selected_candidates']['tiny']['published-msbpg'] = dict(id='rogue', coupling_fraction=.5, momentum_prefactor=.3)
        with self.assertRaisesRegex(ValueError, 'original catalog'):
            c.validate_config(changed)

    def test_artifact_and_prepared_manifest_changes_are_rejected(self):
        f = self.fixture
        artifact = f.base / f.config['selections_provenance']['file']
        original = artifact.read_bytes()
        artifact.write_bytes(original+b' ')
        with self.assertRaisesRegex(ValueError, 'artifact hash'):
            c.run_task(f.args(), executor=f.execute)
        artifact.write_bytes(original)
        manifest = f.prepared / 'tiny/manifest.json'
        manifest.write_bytes(manifest.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError, 'Prepared manifest'):
            c.run_task(f.args(), executor=f.execute)
        self.assertFalse(f.calls)

    def test_existing_run_is_never_overwritten_or_resumed(self):
        f = self.fixture
        c.run_task(f.args(), executor=f.execute)
        directory = c.run_directory(f.out, 'tiny', 'published-msbpg', 'product', 810)
        original = (directory / 'status.json').read_bytes()
        with self.assertRaises(FileExistsError):
            c.run_task(f.args(), executor=f.execute)
        self.assertEqual((directory / 'status.json').read_bytes(), original)
        self.assertEqual(len(f.calls), 1)

    def test_nonzero_worker_exit_preserves_failed_receipt(self):
        f = self.fixture
        with self.assertRaisesRegex(ValueError, 'exit status 19'):
            c.run_task(f.args(), executor=lambda *_: 19)
        directory = c.run_directory(f.out, 'tiny', 'published-msbpg', 'product', 810)
        receipt = c.read_json(directory / 'status.json')
        self.assertEqual((receipt['status'], receipt['returncode']), ('failed', 19))
        self.assertEqual(receipt['request_sha256'], c.sha256(directory / 'request.json'))
        with self.assertRaises(FileExistsError):
            c.run_task(f.args(), executor=f.execute)

    def test_wrong_theoretical_schedule_and_inexact_prox_do_not_complete(self):
        f = self.fixture
        def bad_worker(command, path, timeout):
            f.execute(command, path, timeout)
            target = path.parent / 'output/metadata.json'
            value = c.read_json(target)
            value['inner_trace'][1]['beta_next'] *= 2
            c.write_json(target, value)
            return 0
        with self.assertRaisesRegex(ValueError, 'Theorem parameter mismatch'):
            c.run_task(f.args(), executor=bad_worker)
        def inexact_worker(command, path, timeout):
            f.execute(command, path, timeout)
            target = path.parent / 'output/metadata.json'
            value = c.read_json(target)
            value['inner_trace'][1]['normalized_gap_upper_bound'] = 1e-6
            c.write_json(target, value)
            return 0
        with self.assertRaisesRegex(ValueError, 'tolerance'):
            c.run_task(f.args('published-smd'), executor=inexact_worker)

    def test_metadata_request_and_source_tampering_are_detected(self):
        f = self.fixture
        c.run_task(f.args(), executor=f.execute)
        config = c.load_config(f.path)
        directory = c.run_directory(f.out, 'tiny', 'published-msbpg', 'product', 810)
        for path, message in ((directory / 'output/metadata.json', 'Metadata changed'),
                              (directory / 'request.json', 'Frozen request changed'),
                              (f.source / 'extensions/msbpg_20261005/msbpg.py', 'Request provenance')):
            original = path.read_bytes()
            path.write_bytes(original+b' ')
            with self.assertRaisesRegex(ValueError, message):
                c.validate_run(config, f.source, f.prepared, f.out, 'tiny', 'published-msbpg', 'product', 810)
            path.write_bytes(original)

    def test_invalid_config_and_task_are_rejected(self):
        f = self.fixture
        mutations = [lambda x: x.__setitem__('plan_storage', 'all'),
                     lambda x: x.__setitem__('final_seeds', [710, 711, 712]),
                     lambda x: x['runner'].__setitem__('inner_tol', 1e-6),
                     lambda x: x['run_timeout_seconds_by_method'].__setitem__('published-msbpg', 50000),
                     lambda x: x.__setitem__('final_checkpoint_iterations', [1, 3])]
        for mutate in mutations:
            config = copy.deepcopy(f.config)
            mutate(config)
            with self.assertRaises(ValueError):
                c.validate_config(config)
        for index in (-1, 6, True):
            with self.assertRaises(ValueError):
                c.cell_for_task(f.config, index)


if __name__ == '__main__':
    unittest.main()
