"""Routing is complete, fixed, provenance-bound and never chosen by loss."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_test_combine_long_campaigns', ROOT/'combine_campaigns.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class CombinedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='combined_long_campaigns_')
        self.root = Path(self.temp.name)
        original = c.reporter.read(ROOT/'campaign.json')
        share = copy.deepcopy(original)
        older = c.reporter.read(ROOT.parent/'msbpg_20261005'/'campaign.json')
        share['instances'] = {'shareseq_full': older['instances']['shareseq_full']}
        frozen = c.reporter.read(ROOT/'frozen_selections.json')
        share['selected_candidates'] = {'shareseq_full': {
            method: record['candidate'] for method, record in frozen['selections']['shareseq_full'].items()}}
        self.manifest = dict(schema_version=1, parts={}, routes=[])
        for key, config in [('original', original), ('retry', original), ('share', share)]:
            folder = self.root/key
            folder.mkdir()
            path = folder/'campaign.json'
            path.write_text(json.dumps(config))
            (folder/'frozen_selections.json').write_bytes((ROOT/'frozen_selections.json').read_bytes())
            self.manifest['parts'][key] = dict(config=str(path), config_sha256=c.reporter.sha256(path),
                source=str(folder/'source'), prepared=str(folder/'prepared'), out=str(folder/'results'))
        for dataset in c.DATASETS:
            for method in c.reporter.METHODS:
                key = ('share' if dataset == 'shareseq_full' else 'retry'
                       if (dataset == 'bmmc6224') == (method == 'published-msbpg') else 'original')
                self.manifest['routes'].append(dict(dataset=dataset, method=method, part=key))
        self.path = self.root/'combine.json'
        self.save_manifest()
        self.overrides = {}

    def tearDown(self):
        self.temp.cleanup()

    def save_manifest(self):
        self.path.write_text(json.dumps(self.manifest))

    def change_config(self, key, change, rebind=True):
        part = self.manifest['parts'][key]
        path = Path(part['config'])
        value = c.reporter.read(path)
        change(value)
        path.write_text(json.dumps(value))
        if rebind:
            part['config_sha256'] = c.reporter.sha256(path)
        self.save_manifest()

    def audit_fixture(self, config_path, source, prepared, out):
        config = c.reporter._contract.load_config(config_path)
        key = Path(config_path).parent.name
        rows = []
        for dataset in config['instances']:
            for method in c.reporter.METHODS:
                for start in config['initializations']:
                    for seed in config['final_seeds']:
                        state = self.overrides.get((key, dataset, method, start, seed), 'completed')
                        loss = {'original': .1, 'retry': .4, 'share': .7}[key] + (seed-810)*.01
                        trajectory = [dict(iteration=k, rounded_loss=loss+.01*(k != 20000),
                                           precision='float64' if k == 20000 else 'float32',
                                           kind='last' if k == 20000 else 'current_diagnostic',
                                           actual_solver_seconds=k/1000,
                                           time_to_this_feasible_output=k/1000+.1)
                                      for k in c.reporter.checkpoint_iterations(config)]
                        rows.append(dict(dataset=dataset, method=method, initialization=start, seed=seed,
                            candidate_id=config['selected_candidates'][dataset][method]['id'], status=state,
                            last_rounded_loss=loss, solver_seconds=20., last_feasible_output_seconds=20.1,
                            stochastic_gradient_calls=20000, sketch_probes=2560000, completed_updates=20000,
                            inner_iterations=1000, certificate_checks=100,
                            sampled_normalized_displacement_squared=.001, trajectory=trajectory,
                            reason='Infrastructure failure' if state == 'failed' else None))
        counts = {state: sum(row['status'] == state for row in rows) for state in c.reporter.STATES}
        return dict(config_sha256=c.reporter.sha256(config_path), runs=rows, task_counts=counts,
                    status='completed' if counts['completed'] == len(rows) else 'incomplete')

    def fixture_combine(self):
        with mock.patch.object(c.reporter, 'audit', side_effect=self.audit_fixture) as audit:
            result = c.combine(self.path)
        self.assertEqual(audit.call_count, 3)
        return result

    def test_all_missing_real_auditor_keeps_all_54_runs_and_18_groups(self):
        result = c.combine(self.path)
        self.assertEqual(result['status'], 'incomplete')
        self.assertEqual(result['task_counts']['missing'], 54)
        self.assertEqual(len(result['final_summaries']), 18)
        self.assertEqual(len(result['trajectory_summaries']), 18*len(result['checkpoints']))
        self.assertTrue(all(row['median'] is None for row in result['trajectory_summaries']))
        self.assertEqual(result['nonrouted_attempt_counts']['missing'], 36)
        c.reporter.write_report(result, self.root/'report', no_plots=True)
        saved = c.reporter.read(self.root/'report/report.json')
        self.assertEqual(saved['task_counts']['missing'], 54)
        self.assertTrue((self.root/'report/median_loss_iterations.csv').exists())

    def test_all_parts_audited_routed_population_has_54_complete_runs(self):
        result = self.fixture_combine()
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['task_counts']['completed'], 54)
        self.assertEqual(len(result['final_summaries']), 18)
        row = next(row for row in result['final_summaries']
                   if (row['dataset'], row['method'], row['initialization'])
                   == ('bmmc6224', 'published-msbpg', 'product'))
        self.assertAlmostEqual(row['statistics']['last_rounded_loss']['median'], .41)
        self.assertIn('not time to equal accuracy', result['timing_scope'])

    def test_prior_failed_attempts_remain_separate_from_fixed_retry_population(self):
        for start in ('product', 'randomized'):
            for seed in (810, 811, 812):
                self.overrides[('original', 'bmmc6224', 'published-msbpg', start, seed)] = 'failed'
        result = self.fixture_combine()
        self.assertEqual(result['task_counts']['failed'], 0)
        self.assertEqual(result['nonrouted_attempt_counts']['failed'], 6)
        self.assertEqual(len(result['nonrouted_failed_attempts']), 6)
        self.assertTrue(all(row['campaign_part'] == 'original' for row in result['nonrouted_failed_attempts']))

    def test_missing_routed_seed_never_falls_back_to_completed_original(self):
        self.overrides[('retry', 'bmmc6224', 'published-msbpg', 'product', 811)] = 'missing'
        result = self.fixture_combine()
        self.assertEqual(result['task_counts']['missing'], 1)
        population = [row for row in result['trajectory_summaries'] if row['dataset'] == 'bmmc6224'
                      and row['method'] == 'published-msbpg' and row['initialization'] == 'product']
        self.assertTrue(all(row['n'] == 2 and row['median'] is None for row in population))

    def test_duplicate_missing_and_unknown_routes_rejected(self):
        original = copy.deepcopy(self.manifest)
        for mutate in (lambda m: m['routes'].pop(),
                       lambda m: m['routes'].__setitem__(1, copy.deepcopy(m['routes'][0])),
                       lambda m: m['routes'][0].update(part='unbound')):
            self.manifest = copy.deepcopy(original)
            mutate(self.manifest)
            self.save_manifest()
            with self.subTest(routes=self.manifest['routes']), self.assertRaises(ValueError):
                c.combine(self.path)

    def test_bound_config_hash_cannot_change(self):
        self.change_config('retry', lambda value: value.update(experiment='changed'), rebind=False)
        with self.assertRaisesRegex(ValueError, 'Bound config changed'):
            c.combine(self.path)

    def test_incompatible_horizon_rejected(self):
        self.change_config('retry', lambda value: value.update(final_iterations=30000,
            final_checkpoint_iterations=value['final_checkpoint_iterations']+[30000]))
        with self.assertRaisesRegex(ValueError, 'Incompatible campaign field: final_iterations'):
            c.combine(self.path)

    def test_incompatible_input_dimensions_rejected(self):
        self.change_config('retry', lambda value: value['instances']['bmmc6224'].update(n=6225))
        with self.assertRaisesRegex(ValueError, 'Incompatible frozen input'):
            c.combine(self.path)

    def test_input_array_hashes_must_agree_across_parts(self):
        configs = {key: c.reporter._contract.load_config(part['config'])
                   for key, part in self.manifest['parts'].items()}
        configs['retry']['instances']['bmmc6224']['array_files_sha256']['A.npy'] = 'a'*64
        with self.assertRaisesRegex(ValueError, 'Incompatible frozen input'):
            c.compatible_config(configs)

    def test_tolerances_cannot_be_retuned(self):
        self.change_config('retry', lambda value: value['runner'].update(inner_tol=1e-6))
        with self.assertRaisesRegex(ValueError, 'tolerances'):
            c.combine(self.path)

    def test_selected_candidates_remain_bound_to_frozen_selection_evidence(self):
        self.change_config('retry', lambda value: value['selected_candidates']['bmmc6224'].update(
            {'published-msbpg': value['methods']['published-msbpg'][1]}))
        with self.assertRaisesRegex(ValueError, 'Selected candidate differs from frozen selection evidence'):
            c.combine(self.path)


if __name__ == '__main__':
    unittest.main()
