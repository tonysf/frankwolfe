"""Long-run summaries require complete seed populations and exact checkpoints."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_test_long_horizon_report', ROOT/'report_campaign.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def fixture_config():
    return dict(instances={'tiny': dict(n=8, world_size=1)},
                selected_candidates={'tiny': {method: dict(id='candidate_'+str(i)) for i,method in enumerate(r.METHODS)}},
                final_iterations=20000, final_checkpoint_iterations=[1000,2000,10000,20000],
                final_seeds=[810,811,812], initializations=['product','randomized'])


def metadata(method='smoothing-fw', start='product', seed=810):
    offset = (seed-810)*.1
    trace = [dict(kind=kind, iteration=iteration, rounded_loss=loss+offset,
                  actual_solver_seconds=iteration/1000,
                  time_to_this_feasible_output=iteration/1000+.1,
                  repair_seconds_current=.1)
             for kind,iteration,loss in [('initial',0,1.),('current_diagnostic',1000,.99),
                 ('current_diagnostic',2000,.9),('current_diagnostic',10000,.85),
                 ('current_diagnostic',20000,.75),('last',20000,.70001)]]
    return dict(status='completed',method_key=method,
                config=dict(method=method,initialization=start,seed=seed,iterations=20000,dtype='float32'),
                counters=dict(completed_updates=20000,stochastic_gradient_calls=20000,
                              sketch_probes=20000*128,inner_iterations_total=0,certificate_checks=0),
                final_float64_validation=dict(dtype='float64',exact_objective=True,
                    plans={'last_rounded':dict(loss=.7+offset)}),
                selected_update_statistic=dict(normalized_displacement_squared=.123),
                timing=dict(solver_seconds=20.),trace=trace)


class ResultTests(unittest.TestCase):
    def test_endpoint_uses_final_fp64_but_2000_uses_its_current_diagnostic(self):
        result = r.extract_result(metadata(), fixture_config(), 'smoothing-fw', 'product', 810)
        points = {point['iteration']:point for point in result['trajectory']}
        self.assertEqual(len(points), 5)
        self.assertEqual(points[2000]['kind'], 'current_diagnostic')
        self.assertEqual(points[2000]['rounded_loss'], .9)
        self.assertEqual(points[2000]['precision'], 'float32')
        self.assertEqual(points[20000]['kind'], 'last')
        self.assertEqual(points[20000]['rounded_loss'], .7)
        self.assertEqual(points[20000]['precision'], 'float64')

    def test_missing_exact_checkpoint_cannot_use_nearest_saved_iterate(self):
        value = metadata()
        value['trace'][2]['iteration'] = 2001
        with self.assertRaisesRegex(ValueError, 'Missing prescribed.*2000'):
            r.extract_result(value, fixture_config(), 'smoothing-fw', 'product', 810)

    def test_final_record_at_2000_is_invalid_for_a_20000_update_run(self):
        value = metadata()
        value['trace'][2]['kind'] = 'last'
        with self.assertRaisesRegex(ValueError, 'Final diagnostic before final horizon'):
            r.extract_result(value, fixture_config(), 'smoothing-fw', 'product', 810)

    def test_duplicate_current_checkpoint_is_not_best_value_selected(self):
        value = metadata()
        duplicate = copy.deepcopy(value['trace'][2])
        duplicate['rounded_loss'] = 0.
        value['trace'].insert(3, duplicate)
        with self.assertRaisesRegex(ValueError, 'Duplicate current iterate'):
            r.extract_result(value, fixture_config(), 'smoothing-fw', 'product', 810)

    def test_wrong_method_seed_and_horizon_rejected(self):
        for key,value in [('iterations',2000),('seed',811),('method','published-smd')]:
            changed = metadata()
            changed['config'][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'mismatch'):
                r.extract_result(changed, fixture_config(), 'smoothing-fw', 'product', 810)


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='long_horizon_report_')
        self.root = Path(self.temp.name)
        self.out = self.root/'results'
        self.config = fixture_config()
        self.config_path = self.root/'campaign.json'
        self.config_path.write_text(json.dumps(self.config))
        self.config_patch = mock.patch.object(r._contract, 'load_config', return_value=self.config)
        self.validation_patch = mock.patch.object(r._contract, 'validate_run', side_effect=self.validate)
        self.config_patch.start()
        self.validation_patch.start()

    def tearDown(self):
        self.validation_patch.stop()
        self.config_patch.stop()
        self.temp.cleanup()

    def folder(self, method='smoothing-fw', start='product', seed=810):
        return r._contract.run_directory(self.out, 'tiny', method, start, seed)

    def validate(self, config, source, prepared, out, dataset, method, start, seed):
        folder = r._contract.run_directory(out,dataset,method,start,seed)
        return dict(metadata=json.loads((folder/'output'/'metadata.json').read_text()))

    def complete(self):
        for method in r.METHODS:
            for start in self.config['initializations']:
                for seed in self.config['final_seeds']:
                    folder = self.folder(method,start,seed)
                    (folder/'output').mkdir(parents=True)
                    (folder/'output'/'metadata.json').write_text(json.dumps(metadata(method,start,seed)))
                    (folder/'status.json').write_text(json.dumps(dict(status='completed')))

    def audit(self):
        return r.audit(self.config_path,self.root/'source',self.root/'prepared',self.out)

    def test_missing_campaign_has_no_summary_values(self):
        report = self.audit()
        self.assertEqual(report['status'], 'incomplete')
        self.assertEqual(report['task_counts']['missing'], 18)
        self.assertTrue(all(row['statistics'] is None for row in report['final_summaries']))
        self.assertTrue(all(row['median'] is None for row in report['trajectory_summaries']))
        r.write_report(report,self.out,no_plots=True)
        self.assertTrue((self.out/'median_loss_iterations.csv').is_file())
        self.assertEqual(r.read(self.out/'report.json')['task_counts']['missing'], 18)

    def test_full_campaign_medians_iqr_and_paired_progress(self):
        self.complete()
        report = self.audit()
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(report['task_counts']['completed'], 18)
        self.assertEqual(len(report['trajectory_summaries']), 30)
        final = report['final_summaries'][0]['statistics']['last_rounded_loss']
        self.assertEqual(final['n'], 3)
        self.assertAlmostEqual(final['median'], .8)
        self.assertAlmostEqual(final['q25'], .75)
        self.assertAlmostEqual(final['q75'], .85)
        progress = report['progress_from_reference'][0]
        self.assertEqual(progress['reference_iteration'], 2000)
        self.assertEqual(progress['reference_precision'], 'float32')
        self.assertEqual(progress['final_precision'], 'float64')
        self.assertAlmostEqual(progress['paired_loss_drop']['median'], .2)
        self.assertAlmostEqual(progress['paired_relative_loss_drop_percent']['median'], 20.)
        self.assertIn('not time to equal accuracy', report['timing_scope'])

    def test_missing_seed_prevents_completed_median_at_every_checkpoint(self):
        self.complete()
        folder = self.folder(seed=812)
        (folder/'output'/'metadata.json').unlink()
        (folder/'status.json').unlink()
        report = self.audit()
        self.assertEqual(report['task_counts']['missing'], 1)
        summary = report['final_summaries'][0]
        self.assertEqual(summary['completed_seeds'], [810,811])
        self.assertIsNone(summary['statistics'])
        rows = [row for row in report['trajectory_summaries']
                if row['method']=='smoothing-fw' and row['initialization']=='product']
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(row['n']==2 and row['median'] is None for row in rows))
        self.assertIsNone(report['progress_from_reference'][0]['paired_loss_drop'])

    def test_controller_failure_is_not_overruled_by_metadata(self):
        self.complete()
        (self.folder()/'status.json').write_text(json.dumps(dict(status='failed',error='Hash mismatch')))
        report = self.audit()
        self.assertEqual(report['task_counts']['failed'], 1)
        self.assertEqual(report['runs'][0]['reason'], 'Hash mismatch')
        self.assertIsNone(report['final_summaries'][0]['statistics'])

    def test_controller_validation_error_preserves_invalid_run(self):
        self.complete()
        with mock.patch.object(r._contract,'validate_run',side_effect=ValueError('Frozen config mismatch')):
            report = self.audit()
        self.assertEqual(report['task_counts']['invalid'], 18)
        self.assertTrue(all(row['reason']=='Frozen config mismatch' for row in report['runs']))

    def test_observed_bad_horizon_is_invalid_even_if_outer_receipt_exists(self):
        self.complete()
        path = self.folder()/'output'/'metadata.json'
        value = r.read(path)
        value['config']['iterations'] = 2000
        path.write_text(json.dumps(value))
        report = self.audit()
        self.assertEqual(report['task_counts']['invalid'], 1)
        self.assertIn('iterations', report['runs'][0]['reason'])


class ControllerIntegrationTests(unittest.TestCase):
    def test_actual_controller_receipts_feed_report_without_mocked_validation(self):
        spec = importlib.util.spec_from_file_location('_report_actual_controller_fixture', ROOT/'test_campaign.py')
        fixture_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture_module)
        with tempfile.TemporaryDirectory(prefix='long_report_integration_') as directory:
            fixture = fixture_module.Fixture(directory)
            for method in r.METHODS:
                for index in range(6):
                    fixture_module.c.run_task(fixture.args(method,index), executor=fixture.execute)
            report = r.audit(fixture.path,fixture.source,fixture.prepared,fixture.out)
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(report['task_counts']['completed'], 18)
            self.assertEqual(len(report['trajectory_summaries']), 18)
            self.assertTrue(all(row['statistics']['last_rounded_loss']['median'] == .2
                                for row in report['final_summaries']))
            self.assertTrue(all(row['status']=='incomplete' for row in report['progress_from_reference']))
            # Provenance changes invalidate only affected runs, not the report process.
            source = fixture.source/'extensions/msbpg_20261005/msbpg.py'
            source.write_text(source.read_text()+'\nchanged\n')
            altered = r.audit(fixture.path,fixture.source,fixture.prepared,fixture.out)
            self.assertEqual(altered['task_counts']['invalid'], 6)
            self.assertEqual(altered['task_counts']['completed'], 12)


if __name__ == '__main__':
    unittest.main()
