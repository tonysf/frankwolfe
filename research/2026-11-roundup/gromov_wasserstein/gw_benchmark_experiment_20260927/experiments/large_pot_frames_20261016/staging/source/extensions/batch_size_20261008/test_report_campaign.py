"""Probe-count reports retain failures and compare exact observed checkpoints."""
from __future__ import annotations

import copy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_test_batch_report', ROOT/'report_campaign.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def config():
    return dict(instances={name: dict(n=8, final_iterations=horizon,
                                      final_checkpoint_iterations=[2, horizon])
                           for name, horizon in [('real', 4), ('synthetic', 8)]},
                batches=[128, 1024, 2048], development_seeds=[710, 711, 712],
                runner=dict(dtype='float32'), final_marginal_l1_tolerance=1e-5)


def metadata(dataset='real', q=128, seed=710):
    horizon = config()['instances'][dataset]['final_iterations']
    offset = seed-710
    initial = 1.+offset
    residual = dict(col_l1=.2, col_l2=.1, both_l1=.2)
    feasible = dict(col_l1=1e-8, col_l2=1e-9, both_l1=2e-8)
    trace = [dict(kind=kind, iteration=k, rounded_loss=loss, raw_loss=loss-.05,
                  raw_residuals=residual, rounded_residuals=feasible, repair_l1=.2,
                  actual_solver_seconds=k*.2+.01, repair_seconds_current=.001,
                  time_to_this_feasible_output=k*.2+.011)
             for kind, k, loss in [('initial', 0, initial), ('current_diagnostic', 2, initial*.9),
                                   ('current_diagnostic', horizon, initial*.81),
                                   ('last', horizon, initial*.81001)]]
    memory = {stage: dict(cuda_peak_allocated_bytes=1000, cuda_peak_reserved_bytes=2000)
              for stage in ('optimization_and_diagnostics', 'float64_validation')}
    return dict(status='completed', method_key='smoothing-fw',
                config=dict(method='smoothing-fw', iterations=horizon, batch=q,
                            initialization='product', seed=seed, dtype='float32'),
                counters=dict(completed_updates=horizon, stochastic_gradient_calls=horizon, sketch_probes=horizon*q),
                final_float64_validation=dict(dtype='float64', exact_objective=True, plans={
                    'last_raw': dict(loss=initial*.75, marginal_residuals=residual),
                    'last_rounded': dict(loss=initial*.8, marginal_residuals=feasible)}),
                timing=dict(solver_seconds=horizon*.2+.01, total_elapsed_seconds=horizon*.2+.5),
                ranks=[dict(rank=0, memory=memory)], trace=trace)


class ExtractionTests(unittest.TestCase):
    def extract(self, value):
        return r.extract_result(value, config(), 'real', 128, 710)

    def test_exact_checkpoints_and_independent_fp64_endpoint(self):
        result = self.extract(metadata())
        self.assertEqual([p['iteration'] for p in result['trajectory']], [0, 2, 4])
        final = result['trajectory'][-1]
        self.assertEqual(final['rounded_loss'], .8)
        self.assertEqual(final['raw_loss'], .75)
        self.assertEqual(final['precision'], 'float64')
        self.assertEqual(final['repair_precision'], 'float32')
        self.assertAlmostEqual(result['loss_reduction_percent'], 20.)
        self.assertEqual(result['sketch_probes'], 512)
        self.assertAlmostEqual(final['time_to_this_feasible_output'], .811)

    def test_missing_exact_checkpoint_cannot_be_replaced(self):
        value = metadata()
        value['trace'][1]['iteration'] = 3
        with self.assertRaisesRegex(ValueError, 'Missing prescribed.*2'):
            self.extract(value)

    def test_duplicate_checkpoint_cannot_select_best_loss(self):
        value = metadata()
        value['trace'].insert(2, copy.deepcopy(value['trace'][1]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.extract(value)

    def test_rejects_wrong_batch_work_and_bad_timing(self):
        for change, message in [
            (lambda value: value['config'].update(batch=1024), 'batch'),
            (lambda value: value['counters'].update(sketch_probes=4096), 'budget'),
            (lambda value: value['trace'][1].update(time_to_this_feasible_output=10), 'time')]:
            value = metadata()
            change(value)
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.extract(value)

    def test_fp64_feasibility_cannot_be_replaced_by_float32_trace(self):
        value = metadata()
        value['final_float64_validation']['plans']['last_rounded']['marginal_residuals']['both_l1'] = .1
        with self.assertRaisesRegex(ValueError, 'infeasible'):
            self.extract(value)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='batch_report_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.out = self.root/'results'
        self.config = config()
        self.config_path = self.root/'campaign.json'
        self.config_path.write_text(json.dumps(self.config))
        patcher = mock.patch.object(r._contract, 'load_config', return_value=self.config)
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.object(r._contract, 'validate_run', side_effect=lambda c,s,p,o,d,q,seed:
                                  dict(metadata=r.read(self.folder(d,q,seed)/'output/metadata.json')))
        patcher.start()
        self.addCleanup(patcher.stop)

    def folder(self, dataset='real', q=128, seed=710):
        return r._contract.run_directory(self.out, dataset, q, seed)

    def complete(self):
        for dataset in self.config['instances']:
            for q in self.config['batches']:
                for seed in self.config['development_seeds']:
                    folder = self.folder(dataset, q, seed)
                    (folder/'output').mkdir(parents=True)
                    (folder/'output/metadata.json').write_text(json.dumps(metadata(dataset, q, seed)))
                    (folder/'status.json').write_text(json.dumps(dict(status='completed')))

    def audit(self):
        return r.audit(self.config_path, self.root/'source', self.root/'prepared', self.out)

    def test_missing_campaign_remains_incomplete_and_writes_files(self):
        report = self.audit()
        self.assertEqual(report['task_counts']['missing'], 18)
        self.assertEqual(report['status'], 'incomplete')
        self.assertTrue(all(row['statistics'] is None for row in report['final_summaries']))
        r.write_report(report, self.out, no_plots=True)
        self.assertEqual(r.read(self.out/'report.json')['expected_runs'], 18)
        self.assertTrue((self.out/'observations.csv').is_file())

    def test_cli_preserves_incomplete_report_and_returns_nonzero(self):
        destination = self.root/'report'
        arguments = ['report_campaign.py', '--config', str(self.config_path),
                     '--source', str(self.root/'source'), '--prepared', str(self.root/'prepared'),
                     '--out', str(self.out), '--report-dir', str(destination), '--no-plots']
        with mock.patch('sys.argv', arguments), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(r.main(), 2)
        self.assertEqual(r.read(destination/'report.json')['task_counts']['missing'], 18)
        self.assertFalse((self.out/'report.json').exists())

    def test_complete_groups_use_paired_percentages_and_recorded_time(self):
        self.complete()
        report = self.audit()
        self.assertEqual(report['status'], 'completed')
        self.assertEqual(report['task_counts']['completed'], 18)
        self.assertEqual(len(report['trajectory_summaries']), 18)
        summary = report['final_summaries'][0]['statistics']
        self.assertAlmostEqual(summary['loss_reduction_percent']['median'], 20.)
        self.assertEqual(summary['last_rounded_loss']['median'], 1.6)
        self.assertAlmostEqual(report['trajectory_summaries'][1]['statistics']['actual_solver_seconds']['median'], .41)
        r.write_report(report, self.out, no_plots=True)
        self.assertEqual(len((self.out/'observations.csv').read_text().splitlines()), 55)

    def test_missing_seed_prevents_all_medians_for_group(self):
        self.complete()
        (self.folder(seed=712)/'output/metadata.json').unlink()
        (self.folder(seed=712)/'status.json').unlink()
        report = self.audit()
        self.assertEqual(report['task_counts']['missing'], 1)
        self.assertEqual(report['status'], 'incomplete')
        self.assertEqual(report['final_summaries'][0]['completed_seeds'], [710,711])
        self.assertIsNone(report['final_summaries'][0]['statistics'])
        self.assertTrue(all(row['statistics'] is None for row in report['trajectory_summaries'][:3]))

    def test_failures_and_invalid_provenance_cannot_be_counted_completed(self):
        self.complete()
        (self.folder()/'status.json').write_text(json.dumps(dict(status='failed',error='Source hash mismatch')))
        report = self.audit()
        self.assertEqual(report['task_counts']['failed'], 1)
        self.assertEqual(report['runs'][0]['reason'], 'Source hash mismatch')
        with mock.patch.object(r._contract, 'validate_run', side_effect=ValueError('Frozen provenance mismatch')):
            report = self.audit()
        self.assertEqual(report['task_counts']['failed'], 1)
        self.assertEqual(report['task_counts']['invalid'], 17)

    def test_standalone_figures_for_each_dataset(self):
        self.complete()
        report = self.audit()
        r.write_report(report, self.out)
        self.assertEqual(report['figures']['status'], 'generated')
        self.assertEqual(len(report['figures']['files']), 4)
        self.assertTrue(all(Path(path).stat().st_size > 1000 for path in report['figures']['files']))


class ControllerIntegrationTests(unittest.TestCase):
    def test_actual_controller_receipts_and_source_hashes_feed_report(self):
        spec = importlib.util.spec_from_file_location('_batch_report_controller_fixture', ROOT/'test_campaign.py')
        fixture_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture_module)
        with tempfile.TemporaryDirectory(prefix='batch_report_integration_') as directory:
            fixture = fixture_module.Fixture(directory)

            def execute(command, path, timeout):
                fixture.execute(command, path, timeout)
                target = path.parent/'output/metadata.json'
                value = r.read(target)
                for point in value['trace']:
                    if point['kind'] == 'checkpoint':
                        point['kind'] = 'current_diagnostic'
                    point['time_to_this_feasible_output'] = point['actual_solver_seconds'] + point['repair_seconds_current']
                    for key in ('raw_residuals', 'rounded_residuals'):
                        point[key].update(col_l1=point[key]['both_l1'], col_l2=point[key]['both_l1'])
                for plan in value['final_float64_validation']['plans'].values():
                    plan['marginal_residuals'].update(col_l1=plan['marginal_residuals']['both_l1'], col_l2=plan['marginal_residuals']['both_l1'])
                fixture_module.c.write_json(target, value)
                return 0

            for index in range(9):
                fixture_module.c.run_task(fixture.args(index), executor=execute)
            report = r.audit(fixture.path, fixture.source, fixture.prepared, fixture.out)
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(report['task_counts']['completed'], 9)
            self.assertEqual(len(report['final_summaries']), 3)
            source = fixture.source/'extensions/batch_size_20261008/run_batch.py'
            source.write_text(source.read_text()+'\nchanged\n')
            report = r.audit(fixture.path, fixture.source, fixture.prepared, fixture.out)
            self.assertEqual(report['status'], 'incomplete')
            self.assertEqual(report['task_counts']['invalid'], 9)


if __name__ == '__main__':
    unittest.main()
