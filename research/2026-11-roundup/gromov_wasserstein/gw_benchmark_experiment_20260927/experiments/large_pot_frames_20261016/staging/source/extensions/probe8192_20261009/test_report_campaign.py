"""End-to-end receipt/report integration, baseline audit, and missing seeds."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parent


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


r=load('_probe_report_test',ROOT/'report_campaign.py')
fixture=load('_probe_report_fixture',ROOT/'test_campaign.py')


class ReportTests(unittest.TestCase):
    def test_three_real_controller_receipts_produce_dense_summaries_and_figures(self):
        with tempfile.TemporaryDirectory(prefix='probe_report_') as directory:
            f=fixture.Fixture(directory)
            fixture.c.preflight(f.args(),executor=f.execute)
            for index in range(3):fixture.c.run_task(f.args('run',index),executor=f.execute)
            report=r.audit(f.path,f.source,f.prepared,f.out)
            self.assertEqual(report['status'],'completed')
            self.assertEqual(report['task_counts']['completed'],3)
            self.assertEqual(len(report['direction_summaries']),209)
            self.assertEqual(len(report['trajectory_summaries']),210)
            self.assertAlmostEqual(report['direction_summaries'][1]['statistics']['penalty_to_momentum_ratio']['statistics']['median'],1.5)
            self.assertIn('EMA lag',report['direction_scope'])
            r.write_report(report,Path(directory)/'report')
            self.assertEqual(len(report['figures']['files']),6)
            self.assertTrue(all(Path(path).stat().st_size>1000 for path in report['figures']['files']))
            folder=fixture.c.run_directory(f.out,fixture.c.DATASET,8192,712)
            (folder/'output/metadata.json').unlink();(folder/'status.json').unlink()
            report=r.audit(f.path,f.source,f.prepared,f.out)
            self.assertEqual(report['status'],'incomplete')
            self.assertIsNone(report['final_summaries'][0]['statistics'])
            self.assertIsNone(report['direction_summaries'][0]['statistics']['raw_col_l1']['statistics'])

    def test_previous_q2048_baseline_is_hash_bound_and_parameter_matched(self):
        config=r.contract.load_config(ROOT/'campaign.json')
        baseline=r.baseline(config)
        self.assertEqual(baseline['status'],'validated')
        self.assertEqual([row['seed'] for row in baseline['runs']],[710,711,712])
        config['instances'][r.contract.DATASET]['prefactors']['mu_scale']=100.
        with self.assertRaisesRegex(ValueError,'prefactors'):r.baseline(config)


if __name__=='__main__':unittest.main()
