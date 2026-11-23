"""Reports distinguish frozen references, new failures and the MSBPG statistic."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_msbpg_test_fixtures',ROOT/'test_campaign.py')
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)
spec=importlib.util.spec_from_file_location('_msbpg_report',ROOT/'report_campaign.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
c=fixtures.c


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='msbpg_report_');self.f=fixtures.Fixture(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_missing_new_campaign_keeps_completed_references_but_no_claim(self):
        f=self.f;report=r.audit(f.config_path,f.out)
        self.assertEqual(report['status'],'incomplete')
        self.assertEqual(report['task_counts']['missing'],78)
        frozen=[x for x in report['runs'] if x.get('frozen_reference')]
        self.assertEqual(len(frozen),12);self.assertTrue(all(x['status']=='completed' for x in frozen))
        self.assertTrue(all(x['status']=='incomplete' for x in report['final_comparisons']))
        cli=subprocess.run([sys.executable,str(ROOT/'report_campaign.py'),'--config',str(f.config_path),
                            '--out',str(f.out),'--no-plots'],capture_output=True,text=True,timeout=15)
        self.assertEqual(cli.returncode,0,cli.stderr);self.assertTrue((f.out/'report.csv').exists())

    def test_full_report_uses_three_seed_median_and_separate_theorem_statistic(self):
        f=self.f;f.complete();report=r.audit(f.config_path,f.out)
        self.assertEqual(report['status'],'completed');self.assertEqual(report['task_counts']['completed'],78)
        self.assertEqual(len(report['runs']),90)
        summaries=[x for x in report['final_summaries'] if x['method']=='published-msbpg']
        self.assertEqual(len(summaries),2)
        for summary in summaries:
            stats=summary['statistics'];self.assertAlmostEqual(stats['last_rounded_loss']['median'],.201)
            self.assertAlmostEqual(stats['sampled_normalized_displacement_squared']['median'],.402)
            self.assertAlmostEqual(stats['selected_raw_loss']['median'],100.201)
        for row in report['final_comparisons']:
            self.assertEqual(row['status'],'completed')
            self.assertAlmostEqual(row['msbpg_minus_frames_median_loss'],-.299)
            self.assertEqual(row['ratio_msbpg_to_frames_solver_time'],.5)
            self.assertIn('not time to equal accuracy',row['timing_scope'])
        trajectory=next(x['trajectory'] for x in report['runs'] if x['phase']=='final' and x['method']=='published-msbpg')
        self.assertEqual(trajectory[1]['rounded_loss'],.001)
        self.assertGreater(trajectory[-1]['rounded_loss'],trajectory[1]['rounded_loss'])

    def test_bad_coupling_rejected_even_if_runner_claims_theorem_passes(self):
        f=self.f;f.complete();path=f.out/'tiny/final/msbpg03/product/seed810/output/metadata.json'
        metadata=c.read_json(path);metadata['inner_trace'][1]['alpha']*=2;c.write_json(path,metadata)
        status=path.parent.parent/'status.json';receipt=c.read_json(status);receipt['metadata_sha256']=c.sha256(path);c.write_json(status,receipt)
        report=r.audit(f.config_path,f.out)
        bad=next(x for x in report['runs'] if x['metadata_file']==str(path))
        self.assertEqual(bad['status'],'invalid');self.assertIn('Theorem parameter mismatch',bad['reason'])
        summary=next(x for x in report['final_summaries'] if x['method']=='published-msbpg' and x['initialization']=='product')
        self.assertIsNone(summary['statistics'])
        self.assertEqual(next(x for x in report['final_comparisons'] if x['initialization']=='product')['status'],'incomplete')

    def test_controller_failure_is_not_overruled_by_runner_metadata(self):
        f=self.f;f.complete();directory=f.out/'tiny/final/msbpg03/randomized/seed812'
        c.write_json(directory/'status.json',dict(status='failed',error='Independently rejected provenance'))
        report=r.audit(f.config_path,f.out);failed=[x for x in report['runs'] if x['status']=='failed']
        self.assertEqual(len(failed),1);self.assertIn('Independently rejected provenance',failed[0]['errors'])

    def test_frozen_baseline_tampering_is_a_hard_reporting_error(self):
        f=self.f;path=f.base/f.config['baseline']['file'];value=c.read_json(path)
        value['runs'][0]['last_rounded_loss']=0.;c.write_json(path,value)
        with self.assertRaisesRegex(ValueError,'artifact hash'):r.audit(f.config_path,f.out)


if __name__=='__main__':unittest.main()
