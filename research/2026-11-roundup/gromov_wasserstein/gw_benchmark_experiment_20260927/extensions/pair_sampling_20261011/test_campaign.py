"""CPU contract tests for launch gates; these are not GPU validation receipts."""
from __future__ import annotations
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('pair_campaign_contract',ROOT/'campaign.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)


class CampaignContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='pair_campaign_contract_')
        self.base=Path(self.tmp.name)
        self.path=self.base/'project/extensions/pair_sampling_20261011/campaign.json'
        self.path.parent.mkdir(parents=True)
        self.config=json.loads((ROOT/'campaign.json').read_text())
        for record in self.config['prefactor_provenance'].values():
            target=self.path.parents[2]/record['file'];target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT.parents[1]/record['file'],target)
        shutil.copyfile(ROOT/'theory_audit.json',self.path.parent/'theory_audit.json')
        c.write(self.path,self.config)

    def tearDown(self):self.tmp.cleanup()

    def load(self):return c.load_config(self.path)

    def test_frozen_methods_pairs_starts_and_horizons(self):
        config=self.load();count=0
        for dataset,instance in config['instances'].items():
            for method in c.METHODS:
                for seed in config['seeds']:
                    values=c.arguments(config,self.base/'prepared',self.base/'out',dataset,method,seed,'production')
                    expected=c.expected_config(ROOT.parents[1],values)
                    self.assertEqual(expected['oracle'],'pair-sampling')
                    self.assertEqual(expected['pairs_per_call'],1024)
                    self.assertEqual(expected['iterations'],200000 if dataset=='metric512_s1000' else 20000)
                    self.assertEqual((expected['seed'],expected['init_seed'],expected['output_seed']),(seed,seed+10000,seed+20000))
                    self.assertEqual(expected['initialization'],'product')
                    self.assertEqual(expected['save_plans'],'none')
                    self.assertEqual(expected['inner_tol'],expected['inner_gap_tol'])
                    count+=1
        self.assertEqual(count,36)

    def test_changed_prefactor_or_theory_accuracy_is_rejected(self):
        changes=(lambda x:x['instances']['bmmc6224']['prefactors']['smoothing-fw'].__setitem__('gamma0',.3),
                 lambda x:x['instances']['metric512_s1000']['prefactors']['published-smd'].__setitem__('prefactor_fraction',.3),
                 lambda x:x['instances']['snare_full'].__setitem__('final_iterations',19999),
                 lambda x:x['runner'].__setitem__('inner_gap_tol',1e-6),
                 lambda x:x.__setitem__('pairs_per_call',8192))
        for change in changes:
            value=copy.deepcopy(self.config);change(value);c.write(self.path,value)
            with self.assertRaises(ValueError):self.load()
        c.write(self.path,self.config)
        theory=self.path.parent/'theory_audit.json';theory.write_text(theory.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'audit changed'):self.load()

    def calibration(self):
        return dict(config=dict(iterations=2000),timing=dict(solver_seconds=200.,exact_diagnostic_seconds=3.,repair_seconds=1.,
                    progress_logging_seconds=.1,total_elapsed_seconds=220.),trace=[{}]*6,
                    progress_reporting=dict(completed_updates_recorded=[0,1,1000,2000]),
                    ranks=[dict(memory={name:dict(cuda_peak_reserved_bytes=40,cuda_total_memory_bytes=100)
                                        for name in ('preparation','optimization_and_diagnostics','float64_validation')})])

    def test_runtime_forecast_counts_all_costs_with_margin_and_never_uses_loss(self):
        config=self.load();instance=config['instances']['metric512_s1000'];metadata=self.calibration()
        result=c.resource_checks(config,instance,metadata,dict(process_elapsed_seconds=225.))
        self.assertTrue(result['passed']);self.assertGreater(result['estimated_total_seconds'],20000.)
        self.assertEqual(result['forecast_with_margin_seconds'],2*result['estimated_total_seconds'])
        metadata['some_loss']=1e100
        self.assertEqual(result,c.resource_checks(config,instance,metadata,dict(process_elapsed_seconds=225.)))
        metadata['timing']['solver_seconds']=1000.;metadata['timing']['total_elapsed_seconds']=1020.
        with self.assertRaisesRegex(ValueError,'forecast exceeds'):
            c.resource_checks(config,instance,metadata,dict(process_elapsed_seconds=1025.))

    def test_process_overhead_and_every_rank_memory_can_block_launch(self):
        config=self.load();instance=config['instances']['metric512_s1000'];metadata=self.calibration()
        with self.assertRaisesRegex(ValueError,'forecast exceeds'):
            c.resource_checks(config,instance,metadata,dict(process_elapsed_seconds=60000.))
        bad=copy.deepcopy(metadata['ranks'][0]);bad['memory']['float64_validation']['cuda_peak_reserved_bytes']=91
        metadata['ranks'].append(bad)
        with self.assertRaisesRegex(ValueError,'memory headroom'):
            c.resource_checks(config,instance,metadata,dict(process_elapsed_seconds=225.))

    def test_failed_subprocess_retains_error_and_refuses_reuse(self):
        config=self.load();out=self.base/'results';provenance=dict(config_sha256='x')
        def failure(command,log,timeout):log.write_text('deliberate test failure');return 17
        with mock.patch.object(c,'identity',return_value=provenance),mock.patch.object(c,'expected_config',side_effect=lambda source,values:values):
            with self.assertRaisesRegex(ValueError,'17'):
                c.run_cell(config,ROOT.parents[1],self.base/'prepared',out,'metric512_s1000','smoothing-fw',710,'production',executor=failure)
            target=c.directory(out,'metric512_s1000','smoothing-fw',710)
            status=c.read(target/'status.json');self.assertEqual(status['status'],'failed');self.assertEqual(status['returncode'],17)
            self.assertFalse((target/'output/metadata.json').exists())
            before=(target/'status.json').read_bytes()
            with self.assertRaises(FileExistsError):
                c.run_cell(config,ROOT.parents[1],self.base/'prepared',out,'metric512_s1000','smoothing-fw',710,'production',executor=failure)
            self.assertEqual((target/'status.json').read_bytes(),before)


if __name__=='__main__':unittest.main()
