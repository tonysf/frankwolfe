"""Campaign contract tests: failures, provenance, selection and held-out seeds."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import campaign as c


def config():
    return dict(schema_version=1, instances={'tiny': {'n': 8, 'world_size': 1}},
                methods={'smoothing-fw': [dict(id=f'fw{i:02}', gamma0=g, rho0=.3, mu_scale=m)
                                        for i, (g, m) in enumerate((g, m) for g in (.1, .3, 1.) for m in (.1, 1., 10., 100.))],
                         'published-smd': [dict(id=f'smd{i:02}', prefactor_fraction=1/(i+1)) for i in range(12)]},
                development_seeds=[710,711,712], final_seeds=[810,811,812],
                initializations=['product','randomized'], tuning_iterations=2, final_iterations=4,
                tuning_checkpoint_iterations=[1,2], final_checkpoint_iterations=[1,4], checkpoint_seconds=[10,30,100,300],
                runner=dict(dtype='float32',batch=128,inner_tol=1e-7,inner_gap_tol=1e-7,inner_max=10000,
                            check_every=10,certificate_block_rows=256,exact_block_rows=256,progress_every=100),
                run_timeout_seconds=7200, final_marginal_l1_tolerance=100*2**-23)


def metadata(expected, provenance, *, loss=.2, world=1, n=8):
    horizon, smd = expected['iterations'], expected['method'] == 'published-smd'
    plans = {}
    for name in c.PLANS:
        path = Path(expected['out']) / 'rank0' / (name+'.npy')
        persisted = expected['save_plans'] == 'all'
        if persisted:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'tiny-test-placeholder')
        plans[name] = dict(shape=[n,n], dtype='float32', array_sha256='a'*64, array_bytes=n*n*4,
                           persisted=persisted, file=str(path) if persisted else None,
                           file_sha256=c.sha256(path) if persisted else None, bytes=path.stat().st_size if persisted else 0)
    trace = [dict(converged=True, proximal_optimality_certified=True, marginal_l1=5e-8,
                  normalized_primal_dual_gap=1e-10, normalized_gap_upper_bound=2e-10,
                  inner_iterations=10, certificate_checks=1) for _ in range(horizon)] if smd else []
    return dict(status='completed', config=expected, method_key=expected['method'],
                source_file_sha256=provenance['source_file_sha256'],
                instance_manifest_sha256=provenance['instance_manifest_sha256'],
                array_files_sha256=provenance['array_files_sha256'], dimensions=dict(source=n,target=n),
                environment=dict(world_size=world), ranks=[dict(rank=0,saved_plans=plans)],
                counters=dict(completed_updates=horizon,stochastic_gradient_calls=horizon,sketch_probes=128*horizon,
                              inner_stopping_failures=0,initialization_oracle_calls=int(not smd),
                              inner_iterations_total=10*horizon if smd else 0,certificate_checks=horizon if smd else 0),
                final_float64_validation=dict(dtype='float64',exact_objective=True,
                    plans={name:dict(loss=loss,marginal_residuals=dict(both_l1=1e-8)) for name in c.PLANS}),
                output_selection=dict(horizon=horizon,index=0,last_iterate_index=horizon,independent_rng=True,
                                      output_seed=expected['output_seed']),
                timing={key:1. for key in ('preparation_seconds','solver_seconds','repair_seconds','exact_diagnostic_seconds',
                                          'serialization_seconds','final_float64_validation_seconds','total_elapsed_seconds')},
                inner_trace=trace)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='campaign_contract_')
        self.base = Path(self.temp.name)
        self.source, self.prepared, self.out = (self.base/n for n in ('source','prepared','results'))
        self.config = config()
        self.config_path = self.base/'campaign.json'
        c.write_json(self.config_path,self.config)
        for filename in (*c.SOURCE_FILES,'published_stochastic_20261002/campaign.py'):
            path = self.source/'extensions'/filename
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(filename)
        data = self.prepared/'tiny'
        data.mkdir(parents=True)
        c.write_json(data/'manifest.json',dict(array_files_sha256={name:'b'*64 for name in c.ARRAYS}))
        self.identity = c.identity(self.config_path,self.source,self.prepared,'tiny')
        self.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def args(self,stage,index=0):
        return argparse.Namespace(config=str(self.config_path),source=str(self.source),prepared=str(self.prepared),
                                  out=str(self.out),instance='tiny',stage=stage,task_index=index)

    def fake_execute(self,command,log_path,timeout):
        request = c.read_json(log_path.parent/'request.json')
        expected = request['expected_runner_config']
        self.calls.append(expected)
        Path(expected['out']).mkdir(parents=True)
        # Candidate 03 is the predetermined best for both methods. A final
        # seed changes scores but must not feed back into prefactor selection.
        loss = .2+abs(int(request['candidate_id'][-2:])-3)*.01+(expected['seed']%10)*.001
        value = metadata(expected,self.identity,loss=loss)
        c.write_json(Path(expected['out'])/'metadata.json',value)
        log_path.write_text('fake numerical worker completed\n')
        return 0

    def expected(self):
        return c.runner_config(self.config,self.prepared,self.out/'one','tiny','tuning','published-smd',
                               self.config['methods']['published-smd'][0],'product',710)

    def test_configuration_rejects_seed_leakage_and_relaxed_accuracy(self):
        c.validate_config(self.config)
        for modify in (lambda x:x['final_seeds'].__setitem__(0,710),
                       lambda x:x['runner'].__setitem__('inner_gap_tol',1e-6),
                       lambda x:x['methods']['published-smd'][0].__setitem__('prefactor_fraction',1.1)):
            altered=copy.deepcopy(self.config); modify(altered)
            with self.assertRaises(ValueError):c.validate_config(altered)

    def test_explicit_runner_arguments_match_actual_parser(self):
        from run_published import parser
        expected=self.expected()
        command=c.runner_command(ROOT.parent.parent,1,expected)
        args=parser().parse_args(command[6:])
        self.assertEqual(vars(args),expected)

    def test_grid_is_balanced_and_seed_streams_are_matched(self):
        runs=[r for i in range(12) for r in c.task_runs(self.config,'tiny','tuning',i)]
        self.assertEqual(len(runs),144)
        self.assertEqual(len({(m,can['id'],start,seed) for m,can,start,seed in runs}),144)
        for method,candidate,start,seed in runs:
            expected=c.runner_config(self.config,self.prepared,'unused','tiny','tuning',method,candidate,start,seed)
            self.assertEqual((expected['init_seed'],expected['output_seed']),(seed+10000,seed+20000))
            self.assertEqual(expected['save_plans'],'none')
        self.assertNotEqual(c.task_runs(self.config,'tiny','tuning',0)[0][0],
                            c.task_runs(self.config,'tiny','tuning',1)[0][0])

    def test_metadata_rejects_material_tampering(self):
        expected=self.expected()
        good=metadata(expected,self.identity)
        c.validate_metadata(good,expected,self.identity,self.config['instances']['tiny'],100*2**-23)
        mutations=(lambda x:x['config'].__setitem__('iterations',200),
                   lambda x:x['array_files_sha256'].__setitem__('A.npy','f'*64),
                   lambda x:x['source_file_sha256'].__setitem__(next(iter(x['source_file_sha256'])),'f'*64),
                   lambda x:x['counters'].__setitem__('stochastic_gradient_calls',99),
                   lambda x:x['inner_trace'][0].__setitem__('normalized_gap_upper_bound',1.01e-7),
                   lambda x:x['inner_trace'][0].__setitem__('marginal_l1',1.01e-7),
                   lambda x:x['final_float64_validation']['plans']['last_rounded'].__setitem__('loss',float('nan')),
                   lambda x:x['final_float64_validation']['plans']['last_rounded']['marginal_residuals'].__setitem__('both_l1',1e-3),
                   lambda x:x['environment'].__setitem__('world_size',4),
                   lambda x:x['output_selection'].__setitem__('index',2))
        for mutate in mutations:
            value=copy.deepcopy(good);mutate(value)
            with self.assertRaises(ValueError):
                c.validate_metadata(value,expected,self.identity,self.config['instances']['tiny'],100*2**-23)

    def test_complete_fake_campaign_selects_then_uses_held_out_seeds(self):
        for index in range(12):
            result=c.run_task(self.args('tuning',index),executor=self.fake_execute)
            self.assertEqual(result['completed_runs'],12)
        selected=c.run_task(self.args('select'),executor=self.fake_execute)
        self.assertEqual(selected['selected'],{'smoothing-fw':'fw03','published-smd':'smd03'})
        self.assertEqual(len(self.calls),144)
        for index in range(6):
            result=c.run_task(self.args('final',index),executor=self.fake_execute)
            self.assertEqual(result['completed_runs'],2)
        finals=self.calls[144:]
        self.assertEqual({row['seed'] for row in finals},{810,811,812})
        self.assertEqual(len(finals),12)
        self.assertTrue(all(row['save_plans']=='all' and row['iterations']==4 for row in finals))
        # All completion records remain immutable on accidental resubmission.
        prior=(self.out/'tiny/tasks/final/task_000/status.json').read_bytes()
        with self.assertRaises(FileExistsError):c.run_task(self.args('final',0),executor=self.fake_execute)
        self.assertEqual((self.out/'tiny/tasks/final/task_000/status.json').read_bytes(),prior)
        chosen = self.out/'tiny/tuning/fw03/product/seed710/output/metadata.json'
        chosen.write_text(chosen.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'evidence changed'):
            c.load_selection(self.config,self.identity,self.out,'tiny')
        rebuilt=c.collect_selection(self.config,self.identity,self.source,self.prepared,self.out,'tiny')
        self.assertEqual(rebuilt['selected']['smoothing-fw'],'fw02')
        bad=next(row for row in rebuilt['scores']['smoothing-fw'] if row['candidate_id']=='fw03')
        self.assertFalse(bad['valid'])
        self.assertEqual(len(bad['cells']),5)

    def test_timeout_terminates_rank_group_even_if_launcher_exits_first(self):
        process=mock.Mock(pid=45678)
        process.wait.side_effect=[c.subprocess.TimeoutExpired(['worker'],.01),0,0]
        with mock.patch.object(c.subprocess,'Popen',return_value=process) as popen, \
             mock.patch.object(c.os,'killpg') as killpg:
            with self.assertRaises(c.subprocess.TimeoutExpired):
                c.execute(['worker'],self.base/'timeout.log',.01)
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        self.assertEqual(killpg.call_args_list,[mock.call(45678,c.signal.SIGTERM),mock.call(45678,c.signal.SIGKILL)])

    def test_subprocess_failure_is_preserved_and_blocks_selection(self):
        def failed(command,path,timeout):
            path.write_text('intentional worker failure\n');return 17
        with self.assertRaisesRegex(ValueError,'17'):
            c.run_task(self.args('tuning'),executor=failed)
        run=self.out/'tiny/tuning/fw00/product/seed710'
        self.assertEqual(c.read_json(run/'status.json')['returncode'],17)
        self.assertEqual(c.read_json(self.out/'tiny/tasks/tuning/task_000/status.json')['status'],'failed')
        with self.assertRaises(ValueError):c.run_task(self.args('select'),executor=self.fake_execute)
        self.assertFalse((self.out/'tiny/selection.json').exists())

    def test_selection_tie_and_all_invalid_behavior(self):
        rows=[dict(candidate_id=name,valid=True,median_last_rounded_loss=.2) for name in ('z','a')]
        self.assertEqual(c.choose(rows),'a')
        for row in rows:row['valid']=False
        with self.assertRaisesRegex(ValueError,'No candidate'):c.choose(rows)

    def test_share_storage_policy_retains_both_starts_only_for_first_final_seed(self):
        for start in self.config['initializations']:
            for seed in self.config['final_seeds']:
                expected=c.runner_config(self.config,self.prepared,'out','shareseq_full','final','published-smd',
                                         self.config['methods']['published-smd'][0],start,seed)
                self.assertEqual(expected['save_plans'],'all' if seed==810 else 'none')


if __name__=='__main__':
    unittest.main()
