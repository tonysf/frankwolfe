"""MSBPG campaign contracts: admissibility, immutable prior results, no old reruns."""
from __future__ import annotations
import argparse
import copy
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('_test_msbpg_campaign', ROOT/'campaign.py')
c = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(c)


def metadata(expected, provenance, *, loss=.2, n=8):
    T = expected['iterations']
    plans = {}
    for name in c.PLANS:
        path = Path(expected['out'])/'rank0'/(name+'.npy')
        persisted = expected['save_plans'] == 'all'
        if persisted:
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'tiny-plan-test-placeholder')
        plans[name] = dict(shape=[n,n], dtype='float32', array_sha256='a'*64, array_bytes=n*n*4,
                           persisted=persisted, file=str(path) if persisted else None,
                           file_sha256=c.sha256(path) if persisted else None, bytes=path.stat().st_size if persisted else 0)
    L, M = 12., 1/n
    cap = 1/(2*math.sqrt(2*L*L))
    theta, coupling = expected['momentum_prefactor'], expected['coupling_fraction']*cap
    inner = []
    for k in range(T):
        alpha = coupling*theta/math.sqrt(k+1)
        inner.append(dict(iteration=k, converged=True, certificate_valid=True, proximal_optimality_certified=True,
              marginal_l1=5e-8, normalized_primal_dual_gap=1e-10, normalized_gap_upper_bound=2e-10,
              primal_witness_marginal_l1=1e-14, inner_iterations=10, certificate_checks=1,
              alpha=alpha, eta=alpha/M, proximal_eta=alpha/M,
              beta=None if k==0 else theta/math.sqrt(k), beta_next=theta/math.sqrt(k+1),
              coupling_c=coupling,coupling_fraction=expected['coupling_fraction'],momentum_prefactor=theta,
              theorem_c_upper_bound=cap,alpha_small_step_upper_bound=1/L,
              theorem_c_bound_satisfied=True,alpha_small_step_bound_satisfied=True))
    return dict(status='completed', config=expected, method_key='published-msbpg',
                source_file_sha256=provenance['source_file_sha256'],
                instance_manifest_sha256=provenance['instance_manifest_sha256'],
                instance_manifest=dict(array_files_sha256=provenance['array_files_sha256']),
                array_files_sha256=provenance['array_files_sha256'], dimensions=dict(source=n,target=n),
                environment=dict(world_size=1), ranks=[dict(rank=0,saved_plans=plans)],
                counters=dict(completed_updates=T,stochastic_gradient_calls=T,sketch_probes=128*T,
                              inner_stopping_failures=0,initialization_oracle_calls=1,
                              inner_iterations_total=10*T,certificate_checks=T),
                final_float64_validation=dict(dtype='float64',exact_objective=True,optimizer_state_released=True,
                    plans={name:dict(loss=loss+100 if name.startswith('selected') else loss,
                                    marginal_residuals=dict(both_l1=1e-8,min_entry=0.)) for name in c.PLANS}),
                output_selection=dict(horizon=T,index=0,last_iterate_index=T,independent_rng=True,
                                      output_seed=expected['output_seed'],probability=1/sum((k+1)**-.5 for k in range(T))),
                timing={key:1. for key in ('preparation_seconds','solver_seconds','repair_seconds','exact_diagnostic_seconds',
                                          'serialization_seconds','final_float64_validation_seconds','total_elapsed_seconds')},
                inner_trace=inner,
                theorem_constants=dict(hessian_bound=L,maximum_feasible_entry=M,kernel_scale=M,mu=1.,
                                       relative_smoothness=L,kappa=2*L*L,coupling_c_max=cap,norm='Frobenius',
                                       feasible_divergence='D_phi(Q,P)=M*KL(Q||P)'),
                selected_update_statistic=dict(index=0,alpha=inner[0]['alpha'],proximal_eta=inner[0]['eta'],
                                                normalized_displacement_squared=loss*2),
                trace=[dict(kind=kind,iteration=k,actual_solver_seconds=elapsed,
                            time_to_this_feasible_output=elapsed+.1,repair_seconds_current=.1,rounded_loss=value)
                       for kind,k,elapsed,value in [('initial',0,0.,100.),('current_diagnostic',1,.5,.001),('last',T,1.,loss)]])


class Fixture:
    def __init__(self, base):
        self.base=Path(base); self.source,self.prepared,self.out=(self.base/x for x in ('source','prepared','results'))
        self.config_path=self.base/'campaign.json'
        self.config=c.read_json(ROOT/'campaign.json')
        self.config['instances']={'tiny':dict(n=8,world_size=1)}
        self.config.update(tuning_iterations=2,final_iterations=4,tuning_checkpoint_iterations=[1,2],final_checkpoint_iterations=[1,4])
        for filename in (*c.SOURCE_FILES,'msbpg_20261005/campaign.py'):
            path=self.source/'extensions'/filename; path.parent.mkdir(parents=True,exist_ok=True); path.write_text(filename)
        data=self.prepared/'tiny'; data.mkdir(parents=True)
        c.write_json(data/'manifest.json',dict(array_files_sha256={name:'b'*64 for name in c.ARRAYS}))
        manifest_hash=c.sha256(data/'manifest.json'); hashes=c.read_json(data/'manifest.json')['array_files_sha256']
        self.config['instances']['tiny'].update(array_files_sha256=hashes,instance_manifest_sha256=manifest_hash)
        spec=self.config['baseline']
        baseline=dict(status='completed',origin=dict(report_sha256=spec['source_report_sha256'],
                      config_sha256=spec['source_config_sha256'],source_commit=spec['source_commit']),
                      selections={'tiny':dict(array_files_sha256=hashes,instance_manifest_sha256=manifest_hash,
                                              selected={'smoothing-fw':'fw00','published-smd':'smd00'})},runs=[],final_summaries=[])
        for method, candidate in [('smoothing-fw','fw00'),('published-smd','smd00')]:
            for start in self.config['initializations']:
                for seed in self.config['final_seeds']:
                    baseline['runs'].append(dict(dataset='tiny',method=method,candidate_id=candidate,initialization=start,seed=seed,
                        phase='final',status='completed',completed_updates=4,stochastic_gradient_calls=4,sketch_probes=512,
                        last_rounded_loss=.5,solver_seconds=2.,last_feasible_output_seconds=2.1,trajectory=[]))
                stats={key:dict(n=3,median=value,q25=value,q75=value) for key,value in
                       [('last_rounded_loss',.5),('solver_seconds',2.),('last_feasible_output_seconds',2.1)]}
                baseline['final_summaries'].append(dict(dataset='tiny',method=method,initialization=start,status='completed',statistics=stats))
        c.write_json(self.base/spec['file'],baseline);spec['sha256']=c.sha256(self.base/spec['file'])
        c.write_json(self.config_path,self.config)
        self.identity=c.identity(self.config_path,self.source,self.prepared,'tiny');self.calls=[]

    def args(self,stage,index=0):
        return argparse.Namespace(config=str(self.config_path),source=str(self.source),prepared=str(self.prepared),
                                  out=str(self.out),instance='tiny',stage=stage,task_index=index)

    def execute(self,command,path,timeout):
        request=c.read_json(path.parent/'request.json');expected=request['expected_runner_config'];self.calls.append(expected)
        Path(expected['out']).mkdir(parents=True)
        loss=.2+abs(int(request['candidate_id'][-2:])-3)*.01+(expected['seed']%10)*.001
        c.write_json(Path(expected['out'])/'metadata.json',metadata(expected,self.identity,loss=loss))
        path.write_text('fake numerical worker completed\n');return 0

    def complete(self):
        for index in range(12):c.run_task(self.args('tuning',index),executor=self.execute)
        c.run_task(self.args('select'),executor=self.execute)
        for index in range(6):c.run_task(self.args('final',index),executor=self.execute)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='msbpg_campaign_');self.fixture=Fixture(self.temp.name)
    def tearDown(self):self.temp.cleanup()

    def test_configuration_rejects_seed_leakage_and_theorem_violations(self):
        config=self.fixture.config;c.validate_config(config)
        for modify in (lambda x:x['final_seeds'].__setitem__(0,710),
                       lambda x:x['runner'].__setitem__('inner_gap_tol',1e-6),
                       lambda x:x['methods']['published-msbpg'][0].__setitem__('coupling_fraction',1.1),
                       lambda x:x['methods']['published-msbpg'][0].__setitem__('momentum_prefactor',0),
                       lambda x:x['methods'].__setitem__('smoothing-fw',[])):
            altered=copy.deepcopy(config);modify(altered)
            with self.assertRaises(ValueError):c.validate_config(altered)

    def test_runner_cli_contract_matches_exactly(self):
        import sys
        sys.path.insert(0,str(ROOT))
        from run_msbpg import parser
        f=self.fixture;candidate=f.config['methods']['published-msbpg'][0]
        expected=c.runner_config(f.config,f.prepared,f.out/'cli','tiny','tuning','published-msbpg',candidate,'product',710)
        actual=vars(parser().parse_args(c.runner_command(ROOT.parents[1],1,expected)[6:]))
        self.assertEqual(actual,expected)

    def test_grid_has_only_new_comparator_and_same_seed_streams(self):
        f=self.fixture;runs=[r for i in range(12) for r in c.task_runs(f.config,'tiny','tuning',i)]
        self.assertEqual(len(runs),72);self.assertEqual({r[0] for r in runs},{'published-msbpg'})
        self.assertEqual(len({(x[1]['id'],x[2],x[3]) for x in runs}),72)
        for method,candidate,start,seed in runs:
            actual=c.runner_config(f.config,f.prepared,'unused','tiny','tuning',method,candidate,start,seed)
            self.assertEqual((actual['init_seed'],actual['output_seed']),(seed+10000,seed+20000))
            self.assertEqual(actual['save_plans'],'none')

    def test_baseline_artifact_and_original_data_are_hash_bound(self):
        f=self.fixture
        c.load_baseline(f.config_path)
        path=f.base/f.config['baseline']['file'];original=path.read_bytes();path.write_bytes(original+b' ')
        with self.assertRaisesRegex(ValueError,'artifact hash'):c.load_baseline(f.config_path)
        path.write_bytes(original)
        manifest=f.prepared/'tiny/manifest.json';manifest.write_text(manifest.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'manifest differs'):c.identity(f.config_path,f.source,f.prepared,'tiny')

    def test_independent_metadata_audit_rejects_shift_wrong_kernel_and_tolerance(self):
        f=self.fixture;candidate=f.config['methods']['published-msbpg'][0]
        expected=c.runner_config(f.config,f.prepared,f.out/'one','tiny','tuning','published-msbpg',candidate,'product',710)
        good=metadata(expected,f.identity)
        c.validate_metadata(good,expected,f.identity,f.config['instances']['tiny'],1e-5)
        mutations=(lambda x:x['inner_trace'][0].__setitem__('beta',1.),
                   lambda x:x['inner_trace'][1].__setitem__('beta_next',1.),
                   lambda x:x['inner_trace'][0].__setitem__('eta',x['inner_trace'][0]['alpha']),
                   lambda x:x['theorem_constants'].__setitem__('mu',8.),
                   lambda x:x['selected_update_statistic'].__setitem__('alpha',x['selected_update_statistic']['proximal_eta']),
                   lambda x:x['inner_trace'][0].__setitem__('normalized_gap_upper_bound',1.01e-7),
                   lambda x:x['counters'].__setitem__('stochastic_gradient_calls',99),
                   lambda x:x['array_files_sha256'].__setitem__('A.npy','f'*64))
        for mutate in mutations:
            value=copy.deepcopy(good);mutate(value)
            with self.assertRaises(ValueError):c.validate_metadata(value,expected,f.identity,f.config['instances']['tiny'],1e-5)

    def test_full_controller_selects_development_winner_and_preserves_prior_results(self):
        f=self.fixture;f.complete()
        self.assertEqual(len(f.calls),78)
        selection=c.read_json(f.out/'tiny/selection.json');self.assertEqual(selection['selected'],{'published-msbpg':'msbpg03'})
        self.assertEqual({x['seed'] for x in f.calls[72:]},{810,811,812})
        self.assertEqual({x['method'] for x in f.calls},{'published-msbpg'})
        self.assertTrue(all(x['iterations']==4 for x in f.calls[72:]))
        with self.assertRaises(FileExistsError):c.run_task(f.args('final',0),executor=f.execute)
        chosen=f.out/'tiny/tuning/msbpg03/product/seed710/output/metadata.json';chosen.write_text(chosen.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'evidence changed'):c.load_selection(f.config,f.identity,f.out,'tiny')

    def test_failure_is_preserved_and_blocks_selection(self):
        f=self.fixture
        def fail(command,path,timeout):path.write_text('worker failed\n');return 17
        with self.assertRaisesRegex(ValueError,'17'):c.run_task(f.args('tuning'),executor=fail)
        self.assertEqual(c.read_json(f.out/'tiny/tasks/tuning/task_000/status.json')['status'],'failed')
        with self.assertRaises((ValueError,OSError)):c.run_task(f.args('select'),executor=f.execute)
        self.assertFalse((f.out/'tiny/selection.json').exists())

    def test_real_cpu_runner_metadata_passes_controller_and_report_contracts(self):
        import os
        import subprocess
        import sys
        import numpy as np
        f=self.fixture;f.source=ROOT.parents[1]
        data=f.prepared/'tiny';rng=np.random.default_rng(17)
        x,y=rng.normal(size=(6,3)),rng.normal(size=(6,2))
        arrays=dict(A=np.linalg.norm(x[:,None]-x[None,:],axis=2)/3,
                    B=np.linalg.norm(y[:,None]-y[None,:],axis=2)/3,
                    a=np.full(6,1/6),b=np.full(6,1/6))
        for name,value in arrays.items():np.save(data/(name+'.npy'),value)
        hashes={name:c.sha256(data/name) for name in c.ARRAYS}
        c.write_json(data/'manifest.json',dict(array_files_sha256=hashes))
        manifest=c.sha256(data/'manifest.json')
        f.config['instances']['tiny'].update(n=6,array_files_sha256=hashes,instance_manifest_sha256=manifest)
        path=f.base/f.config['baseline']['file'];baseline=c.read_json(path)
        baseline['selections']['tiny'].update(array_files_sha256=hashes,instance_manifest_sha256=manifest)
        c.write_json(path,baseline);f.config['baseline']['sha256']=c.sha256(path);c.write_json(f.config_path,f.config)
        provenance=c.identity(f.config_path,f.source,f.prepared,'tiny')
        candidate=f.config['methods']['published-msbpg'][3]
        expected=c.runner_config(f.config,f.prepared,f.out/'actual','tiny','tuning','published-msbpg',candidate,'product',710)
        expected['device']='cpu'
        command=[sys.executable,str(ROOT/'run_msbpg.py')]
        for key,value in expected.items():command+=['--'+key.replace('_','-'),str(value)]
        env=dict(os.environ,OMP_NUM_THREADS='1')
        for key in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):env.pop(key,None)
        result=subprocess.run(command,capture_output=True,text=True,env=env,timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        actual=c.read_json(Path(expected['out'])/'metadata.json')
        loss=c.validate_metadata(actual,expected,provenance,f.config['instances']['tiny'],f.config['final_marginal_l1_tolerance'])
        spec=importlib.util.spec_from_file_location('_actual_msbpg_report',ROOT/'report_campaign.py')
        report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
        audited=report.validate_result(actual,f.config,'tiny','tuning',candidate,'product',710)
        self.assertEqual(audited['last_rounded_loss'],loss)
        self.assertEqual(audited['stochastic_gradient_calls'],2)
        self.assertEqual(set(actual['source_file_sha256']),set(c.SOURCE_FILES))

    def test_timeout_terminates_entire_torchrun_process_group(self):
        process=mock.Mock(pid=45678);process.wait.side_effect=[c.subprocess.TimeoutExpired(['worker'],.01),0,0]
        with mock.patch.object(c.subprocess,'Popen',return_value=process),mock.patch.object(c.os,'killpg') as kill:
            with self.assertRaises(c.subprocess.TimeoutExpired):c.execute(['worker'],Path(self.temp.name)/'timeout.log',.01)
        self.assertEqual(kill.call_args_list,[mock.call(45678,c.signal.SIGTERM),mock.call(45678,c.signal.SIGKILL)])

    def test_production_configuration_binds_verified_existing_baselines(self):
        config=c.validate_config(c.read_json(ROOT/'campaign.json'));baseline=c.load_baseline(ROOT/'campaign.json',config)
        self.assertEqual(len(baseline['runs']),36)
        self.assertEqual(config['run_counts'],dict(tuning=216,final=18))
        self.assertEqual({x['world_size'] for x in config['instances'].values()},{1,4})


if __name__=='__main__':unittest.main()
