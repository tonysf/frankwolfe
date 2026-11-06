"""Reporting tests use small JSON receipts; no framework or scientific compute."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from report_campaign import audit,sha256


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False))


class CampaignReportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='smd_report_')
        self.root=Path(self.temp.name)
        self.out=self.root/'campaign'
        self.config_path=self.root/'config.json'
        self.config=dict(instances={name:dict(n=n,world_size=1) for name,n in (('bmmc6224',2),('snare_full',3),('shareseq_full',4))},
            methods={'smoothing-fw':[dict(id='fw00',gamma0=.1,rho0=.3,mu_scale=.1)],
                     'published-smd':[dict(id='smd00',prefactor_fraction=1.)]},
            development_seeds=[710,711,712],final_seeds=[810,811,812],initializations=['product','randomized'],
            tuning_iterations=2,final_iterations=2,runner=dict(dtype='float32',batch=128,inner_tol=1e-7,inner_gap_tol=1e-7),
            final_marginal_l1_tolerance=1e-5)
        write(self.config_path,self.config)
        self.sources={'published_stochastic_20261002/run_published.py':'a'*64}
        self.inputs={name:{f'{x}.npy':hashlib.sha256((name+x).encode()).hexdigest() for x in ('A','B','a','b')} for name in self.config['instances']}

    def tearDown(self):
        self.temp.cleanup()

    def provenance(self,dataset):
        return dict(config_sha256=sha256(self.config_path),source_file_sha256=self.sources,
                    instance_manifest_sha256=hashlib.sha256(dataset.encode()).hexdigest(),array_files_sha256=self.inputs[dataset])

    def selection(self,dataset):
        write(self.out/dataset/'selection.json',dict(status='completed',**self.provenance(dataset),selected={'smoothing-fw':'fw00','published-smd':'smd00'}))

    def result(self,dataset,method,start,seed,loss,phase='final'):
        candidate=self.config['methods'][method][0]
        output=self.out/dataset/phase/candidate['id']/start/f'seed{seed}'/'output'
        T=self.config[phase+'_iterations']; n=self.config['instances'][dataset]['n']
        config=dict(method=method,iterations=T,seed=seed,output_seed=seed+20000,init_seed=seed+10000,initialization=start,
                    **self.config['runner'],**{k:v for k,v in candidate.items() if k!='id'})
        names=['selected_raw','selected_rounded','last_raw','last_rounded']
        plans={name:dict(loss=100+loss if name.startswith('selected') else loss,
                        marginal_residuals=dict(both_l1=1e-8,min_entry=0.)) for name in names}
        inner=[dict(iteration=k,converged=True,certificate_valid=True,proximal_optimality_certified=True,
                    marginal_l1=1e-8,normalized_gap_upper_bound=1e-9,normalized_primal_dual_gap=1e-10,
                    primal_witness_marginal_l1=1e-14,inner_iterations=10,certificate_checks=1) for k in range(T)] if method=='published-smd' else []
        metadata=dict(status='completed',method_key=method,config=config,dimensions=dict(source=n,target=n),
            environment=dict(world_size=1),instance_manifest={'array_files_sha256':self.inputs[dataset]},**{k:v for k,v in self.provenance(dataset).items() if k!='config_sha256'},
            counters=dict(completed_updates=T,stochastic_gradient_calls=T,sketch_probes=T*128,inner_stopping_failures=0,
                          initialization_oracle_calls=int(method=='smoothing-fw'),inner_iterations_total=10*len(inner),certificate_checks=len(inner)),
            output_selection=dict(index=0,horizon=T,last_iterate_index=T,independent_rng=True,probability=1/sum((k+1)**-.5 for k in range(T))),
            inner_trace=inner,final_float64_validation=dict(dtype='float64',exact_objective=True,optimizer_state_released=True,plans=plans),
            ranks=[dict(rank=0,saved_plans={name:dict(shape=[n,n],array_sha256='b'*64) for name in names})],timing=dict(solver_seconds=2.),
            trace=[dict(kind=kind,iteration=k,actual_solver_seconds=float(k),time_to_this_feasible_output=k+.1,
                        repair_seconds_current=.1,rounded_loss=value) for kind,k,value in [('initial',0,100.),('current_diagnostic',1,.001),('last',2,loss)]])
        write(output/'metadata.json',metadata)
        write(output.parent/'request.json',dict(**self.provenance(dataset),stage=phase,instance=dataset,candidate_id=candidate['id'],expected_runner_config=config))
        write(output.parent/'status.json',dict(status='completed',returncode=0,metadata_sha256=sha256(output/'metadata.json')))
        return output

    def test_missing_campaign_is_incomplete_and_cli_exits_zero(self):
        report=audit(self.config_path,self.out)
        self.assertEqual(report['status'],'incomplete')
        self.assertEqual(report['task_counts']['missing'],72)
        self.assertTrue(all(row['statistics'] is None for row in report['final_summaries']))
        result=subprocess.run([sys.executable,str(ROOT/'report_campaign.py'),'--config',str(self.config_path),'--out',str(self.out),'--no-plots'],capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertTrue((self.out/'report.csv').exists())
        self.assertEqual(json.loads((self.out/'report.json').read_text())['status'],'incomplete')

    def test_three_seed_median_is_not_best_seed_or_checkpoint_and_selected_is_separate(self):
        for dataset in self.config['instances']:
            self.selection(dataset)
            for start in self.config['initializations']:
                for method in self.config['methods']:
                    for seed,loss in zip(self.config['final_seeds'],[1.,5.,9.]):
                        self.result(dataset,method,start,seed,loss)
        report=audit(self.config_path,self.out)
        self.assertEqual(report['task_counts']['completed'],36)
        self.assertEqual(report['task_counts']['missing'],36)
        for row in report['final_summaries']:
            self.assertEqual(row['status'],'completed')
            self.assertEqual(row['statistics']['last_rounded_loss'],dict(n=3,median=5.,q25=3.,q75=7.))
            if row['method']=='published-smd':
                self.assertEqual(row['statistics']['selected_raw_loss']['median'],105.)
            else:
                self.assertNotIn('selected_raw_loss',row['statistics'])
        self.assertEqual(report['runs'][-1]['trajectory'][1]['rounded_loss'],.001)
        for dataset in self.config['instances']:
            for start in self.config['initializations']:
                for method in self.config['methods']:
                    for seed in self.config['development_seeds']:
                        self.result(dataset,method,start,seed,7.,phase='tuning')
        completed=audit(self.config_path,self.out)
        self.assertEqual(completed['status'],'completed')
        self.assertEqual(completed['task_counts']['completed'],72)

    def test_failure_and_invalid_certificate_are_retained_and_do_not_form_summary(self):
        self.selection('bmmc6224')
        first=self.result('bmmc6224','published-smd','product',810,2.)
        second=self.result('bmmc6224','published-smd','product',811,3.)
        metadata=json.loads((second/'metadata.json').read_text())
        metadata['inner_trace'][0]['normalized_gap_upper_bound']=1e-5
        write(second/'metadata.json',metadata)
        write(second.parent/'status.json',dict(status='completed',returncode=0,metadata_sha256=sha256(second/'metadata.json')))
        failed=self.out/'bmmc6224/final/smd00/product/seed812'
        write(failed/'status.json',dict(status='failed',error='Runner time limit reached'))
        report=audit(self.config_path,self.out)
        group=[r for r in report['runs'] if r['dataset']=='bmmc6224' and r['phase']=='final' and r['method']=='published-smd' and r['initialization']=='product']
        self.assertEqual([r['status'] for r in group],['completed','invalid','failed'])
        self.assertIn('Guarded inner gap',group[1]['reason'])
        self.assertIn('Runner time limit reached',group[2]['errors'])
        summary=next(r for r in report['final_summaries'] if r['dataset']=='bmmc6224' and r['method']=='published-smd' and r['initialization']=='product')
        self.assertIsNone(summary['statistics'])
        self.assertEqual(summary['completed_seeds'],[810])

    def test_controller_failure_is_retained_even_when_runner_metadata_exists(self):
        self.selection('bmmc6224')
        output=self.result('bmmc6224','smoothing-fw','product',810,2.)
        write(output.parent/'status.json',dict(status='failed',error='Controller provenance audit rejected result'))
        report=audit(self.config_path,self.out)
        row=next(r for r in report['runs'] if r['metadata_file']==str(output/'metadata.json'))
        self.assertEqual(row['status'],'failed')
        self.assertIn('Controller provenance audit rejected result',row['errors'])

    def test_wrong_dataset_method_and_candidate_parameters_are_rejected(self):
        self.selection('bmmc6224')
        output=self.result('bmmc6224','smoothing-fw','product',810,2.)
        metadata=json.loads((output/'metadata.json').read_text())
        metadata['dimensions']['source']=999
        write(output/'metadata.json',metadata)
        write(output.parent/'status.json',dict(status='completed',returncode=0,metadata_sha256=sha256(output/'metadata.json')))
        report=audit(self.config_path,self.out)
        row=next(r for r in report['runs'] if r['metadata_file']==str(output/'metadata.json'))
        self.assertEqual(row['status'],'invalid')
        self.assertIn('Dataset dimensions',row['reason'])


if __name__=='__main__':
    unittest.main()
