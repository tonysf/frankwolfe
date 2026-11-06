"""Audit actual probe budgets, frozen diagnostics and fail-closed launch behavior."""
from __future__ import annotations
import argparse
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_batch_campaign_test', ROOT/'campaign.py')
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
spec = importlib.util.spec_from_file_location('_batch_previous_fixture', ROOT.parent/'msbpg_20261005/test_campaign.py')
previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)


def metadata(expected, provenance):
    expanded = dict(expected, momentum_prefactor=1., coupling_fraction=1.)
    value = previous.metadata(expanded, provenance)
    value.update(config=expected, method_key='smoothing-fw', inner_trace=[],
                 schedule=dict(gamma_exponent=.75, rho_exponent=.5, mu_exponent=.25))
    value['counters'].update(sketch_probes=expected['batch']*expected['iterations'],
                             inner_iterations_total=0, certificate_checks=0)
    memory = dict(cuda_peak_allocated_bytes=100, cuda_peak_reserved_bytes=200, cuda_total_memory_bytes=1000)
    value['ranks'][0]['memory'] = {key:dict(memory) for key in
        ('preparation','optimization_and_diagnostics','float64_validation')}
    value['timing'].update(solver_seconds=.1, total_elapsed_seconds=.2)
    points = {0, expected['iterations'], *map(int, expected['checkpoint_iterations'].split(','))}
    value['trace'] = [dict(kind='last' if k==expected['iterations'] else 'initial' if k==0 else 'checkpoint',
        iteration=k, actual_solver_seconds=k*.001, raw_loss=.15, rounded_loss=.2,
        raw_residuals=dict(both_l1=.3), rounded_residuals=dict(both_l1=1e-8),
        repair_l1=.3, repair_seconds_current=.01) for k in sorted(points)]
    return value


class Fixture:
    def __init__(self, base):
        self.base=Path(base)
        self.source,self.prepared,self.out=(self.base/name for name in ('source','prepared','results'))
        self.path,self.gate=self.base/'campaign.json',self.base/'gate.json'
        self.config=c.read_json(ROOT/'campaign.json')
        for name in (*c.SOURCE_FILES,*c.CONTROLLER_FILES):
            p=self.source/'extensions'/name; p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name)
        d=self.prepared/'tiny';d.mkdir(parents=True)
        hashes={name:'b'*64 for name in c.ARRAYS}
        c.write_json(d/'manifest.json',dict(array_files_sha256=hashes))
        self.config['instances']={'tiny':dict(n=8,world_size=1,array_files_sha256=hashes,
            instance_manifest_sha256=c.sha256(d/'manifest.json'),final_iterations=4,
            final_checkpoint_iterations=[1,4],prefactors=dict(gamma0=1.,rho0=.3,mu_scale=.1),
            run_timeout_seconds=100)}
        self.config['preflight']['iterations']=2
        c.write_json(self.path,self.config)
        c.write_json(self.gate,dict(status='passed',config_sha256=c.sha256(self.path),instances={'tiny':dict(
            provenance=c.identity(self.path,self.source,self.prepared,'tiny'),resource_checks=dict(passed=True))}))
        self.calls=[]

    def args(self,index=0):
        return argparse.Namespace(config=str(self.path),source=str(self.source),prepared=str(self.prepared),
            out=str(self.out),gate=str(self.gate),instance='tiny',task_index=index)

    def execute(self,command,path,timeout):
        req=c.read_json(path.parent/'request.json'); expected=req['expected_runner_config']
        self.calls.append((command,timeout,expected))
        dest=Path(expected['out']);dest.mkdir(parents=True)
        c.write_json(dest/'metadata.json',metadata(expected,req));path.write_text('fixture')
        return 0


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='batch_campaign_');self.f=Fixture(self.tmp.name)
    def tearDown(self): self.tmp.cleanup()

    def test_nine_complete_runs_dispatch_actual_q_and_frozen_prefactors(self):
        f=self.f; config=c.load_config(f.path)
        cells=[c.cell_for_task(config,i) for i in range(9)]
        self.assertEqual(cells,[(q,s) for q in (128,1024,2048) for s in (710,711,712)])
        for i,(q,seed) in enumerate(cells):
            self.assertEqual(c.run_task(f.args(i),executor=f.execute)['status'],'completed')
            run=c.validate_run(config,f.source,f.prepared,f.out,'tiny',q,seed)
            expected=run['request']['expected_runner_config']
            self.assertEqual((expected['batch'],expected['iterations'],expected['gamma0'],expected['rho0'],expected['mu_scale']),
                             (q,4,1.,.3,.1))
            self.assertEqual(expected['init_seed'],seed+10000);self.assertEqual(expected['output_seed'],seed+20000)
            command=run['request']['command']
            self.assertTrue(command[5].endswith('batch_size_20261008/run_batch.py'))
            self.assertEqual(command[command.index('--batch')+1],str(q))
            self.assertEqual(run['metadata']['counters']['sketch_probes'],q*4)
        self.assertEqual(len(f.calls),9)

    def test_legacy_q128_counter_cannot_pass_higher_batch(self):
        f=self.f
        def bad(command,path,timeout):
            f.execute(command,path,timeout);target=path.parent/'output/metadata.json'
            value=c.read_json(target);value['counters']['sketch_probes']=128*4;c.write_json(target,value);return 0
        with self.assertRaisesRegex(ValueError,'Oracle budget'):
            c.run_task(f.args(3),executor=bad)
        self.assertEqual(c.read_json(c.run_directory(f.out,'tiny',1024,710)/'status.json')['status'],'failed')

    def test_failed_worker_preserved_and_not_resumed(self):
        f=self.f
        with self.assertRaisesRegex(ValueError,'exit status 17'):
            c.run_task(f.args(),executor=lambda *_:17)
        directory=c.run_directory(f.out,'tiny',128,710);before=(directory/'status.json').read_bytes()
        with self.assertRaises(FileExistsError): c.run_task(f.args(),executor=f.execute)
        self.assertEqual((directory/'status.json').read_bytes(),before)

    def test_missing_checkpoint_and_schedule_change_are_rejected(self):
        f=self.f
        for index,edit,message in ((0,lambda v:v['trace'].pop(1),'Missing requested'),
              (1,lambda v:v['schedule'].__setitem__('rho_exponent',.75),'schedule exponents')):
            def bad(command,path,timeout):
                f.execute(command,path,timeout);p=path.parent/'output/metadata.json';v=c.read_json(p);edit(v);c.write_json(p,v);return 0
            with self.assertRaisesRegex(ValueError,message):c.run_task(f.args(index),executor=bad)

    def test_source_request_and_metadata_tampering_are_rejected(self):
        f=self.f;c.run_task(f.args(),executor=f.execute);cfg=c.load_config(f.path)
        directory=c.run_directory(f.out,'tiny',128,710)
        for p,message in ((directory/'request.json','Frozen request'),(directory/'output/metadata.json','Metadata changed'),
                         (f.source/'extensions/batch_size_20261008/run_batch.py','Request provenance')):
            original=p.read_bytes();p.write_bytes(original+b' ')
            with self.assertRaisesRegex(ValueError,message):c.validate_run(cfg,f.source,f.prepared,f.out,'tiny',128,710)
            p.write_bytes(original)

    def test_failed_or_stale_gate_blocks_all_execution(self):
        f=self.f;gate=c.read_json(f.gate);gate['status']='failed';c.write_json(f.gate,gate)
        with self.assertRaisesRegex(ValueError,'did not pass'):c.run_task(f.args(),executor=f.execute)
        gate['status']='passed';gate['config_sha256']='a'*64;c.write_json(f.gate,gate)
        with self.assertRaisesRegex(ValueError,'configuration differs'):c.run_task(f.args(),executor=f.execute)
        self.assertFalse(f.calls)

    def test_resource_gate_uses_runtime_and_memory_not_loss(self):
        f=self.f;cfg=c.load_config(f.path)
        expected=c.runner_config(cfg,f.prepared,f.out,'tiny',2048,710,preflight=True)
        value=metadata(expected,c.identity(f.path,f.source,f.prepared,'tiny'))
        value['final_float64_validation']['plans']['last_rounded']['loss']=1.e30
        self.assertTrue(c.resource_checks(cfg,'tiny',value)['passed'])
        value['timing']['solver_seconds']=1000
        with self.assertRaisesRegex(ValueError,'runtime estimate'):c.resource_checks(cfg,'tiny',value)
        value['timing']['solver_seconds']=.1
        value['ranks'][0]['memory']['float64_validation']['cuda_peak_reserved_bytes']=950
        with self.assertRaisesRegex(ValueError,'memory headroom'):c.resource_checks(cfg,'tiny',value)

    def test_preflight_writes_passed_gate_only_after_all_checks(self):
        f=self.f;args=f.args();args.out=str(f.base/'preflight')
        gate=c.preflight(args,executor=f.execute)
        self.assertEqual(gate['status'],'passed');self.assertEqual(set(gate['instances']),{'tiny'})
        self.assertEqual(f.calls[0][2]['batch'],2048)
        self.assertEqual(f.calls[0][2]['iterations'],2)
        cfg=c.load_config(f.path)
        self.assertEqual(c.validate_gate(cfg,f.source,f.prepared,Path(args.out)/'gate.json'),
                         c.sha256(Path(args.out)/'gate.json'))

    def test_invalid_config_and_task_rejected(self):
        for change in (lambda x:x.__setitem__('batches',[128,512,1024]),
             lambda x:x.__setitem__('initializations',['randomized']),
             lambda x:x['instances']['tiny'].__setitem__('instance_manifest_sha256','0'*64),
             lambda x:x['preflight'].__setitem__('runtime_safety_factor',1.1)):
            cfg=copy.deepcopy(self.f.config);change(cfg)
            with self.assertRaises(ValueError):c.validate_config(cfg)
        for task in (-1,9,True):
            with self.assertRaises(ValueError):c.cell_for_task(self.f.config,task)

if __name__=='__main__':unittest.main()
