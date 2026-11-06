"""Real CPU trajectories verify batch adapter parity and actual oracle budgets."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent


class NumericalAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(prefix='batch_adapter_cpu_')
        cls.base=Path(cls.temporary.name);cls.data=cls.base/'data';cls.data.mkdir()
        rng=np.random.default_rng(20261008)
        arrays={}
        for key,d in (('A',2),('B',3)):
            X=rng.normal(size=(8,d));D=np.linalg.norm(X[:,None]-X[None,:],axis=2)
            arrays[key]=(D/D.max()).astype('float32')
        arrays.update(a=np.full(8,1/8),b=np.full(8,1/8))
        for key,value in arrays.items():np.save(cls.data/(key+'.npy'),value,allow_pickle=False)
        hashes={key+'.npy':hashlib.sha256((cls.data/(key+'.npy')).read_bytes()).hexdigest() for key in arrays}
        (cls.data/'manifest.json').write_text(json.dumps(dict(array_files_sha256=hashes)))
        cls.runs={}
        for name,script,q in (('original128',ROOT.parent/'published_stochastic_20261002/run_published.py',128),
                             ('adapter128',ROOT/'run_batch.py',128),('adapter1024',ROOT/'run_batch.py',1024),
                             ('adapter2048',ROOT/'run_batch.py',2048)):
            out=cls.base/name
            command=cls.command(script,out,q)
            result=subprocess.run(command,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),
                                  capture_output=True,text=True,timeout=60)
            if result.returncode:
                raise AssertionError(f'{name} failed:\n{result.stdout}\n{result.stderr}')
            cls.runs[name]=json.loads((out/'metadata.json').read_text())

    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    @classmethod
    def command(cls,script,out,q):
        return [sys.executable,str(script),'--instance',str(cls.data),'--out',str(out),
                '--method','smoothing-fw','--device','cpu','--iterations','4','--seed','710',
                '--init-seed','10710','--output-seed','20710','--batch',str(q),'--save-plans','none',
                '--gamma0','1','--rho0','.3','--mu-scale','.1','--checkpoint-iterations','1,4']

    def test_q128_adapter_has_identical_trajectory_and_outputs(self):
        original,adapter=self.runs['original128'],self.runs['adapter128']
        self.assertEqual(original['counters'],adapter['counters'])
        self.assertEqual(original['final_float64_validation'],adapter['final_float64_validation'])
        self.assertEqual(original['output_selection'],adapter['output_selection'])
        self.assertEqual(original['ranks'][0]['initial_shard_sha256'],adapter['ranks'][0]['initial_shard_sha256'])
        self.assertEqual(original['ranks'][0]['saved_plans'],adapter['ranks'][0]['saved_plans'])
        for a,b in zip(original['trace'],adapter['trace']):
            for key in ('iteration','raw_loss','rounded_loss','raw_residuals','rounded_residuals','repair_l1'):
                self.assertEqual(a[key],b[key])
        expected=dict(original['source_file_sha256'])
        expected['batch_size_20261008/run_batch.py']=hashlib.sha256((ROOT/'run_batch.py').read_bytes()).hexdigest()
        self.assertEqual(adapter['source_file_sha256'],expected)

    def test_high_q_reaches_real_oracle_and_exact_feasible_validation(self):
        for q in (1024,2048):
            value=self.runs['adapter'+str(q)]
            self.assertEqual(value['config']['batch'],q)
            self.assertEqual(value['counters']['sketch_probes'],q*4)
            self.assertEqual(value['counters']['stochastic_gradient_calls'],4)
            self.assertEqual(value['counters']['completed_updates'],4)
            self.assertEqual(value['counters']['initialization_oracle_calls'],1)
            self.assertEqual(value['final_float64_validation']['dtype'],'float64')
            self.assertTrue(value['final_float64_validation']['exact_objective'])
            for key,record in value['final_float64_validation']['plans'].items():
                self.assertTrue(np.isfinite(record['loss']))
                if key.endswith('_rounded'):
                    self.assertLessEqual(record['marginal_residuals']['both_l1'],100*2**-23)
            self.assertFalse(value['plan_storage']['dense_arrays_persisted'])
        # At least one actual output must depend on batch shape/probe stream.
        outputs={self.runs['adapter'+str(q)]['ranks'][0]['saved_plans']['last_raw']['array_sha256']
                 for q in (128,1024,2048)}
        self.assertGreater(len(outputs),1)

    def test_rejects_invalid_batch_without_running(self):
        for q in (0,-1,512):
            out=self.base/('bad'+str(q))
            result=subprocess.run(self.command(ROOT/'run_batch.py',out,q),capture_output=True,text=True,timeout=60)
            self.assertNotEqual(result.returncode,0)
            failure=json.loads((out/'failure_rank0.json').read_text())
            self.assertIn('Batch must be',failure['error'])
            self.assertFalse((out/'metadata.json').exists())

    def test_other_methods_remain_outside_adapter_scope(self):
        out=self.base/'wrong_method';cmd=self.command(ROOT/'run_batch.py',out,1024)
        cmd[cmd.index('--method')+1]='published-smd'
        result=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('FraMES only',json.loads((out/'failure_rank0.json').read_text())['error'])

if __name__=='__main__':unittest.main()
