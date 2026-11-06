"""Actual trajectories and independent formula checks for read-only diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT.parent/'shareseq_25000_20260930'))
from direction_diagnostics import direction_diagnostics
from trajectory_parity import compare_trajectories
from sharded_gw import RowShardedGW


class NumericalParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(prefix='probe8192_cpu_')
        cls.base=Path(cls.temporary.name);cls.data=cls.base/'data';cls.data.mkdir()
        rng=np.random.default_rng(20261009);arrays={}
        for key,d in (('A',2),('B',3)):
            X=rng.normal(size=(8,d));D=np.linalg.norm(X[:,None]-X[None,:],axis=2)
            arrays[key]=(D/D.max()).astype('float32')
        arrays.update(a=np.full(8,1/8),b=np.full(8,1/8))
        for key,value in arrays.items():np.save(cls.data/(key+'.npy'),value,allow_pickle=False)
        hashes={key+'.npy':hashlib.sha256((cls.data/(key+'.npy')).read_bytes()).hexdigest() for key in arrays}
        (cls.data/'manifest.json').write_text(json.dumps(dict(array_files_sha256=hashes)))
        cls.runs={}
        for name,q,script,enabled in (
            ('q2048_legacy',2048,ROOT.parent/'batch_size_20261008/run_batch.py',None),
            ('q2048_enabled',2048,ROOT/'run_diagnostic.py','enabled'),
            ('q8192_disabled',8192,ROOT/'run_diagnostic.py','disabled'),
            ('q8192_enabled',8192,ROOT/'run_diagnostic.py','enabled')):
            out=cls.base/name
            command=cls.command(script,out,q)
            if enabled is not None:command+=['--direction-diagnostics',enabled]
            result=subprocess.run(command,env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1'),
                                  capture_output=True,text=True,timeout=60)
            if result.returncode:raise AssertionError(f'{name} failed:\n{result.stdout}\n{result.stderr}')
            cls.runs[name]=json.loads((out/'metadata.json').read_text())

    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    @classmethod
    def command(cls,script,out,q):
        return [sys.executable,str(script),'--instance',str(cls.data),'--out',str(out),
                '--method','smoothing-fw','--device','cpu','--iterations','5','--seed','710',
                '--init-seed','10710','--output-seed','20710','--batch',str(q),'--save-plans','none',
                '--gamma0','1','--rho0','.3','--mu-scale','10','--checkpoint-iterations','1,2,5']

    def test_q2048_matches_prior_adapter(self):
        self.assertEqual(compare_trajectories(self.runs['q2048_legacy'],self.runs['q2048_enabled'])['status'],'passed')

    def test_q8192_hook_does_not_change_actual_updates_or_rng(self):
        self.assertEqual(compare_trajectories(self.runs['q8192_disabled'],self.runs['q8192_enabled'])['status'],'passed')
        value=self.runs['q8192_enabled']
        self.assertEqual(value['counters']['sketch_probes'],8192*5)
        self.assertEqual(value['counters']['stochastic_gradient_calls'],5)
        self.assertEqual(value['counters']['initialization_oracle_calls'],1)

    def test_preupdate_index_schedule_and_terminal_semantics(self):
        value=self.runs['q8192_enabled'];trace=value['direction_trace']
        self.assertEqual([r['iteration'] for r in trace],[0,1,2])
        self.assertNotIn(5,[r['iteration'] for r in trace])
        self.assertIn(5,[r['iteration'] for r in value['trace']])
        for row in trace:
            k=row['iteration']
            self.assertEqual((row['plan_index'],row['momentum_index']),(k,k))
            self.assertEqual(row['gamma'],1/(k+1)**.75)
            self.assertEqual(row['mu'],10/8/(k+1)**.25)
            self.assertEqual(row['rho_scheduled'],.3/(k+1)**.5)
            self.assertEqual(row['rho_effective'],1. if k==0 else .3/(k+1)**.5)
            self.assertEqual(row['full_gradient_dtype'],'float64')
        self.assertEqual(trace[0]['raw_structure_l1'],0.)
        self.assertEqual(trace[0]['penalty_value'],0.)
        # At k1 the optimizer has actually taken its first gamma=1 row-FW step.
        self.assertGreater(trace[1]['residuals']['col_l1'],0.)
        self.assertEqual(trace[1]['residuals'],next(r for r in value['trace'] if r['iteration']==1)['raw_residuals'])

    def test_observed_timing_is_separate_and_default_output_remains_clean(self):
        enabled,disabled=self.runs['q8192_enabled'],self.runs['q8192_disabled']
        self.assertGreater(enabled['timing']['direction_diagnostic_seconds'],0.)
        self.assertNotIn('direction_trace',disabled)
        self.assertNotIn('direction_diagnostic_seconds',disabled['timing'])
        for rows in (enabled['trace'],enabled['direction_trace']):
            times=[r['actual_end_to_end_seconds'] for r in rows]
            self.assertEqual(times,sorted(times))
            self.assertTrue(all(r['actual_end_to_end_seconds']>=r['actual_solver_seconds'] for r in rows))
            self.assertLessEqual(max(times),enabled['timing']['total_elapsed_seconds'])
        self.assertTrue(all('actual_end_to_end_seconds' not in r for r in disabled['trace']))

    def test_wrong_batch_or_smoothing_setting_rejected(self):
        for name,q,mu in (('q1024',1024,'10'),('mu100',8192,'100')):
            command=self.command(ROOT/'run_diagnostic.py',self.base/name,q)
            command[command.index('--mu-scale')+1]=mu
            result=subprocess.run(command,capture_output=True,text=True,timeout=60)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse((self.base/name/'metadata.json').exists())


class DirectionFormulaTests(unittest.TestCase):
    def setUp(self):
        A=torch.tensor([[0.,1.,4.],[1.,0.,3.],[4.,3.,0.]])/4
        B=torch.tensor([[0.,2.,3.],[2.,0.,1.],[3.,1.,0.]])/3
        a=torch.full((3,),1/3)
        self.problem=RowShardedGW(A,B,a,a.clone(),exact_block_rows=2)
        self.P=torch.tensor([[.25,.03,1/3-.28],[.2,.04,1/3-.24],[.2,.03,1/3-.23]])
        self.d=torch.tensor([[0.,4.,6.],[3.,0.,1.],[2.,1.,0.]])*.01
        self.args=argparse.Namespace(gamma0=1.,rho0=.3,mu_scale=.01)

    def test_metrics_against_independent_four_index_autodiff_and_numpy(self):
        p,d,args=self.P,self.d,self.args;k=5
        result=direction_diagnostics(self.problem,p,d,k,args)
        A,B=self.problem.A.double(),self.problem.B.double()
        variable=p.double().requires_grad_()
        cost=(A[:,None,:,None]-B[None,:,None,:]).square()
        objective=torch.einsum('ijkl,ij,kl->',cost,variable,variable)
        g=torch.autograd.grad(objective,variable)[0].detach().numpy()
        g-=g.mean(axis=1,keepdims=True)
        dn=d.double().numpy();dc=dn-dn.mean(axis=1,keepdims=True)
        error=np.linalg.norm(dc-g);gnorm=np.linalg.norm(g);dnorm=np.linalg.norm(dc)
        mu=args.mu_scale/3/(k+1)**.25
        column_error=p.sum(0)-self.problem.b
        hv=(column_error/mu).double().numpy();hnorm=np.sqrt(3)*np.linalg.norm(hv-hv.mean())
        self.assertAlmostEqual(result['row_centered_full_gradient_norm'],gnorm,places=12)
        self.assertAlmostEqual(result['row_centered_momentum_norm'],dnorm,places=12)
        self.assertAlmostEqual(result['row_centered_momentum_error_norm'],error,places=12)
        self.assertAlmostEqual(result['row_centered_penalty_norm'],hnorm,places=12)
        self.assertAlmostEqual(result['row_centered_momentum_relative_error'],error/gnorm,places=12)
        self.assertAlmostEqual(result['penalty_to_momentum_ratio'],hnorm/dnorm,places=10)
        self.assertAlmostEqual(result['penalty_value'],np.linalg.norm(column_error.double().numpy())**2/(2*mu),places=12)
        self.assertEqual(result['lmo_with_penalty']['selected_column_count'],1)
        self.assertEqual(result['lmo_with_penalty']['largest_row_fraction'],1.)
        self.assertEqual(result['lmo_without_penalty']['selected_column_count'],3)
        self.assertAlmostEqual(result['lmo_agreement_with_without_penalty'],1/3)
        pd=p.double().numpy();an=self.problem.a.double().numpy()
        self.assertAlmostEqual(result['raw_structure_l1'],np.abs(pd-an[:,None]*pd.sum(0)[None,:]).sum(),places=14)

    def test_inputs_and_global_rng_are_untouched(self):
        tensors=(self.P,self.d,self.problem.A,self.problem.B,self.problem.a,self.problem.b)
        originals=[value.clone() for value in tensors];rng=torch.random.get_rng_state().clone()
        direction_diagnostics(self.problem,self.P,self.d,5,self.args)
        self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(tensors,originals)))

    def test_zero_reference_norm_yields_explicit_null_not_fabricated_ratio(self):
        d=torch.zeros_like(self.d)
        result=direction_diagnostics(self.problem,self.P,d,0,self.args)
        self.assertIsNone(result['penalty_to_momentum_ratio'])
        zero=RowShardedGW(torch.zeros_like(self.problem.A),torch.zeros_like(self.problem.B),
                          self.problem.a,self.problem.b)
        result=direction_diagnostics(zero,self.P,self.d,0,self.args)
        self.assertEqual(result['row_centered_full_gradient_norm'],0.)
        self.assertIsNone(result['row_centered_momentum_relative_error'])

if __name__=='__main__':unittest.main()
