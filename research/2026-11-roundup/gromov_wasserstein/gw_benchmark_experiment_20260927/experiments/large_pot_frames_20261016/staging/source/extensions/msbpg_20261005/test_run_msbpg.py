"""Small end-to-end tests; subprocess failures and saved outputs are observable."""
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
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from msbpg import select_output_index
from run_msbpg import parser


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dense_gradient(A, B, P, Z):
    r, c = P.sum(1), P.sum(0)
    return (2*(A.square()@r)[:, None] + 2*(B.square()@c)[None, :]
            -2/Z.shape[1] * ((A@(P@(B@Z)))@Z.T + (A@(P@Z))@(B@Z).T))


def dense_prox(P, G, a, b, eta):
    """Independent ordinary positive-matrix scaling, without certificate code."""
    K = P*torch.exp(-eta*G)
    v = torch.ones_like(b)
    for _ in range(20000):
        u = a/(K@v)
        v = b/(K.T@u)
        Q = u[:, None]*K*v[None, :]
        if ((Q.sum(1)-a).abs().sum() + (Q.sum(0)-b).abs().sum()).item() < 1e-14:
            return Q
    raise AssertionError('Independent tiny dense solve failed.')


class RunnerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='published_msbpg_runner_test_')
        cls.base = Path(cls.temporary.name)
        cls.data = cls.base/'data'
        cls.data.mkdir()
        rng = np.random.default_rng(812)
        coordsA, coordsB = rng.normal(size=(6,3)), rng.normal(size=(6,2))
        A = np.linalg.norm(coordsA[:,None,:]-coordsA[None,:,:], axis=2)/3
        B = np.linalg.norm(coordsB[:,None,:]-coordsB[None,:,:], axis=2)/3
        cls.arrays = dict(A=A, B=B, a=np.full(6,1/6), b=np.full(6,1/6))
        for name, value in cls.arrays.items():
            np.save(cls.data/f'{name}.npy', value)
        manifest = {'array_files_sha256': {f'{name}.npy':digest(cls.data/f'{name}.npy') for name in cls.arrays}}
        (cls.data/'manifest.json').write_text(json.dumps(manifest))
        # Choose a noninitial, nonterminal P_R so off-by-one errors are visible.
        cls.output_seed = next(s for s in range(100) if select_output_index(4,generator=torch.Generator().manual_seed(s)) == 2)
        cls.smd_out, cls.smd_metadata = cls.launch('smd', output_seed=cls.output_seed)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def launch(cls, label, method='published-msbpg', output_seed=0, extra=(), expect_success=True):
        out = cls.base/label
        command = [sys.executable, str(ROOT/'run_msbpg.py'), '--instance', str(cls.data),
                   '--out', str(out), '--method', method, '--device', 'cpu', '--dtype', 'float64',
                   '--iterations', '4', '--seed', '314', '--output-seed', str(output_seed),
                   '--initialization', 'randomized', '--init-seed', '123', '--check-every', '1',
                   '--certificate-block-rows', '2', '--exact-block-rows', '2',
                   '--checkpoint-iterations', '1,3', '--coupling-fraction', '.8',
                   '--momentum-prefactor', '.6', *extra]
        env = dict(os.environ, OMP_NUM_THREADS='1')
        for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
            env.pop(key, None)
        result = subprocess.run(command, text=True, capture_output=True, env=env, timeout=90)
        if expect_success and result.returncode:
            raise AssertionError(result.stdout+'\n'+result.stderr)
        if not expect_success:
            if result.returncode == 0:
                raise AssertionError('Failure was silently accepted.')
            return out, result
        return out, json.loads((out/'metadata.json').read_text())

    def independent_trajectory(self, *, wrong_index=False, no_momentum=False):
        A, B, a, b = (torch.tensor(self.arrays[n], dtype=torch.float64) for n in ('A','B','a','b'))
        P = .9*torch.outer(a,b)
        permutation = np.random.default_rng(123).permutation(len(a))
        P[torch.arange(len(a)), torch.as_tensor(permutation)] += .1/len(a)
        trajectory, d = [P.clone()], None
        generator = torch.Generator().manual_seed(314)
        # Enumerated W, independent of the production row-statistics bound.
        W=(A[:,None,:,None]-B[None,:,None,:]).square().reshape(36,36)
        L=2*W.sum(1).max().item()
        for k in range(4):
            Z = (2*torch.randint(0,2,(len(b),128),generator=generator)-1).double()
            G = dense_gradient(A,B,P,Z)
            beta=.6/(k+1 if wrong_index else max(1,k))**.5
            d=G if d is None or no_momentum else (1-beta)*d+beta*G
            alpha=.8/(2*2**.5*L)*.6/(k+1)**.5
            P=dense_prox(P,d,a,b,alpha/min(a.max().item(),b.max().item()))
            trajectory.append(P.clone())
        return trajectory

    def test_late_step_gate_uses_smallest_requested_schedule_on_both_starts(self):
        out=self.base/'late_gate.json'
        command=[sys.executable,str(ROOT/'validate_late_steps.py'),'--instance',str(self.data),
                 '--device','cpu','--out',str(out),'--horizon','2000',
                 '--coupling-fraction','.125','--momentum-prefactor','.1',
                 '--certificate-block-rows','2','--exact-block-rows','2']
        env=dict(os.environ,OMP_NUM_THREADS='1')
        for key in ('RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT'):
            env.pop(key,None)
        result=subprocess.run(command,text=True,capture_output=True,env=env,timeout=90)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        metadata=json.loads(out.read_text())
        self.assertEqual(metadata['status'],'passed')
        self.assertEqual(metadata['iteration'],1999)
        self.assertEqual({r['initialization'] for r in metadata['checks']},{'product','randomized'})
        self.assertIn('no claim to reproduce a late optimizer trajectory',metadata['scope'])
        for row in metadata['checks']:
            cert=row['certificate']
            self.assertEqual(cert['beta_next'],.1/2000**.5)
            self.assertEqual(cert['coupling_fraction'],.125)
            self.assertLessEqual(cert['normalized_gap_upper_bound'],1e-7)
            self.assertEqual(row['probe_calls'],1)
            self.assertEqual(row['sketch_probes'],128)
            self.assertGreaterEqual(row['stored_plan_changed_entries'],0)

    def test_defaults_are_selected_accuracy(self):
        args = parser().parse_args(['--instance','data','--out','out','--method','published-msbpg',
                                   '--iterations','4','--seed','0','--output-seed','1'])
        self.assertEqual((args.inner_tol,args.inner_gap_tol,args.inner_max), (1e-7,1e-7,10000))
        self.assertEqual(args.save_plans,'all')
        self.assertEqual(args.progress_every,100)

    def test_published_momentum_and_preupdate_output_match_independent_dense_reference(self):
        expected = self.independent_trajectory()
        R = self.smd_metadata['output_selection']['index']
        self.assertEqual(R,2)
        selected = np.load(self.smd_out/'rank0'/'selected_raw.npy')
        last = np.load(self.smd_out/'rank0'/'last_raw.npy')
        np.testing.assert_allclose(selected,expected[R].numpy(),atol=2e-8,rtol=2e-7)
        np.testing.assert_allclose(last,expected[4].numpy(),atol=2e-8,rtol=2e-7)
        self.assertGreater(np.max(np.abs(last-self.independent_trajectory(no_momentum=True)[4].numpy())),1e-7)
        self.assertGreater(np.max(np.abs(last-self.independent_trajectory(wrong_index=True)[4].numpy())),1e-7)
        self.assertEqual(self.smd_metadata['counters']['stochastic_gradient_calls'],4)
        self.assertEqual(self.smd_metadata['counters']['initialization_oracle_calls'],1)
        self.assertEqual(self.smd_metadata['counters']['sketch_probes'],512)

    def test_output_randomness_does_not_change_optimization(self):
        other_seed = next(s for s in range(100) if select_output_index(4,generator=torch.Generator().manual_seed(s)) != 2)
        out, metadata = self.launch('other_output_rng',output_seed=other_seed)
        self.assertNotEqual(metadata['output_selection']['index'],2)
        np.testing.assert_array_equal(np.load(out/'rank0'/'last_raw.npy'),np.load(self.smd_out/'rank0'/'last_raw.npy'))

    def test_no_plan_persistence_preserves_outputs_and_full_validation(self):
        out, metadata = self.launch('no_plan_persistence', output_seed=self.output_seed,
                                    extra=('--save-plans','none','--progress-every','2'))
        self.assertEqual(list(out.rglob('*.npy')),[])
        self.assertEqual(metadata['final_float64_validation'],self.smd_metadata['final_float64_validation'])
        self.assertEqual(metadata['output_selection'],self.smd_metadata['output_selection'])
        self.assertEqual(metadata['inner_trace'],self.smd_metadata['inner_trace'])
        self.assertEqual(metadata['counters'],self.smd_metadata['counters'])
        for name, saved in metadata['ranks'][0]['saved_plans'].items():
            original = self.smd_metadata['ranks'][0]['saved_plans'][name]
            array = np.load(original['file'])
            self.assertTrue(original['persisted'])
            self.assertFalse(saved['persisted'])
            self.assertIsNone(saved['file'])
            self.assertIsNone(saved['file_sha256'])
            self.assertEqual(saved['bytes'],0)
            self.assertEqual(saved['array_bytes'],array.nbytes)
            self.assertEqual(saved['array_sha256'],original['array_sha256'])
            self.assertEqual(saved['array_sha256'],hashlib.sha256(array.tobytes()).hexdigest())
        self.assertFalse(metadata['plan_storage']['dense_arrays_persisted'])
        self.assertTrue(metadata['plan_storage']['all_four_outputs_validated_in_float64'])
        self.assertEqual(metadata['progress_reporting']['completed_updates_recorded'],[0,1,2,3,4])
        self.assertTrue(metadata['progress_reporting']['excluded_from_solver_time'])
        self.assertGreater(metadata['timing']['progress_logging_seconds'],0)
        self.assertEqual(metadata['config']['iterations'],4)

    def test_schedule_constants_and_selected_displacement_are_recorded(self):
        metadata=self.smd_metadata
        constants=metadata['theorem_constants']
        self.assertEqual(constants['mu'],1.)
        self.assertGreater(constants['hessian_bound'],constants['raw_hessian_bound'])
        self.assertEqual(constants['kappa'],2*constants['hessian_bound']**2)
        for k,info in enumerate(metadata['inner_trace']):
            self.assertEqual(info['beta'],None if k==0 else .6/k**.5)
            self.assertEqual(info['beta_next'],.6/(k+1)**.5)
            self.assertEqual(info['alpha'],info['coupling_c']*info['beta_next'])
            self.assertEqual(info['proximal_eta'],info['alpha']/constants['kernel_scale'])
        stat=metadata['selected_update_statistic']
        R=metadata['output_selection']['index']
        expected=self.independent_trajectory()
        value=((expected[R+1]-expected[R])/metadata['inner_trace'][R]['alpha']).square().sum().item()
        self.assertAlmostEqual(stat['normalized_displacement_squared'],value,delta=max(1e-10,value*1e-6))
        self.assertTrue(stat['excluded_from_solver_time'])
        self.assertEqual(stat['index'],R)
        self.assertEqual(metadata['method_key'],'published-msbpg')

    def test_float32_plans_keep_an_unquantized_float64_log_reference(self):
        out, metadata = self.launch('mixed_precision', extra=('--dtype','float32'))
        a = torch.tensor(self.arrays['a'],dtype=torch.float32)
        b = torch.tensor(self.arrays['b'],dtype=torch.float32)
        initial = .9*torch.outer(a,b)
        initial[torch.arange(len(a)),torch.as_tensor(np.random.default_rng(123).permutation(len(a)))] += .1/len(a)
        self.assertAlmostEqual(metadata['inner_trace'][0]['reference_mass'],initial.double().sum().item(),places=13)
        self.assertEqual(np.load(out/'rank0'/'last_raw.npy').dtype,np.float32)
        self.assertTrue(all(info['retained_log_dtype'] == 'torch.float64' for info in metadata['inner_trace']))
        self.assertLessEqual(metadata['inner_acceptance_maxima']['marginal_l1'],1e-7)

    def test_inner_failure_is_nonzero_and_preserved(self):
        out, result = self.launch('failed_inner',extra=('--inner-max','1','--inner-tol','1e-30','--inner-gap-tol','1e-30'),expect_success=False)
        failure = json.loads((out/'failure_rank0.json').read_text())
        inner = json.loads((out/'rank0'/'inner_failure.json').read_text())
        self.assertEqual(failure['status'],'failed')
        self.assertFalse(inner['accepted_failed_update'])
        self.assertEqual(inner['counters']['inner_stopping_failures'],1)
        self.assertEqual(inner['certificate']['inner_iterations'],1)
        self.assertEqual(inner['counters']['inner_iterations_total'],1)
        self.assertEqual(inner['counters']['certificate_checks'],1)
        self.assertFalse((out/'metadata.json').exists())

    def test_metadata_certifies_both_criteria_and_separates_diagnostics(self):
        metadata = self.smd_metadata
        self.assertEqual(len(metadata['inner_trace']),4)
        self.assertGreater(metadata['counters']['certificate_checks'],0)
        for info in metadata['inner_trace']:
            self.assertTrue(info['converged'])
            self.assertTrue(info['proximal_optimality_certified'])
            self.assertLessEqual(info['marginal_l1'],1e-7)
            self.assertLessEqual(info['normalized_primal_dual_gap'],1e-7)
        self.assertEqual(set(metadata['final_float64_validation']['plans']), {'selected_raw','selected_rounded','last_raw','last_rounded'})
        self.assertTrue(metadata['final_float64_validation']['optimizer_state_released'])
        self.assertGreater(metadata['timing']['exact_diagnostic_seconds'],0)
        self.assertEqual(metadata['array_files_sha256'],json.loads((self.data/'manifest.json').read_text())['array_files_sha256'])
        self.assertIn('published_stochastic_20261002/certified_prox.py',metadata['source_file_sha256'])


if __name__ == '__main__':
    unittest.main()
