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
from smd import select_output_index
from run_published import parser


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
        cls.temporary = tempfile.TemporaryDirectory(prefix='published_smd_runner_test_')
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
    def launch(cls, label, method='published-smd', output_seed=0, extra=(), expect_success=True):
        out = cls.base/label
        command = [sys.executable, str(ROOT/'run_published.py'), '--instance', str(cls.data),
                   '--out', str(out), '--method', method, '--device', 'cpu', '--dtype', 'float64',
                   '--iterations', '4', '--seed', '314', '--output-seed', str(output_seed),
                   '--initialization', 'randomized', '--init-seed', '123', '--check-every', '1',
                   '--certificate-block-rows', '2', '--exact-block-rows', '2',
                   '--checkpoint-iterations', '1,3', *extra]
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

    def independent_trajectory(self, method, *, ema=False):
        A, B, a, b = (torch.tensor(self.arrays[n], dtype=torch.float64) for n in ('A','B','a','b'))
        P = .9*torch.outer(a,b)
        permutation = np.random.default_rng(123).permutation(len(a))
        P[torch.arange(len(a)), torch.as_tensor(permutation)] += .1/len(a)
        trajectory, d = [P.clone()], None
        generator = torch.Generator().manual_seed(314)
        cmax = max(abs(A.min().item()-B.max().item()), abs(A.max().item()-B.min().item()))**2
        for k in range(4):
            Z = (2*torch.randint(0,2,(len(b),128),generator=generator)-1).double()
            G = dense_gradient(A,B,P,Z)
            if method == 'smoothing-fw' or ema:
                rho = .3/(k+1)**.5
                d = G if d is None else (1-rho)*d+rho*G
                G = d
            if method == 'published-smd':
                P = dense_prox(P,G,a,b,.25/cmax/(k+1)**.5)
            else:
                H = G+(P.sum(0)-b)[None,:]/(.1/len(b)/(k+1)**.25)
                gamma = .1/(k+1)**.75
                P = (1-gamma)*P
                P[torch.arange(len(a)),H.argmin(1)] += gamma*a
            trajectory.append(P.clone())
        return trajectory

    def test_defaults_are_selected_accuracy(self):
        args = parser().parse_args(['--instance','data','--out','out','--method','published-smd',
                                   '--iterations','4','--seed','0','--output-seed','1'])
        self.assertEqual((args.inner_tol,args.inner_gap_tol,args.inner_max), (1e-7,1e-7,10000))

    def test_fresh_gradient_and_preupdate_output_match_independent_dense_reference(self):
        expected = self.independent_trajectory('published-smd')
        R = self.smd_metadata['output_selection']['index']
        self.assertEqual(R,2)
        selected = np.load(self.smd_out/'rank0'/'selected_raw.npy')
        last = np.load(self.smd_out/'rank0'/'last_raw.npy')
        np.testing.assert_allclose(selected,expected[R].numpy(),atol=2e-8,rtol=2e-7)
        np.testing.assert_allclose(last,expected[4].numpy(),atol=2e-8,rtol=2e-7)
        self.assertGreater(np.max(np.abs(last-self.independent_trajectory('published-smd',ema=True)[4].numpy())),1e-6)
        self.assertEqual(self.smd_metadata['counters']['stochastic_gradient_calls'],4)
        self.assertEqual(self.smd_metadata['counters']['initialization_oracle_calls'],0)
        self.assertEqual(self.smd_metadata['counters']['sketch_probes'],512)

    def test_output_randomness_does_not_change_optimization(self):
        other_seed = next(s for s in range(100) if select_output_index(4,generator=torch.Generator().manual_seed(s)) != 2)
        out, metadata = self.launch('other_output_rng',output_seed=other_seed)
        self.assertNotEqual(metadata['output_selection']['index'],2)
        np.testing.assert_array_equal(np.load(out/'rank0'/'last_raw.npy'),np.load(self.smd_out/'rank0'/'last_raw.npy'))

    def test_fw_preserves_the_existing_update(self):
        out, metadata = self.launch('fw',method='smoothing-fw')
        np.testing.assert_allclose(np.load(out/'rank0'/'last_raw.npy'), self.independent_trajectory('smoothing-fw')[4].numpy(),atol=1e-15,rtol=1e-14)
        self.assertEqual(metadata['counters']['stochastic_gradient_calls'],4)
        self.assertEqual(metadata['counters']['inner_iterations_total'],0)
        self.assertIn('Diagnostic only',metadata['output_selection']['scope'])

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
