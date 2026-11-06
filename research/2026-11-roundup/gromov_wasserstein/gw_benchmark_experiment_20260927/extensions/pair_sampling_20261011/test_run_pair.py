"""Small end-to-end shared-oracle checks plus legacy export regression."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import run_pair
from validate_distributed import dense_slice_gradient, dense_prox


class PairRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='pair_runner_test_')
        cls.base = Path(cls.tmp.name)
        cls.data = cls.base/'data'
        cls.data.mkdir()
        x, y = np.arange(4), np.array([0., .3, 1.1, 1.7])
        cls.arrays = dict(A=np.abs(x[:, None]-x[None, :])/3,
                          B=np.abs(y[:, None]-y[None, :]), a=np.full(4, .25), b=np.full(4, .25))
        hashes = {}
        for name, array in cls.arrays.items():
            path = cls.data/f'{name}.npy'
            np.save(path, array, allow_pickle=False)
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        (cls.data/'manifest.json').write_text(json.dumps(dict(array_files_sha256=hashes)))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def launch(self, method, label, *, entrypoint=None, pair=True, extra=()):
        output = self.base/label
        command = [sys.executable, str(entrypoint or ROOT/'run_pair.py'), '--method', method,
                   '--instance', str(self.data), '--out', str(output), '--device', 'cpu',
                   '--dtype', 'float64', '--iterations', '3', '--seed', '314', '--output-seed', '6',
                   '--check-every', '1', '--inner-tol', '1e-10', '--inner-gap-tol', '1e-8',
                   '--certificate-block-rows', '2', '--exact-block-rows', '2']
        if pair:
            command += ['--oracle', 'pair-sampling', '--pairs-per-call', '17']
        command += list(extra)
        env = dict(os.environ, OMP_NUM_THREADS='1', GLOO_SOCKET_IFNAME='lo')
        for key in ('RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
            env.pop(key, None)
        result = subprocess.run(command, text=True, capture_output=True, env=env, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout+'\n'+result.stderr)
        return output, json.loads((output/'metadata.json').read_text())

    def independent_trajectory(self, method, metadata):
        A, B, a, b = (torch.tensor(self.arrays[key], dtype=torch.float64) for key in ('A','B','a','b'))
        P = torch.outer(a, b)
        generator = torch.Generator().manual_seed(314)
        momentum, trajectory = None, [P.clone()]
        for k in range(3):
            uniforms = torch.rand((2, 17), dtype=torch.float64, generator=generator)
            cdf = P.sum(1).cumsum(0)
            mass = cdf[-1]
            K = torch.searchsorted(cdf/mass, uniforms[0], right=True)
            L = torch.empty_like(K)
            for t in range(17):
                conditional = P[K[t]].cumsum(0)
                L[t] = torch.searchsorted(conditional/conditional[-1], uniforms[1, t], right=True)
            G = dense_slice_gradient(A, B, K, L, mass)
            if method == 'smoothing-fw':
                rho = .3/(k+1)**.5
                momentum = G if k == 0 else (1-rho)*momentum+rho*G
                H = momentum+(P.sum(0)-b)[None, :]/(.1/4/(k+1)**.25)
                gamma = .1/(k+1)**.75
                P = (1-gamma)*P
                P[torch.arange(4), H.argmin(1)] += gamma*a
            else:
                if method == 'published-smd':
                    eta = .25/metadata['cmax']/(k+1)**.5
                    direction = G
                else:
                    beta = None if k == 0 else 1/k**.5
                    momentum = G if k == 0 else (1-beta)*momentum+beta*G
                    constants = metadata['theorem_constants']
                    eta = constants['coupling_c_max']/constants['kernel_scale']/(k+1)**.5
                    direction = momentum
                P = dense_prox(P, direction, a, b, eta)
            trajectory.append(P.clone())
        return trajectory

    def check_method(self, method):
        out, meta = self.launch(method, method)
        counters = meta['counters']
        self.assertEqual((counters['stochastic_gradient_calls'], counters['pair_samples'], counters['sketch_probes']), (3, 51, 0))
        self.assertEqual(counters['initialization_oracle_calls'], int(method != 'published-smd'))
        self.assertEqual(meta['gradient_oracle']['name'], 'pair-sampling')
        self.assertEqual(meta['gradient_oracle']['pair_samples_per_call'], 17)
        self.assertTrue(meta['random_streams']['pairs']['rank_zero_generator'])
        self.assertNotIn('probes', meta['random_streams'])
        self.assertEqual(meta['source_file_sha256'], run_pair.source_hashes())
        expected = self.independent_trajectory(method, meta)
        for kind, index in (('selected', meta['output_selection']['index']), ('last', 3)):
            actual = np.load(out/'rank0'/f'{kind}_raw.npy')
            np.testing.assert_allclose(actual, expected[index].numpy(), rtol=3e-8, atol=3e-10)
        self.assertEqual(meta['final_float64_validation']['plan_values'], 'Promoted CPU snapshot values; no additional re-rounding')
        return out, meta

    def test_frames_same_fresh_pair_stream_and_initial_reuse(self):
        out, meta = self.check_method('smoothing-fw')
        other, other_meta = self.launch('smoothing-fw', 'different-output-seed', extra=('--output-seed','901'))
        self.assertEqual((out/'rank0'/'last_raw.npy').read_bytes(), (other/'rank0'/'last_raw.npy').read_bytes())
        self.assertEqual(meta['counters'], other_meta['counters'])

    def test_smd_uses_fresh_gradient_without_momentum(self):
        self.check_method('published-smd')

    def test_msbpg_preserves_published_momentum_and_initial_reuse(self):
        self.check_method('published-msbpg')

    def test_legacy_exports_need_no_pair_directory_and_keep_source_keysets(self):
        release = self.base/'legacy_export'
        for name in run_pair.source_hashes():
            if name.startswith('pair_sampling_20261011/'):
                continue
            destination = release/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT.parent/name, destination)
        for method, module, implementation in (
                ('published-smd', 'published_stochastic_20261002/run_published.py', run_pair.run_published),
                ('published-msbpg', 'msbpg_20261005/run_msbpg.py', run_pair.run_msbpg)):
            _, meta = self.launch(method, 'legacy-'+method, entrypoint=release/module, pair=False)
            self.assertEqual(meta['config']['oracle'], 'sketch')
            self.assertEqual(meta['counters']['pair_samples'], 0)
            self.assertEqual(meta['counters']['sketch_probes'], 384)
            self.assertEqual(set(meta['source_file_sha256']), set(implementation.source_hashes()))
            self.assertFalse(any(name.startswith('pair_sampling_20261011/') for name in meta['source_file_sha256']))


if __name__ == '__main__':
    unittest.main()
