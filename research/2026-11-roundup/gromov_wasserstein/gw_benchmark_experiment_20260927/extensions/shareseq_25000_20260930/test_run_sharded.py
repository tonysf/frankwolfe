"""Tiny distributed runner integration and fail-closed campaign gates.

CPU fixtures validate bookkeeping and outputs only; their timings are not
performance results. The worker mode below is called by torchrun subprocesses.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import weakref

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PYTHON = REPO / '.venv' / 'bin' / 'python'
METHODS = ('smoothing-fw', 'stochastic-kl-momentum')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _module(name):
    spec = importlib.util.spec_from_file_location('shareseq_' + name + '_under_test', HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _launch(arguments, timeout=120):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(HERE), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1')
    return subprocess.run([str(PYTHON), '-m', 'torch.distributed.run', '--standalone',
                           '--nproc-per-node=2', str(Path(__file__).resolve()), *map(str, arguments)],
                          cwd=HERE, env=env, capture_output=True, text=True, timeout=timeout)


def _direct_loss(A, B, P):
    discrepancy = (A.astype(np.float64)[:, None, :, None]
                   - B.astype(np.float64)[None, :, None, :]) ** 2
    return float(np.einsum('ijkl,ij,kl->', discrepancy, P.astype(np.float64), P.astype(np.float64)))


def _direct_metrics(P, forward):
    n = len(P)
    accuracy, ranks = [], []
    for matrix, mapping in ((P, forward), (P.T, np.argsort(forward))):
        credit, rank = [], []
        for i, j in enumerate(mapping):
            equal = int(np.count_nonzero(matrix[i] == matrix[i, j]))
            higher = int(np.count_nonzero(matrix[i] > matrix[i, j]))
            credit.append(float(higher == 0) / equal)
            rank.append((higher + .5 * (equal - 1)) / max(1, n - 1))
        accuracy.append(float(np.mean(credit)))
        ranks.append(float(np.mean(rank)))
    return dict(correspondence_top1_accuracy=sum(accuracy) / 2,
                correspondence_rank_fraction=sum(ranks) / 2,
                correspondence_forward_top1_accuracy=accuracy[0],
                correspondence_inverse_top1_accuracy=accuracy[1],
                correspondence_true_pair_mass=float(P[np.arange(n), forward].astype(np.float64).sum()))


@pytest.fixture(scope='module')
def tiny_runs(tmp_path_factory):
    root = tmp_path_factory.mktemp('shareseq_sharded_runner')
    instance = root / 'instance'
    instance.mkdir()
    n = 9
    positions = np.random.default_rng(2988).normal(size=(n, 3))
    A = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
    A = (A / A.max()).astype(np.float32)
    permutation = np.random.default_rng(3129).permutation(n)
    B = A[np.ix_(permutation, permutation)]
    marginals = np.full(n, 1/n, dtype=np.float64)
    arrays = dict(A=A, B=B, a=marginals, b=marginals.copy())
    for key, value in arrays.items():
        np.save(instance / (key + '.npy'), value, allow_pickle=False)
    hashes = {key + '.npy': _sha(instance / (key + '.npy')) for key in arrays}
    (instance / 'manifest.json').write_text(json.dumps(dict(n=n, m=n, family='tiny_test_only',
                                                         array_files_sha256=hashes)))
    forward = np.argsort(permutation)
    labels = instance / 'correspondence.npz'
    np.savez(labels, forward=forward, inverse=permutation)
    runs = {}
    for method in METHODS:
        output = root / method
        arguments = ['--worker-run', '--instance', instance, '--out', output,
                     '--method', method, '--device', 'cpu', '--budget', '10',
                     '--checkpoints', '0.000001,0.000002,10', '--max-updates', '3',
                     '--seed', '147', '--init-seed', '917', '--initialization', 'randomized',
                     '--batch', '128', '--rho0', '.3', '--gamma0', '.1', '--mu-scale', '.1',
                     '--eta', '.1', '--inner-tol', '1e-12', '--inner-max', '1',
                     '--exact-block-rows', '3', '--correspondence', labels,
                     '--correspondence-sha256', _sha(labels), '--save-plans']
        completed = _launch(arguments)
        assert completed.returncode == 0, completed.stdout + '\n' + completed.stderr
        metadata = json.loads((output / 'metadata.json').read_text())
        plans = {key: np.concatenate([np.load(output / f'rank{rank}' / (key + '.npy'), allow_pickle=False)
                                     for rank in range(2)], axis=0)
                 for key in ('P_raw', 'P_rounded')}
        runs[method] = dict(output=output, metadata=metadata, plans=plans)
    return dict(root=root, instance=instance, n=n, arrays=arrays, forward=forward,
                labels=labels, hashes=hashes, runs=runs)


@pytest.mark.parametrize('method', METHODS)
def test_saved_shards_and_final_float64_validation(tiny_runs, method):
    run = tiny_runs['runs'][method]
    metadata = run['metadata']
    assert metadata['status'] in ('completed', 'completed_with_inner_failures')
    assert metadata['environment']['world_size'] == 2
    assert metadata['environment']['backend'] == 'gloo'
    assert metadata['counters']['completed_updates'] == 3
    assert metadata['array_files_sha256'] == tiny_runs['hashes']
    assert metadata['config']['correspondence_sha256'] == _sha(tiny_runs['labels'])
    assert metadata['config']['initialization'] == 'randomized'
    for key, field in (('P_raw', 'raw'), ('P_rounded', 'rounded')):
        plan = run['plans'][key]
        assert plan.shape == (tiny_runs['n'], tiny_runs['n']) and plan.dtype == np.float32
        assert np.isfinite(plan).all() and np.all(plan >= 0)
        expected = _direct_loss(tiny_runs['arrays']['A'], tiny_runs['arrays']['B'], plan)
        validation = metadata['final_float64_validation']
        assert validation[field + '_loss'] == pytest.approx(expected, abs=2e-12, rel=1e-10)
        plan64 = plan.astype(np.float64)
        a, b = tiny_runs['arrays']['a'], tiny_runs['arrays']['b']
        residual = np.abs(plan64.sum(1) - a).sum() + np.abs(plan64.sum(0) - b).sum()
        assert validation[field]['both_l1'] == pytest.approx(residual, abs=1e-13)
        for rank, rank_record in enumerate(metadata['ranks']):
            saved = rank_record['saved_plans'][key]
            path = run['output'] / f'rank{rank}' / (key + '.npy')
            array = np.load(path, allow_pickle=False)
            assert saved['file_sha256'] == _sha(path)
            assert saved['array_sha256'] == hashlib.sha256(array.tobytes()).hexdigest()
            assert saved['shape'] == list(array.shape)
    assert metadata['final_float64_validation']['rounded']['both_l1'] <= 1.2e-5
    for rank in range(2):
        release = json.loads((run['output'] / f'rank{rank}' / 'float32_release_check.json').read_text())
        assert release and all(release.values()), release


@pytest.mark.parametrize('method', METHODS)
def test_counter_checkpoint_timer_and_correspondence_contracts(tiny_runs, method):
    metadata = tiny_runs['runs'][method]['metadata']
    counters = metadata['counters']
    assert counters['stochastic_gradient_calls'] == counters['completed_updates'] == 3
    assert counters['initialization_oracle_calls'] == 1
    assert counters['sketch_probes'] == 3 * 128
    trace = metadata['trace']
    evaluated = [row for row in trace if row['kind'] != 'missed']
    assert evaluated[0]['iteration'] == 0 and evaluated[0]['kind'] == 'initial'
    assert evaluated[-1]['iteration'] == 3
    assert [row['iteration'] for row in evaluated] == [0, 1, 3]
    missed = [row for row in trace if row['kind'] == 'missed']
    assert len(missed) == 2
    assert missed[0]['target_solver_seconds'] == 2e-6
    assert missed[1]['target_solver_seconds'] == 10
    assert evaluated[-1]['momentum_point_iteration'] == 2
    assert evaluated[-1]['momentum_error_matches_current_output'] is False
    timing = metadata['timing']
    assert all(np.isfinite(value) and value >= 0 for value in timing.values())
    assert timing['solver_gpu_seconds'] == pytest.approx(2 * timing['solver_seconds'])
    assert timing['solver_seconds'] == evaluated[-1]['actual_solver_seconds']
    assert timing['total_elapsed_seconds'] >= timing['solver_seconds']
    assert evaluated[-1]['time_to_this_feasible_output'] == pytest.approx(
        timing['solver_seconds'] + evaluated[-1]['repair_seconds_current'])
    if method == 'stochastic-kl-momentum':
        assert counters['inner_stopping_failures'] > 0
        assert metadata['status'] == 'completed_with_inner_failures'
    expected = _direct_metrics(tiny_runs['runs'][method]['plans']['P_rounded'], tiny_runs['forward'])
    for key, value in expected.items():
        assert evaluated[-1][key] == pytest.approx(value, abs=2e-13)


def test_matched_initial_shards(tiny_runs):
    fw, kl = (tiny_runs['runs'][method]['metadata'] for method in METHODS)
    assert [row['initial_shard_sha256'] for row in fw['ranks']] == [row['initial_shard_sha256'] for row in kl['ranks']]
    for key in ('seed', 'init_seed', 'batch', 'rho0'):
        assert fw['config'][key] == kl['config'][key]


def test_distributed_correspondence_on_randomized_ties(tmp_path):
    output = tmp_path / 'tie_metrics.json'
    completed = _launch(['--worker-matching', output])
    assert completed.returncode == 0, completed.stdout + '\n' + completed.stderr
    cases = json.loads(output.read_text())
    assert len(cases) == 3
    for case in cases:
        assert case['unchanged']
        for key, expected in case['expected'].items():
            assert case['actual'][key] == pytest.approx(expected, abs=2e-13)


def _valid_gate_result():
    phase = dict(cuda_peak_reserved_bytes=40, cuda_total_memory_bytes=100,
                 cpu_process_peak_rss_kib=1024)
    return dict(status='completed', environment=dict(world_size=4),
                ranks=[dict(rank=i, gpu='NVIDIA A100-SXM4-40GB',
                            memory=dict(solver=copy.deepcopy(phase))) for i in range(4)],
                counters=dict(completed_updates=3),
                final_float64_validation=dict(dtype='float64', exact_objective=True, raw_loss=.2, rounded_loss=.21,
                                              rounded=dict(both_l1=1e-7)))


def test_pilot_gate_accepts_valid_result():
    _module('campaign').gate(_valid_gate_result())


@pytest.mark.parametrize('change', ('failed', 'world', 'gpu_memory', 'host_memory',
                                    'residual', 'zero_updates', 'nan_gpu_memory',
                                    'nan_residual', 'nan_loss', 'duplicate_rank'))
def test_pilot_gate_rejects_invalid_result(change):
    result = _valid_gate_result()
    if change == 'failed':
        result['status'] = 'failed'
    elif change == 'world':
        result['environment']['world_size'] = 2
    elif change == 'gpu_memory':
        result['ranks'][0]['memory']['solver']['cuda_peak_reserved_bytes'] = 80
    elif change == 'host_memory':
        result['ranks'][0]['memory']['solver']['cpu_process_peak_rss_kib'] = 128 * 1024**2
    elif change == 'residual':
        result['final_float64_validation']['rounded']['both_l1'] = .1
    elif change == 'zero_updates':
        result['counters']['completed_updates'] = 0
    elif change == 'nan_gpu_memory':
        result['ranks'][0]['memory']['solver']['cuda_peak_reserved_bytes'] = float('nan')
    elif change == 'nan_residual':
        result['final_float64_validation']['rounded']['both_l1'] = float('nan')
    elif change == 'nan_loss':
        result['final_float64_validation']['rounded_loss'] = float('nan')
    elif change == 'duplicate_rank':
        result['ranks'][3]['rank'] = 0
    with pytest.raises((ValueError, RuntimeError)):
        _module('campaign').gate(result)



def _bound_result(tmp_path):
    source = HERE
    config = dict(instance=str(tmp_path/'instance'),out=str(tmp_path/'output'),
        method='smoothing-fw',device='cuda',budget=30,checkpoints='10,30',seed=450,
        init_seed=100,initialization='product',batch=128,rho0=.3,gamma0=.1,mu_scale=.1,
        eta=.1,inner_tol=1e-4,inner_max=1000,max_updates=0,exact_block_rows=1024,
        correspondence=str(tmp_path/'correspondence.npz'),correspondence_sha256='c'*64,save_plans=False)
    manifest = dict(n=9,m=9,array_files_sha256={key+'.npy':str(i)*64 for i,key in enumerate(('A','B','a','b'))},
                    correspondence_sha256='c'*64)
    result = _valid_gate_result()
    result.update(config=copy.deepcopy(config),instance_manifest=copy.deepcopy(manifest),
                  array_files_sha256=copy.deepcopy(manifest['array_files_sha256']),
                  source_file_sha256={name:_sha(source/name) for name in
                                     ('run_sharded.py','sharded_gw.py','theory_config.json')},
                  timing=dict(solver_seconds=30.,solver_gpu_seconds=120.,total_elapsed_seconds=35.))
    result['environment'].update(backend='nccl',hostname='testnode',slurm_job_id='test123')
    result['counters'].update(stochastic_gradient_calls=3,initialization_oracle_calls=1,sketch_probes=384)
    for rank, record in enumerate(result['ranks']):
        record['initial_shard_sha256'] = str(rank)*64
        record['saved_plans'] = {key:dict(shape=[9//4+int(rank<9%4),9],array_sha256='d'*64)
                                 for key in ('P_raw','P_rounded')}
    return result, config, manifest, source


@pytest.mark.parametrize('change', ('valid','config','array','source','counters','nan_loss',
                                    'labels','rank','missing_plans','shape','saved_file_missing'))
def test_result_binding(tmp_path, change):
    result, expected, manifest, source = _bound_result(tmp_path)
    if change == 'config':
        result['config']['rho0'] = .9
    elif change == 'array':
        result['array_files_sha256']['A.npy'] = 'e'*64
    elif change == 'source':
        result['source_file_sha256']['run_sharded.py'] = 'e'*64
    elif change == 'counters':
        result['counters']['stochastic_gradient_calls'] = 4
    elif change == 'nan_loss':
        result['final_float64_validation']['rounded_loss'] = float('nan')
    elif change == 'labels':
        manifest['correspondence_sha256'] = 'e'*64
        result['instance_manifest'] = copy.deepcopy(manifest)
    elif change == 'rank':
        result['ranks'][3]['rank'] = 0
    elif change == 'missing_plans':
        del result['ranks'][0]['saved_plans']['P_raw']
    elif change == 'shape':
        result['ranks'][0]['saved_plans']['P_raw']['shape'] = [9,9]
    elif change == 'saved_file_missing':
        result['config']['save_plans'] = expected['save_plans'] = True
    if change == 'valid':
        _module('campaign').validate_result(result,expected,manifest,source)
    else:
        with pytest.raises((ValueError,RuntimeError)):
            _module('campaign').validate_result(result,expected,manifest,source)


@pytest.mark.parametrize('change', ('valid','initial','array','budget','seed','allocation','source','method'))
def test_pair_binding(tmp_path, change):
    first, _, _, _ = _bound_result(tmp_path)
    second = copy.deepcopy(first)
    second['config']['method'] = 'stochastic-kl-momentum'
    if change == 'initial':
        second['ranks'][0]['initial_shard_sha256'] = 'e'*64
    elif change == 'array':
        second['array_files_sha256']['B.npy'] = 'e'*64
    elif change == 'budget':
        second['config']['budget'] = 301
    elif change == 'seed':
        second['config']['seed'] = 451
    elif change == 'allocation':
        second['environment']['slurm_job_id'] = 'test124'
    elif change == 'source':
        second['source_file_sha256']['run_sharded.py'] = 'e'*64
    elif change == 'method':
        second['config']['method'] = 'smoothing-fw'
    if change == 'valid':
        _module('campaign').verify_pair(first,second)
    else:
        with pytest.raises((ValueError,RuntimeError)):
            _module('campaign').verify_pair(first,second)


@pytest.mark.parametrize('change', ('valid','world','hardware','source','empty_checks','failed_check',
                                    'coverage','duplicate_rank','device'))
def test_correctness_evidence_binding(change):
    result = dict(status='passed',world_size=4,n=17,m=17,batch=128,momentum_steps=5,
                  precision=['float64','float32'],tf32=False,deterministic_algorithms=True,
                  checks=[dict(name=str(i),passed=True) for i in range(352)],failed_checks=[],
                  source_sha256={Path(name).name:_sha(HERE/name) for name in
                                 ('gpu_correctness.py','sharded_gw.py','reference/gw_core.py')},
                  ranks=[dict(rank=i,local_rank=i,hostname='testnode',device='cuda:%d'%i,
                              gpu_name='NVIDIA A100-SXM4-40GB') for i in range(4)])
    if change == 'world':
        result['world_size'] = 2
    elif change == 'hardware':
        result['ranks'][0]['gpu_name'] = 'CPU'
    elif change == 'source':
        result['source_sha256']['sharded_gw.py'] = 'e'*64
    elif change == 'empty_checks':
        result['checks'] = []
    elif change == 'failed_check':
        result['checks'][0]['passed'] = False
    elif change == 'coverage':
        result['precision'] = ['float32']
    elif change == 'duplicate_rank':
        result['ranks'][3]['rank'] = 0
    elif change == 'device':
        result['ranks'][3]['device'] = 'cuda:0'
    if change == 'valid':
        _module('campaign').correctness_gate(result,HERE)
    else:
        with pytest.raises((ValueError,RuntimeError)):
            _module('campaign').correctness_gate(result,HERE)

def _worker_run(arguments):
    import torch
    import torch.distributed as dist
    import run_sharded as runner
    args = runner.parser().parse_args(arguments)
    device = runner.setup(args.device)
    original_loader = runner.load_problem
    references = {}

    def checked_loader(directory, dtype, device, block_rows):
        if dtype == torch.float64:
            freed = {key: ref() is None for key, ref in references.items()}
            path = Path(args.out) / f'rank{dist.get_rank()}' / 'float32_release_check.json'
            path.write_text(json.dumps(freed))
            if not all(freed.values()):
                raise AssertionError('Float32 solver/problem tensors retained into float64 validation: ' + str(freed))
        problem = original_loader(directory, dtype, device, block_rows)
        if dtype == torch.float32:
            references.update(problem=weakref.ref(problem), A=weakref.ref(problem.A),
                              B=weakref.ref(problem.B), A2=weakref.ref(problem.A2), B2=weakref.ref(problem.B2))
        return problem

    runner.load_problem = checked_loader
    try:
        runner.run(args, device)
    finally:
        dist.destroy_process_group()


def _worker_matching(output):
    import torch
    import torch.distributed as dist
    import run_sharded as runner
    from sharded_gw import RowShardedGW
    device = runner.setup('cpu')
    try:
        n = 9
        rank, world = dist.get_rank(), dist.get_world_size()
        start, stop = runner.shard_bounds(n, rank, world)
        A = torch.eye(n, dtype=torch.float32)
        a = torch.full((n,), 1/n)
        problem = RowShardedGW(A[start:stop], A, a, a, exact_block_rows=3)
        rng = np.random.default_rng(1857)
        forward = rng.permutation(n)
        cases = [np.full((n, n), 1/n**2, dtype=np.float32),
                 rng.integers(0, 4, size=(n, n)).astype(np.float32) / 100,
                 rng.integers(0, 2, size=(n, n)).astype(np.float32)]
        reports = []
        for P in cases:
            local = torch.as_tensor(P[start:stop].copy())
            before = local.clone()
            actual = runner.matching_metrics(problem, local, forward, chunk=2)
            unchanged = torch.tensor(int(torch.equal(before, local)))
            dist.all_reduce(unchanged, op=dist.ReduceOp.MIN)
            reports.append(dict(actual=actual, expected=_direct_metrics(P, forward),
                                unchanged=bool(unchanged.item())))
        if rank == 0:
            Path(output).write_text(json.dumps(reports, allow_nan=False))
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--worker-run':
        _worker_run(sys.argv[2:])
    elif len(sys.argv) == 3 and sys.argv[1] == '--worker-matching':
        _worker_matching(sys.argv[2])
    else:
        raise SystemExit('This file is a pytest suite or an explicitly selected torchrun worker.')
