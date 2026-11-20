"""Prepare immutable prospective Stage C manifests; never execute or submit runs.

Without --selected, generate all 1,164 development tuning cells. After the
independent selection report is frozen, --selected generates the three final
panels. Dataset hashes and product-plan reference values are computed from the
canonical saved arrays. The generator never reads the planted correspondence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')

import numpy as np

ROOT = Path(__file__).resolve().parent
CAMPAIGN_PATH = ROOT / 'configs' / 'stage_c_campaign.json'
STOCHASTIC = {'smoothing-fw', 'stochastic-kl', 'native-sampled-gw', 'smoothing-row-projection'}
TUNABLE = ('smoothing-fw', 'full-kl', 'stochastic-kl', 'native-sampled-gw', 'smoothing-row-projection', 'pg')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def candidate_catalog(campaign):
    """Generate exactly 12 distinct, stable candidate IDs for each method."""
    result = {}
    for method in TUNABLE:
        grid = campaign['candidate_grids'][method]
        axes = [(key, value) for key, value in grid.items() if isinstance(value, list)]
        constants = {key: value for key, value in grid.items() if not isinstance(value, list)}
        candidates = []
        for index, values in enumerate(itertools.product(*(values for _, values in axes))):
            parameters = constants | dict(zip((key for key, _ in axes), values))
            candidates.append(dict(candidate_id=f'c{index:02d}', parameters=parameters))
        if len(candidates) != campaign['candidates_per_tunable_method']:
            raise ValueError(f'{method} does not have the prescribed candidate count.')
        result[method] = candidates
    return result


def synthetic_instances(n, seeds, noise_levels):
    result = []
    for noise in noise_levels:
        infix = '_noisy' if noise else ''
        for seed in seeds:
            result.append(dict(instance=f'metric{n}{infix}_s{seed}.npz', problem_seed=seed,
                               noise=noise, n=n, m=n, family='synthetic'))
    return result


def bmmc_instance(campaign):
    n = campaign['primary_sizes']['bmmc']
    return dict(instance=f'bmmc{n}.npz', problem_seed=None, noise=None, n=n, m=n, family='bmmc')


class CanonicalInstances:
    def __init__(self, study_root):
        self.data = Path(study_root).resolve() / 'data'
        self.cache = {}

    def metadata(self, description):
        basename = description['instance']
        if basename in self.cache:
            return self.cache[basename]
        path = self.data / basename
        with np.load(path, allow_pickle=False) as archive:
            # Never load true_target or any correspondence/label metadata.
            arrays = {key: np.array(archive[key], copy=True) for key in ('A', 'B', 'a', 'b')}
        hashes = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in arrays.items()}
        manifest = json.loads(path.with_suffix('.json').read_text())
        for key in ('A', 'B'):
            if hashes[key] != manifest.get(f'{key}_sha256'):
                raise ValueError(f'{path}: frozen {key} hash mismatch.')
        n, m = len(arrays['a']), len(arrays['b'])
        if (n, m) != (description['n'], description['m']):
            raise ValueError(f'{path}: dimensions differ from the declared panel.')
        A, B, a, b = (arrays[key].astype(np.float64, copy=False) for key in ('A', 'B', 'a', 'b'))
        if any(not np.isfinite(value).all() for value in (A, B, a, b)):
            raise ValueError(f'{path}: nonfinite canonical data.')
        # For P=a*b.T, this O(n²+m²) identity avoids another dense GW product.
        # Retain floating-point marginal sums rather than silently substituting 1.
        reference = float((b.sum() ** 2) * (a @ (A * A) @ a)
                          + (a.sum() ** 2) * (b @ (B * B) @ b)
                          - 2 * (a @ A @ a) * (b @ B @ b))
        if not math.isfinite(reference) or reference <= 0:
            raise ValueError(f'{path}: product-loss normalization requires a strictly positive finite reference.')
        result = dict(instance_sha256=sha256_file(path), expected_array_sha256=hashes,
                      product_loss_float64=reference,
                      product_loss_formula='sum(b)^2*a^T(A**2)a+sum(a)^2*b^T(B**2)b-2*(a^TAa)*(b^TBb)')
        self.cache[basename] = result
        return result


def _number(value):
    return repr(value) if isinstance(value, float) else str(value)


def _argv(method, parameters, description, seed, init_seed, budget, checkpoints, initialization, campaign):
    common = ['--device', 'cuda', '--dtype', 'float32', '--threads', '1',
              '--seed', str(seed), '--init-seed', str(init_seed),
              '--budget', _number(budget), '--checkpoints', ','.join(_number(t) for t in checkpoints)]
    resolved = dict(parameters)
    if method == 'native-sampled-gw':
        if initialization != 'product':
            raise ValueError('Native Sampled GW cannot enter the randomized-start panel.')
        return ['run_native_sampled.py', *common, '--batch', str(parameters['batch']),
                '--epsilon', _number(parameters['epsilon'])], resolved, 'native-sampled-gw'
    implementation = 'smoothing-fw' if method == 'smoothing-fw-exact' else method
    argv = ['run_benchmark.py', '--method', implementation, *common]
    if initialization == 'randomized':
        argv += ['--initialization', 'randomized']
    argv += ['--inner-max', str(campaign['common_inner_max'])]
    if method in ('smoothing-fw', 'smoothing-fw-exact', 'smoothing-row-projection', 'stochastic-kl'):
        oracle = 'exact' if method == 'smoothing-fw-exact' else campaign['custom_oracle']
        argv += ['--oracle', oracle, '--batch', str(campaign['custom_oracle_batch']),
                 '--oracle-selection-status', 'stage_c_prospectively_frozen_q128_no_speed_claim']
        resolved.update(oracle=oracle, batch=campaign['custom_oracle_batch'])
    if method.startswith('smoothing-'):
        argv += ['--mu-scale', _number(parameters['mu_scale']),
                 '--gamma0', _number(parameters.get('gamma0', 1.)),
                 '--rho0', _number(parameters.get('rho0', 1.))]
    if method == 'smoothing-row-projection':
        eta = parameters['eta_coefficient'] * parameters['mu_scale'] / (description['m'] * description['n'])
        argv += ['--eta', _number(eta), '--eta-power', _number(parameters['eta_power'])]
        resolved['eta'] = eta
    elif method == 'pg':
        eta = parameters['eta_coefficient'] / description['n'] ** 2
        argv += ['--eta', _number(eta), '--eta-power', _number(parameters['eta_power']),
                 '--inner-tol', _number(parameters['inner_tol']), '--pg-max-n', str(campaign['small_pg_size'])]
        resolved['eta'] = eta
    elif method in ('full-kl', 'stochastic-kl'):
        argv += ['--eta', _number(parameters['eta']), '--eta-power', _number(parameters['eta_power']),
                 '--inner-tol', _number(parameters['inner_tol'])]
    return argv, resolved, implementation


def make_run(phase, method, candidate, description, seed, init_seed, budget, checkpoints,
             initialization, campaign, instances):
    candidate_id = candidate['candidate_id']
    argv, resolved, implementation = _argv(method, candidate['parameters'], description,
                                          seed, init_seed, budget, checkpoints, initialization, campaign)
    run_id = f"{phase}/{method}/{candidate_id}/{Path(description['instance']).stem}/s{seed}_i{init_seed}"
    return dict(run_id=run_id, method=method, implementation=implementation,
                candidate_id=candidate_id, config_id=candidate_id, **description,
                seed=seed, stochastic_seed=seed, init_seed=init_seed, initialization_seed=init_seed,
                initialization=initialization, deterministic=method not in STOCHASTIC,
                budget=budget, checkpoints=list(checkpoints), parameters=resolved,
                candidate_parameters=dict(candidate['parameters']), argv=argv,
                source_method_for_selection='smoothing-fw' if method == 'smoothing-fw-exact' else method,
                expected_status='pending', **instances.metadata(description))


def make_tuning(campaign, catalog, instances):
    primary = synthetic_instances(campaign['primary_sizes']['synthetic'], campaign['development_problem_seeds'], campaign['noise_levels'])
    primary.append(bmmc_instance(campaign))
    small = synthetic_instances(campaign['small_pg_size'], campaign['development_problem_seeds'], campaign['noise_levels'])
    runs = []
    for method in TUNABLE:
        descriptions = small if method == 'pg' else primary
        seeds = campaign['development_stochastic_seeds'] if method in STOCHASTIC else [campaign['development_stochastic_seeds'][0]]
        for candidate, description, seed in itertools.product(catalog[method], descriptions, seeds):
            runs.append(make_run('tuning', method, candidate, description, seed, 0,
                                 campaign['tuning_budget_seconds'], campaign['tuning_checkpoints_seconds'],
                                 'product', campaign, instances))
    return runs


def selected_catalog(selection, catalog, *, campaign_sha256, tuning_manifest_sha256):
    if selection.get('schema_version') != 1 or not isinstance(selection.get('selections'), dict):
        raise ValueError('Selection must have schema_version=1 and a selections mapping.')
    if selection.get('campaign_config_sha256') != campaign_sha256:
        raise ValueError('Selection campaign hash differs from the frozen current campaign.')
    if selection.get('tuning_manifest_sha256') != tuning_manifest_sha256:
        raise ValueError('Selection tuning-manifest hash differs from the supplied frozen manifest.')
    unavailable = [method for method in TUNABLE if selection['selections'].get(method, {}).get('status') != 'selected']
    if unavailable:
        raise ValueError('Final manifests blocked for review; no selected setting for: ' + ', '.join(unavailable))
    choices = {}
    for method in TUNABLE:
        record = selection['selections'][method]
        score = record.get('score')
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f'{method}: selected score must be finite.')
        chosen_id = record.get('candidate_id')
        matching = [candidate for candidate in catalog[method] if candidate['candidate_id'] == chosen_id]
        if len(matching) != 1:
            raise ValueError(f'{method}: selected candidate {chosen_id!r} is not in the frozen grid.')
        choices[method] = matching[0]
    choices['smoothing-fw-exact'] = choices['smoothing-fw']
    choices['fw'] = dict(candidate_id='untuned', parameters={})
    return choices


def make_finals(campaign, choices, instances):
    primary = synthetic_instances(campaign['primary_sizes']['synthetic'], campaign['heldout_problem_seeds'], campaign['noise_levels'])
    primary.append(bmmc_instance(campaign))
    small = synthetic_instances(campaign['small_pg_size'], campaign['heldout_problem_seeds'], campaign['noise_levels'])
    result = {'final_product': [], 'final_randomized': [], 'small_pg': []}
    final_seeds = campaign['heldout_stochastic_seeds']
    for method in campaign['final_product_methods']:
        seeds = final_seeds if method in STOCHASTIC else [final_seeds[0]]
        for description, seed in itertools.product(primary, seeds):
            result['final_product'].append(make_run(
                'final_product', method, choices[method], description, seed, 100,
                campaign['final_product_budget_seconds'], campaign['final_product_checkpoints_seconds'],
                'product', campaign, instances))
    paired = list(zip(final_seeds, campaign['randomized_initialization_seeds'], strict=True))
    for method, description, (seed, init_seed) in itertools.product(campaign['final_randomized_methods'], primary, paired):
        result['final_randomized'].append(make_run(
            'final_randomized', method, choices[method], description, seed, init_seed,
            campaign['final_randomized_budget_seconds'], campaign['final_randomized_checkpoints_seconds'],
            'randomized', campaign, instances))
    for method, description in itertools.product(campaign['small_pg_methods'], small):
        result['small_pg'].append(make_run(
            'small_pg', method, choices[method], description, final_seeds[0], 100,
            campaign['small_pg_budget_seconds'], campaign['small_pg_checkpoints_seconds'],
            'product', campaign, instances))
    return result


def manifest(phase, runs, campaign, catalog, study_root, selection_path=None, tuning_manifest_path=None):
    expected = campaign['expected_counts'][phase]
    if len(runs) != expected or len({run['run_id'] for run in runs}) != expected:
        raise ValueError(f'{phase}: expected {expected} unique runs, got {len(runs)}.')
    budget = sum(run['budget'] for run in runs)
    if budget != campaign['raw_solver_budget_seconds'][phase]:
        raise ValueError(f'{phase}: raw solver budget mismatch.')
    return dict(schema_version=1, phase=phase, status='ready',
                created_utc=datetime.now(timezone.utc).isoformat(), study_root=str(Path(study_root).resolve()),
                campaign_config_path=str(CAMPAIGN_PATH), campaign_config_sha256=sha256_file(CAMPAIGN_PATH),
                generator_sha256=sha256_file(__file__),
                selection_path=str(Path(selection_path).resolve()) if selection_path else None,
                selection_sha256=sha256_file(selection_path) if selection_path else None,
                tuning_manifest_path=str(Path(tuning_manifest_path).resolve()) if tuning_manifest_path else None,
                tuning_manifest_sha256=sha256_file(tuning_manifest_path) if tuning_manifest_path else None,
                selection_rule=campaign['selection'], candidate_catalog=catalog,
                run_count=len(runs), method_run_counts=dict(sorted(Counter(run['method'] for run in runs).items())),
                raw_solver_budget_seconds=budget, raw_solver_budget_gpu_hours=budget / 3600,
                budget_caveat=campaign['budget_caveat'], runs=runs)


def write_new(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    with Path(path).open('x') as stream:
        stream.write(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-root', type=Path, required=True, help='Experiment root containing canonical data/.')
    parser.add_argument('--out-dir', type=Path, required=True, help='Manifest output directory; individual existing files are never overwritten.')
    parser.add_argument('--selected', type=Path, help='Frozen schema_version=1 selection report; required before final manifests can be created.')
    parser.add_argument('--tuning-manifest', type=Path, help='Exact frozen tuning manifest; required together with --selected.')
    args = parser.parse_args()
    if bool(args.selected) != bool(args.tuning_manifest):
        parser.error('--selected and --tuning-manifest must be supplied together.')
    campaign = json.loads(CAMPAIGN_PATH.read_text())
    catalog = candidate_catalog(campaign)
    instances = CanonicalInstances(args.study_root)
    if args.selected:
        tuning = json.loads(args.tuning_manifest.read_text())
        if tuning.get('phase') != 'tuning' or tuning.get('campaign_config_sha256') != sha256_file(CAMPAIGN_PATH):
            raise ValueError('Tuning manifest does not bind to the frozen current campaign.')
        if tuning.get('candidate_catalog') != catalog:
            raise ValueError('Tuning manifest candidate catalog differs from the frozen current grid.')
        choices = selected_catalog(json.loads(args.selected.read_text()), catalog,
                                   campaign_sha256=sha256_file(CAMPAIGN_PATH),
                                   tuning_manifest_sha256=sha256_file(args.tuning_manifest))
        phases = make_finals(campaign, choices, instances)
    else:
        phases = {'tuning': make_tuning(campaign, catalog, instances)}
    documents = {phase: manifest(phase, runs, campaign, catalog, args.study_root, args.selected, args.tuning_manifest)
                 for phase, runs in phases.items()}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {phase: args.out_dir / f'{phase}.json' for phase in documents}
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError('Refusing to overwrite frozen manifests: ' + ', '.join(existing))
    for phase, document in documents.items():
        write_new(paths[phase], document)
    print(json.dumps({phase: {'manifest': str(paths[phase].resolve()), 'runs': value['run_count'],
                              'raw_solver_gpu_hours': value['raw_solver_budget_gpu_hours']}
                      for phase, value in documents.items()}, indent=2))


if __name__ == '__main__':
    main()
