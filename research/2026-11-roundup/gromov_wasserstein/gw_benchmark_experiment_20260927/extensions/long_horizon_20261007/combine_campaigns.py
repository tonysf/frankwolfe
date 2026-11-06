"""Audit and combine explicitly routed 20,000-update campaigns.

Routes select an entire six-run dataset/method population before results exist.
There is no per-seed fallback or result-based choice between original and retry
attempts. Every part is audited in this interpreter using its own immutable
config, source directory, prepared inputs and result root.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    '_combined_long_horizon_report', Path(__file__).with_name('report_campaign.py'))
reporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reporter)
require = reporter.require
DATASETS = ('bmmc6224', 'snare_full', 'shareseq_full')
SHARED_FIELDS = ('final_iterations', 'final_checkpoint_iterations', 'checkpoint_seconds',
                 'final_seeds', 'initializations', 'methods', 'runner',
                 'final_marginal_l1_tolerance', 'plan_storage', 'selections_provenance')


def validate_manifest(manifest):
    require(manifest['schema_version'] == 1, 'Unsupported combination manifest schema')
    parts, routes = manifest['parts'], manifest['routes']
    require(isinstance(parts, dict) and bool(parts), 'No campaign parts')
    required = {'config', 'config_sha256', 'source', 'prepared', 'out'}
    for key, part in parts.items():
        require(isinstance(key, str) and key and required <= set(part), 'Invalid part specification')
        require(all(isinstance(part[field], str) and Path(part[field]).is_absolute()
                    for field in required - {'config_sha256'}), 'Part paths must be absolute')
        require(reporter._contract._hash(part['config_sha256']), 'Invalid bound config hash')
    require(isinstance(routes, list), 'Routes must be a list')
    identities = []
    for route in routes:
        require(set(route) == {'dataset', 'method', 'part'}, 'Invalid route fields')
        require(route['part'] in parts, 'Route references an unknown part')
        identities.append((route['dataset'], route['method']))
    expected = {(dataset, method) for dataset in DATASETS for method in reporter.METHODS}
    require(len(identities) == len(expected) and set(identities) == expected,
            'Routes must cover each of the nine dataset/method pairs exactly once')
    require(set(parts) == {route['part'] for route in routes}, 'Unreferenced campaign part')


def compatible_config(configs):
    """Merge only identical scientific contracts; wall-time limits may differ."""
    base = next(iter(configs.values()))
    require(base['final_iterations'] == 20000, 'Combined horizon must be 20000')
    require(base['final_seeds'] == [810, 811, 812]
            and base['initializations'] == ['product', 'randomized'], 'Final population changed')
    require(base['runner']['batch'] == 128
            and base['runner']['inner_tol'] == base['runner']['inner_gap_tol'] == 1e-7,
            'Probe count or proximal tolerances changed')
    combined = copy.deepcopy(base)
    combined['instances'], combined['selected_candidates'] = {}, {}
    for config in configs.values():
        for field in SHARED_FIELDS:
            require(config[field] == base[field], 'Incompatible campaign field: ' + field)
        require(config.get('progress_reference_iteration', 2000)
                == base.get('progress_reference_iteration', 2000), 'Incompatible progress reference')
        for dataset, instance in config['instances'].items():
            require(dataset in DATASETS, 'Unexpected dataset: ' + dataset)
            if dataset in combined['instances']:
                old = combined['instances'][dataset]
                for field in ('n', 'world_size', 'instance_manifest_sha256', 'array_files_sha256'):
                    require(instance[field] == old[field], 'Incompatible frozen input: ' + dataset + '/' + field)
                require(config['selected_candidates'][dataset] == combined['selected_candidates'][dataset],
                        'Incompatible selected candidates: ' + dataset)
            combined['instances'][dataset] = copy.deepcopy(instance)
            combined['selected_candidates'][dataset] = copy.deepcopy(config['selected_candidates'][dataset])
    require(set(combined['instances']) == set(DATASETS), 'All three datasets are required')
    combined['instances'] = {dataset: combined['instances'][dataset] for dataset in DATASETS}
    return combined


def combine(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest_hash = reporter.sha256(manifest_path)
    manifest = reporter.read(manifest_path)
    validate_manifest(manifest)
    configs = {}
    for key, part in manifest['parts'].items():
        require(reporter.sha256(part['config']) == part['config_sha256'], 'Bound config changed: ' + key)
        configs[key] = reporter._contract.load_config(part['config'])
    config = compatible_config(configs)
    routing = {(route['dataset'], route['method']): route['part'] for route in manifest['routes']}
    for (dataset, method), key in routing.items():
        require(dataset in configs[key]['instances'], 'Route dataset missing from part: ' + key)
    reports, selected, excluded = {}, [], []
    for key, part in manifest['parts'].items():
        report = reporter.audit(part['config'], part['source'], part['prepared'], part['out'])
        require(report['config_sha256'] == part['config_sha256'], 'Config changed during audit: ' + key)
        expected = {(dataset, method, start, seed) for dataset in configs[key]['instances']
                    for method in reporter.METHODS for start in config['initializations']
                    for seed in config['final_seeds']}
        observed = [(row['dataset'], row['method'], row['initialization'], row['seed'])
                    for row in report['runs']]
        require(len(observed) == len(expected) and set(observed) == expected,
                'Audited part population differs from its config: ' + key)
        for row in report['runs']:
            require(row['status'] in reporter.STATES, 'Unknown audited run state')
            record = dict(row, campaign_part=key)
            destination = selected if routing[(row['dataset'], row['method'])] == key else excluded
            destination.append(record)
        reports[key] = dict(part, status=report['status'], task_counts=report['task_counts'])
    require(len(selected) == 54, 'Combined routed population must contain 54 runs')
    selected.sort(key=lambda row: (DATASETS.index(row['dataset']), reporter.METHODS.index(row['method']),
                                   config['initializations'].index(row['initialization']), row['seed']))
    summaries, trajectories, progress = reporter.summarize(config, selected)
    counts = {state: sum(row['status'] == state for row in selected) for state in reporter.STATES}
    excluded_counts = {state: sum(row['status'] == state for row in excluded) for state in reporter.STATES}
    require(reporter.sha256(manifest_path) == manifest_hash, 'Combination manifest changed during audit')
    return dict(schema_version=1, status='completed' if counts['completed'] == 54 else 'incomplete',
                manifest_file=str(manifest_path), manifest_sha256=manifest_hash,
                routes=manifest['routes'], part_audits=reports, methods=reporter.LABELS,
                instances=config['instances'], selected_candidates=config['selected_candidates'],
                final_iterations=20000, checkpoints=reporter.checkpoint_iterations(config),
                final_seeds=config['final_seeds'], initializations=config['initializations'],
                task_counts=counts, runs=selected, final_summaries=summaries,
                trajectory_summaries=trajectories, progress_from_reference=progress,
                nonrouted_attempt_counts=excluded_counts, nonrouted_attempts=excluded,
                nonrouted_failed_attempts=[row for row in excluded if row['status'] in ('failed', 'invalid')],
                routing_policy='One manifest-fixed part per dataset/method for all six final seeds/starts. No per-seed fallback or result-based source selection. Nonrouted attempts are retained separately.',
                objective='Unregularized squared GW loss of the current rounded feasible plan; no entropy term or factor of one half.',
                summary_policy='Median and linear-interpolation IQR require all three configured final seeds. Missing, failed or invalid routed runs remain visible and prevent complete medians.',
                trajectory_policy='Exact observed checkpoints only; final endpoint uses the independent FP64 last-rounded validation. No interpolation or best-so-far selection.',
                precision_policy='Intermediate diagnostics use optimizer precision; final losses use independent FP64 evaluation.',
                timing_scope='Observed solver times at the same update count, not time to equal accuracy. Diagnostic evaluation and final FP64 validation are outside solver time.',
                theorem_scope='MSBPG follows the theorem parameter prescription with numerical proximal solves at the configured tolerance. Its sampled displacement statistic differs from feasible GW loss.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    report = combine(args.manifest)
    reporter.write_report(report, args.out, args.no_plots)
    print(json.dumps(dict(status=report['status'], task_counts=report['task_counts'],
                          report=str(args.out/'report.json'))))


if __name__ == '__main__':
    main()
