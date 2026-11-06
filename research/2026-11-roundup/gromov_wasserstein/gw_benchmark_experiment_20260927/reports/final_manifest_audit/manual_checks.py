"""Independent production audit; standard library only, no dataset/optimizer reads.
Run from any directory: python /path/to/reports/final_manifest_audit/manual_checks.py
"""
import hashlib
import itertools
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FILES = {
    'campaign': 'configs/stage_c_campaign.json',
    'supplement': 'configs/momentum_ablation.json',
    'selection': 'configs/stage_c_selection.json',
    'tuning': 'configs/stage_c_manifests/tuning.json',
    'report': 'reports/stage_c_tuning/report.json',
    'base_generator': 'prepare_stage_c_campaign.py',
    'supplement_generator': 'prepare_momentum_ablation.py',
    **{phase: f'configs/stage_c_final_manifests/{phase}.json' for phase in ['final_product', 'final_randomized', 'small_pg']},
    **{phase: f'configs/momentum_final_manifests/{phase}.json' for phase in ['momentum_product', 'momentum_randomized']},
}
OUT = Path(__file__).with_name('production_selection.json')
allowed_reads = {(ROOT / path).resolve() for path in FILES.values()} | {Path(__file__).resolve()}
def guard(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = Path(args[0]).resolve()
        assert path in allowed_reads or path == OUT.resolve(), f'Unexpected file access: {path}'
sys.addaudithook(guard)
blobs = {key: (ROOT / value).read_bytes() for key, value in FILES.items()}
hashes = {key: hashlib.sha256(value).hexdigest() for key, value in blobs.items()}
docs = {key: json.loads(value) for key, value in blobs.items() if FILES[key].endswith('.json')}
c, supplement, selection, tuning, report = (docs[key] for key in ['campaign', 'supplement', 'selection', 'tuning', 'report'])
checks = []
def passed(name): checks.append(name)
assert hashes['selection'] == '4cfa1ef2a2ea127ba33987378b7a89c1922cfc6bedd1a433696136641c616fe7'
assert hashes['campaign'] == '02263a91fa7dad587fd222674fe037c6b9cab79a6f6d77af033b54259dbeed5f'
assert hashes['tuning'] == '81d1c301eb4be2657f0aa5907e4be036ce3c475422fefbd016daa5c0b951465b'
assert selection['campaign_config_sha256'] == hashes['campaign'] and selection['tuning_manifest_sha256'] == hashes['tuning']
assert report['selection_sha256'] == hashes['selection'] and report['authoritative_selection'] == selection
assert selection['selection_source'].startswith('tuning phase only;')
assert all(report[key] is True for key in ['execution_complete', 'provenance_valid', 'all_outputs_valid', 'selection_ready'])
assert report['scope'] == 'campaign' and all(not report[key] for key in ['initialization_hash_audit_errors', 'selection_provenance_errors', 'duplicate_scientific_units'])
passed('exact_selection_campaign_tuning_and_complete_report_binding')

catalog = {}
for method, grid in c['candidate_grids'].items():
    axes = [(key, value) for key, value in grid.items() if isinstance(value, list)]
    fixed = {key: value for key, value in grid.items() if not isinstance(value, list)}
    catalog[method] = [{'candidate_id': f'c{i:02d}', 'parameters': fixed | dict(zip([key for key, _ in axes], values))}
                       for i, values in enumerate(itertools.product(*(values for _, values in axes)))]
    assert len(catalog[method]) == 12
assert catalog == tuning['candidate_catalog']
specs = {r['run_id']: r for r in tuning['runs']}
records = {r['run_id']: r for r in report['runs']}
assert len(specs) == len(tuning['runs']) == len(records) == len(report['runs']) == 1164
assert specs.keys() == records.keys()
scores = defaultdict(list)
statuses = Counter()
for rid, r in records.items():
    s = specs[rid]
    assert r['phase'] == 'tuning' and r['source_manifest_sha256'] == hashes['tuning']
    assert all(r[key] is True for key in ['execution_terminal', 'provenance_valid', 'eligible_final_output'])
    assert r['recorded_status'] in c['selection']['eligible_completed_statuses'] and not r['issues']
    for key in ['method', 'candidate_id', 'instance', 'family', 'n', 'noise', 'problem_seed', 'seed', 'init_seed', 'initialization', 'deterministic', 'budget', 'checkpoints', 'parameters']:
        assert r[key] == s[key], (rid, key)
    assert r['instance_sha256'] == s['instance_sha256'] and r['array_sha256'] == s['expected_array_sha256']
    numerator, denominator = r['rounded_loss_float64'], s['product_loss_float64']
    assert math.isfinite(numerator) and math.isfinite(denominator) and denominator > 0
    assert r['product_loss_float64'] == denominator and r['normalized_loss'] == numerator / denominator
    scores[(r['method'], r['candidate_id'])].append(numerator / denominator)
    statuses[r['recorded_status']] += 1
assert len(scores) == 72
score_rows = {(r['method'], r['candidate_id']): r for r in selection['candidate_scores']}
assert len(score_rows) == len(selection['candidate_scores']) == 72
selected = {}
for method, candidates in catalog.items():
    ranked = []
    for candidate in candidates:
        cid = candidate['candidate_id']; values = scores[(method, cid)]; score = statistics.median(values)
        row = score_rows[(method, cid)]
        expected = sum(s['method'] == method and s['candidate_id'] == cid for s in specs.values())
        assert row['eligible'] is True and row['score'] == score
        assert len(values) == row['prescribed_runs'] == row['valid_runs'] == expected and not row['disqualifying_run_ids']
        ranked.append((score, cid))
    score, cid = min(ranked)
    record = selection['selections'][method]
    assert record['status'] == 'selected' and record['candidate_id'] == cid and record['score'] == score and record['eligible_candidate_count'] == 12
    selected[method] = {'candidate_id': cid, 'score': score, 'parameters': next(x['parameters'] for x in candidates if x['candidate_id'] == cid), 'eligible_candidates': 12, 'prescribed_runs_per_candidate': len(scores[(method, cid)])}
passed('all_1164_tuning_rows_match_prescribed_cells_and_float64_actual_plan_normalization')
passed('all_72_flat_medians_and_six_exact_minimum_lexicographic_selections_independently_recomputed')

stochastic = {'smoothing-fw', 'stochastic-kl', 'native-sampled-gw', 'smoothing-row-projection', 'stochastic-kl-momentum'}
phases = ['final_product', 'final_randomized', 'small_pg', 'momentum_product', 'momentum_randomized']
expected_counts = dict(final_product=253, final_randomized=330, small_pg=30, momentum_product=55, momentum_randomized=55)
primary = [(f'metric512{"_noisy" if noise else ""}_s{seed}.npz', 'synthetic', 512, noise, seed) for noise, seed in itertools.product(c['noise_levels'], c['heldout_problem_seeds'])] + [('bmmc1000.npz', 'bmmc', 1000, None, None)]
small = [(f'metric128{"_noisy" if noise else ""}_s{seed}.npz', 'synthetic', 128, noise, seed) for noise, seed in itertools.product(c['noise_levels'], c['heldout_problem_seeds'])]
assert not set(c['heldout_problem_seeds']) & set(c['development_problem_seeds'])
assert not set(c['heldout_stochastic_seeds']) & set(c['development_stochastic_seeds'])
all_ids = set(); instance_evidence = {}; summaries = {}; exact_count = 0; momentum_count = 0
for phase in phases:
    manifest = docs[phase]; runs = manifest['runs']
    assert manifest['phase'] == phase and manifest['status'] == 'ready'
    assert manifest['campaign_config_sha256'] == hashes['campaign'] and manifest['selection_sha256'] == hashes['selection'] and manifest['tuning_manifest_sha256'] == hashes['tuning']
    assert datetime.fromisoformat(manifest['created_utc']) >= datetime.fromisoformat(selection['created_utc'])
    assert len(runs) == manifest['run_count'] == expected_counts[phase]
    assert dict(Counter(r['method'] for r in runs)) == manifest['method_run_counts']
    assert sum(r['budget'] for r in runs) == manifest['raw_solver_budget_seconds']
    supplement_phase = phase.startswith('momentum_')
    comparison = phase.replace('momentum_', 'final_') if supplement_phase else phase
    methods = ['stochastic-kl-momentum'] if supplement_phase else c[f'{phase}_methods']
    budget = c[f'{comparison}_budget_seconds']; checkpoints = c[f'{comparison}_checkpoints_seconds']
    descriptions = small if phase == 'small_pg' else primary
    expected_cells = set()
    for method, description in itertools.product(methods, descriptions):
        seeds = c['heldout_stochastic_seeds'] if comparison == 'final_randomized' or method in stochastic else [100]
        for seed in seeds:
            expected_cells.add((method, *description, seed, 900000 + seed if comparison == 'final_randomized' else 100))
    actual_cells = {(r['method'], r['instance'], r['family'], r['n'], r['noise'], r['problem_seed'], r['seed'], r['init_seed']) for r in runs}
    assert actual_cells == expected_cells and len(actual_cells) == len(runs)
    assert manifest['generator_sha256'] == hashes['supplement_generator' if supplement_phase else 'base_generator']
    if supplement_phase:
        assert manifest['comparison_phase'] == comparison and manifest['parent_campaign_config_sha256'] == hashes['campaign']
        assert manifest['supplemental_config_sha256'] == hashes['supplement'] and manifest['base_manifest_sha256'] == hashes[comparison]
        assert manifest['momentum_beta'] == 0.5 and manifest['selection_status'] == 'unretuned_momentum_ablation'
    else:
        assert manifest['candidate_catalog'] == catalog
    for r in runs:
        method = r['method']; rid = r['run_id']
        assert rid not in all_ids; all_ids.add(rid)
        assert r['budget'] == budget and r['checkpoints'] == checkpoints and r['n'] == r['m']
        assert r['stochastic_seed'] == r['seed'] and r['initialization_seed'] == r['init_seed']
        assert r['initialization'] == ('randomized' if comparison == 'final_randomized' else 'product')
        assert r['deterministic'] == (method not in stochastic) and r['expected_status'] == 'pending'
        source_method = 'stochastic-kl' if supplement_phase else 'smoothing-fw' if method == 'smoothing-fw-exact' else method
        choice = selected.get(source_method, {'candidate_id': 'untuned', 'parameters': {}})
        params = dict(choice['parameters']); cid = choice['candidate_id']
        if supplement_phase:
            params.update(rho0=selected['smoothing-fw']['parameters']['rho0'], momentum_beta=0.5)
            cid = f"kl_{cid}_rho_{selected['smoothing-fw']['candidate_id']}"
        assert r['candidate_id'] == r['config_id'] == cid and r['candidate_parameters'] == params
        assert r['source_method_for_selection'] == source_method
        assert rid == f"{phase}/{method}/{cid}/{Path(r['instance']).stem}/s{r['seed']}_i{r['init_seed']}"
        resolved = dict(params)
        if method in {'smoothing-fw', 'smoothing-fw-exact', 'smoothing-row-projection', 'stochastic-kl', 'stochastic-kl-momentum'}:
            resolved.update(oracle='exact' if method == 'smoothing-fw-exact' else 'sketch', batch=128)
        if method == 'smoothing-row-projection': resolved['eta'] = params['eta_coefficient'] * params['mu_scale'] / (r['m'] * r['n'])
        if method == 'pg': resolved['eta'] = params['eta_coefficient'] / r['n'] ** 2
        assert r['parameters'] == resolved
        argv = r['argv']; assert len(argv) % 2 == 1
        cli = dict(zip(argv[1::2], argv[2::2])); assert len(cli) == len(argv[1::2])
        expected_cli = {'--device': 'cuda', '--dtype': 'float32', '--threads': '1', '--seed': str(r['seed']), '--init-seed': str(r['init_seed']), '--budget': str(budget), '--checkpoints': ','.join(map(str, checkpoints))}
        if method == 'native-sampled-gw':
            assert argv[0] == 'run_native_sampled.py'; expected_cli.update({'--batch': str(params['batch']), '--epsilon': str(params['epsilon'])})
        else:
            assert argv[0] == 'run_benchmark.py'
            expected_cli.update({'--method': 'smoothing-fw' if method == 'smoothing-fw-exact' else method, '--inner-max': '1000'})
            if comparison == 'final_randomized': expected_cli['--initialization'] = 'randomized'
            if 'oracle' in resolved: expected_cli.update({'--oracle': resolved['oracle'], '--batch': '128', '--oracle-selection-status': 'stage_c_prospectively_frozen_q128_no_speed_claim'})
            keys = ['mu_scale', 'gamma0', 'rho0'] if method.startswith('smoothing-') else []
            if method in {'full-kl', 'stochastic-kl', 'stochastic-kl-momentum', 'pg'}: keys += ['eta', 'eta_power', 'inner_tol']
            if method == 'smoothing-row-projection': keys += ['eta', 'eta_power']
            if supplement_phase: keys += ['rho0']
            expected_cli.update({f'--{key.replace("_", "-")}': str(resolved[key]) for key in keys})
            if method == 'pg': expected_cli['--pg-max-n'] = '128'
        assert cli == expected_cli and r['implementation'] == cli.get('--method', 'native-sampled-gw'), rid
        assert re.fullmatch('[a-f0-9]{64}', r['instance_sha256']) and set(r['expected_array_sha256']) == {'A', 'B', 'a', 'b'}
        assert all(re.fullmatch('[a-f0-9]{64}', h) for h in r['expected_array_sha256'].values())
        evidence = (r['instance_sha256'], r['expected_array_sha256'], r['product_loss_float64'])
        assert math.isfinite(evidence[2]) and evidence[2] > 0
        assert r['instance'] not in instance_evidence or instance_evidence[r['instance']] == evidence
        instance_evidence[r['instance']] = evidence
        if method == 'smoothing-fw-exact':
            paired = next(x for x in runs if x['method'] == 'smoothing-fw' and x['instance'] == r['instance'] and x['seed'] == r['seed'] and x['init_seed'] == r['init_seed'])
            assert r['parameters'] == {**paired['parameters'], 'oracle': 'exact'} and r['candidate_parameters'] == paired['candidate_parameters']
            exact_count += 1
        if supplement_phase:
            parents = {x['run_id']: x for x in docs[comparison]['runs'] if x['method'] == 'stochastic-kl'}
            parent = parents[r['parent_run_id']]
            for key in ['instance', 'family', 'n', 'm', 'noise', 'problem_seed', 'seed', 'stochastic_seed', 'init_seed', 'initialization_seed', 'initialization', 'deterministic', 'budget', 'checkpoints', 'instance_sha256', 'expected_array_sha256', 'product_loss_float64']:
                assert r[key] == parent[key]
            assert r['parent_manifest_sha256'] == hashes[comparison] and r['comparison_phase'] == comparison
            assert r['source_candidate_id'] == parent['candidate_id'] and r['momentum_source_method_for_selection'] == 'smoothing-fw' and r['momentum_source_candidate_id'] == selected['smoothing-fw']['candidate_id']
            parent_cli = dict(zip(parent['argv'][1::2], parent['argv'][2::2]))
            assert cli == {**parent_cli, '--method': 'stochastic-kl-momentum', '--rho0': str(selected['smoothing-fw']['parameters']['rho0'])}
            momentum_count += 1
    summaries[phase] = {'runs': len(runs), 'method_run_counts': manifest['method_run_counts'], 'budget_seconds_per_run': budget, 'checkpoints_seconds': checkpoints, 'nominal_solver_seconds': manifest['raw_solver_budget_seconds'], 'sha256': hashes[phase]}
assert len(all_ids) == 723 and exact_count == 66 and momentum_count == 110
assert sum(x['nominal_solver_seconds'] for x in summaries.values()) == 104850
passed('723_unique_prescribed_final_cells_with_disjoint_synthetic_and_stochastic_heldout_seeds')
passed('all_phase_counts_30_and_300_second_budgets_checkpoint_opportunities_and_initializations_match')
passed('all_selected_parameters_resolved_step_sizes_precision_and_runner_cli_flags_match')
passed('66_exact_smoothing_cells_inherit_selected_constants_and_change_only_oracle')
passed('110_momentum_cells_exactly_pair_parent_inputs_seeds_initializations_budgets_and_kl_parameters')
passed('all_five_manifests_bind_exact_selection_config_generator_and_parent_hashes')
passed('instance_hashes_and_product_normalizers_consistent_across_methods_and_phases_without_dataset_reads')
result = {'schema_version': 1, 'status': 'passed', 'created_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'Independent manifest and completed tuning-report audit only; no held-out arrays, optimizer calls, or final-result reads.', 'checks': checks, 'check_count': len(checks), 'input_sha256': {FILES[k]: v for k, v in hashes.items()}, 'audit_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'selection': selected, 'tuning_rows_checked': 1164, 'tuning_status_counts': dict(statuses), 'candidate_scores_recomputed': 72, 'final_manifests': summaries, 'total_final_runs': 723, 'nominal_final_solver_seconds': 104850, 'nominal_final_solver_gpu_hours': 104850 / 3600, 'exact_smoothing_paired_cells_checked': exact_count, 'momentum_paired_cells_checked': momentum_count, 'unique_final_instances': len(instance_evidence), 'blocking_issues': [], 'limitations': ['Input array contents and runtime outputs are outside this metadata-only audit; dispatch and result validation enforce frozen hashes.', 'Selection scores compare settings within each method, not cross-method performance: PG uses the n=128 panel.', 'Nominal solver budgets are not wall-clock predictions; warnings and inner failures remain reported by the tuning report.'], 'reproduce': '.venv/bin/python reports/final_manifest_audit/manual_checks.py'}
OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
print(json.dumps({'status': result['status'], 'checks': len(checks), 'final_runs': 723, 'selection': {k: v['candidate_id'] for k,v in selected.items()}, 'output': str(OUT)}, indent=2))
