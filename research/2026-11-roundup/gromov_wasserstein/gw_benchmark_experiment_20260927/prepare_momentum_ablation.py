"""Bind the protocol's unretuned momentum-KL supplement to frozen base finals.

No dataset is opened. The already selected stochastic-KL cells supply hashes,
seeds, initialization and timing settings; smoothing supplies only rho0.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re

import prepare_stage_c_campaign as base

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / 'configs/momentum_ablation.json'
METHOD = 'stochastic-kl-momentum'


class NoDataset:
    def metadata(self, description):
        return {}


def resolve_selection(campaign, tuning, selection, supplement, *, campaign_hash, tuning_hash):
    if supplement.get('schema_version') != 1:
        raise ValueError('Unsupported supplemental configuration schema.')
    if supplement.get('parent_campaign_config_sha256') != campaign_hash or supplement.get('tuning_manifest_sha256') != tuning_hash:
        raise ValueError('Supplement does not bind to the frozen base campaign and tuning manifest.')
    catalog = base.candidate_catalog(campaign)
    if tuning.get('schema_version') != 1 or tuning.get('phase') != 'tuning' or tuning.get('campaign_config_sha256') != campaign_hash or tuning.get('candidate_catalog') != catalog:
        raise ValueError('Tuning manifest does not bind to the frozen base grid.')
    choices = base.selected_catalog(selection, catalog, campaign_sha256=campaign_hash, tuning_manifest_sha256=tuning_hash)
    if not str(selection.get('selection_source', '')).startswith('tuning phase only;'):
        raise ValueError('Supplement requires an authoritative tuning-only selection report.')
    for method in base.TUNABLE:
        candidate = choices[method]['candidate_id']
        audit = [row for row in selection.get('candidate_scores', []) if row.get('method') == method and row.get('candidate_id') == candidate]
        expected = sum(r['method'] == method and r['candidate_id'] == candidate for r in tuning['runs'])
        if (len(audit) != 1 or expected <= 0 or audit[0].get('eligible') is not True
                or audit[0].get('valid_runs') != expected or audit[0].get('prescribed_runs') != expected
                or audit[0].get('disqualifying_run_ids') != []
                or audit[0].get('score') != selection['selections'][method]['score']):
            raise ValueError(f'{method}: selected candidate lacks its complete eligible tuning audit.')
    return choices


def _valid_digest(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def make_supplements(campaign, choices, base_manifests, supplement, bindings):
    """Pure structural transformation of source manifests; no dataset or solver calls."""
    expected = base.make_finals(campaign, choices, NoDataset())
    kl = choices['stochastic-kl']
    smooth = choices['smoothing-fw']
    rho0 = smooth['parameters']['rho0']
    if not math.isfinite(rho0) or not 0 < rho0 <= 1 or supplement['momentum_beta'] != campaign['schedules']['beta']:
        raise ValueError('Invalid inherited momentum schedule.')
    candidate_id = f"kl_{kl['candidate_id']}_rho_{smooth['candidate_id']}"
    result = {}
    for phase, policy in supplement['phases'].items():
        comparison = policy['comparison_phase']
        original = base_manifests[comparison]
        if (original.get('schema_version') != 1 or original.get('phase') != comparison
                or original.get('campaign_config_sha256') != bindings['campaign_config_sha256']
                or original.get('tuning_manifest_sha256') != bindings['tuning_manifest_sha256']
                or original.get('selection_sha256') != bindings['selection_sha256']):
            raise ValueError('Base final manifest has stale campaign, tuning or selection provenance.')
        if original.get('run_count') != campaign['expected_counts'][comparison] or len(original['runs']) != original['run_count']:
            raise ValueError('Base final manifest is incomplete.')
        if len({r['run_id'] for r in original['runs']}) != original['run_count']:
            raise ValueError('Duplicate base final run IDs.')
        source = [r for r in original['runs'] if r['method'] == 'stochastic-kl']
        template = {r['run_id']: r for r in expected[comparison] if r['method'] == 'stochastic-kl'}
        if len(source) != policy['run_count'] or {r['run_id'] for r in source} != set(template):
            raise ValueError('Base stochastic-KL cells differ from the prescribed final coverage.')
        runs = []
        for parent in source:
            frozen = template[parent['run_id']]
            if any(parent.get(key) != value for key, value in frozen.items()):
                raise ValueError('Base stochastic-KL settings differ from the selected frozen configuration.')
            if (not _valid_digest(parent.get('instance_sha256'))
                    or set(parent.get('expected_array_sha256', {})) != {'A', 'B', 'a', 'b'}
                    or not all(_valid_digest(v) for v in parent['expected_array_sha256'].values())):
                raise ValueError('Missing or malformed inherited instance/array hashes.')
            product = parent.get('product_loss_float64')
            if isinstance(product, bool) or not isinstance(product, (int, float)) or not math.isfinite(product) or product <= 0:
                raise ValueError('Missing positive inherited product-loss reference.')
            if parent['budget'] != policy['budget_seconds'] or parent['checkpoints'] != policy['checkpoints_seconds']:
                raise ValueError('Supplement budget differs from its matching base cell.')
            run = copy.deepcopy(parent)
            run['parent_run_id'] = parent['run_id']
            run['parent_manifest_sha256'] = bindings['base_manifest_sha256'][comparison]
            run['comparison_phase'] = comparison
            run['method'] = run['implementation'] = METHOD
            run['candidate_id'] = run['config_id'] = candidate_id
            run['source_method_for_selection'] = 'stochastic-kl'
            run['source_candidate_id'] = kl['candidate_id']
            run['momentum_source_method_for_selection'] = 'smoothing-fw'
            run['momentum_source_candidate_id'] = smooth['candidate_id']
            run['selection_status'] = 'unretuned_momentum_ablation'
            run['run_id'] = f"{phase}/{METHOD}/{candidate_id}/{Path(run['instance']).stem}/s{run['seed']}_i{run['init_seed']}"
            run['parameters'].update(rho0=rho0, momentum_beta=supplement['momentum_beta'])
            run['candidate_parameters'].update(rho0=rho0, momentum_beta=supplement['momentum_beta'])
            argv = run['argv']
            if '--rho0' in argv or argv[0] != 'run_benchmark.py' or argv[argv.index('--method') + 1] != 'stochastic-kl':
                raise ValueError('Unexpected base stochastic-KL runner command.')
            argv[argv.index('--method') + 1] = METHOD
            argv.extend(['--rho0', repr(rho0)])
            runs.append(run)
        if len({r['run_id'] for r in runs}) != policy['run_count']:
            raise ValueError('Supplement run identities are not unique.')
        seconds = sum(r['budget'] for r in runs)
        if seconds != supplement['raw_solver_budget_seconds'][phase]:
            raise ValueError('Supplement total budget differs from its prospective plan.')
        result[phase] = dict(schema_version=1, phase=phase, comparison_phase=comparison, status='ready',
            created_utc=datetime.now(timezone.utc).isoformat(),
            campaign_config_sha256=bindings['campaign_config_sha256'],
            parent_campaign_config_sha256=bindings['campaign_config_sha256'],
            supplemental_config_path=bindings['supplemental_config_path'], supplemental_config_sha256=bindings['supplemental_config_sha256'],
            tuning_manifest_path=bindings['tuning_manifest_path'], tuning_manifest_sha256=bindings['tuning_manifest_sha256'],
            selection_path=bindings['selection_path'], selection_sha256=bindings['selection_sha256'],
            base_manifest_path=bindings['base_manifest_path'][comparison], base_manifest_sha256=bindings['base_manifest_sha256'][comparison],
            generator_sha256=base.sha256_file(__file__), selection_status='unretuned_momentum_ablation',
            freeze_reason=supplement['freeze_reason'], parameter_rule=supplement['parameter_rule'],
            momentum_beta=supplement['momentum_beta'], run_count=len(runs), method_run_counts={METHOD:len(runs)},
            raw_solver_budget_seconds=seconds, raw_solver_budget_gpu_hours=seconds/3600,
            budget_caveat=supplement['budget_note'], runs=runs)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected',type=Path,required=True)
    parser.add_argument('--tuning-manifest',type=Path,required=True)
    parser.add_argument('--base-product-manifest',type=Path,required=True)
    parser.add_argument('--base-randomized-manifest',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args()
    read=lambda path:json.loads(path.read_text())
    campaign=read(base.CAMPAIGN_PATH);supplement=read(CONFIG_PATH)
    config_hash=base.sha256_file(base.CAMPAIGN_PATH);tuning_hash=base.sha256_file(args.tuning_manifest)
    choices=resolve_selection(campaign,read(args.tuning_manifest),read(args.selected),supplement,
                              campaign_hash=config_hash,tuning_hash=tuning_hash)
    theory=read(ROOT/'configs/theory_config.json')
    if theory['beta'] != supplement['momentum_beta']:
        raise ValueError('Runner momentum exponent differs from the frozen supplement.')
    paths={'final_product':args.base_product_manifest,'final_randomized':args.base_randomized_manifest}
    bindings=dict(campaign_config_sha256=config_hash,tuning_manifest_path=str(args.tuning_manifest.resolve()),
        tuning_manifest_sha256=tuning_hash,selection_path=str(args.selected.resolve()),selection_sha256=base.sha256_file(args.selected),
        supplemental_config_path=str(CONFIG_PATH),supplemental_config_sha256=base.sha256_file(CONFIG_PATH),
        base_manifest_path={k:str(p.resolve()) for k,p in paths.items()},base_manifest_sha256={k:base.sha256_file(p) for k,p in paths.items()})
    documents=make_supplements(campaign,choices,{k:read(p) for k,p in paths.items()},supplement,bindings)
    targets={phase:args.out_dir/f'{phase}.json' for phase in documents}
    if any(path.exists() for path in targets.values()):
        raise FileExistsError('Refusing to overwrite a frozen supplemental manifest.')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    for phase,document in documents.items():base.write_new(targets[phase],document)
    print(json.dumps({phase:{'manifest':str(targets[phase].resolve()),'runs':doc['run_count'],'raw_solver_gpu_hours':doc['raw_solver_budget_gpu_hours']} for phase,doc in documents.items()},indent=2))


if __name__=='__main__':
    main()
