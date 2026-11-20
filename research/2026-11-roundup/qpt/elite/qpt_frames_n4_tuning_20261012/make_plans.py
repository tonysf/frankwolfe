"""Predeclared local n=4 calibration and deterministic, audit-bound selection."""
import argparse
import json
import math
from pathlib import Path
from run_tuning import ROOT, COMMIT, GATE, digest, now, source, write

PREVIOUS = Path('/home/tony/qpt_frames_local_followup_20261012')
HELPER_SHA = '8dcf5e7fab5573b8231671b1230f79baae0af0b4a8bb214cb74f725e0838b7e3'


def specification():
    base = dict(alpha0=5e-5, penalty_exponent=.25, step_scale=4., step_offset=4.,
                step_exponent=1., rho_scale=4., rho_offset=8., rho_exponent=2/3,
                batch_size=32, tau=4.)
    arms = [
        ('incumbent', {}),
        ('penalty-div8', {'alpha0': 5e-5/8}),
        ('penalty-div64', {'alpha0': 5e-5/64}),
        ('penalty-div512', {'alpha0': 5e-5/512}),
        ('penalty-slow', {'penalty_exponent': .125}),
        ('step-wide', {'step_scale': 16., 'step_offset': 16.}),
        ('step-slow', {'step_scale': 4**.75, 'step_exponent': .75}),
        ('batch128', {'batch_size': 128}),
        ('rho-half', {'rho_scale': math.sqrt(8), 'rho_exponent': .5}),
        ('penalty64-stepwide', {'alpha0': 5e-5/64, 'step_scale': 16., 'step_offset': 16.}),
        ('penalty64-stepslow', {'alpha0': 5e-5/64, 'step_scale': 4**.75, 'step_exponent': .75}),
        ('penalty64-batch128', {'alpha0': 5e-5/64, 'batch_size': 128}),
    ]
    configs = [{**base, **changes, 'id': name} for name, changes in arms]
    ids = [c['id'] for c in configs]
    def phase(channels, inits, streams, row_budget, checkpoint_rows, config_ids):
        return dict(channels=channels, initialization_replicates=inits, sampling_replicates=streams,
                    charged_row_budget=row_budget, checkpoint_charged_rows=checkpoint_rows, config_ids=config_ids)
    return {
        'schema': 'frames_n4_local_calibration_spec_v1', 'created_utc': now(),
        'configs': configs,
        'phases': {
            'pilot': phase([0], [0], [0], 64000, 32000, ['incumbent', 'penalty64-stepslow', 'batch128']),
            'screen': phase([0, 1], [0], [0], 1536000, 128000, ids),
            'long': phase([0, 1], [0], [0], 12800000, 128000, [c for c in ids if c != 'incumbent']),
            'confirmation': phase([2, 3], [1], [1], 12800000, 128000, ids),
        },
        'gate': GATE, 'reserved_final_channels': list(range(1000, 1020)),
        'physics_audit_helper': {'path': str(PREVIOUS/'audit_followup.py'), 'sha256': HELPER_SHA},
        'pilot_resource_gate': {'max_peak_rss_mib': 1024., 'max_projected_long_seconds': 1200.,
            'projection': 'optimizer_seconds*(12800000/pilot.charged_row_budget)+total_seconds',
            'numerical_success_used_for_budget_selection': False},
        'selection_rules': {
            'screen': 'Among new configurations completing both channels, nominate minimum worst-channel joint deficit and maximum worst-channel fidelity; deduplicate and fill the second slot by joint-deficit order. Ties: worst deficit, config ID for fidelity; config ID for deficit.',
            'joint_deficit': 'max(RMSE/1e-6, TPnorm/1e-3, max(0,1-F)/0.01)',
            'long': 'Consider the two nominated configurations plus the archived same-seed incumbent. Eligible only if both calibration channels pass. Rank by worst persistent recorded charged-row gate crossing, then worst endpoint joint deficit, then config ID.',
            'confirmation': 'Only an eligible frozen winner is run on n4 channels 2/3 with initialization replicate 1 and sampling replicate 1. Otherwise explicitly skip: no recovery winner. No new retuning from confirmation.',
            'further_adaptive_search_in_this_round': False,
        },
        'qualifications': [
            'Noiseless rank-one full Quiroga-style design; n=4, d=16, m=8192. No claims for noise or higher rank.',
            'All runs restart. Repeated prefixes and resource pilots count toward total work.',
            'Schedules use zero-based optimizer updates, not sampled rows. A larger batch has fewer updates at matched sampled rows and a different terminal penalty: this is joint sampling/update-budget tuning, not an isolated variance experiment.',
            'The stochastic loss is a mean. Summed-equivalent TP coefficient is lambda=m*alpha, not batch_size*alpha. Penalty growth remains positive for all candidates.',
            'The twelve-arm grid includes selected interactions but is not an exhaustive factorial search or a claim of globally optimal tuning.',
            'Selection uses truth fidelity on calibration channels only. Confirmation is not the final untouched held-out evaluation.',
            'Recorded row costs exclude TP/LMO work and metric evaluation. No equality of runtime or floating-point work is asserted.',
            'Native NumPy sampling seed/source are pinned; realized batch arrays are not archived.',
            'No Ruche, source-repository modification, or old-result overwrites. Published Quiroga remains a separate unresolved baseline.',
        ],
    }


def trial(config, phase, channel, initrep, samplerep):
    batch = config['batch_size']
    return {**{k: v for k, v in config.items() if k != 'id'},
        'id': f"{config['id']}-c{channel}-i{initrep}-s{samplerep}", 'config_id': config['id'],
        'n_qubits': 4, 'channel': channel, 'initialization_replicate': initrep,
        'initialization_seed': 200000+10*channel+initrep,
        'sampling_replicate': samplerep, 'sampling_seed': 300000+100*channel+samplerep,
        'steps': phase['charged_row_budget']//batch,
        'metrics_every': phase['checkpoint_charged_rows']//batch,
        'charged_row_budget': phase['charged_row_budget']}


def make_plan(phase, selected=None, selection=None):
    spec = json.loads((ROOT/'campaign_spec.json').read_text())
    p = spec['phases'][phase]
    ids = p['config_ids'] if selected is None else selected
    configs = {c['id']: c for c in spec['configs']}
    trials = [trial(configs[c], p, channel, initrep, stream) for c in ids
              for channel in p['channels'] for initrep in p['initialization_replicates']
              for stream in p['sampling_replicates']]
    value = dict(schema_version=1, scope='local_frames_n4_tuning', phase=phase, created_utc=now(),
        source_commit=COMMIT, source_sha256=source(),
        parent_pins={str(PREVIOUS/name): digest(PREVIOUS/name) for name in ('completion_manifest.json', 'audit_transfer.json')},
        harness_sha256={name: digest(ROOT/name) for name in ('run_tuning.py', 'run_batch.py', 'make_plans.py')},
        campaign_spec_sha256=digest(ROOT/'campaign_spec.json'), gate=GATE,
        reserved_final_channels=list(range(1000, 1020)), trials=trials,
        total_charged_row_budget=sum(t['charged_row_budget'] for t in trials),
        resources={'max_workers': 1 if phase == 'pilot' else 2, 'cpu_affinity': [0, 1], 'nice': 10,
                   'trial_wall_seconds': 120 if phase == 'pilot' else 1200,
                   'batch_wall_seconds': 300 if phase == 'pilot' else 3600})
    if selected is not None:
        value['selected_config_ids'] = selected
        value['selection'] = {'path': str(ROOT/selection), 'sha256': digest(ROOT/selection)}
    write(ROOT/f'plan_{phase}.json', value)
    return value


def audited(phase):
    path = ROOT/f'audit_{phase}.json'
    report = json.loads(path.read_text())
    assert report['phase_complete_and_valid'] and report['plan_sha256'] == digest(ROOT/f'plan_{phase}.json')
    assert report['batch_inventory_sha256'] == digest(ROOT/f'inventory_{phase}.json')
    for row in report['trials']:
        assert row['valid']
        for filename, key in [('result.json', 'json_sha256'), ('result.npz', 'npz_sha256'),
                              ('summary.json', 'summary_sha256'), ('started.json', 'started_sha256')]:
            assert digest(ROOT/phase/row['id']/filename) == row[key]
    return report


def rank_groups(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r['config_id'], []).append(r)
    values = []
    for config, group in groups.items():
        assert len(group) == 2 and {r['channel'] for r in group} == {0, 1}
        if any(r['status'] != 'complete' for r in group):
            continue
        passed = all(r['gate_pass'] for r in group)
        values.append({'config_id': config, 'worst_deficit': max(r['deficit'] for r in group),
            'worst_fidelity': min(r['metrics']['process_fidelity'] for r in group),
            'both_pass': passed,
            'worst_persistent_gate_rows': max(r['persistent_recorded_gate_step'] *
                (r['charged_rows']//r['completed_steps']) for r in group) if passed else None})
    return values


def select_screen(rows):
    groups = [g for g in rank_groups(rows) if g['config_id'] != 'incumbent']
    assert len(groups) >= 2, 'Fewer than two numerically complete new configurations; inspect failures.'
    deficit = sorted(groups, key=lambda g: (g['worst_deficit'], g['config_id']))
    fidelity = sorted(groups, key=lambda g: (-g['worst_fidelity'], g['worst_deficit'], g['config_id']))
    chosen = list(dict.fromkeys([deficit[0]['config_id'], fidelity[0]['config_id']]))
    chosen += [g['config_id'] for g in deficit if g['config_id'] not in chosen][:2-len(chosen)]
    return chosen, groups


def select_long(rows):
    groups = rank_groups(rows)
    eligible = sorted([g for g in groups if g['both_pass']],
        key=lambda g: (g['worst_persistent_gate_rows'], g['worst_deficit'], g['config_id']))
    return ([eligible[0]['config_id']] if eligible else []), groups


def initial():
    write(ROOT/'campaign_spec.json', specification())
    for phase in ('pilot', 'screen'):
        plan = make_plan(phase)
        print(json.dumps({'phase': phase, 'trials': len(plan['trials']), 'rows': plan['total_charged_row_budget']}))


def screen_ready():
    a = audited('pilot')
    spec = json.loads((ROOT/'campaign_spec.json').read_text())
    rule = spec['pilot_resource_gate']; estimates = []
    for r in a['trials']:
        summary = json.loads((ROOT/'pilot'/r['id']/'summary.json').read_text())
        estimate = summary['optimizer_seconds'] * 12800000/r['charged_rows'] + summary['total_seconds']
        assert summary['status'] == 'complete' and summary['peak_rss_mib'] < rule['max_peak_rss_mib']
        assert estimate < rule['max_projected_long_seconds'], 'Resource gate failed; no automatic budget expansion.'
        estimates.append({'id': r['id'], 'projected_long_seconds': estimate, 'peak_rss_mib': summary['peak_rss_mib']})
    write(ROOT/'resource_decision.json', {'created_utc': now(), 'pilot_audit_sha256': digest(ROOT/'audit_pilot.json'),
         'campaign_spec_sha256': digest(ROOT/'campaign_spec.json'), 'allowed': True, 'estimates': estimates,
         'quality_metrics_used_for_resource_decision': False})
    print(json.dumps(estimates))


def long():
    report = audited('screen')
    chosen, groups = select_screen(report['trials'])
    write(ROOT/'selection_screen.json', {'created_utc': now(), 'selected_config_ids': chosen, 'ranking_inputs': groups,
        'audit_sha256': digest(ROOT/'audit_screen.json'), 'spec_sha256': digest(ROOT/'campaign_spec.json'),
        'rule': specification()['selection_rules']['screen']})
    make_plan('long', chosen, 'selection_screen.json')
    print(json.dumps({'selected': chosen}))


def confirmation():
    report = audited('long')
    oldpath = PREVIOUS/'audit_transfer.json'
    old = json.loads(oldpath.read_text())
    assert old['phase_complete_and_valid'] and old['source_commit'] == COMMIT
    spec = json.loads((ROOT/'campaign_spec.json').read_text())
    incumbent = next(c for c in spec['configs'] if c['id'] == 'incumbent')
    archived = []
    for r in old['trials']:
        if r['n_qubits'] != 4: continue
        for filename, key in [('result.json', 'json_sha256'), ('result.npz', 'npz_sha256'),
                              ('summary.json', 'summary_sha256'), ('started.json', 'started_sha256')]:
            assert digest(PREVIOUS/'transfer'/r['id']/filename) == r[key]
        start = json.loads((PREVIOUS/'transfer'/r['id']/'started.json').read_text())
        expected = trial(incumbent, spec['phases']['long'], r['channel'], 0, 0)
        assert start['source_commit'] == COMMIT and start['source_sha256'] == source()
        assert all(start['trial'][key] == value for key, value in expected.items()
                   if key not in ('id', 'config_id', 'charged_row_budget'))
        assert r['charged_rows'] == expected['charged_row_budget']
        archived.append({**r, 'config_id': 'incumbent'})
    assert len(archived) == 2 and {r['channel'] for r in archived} == {0, 1}
    chosen, groups = select_long(report['trials']+archived)
    value = {'created_utc': now(), 'selected_config_ids': chosen, 'ranking_inputs': groups,
        'audit_sha256': digest(ROOT/'audit_long.json'), 'archived_control_audit_sha256': digest(oldpath),
        'spec_sha256': digest(ROOT/'campaign_spec.json'), 'rule': specification()['selection_rules']['long'],
        'confirmation_status': 'planned' if chosen else 'skipped_no_joint_gate_eligible_configuration',
        'archived_control_extra_compute_this_round': 0}
    write(ROOT/'selection_long.json', value)
    if chosen: make_plan('confirmation', chosen, 'selection_long.json')
    print(json.dumps(value))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['initial', 'screen_ready', 'long', 'confirmation'])
    globals()[parser.parse_args().stage]()
