"""Freeze all algorithm choices before new data; transfer horizons use resources only."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
from run_followup import ROOT, PARENT, COMMIT, GATE, digest, source, now, write

RULE = {
    'pilot_steps': 2000,
    'estimate_400k_seconds': 'pilot.optimizer_seconds / pilot.steps * 400000 + pilot.total_seconds',
    'max_peak_rss_mib': 2048.,
    'full_steps': 400000, 'max_estimate_for_full_seconds': 600.,
    'fallback_steps': 100000, 'max_estimate_for_fallback_seconds': 300.,
    'fallback_estimate': 'estimate_400k_seconds / 4',
    'otherwise': 'omit this dimension; preserve and report pilot',
    'numerical_gate_or_loss_may_affect_horizon': False,
}


def trial(n, channel, initrep, samplerep, steps, frequency):
    return {'id': f'n{n}-c{channel}-i{initrep}-s{samplerep}',
            'config_id': 'rho-standard', 'n_qubits': n, 'channel': channel,
            'initialization_replicate': initrep, 'initialization_seed': 200000 + 10 * channel + initrep,
            'sampling_replicate': samplerep, 'sampling_seed': 300000 + 100 * channel + samplerep,
            'steps': steps, 'metrics_every': frequency, 'batch_size': 32, 'tau': math.sqrt(2 ** n),
            'alpha0': 5e-5, 'penalty_exponent': .25,
            'rho_scale': 4., 'rho_offset': 8., 'rho_exponent': 2 / 3,
            'step_scale': 4., 'step_offset': 4., 'step_exponent': 1.}


def initial():
    selection = json.loads((PARENT / 'selection_long.json').read_text())
    assert digest(PARENT / 'selection_long.json') == '64e822315a131d88aa19372aa201b564b04343cb4932a78f5b7efad97e986a98'
    assert selection['selected_config_ids'] == ['rho-standard']
    assert json.loads((PARENT / 'completion_manifest.json').read_text())['selected_config_passes_all_confirmation_trials']
    parent_pins = {str(PARENT / name): digest(PARENT / name) for name in
                   ('selection_long.json', 'audit_confirmation.json', 'completion_manifest.json', 'jax_backend.py')}
    harness = {name: digest(ROOT / name) for name in
               ('run_followup.py', 'run_batch.py', 'make_plans.py', 'backend_followup.py')}
    base = {'schema_version': 1, 'scope': 'local_frames_frozen_followup', 'created_utc': now(),
            'source_commit': COMMIT, 'source_sha256': source(), 'parent_pins': parent_pins,
            'harness_sha256': harness, 'gate': GATE, 'reserved_final_channels': list(range(1000, 1020)),
            'resource_rule': RULE,
            'qualifications': [
                'No retuning or candidate selection from these results. All runs start afresh.',
                'Robustness uses previously exposed channels, one new initializer and two new sampling streams per channel.',
                'Tail run repeats the original channel-0 trajectory beyond its narrow 400k endpoint pass; duplicated prefix charged.',
                'Dimension transfer preserves mean-loss alpha0, not summed lambda or epochs; tau=sqrt(d) and valid dimension-specific POVM.',
                'Higher dimensions use unchanged matrix-free NumPy FRAMES; n2 uses the pinned prior JAX CPU adapter.',
                'Same absolute RMSE gate across dimensions is not equivalent relative accuracy; relative residual is also reported.',
                'No Ruche, no final held-out channels, no claim of superiority to published Quiroga.',
            ]}
    phases = {
        'pilot': [trial(n, 0, 0, 0, 2000, 1000) for n in (3, 4)],
        'robustness': [trial(2, c, 1, s, 400000, 4000) for c in range(2, 8) for s in (10, 11)],
        'tail': [trial(2, 0, 0, 0, 800000, 4000)],
    }
    for phase, trials in phases.items():
        assert not (ROOT / f'plan_{phase}.json').exists()
    for phase, trials in phases.items():
        plan = {**base, 'phase': phase, 'trials': trials,
                'resources': {'max_workers': 2 if phase == 'robustness' else 1,
                              'cpu_affinity': [0, 1], 'nice': 10,
                              'batch_wall_seconds': 3600, 'trial_wall_seconds': 1200}}
        write(ROOT / f'plan_{phase}.json', plan)
    print(json.dumps({k: len(v) for k, v in phases.items()}))


def transfer():
    parent = ROOT / 'plan_pilot.json'
    pilot = json.loads(parent.read_text())
    assert pilot['resource_rule'] == RULE
    assert all(digest(ROOT / p) == h for p, h in pilot['harness_sha256'].items())
    audit_path = ROOT / 'audit_pilot.json'
    audit = json.loads(audit_path.read_text())
    assert audit['phase_complete_and_valid'] and audit['plan_sha256'] == digest(parent)
    inventory = ROOT / 'inventory_pilot.json'
    assert json.loads(inventory.read_text())['status'] == 'complete'
    assert not (ROOT / 'resource_decision.json').exists() and not (ROOT / 'plan_transfer.json').exists()
    decision = {'created_utc': now(), 'rule': RULE, 'pilot_plan_sha256': digest(parent),
                'pilot_audit_sha256': digest(audit_path), 'pilot_inventory_sha256': digest(inventory),
                'pilot_files_sha256': {}, 'dimensions': {}}
    trials = []
    for c in pilot['trials']:
        directory = ROOT / 'pilot' / c['id']
        row = json.loads((directory / 'summary.json').read_text())
        assert row['status'] == 'complete'
        estimate = row['optimizer_seconds'] / c['steps'] * 400000 + row['total_seconds']
        steps = (400000 if estimate <= 600. else (100000 if estimate / 4 <= 300. else 0)) if row['peak_rss_mib'] < 2048. else 0
        decision['dimensions'][str(c['n_qubits'])] = {
            'estimate_400k_seconds': estimate, 'peak_rss_mib': row['peak_rss_mib'],
            'selected_steps': steps, 'basis': 'resource measurements only; loss and gate ignored'}
        decision['pilot_files_sha256'][c['id']] = {
            name: digest(directory / name) for name in ('started.json', 'summary.json', 'result.json', 'result.npz')}
        if steps:
            trials.extend(trial(c['n_qubits'], ch, 0, 0, steps, 4000) for ch in (0, 1))
    write(ROOT / 'resource_decision.json', decision)
    if trials:
        plan = deepcopy(pilot)
        plan.update(phase='transfer', trials=trials, created_utc=now(),
                    resource_decision={'path': str(ROOT / 'resource_decision.json'),
                                       'sha256': digest(ROOT / 'resource_decision.json')})
        plan['resources']['max_workers'] = 2
        write(ROOT / 'plan_transfer.json', plan)
    print(json.dumps(decision['dimensions']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['initial', 'transfer'])
    args = parser.parse_args()
    initial() if args.stage == 'initial' else transfer()
