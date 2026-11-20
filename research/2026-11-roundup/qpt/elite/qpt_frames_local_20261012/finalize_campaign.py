"""Seal local calibration evidence without changing experiments or the repository."""
from pathlib import Path
import datetime
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    with Path(path).open('x') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    output = ROOT / 'completion_manifest.json'
    assert not output.exists(), 'Never overwrite completed evidence.'
    from run_local import source, COMMIT
    current_source = source()
    stages = {}
    evidence_hashes = {}
    for phase in ('screen', 'long', 'confirmation'):
        plan_path = ROOT / f'plan_{phase}.json'
        audit_path = ROOT / f'audit_{phase}.json'
        inventory_path = ROOT / f'inventory_{phase}.json'
        plan, audit, inventory = map(read, (plan_path, audit_path, inventory_path))
        assert plan['source_commit'] == COMMIT
        assert plan['source_sha256'] == current_source
        assert audit['plan_sha256'] == sha(plan_path)
        assert audit['phase_complete_and_valid'] is True
        assert inventory['status'] == 'complete'
        assert audit['expected_trials'] == audit['validated_trials'] == len(plan['trials'])
        assert {t['id'] for t in plan['trials']} == set(inventory['trials'])
        for name, checksum in plan['harness_sha256'].items():
            assert sha(ROOT / name) == checksum
        for trial in audit['trials']:
            assert trial['valid'] and trial['status'] == 'complete'
            directory = ROOT / phase / trial['id']
            for name, field in (('result.npz', 'npz_sha256'), ('result.json', 'json_sha256'),
                                ('started.json', 'started_sha256'), ('summary.json', 'summary_sha256')):
                assert sha(directory / name) == trial[field]
        stages[phase] = {
            'complete_trials': len(plan['trials']),
            'endpoint_gate_passes': audit['passing_trials'],
            'distinct_channels': len({t['channel'] for t in plan['trials']}),
            'updates': sum(t['steps'] for t in plan['trials']),
            'charged_measurement_rows': sum(t['steps'] * t['batch_size'] for t in plan['trials']),
            'numerical_failures': sum(t['status'] != 'complete' for t in audit['trials']),
        }
        for path in (plan_path, audit_path, inventory_path):
            evidence_hashes[path.name] = sha(path)
    selection = read(ROOT / 'selection_long.json')
    assert selection['status'] == 'selected'
    selected = selection['selected_config_ids']
    confirmation = read(ROOT / 'plan_confirmation.json')
    assert confirmation['selected_config_ids'] == selected
    assert confirmation['selection']['sha256'] == sha(ROOT / 'selection_long.json')
    tests = read(ROOT / 'validation_tests.json')
    assert tests['exit_code'] == 0
    parity = read(ROOT / 'backend_parity.json')
    assert parity['status'] == 'pass'
    for name in ('selection_screen.json', 'selection_long.json', 'validation_tests.json',
                 'backend_parity.json'):
        evidence_hashes[name] = sha(ROOT / name)
    full_confirmation = stages['confirmation']['endpoint_gate_passes'] == stages['confirmation']['complete_trials']
    result = {
        'schema': 'qpt_frames_local_calibration_completion_v1',
        'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'complete', 'source_commit': COMMIT,
        'experiment_scope': 'n=2, noiseless, rank-one Haar channels, full 128-row Quiroga-style design, Gell-Mann initialization',
        'stages': stages,
        'selected_config_ids': selected,
        'selected_config': confirmation['trials'][0],
        'selected_config_passes_all_confirmation_trials': full_confirmation,
        'total_updates_excluding_timing_pilot': sum(s['updates'] for s in stages.values()),
        'total_charged_measurement_rows_excluding_timing_pilot': sum(s['charged_measurement_rows'] for s in stages.values()),
        'resources': {'execution': 'local CPU only', 'logical_cpus': [0, 1], 'max_optimization_workers': 2,
                      'nice': 10, 'ruche_used': False},
        'final_heldout_used': False, 'final_heldout_channels_reserved': list(range(1000, 1020)),
        'best_for_best_ready': False,
        'remaining': ([ ] if full_confirmation else ['FRAMES did not pass every frozen confirmation run; any further tuning must be a new declared round.']) + [
            'Resolve published Quiroga adaptive-step reproduction before labeling a published-method best-for-best comparison.',
            'Freeze both methods and the final held-out evaluation and timing protocol.',
            'Extend matched sensing to higher qubit counts after the small-scale comparison is validated.'
        ],
        'qualifications': [
            'Channels 0/1 were tuning channels. Confirmation channels 2..7 were previously exposed, not unseen final tests.',
            'Two sampling replicates per channel do not make twelve independent channels.',
            'Tuning and confirmation were fresh starts; duplicated screen prefixes are charged in totals.',
            'Measurement-row counts exclude TP evaluation, checkpointing and compilation; they are not FLOP or wall-time parity.',
            'JAX CPU64 scan uses a dense bridge at n=2 and does not establish high-qubit scalability.',
            'The modified exact-line-search FGD diagnostic is not the published Quiroga method.',
            'The separate 2,000-update timing pilot is retained but excluded from audited scientific stage counts; its pre-freeze runner snapshot is not retained.',
            'All original results and source repository files were preserved; no remote writes or author messages were made.'
        ],
        'evidence_sha256': evidence_hashes,
        'script_sha256': {p.name: sha(p) for p in ROOT.glob('*.py')},
    }
    write(output, result)
    print(json.dumps({k: result[k] for k in ('status', 'selected_config_ids',
          'selected_config_passes_all_confirmation_trials', 'total_updates_excluding_timing_pilot')}))


if __name__ == '__main__':
    main()
