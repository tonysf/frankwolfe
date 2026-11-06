"""Seal the completed local test round; gate failures are results, not missing work."""
import json
from pathlib import Path
from run_followup import ROOT, COMMIT, digest, now, validate_plan, write


def main():
    destination = ROOT / 'completion_manifest.json'
    assert not destination.exists()
    evidence, stages = {}, {}
    for phase in ('pilot', 'robustness', 'tail', 'transfer'):
        paths = {kind: ROOT / f'{kind}_{phase}.json' for kind in ('plan', 'audit', 'inventory')}
        values = {kind: json.loads(path.read_text()) for kind, path in paths.items()}
        plan, audit, inventory = (values[k] for k in ('plan', 'audit', 'inventory'))
        validate_plan(plan)
        assert audit['phase_complete_and_valid'] and audit['batch_inventory_valid']
        assert audit['plan_sha256'] == digest(paths['plan'])
        assert audit['batch_inventory_sha256'] == digest(paths['inventory'])
        assert inventory['status'] == 'complete'
        assert audit['expected_trials'] == audit['validated_trials'] == len(plan['trials'])
        assert [row['id'] for row in audit['trials']] == [trial['id'] for trial in plan['trials']]
        for row in audit['trials']:
            assert row['valid']
            directory = ROOT / phase / row['id']
            for name, field in (('started.json', 'started_sha256'), ('summary.json', 'summary_sha256'),
                                ('result.json', 'json_sha256'), ('result.npz', 'npz_sha256')):
                assert digest(directory / name) == row[field]
        stages[phase] = {
            'planned_trials': len(plan['trials']), 'valid_archives': audit['validated_trials'],
            'endpoint_gate_passes': audit['passing_trials'],
            'numerical_failures': sum(r['status'] != 'complete' for r in audit['trials']),
            'updates': sum(r['completed_steps'] for r in audit['trials']),
            'charged_rows': sum(r['charged_rows'] for r in audit['trials']),
            'channel_dimension_pairs': len({(r['n_qubits'], r['channel']) for r in audit['trials']}),
            'outcomes': [{k: r[k] for k in ('id', 'n_qubits', 'channel', 'initialization_replicate',
                                          'sampling_replicate', 'status', 'gate_pass', 'metrics',
                                          'persistent_recorded_gate_step', 'tail_stability')} for r in audit['trials']],
        }
        evidence.update({p.name: digest(p) for p in paths.values()})
    validation = ROOT / 'validation_tests.json'
    assert json.loads(validation.read_text())['exit_code'] == 0
    evidence.update({p.name: digest(p) for p in (validation, ROOT / 'resource_decision.json')})
    manifest = {
        'schema': 'qpt_frames_local_followup_completion_v1', 'status': 'complete', 'finished_utc': now(),
        'source_commit': COMMIT, 'all_old_results_preserved': True, 'repository_modified': False,
        'ruche_used': False, 'final_heldout_used': False, 'reserved_final_channels': list(range(1000, 1020)),
        'optimizer_or_settings_changed_in_round': False,
        'stages': stages, 'total_updates_including_resource_pilots': sum(v['updates'] for v in stages.values()),
        'total_charged_rows_including_resource_pilots': sum(v['charged_rows'] for v in stages.values()),
        'best_for_best_complete': False,
        'qualifications': [
            'Frozen-setting validation and dimension transfer, not a new hyperparameter search.',
            'New initialization and sampling streams change together; their individual effects are not isolated.',
            'Six robustness channels, two nested sampling streams per channel, one new initializer each; not twelve independent channels.',
            'The longer tail repeats and charges the original 400k prefix; recorded checkpoints do not certify every intermediate iterate.',
            'Dimension transfer preserves alpha0 in mean-loss units and sampled-row budget, not summed lambda or logical epochs.',
            'Identical absolute RMSE gates do not imply equal relative accuracy across dimensions; relative measurement residuals are reported.',
            'n2 uses the frozen CPU JAX adapter; n3/n4 use unchanged exact matrix-free NumPy FRAMES. No fair cross-backend timing claim.',
            'Published Quiroga was not run or resolved here. Final held-out comparison remains unopened.',
        ],
        'evidence_sha256': evidence,
        'script_sha256': {p.name: digest(p) for p in ROOT.glob('*.py')},
    }
    write(destination, manifest)
    print(json.dumps({'status': 'complete', 'stages': {k: {f: v[f] for f in
          ('planned_trials', 'endpoint_gate_passes', 'numerical_failures')} for k, v in stages.items()}}))


if __name__ == '__main__':
    main()
