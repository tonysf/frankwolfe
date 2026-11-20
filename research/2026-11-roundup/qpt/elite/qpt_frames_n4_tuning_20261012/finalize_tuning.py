"""Seal a complete declared local tuning round, including negative outcomes."""
import json
import subprocess
from run_tuning import ROOT, COMMIT, digest, now, write, validate_plan, execution_prerequisites
from make_plans import audited, select_screen, select_long, trial, PREVIOUS


def main():
    path = ROOT/'completion_manifest.json'
    assert not path.exists()
    selection = json.loads((ROOT/'selection_long.json').read_text())
    chosen = selection['selected_config_ids']
    phases = ['pilot', 'screen', 'long']
    if chosen:
        assert selection['confirmation_status'] == 'planned'
        phases.append('confirmation')
    else:
        assert selection['confirmation_status'] == 'skipped_no_joint_gate_eligible_configuration'
        assert not any((ROOT/f'{kind}_confirmation.json').exists() for kind in ('plan', 'audit', 'inventory'))
    evidence, stages = {}, {}
    for phase in phases:
        plan = json.loads((ROOT/f'plan_{phase}.json').read_text())
        spec = validate_plan(plan); execution_prerequisites(plan, spec)
        audit = audited(phase)
        assert [r['id'] for r in audit['trials']] == [t['id'] for t in plan['trials']]
        assert audit['expected_trials'] == audit['validated_trials'] == len(plan['trials'])
        assert all(r['valid'] and r['status'] in ('complete', 'failed') for r in audit['trials'])
        failed = [r['id'] for r in audit['trials'] if r['status'] == 'failed']
        stages[phase] = {'planned_trials': len(plan['trials']), 'valid_archives': audit['validated_trials'],
            'gate_passes': audit['passing_trials'], 'numerical_failures': len(failed),
            'numerical_failure_trial_ids': failed,
            'updates': sum(r['completed_steps'] for r in audit['trials']),
            'charged_rows': sum(r['charged_rows'] for r in audit['trials'])}
        for kind in ('plan', 'audit', 'inventory'):
            name = f'{kind}_{phase}.json'; evidence[name] = digest(ROOT/name)
    tests = json.loads((ROOT/'validation_tests.json').read_text())
    assert tests['exit_code'] == 0 and tests['log_sha256'] == digest(ROOT/'validation_tests.log')
    for filename, sha in tests['test_sha256'].items():
        assert digest(filename) == sha
    for name in ('run_tuning.py', 'run_batch.py', 'make_plans.py', 'audit_tuning.py'):
        assert digest(ROOT/name) == tests['code_sha256'][name]
    for name in ('campaign_spec.json', 'resource_decision.json', 'selection_screen.json',
                 'selection_long.json', 'validation_tests.json', 'validation_tests.log'):
        evidence[name] = digest(ROOT/name)
    prefix_path = ROOT/'prefix_checks.json'
    if prefix_path.exists():
        prefix = json.loads(prefix_path.read_text())
        assert prefix['all_prefixes_match']
        for check in prefix['checks']:
            assert digest(check['short']) == check['short_sha256']
            assert digest(check['long']) == check['long_sha256']
        evidence[prefix_path.name] = digest(prefix_path)
    # Recompute screen nomination and the two-channel recovery eligibility.
    screen = audited('screen')
    screen_selection = json.loads((ROOT/'selection_screen.json').read_text())
    assert screen_selection['selected_config_ids'] == select_screen(screen['trials'])[0]
    assert screen_selection['audit_sha256'] == evidence['audit_screen.json']
    old = json.loads((PREVIOUS/'audit_transfer.json').read_text())
    assert old['phase_complete_and_valid'] and old['source_commit'] == COMMIT
    controls = [{**r, 'config_id': 'incumbent'} for r in old['trials'] if r['n_qubits'] == 4]
    assert len(controls) == 2 and {r['channel'] for r in controls} == {0, 1}
    incumbent = next(c for c in spec['configs'] if c['id'] == 'incumbent')
    for row in controls:
        folder = PREVIOUS/'transfer'/row['id']
        for filename, key in [('result.json', 'json_sha256'), ('result.npz', 'npz_sha256'),
                              ('summary.json', 'summary_sha256'), ('started.json', 'started_sha256')]:
            assert digest(folder/filename) == row[key]
        start = json.loads((folder/'started.json').read_text())
        expected = trial(incumbent, spec['phases']['long'], row['channel'], 0, 0)
        assert start['source_commit'] == COMMIT and start['source_sha256'] == plan['source_sha256']
        assert all(start['trial'][key] == value for key, value in expected.items()
                   if key not in ('id', 'config_id', 'charged_row_budget'))
        assert row['charged_rows'] == expected['charged_row_budget']
    assert chosen == select_long(audited('long')['trials']+controls)[0]
    assert selection['audit_sha256'] == evidence['audit_long.json']
    assert selection['archived_control_audit_sha256'] == digest(PREVIOUS/'audit_transfer.json')
    assert selection['spec_sha256'] == evidence['campaign_spec.json']
    manifest = dict(schema='qpt_frames_n4_tuning_completion_v1', status='complete', finished_utc=now(),
        source_commit=COMMIT, campaign_git_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        source_repository_modified=False, all_previous_results_preserved=True,
        ruche_used=False, final_heldout_used=False, reserved_final_channels=list(range(1000,1020)),
        stages=stages, selected_config_ids=chosen, recovery_winner_found=bool(chosen),
        confirmation_status=('complete' if chosen else selection['confirmation_status']),
        all_confirmation_gates_pass=(stages['confirmation']['gate_passes'] == stages['confirmation']['planned_trials']) if chosen else None,
        total_updates_including_pilots=sum(s['updates'] for s in stages.values()),
        total_charged_rows_including_pilots=sum(s['charged_rows'] for s in stages.values()),
        recorded_work_is_lower_bound=any(s['numerical_failures'] for s in stages.values()),
        unrecorded_partial_work_unknown=any(s['numerical_failures'] for s in stages.values()),
        repeated_prefixes_charged=True, archived_control_added_compute_this_round=0,
        best_for_best_complete=False, quiroga_run=False,
        qualifications=spec['qualifications'], evidence_sha256=evidence,
        script_sha256={p.name:digest(p) for p in ROOT.glob('*.py')})
    write(path, manifest)
    print(json.dumps({k:manifest[k] for k in ('status','selected_config_ids','confirmation_status',
                     'all_confirmation_gates_pass','total_charged_rows_including_pilots')},indent=2))


if __name__ == '__main__': main()
