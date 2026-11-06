"""Seal the separate one-arm local follow-up, including negative outcomes."""
import json
import subprocess
from run_polish import ROOT, COMMIT, CONFIG_ID, digest, now, write, validate_plan, confirmation_guard


def main():
    assert not (ROOT/'completion_manifest.json').exists()
    selection = json.loads((ROOT/'selection_calibration.json').read_text())
    chosen = selection['selected_config_ids']
    assert chosen in ([], [CONFIG_ID])
    phases = ['calibration']
    if chosen:
        assert selection['confirmation_status'] == 'planned'
        phases.append('confirmation')
    else:
        assert selection['confirmation_status'] == 'skipped_no_joint_gate_eligible_configuration'
        assert not any((ROOT/f'{kind}_confirmation.json').exists() for kind in ('plan', 'audit', 'inventory'))
        assert not (ROOT/'confirmation').exists()
    evidence, stages, audits = {}, {}, {}
    for phase in phases:
        plan_path = ROOT/f'plan_{phase}.json'
        plan = json.loads(plan_path.read_text()); validate_plan(plan); confirmation_guard(plan)
        audit = json.loads((ROOT/f'audit_{phase}.json').read_text())
        assert audit['phase_complete_and_valid'] and audit['auditor_sha256'] == digest(ROOT/'audit_polish.py')
        assert audit['plan_sha256'] == digest(plan_path)
        assert audit['batch_inventory_sha256'] == digest(ROOT/f'inventory_{phase}.json')
        assert audit['source_commit'] == COMMIT
        assert digest(audit['physical_helper']['path']) == audit['physical_helper']['sha256']
        assert [r['id'] for r in audit['trials']] == [t['id'] for t in plan['trials']]
        assert audit['expected_trials'] == audit['validated_trials'] == len(plan['trials']) == 2
        assert all(r['valid'] and r['status'] in ('complete', 'failed') for r in audit['trials'])
        for row in audit['trials']:
            for filename, key in [('result.json', 'json_sha256'), ('result.npz', 'npz_sha256'),
                                  ('summary.json', 'summary_sha256'), ('started.json', 'started_sha256')]:
                assert digest(ROOT/phase/row['id']/filename) == row[key]
            if phase == 'calibration' and row['status'] == 'complete':
                assert row['prefix_check']['verified']
                prefix = row['prefix_check']
                assert digest(prefix['archived_parent_npz_path']) == prefix['archived_parent_npz_sha256']
        failed = [r['id'] for r in audit['trials'] if r['status'] == 'failed']
        stages[phase] = dict(planned_trials=2, valid_archives=2, gate_passes=audit['passing_trials'],
            numerical_failures=len(failed), numerical_failure_trial_ids=failed,
            updates=sum(r['completed_steps'] for r in audit['trials']),
            charged_rows=sum(r['charged_rows'] for r in audit['trials']))
        audits[phase] = audit
        for kind in ('plan', 'audit', 'inventory'):
            name = f'{kind}_{phase}.json'; evidence[name] = digest(ROOT/name)
    eligible = all(r['status'] == 'complete' and r['gate_pass'] for r in audits['calibration']['trials'])
    assert chosen == ([CONFIG_ID] if eligible else [])
    assert selection['audit_sha256'] == evidence['audit_calibration.json']
    assert selection['calibration_plan_sha256'] == evidence['plan_calibration.json']
    tests = json.loads((ROOT/'validation_tests.json').read_text())
    assert tests['exit_code'] == 0 and tests['log_sha256'] == digest(ROOT/'validation_tests.log')
    for filename, sha in tests['test_sha256'].items():
        assert digest(filename) == sha
    for name in ('run_polish.py', 'run_batch.py', 'prepare_polish.py', 'audit_polish.py'):
        assert digest(ROOT/name) == tests['code_sha256'][name]
    for name in ('selection_calibration.json', 'validation_tests.json', 'validation_tests.log'):
        evidence[name] = digest(ROOT/name)
    failed = any(s['numerical_failures'] for s in stages.values())
    manifest = dict(schema='qpt_frames_n4_polish_completion_v1', status='complete', finished_utc=now(),
        source_commit=COMMIT, campaign_git_commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        source_repository_modified=False, all_previous_results_preserved=True,
        ruche_used=False, final_heldout_used=False, reserved_final_channels=list(range(1000,1020)),
        parent_pins=plan['parent_pins'], stages=stages, selected_config_ids=chosen, recovery_winner_found=eligible,
        confirmation_status='complete' if chosen else selection['confirmation_status'],
        all_confirmation_gates_pass=(stages['confirmation']['gate_passes'] == 2) if chosen else None,
        total_updates_including_pilots=sum(s['updates'] for s in stages.values()),
        total_charged_rows_including_pilots=sum(s['charged_rows'] for s in stages.values()),
        recorded_work_is_lower_bound=failed, unrecorded_partial_work_unknown=failed,
        repeated_prefixes_charged=True, archived_control_added_compute_this_round=0,
        best_for_best_complete=False, quiroga_run=False, qualifications=plan['qualifications'],
        evidence_sha256=evidence, script_sha256={p.name:digest(p) for p in ROOT.glob('*.py')})
    write(ROOT/'completion_manifest.json', manifest)
    print(json.dumps({k:manifest[k] for k in ('status','selected_config_ids','confirmation_status',
        'all_confirmation_gates_pass','total_charged_rows_including_pilots')}, indent=2))


if __name__ == '__main__': main()
