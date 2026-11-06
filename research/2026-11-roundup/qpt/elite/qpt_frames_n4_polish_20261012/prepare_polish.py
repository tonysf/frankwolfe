"""One separately declared, adaptively motivated n=4 late-step experiment."""
import argparse
import json
from pathlib import Path
from run_polish import ROOT, COMMIT, GATE, digest, source, write, now

PARENT = Path('/home/tony/qpt_frames_n4_tuning_20261012')
CONFIG_ID = 'penalty64-two-phase'


def trial(channel, replicate):
    return dict(id=f'{CONFIG_ID}-c{channel}-i{replicate}-s{replicate}', config_id=CONFIG_ID,
        n_qubits=4, channel=channel, initialization_replicate=replicate, sampling_replicate=replicate,
        initialization_seed=200000+10*channel+replicate, sampling_seed=300000+100*channel+replicate,
        alpha0=5e-5/64, penalty_exponent=.25, rho_scale=4., rho_offset=8., rho_exponent=2/3,
        step_scale=4**.75, step_offset=4., step_exponent=.75,
        switch_update=48000, post_step_scale=4., post_step_offset=4., post_step_exponent=1.,
        batch_size=32, tau=4., steps=400000, metrics_every=4000, charged_row_budget=12800000)


def check_parent():
    manifest = json.loads((PARENT/'completion_manifest.json').read_text())
    selection = json.loads((PARENT/'selection_long.json').read_text())
    audit = json.loads((PARENT/'audit_long.json').read_text())
    assert manifest['status'] == 'complete' and not manifest['ruche_used'] and not manifest['final_heldout_used']
    assert selection['selected_config_ids'] == [] and selection['confirmation_status'] == 'skipped_no_joint_gate_eligible_configuration'
    assert manifest['selected_config_ids'] == [] and manifest['source_commit'] == COMMIT
    for name, sha in manifest['evidence_sha256'].items():
        assert digest(PARENT/name) == sha
    assert audit['phase_complete_and_valid'] and audit['source_commit'] == COMMIT
    controls = [r for r in audit['trials'] if r['config_id'] == 'penalty64-stepslow']
    assert len(controls) == 2 and {r['channel'] for r in controls} == {0, 1}
    resources = []
    for row in controls:
        folder = PARENT/'long'/row['id']
        for name, key in [('started.json','started_sha256'), ('summary.json','summary_sha256'),
                          ('result.json','json_sha256'), ('result.npz','npz_sha256')]:
            assert digest(folder/name) == row[key]
        assert row['status'] == 'complete' and not row['gate_pass']
        assert row['metrics']['process_fidelity'] >= .99 and row['metrics']['normalized_tp_violation'] <= .001
        assert 1e-6 < row['metrics']['rmse'] < 1e-5
        summary = json.loads((folder/'summary.json').read_text())
        assert summary['total_seconds'] < 600 and summary['peak_rss_mib'] < 1024
        resources.append({'id': row['id'], 'total_seconds': summary['total_seconds'],
                          'peak_rss_mib': summary['peak_rss_mib']})
    return resources


def make_plan(phase):
    resources = check_parent()
    trials = [trial(c, 0 if phase == 'calibration' else 1) for c in ((0,1) if phase == 'calibration' else (2,3))]
    plan = dict(schema_version=1, scope='local_frames_n4_polish', phase=phase, created_utc=now(),
        source_commit=COMMIT, source_sha256=source(), gate=GATE,
        reserved_final_channels=list(range(1000,1020)), trials=trials,
        total_charged_row_budget=25600000,
        parent_pins={str(PARENT/name):digest(PARENT/name) for name in
                     ('completion_manifest.json','audit_long.json','selection_long.json')},
        harness_sha256={name:digest(ROOT/name) for name in ('run_polish.py','run_batch.py','prepare_polish.py')},
        resources={'max_workers':2,'cpu_affinity':[0,1],'nice':10,'trial_wall_seconds':1200,'batch_wall_seconds':2400},
        resource_basis={'observed_same_backend_same_dimension_same_budget':resources,
                        'new_allocation':'two low-priority local workers; no Ruche'},
        selection_rule='Run confirmation only if both calibration endpoints pass every unchanged gate; no retuning from confirmation.',
        motivation='Separate adaptive follow-up after the frozen twelve-arm screen and long-budget stage produced no eligible recovery winner. Test whether a smaller late step reduces residual normalization error.',
        schedule={'clock':'global zero-based optimizer update', 'switch_update':48000,
                  'before':'(4/(t+4))**0.75', 'after':'4/(t+4)',
                  'momentum_reset':False,'penalty_clock_reset':False,'tp_projection':False,'resume':False},
        qualifications=[
            'This single arm is adaptively motivated by calibration outcomes, not part of the earlier frozen twelve-arm sweep.',
            'Fresh starts; all 400k updates and repeated 48k/400k work are charged. Same 12.8M sampled-row budget as the paired prior-best control.',
            'Exact initial factor and first 48k physical checkpoint/schedule agreement are audited on calibration channels only.',
            'The complete piecewise schedule is recorded explicitly; native callable metadata alone is not sufficient provenance.',
            'No optimizer update, momentum reset, TP projection, sensing design, initialization distribution or observation change.',
            'Noiseless rank-one nonconvex factorized FRAMES experiment only; no convergence-theorem, noise or higher-rank claim.',
            'Fresh confirmation channels are not the final held-out set. Channels 1000–1019 remain unused.',
            'No published-Quiroga or completed best-for-best claim. Realized native sampling batches are not archived.',
        ])
    if phase == 'confirmation':
        plan['selected_config_ids'] = [CONFIG_ID]
        plan['selection'] = {'path':str(ROOT/'selection_calibration.json'), 'sha256':digest(ROOT/'selection_calibration.json')}
    write(ROOT/f'plan_{phase}.json', plan)


def initial():
    make_plan('calibration')
    print(json.dumps({'phase':'calibration','trials':2,'updates_per_trial':400000,'rows_per_trial':12800000}))


def confirmation():
    path = ROOT/'audit_calibration.json'; report = json.loads(path.read_text())
    assert report['phase_complete_and_valid'] and report['plan_sha256'] == digest(ROOT/'plan_calibration.json')
    assert report['auditor_sha256'] == digest(ROOT/'audit_polish.py') and report['source_commit'] == COMMIT
    assert report['batch_inventory_sha256'] == digest(ROOT/'inventory_calibration.json')
    assert report['expected_trials'] == report['validated_trials'] == len(report['trials']) == 2
    assert {r['channel'] for r in report['trials']} == {0,1}
    for row in report['trials']:
        assert row['valid']
        if row['status'] == 'complete': assert row['prefix_check']['verified']
        for name, key in [('started.json','started_sha256'), ('summary.json','summary_sha256'),
                          ('result.json','json_sha256'), ('result.npz','npz_sha256')]:
            assert digest(ROOT/'calibration'/row['id']/name) == row[key]
    eligible = all(r['status'] == 'complete' and r['gate_pass'] for r in report['trials'])
    value = dict(created_utc=now(), selected_config_ids=[CONFIG_ID] if eligible else [],
        audit_sha256=digest(path), calibration_plan_sha256=digest(ROOT/'plan_calibration.json'),
        confirmation_status='planned' if eligible else 'skipped_no_joint_gate_eligible_configuration',
        rule='Both calibration channels must pass fidelity, RMSE and TP gates; no further candidate selection.')
    write(ROOT/'selection_calibration.json', value)
    if eligible: make_plan('confirmation')
    print(json.dumps(value))


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('stage',choices=['initial','confirmation'])
    globals()[p.parse_args().stage]()
