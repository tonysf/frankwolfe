"""Independent physical and adversarial archive tests; no optimizer runs."""
from copy import deepcopy
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_tuning as audit


def declared(phase='screen'):
    base = dict(alpha0=5e-5, penalty_exponent=.25, step_scale=4., step_offset=4.,
                step_exponent=1., rho_scale=4., rho_offset=8., rho_exponent=2/3,
                batch_size=32, tau=4.)
    configs = [dict(base, id='a'), dict(base, id='b', batch_size=128, alpha0=5e-5/64,
                                     step_scale=4**.75, step_exponent=.75)]
    axes = {'pilot': ([0], [0], [0]), 'screen': ([0, 1], [0], [0]),
            'long': ([0, 1], [0], [0]), 'confirmation': ([2, 3], [1], [1])}
    phases = {name: dict(channels=x[0], initialization_replicates=x[1], sampling_replicates=x[2],
                         config_ids=['a', 'b'], charged_row_budget=640, checkpoint_charged_rows=128)
              for name, x in axes.items()}
    spec = dict(configs=configs, phases=phases, gate=audit.GATE,
                reserved_final_channels=list(range(1000, 1020)),
                physics_audit_helper=dict(path=str(audit.HELPER_PATH), sha256=audit.HELPER_SHA256))
    selected = configs[:1] if phase == 'confirmation' else configs
    trials = []
    for cfg, channel, init, sample in itertools.product(selected, *axes[phase]):
        b = cfg['batch_size']
        trials.append({**{k: v for k, v in cfg.items() if k != 'id'},
            'id': f"{cfg['id']}-c{channel}-i{init}-s{sample}", 'config_id': cfg['id'],
            'n_qubits': 4, 'channel': channel, 'initialization_replicate': init,
            'sampling_replicate': sample, 'initialization_seed': 200000+10*channel+init,
            'sampling_seed': 300000+100*channel+sample, 'steps': 640//b,
            'metrics_every': 128//b, 'charged_row_budget': 640})
    plan = dict(scope='local_frames_n4_tuning', phase=phase, gate=audit.GATE,
                source_commit=audit.COMMIT, source_sha256={}, parent_pins=audit.PARENT_PINS,
                harness_sha256={'run_tuning.py': 'test-runner'}, campaign_spec_sha256='test-spec',
                reserved_final_channels=list(range(1000, 1020)), trials=trials,
                total_charged_row_budget=640*len(trials))
    if phase in ('long', 'confirmation'):
        plan.update(selected_config_ids=[c['id'] for c in selected], selection={'path': 'fixture', 'sha256': 'fixture'})
    return plan, spec


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


def archive_fixture(tmp_path, config_index=0):
    """A physically coherent synthetic trace, not claimed to be optimizer output."""
    plan, spec = declared()
    c = plan['trials'][config_index*2]
    path = tmp_path/'plan_screen.json'; write_json(path, plan)
    folder = tmp_path/'screen'/c['id']; folder.mkdir(parents=True)
    for name in ('resource_decision.json', 'audit_pilot.json'):
        write_json(tmp_path/name, {'synthetic_fixture_only': True})
    initial = audit.physical.independent_initializer(c['initialization_seed'], 16)
    truth = audit.physical.independent_truth(c['channel'], 16)
    targets = audit.physical.born_values(truth, 16)
    steps = np.arange(0, c['steps']+1, c['metrics_every'])
    schedules = audit.schedule_histories(c)
    beta = schedules['smoothing_parameter'][np.minimum(steps, c['steps']-1)]
    states = [(1-t/c['steps'])*initial+(t/c['steps'])*truth for t in steps]
    metrics = [audit.physical.independent_metrics(x, truth, targets, 1/(2*b), 16) for x, b in zip(states, beta)]
    arrays = dict(active_rows=np.arange(8192), active_observations=targets, initial_factor=initial,
                  truth_factor=truth, frames_final_factor=truth, frames_checkpoint_steps=steps,
                  frames_optimizer_seconds=np.arange(len(steps), dtype=float)*.1,
                  frames_cumulative_data_row_accesses=steps*c['batch_size'])
    arrays.update({'frames_'+k: np.array([m[k] for m in metrics]) for k in audit.COMMON})
    arrays.update({'frames_diagnostic_'+k: v for k, v in schedules.items()})
    arrays.update(frames_diagnostic_checkpoint_smoothing_parameter=beta,
                  frames_diagnostic_checkpoint_tp_weight_per_row_alpha=1/(2*beta),
                  frames_diagnostic_checkpoint_tp_weight_lambda=8192/(2*beta),
                  frames_diagnostic_estimated_gap=np.zeros(c['steps']),
                  frames_diagnostic_exact_smoothed_gap=np.zeros(len(steps)))
    started = dict(trial=c, plan_sha256=audit.digest(path), source_commit=plan['source_commit'],
        source_sha256=plan['source_sha256'], harness_sha256=plan['harness_sha256'],
        parent_pins=plan['parent_pins'], runner_sha256='test-runner', campaign_spec_sha256='test-spec',
        host='elite', execution='local_cpu', cpu_affinity=[0, 1], nice=10,
        threads={k:'1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
        split='screen', charged_row_budget=c['charged_row_budget'], execution_prerequisites_sha256={
            name: audit.digest(tmp_path/name) for name in ('resource_decision.json', 'audit_pilot.json')})
    final = metrics[-1]; passed, deficit = audit.physical.gate_and_deficit(final)
    passing = [(m['rmse'] <= 1e-6 and m['normalized_tp_violation'] <= .001 and m['process_fidelity'] >= .99) for m in metrics]
    first, persistent = audit.physical.gate_crossings(steps, passing)
    conf = dict(methods=['frames'], rank=1, initialization_basis='gell-mann',
        initialization_seed=c['initialization_seed'], sampling_seed=c['sampling_seed'],
        frames_steps=c['steps'], frames_metrics_frequency=c['metrics_every'], batch_size=c['batch_size'],
        tau=4., frames_penalty_mode='independent_schedule', frames_beta0=1/(2*c['alpha0']),
        frames_initial_tp_weight_per_row_alpha=c['alpha0'], frames_initial_tp_weight_lambda=8192*c['alpha0'])
    for key, scale, offset, exponent, cap in (
        ('frames_smoothing_schedule', 1/(2*c['alpha0']), 1., c['penalty_exponent'], None),
        ('rho_schedule', c['rho_scale'], c['rho_offset'], c['rho_exponent'], 1.),
        ('step_size_schedule', c['step_scale'], c['step_offset'], c['step_exponent'], 1.)):
        conf[key] = dict(type='PowerSchedule', scale=scale, offset=offset, exponent=exponent, cap=cap)
    metadata = dict(format='qpt_quiroga_campaign', tuning_configuration=c, configuration=conf,
        design=dict(n_qubits=4, d=16, rows=8192, probes=256, outcomes=32, povm_a=1/37, povm_b=1/37),
        observation_mode='noiseless', active_row_count=8192, initialization=dict(basis='gell-mann'),
        data_metadata=dict(channel_seed=c['channel']), source_sha256={},
        tuning_backend=dict(name='structured_numpy', precision='complex128', dense_sensing_bank=False,
            resume=False, observed_batch_indices_archived=False,
            repository_runner='paper.experiments.qpt_quiroga_campaign.run_campaign'),
        tuning_provenance=dict(started, endpoint_gate=passed, gate_deficit=deficit,
            first_recorded_gate_step=first, persistent_recorded_gate_step=persistent),
        methods=dict(frames=dict(final, status='complete', failure_reason=None, completed_steps=c['steps'],
                               cumulative_data_row_accesses=c['charged_row_budget'])))
    metadata.update({k+'_sha256': audit.array_digest(arrays[k]) for k in ('active_rows', 'active_observations', 'initial_factor', 'truth_factor')})
    summary = dict(trial=c, status='complete', gate_pass=passed, deficit=deficit,
        rmse=final['rmse'], normalized_tp=final['normalized_tp_violation'], fidelity=final['process_fidelity'],
        relative_rmse=final['relative_rmse'], outcome_scaled_rmse=32*final['rmse'], optimizer_seconds=float(arrays['frames_optimizer_seconds'][-1]),
        total_seconds=10., peak_rss_mib=120., completed_steps=c['steps'], charged_rows=c['charged_row_budget'],
        charged_row_budget=c['charged_row_budget'], first_recorded_gate_step=first, persistent_recorded_gate_step=persistent)
    write_json(folder/'started.json', started)
    persist(folder, metadata, arrays, summary)
    return plan, path, c, folder, metadata, arrays, summary


def persist(folder, metadata, arrays, summary):
    write_json(folder/'result.json', metadata)
    np.savez(folder/'result.npz', **arrays, metadata_json=json.dumps(metadata, sort_keys=True))
    summary.update(json_sha256=audit.digest(folder/'result.json'), npz_sha256=audit.digest(folder/'result.npz'))
    write_json(folder/'summary.json', summary)


@pytest.mark.parametrize('phase', ['pilot', 'screen', 'long', 'confirmation'])
def test_complete_declared_cohorts(phase):
    audit.check_plan(*declared(phase))


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'seed', 'heldout', 'alpha', 'batch', 'cadence', 'budget', 'total', 'helper'])
def test_plan_tampering(mutation):
    p, s = declared()
    if mutation == 'missing': p['trials'].pop()
    elif mutation == 'duplicate': p['trials'].append(deepcopy(p['trials'][0]))
    elif mutation == 'helper': s['physics_audit_helper']['sha256'] = 'bad'
    elif mutation == 'total': p['total_charged_row_budget'] += 1
    else:
        key, value = {'seed': ('sampling_seed', 3), 'heldout': ('channel', 1000), 'alpha': ('alpha0', .001),
                      'batch': ('batch_size', 64), 'cadence': ('metrics_every', 5), 'budget': ('charged_row_budget', 1280)}[mutation]
        p['trials'][0][key] = value
    with pytest.raises(ValueError): audit.check_plan(p, s)


@pytest.mark.parametrize('config_index', [0, 1])
def test_physical_archive_and_varied_batch_schedule(tmp_path, config_index):
    p, path, c, folder, m, z, summary = archive_fixture(tmp_path, config_index)
    result = audit.check_trial(tmp_path, path, p, c)
    assert result['valid'] and result['gate_pass'] and result['charged_rows'] == 640
    assert result['metrics']['rmse'] == 0
    assert np.allclose(z['frames_diagnostic_tp_weight_lambda'], 8192*z['frames_diagnostic_tp_weight_per_row_alpha'])


@pytest.mark.parametrize('mutation', ['born', 'init', 'truth', 'factor', 'row_charge', 'beta', 'lambda', 'checkpoint_beta',
                                    'gamma', 'rho', 'loss', 'tp', 'gate', 'crossing', 'hash', 'metadata', 'batch_claim', 'cpu'])
def test_archive_tampering_even_after_outer_hashes_recomputed(tmp_path, mutation):
    p, path, c, folder, m, z, summary = archive_fixture(tmp_path)
    if mutation in ('born', 'init', 'truth'):
        key = {'born':'active_observations', 'init':'initial_factor', 'truth':'truth_factor'}[mutation]
        z[key] = z[key].copy(); z[key].flat[0] += .01
        m[key+'_sha256'] = audit.array_digest(z[key])
    elif mutation == 'factor': z['frames_final_factor'] = z['frames_final_factor']*.99
    elif mutation == 'row_charge': z['frames_cumulative_data_row_accesses'][-1] -= 32
    elif mutation in ('beta', 'lambda', 'checkpoint_beta', 'gamma', 'rho'):
        key = {'beta':'smoothing_parameter', 'lambda':'tp_weight_lambda', 'checkpoint_beta':'checkpoint_smoothing_parameter',
               'gamma':'step_size', 'rho':'momentum_weight'}[mutation]
        z['frames_diagnostic_'+key][-1] *= 1.01
    elif mutation in ('loss', 'tp'): z['frames_'+('measurement_loss_mean' if mutation == 'loss' else 'tp_violation')][-1] = .1
    elif mutation == 'gate': summary['gate_pass'] = False
    elif mutation == 'crossing': summary['persistent_recorded_gate_step'] = 0
    elif mutation == 'hash': m['initial_factor_sha256'] = 'bad'
    elif mutation == 'metadata': m['configuration']['frames_initial_tp_weight_lambda'] = c['batch_size']*c['alpha0']
    elif mutation == 'batch_claim': m['tuning_backend']['observed_batch_indices_archived'] = True
    elif mutation == 'cpu':
        started = audit.read(folder/'started.json'); started['cpu_affinity'] = [0, 1, 2]; write_json(folder/'started.json', started)
    persist(folder, m, z, summary)
    with pytest.raises(ValueError): audit.check_trial(tmp_path, path, p, c)


def test_independent_physics_without_process_space_bank():
    probes, effects = audit.physical.physical_design(16)
    assert probes.shape == (256, 16) and effects.shape == (32, 16, 16)
    np.testing.assert_allclose(effects.sum(axis=0), np.eye(16), atol=1e-15)
    assert np.linalg.eigvalsh(effects).min() > -1e-14
    identity = np.eye(16, dtype=complex).T.reshape(256, 1)
    probabilities = audit.physical.born_values(identity, 16).reshape(256, 32)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1., atol=1e-14)
    for row, outcome in ((0, 0), (16, 1), (255, 31)):
        vector = probes[row]
        expected = np.vdot(vector, effects[outcome]@vector).real
        np.testing.assert_allclose(probabilities[row, outcome], expected, atol=1e-15)
    metric = audit.physical.independent_metrics(identity, identity, probabilities.ravel(), 1e-5, 16)
    assert metric['tp_violation'] == 0 and metric['process_fidelity'] == 1 and metric['rmse'] == 0


def ranking_fixture():
    rows = []
    for cid, deficit, fidelity, passing, crossing, batch in (
        ('a', .9, .999, True, 100, 32), ('b', .8, .9999, True, 30, 128), ('c', 3., 1., False, None, 32)):
        for channel in (0, 1):
            rows.append(dict(config_id=cid, channel=channel, status='complete', gate_pass=passing,
                deficit=deficit, metrics=dict(process_fidelity=fidelity), persistent_recorded_gate_step=crossing,
                completed_steps=200, charged_rows=200*batch))
    return rows


def test_screen_diversity_and_long_uses_rows_not_update_steps():
    rows = ranking_fixture()
    assert audit.selected_from_rows(rows, 'screen')[0] == ['b', 'c']
    assert audit.selected_from_rows(rows, 'long')[0] == ['a']
    for row in rows: row['gate_pass'] = False
    assert audit.selected_from_rows(rows, 'long')[0] == []


def test_missing_archive_is_reported_not_dropped(tmp_path):
    p, _ = declared(); path = tmp_path/'plan_screen.json'; write_json(path, p)
    result = audit.check_trial(tmp_path, path, p, p['trials'][0])
    assert not result['valid'] and len(result['missing']) == 4


def test_nonnumeric_and_incomplete_histories_rejected(tmp_path):
    p, path, c, folder, m, z, summary = archive_fixture(tmp_path)
    z['frames_diagnostic_momentum_weight'][0] = np.nan
    persist(folder, m, z, summary)
    with pytest.raises(ValueError, match='nonfinite'): audit.check_trial(tmp_path, path, p, c)
