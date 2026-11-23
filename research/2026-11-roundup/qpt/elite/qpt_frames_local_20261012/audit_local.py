"""Independent, read-only, plan-complete audit of local FRAMES archives.

No optimizer/project modules are imported. Born probabilities, Gell-Mann
initialization, unitary truth, TP error and fidelity are rebuilt with NumPy.
An absent/partial trial is never mistaken for a complete phase. The CLI writes
only its explicitly requested report, exclusively (never overwrites).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = Path('/home/tony/frankwolfe')
COMMIT = '4ce1108a0692edc43b47ef465dbc74a42397765e'
GATE = {'fidelity_min': .99, 'rmse_max': 1e-6, 'normalized_tp_max': .001}
COMMON = ('measurement_loss_sum', 'measurement_loss_mean', 'rmse',
          'tp_violation', 'normalized_tp_violation', 'penalized_objective',
          'process_fidelity')
CONFIG_KEYS = ('config_id','alpha0','penalty_exponent','step_scale','step_offset',
               'step_exponent','rho_scale','rho_offset','rho_exponent','tau','batch_size')


def require(condition, explanation):
    if not condition:
        raise ValueError(explanation)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(array):
    array = np.ascontiguousarray(array)
    sha = hashlib.sha256(str((array.shape, str(array.dtype))).encode())
    sha.update(memoryview(array).cast('B'))
    return sha.hexdigest()


def close(actual, expected, label, rtol=2e-10, atol=2e-14):
    require(np.shape(actual) == np.shape(expected), label + ': shape mismatch')
    require(np.all(np.isfinite(actual)), label + ': nonfinite value')
    require(np.allclose(actual, expected, rtol=rtol, atol=atol), label + ': numerical mismatch')


def dense_design(d=4, a=.1, b=.1):
    """Build physical vectors and dense POVM from their documented formulas."""
    eye = np.eye(d, dtype=np.complex128)
    pairs = [(j, k) for j in range(d) for k in range(j + 1, d)]
    probes = [eye[j] for j in range(d)]
    probes += [(eye[j] + eye[k]) / math.sqrt(2) for j, k in pairs]
    probes += [(eye[j] + 1j * eye[k]) / math.sqrt(2) for j, k in pairs]
    effects = [a * np.outer(eye[0], eye[0])]
    for k in range(1, d):
        effects.append(b * (eye + np.outer(eye[0], eye[k]) + np.outer(eye[k], eye[0])))
    for k in range(1, d):
        effects.append(b * (eye + 1j * np.outer(eye[0], eye[k]) - 1j * np.outer(eye[k], eye[0])))
    effects.append(eye - np.sum(effects, axis=0))
    return np.asarray(probes), np.asarray(effects)


def born_values(factor, d=4, a=.1, b=.1):
    probes, effects = dense_design(d, a, b)
    kraus = np.asarray(factor).reshape(d, d, -1).transpose(2, 1, 0)
    values = []
    for probe in probes:
        outputs = kraus @ probe
        state = outputs.T @ outputs.conj()
        values.extend(np.trace(effect @ state).real for effect in effects)
    return np.asarray(values)


def independent_metrics(factor, truth, targets, alpha, d=4):
    residual = born_values(factor, d) - targets
    loss = .5 * float(residual @ residual)
    kraus = np.asarray(factor).reshape(d, d, -1).transpose(2, 1, 0)
    gram = sum(k.conj().T @ k for k in kraus)
    tp = float(np.linalg.norm(gram - np.eye(d), 'fro'))
    fidelity = float(np.sum(np.abs(truth.conj().T @ factor)**2)
                     / (np.vdot(truth, truth).real * np.vdot(factor, factor).real))
    return {'measurement_loss_sum': loss, 'measurement_loss_mean': loss / len(targets),
            'rmse': math.sqrt(2 * loss / len(targets)), 'tp_violation': tp,
            'normalized_tp_violation': tp / math.sqrt(d),
            'process_fidelity': fidelity,
            'penalized_objective': loss + len(targets) * alpha * tp**2}


def independent_initializer(seed, d=4):
    rng = np.random.default_rng(seed)
    coeff = rng.uniform(0, 1, (d*d, 1)) + 1j*rng.uniform(0, 1, (d*d, 1))
    coeff *= math.sqrt(d) / np.linalg.norm(coeff)
    eye = np.eye(d, dtype=np.complex128)
    pairs = [(j, k) for j in range(d) for k in range(j+1, d)]
    basis = [eye / math.sqrt(d)]
    basis += [(np.outer(eye[j], eye[k]) + np.outer(eye[k], eye[j])) / math.sqrt(2) for j, k in pairs]
    basis += [(-1j*np.outer(eye[j], eye[k]) + 1j*np.outer(eye[k], eye[j])) / math.sqrt(2) for j, k in pairs]
    for level in range(1, d):
        entries = np.zeros(d)
        entries[:level] = 1
        entries[level] = -level
        basis.append(np.diag(entries) / math.sqrt(level * (level+1)))
    kraus = sum(c * matrix for c, matrix in zip(coeff[:, 0], basis))
    return kraus.T.reshape(d*d, 1)


def independent_truth(seed, d=4):
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed, spawn_key=(0,))))
    matrix = (rng.standard_normal((d,d)) + 1j*rng.standard_normal((d,d))) / math.sqrt(2)
    unitary, triangular = np.linalg.qr(matrix)
    diagonal = np.diag(triangular)
    unitary *= (diagonal / np.abs(diagonal))[None, :]
    return unitary.T.reshape(d*d, 1)


def gate_and_deficit(metrics, status='complete'):
    gate = (status == 'complete' and metrics['rmse'] <= GATE['rmse_max']
            and metrics['normalized_tp_violation'] <= GATE['normalized_tp_max']
            and metrics['process_fidelity'] >= GATE['fidelity_min'])
    deficit = max(metrics['rmse']/GATE['rmse_max'],
                  metrics['normalized_tp_violation']/GATE['normalized_tp_max'],
                  max(0., 1-metrics['process_fidelity'])/(1-GATE['fidelity_min']))
    return bool(gate), float(deficit)


def gate_crossings(steps, passing):
    passing = np.asarray(passing, dtype=bool)
    first = int(steps[np.flatnonzero(passing)[0]]) if passing.any() else None
    persistent = np.logical_and.accumulate(passing[::-1])[::-1]
    stable = int(steps[np.flatnonzero(persistent)[0]]) if persistent.any() else None
    return first, stable


def check_plan(plan):
    require(plan['scope'] == 'local_n2_full_noiseless_frames_calibration', 'Wrong scope')
    require(plan['source_commit'] == COMMIT, 'Wrong source commit')
    require(plan['gate'] == GATE, 'Gate differs from preregistered common gate')
    require(plan['trials'], 'Empty plan')
    require(len({c['id'] for c in plan['trials']}) == len(plan['trials']), 'Duplicate trial IDs')
    require(plan['phase'] in ('pilot','screen','long','full_horizon','extended','confirmation','sampling_confirmation'),
            'Unknown calibration phase')
    for trial in plan['trials']:
        require(Path(trial['id']).name == trial['id'] and trial['id'] not in ('.','..'), 'Unsafe trial path')
        require(type(trial['channel']) is int and 0 <= trial['channel'] < 100,
                'Historical/final held-out channels are forbidden')
        if plan['phase'] in ('pilot', 'screen', 'long', 'full_horizon', 'extended'):
            require(trial['channel'] in (0, 1), 'Screen/extension leaks confirmation channels')
        if plan['phase'] in ('confirmation','sampling_confirmation'):
            require(trial['channel'] in range(2, 8), 'Confirmation channels must be 2..7')
            pin = plan.get('selection',{})
            require(set(pin) == {'path','sha256','parent_phase'} and pin['parent_phase'] == 'long',
                    'Confirmation needs pinned long-stage selection provenance')
        for name in ('steps', 'metrics_every', 'batch_size'):
            require(type(trial[name]) is int and trial[name] > 0, 'Invalid ' + name)
        require(trial['steps'] % trial['metrics_every'] == 0, 'Checkpoint interval does not divide horizon')
        require(trial['tau'] == 2. and trial['batch_size'] == 32, 'Radius/batch changed')
        require(type(trial.get('sampling_replicate', 0)) is int and 0 <= trial.get('sampling_replicate', 0) < 100,
                'Invalid/colliding sampling replicate')
        for name in ('alpha0', 'penalty_exponent', 'step_scale', 'step_offset', 'step_exponent',
                     'rho_scale', 'rho_offset', 'rho_exponent'):
            require(np.isfinite(trial[name]) and trial[name] > 0, 'Invalid ' + name)


def _pinned_json(pin, root, label):
    path = Path(pin['path']).resolve()
    require(path.is_relative_to(root), label + ': path is outside campaign')
    require(path.is_file() and digest(path) == pin['sha256'], label + ': checksum mismatch')
    return path, json.loads(path.read_text())


def _selection_groups(outcomes):
    groups = {}
    for row in outcomes:
        groups.setdefault(row['config_id'],[]).append(row)
    result = []
    for config_id,rows in groups.items():
        require(sorted(r['channel'] for r in rows) == [0,1], 'Selection requires paired channels 0/1')
        require(all(r['sampling_replicate'] == 0 for r in rows), 'Selection may use only sampling replicate zero')
        failed = sum(r['failed'] for r in rows)
        passes = sum(r['gate_pass'] for r in rows)
        finite = [r['deficit'] for r in rows if r['deficit'] is not None]
        persistent = [r['persistent_recorded_gate_step'] for r in rows]
        result.append(dict(config_id=config_id,failed_trials=failed,gate_passes=passes,
                           worst_endpoint_deficit=max(finite) if finite else math.inf,
                           mean_log_deficit=(sum(math.log(max(float(v),np.finfo(float).tiny)) for v in finite)/len(finite))
                                            if finite else math.inf,
                           worst_persistent_recorded_gate_step=max(persistent)
                             if passes == 2 and all(v is not None for v in persistent) else None))
    return result


def check_selection_chain(plan_path, plan, seen=None):
    """Bind each promotion to complete audited parent evidence and its rule.

    Older audit artifacts retain their original auditor checksum: matching the
    current auditor version is deliberately not required. Their pinned file
    inventories, source revision, physical outcomes and decision are checked.
    """
    plan_path = Path(plan_path).resolve()
    root = plan_path.parent
    seen = set() if seen is None else set(seen)
    require(plan_path not in seen, 'Cyclic selection provenance')
    seen.add(plan_path)
    phase = plan['phase']
    if phase in ('screen','pilot'):
        require('selection' not in plan, 'Initial phase must not hide a selection parent')
        return []
    require(phase in ('long','confirmation','sampling_confirmation'), 'Unsupported selection phase')
    expected_parent = 'screen' if phase == 'long' else 'long'
    pin = plan.get('selection',{})
    require(set(pin) == {'path','sha256','parent_phase'} and pin['parent_phase'] == expected_parent,
            'Wrong selection pin schema or parent phase')
    selection_path, selection = _pinned_json(pin,root,'Selection')
    require(selection['phase'] == expected_parent and selection['status'] == 'selected', 'Parent made no compatible selection')
    require(Path(selection['next_plan_path']).resolve() == plan_path, 'Selection points to a different next plan')
    parent_path, parent = _pinned_json(selection['input_plan'],root,'Parent plan')
    require(parent['phase'] == expected_parent, 'Parent plan phase mismatch')
    check_plan(parent)
    chain = check_selection_chain(parent_path,parent,seen)
    for key in ('source_commit','source_sha256','gate','scope','backend','final_heldout'):
        require(plan[key] == parent[key], 'Frozen parent setting changed: '+key)
    require(selection['source_commit'] == parent['source_commit'] and selection['source_sha256'] == parent['source_sha256'],
            'Selection source mismatch')
    require(plan['harness_sha256'] == selection['harness_sha256'], 'Selected harness pin mismatch')
    require(all(plan['harness_sha256'].get(k) == v for k,v in parent['harness_sha256'].items()), 'Parent harness changed')
    audit_path, report = _pinned_json(selection['input_audit'],root,'Parent audit')
    require(report['phase_complete_and_valid'] is True and report['plan_sha256'] == digest(parent_path),
            'Parent audit is incomplete or for another plan')
    require(report['source_commit'] == parent['source_commit'], 'Parent audit source mismatch')
    trials = {t['id']:t for t in parent['trials']}
    audited = {t['id']:t for t in report['trials']}
    outcomes = {t['trial_id']:t for t in selection['trial_outcomes']}
    files = selection['trial_files_sha256']
    require(report['expected_trials'] == report['validated_trials'] == len(trials)
            and len(report['trials']) == len(audited) == len(trials)
            and len(selection['trial_outcomes']) == len(outcomes) == len(trials)
            and set(audited) == set(outcomes) == set(files) == set(trials), 'Selection inventory is incomplete or duplicated')
    fields = {'started.json':'started_sha256','summary.json':'summary_sha256',
              'result.json':'json_sha256','result.npz':'npz_sha256','failure.json':'failure_sha256'}
    for trial_id,trial in trials.items():
        row, outcome = audited[trial_id],outcomes[trial_id]
        require(row['valid'] and row['archive_complete'], 'Selection used an invalid parent trial')
        inventory = {name:row[field] for name,field in fields.items() if field in row}
        require(files[trial_id] == inventory, 'Selection/audit file inventories differ')
        for name,sha in inventory.items():
            require(digest(root/expected_parent/trial_id/name) == sha, 'Parent trial file changed: '+trial_id+'/'+name)
        for key in ('config_id','channel','sampling_replicate'):
            require(outcome[key] == trial[key], 'Selection outcome/trial mismatch: '+key)
        require(outcome['status'] == row['status'] and outcome['failed'] == (row['status'] != 'complete')
                and outcome['gate_pass'] == row['gate_pass'], 'Selection outcome status/gate mismatch')
        close(outcome['deficit'],row['deficit'],'Selection deficit')
        require(outcome['persistent_recorded_gate_step'] == (row['persistent_recorded_gate_step'] if row['gate_pass'] else None),
                'Selection persistent crossing mismatch')
    grouped = _selection_groups(list(outcomes.values()))
    if expected_parent == 'screen':
        ranking = sorted(grouped,key=lambda r:(r['failed_trials'],-r['gate_passes'],
                         r['worst_endpoint_deficit'],r['mean_log_deficit'],r['config_id']))
        expected_ids = list(dict.fromkeys([r['config_id'] for r in ranking[:2]]+[parent['incumbent_config_id']]))
        require(selection['selection_rule'] == parent['screen_selection_rule'], 'Screen selection rule changed')
    else:
        eligible = [r for r in grouped if r['failed_trials'] == 0 and r['gate_passes'] == 2
                    and r['worst_persistent_recorded_gate_step'] is not None]
        eligible.sort(key=lambda r:(r['worst_persistent_recorded_gate_step'],r['worst_endpoint_deficit'],r['config_id']))
        require(eligible,'Confirmation opened without an eligible winner')
        expected_ids = [eligible[0]['config_id']]
        require(selection['selection_rule'] == parent['long_stage']['selection'], 'Long-stage selection rule changed')
    require(selection['selected_config_ids'] == plan['selected_config_ids'] == expected_ids,
            'Selected configuration does not follow the declared ranking rule')
    settings = parent['long_stage' if phase == 'long' else 'confirmation']
    channels = [0,1] if phase == 'long' else list(range(2,8))
    replicates = [0] if phase == 'long' else [0,1]
    require(settings['channels'] == channels, 'Selection channel split changed')
    if phase != 'long':
        require(settings['sampling_replicates'] == replicates, 'Confirmation sampling replicates changed')
    configs = {}
    for trial in parent['trials']:
        config = {key:trial[key] for key in CONFIG_KEYS}
        require(configs.setdefault(trial['config_id'],config) == config, 'Parent configuration varies across channels')
    expected_trials = []
    for config_id in expected_ids:
        for channel in channels:
            for replicate in replicates:
                expected_trials.append({**configs[config_id],'id':f'{config_id}-c{channel}-s{replicate}',
                                        'channel':channel,'sampling_replicate':replicate,'steps':settings['steps'],
                                        'metrics_every':parent['long_stage']['metrics_every']})
    require(plan['trials'] == expected_trials, 'Promoted trial configuration, seed, budget or inventory changed')
    require(datetime.datetime.fromisoformat(plan['created_utc']) >= datetime.datetime.fromisoformat(selection['created_utc']),
            'Plan predates the recorded selection')
    return chain+[{'phase':phase,'parent_phase':expected_parent,'parent_plan_sha256':digest(parent_path),
                   'selection_path':str(selection_path),'selection_sha256':digest(selection_path),
                   'parent_audit_sha256':digest(audit_path),'selected_config_ids':expected_ids}]


def check_source(plan, repo=REPO):
    listed = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-r', '--name-only',
                                     plan['source_commit'], '--', 'paper', 'frank_wolfe'], text=True).splitlines()
    expected = {name for name in listed if name.endswith('.py')}
    require(set(plan['source_sha256']) == expected, 'Incomplete source manifest')
    for name, sha in plan['source_sha256'].items():
        original = subprocess.check_output(['git', '-C', str(repo), 'show', plan['source_commit']+':'+name])
        require(hashlib.sha256(original).hexdigest() == sha, 'Git source mismatch: ' + name)
        require(digest(repo/name) == sha, 'Working source mismatch: ' + name)
    return len(expected)


def check_trial(root, plan_path, plan, trial):
    folder = root/plan['phase']/trial['id']
    required = ('started.json', 'result.json', 'result.npz', 'summary.json')
    missing = [name for name in required if not (folder/name).is_file()]
    if missing:
        return {'id': trial['id'], 'archive_complete': False, 'valid': False,
                'missing': missing, 'failure_record_present': (folder/'failure.json').exists()}
    started, meta, summary = (json.loads((folder/name).read_text()) for name in ('started.json','result.json','summary.json'))
    with np.load(folder/'result.npz', allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    require(json.loads(str(payload['metadata_json'])) == meta, 'JSON/NPZ metadata mismatch')
    for name in ('json', 'npz'):
        require(summary[name+'_sha256'] == digest(folder/('result.'+name)), 'Archive hash mismatch')
    require(summary['trial'] == trial and started['trial'] == trial, 'Trial/plan mismatch')
    local = meta['local_calibration']
    for name in ('trial','source_commit','source_sha256','plan_sha256','runner_sha256'):
        require(local[name] == started[name], 'Started/archive provenance differs: ' + name)
    require(started['plan_sha256'] == digest(plan_path), 'Plan hash mismatch')
    require(started['source_commit'] == plan['source_commit'] and started['source_sha256'] == plan['source_sha256'],
            'Source provenance differs from plan')
    runner_files = [root/'run_local.py', *root.glob('run_local_*.py')]
    require(any(p.is_file() and digest(p) == started['runner_sha256'] for p in runner_files),
            'Recorded runner hash has no matching preserved source')
    for filename, sha in meta['source_sha256'].items():
        path = filename if '/' in filename else 'paper/experiments/'+filename
        require(plan['source_sha256'][path] == sha, 'Native source mismatch')
    for filename, sha in plan.get('harness_sha256',{}).items():
        require(digest(root/filename) == sha, 'Frozen harness source changed: '+filename)
    for owner in (started, local):
        if 'backend_sha256' in owner:
            backend_files = [root/'jax_backend.py', *root.glob('jax_backend_*.py')]
            require(any(p.is_file() and digest(p) == owner['backend_sha256'] for p in backend_files),
                    'Recorded backend hash has no matching preserved source')
    is_jax = meta['format'] == 'qpt_frames_local_dense_jax_calibration_v1'
    require((is_jax or meta['format'] == 'qpt_quiroga_campaign') and meta['schema_version'] == 1, 'Unknown archive schema')
    if is_jax:
        require(plan['backend'] == 'jax_dense_cpu64_scan', 'JAX archive with incompatible plan backend')
        require(meta['adapter_sha256'] == plan['harness_sha256']['jax_backend.py'], 'Adapter hash mismatch')
    design = meta['design']
    require(all(design[key] == value for key,value in {'n_qubits':2,'d':4,'rows':128,'probes':16,'outcomes':8,
                                                     'povm_a':.1,'povm_b':.1}.items()), 'Wrong physical design')
    require(meta['observation_mode'] == 'noiseless' and meta['active_row_count'] == 128, 'Wrong data regime')
    conf = meta['configuration']
    expected = {'methods':['frames'],'rank':1,'initialization_basis':'gell-mann',
                'initialization_seed':200000+10*trial['channel'],
                'sampling_seed':300000+100*trial['channel']+trial.get('sampling_replicate',0),
                'frames_steps':trial['steps'],'frames_metrics_frequency':trial['metrics_every'],
                'tau':trial['tau'],'batch_size':trial['batch_size'],'frames_penalty_mode':'independent_schedule'}
    beta0 = 1/(2*trial['alpha0'])
    schedules = {'frames_smoothing_schedule': (beta0,1.,trial['penalty_exponent'],None),
                 'step_size_schedule': (trial['step_scale'],trial['step_offset'],trial['step_exponent'],1.),
                 'rho_schedule': (trial['rho_scale'],trial['rho_offset'],trial['rho_exponent'],1.)}
    if is_jax:
        require(conf == trial, 'JAX trial configuration mismatch')
        require(meta['sampling_seed'] == expected['sampling_seed'], 'JAX sampling seed mismatch')
        require(meta['initialization']['basis'] == 'gell-mann' and meta['independent_penalty'], 'JAX initialization/penalty mismatch')
        close(meta['frames_initial_tp_weight_per_row_alpha'],trial['alpha0'],'JAX alpha0')
        close(meta['frames_initial_tp_weight_lambda'],128*trial['alpha0'],'JAX lambda0')
    else:
        for key,value in expected.items():
            require(conf[key] == value, 'Wrong configuration: ' + key)
        close(conf['frames_beta0'], beta0, 'beta0')
        close(conf['frames_initial_tp_weight_per_row_alpha'], trial['alpha0'], 'alpha0')
        close(conf['frames_initial_tp_weight_lambda'], 128*trial['alpha0'], 'lambda0')
        for key,(scale,offset,exponent,cap) in schedules.items():
            require(conf[key] == {'type':'PowerSchedule','scale':scale,'offset':offset,'exponent':exponent,'cap':cap},
                    'Schedule configuration mismatch: ' + key)
    for name in ('active_rows','active_observations','initial_factor','truth_factor'):
        sha = (hashlib.sha256(memoryview(np.ascontiguousarray(payload[name])).cast('B')).hexdigest()
               if is_jax else array_digest(payload[name]))
        require(meta[name+'_sha256'] == sha, 'Payload checksum: ' + name)
    require(np.array_equal(payload['active_rows'], np.arange(128)), 'Not all canonical rows')
    for name in ('initial_factor','truth_factor','frames_final_factor'):
        require(payload[name].shape == (16,1), 'Wrong factor shape')
        require(np.all(np.isfinite(payload[name])), 'Nonfinite factor')
    close(payload['truth_factor'], independent_truth(trial['channel']), 'Independently regenerated truth')
    close(payload['initial_factor'], independent_initializer(200000+10*trial['channel']), 'Independent Gell-Mann initializer')
    close(payload['active_observations'], born_values(payload['truth_factor']), 'Independent target Born probabilities')
    require(meta['data_metadata']['channel_seed'] == trial['channel'], 'Wrong channel seed')
    endpoint = meta['endpoints' if is_jax else 'methods']['frames']
    status = endpoint['status']
    require(status == summary['status'] and status in ('complete','failed'), 'Conflicting/unknown status')
    steps = payload['frames_checkpoint_steps']
    require(steps.ndim == 1 and np.issubdtype(steps.dtype,np.integer) and steps.size >= 1
            and steps[0] == 0 and np.all(np.diff(steps) > 0), 'Invalid checkpoint steps')
    require(np.array_equal(payload['frames_cumulative_data_row_accesses'], steps*trial['batch_size']), 'Optimizer row charges mismatch')
    times = payload['frames_optimizer_seconds']
    require(times.shape == steps.shape, 'Wrong timing shape')
    if is_jax and status == 'complete':
        require(times[0] == 0 and np.isfinite(times[-1]) and times[-1] >= 0 and np.all(np.isnan(times[1:-1])),
                'JAX scan must retain measured endpoints and unavailable interior times')
        require(meta['timing']['scan_time_includes_jit_compilation'] and meta['timing']['warmup'] is False,
                'JAX timing convention changed')
        require(meta['jax']['precision'] == '64' and meta['jax']['device_platform'] == 'cpu', 'JAX platform/precision changed')
        require(meta['metric_checkpoint_row_count'] == 128*len(steps), 'Checkpoint row count mismatch')
    else:
        require(np.all(np.isfinite(times)) and np.all(times >= 0) and np.all(np.diff(times) >= 0), 'Invalid optimizer timing')
    for name in COMMON:
        values = payload['frames_'+name]
        require(values.shape == steps.shape and np.all(np.isfinite(values)), 'Invalid checkpoint metric: '+name)
    close(payload['frames_measurement_loss_mean'], .5*payload['frames_rmse']**2, 'Mean loss history')
    close(payload['frames_measurement_loss_sum'], 128*payload['frames_measurement_loss_mean'], 'Sum loss history')
    close(payload['frames_normalized_tp_violation'], payload['frames_tp_violation']/2, 'TP normalization history')
    if status == 'complete':
        require(np.array_equal(steps, np.arange(0,trial['steps']+1,trial['metrics_every'])), 'Incomplete checkpoint history')
        iteration = np.arange(trial['steps'])
        beta = beta0/(iteration+1)**trial['penalty_exponent']
        diagnostic = 'frames_diagnostic_'
        for name,values in {'smoothing_parameter':beta,'tp_weight_per_row_alpha':1/(2*beta),
                            'tp_weight_lambda':128/(2*beta),
                            'step_size':np.minimum(1,trial['step_scale']/(iteration+trial['step_offset'])**trial['step_exponent']),
                            'momentum_weight':np.minimum(1,trial['rho_scale']/(iteration+trial['rho_offset'])**trial['rho_exponent'])}.items():
            close(payload[diagnostic+name], values, 'Schedule history '+name)
        if is_jax:
            batches = payload[diagnostic+'batch_indices']
            require(batches.shape == (trial['steps'],trial['batch_size']) and np.issubdtype(batches.dtype,np.integer),
                    'Wrong saved sampling plan shape/type')
            rng = np.random.default_rng(expected['sampling_seed'])
            for start in range(0,trial['steps'],4096):
                block = batches[start:start+4096]
                require(np.array_equal(block,rng.integers(0,128,size=block.shape)), 'Saved sampling plan differs from declared stream')
            require(np.all(np.isfinite(payload[diagnostic+'final_gradient_estimate'])), 'Nonfinite final momentum state')
        checkpoint_beta = beta[np.minimum(steps, trial['steps']-1)]
        for name,values in {'checkpoint_smoothing_parameter':checkpoint_beta,
                            'checkpoint_tp_weight_per_row_alpha':1/(2*checkpoint_beta),
                            'checkpoint_tp_weight_lambda':128/(2*checkpoint_beta)}.items():
            close(payload[diagnostic+name], values, name)
        close(payload['frames_penalized_objective'], payload['frames_measurement_loss_sum']+
              128/(2*checkpoint_beta)*payload['frames_tp_violation']**2, 'Objective history')
        alpha_end = 1/(2*checkpoint_beta[-1])
        require(np.linalg.norm(payload['frames_final_factor'],2) <= trial['tau']+2e-12, 'Final factor outside declared domain')
    else:
        alpha_end = trial['alpha0']
        require(bool(endpoint['failure_reason']), 'Failed run missing failure reason')
    metric = independent_metrics(payload['frames_final_factor'],payload['truth_factor'],payload['active_observations'],alpha_end)
    for name,value in metric.items():
        close(payload['frames_'+name][-1], value, 'Independent endpoint '+name)
        close(endpoint[name], value, 'JSON endpoint '+name)
    initial = independent_metrics(payload['initial_factor'],payload['truth_factor'],payload['active_observations'],trial['alpha0'])
    for name,value in initial.items():
        close(payload['frames_'+name][0], value, 'Independent initial '+name)
    passed, deficit = gate_and_deficit(metric,status)
    require(summary['gate_pass'] == passed and local['endpoint_gate'] == passed, 'Incorrect endpoint gate')
    close(summary['optimizer_seconds'],times[-1],'Summary optimizer time')
    close(endpoint['optimizer_seconds'],times[-1],'Endpoint optimizer time')
    require(endpoint['completed_steps'] == int(steps[-1]) and endpoint['cumulative_data_row_accesses'] == int(steps[-1]*trial['batch_size']),
            'Endpoint accounting mismatch')
    close(summary['deficit'], deficit, 'Summary deficit')
    close(local['gate_deficit'], deficit, 'Metadata deficit')
    for name,key in (('rmse','rmse'),('normalized_tp','normalized_tp_violation'),('fidelity','process_fidelity')):
        close(summary[name], metric[key], 'Summary '+name)
    passing = ((payload['frames_rmse']<=1e-6) & (payload['frames_normalized_tp_violation']<=.001)
               & (payload['frames_process_fidelity']>=.99))
    first, persistent = gate_crossings(steps,passing)
    require(summary['first_recorded_gate_step'] == first and local['first_recorded_gate_step'] == first,
            'First recorded gate mismatch')
    for owner in (summary,local):
        if 'persistent_recorded_gate_step' in owner:
            require(owner['persistent_recorded_gate_step'] == persistent, 'Persistent recorded gate mismatch')
    return {'id':trial['id'],'channel':trial['channel'],'sampling_replicate':trial.get('sampling_replicate',0),
            'archive_complete':True,'valid':True,'status':status,'gate_pass':passed,'deficit':deficit,
            'metrics':metric,'first_recorded_gate_step':first,'persistent_recorded_gate_step':persistent,
            'completed_steps':int(steps[-1]),'charged_rows':int(steps[-1]*trial['batch_size']),
            'optimizer_seconds':float(times[-1]),'json_sha256':digest(folder/'result.json'),
            'npz_sha256':digest(folder/'result.npz'),
            'summary_sha256':digest(folder/'summary.json'),'started_sha256':digest(folder/'started.json'),
            'paired_arrays_sha256':{name:array_digest(payload[name]) for name in ('active_rows','active_observations','initial_factor','truth_factor')}}


def audit(plan_path, root=None, check_git=True):
    plan_path = Path(plan_path).resolve()
    root = plan_path.parent if root is None else Path(root)
    plan = json.loads(plan_path.read_text())
    check_plan(plan)
    selection_chain = check_selection_chain(plan_path,plan)
    source_count = check_source(plan) if check_git else None
    results = []
    pairs = {}
    for trial in plan['trials']:
        try:
            result = check_trial(root,plan_path,plan,trial)
            if result.get('valid'):
                prior = pairs.setdefault(trial['channel'],result['paired_arrays_sha256'])
                require(prior == result['paired_arrays_sha256'], 'Same-channel arms do not share identical data/initial factor')
        except Exception as error:
            result = {'id':trial['id'],'valid':False,'archive_complete':False,
                      'audit_error':type(error).__name__+': '+str(error)}
        results.append(result)
    valid = all(result['valid'] for result in results)
    return {'schema':'local_frames_independent_audit_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'plan_path':str(plan_path),'plan_sha256':digest(plan_path),'auditor_sha256':digest(__file__),
            'source_commit':plan['source_commit'],'source_file_count':source_count,'expected_trials':len(plan['trials']),
            'selection_chain':selection_chain,
            'validated_trials':sum(result['valid'] for result in results),'phase_complete_and_valid':valid,
            'passing_trials':sum(result.get('gate_pass',False) for result in results),
            'unique_channels':len(pairs),'trials':results,
            'qualification':'Plan-complete archive validation, independent initial/final physical metrics; no optimization or remote access. Charged rows exclude TP, checkpoint and compilation costs.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    report = audit(args.plan)
    rendered = json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as handle:
            handle.write(rendered)
    print(rendered,end='')
    raise SystemExit(0 if report['phase_complete_and_valid'] else 2)
