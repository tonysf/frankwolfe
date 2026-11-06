"""Independent read-only physical and provenance audit of frozen FRAMES followups.

No optimization or project imports. Physical checks form d-by-d Kraus/POVM
matrices, never a bank of d**2-by-d**2 sensing operators. Reports are published
only to a requested new path, without overwriting any prior evidence.
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
OLD = Path('/home/tony/qpt_frames_local_20261012')
REPO = Path('/home/tony/frankwolfe')
COMMIT = '4ce1108a0692edc43b47ef465dbc74a42397765e'
GATE = {'fidelity_min':.99,'rmse_max':1e-6,'normalized_tp_max':.001}
FROZEN = {'alpha0':5e-5,'penalty_exponent':.25,'step_scale':4.,'step_offset':4.,
          'step_exponent':1.,'rho_scale':4.,'rho_offset':8.,'rho_exponent':2/3,'batch_size':32}
COMMON = ('measurement_loss_sum','measurement_loss_mean','rmse','tp_violation',
          'normalized_tp_violation','penalized_objective','process_fidelity')
PARENT_HASHES = {
    'selection_long.json':'64e822315a131d88aa19372aa201b564b04343cb4932a78f5b7efad97e986a98',
    'audit_confirmation.json':'4035c93a365bad185ebcbf42d9bf10137b6c632c1bb1fe901130a42723bcb4f5',
    'completion_manifest.json':'3114f248f605a17996062077519aa03b85a6c83da6b46ac0f038bce2084f38a7',
    'jax_backend.py':'a96b2931d3be190c221a5025e722363d6fc0e494b1a508da9f6e3b3cbb30f44e',
}
RESOURCE_RULE = {'pilot_steps':2000,
    'estimate_400k_seconds':'pilot.optimizer_seconds / pilot.steps * 400000 + pilot.total_seconds',
    'max_peak_rss_mib':2048.,'full_steps':400000,'max_estimate_for_full_seconds':600.,
    'fallback_steps':100000,'max_estimate_for_fallback_seconds':300.,
    'fallback_estimate':'estimate_400k_seconds / 4','otherwise':'omit this dimension; preserve and report pilot',
    'numerical_gate_or_loss_may_affect_horizon':False}


def require(condition,message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def array_digest(array,shape_tag=True):
    array = np.ascontiguousarray(array)
    sha = hashlib.sha256(str((array.shape,str(array.dtype))).encode() if shape_tag else b'')
    sha.update(memoryview(array).cast('B'))
    return sha.hexdigest()


def close(actual,expected,label,rtol=3e-10,atol=3e-14):
    require(np.shape(actual) == np.shape(expected),label+': shape mismatch')
    require(np.all(np.isfinite(actual)) and np.all(np.isfinite(expected)),label+': nonfinite')
    require(np.allclose(actual,expected,rtol=rtol,atol=atol),label+': numerical mismatch')


def dimensions(n):
    require(type(n) is int and n in (2,3,4),'Unsupported dimension')
    d = 2**n
    return d,d*d,2*d**3


def povm_constants(d):
    value = 2/(4*d-1+math.sqrt(8*d-7))
    return value,value


def physical_design(d):
    a,b = povm_constants(d)
    eye = np.eye(d,dtype=np.complex128)
    pairs = [(j,k) for j in range(d) for k in range(j+1,d)]
    probes = [eye[j] for j in range(d)]
    probes += [(eye[j]+eye[k])/math.sqrt(2) for j,k in pairs]
    probes += [(eye[j]+1j*eye[k])/math.sqrt(2) for j,k in pairs]
    effects = [a*np.outer(eye[0],eye[0])]
    effects += [b*(eye+np.outer(eye[0],eye[k])+np.outer(eye[k],eye[0])) for k in range(1,d)]
    effects += [b*(eye+1j*np.outer(eye[0],eye[k])-1j*np.outer(eye[k],eye[0])) for k in range(1,d)]
    effects.append(eye-np.sum(effects,axis=0))
    return np.asarray(probes),np.asarray(effects)


def born_values(factor,d):
    probes,effects = physical_design(d)
    kraus = np.asarray(factor).reshape(d,d,-1).transpose(2,1,0)
    # Output states are only d-by-d. There is no process-space sensing bank.
    outputs = np.einsum('roi,pi->pro',kraus,probes)
    states = np.einsum('pro,prs->pos',outputs,outputs.conj())
    return np.einsum('jab,pba->pj',effects,states).real.reshape(-1)


def independent_metrics(factor,truth,targets,alpha,d):
    residual = born_values(factor,d)-targets
    loss = .5*float(residual@residual)
    kraus = np.asarray(factor).reshape(d,d,-1).transpose(2,1,0)
    gram = sum(k.conj().T@k for k in kraus)
    tp = float(np.linalg.norm(gram-np.eye(d),'fro'))
    fidelity = float(np.sum(abs(truth.conj().T@factor)**2)/(np.vdot(truth,truth).real*np.vdot(factor,factor).real))
    return {'measurement_loss_sum':loss,'measurement_loss_mean':loss/len(targets),
            'rmse':math.sqrt(2*loss/len(targets)),'relative_rmse':float(np.linalg.norm(residual)/np.linalg.norm(targets)),
            'tp_violation':tp,'normalized_tp_violation':tp/math.sqrt(d),'process_fidelity':fidelity,
            'penalized_objective':loss+len(targets)*alpha*tp**2}


def independent_initializer(seed,d):
    rng = np.random.default_rng(seed)
    coeff = rng.uniform(0,1,(d*d,1))+1j*rng.uniform(0,1,(d*d,1))
    coeff *= math.sqrt(d)/np.linalg.norm(coeff)
    eye = np.eye(d,dtype=np.complex128)
    pairs = [(j,k) for j in range(d) for k in range(j+1,d)]
    basis = [eye/math.sqrt(d)]
    basis += [(np.outer(eye[j],eye[k])+np.outer(eye[k],eye[j]))/math.sqrt(2) for j,k in pairs]
    basis += [(-1j*np.outer(eye[j],eye[k])+1j*np.outer(eye[k],eye[j]))/math.sqrt(2) for j,k in pairs]
    for level in range(1,d):
        diagonal = np.zeros(d)
        diagonal[:level] = 1
        diagonal[level] = -level
        basis.append(np.diag(diagonal)/math.sqrt(level*(level+1)))
    kraus = sum(c*matrix for c,matrix in zip(coeff[:,0],basis))
    return kraus.T.reshape(d*d,1)


def independent_truth(seed,d):
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed,spawn_key=(0,))))
    matrix = (rng.standard_normal((d,d))+1j*rng.standard_normal((d,d)))/math.sqrt(2)
    unitary,triangular = np.linalg.qr(matrix)
    diagonal = np.diag(triangular)
    unitary *= (diagonal/abs(diagonal))[None,:]
    return unitary.T.reshape(d*d,1)


def gate_and_deficit(metric,status='complete'):
    passed = (status == 'complete' and metric['process_fidelity'] >= .99 and metric['rmse'] <= 1e-6
              and metric['normalized_tp_violation'] <= .001)
    deficit = max(metric['rmse']/1e-6,metric['normalized_tp_violation']/.001,
                  max(0.,1-metric['process_fidelity'])/.01)
    return bool(passed),float(deficit)


def gate_crossings(steps,passing):
    passing = np.asarray(passing,dtype=bool)
    first = int(steps[np.flatnonzero(passing)[0]]) if passing.any() else None
    persistent = np.logical_and.accumulate(passing[::-1])[::-1]
    stable = int(steps[np.flatnonzero(persistent)[0]]) if persistent.any() else None
    return first,stable


def pair_keys(trial):
    channel = (trial['n_qubits'],trial['channel'])
    return channel,channel+(trial['initialization_replicate'],),channel+(trial['sampling_replicate'],)


def validate_trial(c):
    d,size,m = dimensions(c['n_qubits'])
    require(type(c['channel']) is int and 0 <= c['channel'] < 100,'Forbidden historical/final held-out channel')
    i,s = c['initialization_replicate'],c['sampling_replicate']
    require(type(i) is int and 0 <= i < 10,'Invalid/colliding initialization replicate')
    require(type(s) is int and 0 <= s < 100,'Invalid/colliding sampling replicate')
    require(type(c['initialization_seed']) is int and c['initialization_seed'] == 200000+10*c['channel']+i,'Initialization seed mapping changed')
    require(type(c['sampling_seed']) is int and c['sampling_seed'] == 300000+100*c['channel']+s,'Sampling seed mapping changed')
    require(all(c[key] == value for key,value in FROZEN.items()),'Frozen optimizer parameter changed')
    close(c['tau'],math.sqrt(d),'Dimension-adjusted radius',rtol=0,atol=1e-14)
    for key in ('steps','metrics_every'):
        require(type(c[key]) is int and c[key] > 0,'Invalid '+key)
    require(c['steps'] % c['metrics_every'] == 0,'Checkpoint interval does not divide horizon')
    require(Path(c['id']).name == c['id'] and c['id'] not in ('.','..'),'Unsafe trial path')


def check_plan(plan):
    require(plan['scope'] == 'local_frames_frozen_followup','Wrong experiment scope')
    require(plan['source_commit'] == COMMIT and plan['gate'] == GATE,'Source/gate changed')
    require(plan['phase'] in ('pilot','robustness','tail','transfer'),'Unknown phase')
    require(plan['reserved_final_channels'] == list(range(1000,1020)),'Reserved final cohort changed')
    require(plan['resource_rule'] == RESOURCE_RULE,'Predeclared resource rule changed')
    trials = plan['trials']
    require(trials and len({t['id'] for t in trials}) == len(trials),'Empty/duplicate trial inventory')
    for c in trials:
        validate_trial(c)
    axes = [(c['n_qubits'],c['channel'],c['initialization_replicate'],c['sampling_replicate']) for c in trials]
    require(len(set(axes)) == len(axes),'Duplicate physical replicate in phase')
    phase = plan['phase']
    if phase == 'pilot':
        expected = [(n,0,0,0) for n in (3,4)]
        require(all(c['steps'] == 2000 and c['metrics_every'] == 1000 for c in trials),'Pilot budget changed')
    elif phase == 'robustness':
        expected = [(2,c,1,s) for c in range(2,8) for s in (10,11)]
        require(all(c['steps'] == 400000 and c['metrics_every'] == 4000 for c in trials),'Robustness budget changed')
    elif phase == 'tail':
        expected = [(2,0,0,0)]
        require(all(c['steps'] == 800000 and c['metrics_every'] == 4000 for c in trials),'Tail budget changed')
    else:
        dims = sorted({c['n_qubits'] for c in trials})
        require(set(dims) <= {3,4},'Transfer dimension changed')
        expected = [(n,c,0,0) for n in dims for c in (0,1)]
        require(all(c['steps'] in (100000,400000) and c['metrics_every'] == 4000 for c in trials),'Transfer budget changed')
        require('resource_decision' in plan,'Transfer lacks resource decision pin')
    require(sorted(axes) == sorted(expected),'Phase trial inventory differs from declared axes')
    require(plan['parent_pins'] == {str(OLD/name):sha for name,sha in PARENT_HASHES.items()},'Immutable parent pin changed')


def check_sources(plan,root):
    for path,sha in plan['parent_pins'].items():
        require(digest(path) == sha,'Immutable parent file changed: '+path)
    selection = read(OLD/'selection_long.json')
    require(selection['selected_config_ids'] == ['rho-standard'] and selection['status'] == 'selected','Parent winner changed')
    old_audit = read(OLD/'audit_confirmation.json')
    require(old_audit['phase_complete_and_valid'] is True and old_audit['expected_trials'] == 12
            and old_audit['passing_trials'] == 12,'Parent confirmation not complete/pass')
    tracked = subprocess.check_output(['git','-C',str(REPO),'ls-tree','-r','--name-only',COMMIT,'--','paper','frank_wolfe'],text=True).splitlines()
    require(set(plan['source_sha256']) == {p for p in tracked if p.endswith('.py')},'Incomplete source manifest')
    for filename,sha in plan['source_sha256'].items():
        original = subprocess.check_output(['git','-C',str(REPO),'show',COMMIT+':'+filename])
        require(hashlib.sha256(original).hexdigest() == sha and digest(REPO/filename) == sha,'Repository source mismatch: '+filename)
    for filename,sha in plan['harness_sha256'].items():
        require(Path(filename).name == filename and filename not in ('.','..'),'Unsafe harness path')
        require(digest(root/filename) == sha,'Frozen harness changed: '+filename)
    return len(plan['source_sha256'])


def resource_horizon(optimizer_seconds,total_seconds,peak_rss_mib,steps=2000):
    require(steps == 2000,'Resource pilot horizon changed')
    require(all(np.isfinite(v) and v >= 0 for v in (optimizer_seconds,total_seconds,peak_rss_mib)),'Invalid resource measurements')
    estimate = optimizer_seconds/steps*400000+total_seconds
    horizon = (400000 if estimate <= 600 else (100000 if estimate/4 <= 300 else 0)) if peak_rss_mib < 2048 else 0
    return estimate,horizon


def check_resource_decision(plan,root):
    if plan['phase'] != 'transfer':
        return None
    pin = plan['resource_decision']
    path = Path(pin['path']).resolve()
    require(path == root/'resource_decision.json' and digest(path) == pin['sha256'],'Resource decision pin mismatch')
    decision = read(path)
    require(decision['rule'] == RESOURCE_RULE,'Resource decision rule changed')
    pilot_path,audit_path,inventory_path = (root/name for name in ('plan_pilot.json','audit_pilot.json','inventory_pilot.json'))
    for p,key in ((pilot_path,'pilot_plan_sha256'),(audit_path,'pilot_audit_sha256'),(inventory_path,'pilot_inventory_sha256')):
        require(digest(p) == decision[key],'Resource evidence checksum mismatch: '+key)
    pilot,report,inventory = read(pilot_path),read(audit_path),read(inventory_path)
    check_plan(pilot)
    for key in ('gate','source_commit','source_sha256','harness_sha256','parent_pins','resource_rule'):
        require(plan[key] == pilot[key],'Transfer/pilot frozen field mismatch: '+key)
    require(report['phase_complete_and_valid'] and report['plan_sha256'] == digest(pilot_path)
            and report['expected_trials'] == report['validated_trials'] == 2,'Resource decision used incomplete pilot audit')
    require(inventory['status'] == 'complete','Pilot batch inventory incomplete')
    require(inventory['plan_sha256'] == digest(pilot_path) and set(inventory['trials']) == {c['id'] for c in pilot['trials']}
            and all(v['state'] == 'finished' and v['exit_code'] == 0 for v in inventory['trials'].values()),'Pilot execution inventory mismatch')
    rows = {r['id']:r for r in report['trials']}
    require(len(report['trials']) == len(rows) == 2 and set(rows) == {c['id'] for c in pilot['trials']},'Pilot audit inventory mismatch')
    require(set(decision['dimensions']) == {'3','4'} and set(decision['pilot_files_sha256']) == set(rows),'Resource decision inventory mismatch')
    selected = {}
    for c in pilot['trials']:
        directory = root/'pilot'/c['id']
        row = read(directory/'summary.json')
        require(row['trial'] == c and row['status'] == 'complete','Invalid resource pilot endpoint')
        fields = {'started.json':'started_sha256','summary.json':'summary_sha256','result.json':'json_sha256','result.npz':'npz_sha256'}
        files = decision['pilot_files_sha256'][c['id']]
        require(set(files) == set(fields),'Incomplete resource pilot file inventory')
        for filename,field in fields.items():
            require(digest(directory/filename) == files[filename] == rows[c['id']][field],'Resource pilot file changed')
        estimate,horizon = resource_horizon(row['optimizer_seconds'],row['total_seconds'],row['peak_rss_mib'],c['steps'])
        recorded = decision['dimensions'][str(c['n_qubits'])]
        close(recorded['estimate_400k_seconds'],estimate,'Resource estimate')
        close(recorded['peak_rss_mib'],row['peak_rss_mib'],'Resource RSS')
        require(recorded['selected_steps'] == horizon,'Resource-selected budget changed')
        if horizon:
            selected[c['n_qubits']] = horizon
    require({c['n_qubits'] for c in plan['trials']} == set(selected),'Transfer omitted/added an eligible dimension')
    require(all(c['steps'] == selected[c['n_qubits']] for c in plan['trials']),'Transfer horizon differs from resource-only decision')
    require(datetime.datetime.fromisoformat(plan['created_utc']) >= datetime.datetime.fromisoformat(decision['created_utc']),
            'Transfer plan predates resource decision')
    return {'path':str(path),'sha256':digest(path),'selected_steps_by_dimension':selected,
            'qualification':'Budgets recomputed from pilot time/memory only, without using loss or gate.'}


def tail_stability(z):
    """Compare the repeated 400k prefix without inventing intermediate factors."""
    completion = read(OLD/'completion_manifest.json')
    old_audit_path = OLD/'audit_long.json'
    require(digest(old_audit_path) == completion['evidence_sha256']['audit_long.json'],'Pinned historical long audit changed')
    old_audit = read(old_audit_path)
    trial_id = 'rho-standard-c0-s0'
    row = next(r for r in old_audit['trials'] if r['id'] == trial_id)
    folder = OLD/'long'/trial_id
    for filename,field in (('result.json','json_sha256'),('result.npz','npz_sha256')):
        require(digest(folder/filename) == row[field],'Pinned historical prefix archive changed')
    steps = z['frames_checkpoint_steps']
    prefix = steps <= 400000
    with np.load(folder/'result.npz',allow_pickle=False) as old:
        require(np.array_equal(steps[prefix],old['frames_checkpoint_steps']),'Tail prefix checkpoint grid differs')
        for name in ('active_rows','active_observations','truth_factor','initial_factor'):
            require(np.array_equal(z[name],old[name]),'Tail changed physical data/initial factor')
        comparisons = {}
        # At x_400000 old beta is last-applied; new beta is next-update.
        # Thus do not falsely require endpoint penalized objectives to agree.
        for name in ('measurement_loss_sum','measurement_loss_mean','rmse','tp_violation','normalized_tp_violation','process_fidelity'):
            new,previous = z['frames_'+name][prefix],old['frames_'+name]
            close(new,previous,'Repeated prefix '+name,rtol=1e-8,atol=1e-12)
            comparisons[name] = float(np.max(abs(new-previous)))
        for name in ('momentum_weight','step_size','smoothing_parameter'):
            require(np.array_equal(z['frames_diagnostic_'+name][:400000],old['frames_diagnostic_'+name]),'Tail changed prefix schedule')
        require(np.array_equal(z['frames_diagnostic_batch_indices'][:400000],old['frames_diagnostic_batch_indices']),
                'Tail changed prefix sampling')
    after = steps > 400000
    passing = ((z['frames_rmse']<=1e-6)&(z['frames_normalized_tp_violation']<=.001)&(z['frames_process_fidelity']>=.99))
    failures = steps[after & ~passing]
    return {'prefix_metrics_match':True,'prefix_max_abs_metric_differences':comparisons,
            'prefix_reference_npz_sha256':row['npz_sha256'],'prefix_updates_repeated_and_charged':400000,
            'post_400k_recorded_checkpoints':int(np.count_nonzero(after)),
            'post_400k_passing_checkpoints':int(np.count_nonzero(after&passing)),
            'post_400k_failing_checkpoints':int(len(failures)),
            'all_recorded_after_400k_pass':bool(np.all(passing[after])),
            'first_post400k_failure_step':int(failures[0]) if len(failures) else None,
            'post_400k_rmse_min':float(np.min(z['frames_rmse'][after])),
            'post_400k_rmse_max':float(np.max(z['frames_rmse'][after])),
            'qualification':'Stored checkpoint metric/schedule/batch prefix agreement; intermediate factors are not archived. Terminal-vs-interior beta convention prevents direct endpoint penalized-objective equality.'}


def check_trial(root,plan_path,plan,c):
    folder = root/plan['phase']/c['id']
    missing = [name for name in ('started.json','result.json','result.npz','summary.json') if not (folder/name).is_file()]
    if missing:
        return {'id':c['id'],'valid':False,'archive_complete':False,'missing':missing,
                'failure_record_present':(folder/'failure.json').exists()}
    require(not (folder/'failure.json').exists(),'Ambiguous successful archive plus infrastructure failure record')
    started,meta,summary = (read(folder/name) for name in ('started.json','result.json','summary.json'))
    with np.load(folder/'result.npz',allow_pickle=False) as archive:
        z = {key:archive[key] for key in archive.files}
    require(json.loads(str(z['metadata_json'])) == meta,'NPZ/JSON metadata mismatch')
    for kind in ('json','npz'):
        require(summary[kind+'_sha256'] == digest(folder/('result.'+kind)),'Archive hash mismatch')
    require(summary['trial'] == started['trial'] == meta['followup_configuration'] == c,'Frozen trial mismatch')
    provenance = meta['followup_provenance']
    for key in ('trial','plan_sha256','source_commit','source_sha256','harness_sha256','parent_pins','runner_sha256'):
        require(provenance[key] == started[key],'Started/archive provenance mismatch: '+key)
    require(started['plan_sha256'] == digest(plan_path),'Plan hash mismatch')
    for key in ('source_commit','source_sha256','harness_sha256','parent_pins'):
        require(started[key] == plan[key],'Started/plan mismatch: '+key)
    require(started['runner_sha256'] in plan['harness_sha256'].values(),'Runner not in frozen harness')
    d,size,m = dimensions(c['n_qubits'])
    a,b = povm_constants(d)
    design = meta['design']
    for key,expected in {'n_qubits':c['n_qubits'],'d':d,'rows':m,'probes':d*d,'outcomes':2*d}.items():
        require(design[key] == expected,'Dimension metadata mismatch: '+key)
    close(design['povm_a'],a,'POVM a',rtol=0,atol=1e-15)
    close(design['povm_b'],b,'POVM b',rtol=0,atol=1e-15)
    require(meta['observation_mode'] == 'noiseless' and meta['active_row_count'] == m,'Data regime changed')
    require(meta['initialization']['basis'] == 'gell-mann','Initialization basis changed')
    require(meta['data_metadata']['channel_seed'] == c['channel'],'Channel seed metadata changed')
    is_jax = meta['format'] == 'qpt_frames_local_dense_jax_calibration_v1'
    require((c['n_qubits'] == 2) == is_jax,'Backend/dimension mismatch')
    require(is_jax or meta['format'] == 'qpt_quiroga_campaign','Unexpected native format')
    backend = meta['followup_backend']
    require(backend['name'] == ('frozen_n2_dense_jax' if is_jax else 'native_quiroga_structured_numpy'),'Wrong declared backend')
    require(backend['wrapper_sha256'] == plan['harness_sha256']['backend_followup.py'],'Backend wrapper hash changed')
    require(backend['explicit_initialization_seed'] == c['initialization_seed']
            and backend['explicit_sampling_seed'] == c['sampling_seed'],'Backend seed declaration changed')
    require(backend['initial_factor_verified_before_and_after'] and backend['independent_frames_penalty']
            and backend['higher_dimension_dense_banks'] is False and backend['resume_supported'] is False,'Backend execution convention changed')
    require(backend['observed_batch_indices_archived'] is is_jax,'Sampling evidence declaration changed')
    conf = meta['configuration']
    if is_jax:
        require(conf == c and meta['sampling_seed'] == c['sampling_seed'],'JAX configuration changed')
        require(meta['adapter_sha256'] == PARENT_HASHES['jax_backend.py'],'Frozen JAX adapter changed')
    else:
        expected = {'methods':['frames'],'rank':1,'initialization_basis':'gell-mann',
                    'initialization_seed':c['initialization_seed'],'sampling_seed':c['sampling_seed'],
                    'frames_steps':c['steps'],'frames_metrics_frequency':c['metrics_every'],
                    'batch_size':c['batch_size'],'tau':c['tau'],'frames_penalty_mode':'independent_schedule'}
        for key,value in expected.items():
            require(conf[key] == value,'Native configuration mismatch: '+key)
        close(conf['frames_beta0'],1/(2*c['alpha0']),'Initial beta')
    for filename,sha in meta['source_sha256'].items():
        path = filename if '/' in filename else 'paper/experiments/'+filename
        require(plan['source_sha256'][path] == sha,'Native source differs from pinned source')
    for name in ('active_rows','active_observations','initial_factor','truth_factor'):
        require(meta[name+'_sha256'] == array_digest(z[name],not is_jax),'Native array hash mismatch: '+name)
    require(np.array_equal(z['active_rows'],np.arange(m)),'Not canonical complete rows')
    require(z['active_observations'].shape == (m,),'Wrong observation shape')
    for name in ('initial_factor','truth_factor','frames_final_factor'):
        require(z[name].shape == (size,1) and np.all(np.isfinite(z[name])),'Wrong/nonfinite factor: '+name)
    close(z['truth_factor'],independent_truth(c['channel'],d),'Independent truth')
    close(z['initial_factor'],independent_initializer(c['initialization_seed'],d),'Independent initialization')
    close(z['active_observations'],born_values(z['truth_factor'],d),'Independent physical Born observations')
    endpoint = meta['endpoints']['frames']
    status = endpoint['status']
    require(status in ('complete','failed') and summary['status'] == status,'Conflicting execution status')
    if not is_jax:
        require(meta['methods']['frames'] == endpoint,'Native/uniform endpoint mismatch')
    steps,times = z['frames_checkpoint_steps'],z['frames_optimizer_seconds']
    require(steps.ndim == 1 and np.issubdtype(steps.dtype,np.integer) and len(steps)>0 and steps[0] == 0
            and np.all(np.diff(steps)>0),'Invalid checkpoints')
    require(np.array_equal(z['frames_cumulative_data_row_accesses'],steps*c['batch_size']),'Wrong optimizer row charges')
    require(times.shape == steps.shape,'Wrong timing shape')
    if is_jax and status == 'complete':
        require(times[0] == 0 and np.isfinite(times[-1]) and times[-1] >= 0 and np.all(np.isnan(times[1:-1])),
                'JAX scan fabricated interior times')
    else:
        require(np.all(np.isfinite(times)) and np.all(times>=0) and np.all(np.diff(times)>=0),'Invalid measured timing')
    for name in COMMON:
        require(z['frames_'+name].shape == steps.shape and np.all(np.isfinite(z['frames_'+name])),'Invalid metric history: '+name)
    close(z['frames_measurement_loss_mean'],.5*z['frames_rmse']**2,'Mean loss history')
    close(z['frames_measurement_loss_sum'],m*z['frames_measurement_loss_mean'],'Sum loss history')
    close(z['frames_normalized_tp_violation'],z['frames_tp_violation']/math.sqrt(d),'TP history normalization')
    batch_hash = None
    if status == 'complete':
        require(np.array_equal(steps,np.arange(0,c['steps']+1,c['metrics_every'])),'Incomplete checkpoint grid')
        iteration = np.arange(c['steps'])
        beta = 1/(2*c['alpha0'])/(iteration+1)**c['penalty_exponent']
        histories = {'smoothing_parameter':beta,'tp_weight_per_row_alpha':1/(2*beta),'tp_weight_lambda':m/(2*beta),
                     'step_size':np.minimum(1,c['step_scale']/(iteration+c['step_offset'])**c['step_exponent']),
                     'momentum_weight':np.minimum(1,c['rho_scale']/(iteration+c['rho_offset'])**c['rho_exponent'])}
        prefix = 'frames_diagnostic_'
        for name,values in histories.items():
            close(z[prefix+name],values,'Schedule '+name)
        checkpoint_beta = beta[np.minimum(steps,c['steps']-1)]
        for name,values in {'checkpoint_smoothing_parameter':checkpoint_beta,
                            'checkpoint_tp_weight_per_row_alpha':1/(2*checkpoint_beta),
                            'checkpoint_tp_weight_lambda':m/(2*checkpoint_beta)}.items():
            close(z[prefix+name],values,'Checkpoint '+name)
        close(z['frames_penalized_objective'],z['frames_measurement_loss_sum']+m/(2*checkpoint_beta)*z['frames_tp_violation']**2,
              'Penalized objective history')
        alpha_end = 1/(2*checkpoint_beta[-1])
        require(np.linalg.norm(z['frames_final_factor'],2) <= c['tau']+3e-12,'Final point outside declared radius')
        if is_jax:
            batches = z[prefix+'batch_indices']
            require(batches.shape == (c['steps'],c['batch_size']) and np.issubdtype(batches.dtype,np.integer),'Wrong batch plan')
            rng = np.random.default_rng(c['sampling_seed'])
            for start in range(0,c['steps'],4096):
                block = batches[start:start+4096]
                require(np.array_equal(block,rng.integers(0,m,size=block.shape)),'Saved sampling stream differs from seed')
            batch_hash = array_digest(batches)
    else:
        alpha_end = c['alpha0']
        require(bool(endpoint['failure_reason']),'Failed status without reason')
    metric = independent_metrics(z['frames_final_factor'],z['truth_factor'],z['active_observations'],alpha_end,d)
    initial = independent_metrics(z['initial_factor'],z['truth_factor'],z['active_observations'],c['alpha0'],d)
    for name in COMMON:
        close(z['frames_'+name][-1],metric[name],'Independent endpoint '+name)
        close(endpoint[name],metric[name],'JSON endpoint '+name)
        close(z['frames_'+name][0],initial[name],'Independent initial '+name)
    passed,deficit = gate_and_deficit(metric,status)
    require(summary['gate_pass'] == passed and provenance['endpoint_gate'] == passed,'Physical gate mismatch')
    close(summary['deficit'],deficit,'Gate deficit')
    if 'gate_deficit' in provenance:
        close(provenance['gate_deficit'],deficit,'Provenance deficit')
    for name,key in (('rmse','rmse'),('normalized_tp','normalized_tp_violation'),('fidelity','process_fidelity'),('relative_rmse','relative_rmse')):
        close(summary[name],metric[key],'Summary '+name)
    close(summary['optimizer_seconds'],times[-1],'Summary optimizer time')
    close(summary['outcome_scaled_rmse'],2*d*metric['rmse'],'Outcome-scaled RMSE')
    require(np.isfinite(summary['total_seconds']) and summary['total_seconds'] >= times[-1]
            and np.isfinite(summary['peak_rss_mib']) and summary['peak_rss_mib'] > 0,'Invalid complete-run resource measurements')
    require(endpoint['completed_steps'] == int(steps[-1]) and endpoint['cumulative_data_row_accesses'] == int(steps[-1]*c['batch_size']),
            'Endpoint work counters differ')
    passing = ((z['frames_rmse']<=1e-6)&(z['frames_normalized_tp_violation']<=.001)&(z['frames_process_fidelity']>=.99))
    first,persistent = gate_crossings(steps,passing)
    require(summary['first_recorded_gate_step'] == first and summary['persistent_recorded_gate_step'] == persistent
            and provenance['persistent_recorded_gate_step'] == persistent,'Gate crossing mismatch')
    if 'first_recorded_gate_step' in provenance:
        require(provenance['first_recorded_gate_step'] == first,'Provenance first gate mismatch')
    tail = tail_stability(z) if plan['phase'] == 'tail' and status == 'complete' else None
    return {'id':c['id'],'n_qubits':c['n_qubits'],'channel':c['channel'],
            'initialization_replicate':c['initialization_replicate'],'sampling_replicate':c['sampling_replicate'],
            'archive_complete':True,'valid':True,'status':status,'gate_pass':passed,'deficit':deficit,'metrics':metric,
            'first_recorded_gate_step':first,'persistent_recorded_gate_step':persistent,'completed_steps':int(steps[-1]),
            'charged_rows':int(steps[-1]*c['batch_size']),'optimizer_seconds':float(times[-1]),
            'json_sha256':digest(folder/'result.json'),'npz_sha256':digest(folder/'result.npz'),
            'summary_sha256':digest(folder/'summary.json'),'started_sha256':digest(folder/'started.json'),
            'paired_arrays_sha256':{name:array_digest(z[name]) for name in ('active_rows','active_observations','initial_factor','truth_factor')},
            'observed_batch_plan_sha256':batch_hash,
            'tail_stability':tail,
            'sampling_verification':'saved realized batches match declared PCG64 seed' if is_jax else 'declared seed and unchanged runner verified; realized batches not archived'}


def audit(plan_path,root=None):
    plan_path = Path(plan_path).resolve()
    root = plan_path.parent if root is None else Path(root)
    plan = read(plan_path)
    check_plan(plan)
    source_count = check_sources(plan,root)
    resource_decision = check_resource_decision(plan,root)
    results = []
    data_pairs,init_pairs,batch_pairs = {},{},{}
    for c in plan['trials']:
        try:
            result = check_trial(root,plan_path,plan,c)
            if result['valid']:
                data_key,init_key,batch_key = pair_keys(c)
                arrays = result['paired_arrays_sha256']
                shared = {k:arrays[k] for k in ('active_rows','active_observations','truth_factor')}
                require(data_pairs.setdefault(data_key,shared) == shared,'Same-channel physical data mismatch')
                require(init_pairs.setdefault(init_key,arrays['initial_factor']) == arrays['initial_factor'],'Same initializer group mismatch')
                if result['observed_batch_plan_sha256'] is not None:
                    key = batch_key+(c['steps'],c['batch_size'])
                    require(batch_pairs.setdefault(key,result['observed_batch_plan_sha256']) == result['observed_batch_plan_sha256'],
                            'Same sampling group has inconsistent realized batches')
        except Exception as error:
            result = {'id':c['id'],'valid':False,'archive_complete':False,'audit_error':type(error).__name__+': '+str(error)}
        results.append(result)
    inventory_path = root/f"inventory_{plan['phase']}.json"
    inventory_valid = False
    inventory_error = None
    try:
        inventory = read(inventory_path)
        require(inventory['phase'] == plan['phase'] and inventory['plan_sha256'] == digest(plan_path),'Batch inventory plan/phase mismatch')
        require(inventory['status'] == 'complete' and set(inventory['trials']) == {c['id'] for c in plan['trials']},'Batch inventory is partial/incomplete')
        require(all(v['state'] == 'finished' and v['exit_code'] == 0 for v in inventory['trials'].values()),'Batch infrastructure failure')
        inventory_valid = True
    except Exception as error:
        inventory_error = type(error).__name__+': '+str(error)
    return {'schema':'frozen_frames_followup_independent_audit_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'plan_path':str(plan_path),'plan_sha256':digest(plan_path),'auditor_sha256':digest(__file__),
            'source_commit':COMMIT,'source_file_count':source_count,'expected_trials':len(plan['trials']),
            'resource_decision':resource_decision,
            'validated_trials':sum(r['valid'] for r in results),'phase_complete_and_valid':inventory_valid and all(r['valid'] for r in results),
            'batch_inventory_valid':inventory_valid,'batch_inventory_error':inventory_error,
            'batch_inventory_sha256':digest(inventory_path) if inventory_path.exists() else None,
            'passing_trials':sum(r.get('gate_pass',False) for r in results),'distinct_channel_dimension_pairs':len(data_pairs),
            'distinct_initialization_groups':len(init_pairs),'trials':results,
            'qualification':'Frozen-setting followup, not new tuning or final unseen-channel comparison. Complete plan inventory required; row counts exclude TP/checkpoints/compilation. Independent Born validation uses physical d-by-d output states and dimension-specific POVM.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    report = audit(args.plan)
    rendered = json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(rendered)
    print(rendered,end='')
    raise SystemExit(0 if report['phase_complete_and_valid'] else 2)
