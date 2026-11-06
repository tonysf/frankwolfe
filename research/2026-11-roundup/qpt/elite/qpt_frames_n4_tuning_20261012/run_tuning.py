"""Pinned local n=4 FRAMES calibration using the unchanged structured runner."""
import argparse
import datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import resource
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
PARENT = Path('/home/tony/qpt_frames_local_followup_20261012')
REPO = Path('/home/tony/frankwolfe')
COMMIT = '4ce1108a0692edc43b47ef465dbc74a42397765e'
GATE = {'fidelity_min': .99, 'rmse_max': 1e-6, 'normalized_tp_max': .001}
CONFIG_FIELDS = ('alpha0', 'penalty_exponent', 'step_scale', 'step_offset', 'step_exponent',
                 'rho_scale', 'rho_offset', 'rho_exponent', 'batch_size', 'tau')
HARNESS_FILES = {'run_tuning.py', 'run_batch.py', 'make_plans.py'}
THREAD_ENV = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')
sys.path.insert(0, str(REPO))
import numpy as np
import scipy
from paper.experiments.qpt_quiroga_campaign import run_campaign, evaluate_common_metrics, save_campaign
from paper.experiments.qpt_quiroga_sensing import generate_quiroga_sensing_data
from paper.experiments.qpt_quiroga_initialization import gell_mann_coefficients_to_choi
from paper.experiments.quantum_process_tomography import PowerSchedule, make_factor_initial_point, unpack_factor


def require(test, explanation):
    if not test:
        raise ValueError(explanation)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def source():
    require(subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip() == COMMIT,
            'Repository commit differs from pinned source.')
    require(not subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True).strip(),
            'Repository has uncommitted changes.')
    names = subprocess.check_output(['git', '-C', str(REPO), 'ls-files', 'paper', 'frank_wolfe'], text=True).splitlines()
    return {name: digest(REPO / name) for name in names if name.endswith('.py')}


def plan_path_in_root(path):
    path = Path(path).resolve()
    require(path.parent == ROOT and path.name.startswith('plan_') and path.suffix == '.json',
            'Plan must be a local plan_*.json file directly inside this campaign root.')
    return path


def runtime_guard():
    require(socket.gethostname().split('.')[0] == 'elite', 'This campaign is local Elite CPU only.')
    require(not any(os.environ.get(k) for k in ('SLURM_JOB_ID', 'SLURM_STEP_ID', 'PBS_JOBID')),
            'Scheduler environment forbidden for this local-only campaign.')
    affinity = set(os.sched_getaffinity(0))
    require(affinity and affinity <= {0, 1}, 'CPU affinity must be restricted to CPUs0,1.')
    require(all(os.environ.get(k) == '1' for k in THREAD_ENV), 'All BLAS/OMP thread settings must be1.')
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'Low-priority execution (nice>=10) required.')


def resource_guard(resources):
    require(type(resources['max_workers']) is int and 1 <= resources['max_workers'] <= 2, 'At most two workers.')
    cpus = resources['cpu_affinity']
    require(isinstance(cpus, list) and cpus and all(type(v) is int for v in cpus)
            and set(cpus) <= {0, 1} and len(cpus) == len(set(cpus)), 'Invalid CPU allocation.')
    require(type(resources['nice']) is int and 10 <= resources['nice'] <= 19, 'Invalid nice priority.')
    for name in ('batch_wall_seconds', 'trial_wall_seconds'):
        require(type(resources[name]) is int and 1 <= resources[name] <= 7200, 'Invalid bounded timeout.')
    require(resources['trial_wall_seconds'] <= resources['batch_wall_seconds'], 'Trial timeout exceeds batch budget.')


def validate_plan(plan, *, spec=None, check_files=True):
    """Validate a complete declared cohort; tests can omit filesystem checks."""
    require(plan['scope'] == 'local_frames_n4_tuning' and plan['gate'] == GATE, 'Wrong scope or recovery gate.')
    phase = plan['phase']
    require(phase in ('pilot', 'screen', 'long', 'confirmation'), 'Unknown phase.')
    require(plan['source_commit'] == COMMIT, 'Wrong source commit.')
    require(plan['reserved_final_channels'] == list(range(1000, 1020)), 'Reserved final channels changed.')
    spec = json.loads((ROOT / 'campaign_spec.json').read_text()) if spec is None else spec
    configs = {c['id']: c for c in spec['configs']}
    require(len(configs) == len(spec['configs']) and configs, 'Duplicate or empty configuration grid.')
    for c in configs.values():
        require(set(c) == {'id', *CONFIG_FIELDS}, 'Configuration definition has missing or unknown knobs.')
        require(isinstance(c['id'], str) and c['id'] and Path(c['id']).name == c['id']
                and c['id'] not in ('.', '..'), 'Unsafe configuration ID.')
        require(type(c['batch_size']) is int and c['batch_size'] > 0 and c['tau'] == 4., 'Invalid n4 batch/radius.')
        require(all(not isinstance(c[k], bool) and np.isfinite(c[k]) and c[k] > 0 for k in CONFIG_FIELDS),
                'Every schedule parameter must be positive and finite; p=0 is not this lane.')
    rule = spec['phases'][phase]
    allowed = rule['config_ids']
    require(isinstance(allowed, list) and allowed and len(set(allowed)) == len(allowed)
            and set(allowed) <= set(configs), 'Invalid phase configuration list.')
    selected = plan.get('selected_config_ids', allowed)
    if phase in ('long', 'confirmation'):
        require('selected_config_ids' in plan and 1 <= len(selected) <= (2 if phase == 'long' else 1),
                'Later stages require the declared frozen selection.')
    else:
        require(selected == allowed, 'Pilot/screen may not silently prune configurations.')
    require(len(set(selected)) == len(selected) and set(selected) <= set(allowed), 'Selected configuration not allowed.')
    for key, limit in (('channels', 1000), ('initialization_replicates', 10), ('sampling_replicates', 100)):
        values = rule[key]
        require(isinstance(values, list) and values and len(set(values)) == len(values)
                and all(type(v) is int and 0 <= v < limit for v in values), 'Invalid phase cohort: ' + key)
    budget, cadence = rule['charged_row_budget'], rule['checkpoint_charged_rows']
    require(type(budget) is int and type(cadence) is int and 0 < cadence <= budget and budget % cadence == 0,
            'Invalid charged-row horizon/checkpoints.')
    trials = plan['trials']
    require(trials and len({c['id'] for c in trials}) == len(trials), 'Missing or duplicate trials.')
    expected = set(itertools.product(selected, rule['channels'], rule['initialization_replicates'], rule['sampling_replicates']))
    observed = set()
    for c in trials:
        cid = c['config_id']
        require(cid in selected, 'Trial configuration is not selected.')
        require({k: c[k] for k in CONFIG_FIELDS} == {k: configs[cid][k] for k in CONFIG_FIELDS}, 'Trial changes a frozen knob.')
        for key in ('channel', 'initialization_replicate', 'sampling_replicate', 'initialization_seed', 'sampling_seed',
                    'n_qubits', 'steps', 'metrics_every', 'charged_row_budget'):
            require(type(c[key]) is int, 'Integer field required: ' + key)
        require(c['n_qubits'] == 4 and c['channel'] not in plan['reserved_final_channels'], 'Wrong dimension or held-out leakage.')
        require(c['initialization_seed'] == 200000 + 10*c['channel'] + c['initialization_replicate'], 'Initialization seed mismatch.')
        require(c['sampling_seed'] == 300000 + 100*c['channel'] + c['sampling_replicate'], 'Sampling seed mismatch.')
        require(c['id'] == f"{cid}-c{c['channel']}-i{c['initialization_replicate']}-s{c['sampling_replicate']}", 'Trial ID mismatch.')
        batch = c['batch_size']
        require(budget % batch == cadence % batch == 0 and c['steps'] == budget // batch
                and c['metrics_every'] == cadence // batch, 'Trial steps/checkpoints violate charged-row budget.')
        require(c['charged_row_budget'] == budget == c['steps']*batch, 'Trial row budget mismatch.')
        observed.add((cid, c['channel'], c['initialization_replicate'], c['sampling_replicate']))
    require(observed == expected and len(trials) == len(expected), 'Incomplete or undeclared phase inventory.')
    require(plan['total_charged_row_budget'] == sum(c['charged_row_budget'] for c in trials), 'Total row budget mismatch.')
    resource_guard(plan['resources'])
    if check_files:
        require(plan['campaign_spec_sha256'] == digest(ROOT / 'campaign_spec.json'), 'Campaign specification changed.')
        require(plan['source_sha256'] == source(), 'Source manifest differs from clean pinned checkout.')
        parents = {str(PARENT / n) for n in ('completion_manifest.json', 'audit_transfer.json')}
        require(set(plan['parent_pins']) == parents, 'Unexpected parent evidence.')
        require(all(digest(p) == h for p, h in plan['parent_pins'].items()), 'Parent evidence changed.')
        require(set(plan['harness_sha256']) == HARNESS_FILES, 'Incomplete harness pin set.')
        require(all(digest(ROOT / p) == h for p, h in plan['harness_sha256'].items()), 'Frozen harness changed.')
    return spec


def initial_factor(data, seed):
    return gell_mann_coefficients_to_choi(unpack_factor(make_factor_initial_point(data, rank=1, seed=seed),
                                                      data.process_dimension, 1))


def audited_inputs(phase):
    """Bind prerequisites to their complete, independently audited archives."""
    audit_path, plan_path = ROOT/f'audit_{phase}.json', ROOT/f'plan_{phase}.json'
    audit, plan = json.loads(audit_path.read_text()), json.loads(plan_path.read_text())
    require(audit['phase_complete_and_valid'] is True and audit['plan_sha256'] == digest(plan_path),
            'Prerequisite audit is incomplete or stale.')
    require(audit['batch_inventory_sha256'] == digest(ROOT/f'inventory_{phase}.json'), 'Prerequisite inventory changed.')
    require([r['id'] for r in audit['trials']] == [t['id'] for t in plan['trials']], 'Prerequisite trial inventory changed.')
    for row in audit['trials']:
        require(row['valid'], 'Prerequisite contains an invalid trial.')
        for name, key in [('result.json','json_sha256'),('result.npz','npz_sha256'),
                          ('started.json','started_sha256'),('summary.json','summary_sha256')]:
            require(digest(ROOT/phase/row['id']/name) == row[key], 'Prerequisite archive changed.')
    return audit


def selection_contract(plan, selected):
    require(selected['selected_config_ids'] == plan['selected_config_ids'], 'Selection configuration mismatch.')
    require(selected['spec_sha256'] == plan['campaign_spec_sha256'], 'Selection campaign specification mismatch.')


def execution_prerequisites(plan, spec):
    """Screen requires resources; later phases additionally require selection."""
    if plan['phase'] == 'pilot': return
    decision = json.loads((ROOT/'resource_decision.json').read_text())
    require(decision.get('allowed') is True and decision.get('quality_metrics_used_for_resource_decision') is False,
            'Screen resource gate absent, failed, or conditioned on numerical quality.')
    require(decision['campaign_spec_sha256'] == plan['campaign_spec_sha256'], 'Resource decision specification mismatch.')
    require(decision['pilot_audit_sha256'] == digest(ROOT/'audit_pilot.json'), 'Resource decision pilot audit changed.')
    audit = audited_inputs('pilot'); rule = spec['pilot_resource_gate']
    estimates = {r['id']:r for r in decision['estimates']}
    require(len(estimates) == len(decision['estimates']) == len(audit['trials']), 'Incomplete pilot resource estimates.')
    for row in audit['trials']:
        summary = json.loads((ROOT/'pilot'/row['id']/'summary.json').read_text())
        projected = summary['optimizer_seconds']*(spec['phases']['long']['charged_row_budget']/row['charged_rows']) + summary['total_seconds']
        require(summary['status'] == 'complete' and projected < rule['max_projected_long_seconds']
                and summary['peak_rss_mib'] < rule['max_peak_rss_mib'], 'Measured pilot exceeds resource gate.')
        require(np.isclose(estimates[row['id']]['projected_long_seconds'], projected, rtol=1e-12, atol=1e-12)
                and estimates[row['id']]['peak_rss_mib'] == summary['peak_rss_mib'], 'Resource projection mismatch.')
    if plan['phase'] == 'screen': return
    previous = 'screen' if plan['phase'] == 'long' else 'long'
    path = ROOT/f'selection_{previous}.json'
    require(Path(plan['selection']['path']).resolve() == path and plan['selection']['sha256'] == digest(path),
            'Frozen selection path/hash mismatch.')
    selection = json.loads(path.read_text()); selection_contract(plan, selection)
    prior = audited_inputs(previous)
    require(selection['audit_sha256'] == digest(ROOT/f'audit_{previous}.json'), 'Selection input audit changed.')
    from make_plans import select_screen, select_long
    if previous == 'screen':
        expected, _ = select_screen(prior['trials'])
    else:
        old_path = PARENT/'audit_transfer.json'; old = json.loads(old_path.read_text())
        require(selection['archived_control_audit_sha256'] == digest(old_path) and old['phase_complete_and_valid'],
                'Archived incumbent evidence changed.')
        controls = []
        for row in old['trials']:
            if row['n_qubits'] != 4: continue
            for name, key in [('result.json','json_sha256'),('result.npz','npz_sha256'),
                              ('started.json','started_sha256'),('summary.json','summary_sha256')]:
                require(digest(PARENT/'transfer'/row['id']/name) == row[key], 'Archived incumbent trial changed.')
            controls.append({**row, 'config_id':'incumbent'})
        require(len(controls) == 2 and {r['channel'] for r in controls} == {0,1}, 'Incomplete incumbent comparison.')
        expected, _ = select_long(prior['trials'] + controls)
    require(expected and expected == plan['selected_config_ids'], 'Frozen selection does not match its audited rule.')


def run_native(data, factor0, c):
    """Thin adapter only: every optimization update remains repository-owned."""
    require(data.n_qubits == 4 and data.m == 8192 and data.observation_mode == 'noiseless', 'Wrong sensing regime.')
    require(data.metadata['channel_seed'] == c['channel'], 'Channel data mismatch.')
    require(np.array_equal(factor0, initial_factor(data, c['initialization_seed'])), 'Declared initial factor mismatch.')
    before = factor0.copy(); beta0 = 1 / (2*c['alpha0'])
    metadata, payload, traces = run_campaign(
        data, methods=['frames'], rank=1, initialization_basis='gell-mann',
        initialization_seed=c['initialization_seed'], sampling_seed=c['sampling_seed'],
        n_steps=c['steps'], frames_steps=c['steps'], metrics_frequency=c['metrics_every'],
        frames_metrics_frequency=c['metrics_every'], eta_scale=1., tp_weight=1., tau=4., batch_size=c['batch_size'],
        frames_beta0=beta0, frames_smoothing_schedule=PowerSchedule(beta0, 1., c['penalty_exponent']),
        rho_schedule=PowerSchedule(c['rho_scale'], c['rho_offset'], c['rho_exponent'], cap=1.),
        step_size_schedule=PowerSchedule(c['step_scale'], c['step_offset'], c['step_exponent'], cap=1.), show_progress=False)
    require(np.array_equal(factor0, before) and np.array_equal(payload['initial_factor'], before), 'Initial factor changed.')
    metadata['tuning_configuration'] = dict(c)
    metadata['tuning_backend'] = dict(name='structured_numpy', precision='complex128', dense_sensing_bank=False,
        repository_runner='paper.experiments.qpt_quiroga_campaign.run_campaign', resume=False,
        sampling='PCG64 with replacement; no realized batch archive', observed_batch_indices_archived=False,
        failure_archival='native runner retains initial factor on FRAMES exception')
    return metadata, payload, traces['frames']


def execute(plan_path, trial_id):
    runtime_guard(); plan_path = plan_path_in_root(plan_path)
    plan = json.loads(plan_path.read_text()); spec = validate_plan(plan); execution_prerequisites(plan, spec)
    matches = [t for t in plan['trials'] if t['id'] == trial_id]
    require(len(matches) == 1, 'Trial not in frozen plan.'); c = matches[0]
    out = ROOT / plan['phase'] / trial_id; out.mkdir(parents=True, exist_ok=False)
    prerequisites = {}
    if plan['phase'] != 'pilot':
        prerequisites = {name:digest(ROOT/name) for name in ('resource_decision.json','audit_pilot.json')}
    if plan['phase'] in ('long','confirmation'):
        previous = 'screen' if plan['phase']=='long' else 'long'
        prerequisites.update({name:digest(ROOT/name) for name in (f'selection_{previous}.json',f'audit_{previous}.json')})
    started = dict(trial=c, plan_path=str(plan_path), plan_sha256=digest(plan_path),
        campaign_spec_sha256=plan['campaign_spec_sha256'], source_commit=COMMIT, source_sha256=plan['source_sha256'],
        harness_sha256=plan['harness_sha256'], parent_pins=plan['parent_pins'], runner_sha256=digest(__file__),
        host=socket.gethostname(), platform=platform.platform(), python=platform.python_version(), numpy=np.__version__,
        scipy=scipy.__version__, cpu_affinity=sorted(os.sched_getaffinity(0)), nice=os.getpriority(os.PRIO_PROCESS, 0),
        threads={k: os.environ.get(k) for k in THREAD_ENV}, pid=os.getpid(), started_utc=now(), execution='local_cpu',
        split=plan['phase'], charged_row_budget=c['charged_row_budget'], execution_prerequisites_sha256=prerequisites)
    write(out/'started.json', started); began = time.perf_counter()
    try:
        data = generate_quiroga_sensing_data(4, channel_seed=c['channel'], observation_mode='noiseless')
        factor0 = initial_factor(data, c['initialization_seed']); metadata, payload, trace = run_native(data, factor0, c)
        targets = data.all_observations()
        metrics = evaluate_common_metrics(data, trace.final_factor, np.arange(data.m, dtype=np.int64), targets, 0.)
        gate = bool(trace.status == 'complete' and metrics['rmse'] <= 1e-6
                    and metrics['normalized_tp_violation'] <= .001 and metrics['process_fidelity'] >= .99)
        deficit = max(metrics['rmse']/1e-6, metrics['normalized_tp_violation']/.001, max(0.,1.-metrics['process_fidelity'])/.01)
        steps = payload['frames_checkpoint_steps']; charged = payload['frames_cumulative_data_row_accesses']
        require(np.array_equal(charged, steps*c['batch_size']), 'Archive row accounting mismatch.')
        if trace.status == 'complete':
            require(np.array_equal(steps, np.arange(0,c['steps']+1,c['metrics_every'])), 'Checkpoint grid mismatch.')
        require(np.isclose(metrics['rmse'], payload['frames_rmse'][-1], rtol=1e-9, atol=1e-15), 'Endpoint metrics mismatch.')
        passes = (payload['frames_rmse']<=1e-6)&(payload['frames_normalized_tp_violation']<=.001)&(payload['frames_process_fidelity']>=.99)
        persistent = np.logical_and.accumulate(passes[::-1])[::-1]
        first = int(steps[np.flatnonzero(passes)[0]]) if passes.any() else None
        stable = int(steps[np.flatnonzero(persistent)[0]]) if persistent.any() else None
        metadata['tuning_provenance'] = dict(started, endpoint_gate=gate, gate_deficit=float(deficit),
            first_recorded_gate_step=first, persistent_recorded_gate_step=stable)
        save_campaign(out/'result.npz', out/'result.json', metadata, payload)
        result = dict(trial=c, status=trace.status, failure_reason=trace.failure_reason, gate_pass=gate, deficit=float(deficit),
            rmse=float(metrics['rmse']), normalized_tp=float(metrics['normalized_tp_violation']), fidelity=float(metrics['process_fidelity']),
            relative_rmse=float(metrics['rmse']*math.sqrt(data.m)/np.linalg.norm(targets)), outcome_scaled_rmse=float(2*data.d*metrics['rmse']),
            completed_steps=int(steps[-1]), charged_rows=int(charged[-1]), charged_row_budget=c['charged_row_budget'],
            first_recorded_gate_step=first, persistent_recorded_gate_step=stable, optimizer_seconds=float(trace.optimizer_seconds[-1]),
            total_seconds=time.perf_counter()-began, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,
            json_sha256=digest(out/'result.json'), npz_sha256=digest(out/'result.npz'), finished_utc=now())
        write(out/'summary.json', result); print(json.dumps(result), flush=True); return result
    except Exception as error:
        write(out/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(),
                                      total_seconds=time.perf_counter()-began, finished_utc=now()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True); parser.add_argument('--trial', required=True)
    args = parser.parse_args(); result = execute(args.plan, args.trial)
    raise SystemExit(0 if result['status'] == 'complete' else 2)
