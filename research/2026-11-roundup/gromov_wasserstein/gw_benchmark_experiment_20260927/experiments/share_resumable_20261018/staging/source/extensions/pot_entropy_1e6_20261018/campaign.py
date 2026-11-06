"""Add only epsilon=1e-6, after each original entropy stream terminates.

Reuse the existing numerical runner and fault-aware controller. A CPU bridge
follows scheduler dependencies through the original continuation dispatchers;
it does not poll and never holds a GPU waiting for the preceding campaign.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import importlib.util
import os
from pathlib import Path
import traceback

HERE = Path(__file__).resolve().parent
EXT = Path('extensions/pot_entropy_1e6_20261018')
BASE_EXT = Path('extensions/pot_entropy_init_20261018')
spec = importlib.util.spec_from_file_location('entropy_addon_controller_base', HERE.parent/'pot_entropy_init_20261018/campaign.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
require, read, write, now = base.require, base.read, base.write, base.now
DATASETS = base.DATASETS
original_runner_command = base.runner_command
original_load_config = base.load_config


def load_config(path):
    c = read(path)
    reference = original_load_config(HERE.parent/'pot_entropy_init_20261018/campaign.json')
    for key in ('schema_version', 'dtype', 'method', 'instances', 'initializations',
                'init_mixture_weight', 'iterations', 'inner_tol', 'inner_gap_tol',
                'inner_max', 'outer_tolerance', 'outer_min_iterations',
                'outer_stable_updates', 'runner', 'scheduler'):
        require(c[key] == reference[key], 'Parent scientific/runtime specification changed: '+key)
    require(c['experiment'] == 'pot_entropy_1e6_20261018' and c['epsilons'] == [1e-6],
            'Addon must add exactly epsilon=1e-6')
    parent = c['additive_grid_parent']
    require(parent['experiment'] == 'pot_entropy_init_20261018' and
            Path(parent['root']).is_absolute(), 'Invalid parent campaign')
    require(set(parent['initial_dispatchers']) == set(DATASETS) and
            all(x.isdigit() for x in parent['initial_dispatchers'].values()), 'Invalid parent dispatchers')
    require(c['storage']['required_quota_available_bytes'] == 68*1024**3,
            'Reservation must cover 64 GiB SHARE plus 4 GiB serialized entropy')
    return c


def check_storage(root, c, fresh=False):
    receipt = read(Path(root)/'storage_receipt.json')
    require(receipt.get('quota_verified') is True, 'Storage receipt must establish quota')
    require(receipt['quota_available_bytes'] >= c['storage']['required_quota_available_bytes'],
            'Verified quota headroom below 68 GiB: 64 GiB retained SHARE plus 4 GiB entropy')
    folder = Path(receipt['checkpoint_root'])
    require(folder.is_absolute() and folder.is_dir(), 'Checkpoint directory missing')
    if fresh:
        age = (dt.datetime.now(dt.timezone.utc)-dt.datetime.fromisoformat(receipt['verified_utc'])).total_seconds()
        require(0 <= age <= c['storage']['receipt_max_age_seconds'], 'Quota receipt stale')
    return folder


def runner_command(source, root, c, dataset, task, seconds, checkpoint_root):
    command = original_runner_command(source, root, c, dataset, task, seconds, checkpoint_root)
    command[1] = str(Path(source)/BASE_EXT/'runner.py')
    return command


def parent_identity(c):
    parent = c['additive_grid_parent']
    folder = Path(parent['root'])
    expected = {k: parent[k] for k in ('config_sha256', 'source_manifest_sha256')}
    actual = read(folder/'campaign_identity.json')
    require(all(actual.get(k) == v for k, v in expected.items()), 'Parent source/config identity differs')
    return folder, expected


def submit_initial(source, root, config):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    c = load_config(config)
    base.check_source(source)
    check_storage(root, c, fresh=True)
    parent_identity(c)
    (root/'submission_lock').mkdir()
    (root/'scheduler').mkdir(exist_ok=True)
    write(root/'campaign_identity.json', dict(**base.identity(source, config), submitted_utc=now(), source=str(source)))
    cpu = base.submit_job(source, root, 'cpu-gate', cpu=True, request_key='initial:cpu-gate')
    jobs, datasets = [cpu], {}
    for dataset in DATASETS:
        parent = c['additive_grid_parent']['initial_dispatchers'][dataset]
        bridge = base.submit_job(source, root, 'await-parent', dataset, cpu=True, parent_job=parent,
                                 dependency=f'afterok:{cpu},afterany:{parent}',
                                 request_key=f'initial:{dataset}:await-parent:{parent}')
        jobs.append(bridge)
        datasets[dataset] = dict(parent_dispatcher_job_id=parent, bridge_job_id=bridge)
    write(root/'initial_submission.json', dict(status='queued', jobs=jobs, datasets=datasets,
          parent=c['additive_grid_parent'], time=now()))


def await_parent(source, root, config, dataset, parent_job):
    """Follow the whole parent continuation chain using afterany dependencies."""
    root, c = Path(root), load_config(config)
    job = os.environ.get('SLURM_JOB_ID')
    require(job and job.isdigit(), 'Bridge requires Slurm CPU allocation')
    base.require_gate(root/'checks/cpu/gate.json', base.identity(source, config))
    parent_root, expected = parent_identity(c)
    parent_dataset = parent_root/'results'/dataset
    bridge_root = root/'bridges'/dataset
    bridge_root.mkdir(parents=True, exist_ok=True)
    decision_path = bridge_root/f'after_{parent_job}.json'
    with (bridge_root/'bridge.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if decision_path.exists():
            return read(decision_path)
        try:
            prior = read(bridge_root/'state.json') if (bridge_root/'state.json').exists() else None
            if prior:
                require(prior['next_parent_dispatcher'] == str(parent_job), 'Stale bridge cannot fork stream')
            else:
                require(str(parent_job) == c['additive_grid_parent']['initial_dispatchers'][dataset],
                        'Wrong initial parent dispatcher')
            scheduler = base.slurm_terminal_record(parent_job)
            require(scheduler['state'] == 'COMPLETED' and scheduler['exit_code'] == '0:0',
                    'Parent dispatcher did not succeed: '+str(scheduler))
            require(not (parent_dataset/'failed.json').exists(), 'Parent dataset is frozen; manual review required')
            if (parent_dataset/'completed.json').exists():
                complete = read(parent_dataset/'completed.json')
                require(complete['status'] == 'completed' and complete['runs'] == 16 and
                        complete['successful_runs']+complete['unsuccessful_runs'] == 16,
                        'Parent did not account for all 16 trials')
                require(all(complete.get(k) == v for k, v in expected.items()), 'Parent completion identity differs')
                gate = base.submit_job(source, root, 'dataset-gate', dataset, dependency='afterok:'+job,
                                       seconds=3600, request_key=f'{dataset}:gate')
                run = base.submit_job(source, root, 'production', dataset, dependency='afterok:'+gate,
                                      request_key=f'{dataset}:production')
                dispatcher = base.submit_job(source, root, 'dispatch', dataset, dependency='afterany:'+run,
                                             cpu=True, parent_job=run, request_key=f'{dataset}:{run}:dispatch')
                decision = dict(status='released', parent_completion_sha256=base.sha256(parent_dataset/'completed.json'),
                                gate_job_id=gate, production_job_id=run, dispatcher_job_id=dispatcher)
            else:
                continuation = read(parent_dataset/'dispatch_state.json')
                require(continuation['status'] == 'continued', 'Parent has no continuation')
                next_dispatcher = continuation['next_dispatch_job_id']
                require(next_dispatcher.isdigit() and next_dispatcher != str(parent_job),
                        'Parent dispatcher chain made no progress')
                following = base.submit_job(source, root, 'await-parent', dataset, cpu=True,
                          parent_job=next_dispatcher, dependency=f'afterok:{job},afterany:{next_dispatcher}',
                          request_key=f'{dataset}:await-parent:{next_dispatcher}')
                decision = dict(status='waiting', next_parent_dispatcher=next_dispatcher, next_bridge_job_id=following,
                                parent_dispatch_state_sha256=base.sha256(parent_dataset/'dispatch_state.json'))
                write(bridge_root/'state.json', decision)
            decision.update(time=now(), parent_dispatcher=parent_job, scheduler=scheduler)
            write(decision_path, decision)
            return decision
        except BaseException as exc:
            failure = dict(status='failed', time=now(), error=repr(exc), traceback=traceback.format_exc())
            write(bridge_root/'failed.json', failure)
            write(decision_path, failure)
            raise


# Explicit bindings keep one implementation of production/restart/failure logic.
base.EXT = EXT
base.load_config = load_config
base.check_storage = check_storage
base.runner_command = runner_command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('submit', 'cpu-gate', 'await-parent', 'dataset-gate', 'production', 'dispatch'))
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--dataset', choices=DATASETS)
    p.add_argument('--world', type=int, choices=(1,), default=1)
    p.add_argument('--parent-job')
    args = p.parse_args()
    source, root = args.source.resolve(), args.root.resolve()
    config = source/EXT/'campaign.json'
    base.check_source(source)
    if args.phase == 'submit':
        submit_initial(source, root, config)
    elif args.phase == 'cpu-gate':
        base.cpu_gate(source, root, config)
    else:
        require(args.dataset, 'Dataset required')
        if args.phase in ('await-parent', 'dispatch'):
            require(args.parent_job, 'Parent job required')
            operation = await_parent if args.phase == 'await-parent' else base.dispatch
            operation(source, root, config, args.dataset, args.parent_job)
        elif args.phase == 'dataset-gate':
            base.dataset_gate(source, root, config, args.dataset)
        else:
            base.production(source, root, config, args.dataset)


if __name__ == '__main__':
    main()
