"""Dispatch one Slurm array task to a frozen panel shard, with a shared GPU cap.

The plan is a scheduling map, not a new scientific manifest. Every child retains
its original manifest hash, output root, and local shard index in its run ledger.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_plan(plan, task_index):
    tasks = plan['tasks']
    if plan.get('schema_version') != 1 or not tasks or not 0 <= task_index < len(tasks):
        raise ValueError('Invalid dispatch plan or task index.')
    groups = {}
    normalized_tasks = []
    for task in tasks:
        if not all(Path(task[key]).is_absolute() for key in ('manifest', 'out_root')):
            raise ValueError('Dispatch paths must be absolute.')
        task = dict(task, manifest=str(Path(task['manifest']).resolve()),
                    out_root=str(Path(task['out_root']).resolve()))
        normalized_tasks.append(task)
        index, count = task['task_index'], task['task_count']
        if type(index) is not int or type(count) is not int or not 0 <= index < count:
            raise ValueError('Invalid local shard index/count.')
        key = (task['manifest'], task['manifest_sha256'], task['out_root'], count)
        groups.setdefault(key, []).append(index)
    manifests = [key[0] for key in groups]
    if len(manifests) != len(set(manifests)):
        raise ValueError('A scientific manifest must have exactly one shard group.')
    output_roots = [Path(key[2]) for key in groups]
    for i, left in enumerate(output_roots):
        for right in output_roots[i+1:]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError('Panel output roots must be disjoint, not aliases or nested paths.')
    for key, indices in groups.items():
        if sorted(indices) != list(range(key[3])):
            raise ValueError('Every panel shard must appear exactly once.')
    task = normalized_tasks[task_index]
    if sha256(task['manifest']) != task['manifest_sha256']:
        raise ValueError('Frozen scientific manifest hash mismatch.')
    manifest = json.loads(Path(task['manifest']).read_text())
    if len(manifest['runs']) < task['task_count']:
        raise ValueError('A panel shard would be empty.')
    return task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--task-index', type=int, required=True)
    parser.add_argument('--source-directory', type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    if sha256(args.plan) != args.plan_sha256:
        raise ValueError('Frozen scheduling plan hash mismatch.')
    source = args.source_directory.resolve()
    if source != Path(__file__).resolve().parent:
        raise ValueError('Dispatcher must run from its declared immutable source.')
    if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip() != args.source_commit:
        raise ValueError('Source revision mismatch.')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=source, text=True).strip():
        raise ValueError('Dirty source release.')
    task = validate_plan(json.loads(args.plan.read_text()), args.task_index)
    control = Path(task['out_root']) / '_campaign'
    control.mkdir(parents=True, exist_ok=True)
    receipt = dict(plan=str(args.plan.resolve()), plan_sha256=args.plan_sha256,
                   global_task_index=args.task_index, local_shard=task,
                   source_commit=args.source_commit, slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                   slurm_array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),
                   dispatcher_sha256=sha256(__file__),
                   launcher_sha256=sha256(source / 'slurm/a100_campaign.sbatch'))
    # Never overwrite a prior launch record, including a failed attempt.
    receipt_path = control / f"dispatch_{args.task_index:03d}_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    with receipt_path.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    command = ['bash', str(source / 'slurm/a100_campaign.sbatch'), task['manifest'],
               task['out_root'], str(source), args.source_commit,
               str(task['task_count']), str(task['task_index'])]
    raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == '__main__':
    main()
