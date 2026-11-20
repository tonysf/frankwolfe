"""User-authorized parallel continuation; iteration counts are the comparison axis.

Reuse the frozen numerical prescription and completed tuning unchanged. The
original seed811 controller continues; seeds812/813 use spare CPU cores.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import time

import compare_frames as frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('run', 'finalize'), required=True)
    args = parser.parse_args()
    out = frozen.DEFAULT_OUT.resolve()
    config = frozen.frozen_config(out)
    _, binding = frozen.inputs(out)
    chosen = frozen.selection(out, config, binding)
    amendment_path = out/'execution_amendment.json'
    if args.phase == 'run':
        frozen.require(frozen.read(out/'stage_confirm_seed811.json')['status'] in ('running', 'completed'),
                       'Original seed811 must already be launched.')
        frozen.create_json(amendment_path, dict(
            created_utc=frozen.now(), reason='Explicit user instruction to use available CPU cores and assess loss versus iterations; timing is not important at this stage.',
            user_instruction='timing is not so important at this stage, we can just see it works with # of iterations',
            config_sha256=frozen.sha256(out/'local_config.json'),
            frozen_controller_sha256=frozen.sha256(frozen.__file__),
            parallel_controller_sha256=frozen.sha256(__file__),
            remaining_seeds=[812, 813], existing_seed=811,
            additional_workers=2, maximum_total_workers=3, cpu_threads_per_process=1,
            scientific_prescription_unchanged=True, comparison_axis='completed_updates',
            confirmation_timings_comparable=False,
            timing_scope='Confirmation processes may overlap. Their observed times are archived but excluded from method/precision speed comparisons.'))
        stage_path = out/'stage_parallel_confirm.json'
        stage = dict(status='running', started_utc=frozen.now(),
                     execution_amendment_sha256=frozen.sha256(amendment_path))
        frozen.create_json(stage_path, stage)

        def one(seed):
            value = frozen.execute(out, 'confirm', chosen['selected'], seed, binding)
            frozen.create_json(out/f'confirmation_seed{seed}.json', dict(
                status='completed', run=value,
                config_sha256=frozen.sha256(out/'local_config.json'),
                selection_sha256=frozen.sha256(out/'selection.json'),
                execution_amendment_sha256=frozen.sha256(amendment_path)))
            return value

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                values = list(pool.map(one, [812, 813]))
            stage.update(status='completed', finished_utc=frozen.now(), confirmation=values)
            frozen.write_json(stage_path, stage)
        except Exception as exc:
            stage.update(status='failed', error=repr(exc), finished_utc=frozen.now())
            frozen.write_json(stage_path, stage)
            raise
    else:
        amendment = frozen.read(amendment_path)
        frozen.require(amendment['config_sha256'] == frozen.sha256(out/'local_config.json')
                       and amendment['parallel_controller_sha256'] == frozen.sha256(__file__)
                       and amendment['frozen_controller_sha256'] == frozen.sha256(frozen.__file__),
                       'Execution amendment source/config binding changed.')
        frozen.require(frozen.read(out/'stage_parallel_confirm.json')['status'] == 'completed'
                       and frozen.read(out/'stage_confirm_seed811.json')['status'] == 'completed',
                       'All confirmation stages must complete first.')
        confirmed = [frozen.audited_run(out, 'confirm', chosen['selected'], seed, binding)
                     for seed in frozen.SEEDS]
        for seed, run in zip(frozen.SEEDS, confirmed):
            receipt = frozen.read(out/f'confirmation_seed{seed}.json')
            frozen.require(receipt['status'] == 'completed' and receipt['run'] == run
                           and receipt['config_sha256'] == frozen.sha256(out/'local_config.json')
                           and receipt['selection_sha256'] == frozen.sha256(out/'selection.json'),
                           'Confirmation receipt changed.')
            if seed != 811:
                frozen.require(receipt['execution_amendment_sha256'] == frozen.sha256(amendment_path),
                               'Parallel confirmation amendment binding changed.')
        value = dict(status='completed', config=config, tuning=chosen['tuning'],
                     selected=chosen['selected'], confirmation=confirmed,
                     execution_amendment_sha256=frozen.sha256(amendment_path),
                     confirmation_timings_comparable=False)
        frozen.create_json(out/'completed.json', value)
        frozen.create_json(out/'stage_finalize_parallel.json', dict(
            status='completed', finished_utc=frozen.now(),
            execution_amendment_sha256=frozen.sha256(amendment_path),
            completed_sha256=frozen.sha256(out/'completed.json')))
    print(json.dumps(dict(phase=args.phase, status='completed')), flush=True)


if __name__ == '__main__':
    main()
