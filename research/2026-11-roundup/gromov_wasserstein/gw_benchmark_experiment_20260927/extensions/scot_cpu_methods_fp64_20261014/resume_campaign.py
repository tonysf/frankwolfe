"""Resume orchestration after existing SCOT tuning finishes; never rerun tuning.

Run in the host PID namespace (for example, a transient user service), with an
independently captured process-identity JSON. The numerical controller is loaded
unchanged. A separate receipt records this wrapper and preserves failed stages.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('_scot_resume_controller', HERE / 'run_comparison.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


@contextlib.contextmanager
def exclusive_lock(path):
    """One recovery writer across all invocations, including alternate --out."""
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another recovery controller holds the exclusive lock.') from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def process_identity(pid, proc=Path('/proc')):
    """Read PID identity without confusing command-name spaces with stat fields."""
    directory = proc / str(pid)
    try:
        before = (directory / 'stat').read_text()
        fields = before[before.rfind(')') + 2:].split()
        command = (directory / 'cmdline').read_bytes().split(b'\0')
        after = (directory / 'stat').read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    latter = after[after.rfind(')') + 2:].split()
    if fields[19] != latter[19]:
        raise RuntimeError('PID was reused while its identity was being read.')
    return dict(pid=pid, start_ticks=int(latter[19]), state=latter[0],
                cmdline=[word.decode() for word in command if word])


def descendant_identities(pid, proc=Path('/proc')):
    """Include children created by any controller thread, recursively."""
    result, pending, seen = [], [pid], {pid}
    while pending:
        parent = pending.pop()
        for path in (proc / str(parent) / 'task').glob('*/children'):
            try:
                children = [int(word) for word in path.read_text().split()]
            except (FileNotFoundError, ProcessLookupError):
                continue
            for child in children:
                if child in seen:
                    continue
                seen.add(child)
                identity = process_identity(child, proc)
                if identity is not None:
                    result.append(identity)
                    pending.append(child)
    return result


def host_identity(expected, proc=Path('/proc')):
    observed = dict(pid1_comm=(proc / '1/comm').read_text().strip(),
                    pid_namespace=os.readlink(proc / 'self/ns/pid'))
    c.require(observed['pid1_comm'] == expected['pid1_comm'] == 'systemd'
              and observed['pid_namespace'] == expected['pid_namespace'],
              'Host PID namespace does not match the independently captured identity.')
    return observed


def wait_for_original(expected, timeout=10800, poll=60, *, proc=Path('/proc'),
                      identity=process_identity, descendants=descendant_identities,
                      clock=time.monotonic, sleep=time.sleep):
    """Wait for the exact controller and its observed descendants, without signals."""
    c.require(0 < timeout <= 10800 and 60 <= poll <= 300, 'Invalid bounded wait policy.')
    initial = identity(expected['pid'], proc)
    c.require(initial is not None and initial['state'] != 'Z'
              and initial['start_ticks'] == expected['start_ticks']
              and initial['cmdline'] == expected['cmdline'],
              'Original controller identity is missing or changed; refusing to assume it exited.')
    start = clock()
    tracked = {(initial['pid'], initial['start_ticks']): initial}
    observations = 0
    while True:
        observations += 1
        live = []
        for key, recorded in list(tracked.items()):
            current = identity(recorded['pid'], proc)
            if current is None or current['start_ticks'] != key[1] or current['state'] == 'Z':
                continue
            if key == (expected['pid'], expected['start_ticks']):
                c.require(current['cmdline'] == expected['cmdline'], 'Original controller changed command.')
            live.append(current)
            for child in descendants(current['pid'], proc):
                tracked.setdefault((child['pid'], child['start_ticks']), child)
                if child['state'] != 'Z':
                    live.append(child)
        if not live:
            return dict(status='exited', original_identity=initial,
                        observed_processes=list(tracked.values()), observations=observations,
                        elapsed_seconds=clock() - start, poll_seconds=poll,
                        timeout_seconds=timeout, signals_sent=False)
        remaining = timeout - (clock() - start)
        c.require(remaining > 0, 'Original controller/descendants did not exit within the bounded wait.')
        sleep(min(poll, remaining))


def require_no_later_stages(out):
    paths = [out / name for name in ('selection.json', 'completed.json', 'stage_select.json',
             'stage_confirm_both_0_end.json', 'stage_finalize.json', 'stage_report.json', 'report.log', 'report')]
    paths += list(out.glob('confirm_*_receipt.json')) + list((out / 'runs').glob('confirm_*'))
    c.require(not any(path.exists() for path in paths),
              'Selection, confirmation, completion, or reporting already exists; refusing to overwrite or duplicate work.')


def preserve_failed_stages(out):
    paths = [out / 'stage_campaign.json', out / 'stage_tune_both_0_end.json']
    for path in paths:
        c.require(c.read(path)['status'] == 'failed', 'Only the known failed orchestration stages may be recovered.')
    hashes = {path.name: c.sha256(path) for path in paths}
    archive = out / 'recovery_evidence'
    archive.mkdir()
    shutil.copy2(paths[0], archive / paths[0].name)
    paths[1].rename(archive / paths[1].name)
    for name, value in hashes.items():
        c.require(c.sha256(archive / name) == value, 'Failed-stage preservation hash differs.')
    c.require(c.sha256(paths[0]) == hashes[paths[0].name], 'Original failed campaign was changed.')
    return dict(directory=str(archive), original_sha256=hashes,
                replaced_stage='stage_tune_both_0_end.json',
                campaign_failure_preserved_in_place=True)


def render_report(out):
    script = HERE / 'summarize.py'
    script_hash = c.sha256(script)
    command = [sys.executable, str(script), '--root', str(out)]
    with (out / 'report.log').open('x') as stream:
        result = subprocess.run(command, cwd=c.ROOT, env=dict(os.environ, **c.THREAD_ENV),
                                stdout=stream, stderr=subprocess.STDOUT, timeout=3600)
    c.require(result.returncode == 0, f'Final report failed with exit {result.returncode}; see report.log.')
    return dict(command=command, exit_code=result.returncode, log=str(out / 'report.log'),
                reporter_sha256=script_hash)


def resume_after_wait(out, record):
    """The only tuning-stage replacement is synthesized from eight audited runs."""
    require_no_later_stages(out)
    config, binding = c.configuration(out)
    tuned = [c.audited(out, config, binding, 'tune', *cell) for cell in c.tune_cells(config)]
    c.require(len(tuned) == 8, 'Recovery requires all eight completed tuning cells.')
    record['audited_tuning_count'] = len(tuned)
    record['failed_stage_preservation'] = preserve_failed_stages(out)
    c.write_json(out / 'recovery_campaign.json', record)
    c.durable_stage(out, 'tune_both_0_end', lambda: dict(runs=tuned, selection_sha256=None))
    c.durable_stage(out, 'select', lambda: c.choose(out, config, binding))
    args = SimpleNamespace(phase='confirm', method=None, start=0, stop=None)
    c.durable_stage(out, 'confirm_both_0_end', lambda: c.wave(args, out, config, binding))
    c.durable_stage(out, 'finalize', lambda: c.finalize(out, config, binding))
    c.durable_stage(out, 'report', lambda: render_report(out))
    return dict(completed_file=str(out / 'completed.json'), report_log=str(out / 'report.log'))


def run(out, identity_path, timeout, poll):
    c.require(out.is_dir(), 'Experiment directory is absent.')
    # The canonical lock is shared even when --out points to another directory.
    with exclusive_lock(c.DEFAULT_OUT / 'recovery.lock'):
        expected = c.read(identity_path)
        observed = host_identity(expected)
        require_no_later_stages(out)
        receipt_path = out / 'recovery_campaign.json'
        record = dict(status='waiting_for_original', created_utc=c.now(), pid=os.getpid(),
                      recovery_wrapper_sha256=c.sha256(__file__), controller_sha256=c.sha256(HERE / 'run_comparison.py'),
                      config_sha256=c.sha256(out / 'local_config.json'), host_identity=observed,
                      expected_process_identity=expected, process_identity_file=str(identity_path),
                      process_identity_file_sha256=c.sha256(identity_path),
                      policy='Reuse all audited tuning outputs; no tuning runs, numerical edits, or failed-campaign receipt overwrite.')
        c.create_json(receipt_path, record)
        try:
            record['wait'] = wait_for_original(expected, timeout, poll)
            c.require(c.sha256(out / 'local_config.json') == record['config_sha256'], 'Configuration changed during recovery wait.')
            record.update(status='resuming', original_exited_utc=c.now())
            c.write_json(receipt_path, record)
            record['result'] = resume_after_wait(out, record)
            record.update(status='completed', finished_utc=c.now())
            c.write_json(receipt_path, record)
        except Exception as exc:
            record.update(status='failed', error=repr(exc), traceback=traceback.format_exc(), finished_utc=c.now())
            c.write_json(receipt_path, record)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=c.DEFAULT_OUT)
    parser.add_argument('--wait-identity', type=Path, required=True)
    parser.add_argument('--wait-timeout-seconds', type=float, default=10800)
    parser.add_argument('--poll-seconds', type=float, default=60)
    args = parser.parse_args()
    run(args.out.resolve(), args.wait_identity.resolve(), args.wait_timeout_seconds, args.poll_seconds)


if __name__ == '__main__':
    main()
