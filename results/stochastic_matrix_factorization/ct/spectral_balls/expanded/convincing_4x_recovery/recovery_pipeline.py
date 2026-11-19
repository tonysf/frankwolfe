"""Durable evaluation recovery and publication of the existing CT report."""

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parent
PYTHON = Path('/home/tony/zofw_experiments/.venv/bin/python')
DEPS = Path('/home/tony/frames_ct_2detect_20261104/deps')
LATEX = Path('/home/tony/.local/share/latex-tools/default-compiler/bin')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(state):
    pending = ROOT / 'pipeline_state.json.tmp'
    pending.write_text(json.dumps(state, indent=2, allow_nan=False) + '\n')
    pending.replace(ROOT / 'pipeline_state.json')
    print(json.dumps(state, separators=(',', ':')), flush=True)


def execute():
    compatibility = read(ROOT / 'compatibility_manifest.json')
    for name, expected in compatibility['current_source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Recovery source changed: {name}')
    smoke = read(ROOT / 'recovery_smoke_anchor.json')
    if digest(ROOT / 'recovery_input_anchor.json') != smoke['recovery_input_anchor_sha256']:
        raise ValueError('Recovery input anchor differs from the validated import test')
    if digest(ROOT / smoke['engine_smoke_path']) != smoke['engine_smoke_sha256']:
        raise ValueError('Recovery import validation record changed')
    env = dict(os.environ)
    env.update({name: '1' for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                                      'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    env['PYTHONPATH'] = str(DEPS)
    env['PATH'] = f'{LATEX}:{PYTHON.parent}:/usr/local/bin:/usr/bin:/bin'
    env['FRAMES_CAMPAIGN_CPUS'] = '0,2,4,6'
    with (ROOT / 'confirmation.log').open('a', buffering=1) as stream:
        stream.write(f'\n[{now()}] Durable recovery starts; frozen tuning is reused\n')
        child = subprocess.Popen([str(PYTHON), '-u', str(ROOT / 'recovery_confirm.py')],
                                 cwd=ROOT, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT)
        state.update(status='running', phase='evaluation_and_report',
                     child_pid=child.pid, phase_started_utc=now())
        save(state)
        code = child.wait()
    if code:
        raise RuntimeError(f'Evaluation recovery exited with code {code}; see confirmation.log')
    completed = read(ROOT / 'completion.json')
    if completed['evaluation_runs'] != 308:
        raise ValueError('The final cohort is incomplete')
    audit = read(ROOT / 'reports/report_update.json')
    pdf = Path(completed['paper_pdf'])
    if audit['report_pdf_sha256'] != digest(pdf):
        raise ValueError('The published paper differs from its validation record')
    source = pdf.with_name('ct_experiments.tex').read_text()
    if source.count('\\begin{figure}') != 3 or source.count('\\begin{table}') != 2:
        raise ValueError('The report added figures or tables')
    state.update(status='completed', phase='report_published', finished_utc=now(),
                 evaluation_runs=308,
                 observations=audit['native_iteration_observations'], paper_pdf=str(pdf))
    save(state)


state = dict(status='starting', phase='preflight', started_utc=now(),
             pipeline_pid=os.getpid(), pipeline_sha256=digest(Path(__file__)),
             historical_development_runs=960, expected_evaluation_runs=308)


if __name__ == '__main__':
    with (ROOT / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            execute()
        except BaseException as exc:
            state.update(status='failed', error=repr(exc), finished_utc=now())
            save(state)
            traceback.print_exc()
            sys.exit(1)
