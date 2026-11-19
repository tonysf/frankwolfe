"""Durable, one-off execution of the authorized expanded CT campaign."""

from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
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


def launch(stage, script, log_name, arguments=()):
    env = dict(os.environ)
    env.update({name: '1' for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS',
                                      'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    env['PYTHONPATH'] = str(DEPS)
    env['PATH'] = f'{LATEX}:{PYTHON.parent}:/usr/local/bin:/usr/bin:/bin'
    env['FRAMES_CAMPAIGN_CPUS'] = '0,2,4,6'
    with (ROOT / log_name).open('a', buffering=1) as stream:
        stream.write(f'\n[{now()}] Durable pipeline starts {stage}\n')
        child = subprocess.Popen([str(PYTHON), '-u', str(ROOT / script), *arguments],
                                 cwd=ROOT, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT)
        state.update(status='running', phase=stage, child_pid=child.pid,
                     phase_started_utc=now())
        save(state)
        code = child.wait()
    if code:
        raise RuntimeError(f'{stage} exited with code {code}; see {log_name}')


def execute():
    spec = importlib.util.spec_from_file_location('ct_campaign_provenance', ROOT / 'audit_provenance.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    audit.freeze_input_anchor(ROOT)
    audit.audit_development(ROOT, require_complete=False, verify_files=True)
    launch('baseline_development', 'campaign.py', 'campaign.log')
    audit.audit_development(ROOT, require_complete=True, verify_files=True)
    launch('evaluation_and_report', 'confirm.py', 'confirmation.log', ('--wait',))
    completed = read(ROOT / 'completion.json')
    if completed['evaluation_runs'] != 308:
        raise ValueError('The final cohort is incomplete')
    report_audit = read(ROOT / 'reports/report_update.json')
    pdf = Path(completed['paper_pdf'])
    if report_audit['report_pdf_sha256'] != digest(pdf):
        raise ValueError('The published paper does not match its validation record')
    state.update(status='completed', phase='report_published', finished_utc=now(),
                 evaluation_runs=completed['evaluation_runs'],
                 observations=completed['observations'], paper_pdf=str(pdf))
    save(state)


state = dict(status='starting', phase='preflight', started_utc=now(),
             pipeline_pid=os.getpid(), pipeline_sha256=digest(Path(__file__)))


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
