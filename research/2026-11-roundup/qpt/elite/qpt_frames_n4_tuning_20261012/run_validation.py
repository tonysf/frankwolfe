"""Archive local test evidence without replacing earlier failed checks."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main(tag):
    assert tag and Path(tag).name == tag and tag not in ('.', '..')
    files = [ROOT/name for name in ('test_runner.py', 'test_audit.py', 'test_selection.py')]
    files += [Path('/home/tony/frankwolfe/tests')/name for name in
              ('test_qpt_quiroga_frames.py', 'test_qpt_quiroga_initialization.py')]
    log, result_path = ROOT/(tag+'.log'), ROOT/(tag+'.json')
    assert not log.exists() and not result_path.exists()
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *map(str, files)]
    stamp = lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    started = stamp()
    with log.open('x') as stream:
        p = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
    result = {'command': command, 'exit_code': p.returncode, 'started_utc': started,
              'finished_utc': stamp(), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'log': log.read_text(), 'log_sha256': digest(log),
              'test_sha256': {str(f): digest(f) for f in files},
              'code_sha256': {f.name: digest(f) for f in ROOT.glob('*.py')}}
    with result_path.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
    print(result['log'])
    return p.returncode


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--tag', default='validation_tests')
    sys.exit(main(p.parse_args().tag))
