"""Record bounded local regression checks as exclusive evidence artifacts."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
files = [ROOT / n for n in ('test_backend_followup.py', 'test_audit_followup.py', 'test_plan_rules.py')]
files += [Path('/home/tony/frankwolfe/tests') / n for n in
          ('test_qpt_quiroga_frames.py', 'test_qpt_quiroga_initialization.py')]
log, result_path = ROOT / 'validation_tests.log', ROOT / 'validation_tests.json'
assert not log.exists() and not result_path.exists()
env = os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           JAX_PLATFORMS='cpu', XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *map(str, files)]
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
with log.open('x') as stream:
    p = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
result = {'command': command, 'exit_code': p.returncode, 'started_utc': started,
          'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'cpu_affinity': sorted(os.sched_getaffinity(0)), 'log': log.read_text(),
          'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest(),
          'test_sha256': {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
with result_path.open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
print(result['log'])
sys.exit(p.returncode)
