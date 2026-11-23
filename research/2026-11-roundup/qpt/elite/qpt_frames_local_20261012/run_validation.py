"""Record a bounded local regression suite as new evidence, without optimization campaigns."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
tests = [ROOT / name for name in ('test_jax_backend.py', 'test_audit_local.py', 'test_select_stage.py')]
tests += [Path('/home/tony/frankwolfe/tests') / name for name in
          ('test_qpt_quiroga_frames.py', 'test_qpt_quiroga_initialization.py')]
output = ROOT / 'validation_tests.json'
log = ROOT / 'validation_tests.log'
assert not output.exists() and not log.exists()
env = os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
           MKL_NUM_THREADS='1', JAX_PLATFORMS='cpu',
           XLA_FLAGS='--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1')
command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', *map(str, tests)]
started = datetime.datetime.now(datetime.timezone.utc).isoformat()
with log.open('x') as stream:
    completed = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
result = {'started_utc': started, 'finished_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'command': command, 'exit_code': completed.returncode,
          'log_sha256': hashlib.sha256(log.read_bytes()).hexdigest(),
          'test_file_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in tests},
          'log': log.read_text(), 'cpu_affinity': sorted(os.sched_getaffinity(0))}
with output.open('x') as stream:
    json.dump(result, stream, indent=2, sort_keys=True)
print(result['log'])
sys.exit(completed.returncode)
