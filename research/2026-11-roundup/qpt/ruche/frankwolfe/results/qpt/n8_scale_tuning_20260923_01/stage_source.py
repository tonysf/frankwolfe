"""Standard-library-only source snapshot preparation; no framework execution."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parent
plan = json.loads((root/'plan.json').read_text())
original, target = Path(plan['source']), Path(plan['optimized_source'])
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
assert {name:sha(original/name) for name in plan['source_sha256']} == plan['source_sha256']
original_hashes = {str(p.relative_to(original)):sha(p) for p in sorted(original.rglob('*.py'))}
shutil.copytree(str(original), str(target), ignore=shutil.ignore_patterns('__pycache__', '.pytest_cache'))
runner = target/'paper/experiments/quantum_process_tomography_structured_jax.py'
text = runner.read_text()
old = '''            symbols = data.sample_symbols(rng, length * batch_size).reshape(length, batch_size, data.n_qubits)
            observations = (
                data.noise_for_symbols(symbols.reshape(-1, data.n_qubits)).reshape(length, batch_size) if noiseless
                else data.observations_for_symbols(symbols.reshape(-1, data.n_qubits)).reshape(length, batch_size)
            )'''
new = '''            if data.observation_mode == "synthetic-noisy" and 1 <= data.n_qubits <= 13:
                from .qpt_batch_preparation import prepare_on_demand_batch
                symbols, observations = prepare_on_demand_batch(data, rng, length * batch_size)
                symbols = symbols.reshape(length, batch_size, data.n_qubits)
                observations = observations.reshape(length, batch_size)
            else:
                symbols = data.sample_symbols(rng, length * batch_size).reshape(length, batch_size, data.n_qubits)
                observations = (
                    data.noise_for_symbols(symbols.reshape(-1, data.n_qubits)).reshape(length, batch_size) if noiseless
                    else data.observations_for_symbols(symbols.reshape(-1, data.n_qubits)).reshape(length, batch_size)
                )'''
assert text.count(old) == 1
text = text.replace(old, new)
ast.parse(text)
runner.write_text(text)
shutil.copy2(str(root/'qpt_batch_preparation.py'), str(target/'paper/experiments/qpt_batch_preparation.py'))
optimized_hashes = {str(p.relative_to(target)):sha(p) for p in sorted(target.rglob('*.py'))}
assert [name for name in original_hashes if original_hashes[name] != optimized_hashes[name]] == ['paper/experiments/quantum_process_tomography_structured_jax.py']
report = dict(original_source=str(original), original_sha256=original_hashes,
              optimized_source=str(target), optimized_sha256=optimized_hashes,
              added_file='paper/experiments/qpt_batch_preparation.py',
              changed_existing_file='paper/experiments/quantum_process_tomography_structured_jax.py')
(root/'source_deployment.json').write_text(json.dumps(report,indent=2)+'\n')
print('Staged isolated host-preparation candidate; original source preserved.')
