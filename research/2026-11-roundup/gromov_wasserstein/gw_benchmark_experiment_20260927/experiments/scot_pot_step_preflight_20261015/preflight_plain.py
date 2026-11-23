"""One actual-data fixed-POT-step solve; no trajectory or frozen-code changes."""
from pathlib import Path
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import sys, time, json, hashlib, importlib.util
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('entropy_previous', ROOT/'extensions/scot_entropy_md_20261015/run.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
oracle = sys.argv[1]
out = Path(__file__).resolve().parent / (oracle + '_plain')
out.mkdir(exist_ok=False)
t0 = time.perf_counter()
problem, cmax, hashes, manifest_hash = module.load_problem(ROOT/'experiments/scot_cpu_20261014/reference/prepared/scot1047')
product = problem.a[:,None] * problem.b[None,:]
logref = product.log()
rng = torch.Generator(device='cpu').manual_seed(811)
with torch.no_grad():
    gradient = problem.gradient(product) if oracle == 'full' else module.pair_gradient(problem, module.sample_pairs(problem, product, 1024, generator=rng))
    cost = gradient
    prox = module.CertifiedKLProx(problem, gap_tol=1e-7, certificate_block_rows=256)
    t1 = time.perf_counter()
    plan, logplan, cert = prox(logref, cost, eta=1000., tol=1e-7, max_iter=10000, check_every=10)
    solve_time = time.perf_counter()-t1
    rounded, diagnosis = module.diagnose(problem, plan, .001)
record = dict(oracle=oracle, seed=811, q=1024 if oracle=='pair' else None, eta=1000., epsilon=.001,
    stopping=dict(marginal_l1=1e-7, guarded_normalized_primal_dual_gap=1e-7, inner_max=10000, check_every=10),
    fixed_reference='unit-mass product a*b, cost G (production convention)',
    kernel_identity_max_error=float((logref-1000.*cost+1000.*gradient-logref).abs().max()),
    cmax=cmax, array_hashes=hashes, manifest_hash=manifest_hash,
    certificate=cert, diagnosis=diagnosis,
    timings=dict(solve_seconds=solve_time, total_seconds=time.perf_counter()-t0),
    environment=dict(affinity=sorted(os.sched_getaffinity(0)), torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads()),
    minimum_log_plan=float(logplan.min()), maximum_log_plan=float(logplan.max()))
np.save(out/'plan.npy', plan.numpy())
np.save(out/'log_plan.npy', logplan.numpy())
np.save(out/'gradient.npy', gradient.numpy())
(out/'result.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
print(json.dumps(record, allow_nan=False), flush=True)
