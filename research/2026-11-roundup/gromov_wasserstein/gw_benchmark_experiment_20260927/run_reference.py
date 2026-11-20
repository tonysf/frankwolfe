"""Small reference baseline runs, with exact external diagnostics.

NOT a production benchmark harness: diagnostic overhead is intentionally high.
Supported methods: full-gradient FW/assignment, KL-prox, oracle-matched
stochastic KL-prox, and small-instance Euclidean PG/Dykstra.
Your unpublished method is deliberately not implemented here.
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from gw_core import (GWProblem, assignment_lmo, kl_prox, round_transport,
                     marginal_residuals, original_fw_gap, rademacher,
                     project_transport_dykstra)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--instance', required=True)
    p.add_argument('--method', choices=['fw', 'kl', 'stochastic-kl', 'pg'], required=True)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--eta', type=float, default=1.)
    p.add_argument('--eta-power', type=float, default=0.)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--inner-tol', type=float, default=1e-9)
    p.add_argument('--inner-max', type=int, default=5000)
    p.add_argument('--seed', type=int, default=100)
    p.add_argument('--init', choices=['product', 'mixed-permutation'], default='product')
    p.add_argument('--device', default='cpu')
    p.add_argument('--dtype', choices=['float32', 'float64'], default='float64')
    p.add_argument('--threads', type=int, default=8)
    p.add_argument('--eval-every', type=int, default=10)
    p.add_argument('--gap-max-n', type=int, default=512)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    if args.steps < 1 or args.eval_every < 1 or args.eta <= 0:
        raise ValueError('Positive steps, eta and evaluation spacing required.')
    torch.set_num_threads(args.threads)
    dtype, device = getattr(torch, args.dtype), torch.device(args.device)
    arrays = np.load(args.instance)
    cast = lambda x: torch.as_tensor(np.array(x), dtype=dtype, device=device)
    problem = GWProblem(*(cast(arrays[k]) for k in ('A', 'B', 'a', 'b')))
    n, m = len(problem.a), len(problem.b)
    P = torch.outer(problem.a, problem.b)
    if args.init == 'mixed-permutation':
        if n != m:
            raise ValueError('Mixed-permutation initialization needs a square instance.')
        rng = np.random.default_rng(args.seed+900000)
        R = torch.zeros_like(P)
        R[torch.arange(n, device=device), torch.tensor(rng.permutation(n), device=device)] = 1/n
        P = .9*P+.1*R
    logP = P.log()
    gen = torch.Generator(device=device).manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    # Account separately for solver, output repair and diagnostic computation.
    rows = []
    solver_seconds = repair_seconds = diagnostic_seconds = 0.
    def evaluate(k, info):
        nonlocal repair_seconds, diagnostic_seconds
        sync(device); t = time.perf_counter()
        R = round_transport(P, problem.a, problem.b)
        sync(device); repair = time.perf_counter()-t
        repair_seconds += repair
        t = time.perf_counter()
        row = dict(iteration=k, solver_seconds=solver_seconds,
                   repair_seconds_current=repair,
                   repair_seconds_cumulative=repair_seconds,
                   time_to_this_feasible_output=solver_seconds+repair,
                   raw_loss=problem.loss(P).item(),
                   rounded_loss=problem.loss(R).item(),
                   repair_l1=(R-P).abs().sum().item(), **marginal_residuals(P, problem.a, problem.b))
        row['rounded_both_l1'] = marginal_residuals(R, problem.a, problem.b)['both_l1']
        row['original_fw_gap'] = original_fw_gap(problem, R) if max(n, m) <= args.gap_max_n else None
        row['inner_iterations'] = info.get('inner_iterations', 0)
        row['inner_converged'] = info.get('converged', True)
        row['projection_primal_dual_gap'] = info.get('projection_primal_dual_gap', None)
        row['underflow_entries'] = info.get('underflow_entries', 0)
        sync(device); diagnostic_seconds += time.perf_counter()-t
        row['diagnostic_seconds_cumulative'] = diagnostic_seconds
        rows.append(row)
    evaluate(0, {})
    for k in range(args.steps):
        sync(device); t = time.perf_counter()
        info = {}
        if args.method == 'stochastic-kl':
            G = problem.sketch_gradient(P, rademacher(m, args.batch, like=P, generator=gen))
        else:
            G = problem.gradient(P)
        eta = args.eta/(k+1)**args.eta_power
        if args.method == 'fw':
            S = assignment_lmo(G, problem.a, problem.b)
            D = S-P
            linear = (G*D).sum().item()
            # Exact quadratic line search without another dense APB product.
            # Uniform square S is a scaled permutation, so f(S) is O(n^2).
            perm = S.argmax(1)
            fS = (problem.A - problem.B[perm][:, perm]).square().mean().item()
            fP = 0.5 * (G * P).sum().item()  # Euler identity for a homogeneous quadratic
            quadratic = fS - fP - linear
            if quadratic > 0:
                gamma = min(1., max(0., -linear/(2*quadratic)))
            else:
                gamma = 1. if linear+quadratic < 0 else 0.
            P = P+gamma*D
        elif args.method in ('kl', 'stochastic-kl'):
            P, logP, info = kl_prox(logP, G, problem.a, problem.b, eta=eta,
                                    tol=args.inner_tol, max_iter=args.inner_max)
        else:
            P, info = project_transport_dykstra(P-eta*G, problem.a, problem.b,
                                               gap_tol=args.inner_tol, max_iter=args.inner_max)
        sync(device); solver_seconds += time.perf_counter()-t
        if (k+1) % args.eval_every == 0 or k+1 == args.steps:
            evaluate(k+1, info)
    with (out/'trace.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    R = round_transport(P, problem.a, problem.b)
    np.savez(out/'final.npz', P_raw=P.cpu().numpy(), P_rounded=R.cpu().numpy())
    meta = vars(args) | dict(torch_version=torch.__version__, numpy_version=np.__version__,
                             diagnostic_warning='Reference driver; diagnostics deliberately frequent.',
                             gpu_tested_by_author=False)
    (out/'config.json').write_text(json.dumps(meta, indent=2))
    print(json.dumps(rows[-1], indent=2))

if __name__ == '__main__':
    main()
