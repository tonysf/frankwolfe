# FP64 CPU comparison on the SCOT1047 geometry

This experiment adds the published SMD and MSBPG updates to the completed FP64
FraMES experiment. All three use the same released SCOT geometry (1,047 cells
per modality), uniform product initialization, and an unbiased coupling-pair
oracle with 1,024 iid pairs per update. Each run has 2,000 updates. Numerical
optimizer source files from the existing pair-sampling campaign are unchanged.

Four admissible prefactor settings per method are evaluated on seed 810. The
terminal feasible objective selects one setting, which is then run on fresh
seeds 811–813. The existing FraMES runs used the same tuning and confirmation
budgets and are reused. Plots show median full squared GW loss and the range
over the three fresh seeds at common iteration checkpoints. The SCOT/POT
feasible endpoint is a reference, not a certified optimum.

## Paper correspondence

- **SMD:** Fatkhullin–He (AISTATS 2024), equation (2), Theorem 4.3, with fresh
  gradients and the admissible specialization
  `eta_k = u / (4 Cmax sqrt(k+1))`. Test `u = 0.1, 0.3, 0.6, 1`.
- **MSBPG:** Ding–Li–Toh (JMLR 26(39), 2025), equations (8)–(9), Theorem 24
  and Remark 25. Use the existing globally strongly convex Legendre kernel,
  conservative GW Hessian bound, and coupled step/momentum prescription.
  Test coupling fractions `0.3, 1` crossed with momentum prefactors `0.3, 1`.
  This applies the general theorem to GW; it is not the paper's neural-network
  empirical configuration. The conservative bound can limit progress.

The papers do not uniquely prescribe these free prefactors. Both KL proximal
solves retain the requested `1e-7` actual marginal residual and guarded,
normalized primal-dual-gap criteria. These are finite-accuracy numerical
solves; the plots do not establish exact-proximal convergence guarantees or
theoretical guarantees for the last iterate.

## Execution and artifacts

The controller runs at most four processes at once, each restricted to one
distinct physical CPU core with one BLAS/OpenMP thread. Tuning, selection,
confirmation, validation, and report generation are automated. Execution times
are recorded for operations only; method speed comparisons are excluded.

The new experiment directory is
`experiments/scot_cpu_methods_fp64_20261014`. Frozen configuration, per-run
receipts, logs, plan hashes, numerical certificates, and the final report live
there. Input symlinks point to the prior verified SCOT reference and FP64
FraMES results. Existing experiments are not modified.

The following was the second launch command, retained as incident evidence.
Do not repeat it on this populated experiment directory.

```bash
systemd-run --user --unit=scot-fp64-methods-20261014 \
  --working-directory=/home/tony/gw_benchmark_experiment_20260927 \
  --property=StandardOutput=append:/home/tony/gw_benchmark_experiment_20260927/experiments/scot_cpu_methods_fp64_20261014/systemd_campaign.log \
  --property=StandardError=inherit --property=RuntimeMaxSec=36000 \
  --setenv=OMP_NUM_THREADS=1 --setenv=OPENBLAS_NUM_THREADS=1 \
  --setenv=MKL_NUM_THREADS=1 --setenv=NUMEXPR_NUM_THREADS=1 \
  /home/tony/gw_benchmark_experiment_20260927/.venv/bin/python \
  /home/tony/gw_benchmark_experiment_20260927/extensions/scot_cpu_methods_fp64_20261014/run_comparison.py \
  --phase campaign \
  --spec /home/tony/gw_benchmark_experiment_20260927/extensions/scot_cpu_methods_fp64_20261014/specification.json
```

Execution needs access to the local user service manager and the same local
loopback access as the existing single-rank Gloo runner. It does not submit work
to Ruche.

## Launch incident and continuation

The first detached controller survived. An incorrect process check from the
sandbox could not see host PIDs, leading to archival of live paths and a duplicate
launch. The second controller duplicated SMD work, then failed on the first
controller's existing MSBPG receipts. The original four MSBPG jobs remained
active. `launch_attempts/attempt1/ABORTED.json` contains the original incorrect
diagnosis; `launch_incident_20261014.json` explicitly corrects it. Both records
are preserved.

The four final SMD artifacts pass the frozen controller audit and independent
NumPy/POT objective and feasibility checks. The two configurations differ only
in their creation timestamp. Numerical sources, parameter choices, arrays,
iterations, precision, and seeds are identical. Duplicate executions do not
count as additional replications.

The separate `resume_campaign.py` continuation waits for the original controller
and its active workers to exit, using verified host PID identity. It audits all
eight tuning results before selecting parameters and running the six fresh
confirmations and report. It does not rerun tuning or change frozen optimizer
code. Failed stage records are archived before reconstructing the completed
tuning stage; the original campaign failure remains in the incident history.

Host process checks must use the host PID namespace. Absence from a sandboxed
process table is not evidence that a host process exited. The continuation
refuses the sandbox PID view and records its own source hash and launch receipt.

The continuation runs as user service
`scot-fp64-methods-recovery-20261014.service`. Consult
`recovery_campaign.json` for its current state and
`recovery_service_launch.json` for verified launch evidence. The old
`stage_campaign.json` failure is historical and does not describe this separate
continuation. No interactive monitoring is required.
