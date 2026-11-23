# Compute allocation accounting

**Complete accounting for the bound ledger.**

The one scoped scheduler snapshot was captured 2026-09-28T10:17:51.548649+00:00 to 2026-09-28T10:17:53.004440+00:00. It accounts for 113/113 expected primary allocation/task records across 15 job IDs. Known A100 allocation to this snapshot: **200,880 GPU-seconds (55.800000 GPU-hours)**, of which 200,880 GPU-seconds are terminal and 0 have accrued in nonterminal allocations. All scoped allocations are terminal and accounted for.

| Job / array | Purpose | States (primary tasks) | Accounted / expected | A100 GPU-seconds so far | A100 GPU-hours so far |
|---|---|---|---:|---:|---:|
| 1961444 | CPU environment preparation | FAILED: 1 | 1/1 | 0 | 0.000000 |
| 1961445 | CPU environment preparation retry | COMPLETED: 1 | 1/1 | 0 | 0.000000 |
| 1961447 | A100 Stage A/B validation and diagnostics | COMPLETED: 1 | 1/1 | 263 | 0.073056 |
| 1961453 | first practical pilot launch | FAILED: 1 | 1/1 | 127 | 0.035278 |
| 1961459 | 30-second practical pilot retry | COMPLETED: 1 | 1/1 | 734 | 0.203889 |
| 1961460 | ten independent small theory replicates | COMPLETED: 1 | 1/1 | 172 | 0.047778 |
| 1961471 | A100 full Stage C source and randomized initialization preflight | FAILED: 1 | 1/1 | 75 | 0.020833 |
| 1961477 | deterministic CUDA projection and full runner preflight retry | COMPLETED: 1 | 1/1 | 125 | 0.034722 |
| 1961478 | Stage C prescribed equal-budget tuning; four one-A100 shards | CANCELLED: 1, COMPLETED: 3 | 4/4 | 12,476 | 3.465556 |
| 1961516 | Scheduling-only replacement of unstarted tuning shard3: exact same291 prescribed trials split into97-run tasks | COMPLETED: 3 | 3/3 | 4,350 | 1.208333 |
| 1961539 | A100 validation of required momentum-KL comparator and shared final-panel dispatcher | COMPLETED: 1 | 1/1 | 204 | 0.056667 |
| 1961580 | Selected Stage C final product/randomized/small-PG panels and required momentum-KL supplement | COMPLETED: 56 | 56/56 | 129,644 | 36.012222 |
| 1961687 | A100 exact momentum-diagnostic replay and prospective logging preflight | COMPLETED: 1 | 1/1 | 666 | 0.185000 |
| 1961715 | tuning fixed-iteration momentum diagnostics; original outputs/timing read-only | COMPLETED: 8 | 8/8 | 8,086 | 2.246111 |
| 1961716 | finals fixed-iteration momentum diagnostics; original outputs/timing read-only | COMPLETED: 32 | 32/32 | 43,958 | 12.210556 |

Allocation categories (never added to solver or replay-controller times):

- momentum_diagnostic_reconstruction_and_validation: 52,710 A100 GPU-seconds (14.641667 hours); all terminal/accounted: True.
- original_study_and_validation: 148,170 A100 GPU-seconds (41.158333 hours); all terminal/accounted: True.

Raw scheduler rows: 337; primary rows before compressed-array expansion: 113. Excluded rows: {"job_step": 224}. CPU-only started allocations: 2, summed allocation elapsed 125 seconds, contributing zero GPU time.

Missing allocation records: none.
Accounting issues: none.

Sum primary allocation ElapsedRaw * allocated GPU count. Exclude every job step and aggregate array parent. Generic and typed GPU TRES are not added twice. Failed started allocations count; proven unstarted cancellations/pending tasks have zero accrued allocation. Active/missing/invalid accounting prevents final totals.

Limits:

- Scope is exactly the bound ledger job IDs, not unlisted local CPU work or unrelated cluster jobs.
- Allocated time is not measured GPU utilization, solver work, campaign makespan, or billed project credits.
- Original solver budgets and replay-controller times remain separate and are not added to allocation time.
- Scheduler Submit/Start/End strings retain their returned timezone convention; totals use ElapsedRaw, not timestamp subtraction.
- A100 identity follows documented Ruche gpua100 partition or explicit typed GPU TRES.
- Snapshot totals remain provisional until all prescribed allocations are terminal and accounting is complete.

Raw evidence: `/home/tony/gw_benchmark_experiment_20260927/reports/compute_accounting_snapshots/20260928T101751548649Z/sacct.psv`, SHA-256 `511e6cc97d4d5a14d2465f6f5f7d098170c7bb3b3d05861a01238925d50adeb0`. The same snapshot directory contains stderr, the exact query command/timestamps, and the bound ledger. Helper SHA-256: `eae8f1da02f02686d1fcccbf1f61c82b9b25ca4e2edc7381e48b3c981c90a412`.

Offline regeneration:

```bash
.venv/bin/python account_compute_resources.py --snapshot /home/tony/gw_benchmark_experiment_20260927/reports/compute_accounting_snapshots/20260928T101751548649Z --out reports/COMPUTE_ACCOUNTING
```

After final replay completion, an explicitly authorized refresh captures one new immutable snapshot:

```bash
.venv/bin/python account_compute_resources.py --query --ledger reports/job_ledger.json --out reports/COMPUTE_ACCOUNTING
```
