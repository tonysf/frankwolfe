# FraMES with 8,192 probes on the frozen synthetic instance

This experiment tests whether increasing the stochastic gradient probe count from
2,048 to 8,192 brings forward the synthetic loss reduction. It reuses the exact
`metric512_s1000` matrices and three development seeds (710, 711, 712). Each run
starts from the product coupling and executes 200,000 updates. There is no
loss-based stopping or pruning.

The fixed schedules are `gamma_k=(k+1)^(-3/4)`,
`rho_k=0.3*(k+1)^(-1/2)` after initializing momentum with the first fresh gradient,
and `mu_k=10/[512*(k+1)^(1/4)]`. The smoothing prefactor remains 10. The user
replaced the proposed smoothing-prefactor comparison with this probe-count test;
no smoothing-comparison jobs were submitted.

## Diagnostics and interpretation

Loss and marginal diagnostics are recorded at initial state, updates
1, 2, 5, 10, 20, 50, 100, 200, 500, and every 1,000 updates through 200,000.
The current and final losses are optimization diagnostics, not claims about a
theorem-prescribed randomly selected output. The final raw and rounded snapshots
receive independent FP64 objective and marginal checks; dense plans are not saved.

At the same plan indices strictly below the terminal horizon, a read-only hook
observes the actual momentum after its fresh-gradient update and before the FW
step. It records the row-centered momentum and analytical full-gradient norms,
momentum error, penalty-to-momentum norm ratio, LMO choices with and without the
penalty, selected-column concentration, and deviation from a product coupling.
The full gradient is used only for diagnostics. A common row offset cannot affect
the row-simplex LMO, so direction norms remove that offset. Norm ratios alone do
not prove that the penalty determines the LMO; choice agreement is also recorded.
The momentum discrepancy includes both sampling error and momentum lag; it is
not an isolated estimate of oracle variance. No terminal momentum diagnostic is reported because no momentum at index T is
formed.

Diagnostics consume no probes and do not modify the optimization tensors. Their
elapsed time is recorded separately from solver wall-clock time. End-to-end
checkpoint time includes preceding instrumentation and preparation. Solver time
includes initialization, probe generation, stochastic gradients, momentum/FW
updates and acceptance audits; it excludes diagnostic full gradients, losses,
rounding, logging, snapshots, and final validation.

`baseline_report.json` is the byte-identical, previously validated completed
higher-batch report, SHA256
`6ce0b0198875649d7a524abceef5d8c5e4d904c7711bd8d22949c28534808640`.
The comparison uses its three q=2,048 synthetic runs. Their checkpoints are much
sparser, and shared seeds do not imply nested probe tensors across batch shapes.
Report lines connect observations and do not identify an exact transition time.
This one clean synthetic development instance does not establish performance on
real data or a general method ranking.

## Ruche execution

The new source is exported from a committed revision into a separate immutable
source directory and verified by SHA256 before submission and execution. Prepared
input is linked to the existing frozen synthetic export and revalidated.

An A100 preflight checks instrumented/uninstrumented trajectory parity, the
q=2,048 adapter against the prior runner, and q=8,192 production-shape memory,
feasibility and runtime. Three independent production tasks depend on successful
preflight. Resource validation includes dense diagnostic overhead and a runtime
safety factor before permitting the fixed horizon.

Each GPU task requests one A100, two CPU cores, and 16 GB host RAM. Production
allocations last at most 2h15 with a 2h process timeout, below Ruche's 24h limit.
The array permits at most three concurrent tasks; Slurm additionally enforces the
user's four-A100 limit. The automatic CPU report follows all terminal task states
and preserves incomplete/failed results. No optimizer-resume claim is made:
checkpoints contain diagnostics, not restartable state.

`submit_campaign.sh` preserves partial job receipts and refuses duplicate
submission to the same campaign directory. No active monitoring, automatic
resubmission, or changes to earlier campaign source directories are performed.
