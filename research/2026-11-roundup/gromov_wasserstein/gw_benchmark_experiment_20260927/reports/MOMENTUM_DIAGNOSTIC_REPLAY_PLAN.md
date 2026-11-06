# Separate replay of missing practical momentum diagnostics

Prepared 2026-09-28. This is an acceptance plan, not evidence that replay has succeeded. The original optimization results, timings, counters, parameter selection and failure classifications remain authoritative and unchanged. Replay produces separate diagnostic attachments only after trajectory verification. It must not retune, continue optimization beyond the original completed updates, replace failed outputs, or create new benchmark timings.

[Protocol §10, lines 425–435](../PROTOCOL.md#10-metrics-and-logs) requests momentum-gradient error at the same point, alongside current raw/rounded output metrics and true gaps. The archived practical smoothing runners explicitly omitted that error because the stored momentum and post-update output have different indices. [Protocol §11](../PROTOCOL.md#11-timing-and-checkpoint-rules) also requires diagnostic costs to be separate and prohibits treating sparse checkpoints as a full-trajectory theorem average.

## Frozen scope and original archives

| Original panel | Required trajectories | Methods | Source commit |
|---|---:|---|---|
| Development tuning | 504 | 252 smoothing-FW + 252 smoothing-row-projection | `aaeb5fa67fe8cd43ae05fc7aad9fa529188f8caf` |
| Selected product starts | 121 | 55 smoothing-FW + 11 exact-gradient smoothing + 55 row projection | `af7b7c549351b81de2aa886b987bc3e56ffeff9c` |
| Selected randomized starts | 165 | 55 each of the same three methods | `af7b7c549351b81de2aa886b987bc3e56ffeff9c` |
| Original practical pilot | 6 | Three instances × sketch/exact smoothing | `564d56dc5ee7563ee8d689b9d8f213bc6e68a699` |
| **Total** | **796** | **504 tuning + 286 selected finals + 6 pilot** | Separate source bindings |

These are prescribed records, not a claim that all selected finals have completed. Missing, failed or incompatible originals must remain visible in coverage. The 110 momentum-KL supplemental cells already contain a separately labeled last-update momentum diagnostic and are outside this repair's prescribed scope. Native Sampled GW, plain stochastic KL and unrelated pilot/theory runs are not added to it.

Tuning archives are under `ruche_results/tuning_aaeb5fa/tuning/`; the six pilot originals are under `ruche_results/1961459/{metric512_s1001,metric512_noisy_s1001,bmmc1000}/{smoothing-fw,smoothing-fw-exact}/`. A complete practical run contains `metadata.json`, `trace.csv`, `final.npz` when available, `failures.json`, and `source_snapshot/{run_benchmark.py,smoothing_fw.py,gw_core.py,theory_config.json}`. Campaign runs also have `campaign_receipt.json` and a source/manifest-bound shard ledger. The pilot predates these campaign receipts and initial-plan hashes; absence must be recorded as a historical schema distinction, not filled with fabricated provenance.

The pilot's directory label `smoothing-fw-exact` still has `config.method="smoothing-fw"` and `config.oracle="exact"`. Preserve both the scientific label and actual implementation when identifying records. All six inspected pilot runs have the stated `564d56d` source and completed status.

| File | Tuning `aaeb5fa` | Selected `af7b7c5` | Pilot `564d56d` |
|---|---|---|---|
| `run_benchmark.py` SHA256 | `56540d3db1b13260b5200acb8719392c2f97d466a76d13bc9ab44d5ea8023da3` | `569e0bce0eee05b6897dbff1b4fd5056e87145305f2d26e38f06b5caf5889485` | `e553cf4890e73caeb729e24c2a205857f82e8e495d8933c72cabc97283d19ac0` |
| `smoothing_fw.py` SHA256 | `edf880d12394c681e2cbc36b1bf53b45d51d22ba8988c3ce671fce8d6bb45dd6` | Same as tuning | `60607e0d49634bbaa4d7b18aae94eafd6314573449ff8de629f3924cfef5a2d7` |
| `gw_core.py` SHA256 | `5df67e09c61f3cf273128d26ae8ba840fac2f009bc23062d5cbe9f67d58ec444` | Same as tuning | `2cf01a2d084c6f6535ce9ba81838e6c53c33c321b9f07caadec572b9a95b3d04` |

The archived theory config SHA256 is `574be9cf43846a382fc35ea1f52df824ae774aa1c3bb4d73927372d00bbe7484` in the inspected tuning and pilot snapshots. Verify its contents against recorded theory metadata. The pilot initialization API has no supplied-P0 argument, its config has no `initialization` key, and its nullable gamma0/rho0 resolve to one in `smoothing_config`. Later sources support matched randomized P0 and replace CUDA simplex `cumsum` with a fixed-order scan. Use the original, hash-verified module versions in isolation; never silently import the current working copy instead. The three-run early-output preview intentionally lacks source snapshots and is not a complete replay source archive.

## Required mathematical alignment

The original `StepInfo` contract pairs `P_before`, `d_used` and `gradient_estimate` at update index k. After that step, `state.P=P_(k+1)` and `state.d=d_k`. A practical trace row with `iteration=K` evaluates output P_K, after K completed updates.

The agreed replay attaches only quantities from actual original states:

| Original output row | Primary actual-used momentum error | Separate current-output tracking error | Required labels |
|---|---|---|---|
| Initial, K=0 | `||d0 - grad f(P0)||_F`, with d0=g0 | Same quantity | output index 0, momentum point 0, alignment true |
| Any output, K>=1 | `||d_(K-1) - grad f(P_(K-1))||_F` | `||d_(K-1) - grad f(P_K)||_F` | output index K, momentum point K-1, alignment false; tracking error explicitly stale |

Capture the last actual `StepInfo` pair; do not relabel the post-update output as its pre-update input. For K=1 the primary error is still the initialized d0/P0 error. Initialization draws exactly one gradient; the first update reuses it. No new terminal gradient sample or EMA update is permitted. At final output P_N the original run has no d_N; replay must not invent one or advance the solver to obtain it. Extra exact gradients for measurement consume no oracle randomness and belong only to replay diagnostic counts.

Both norms concern the **smooth** gradient. The exact column penalty is excluded from momentum and from its reference gradient. Keep the runtime diagnostic dtype and norm convention explicit; a promoted float64 check, if included, is a distinct field. Absolute Frobenius error is required; relative error is optional and must be null or explicitly undefined when the reference gradient norm is zero. Exact-gradient smoothing generally has nonzero EMA error after initialization when rho<1; it must not be reported as identically zero.

**Literal protocol limitation:** the primary recovered estimator error at a K>=1 checkpoint belongs to P_(K-1), whereas the displayed loss, feasibility and true smoothed gap belong to P_K. The stale tracking diagnostic does not convert d_(K-1) into a current-point d_K. This attachment makes the actual-state alignment explicit and remedies the missing error log, but it must not claim that a d_K/P_K momentum measurement was present at every current output. It also supplies no expectation estimate, fitted rate, last-iterate theorem, or full-trajectory average.

## Acceptance before attaching an error series

1. Bind each record to its original metadata/trace/final/source hashes and, where present, campaign receipt, shard manifest, source revision and frozen selection. Verify canonical NPZ and A/B/a/b hashes; load only these mathematical arrays, never planted correspondence. Hash originals before and after replay to demonstrate no mutation.
2. Reproduce the original dtype, device class, PyTorch/CUDA numerical environment, deterministic settings, CUBLAS workspace, TF32/AMP policy, thread count, oracle seed and archived module behavior. CPU checks test the utility but do not establish CUDA trajectory identity. A differing environment is a recorded incompatibility unless exact trajectory verification succeeds under an explicitly recorded compatibility policy.
3. Reconstruct product or randomized initialization using the original routine and independent initialization RNG. Match the recorded initial-plan hash/permutation when available. For the historical pilot, use the archived product construction and record that no original initial hash exists. Reconstruct the resolved `smoothing_config`, including exact mu0, gamma/rho powers, oracle/batch and row-projection eta; do not replace nullable pilot CLI values by new defaults.
4. Replay a fixed number of original **completed iterations**, not the old wall-time budget. Visit the exact saved trace iteration indices. Preserve each row's kind, target window and original actual solver/output time. Do not create an error row for a missed window, invent a new checkpoint, or use replay wall time as the original x-axis.
5. At each observed row, compare reconstructed raw loss; row-L1, column-L1/L2, mass error and minimum entry; rounded loss/residuals; repair L1; true smoothed gap and mu; original assignment gap where recorded; last-update schedule parameters; and cumulative oracle/probe/LMO/projection/update counts. Compare numerical fields in their original precision, without clamping negative cancellation. Timing fields are preserved, not numerically replayed. Missing or nonfinite recorded values cannot silently pass. Field coverage must be recorded, not reduced to one convenient loss check.
6. For a successful original, require matching saved final raw **and rounded** arrays, including shape and dtype. Prefer bitwise element identity for a deterministic replay under the same numerical environment; record hashes and maximum differences even on rejection. Do not substitute a marginal-only or objective-only match. Archive any tolerance policy before production replay; never loosen it after seeing discrepancies. Scalar comparisons should use the same explicit fixed policy, with exact integer/index/count equality.
7. Verify final original algorithmic counts: for N>=1 smoothing updates there are N oracle evaluations, one initialization oracle call, and q*N probes for sketch; exact smoothing has N exact oracle calls. New exact diagnostic calls are separate. Capturing a checkpoint must leave the oracle RNG, state and subsequent trajectory unchanged. At the terminal checkpoint, no extra call/update is allowed.
8. Only verified rows/runs may feed attached error tables or plots. Keep a coverage table with all 796 prescribed records and distinct missing, pending, failed-original, replay-mismatch and verified categories. Replay success is an integrity finding about an existing trajectory, not a replacement optimization result. Preserve original source and trial paths alongside replay source/environment/timing and diagnostics.

## Failed originals and rejected replays

An original `failed` status stays failed and has no eligible replacement final score. A replay must not run beyond its recorded completed prefix, retry a stochastic sample to avoid the failure, synthesize the missing budget checkpoint, or turn a preserved earlier output into a successful endpoint. Retain the original exception, failure events, attempted/completed counts and incomplete timing caveats.

The bounded utility will reject these attachments as `unsupported_original_failure`, retaining the prescribed record and original status in coverage. This avoids assuming that a saved raw array necessarily belongs to the last completed update after a partially failed attempt. Recovering a failed run's prefix would require a separately anchored, explicitly partial verification; that additional recovery is outside this utility's scope. Reproducing a finite earlier prefix alone does not diagnose the original failure's first invalid operation. Missing originals remain pending/missing; no fallback or substitute run is accepted.

Source/input mismatch, final-array mismatch, scalar/count/index mismatch, unverified archive fallback, altered RNG stream or incompatible initialization must fail closed: archive comparison evidence and withhold a verified error series. A failed replay never changes the original scientific status or frozen selection. Native Sampled GW's already documented failed final replication is outside this smoothing replay and remains in the final campaign failure counts.

## Bounded utility tests and production preflight evidence

| Test | Concrete acceptance/failure condition |
|---|---|
| Two-step indexing and first-sample reuse | Hand-computed d0/P0 and d1/P1 errors match; output K=1 uses d0/P0; stale tracking field uses P1; no extra initialization sample. |
| Exact-oracle EMA control | Initial error is zero to diagnostic arithmetic; a changing-gradient example with rho<1 has the expected nonzero subsequent EMA error. |
| Checkpoint instrumentation isolation | Enabling sparse diagnostics leaves every replayed final raw/rounded array and RNG/count state equal to a replay without diagnostics. No terminal sample/update. |
| Archived-source isolation | Deliberately wrong source hash or ambient-module substitution is rejected; pilot's old initialization API and later randomized API each use their own source. |
| Initialization/config binding | Wrong oracle seed, initialization seed/permutation, rho0, eta or mu0 is detected by provenance or trajectory comparison. |
| Scalar/count tampering | Alter a recorded residual, gap, schedule or counter while retaining the final arrays; replay still rejects the attachment. |
| Final-state tampering | Alter one raw or rounded saved entry without altering trace metrics; replay rejects on saved-array validation. |
| Missed-window handling | A missed row yields no fabricated error/output; original row indices and actual times survive unchanged. |
| Original failure handling | A failed original remains failed and returns `unsupported_original_failure`; no diagnostic CSV, substituted endpoint or hidden coverage omission. |
| Original immutability | Metadata, trace, final plan, receipt and source/input hashes remain unchanged after replay, including on utility failure. |

Before a full replay allocation, use a bounded allocated-A100 preflight covering archived pilot sketch/exact, campaign sketch/exact, a row-projection trajectory and a matched randomized start. Check actual original saved states and scalar fields under the production numerical environment. Local toy tests alone are insufficient. Keep this preflight separate from optimizer performance evidence. The implementation's supported comparison fields, tolerances and failure statuses must be reviewed against this plan before treating production attachments as verified.

## Frozen preparation and local validation

The three [companion manifests](../configs/momentum_diagnostic_replays/) bind all 796 prescribed originals and disjoint output roots. Successful reconstruction requires exact scalar/count equality and bitwise equality of both final arrays; no numerical tolerance will be relaxed after observing a mismatch. Direct compilation of the hash-bound archived source bypasses cached bytecode.

Local validation passed 67 integrated focused tests, followed by an 11-test utility rerun after the cached-bytecode guard. Independent reviews cover mathematical alignment, archive isolation, failure evidence, the guarded launcher and Slurm integration. [Validation and source hashes](momentum_replay_cpu_validation.json). These CPU checks are not A100 validation.

The allocated preflight is prepared to verify four archived af7 toy trajectories, two actual aaeb tuning trajectories, six prospective toy runs with the corrected logger, and all six historical pilot trajectories. The verified pilot attachments will count toward the prescribed 796 records and will not be repeated as new scientific runs. The remaining 504 tuning and 286 final replays require a successful preflight; final replays also wait for original array 1961580.

The original affected solver budgets total 46,470 seconds (12.9083 GPU-hours). This is a scope/cost proxy, not a replay runtime prediction: fixed-iteration reconstruction has separate startup, synchronization and diagnostic overhead. Record actual replay process and scheduler costs.
