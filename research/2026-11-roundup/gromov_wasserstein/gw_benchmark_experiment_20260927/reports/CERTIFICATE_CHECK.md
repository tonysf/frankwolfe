# Recorded certificate consistency check

**18/18 runs and 11,618/11,618 evaluated points have finite required fields; 0 recorded positive excesses.** Coverage comprises 11,600 pre-update points and 18 separately evaluated terminal points. This optional scalar consistency table is not a proof, a new experiment or a replacement for the [true-gap figures](a100_diagnostics/report.md).

The recorded comparison is $G_U(\widehat P)\le G_{\mu,C}(P)+6\kappa\|P^T1-b\|_1$. Signed excess is left minus right. A positive recorded value is listed without clipping or an arbitrary passing tolerance.

| Track / instance | Runs | Expected / observed / finite | RHS min / max | Signed excess min / max | Positive |
|---|---:|---:|---:|---:|---:|
| stage_b / metric128_s1000.npz | 4 | 804 / 804 / 804 | 0.00636030734 / 165.231317 | -165.23081 / -0.00585366332 | 0 |
| stage_b / metric512_s1000.npz | 4 | 804 / 804 / 804 | 0.00462072343 / 621.970437 | -621.97026 / -0.00444306438 | 0 |
| theory_replicates / metric128_s1000.npz | 10 | 10010 / 10010 / 10010 | 0.00636030734 / 165.231317 | -165.23081 / -0.00585366332 | 0 |

Stage B has eight 200-update trajectories (n=128/512; exact and q=8/32/128 sketch, seed 0). The theory track has ten 1,000-update q=128 trajectories on one fixed n=128 development instance, seeds 0–9. These are not 18 independent datasets. Terminal points enter this pointwise inventory only, not theorem averages.

All 54 config/summary/CSV hashes match the transferred SHA256SUMS and remain unchanged. Git objects match recorded source hashes: Stage B revision `8da53a73676cd645f0707f70819f60c0a519e328`; theory revision `564d56dc5ee7563ee8d689b9d8f213bc6e68a699`. The runner/core/kernel source hashes agree across those revisions. [Machine-readable evidence](CERTIFICATE_CHECK.json) preserves all input/source hashes, coverage and per-run extrema.

The archived runner evaluates the exact smooth gradient at retained P_k, adds the current positive penalty, and computes a separate exact-assignment gap at the rounded copy of that same P_k. It subsequently evaluates terminal P_T separately. No noisy momentum gradient enters this certificate. Source anchors: `run_smoothing.py:214–215` defines κ from canonical extrema; `:221–233` computes repair and gaps; `:251–252` records RHS/excess; `:279` evaluates `info.P_before`; `:288` evaluates terminal `state.P`; `gw_core.py:179–182` computes the rounded-plan full-gradient assignment gap.

Numeric κ was not separately logged. Its source definition is verified, but its value was not independently checked against arrays. JSON reports ranges inferred algebraically from recorded nonzero column residuals solely as scalar consistency information; no NPZ was opened.

| Run | Points | Signed excess min / max | Terminal excess |
|---|---:|---:|---:|
| stage_b / smoothing/n128_exact | 201 | -92.6862645 / -0.00585366332 | -8.25441052 |
| stage_b / smoothing/n128_sketch_q8 | 201 | -165.23081 / -0.00585366332 | -9.31310415 |
| stage_b / smoothing/n128_sketch_q32 | 201 | -165.23081 / -0.00585366332 | -8.95382655 |
| stage_b / smoothing/n128_sketch_q128 | 201 | -165.23081 / -0.00585366332 | -8.44109263 |
| stage_b / smoothing/n512_exact | 201 | -605.494927 / -0.00444306438 | -21.5470679 |
| stage_b / smoothing/n512_sketch_q8 | 201 | -621.97026 / -0.00444306438 | -29.0287473 |
| stage_b / smoothing/n512_sketch_q32 | 201 | -621.97026 / -0.00444306438 | -28.9954221 |
| stage_b / smoothing/n512_sketch_q128 | 201 | -621.97026 / -0.00444306438 | -28.9676126 |
| theory_replicates / theory_replicates/seed0 | 1001 | -165.23081 / -0.00585366332 | -3.2899804 |
| theory_replicates / theory_replicates/seed1 | 1001 | -165.230795 / -0.00585366332 | -3.23020016 |
| theory_replicates / theory_replicates/seed2 | 1001 | -165.23081 / -0.00585366332 | -3.28688443 |
| theory_replicates / theory_replicates/seed3 | 1001 | -165.23081 / -0.00585366332 | -3.05564624 |
| theory_replicates / theory_replicates/seed4 | 1001 | -165.23078 / -0.00585366332 | -3.12778883 |
| theory_replicates / theory_replicates/seed5 | 1001 | -165.23081 / -0.00585366332 | -3.44773879 |
| theory_replicates / theory_replicates/seed6 | 1001 | -165.23081 / -0.00585366332 | -3.02739671 |
| theory_replicates / theory_replicates/seed7 | 1001 | -94.5371401 / -0.00585366332 | -3.22663988 |
| theory_replicates / theory_replicates/seed8 | 1001 | -165.230795 / -0.00585366332 | -3.22742208 |
| theory_replicates / theory_replicates/seed9 | 1001 | -165.23081 / -0.00585366332 | -3.13283719 |

Maximum absolute difference between stored excess and regrouped `original_gap - certificate_rhs` is 4.4408920985006262e-16. The source computes `(original_gap - true_gap) - penalty`, so binary64 subtraction grouping can differ. No tolerance was used to classify signs.

Gradients, reductions and rounded plans used float32 on A100; extracted scalar combinations used Python float. Float32 epsilon is 1.1920928955078125e-07. Maximum raw row-L1 was 3.67872417e-07; maximum rounded both-marginal-L1 was 5.88130206e-07. Exact-domain feasibility and exact arithmetic therefore remain uncertified. Epsilon times term scale is descriptive only, not a general forward-error bound. Any positive excess would be retained with its terms and scale before interpretation; none occurs in these logs.

Negative observed excess is a finite-horizon consistency observation. It proves neither the analytic inequality, manuscript correctness, a convergence rate nor optimizer competitiveness. No Stage C output or runtime changed.
