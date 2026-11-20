# Initial A100 diagnostic candidate pilot

One development instance per size, one optimization seed, and prespecified candidate batches. This is a diagnostic candidate pilot, not a completed tuning exercise, held-out final study, convergence proof, or algorithmic speed comparison.

Panel status: **incomplete**. Completed smoothing runs: 0/8; completed oracle reports: 0/2.

| Expected run/report | Status | Missing evidence or issue |
|---|---|---|
| n128_exact | missing | Missing config.json, summary.json, trace.csv |
| n128_sketch_q8 | missing | Missing config.json, summary.json, trace.csv |
| n128_sketch_q32 | missing | Missing config.json, summary.json, trace.csv |
| n128_sketch_q128 | missing | Missing config.json, summary.json, trace.csv |
| n512_exact | missing | Missing config.json, summary.json, trace.csv |
| n512_sketch_q8 | missing | Missing config.json, summary.json, trace.csv |
| n512_sketch_q32 | missing | Missing config.json, summary.json, trace.csv |
| n512_sketch_q128 | missing | Missing config.json, summary.json, trace.csv |
| oracle n=128 | missing | Expected oracle report is missing. |
| oracle n=512 | missing | Expected oracle report is missing. |

Figure interpretation: plotted objective values are current rounded outputs, never best-selected values. Time plots show only recorded checkpoints. True-gap means include every pre-update iterate and exclude the extra terminal point. With one optimization seed, these are realized trajectory statistics, not estimates of theorem expectations. The exact-gradient ablation retains the same momentum schedule. Signed numerical values are preserved; gap plots use a symmetric logarithmic scale near zero. No slopes are fitted, and no GPU/CPU algorithmic speedup conclusion is drawn.

No valid recorded trajectories or oracle panels were available; no curves were generated.
