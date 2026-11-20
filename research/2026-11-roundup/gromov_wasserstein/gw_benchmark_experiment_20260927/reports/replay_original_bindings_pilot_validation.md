# Real pilot cross-archive audit validation

**PASSED**: 6/6 real pilot companions matched their separately collected original archives and canonical NPZ files.

Six real pilot helper-level cross-archive checks only. This validation fixture is not a complete 796-record audit or final campaign report.

The existing replay reporter inspected the real manifest, controller ledger, receipts and proofs. The new auditor then independently compared original metadata, trace, final plans, three archived Python sources, theory configuration, and canonical NPZ hashes. It also compared original configuration, timing and counter metadata with each proof. No arrays were loaded and no trajectory was rerun.

Remote original root `/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/a100_practical/1961459` mapped to local `/home/tony/gw_benchmark_experiment_20260927/ruche_results/1961459`. Relative paths are `dataset/method`; the companion-only `pilot/` prefix was not inserted into the original archive path.

| Original path relative to collected 1961459 | Matched |
|---|---|
| `metric512_s1001/smoothing-fw` | True |
| `metric512_s1001/smoothing-fw-exact` | True |
| `metric512_noisy_s1001/smoothing-fw` | True |
| `metric512_noisy_s1001/smoothing-fw-exact` | True |
| `bmmc1000/smoothing-fw` | True |
| `bmmc1000/smoothing-fw-exact` | True |

The cache hashed 37,154,279 bytes across 57 canonical paths; all inspected files remained unchanged. Detailed hash comparisons are retained in `replay_original_bindings_pilot_validation.json`.
