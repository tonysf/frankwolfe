# Completed full SHARE-seq experiment on four Ruche A100s

**Scientific interpretation corrected on 2 October 2026:** the completed momentum-KL comparisons are exploratory tests against a custom comparator. They do not supply the requested comparison against a published stochastic algorithm under its stated theory. Their measurements are retained; the primary-baseline designation is withdrawn. See [the correction](COMPARISON_SCOPE_CORRECTION.md). Replacement GPU results are not yet available.

All six final runs completed successfully on **31,890 paired cells**, using four A100s per method. The GPU correctness gate passed all 352 checks, both full-size pilots passed, and final rounded outputs passed float64 validation. The [report and figures](shareseq_full_final/report.md) are collected locally; all ten downloaded report/figure files match remote SHA-256 digests.

At the 300-second checkpoint, the medians over three paired stochastic seeds are:

| Method | Float64 rounded GW loss | Outer updates |
|---|---:|---:|
| Smoothed row FW + momentum (sketch gradient) | 0.116899656 | 3,412 |
| Transport KL-proximal gradient + momentum (sketch gradient) | 0.116552011 | 874 |

FW completed 3.90 times as many outer updates. KL attained 0.2974% lower median final loss and lower loss in all three pairs; its median loss was also lower at the 10/30/100-second checkpoints. At 30 seconds the median update counts were 341 for FW and 122 for KL. These are comparisons of prescribed solver-time budgets, not equal oracle work or a general convergence ranking. Parameters were transferred unchanged from SNARE-10,000, without SHARE-seq tuning.

Both methods peaked at 27.414 GiB of PyTorch reserved memory on any GPU, including final validation. There were zero inner stopping failures. KL nevertheless recorded exponentiation underflow on 2,049 updates, totaling 7,685,560 zero-entry occurrences across updates; its log-state was retained, and every returned output passed validation. Maximum final rounded marginal-L1 error was 1.88e-7 or less.

Exact-cell top-1 matching remains extremely low: medians 3.3676e-5 (FW) and 4.7037e-5 (KL), against uniform baseline 3.1358e-5. Lower objective loss does not establish useful biological alignment, and no significance test against chance is claimed.

| Stage | Job | Completion |
|---|---:|---|
| Full-population distance preparation | 1981990 | 5 minutes 28 seconds, exit 0 |
| Successful four-A100 campaign | 1983902 | 41 minutes 58 seconds, exit 0, ruche-gpu19 |
| Report generation | 1983903 | 45 seconds, exit 0 |

The successful GPU allocation used 2.7978 GPU-hours. The initial hardware/driver initialization failure on ruche-gpu16 (job1981992, before scientific runs) used another 0.1567 GPU-hours, totaling 2.9544 allocated GPU-hours. Final solver work alone was 2.00071 GPU-hours; diagnostics, pilots, setup and allocation overhead are distinct. Failed-attempt logs remain preserved.

All 31,890 cells passed the unchanged joint QC from 32,231 curated source pairs. Representations were reused byte-for-byte without refitting or subsampling. Earlier 25,000-cell GPU/report jobs 1981933/1981934 were cancelled before starting.

Runtime/report revision: `8f3b092777285e625041894d406b018b95f899cf`. Preparation revision: `78b146153a0ee87d7de6394782c764c178ce3a84`.

Ruche output and approximately 48.8 GB of saved coupling shards remain at `/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/shareseq_20260930/full_campaign_8f3b092_retry_20261001`. This collection copied small reports and figures; it did not reread or download every dense plan.

No jobs remain to monitor for this experiment.
