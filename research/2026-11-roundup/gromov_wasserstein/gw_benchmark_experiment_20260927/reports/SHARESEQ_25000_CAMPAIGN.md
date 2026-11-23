# SHARE-seq 25,000 cells on four Ruche A100s

Superseded at the user’s request by the full QC-passing population of 31,890 pairs. GPU job 1981933 and report job 1981934 were cancelled while pending, with zero elapsed time and no start date. The completed 25,000-cell preparation is retained for provenance. See SHARESEQ_FULL_CAMPAIGN.md for the replacement campaign.

- Preparation job **1981928 completed** in 8 minutes 3 seconds, peak process RSS 5,974,724 KiB. Of 32,231 source pairs, **31,890 passed joint QC**; the fixed paired 25,000-cell subset and cost matrices are frozen.
- Four-A100 job **1981933 was cancelled before starting**. The original submitted design was: It runs 352 allocated GPU correctness checks, then two 30-second full-size pilots. Memory, feasibility, source and matching gates must pass before final runs.
- CPU report job **1981934 was cancelled before starting**.

The two methods are **Smoothed row FW + momentum (sketch gradient)** and **Transport KL-proximal gradient + momentum (sketch gradient)**. Each uses all four GPUs sequentially, with matched seeds and product initialization. Three seeds receive 300 solver seconds per method; the order alternates. Constants are transferred unchanged from selected SNARE-10,000 configurations. This is an exploratory SHARE-seq comparison without dataset-specific tuning.

Rows of the coupling, momentum, A and A squared are distributed; B and B squared are replicated. Communication enters solver time. Final float64 validation follows release of solver state and validates the two outputs sequentially. About 30 GB of final coupling shards will be saved. The unchanged objective and estimator are retained, with floating-point differences from distributed reductions explicitly acknowledged.

Local checks passed: seven preparation tests, three kernel test groups, 352 four-process CPU agreement checks, and 45 runner/campaign tests. The allocated GPU checks must still establish A100/NCCL correctness and 25,000-cell memory fit. The report script also passed six-run synthetic rendering and failure-case checks.

Source: `extensions/shareseq_25000_20260930/`. GPU runtime revision: `44d54fc814893d7d057d866c58e6b195f9d6c90e`.

Ruche output: `/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/shareseq_20260930/campaign_44d54fc`.

Submission receipts and frozen preparation manifests: `logs/shareseq_20260930/`. Check these named jobs only; no repeated scheduler polling is scheduled.
