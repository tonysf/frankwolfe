# Pair-sampling comparison

This isolated experiment compares **FraMES with gradient momentum**, **stochastic entropic mirror descent with fresh gradients** (Fatkhullin–He), and **MSBPG with the published coupled step and momentum schedules** (Ding–Li–Toh). All three use the same fresh pair-sampling oracle. Existing experiment outputs remain separate.

At a nonnegative raw coupling `P`, each call independently draws 1,024 index pairs with replacement from `P / sum(P)` and returns the full squared-GW gradient estimate with factor `2 * sum(P)`. Workers share the same global pairs. Sampling uses actual represented row/conditional masses, including off-marginal FraMES iterates. There is no sampled-maximum gradient normalization, marginal substitution, or dense coupling gather. Pair counts are reported separately from historical random-projection probe counts. Sampling, communication and gradient construction are included in solver time.

`theory_audit.json` records conditional unbiasedness and bounded variance. Its FraMES qualification distinguishes these sufficient noise conditions from the draft's fixed-law sample-function notation. The experiment borrows the published sampling oracle, not the complete SaGroW algorithm. Both proximal baselines require actual stored-plan marginal error and guarded normalized primal-dual gap at most `1e-7`; these finite numerical tolerances are not an inexact outer convergence theorem.

## Frozen comparison

There are **36 production cells**: four datasets × three methods × three seeds (`710–712`), all from product initialization. The synthetic 512-point instance uses 200,000 updates; BMMC (6,224 cells), SNARE-seq (10,000), and SHARE-seq (31,890) each use 20,000. SHARE uses four A100s; the other instances use one. Each update makes one oracle call of 1,024 pairs, reusing the first gradient to initialize momentum where prescribed.

Real-data prefactors remain those previously selected with the Rademacher oracle. Synthetic FraMES retains its latest fixed prefactors; the two synthetic proximal baselines use explicitly prespecified theorem-safe defaults. There is no new tuning. This is a fixed-prefactor comparison, not a ranking after independently optimizing each method's parameters for the new oracle.

Rounded current/last GW loss, time to a feasible output, solver time, and work counts are empirical comparisons. Independently sampled baseline outputs remain separate. Every selected/last raw/rounded snapshot is hashed and independently evaluated in FP64; dense arrays are not persisted. Nothing is deleted from previous experiments.

## Gates and execution

`submit_campaign.sh SOURCE CAMPAIGN_ROOT` queues the entire bounded dependency graph:

1. Scheduled CPU correctness tests.
2. Scheduled numerical proofs on one and four GPUs, including independent dense comparisons for all three updates.
3. Twelve full-dataset preflights, one per dataset/method, with the actual production topology and precision. Pilots have 2,000 synthetic or 200 real-data updates. Gates check immutable source/input hashes, numerical certificates, GPU memory headroom, and a full-horizon runtime forecast including diagnostics, repairs, validation, logging and process overhead. A factor of two must fit within the 23-hour subprocess limit and 24-hour Slurm limit.
4. Three-seed production arrays after each corresponding gate succeeds; one report job runs after all submitted dependencies terminate, including failures.

No gate checks whether loss improved, truncates a horizon, changes prefactors, or silently relaxes accuracy. Failed gates block their production cells and retain their receipts. There is no background monitoring or automatic resubmission. A passed CPU check does not establish a passed A100 check; actual receipts establish each stage separately.

`stage_source.py --repo REPOSITORY --out NEW_DIRECTORY` exports a committed source package with `SHA256SUMS`. Existing canonical input directories are referenced under `CAMPAIGN_ROOT/prepared/DATASET`; manifest and array hashes are frozen in `campaign.json`.

The scheduled commands are `verify_checks.py --mode cpu|gpu`, `campaign.py preflight`, `campaign.py run`, and `report_campaign.py`. Source-bound gate receipts live under `preflight/DATASET/METHOD/gate.json`; results and failure receipts under `results/DATASET/METHOD/seedSEED/`. `report/report.json` and `report/report.csv` distinguish completed, missing, invalid and failed cells and provide summaries only for complete three-seed populations.

## CUDA determinism repair (13 October 2026)

The first one- and four-A100 checks failed before any production run: Ruche's PyTorch 2.5.1 rejects floating-point CUDA `cumsum` with strict determinism enabled. CUDA CDF construction now uses an explicit, fixed addition tree; CPU CDF arithmetic is unchanged. The repair preserves the pair law, actual-mass correction, precision and strict deterministic settings. Fresh scheduled checks cover the production CDF widths, singleton conditional blocks, repeatability and all three method updates before releasing production.
