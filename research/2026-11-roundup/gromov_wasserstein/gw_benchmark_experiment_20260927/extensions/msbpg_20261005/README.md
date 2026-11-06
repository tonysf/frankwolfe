# Published MSBPG comparison

This extension adds Ding–Li–Toh (JMLR 2025) MSBPG, equations (8)–(9), with the
coupled parameter prescription of Theorem 24 and the diminishing schedule in
Remark 25. [THEORY.md](THEORY.md) gives the complete GW specialization,
global kernel construction, computable constants and qualifications.

The new method key is `published-msbpg`. Historical constant-step momentum-KL
runs are not used as this method. The completed FraMES and fresh-gradient SMD
results remain unchanged and are bound by SHA256 in `frozen_baselines.json`.

The new campaign uses the same BMMC 6,224, SNARE 10,000 and full SHARE 31,890
cell instances, q=128 fresh Rademacher probes per update, product/randomized
starts, development seeds 710–712 and final seeds 810–812. There are 216 new
development runs (12 admissible parameter candidates per dataset, 200 updates)
and 18 new final runs (2,000 updates). Only the theorem's free constants are
tuned, inside their proved bounds. Selection uses the same development loss
criterion as the previous campaign and never reads final outcomes.

Both inner stopping tolerances remain 1e-7. The mathematical theorem assumes
exact proximal steps; finite-tolerance, finite-precision runs are qualified
numerical implementations. The theorem's independently selected normalized
update statistic is recorded separately from endpoint GW loss and uses the
theoretical alpha, not the effective KL step eta.

`submit_campaign.sh` submits a Slurm dependency graph on Ruche: validation,
tuning, selection, final runs, then a report. Production tasks cannot start
unless the new one/four-A100 checks and actual-size smoke runs pass. Small
instances use one A100 per trial, and SHARE uses four. Prepared arrays are
reused without copies. Dense output retention matches the previous campaign
for this one additional method (approximately 46 GB); smokes and development
runs retain validated metrics and hashes without dense output files.

The source package is frozen by commit and checksum before submission. Run
requests, parameter constants, every inner certificate, output hashes and
scheduler receipts are retained. A failure stops dependent work and is
reported; there is no silent resumption, fallback candidate or retuning.

Wall-time comparisons against the frozen baseline use separate campaign
allocations on the same hardware class. All stochastic gradients, momentum,
Sinkhorn iterations and certificates count as solver time. Diagnostics,
serialization and final FP64 validation are measured separately. These are
fixed-update comparisons; differences in endpoint loss and solver time do not
by themselves establish time to equal accuracy.
