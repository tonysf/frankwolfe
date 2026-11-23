# Longer stochastic GW runs

This extension increases the horizon from 2,000 to 20,000 updates using the
previously selected FraMES, fresh-gradient entropic SMD, and theorem-parameter
MSBPG settings. It asks whether the loss continues decreasing substantially.
It adds no tuning and changes no numerical optimizer source.

The initial scope is BMMC (6,224 cells) and SNARE (10,000 cells), three final
seeds and both original starts: 36 runs, approximately 50 A100 GPU-hours based
on the completed short runs. Full SHARE would add approximately 364 GPU-hours
and is deferred pending the user's scope preference. Timing estimates exclude
queuing, extra diagnostics and possible growth in inner-solver work.

The runners restart with the original seeds because prior artifacts contain
plans but not complete momentum, FP64 log-plan, and RNG state. The schedules
depend only on iteration index. The independent theorem-output sampling law
uses the new horizon; it does not change the optimization trajectory.

`campaign.json` binds the inputs and candidate catalogs. `frozen_selections.json`
binds each chosen prefactor to the prior selection file and input hashes.
All runs retain q=128 fresh probes, one gradient call per update, original
initialization/output seeds, and both inner tolerances of 1e-7. MSBPG preserves
the precise coupling and index shift in its documented theorem specialization.

Common current-output loss checkpoints are 0, 100, 300, 1,000, 2,000, 3,000,
5,000, 7,500, 10,000, 12,500, 15,000, 17,500 and 20,000. Intermediate diagnostics
use the optimizer's FP32 arithmetic; independently selected and final outputs
also receive FP64 validation. Neither the 1e-7 inner certificates nor a small
objective change establishes an outer GW optimality gap.

Ruche's workdir was 469/500 GB on 7 October. These runs save scalar traces,
certificates, provenance and all four output hashes, with independent FP64
validation, but do not persist dense plans. They therefore support loss and
timing analysis; new correspondence metrics cannot later be computed from
these hashes alone. Existing dense artifacts are preserved.

Submit an immutable source package once with `submit_campaign.sh`. One small
scheduled GPU integration job tests the new controller with all three methods
and both starts. Each dataset/method then runs the complete 20,000-update
seed-810 product-start trajectory. Its remaining five replicas depend on its
successful numerical validation, regardless of the observed loss. The report
runs after all jobs reach terminal states and exposes failures or omissions.
No background polling or automatic resubmission is installed.
