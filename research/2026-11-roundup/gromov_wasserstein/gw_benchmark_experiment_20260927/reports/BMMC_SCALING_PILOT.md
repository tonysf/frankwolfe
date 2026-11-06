# Larger real-data extension: BMMC scaling pilot

**Pilot 1968346 failed during its 180-second environment-validation timeout, before numerical work. Retry 1968437 completed all four runs at both sizes and passed eligibility. Its one-seed outputs remain pilot evidence; the replicated campaign is complete; see its final report.**

The [prospective plan](../extensions/bmmc_scaling_20260928/README.md) freezes a paired comparison between **Smoothed row FW + momentum (sketch gradient)** and **Transport KL-proximal gradient + momentum (sketch gradient)**. Both use q=128, the same momentum schedule, product initialization, stochastic seed 100 and 300 solver seconds. They run sequentially in the same one-A100 allocation. Checkpoints are 30/100/300 seconds, with full objective evaluation, repair and float64 endpoint validation retained.

Start at 3,000 cells per view (9 million coupling variables). Advance to 6,224 cells per view (38,738,176 variables) only after both smaller outputs validate and the memory/deadline gate passes. The embedding dimensions remain 50 RNA and 38 ATAC features. This increases transport problem size rather than feature dimension.

The pilot transfers the original fixed settings; momentum-KL is not independently retuned. One seed supports feasibility and preliminary scaling evidence, not a replicated ranking. A larger dimension does not guarantee a larger FW advantage because both methods share the gradient estimator. Within-size feasible loss versus solver-plus-output-repair time is the primary comparison; actual updates, probes, inner iterations/failures and memory remain visible.

The new launcher/configuration is committed at `157e143660d1138a68bd93a19e1a2c427dded7e9`. It invokes existing immutable runtime `b5851980646be55a148b083a89cfbf6153dac77e`. Only the reviewed extension files and a public h5py wheel were staged, with all transferred checksums verified. The exact data-reader reconstruction check runs in the allocation before larger inputs are generated; original raw BMMC files already on Ruche are reused.

[Submission record](BMMC_SCALING_PILOT.json). Remote outputs: `/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/bmmc_scaling_20260928/job_1968346`. Slurm log: `/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/logs/bmmc-scale-1968346.out`.

The request is one A100, two CPUs, 32 GiB host RAM and a two-hour ceiling; actual allocation time will be recorded after termination. Monitoring should be sparse and limited to this job; unchanged state does not warrant repeated reports. No automatic tuning or replication campaign is launched. Original results, failures and source releases are unchanged.

The subsequent user go-ahead authorized the [full independently tuned campaign](BMMC_SCALING_CAMPAIGN.md), now submitted behind retry 1968437 and a separate CPU eligibility check. The original failed attempt and immutable source are preserved.
