# QPT campaign: reach 99% fidelity at ten qubits

The user's requested target is at least 0.99 factor fidelity at n=10, with
larger batches and more iterations authorized. SSH, edits within the Ruche
repository, submissions and monitoring are authorized. No commit/push/switch/
pull. Preserve earlier datasets/results and the user's remote .gitignore and
staged scripts/gen_data.py. Primary noise remains sigma=.05 unless the user
answers the pending noise clarification differently. Reduced noise is a
separate control, never evidence of success at sigma=.05.

## Implementation and validation

New opt-in `product-state` measurement backend factors the existing local
sensing vectors after their Pauli transform, verifies the factorization, and
uses matrix products with O(B*2**n) workspace instead of O(B*4**n). Predictions
and real-Frobenius gradients are unchanged mathematically; no conjugation or
FRAMES/TP formula was changed. Auto/default backend behavior is unchanged.

The direct structured runner now supports --restart-path, --resume,
--fidelity-target .99, and --prefetch. Restarts retain factor, measurement
momentum, sampling RNG and absolute schedule position. Data/configuration and
schedule prefixes are validated. --steps means total steps, not added steps.
Changing batch/radius/backend/past schedules on resume is rejected. Chunk,
metrics and prefetch can change. Results contain segment-local timing and
scalar schedules, with absolute checkpoint steps. Checkpoints never include
unconsumed prefetched rows. Only one additional host chunk is prepared.

Local v3: 428 passed, 56 optional-dependency skips. Ruche v2: 482 passed,
plus 12 GPU tests. v3 full/GPU checks passed in job 1888457: 484 tests plus 13 GPU checks.
Long jobs depend on its successful exit, including the prefetch memory profile.
Frozen v3 source: results/qpt/fidelity99_prefetch_validation_20260912_01/source
under /gpfs/workdir/silvetian/frankwolfe. Hashes: source_v3_sha256.json.

## Jobs and results

All remote directories below are under the Ruche repo's results/qpt.

- 1887087: fidelity99_n8_control_20260912_01, COMPLETED 32m23s. Old immutable
  rank-one backend, n8 reduced sigma=.00625, B512, beta1e7, gamma2/(k+2),
  500k fresh steps. Final fidelity .7297081536, TP .18339126. 1777.23 optimizer s.
- 1887097: fidelity99_product_profile_20260912_01, COMPLETED 6m28s. v1 backend
  validation/profiling. A later conditional cancellation required PENDING;
  it had begun, so it was allowed to complete. Do not label this canceled.
- 1887112: fidelity99_product_profile_20260912_02, COMPLETED 15m38s, A100 gpu15.
  v2 validation, n10 batch profiles, four n8 pilots. Results below.
- 1888457: fidelity99_prefetch_validation_20260912_01, v3 validation and n10
  B65536 prefetch profile, 15-minute cap. An earlier SSH connection died before
  creating this directory or submitting; absence and scheduler history were
  inspected before submitting the actual job. Do not duplicate it.
- 1888459: fidelity99_n10_fixed_20260912_01. Main sigma=.05 run.
- 1888460: fidelity99_n10_relative_20260912_01. Diagnostic sigma=.0015625 run.

The two n10 jobs depend afterok on 1888457. Both use A100, one GPU, 2 CPUs,
12G RAM, 4-hour caps; product-state, complex128, B65536, chunk10, prefetch,
100k total steps maximum, tau40, rho=2/(k+4)^.6, beta=1e9/(k+1)^.25,
gamma=10/(k+10). Metrics every1000, fixed512 rows, metricbatch512, seed12345.
Initialization/sampling seeds0; existing truth/noise archives reused.
They save latest_restart.npz every checkpoint and final.npz on successful end,
and stop at .99 fidelity. Neither starts from truth or an earlier final factor.

All submissions used fixed new mkdir guards, quoted sbatch heredocs,
export=NONE/propagate=NONE, environment loading in the allocation, preserved
CUDA_VISIBLE_DEVICES, disabled GPU autotuning. Excluded ruche-gpu13 because
earlier CUDA initialization failed there. Anaconda module and qpt environment
are unchanged. Test dependencies are isolated under previous results.

## v2 measurements

A100 n10, 100-step profiles, two timing repeats and a separate memory run:

| Batch | Optimizer seconds | JAX peak MiB | Sampled process VRAM MiB |
|---:|---:|---:|---:|
| 512 | .7665 | 240.3 | 696 |
| 8192 | 1.7269 | 651.2 | 1720 |
| 65536 | about 9.49 | 4347.3 | 8888 |

These are different memory definitions; sampled VRAM can miss peaks.
At B65536 CPU sampling/transfer took 5.04s in addition to 9.49s optimizer,
motivating the opt-in prefetch. Its speedup still needs the v3 measurements.

n8 B8192, beta1e7, gamma=a/(k+a), 30k maximum steps, seeds0:

| sigma | a | Final step | Fidelity | TP |
|---:|---:|---:|---:|---:|
| .05 | 2 | 30000 | .00282159923 | 1.9037408 |
| .05 | 10 | 30000 | .57194503208 | 1.3561853 |
| .00625 | 2 | 30000 | .92238871712 | .17709169 |
| .00625 | 10 | 26000 | .99017278638 | .18074091 |

The reduced-noise 99% result took 47.18 optimizer seconds; elapsed includes
CPU preparation, metrics, compilation and checkpoint I/O. This validates n8
recovery under that control, not n10 or original-noise success. The larger
step schedule is promising for the new n10 runs. The old B512/500k control
only reached 73%; more iterations alone did not close the gap.

## Follow-through

Native thread heartbeat reach-99-qpt-fidelity checks every30 minutes and can
extend promising completed runs within the user's authorization. Keep at most
two long production GPU runs active. Inspect exact IDs and checkpoints before
submitting; preserve rolling checkpoints and earlier segment results. Before
claiming the user's n10 target, independently recompute complete-factor
fidelity in an allocation and report TP and the noise level. Pause the
heartbeat once the clarified target is verified.

Validated v3 is now applied to the main Ruche checkout as well as the frozen
snapshot. All seven affected files match local SHA-256 values. Original
versions are backed up under results/qpt/fidelity99_main_backup_20260912_01.
The expected old file hashes were checked before copying. The staged diff and
.gitignore hash were unchanged afterward. No commit or push occurred.

1888457 COMPLETED with exit0 in 3m54s. Prefetched B65536 timing repeats took
9.505/9.497 optimizer seconds per100 steps, 15.285/15.252 runner seconds.
Sampling/transfer blocking time fell to 1.157/1.074s, while actual host
preparation took 5.094/4.814s and overlapped GPU execution. Metric batch size
also changed from32 to512, so the entire runner-time reduction is not solely
attributed to prefetch. Raw v3 reports are in local validation_reports/.

Latest inspected production checkpoint: both n10 jobs running on gpu15,
9000 steps at 2026-09-12 18:00 UTC. Reduced sigma=.0015625: fidelity
.93926792479, TP1.3809691; fixed sigma=.05: fidelity .0000005890571993,
TP24.3129099. No n10 99% result yet.
Keep following both trajectories; do not silently label the control as
primary success. The pending noise clarification can steer the target.

Status check, 2026-09-12 18:20 UTC: both jobs RUNNING after36m17s,
21000 steps, no stderr. Reduced-noise fidelity .97323620259205, TP
.8737388171612455; original-noise fidelity .00000175637032604,
TP19.9177963103. Reduced-noise recovery is still improving; original-noise
recovery remains near zero. No new jobs or code changes during this check.

Status check, 2026-09-12 19:33 UTC: reduced-noise job1888460 COMPLETED0
after1h37m40s, stopped at57000 steps upon reaching fidelity
.9900454746610079, TP.510533018353108, sampled loss1.1605935267274708e-6.
Optimizer5410.439s. Original sigma=.05 job1888459 still RUNNING at64000steps:
fidelity6.9990771054966644e-6, TP12.41083518207164. No stderr in either.

Independent NumPy verification submitted as job1888537 in cpu_short
(2CPUs,4G,10min) against the completed reduced-noise final archive.
First submission with afterok:1888460 was rejected with a dependency error;
inspection confirmed completed predecessor, no verification job, and only
the script in the guard directory. Retried without dependency using a new
submission_retry_01 mkdir guard; retained the original directory.
Verification is pending. Reduced-noise 99% does not fulfill the sigma=.05
primary target; follow-up automation remains active.

Verified status, 2026-09-12 21:39 UTC: both production jobs COMPLETED0.
Original-noise1888459 reached100000steps, fidelity4.014800202174878e-5
(0.0040148%), TP10.407019880726523, sampled loss.0011618599174532053;
optimizer9489.553s, whole-job elapsed2h50m09s. No stderr.
Independent CPU job1888537 COMPLETED0 in8s at20:09UTC. NumPy coefficient
fidelity.9900454746610081 and matrix fidelity.9900454746610085 agree with
saved JAX value. TP.5105330183531078 agrees; truth TP1.216e-13.
This verifies the reduced-noise n10 99% result, not sigma=.05 recovery.
Local verification_n10_relative.json preserves the full checks and hashes.
The completed original-noise trajectory does not justify a blind extension;
next work should diagnose stochastic gradient variance/schedule settings at
fixed sigma=.05 before another long run. Primary target remains unmet.

Heartbeat, 2026-09-13: confirmed all previous jobs complete and user queue empty.
Submitted short A100 gradient diagnostic1888709 with fixed mkdir guard,
2CPUs,12G,15min, excludinggpu13. Uses frozen validated v3 source and
existing data/factors; no main-source or dataset changes. For each of U0,
fixed-noise final, and relative-noise final, computes same-row clean and
fixed sigma=.05 noisy gradients over128 independent batches of65536.
Reports noise/tangent norms, fidelity directional derivatives, split-half
Monte Carlo stability, saved momentum alignment and exact TP gradients.
Truth usage is diagnostic only and does not initialize/update an optimizer.
Script: diagnose_fixed_noise.py; remote directory fidelity99_gradient_diagnostic_20260913_01.

Gradient diagnostic1888709 COMPLETED0 in1m57s on A10040GB. Source hashes
match validated v3. At original-noise final factor, over8,388,608 rows:
clean mean gradient norm3.8884e-8, noise mean norm8.8836e-7 (~22.85x),
TP penalized gradient norm2.9176e-7 at beta0=1e9 (~7.50x clean mean).
Noisy split-half cosine.001778; clean split-half cosine.423863 and half
difference norm4.9469e-8, so the clean mean is also uncertain. These are
finite Monte Carlo diagnostics, not exact-gradient ratios or a proof of
impossibility. Target RMS.00138086 versus noise RMS.0500049. Momentum
norm3.1795e-7, cosine to estimated clean gradient-.00675.

Submitted array1888711, two concurrent A100 tasks, each2CPU/12G/1h.
New guarded directory fidelity99_fixed_pilots_20260913_01. No main source
or data changes. All pilots sigma=.05, rank1,tau40, same U0/seeds0,
product-state/64bit, sampled512 metrics every1000, prefetch, beta0=1e10.
Task0 sequential: (a) B65536/10k, original rho2/(k+4)^.6, gamma10/(k+10);
(b) B65536/10k, rho.2/(k+4)^.6, gamma1/(k+100). Both fresh.
Task1: B262144/chunk2/100step smoke, then fresh5000step pilot if smoke
finishes successfully, original rho/gamma. Other chunk size10.
Task0 changes momentum and steps together as a coherent lag/variance test,
not an isolated momentum effect. Batch comparisons must include both
iteration and sample budgets. fixed_noise_pilot_plan.json records choices.
Both tasks confirmed RUNNING, no stderr; source checksums match v3.

Latest pilot check: both array tasks RUNNING after8m11s, no stderr.
Weaker-TP B65536 at4000steps: fidelity1.0278680458921558e-6, TP27.4771.
Large-batch smoke passed100steps:36.3656 optimizer seconds, compiled
executable memory estimate17,337,175,764bytes (~16.15GiB); this is not
measured peak VRAM. Fresh B262144 pilot at1000steps: fidelity
7.29165364469075e-7, TP26.7466. Stronger-averaging pilot follows the
first task's weaker-TP run sequentially and has not started yet.
No early recovery claim; inspect completed comparisons at next heartbeat.

Heartbeat at2026-09-12 23:09UTC: array1888711_0 COMPLETED0 in35m23s,
array1888711_1 COMPLETED0 in34m21s; user queue empty, stderr empty.
All new pilots failed to show useful recovery. Weaker TP finalF4.9590855e-7
TP27.9749 at10k; averaging F2.2549292e-6 TP26.7400 at10k; larger-batch
F3.8156164e-6 TP27.2364 at5k. At equal1.31B draws, baseline20k F2.58917e-6.
Completed scalar traces preserved in completed_fixed_pilots_20260913.json;
comparison with configurations/timing in fixed_noise_pilot_comparison.md.

Submitted matched stronger-TP array1888731 (tasks0/1), beta1e7/1e8,
B65536/10k, originalrho2/(k+4)^.6,gamma10/(k+10), same U0 and sigma=.05.
Fresh fixed mkdir guard fidelity99_stronger_tp_20260913_01; two A100 tasks,
2CPUs,12G,30min caps, no source/default edits. Tests stronger enforcement
after weakening failed, rather than treating gradient norm dominance as
proof that the penalty causes recovery failure. Earlier beta1e7 tests
usedB512, so these complete the matched large-batch penalty comparison.
If all four beta scales remain near random overlap, no blind long extension
is justified. Primary n10 sigma=.05 99% target remains unmet.

Heartbeat result checked2026-09-13 06:46UTC: stronger-TP array1888731
tasks COMPLETED0 in18m30/18m26; queue empty, no stderr. At10k, beta1e7
F1.882388595e-6,TP.458487; beta1e8 F4.481089985e-6,TP4.316467.
Neither is useful recovery. Beta1e8 has best final fidelity among matched
10k beta1e7/1e8/1e9/1e10 tests, still only about2x initial overlap.

Reassessment of original beta1e9 baseline: describing it as fully stalled
was premature. From50k to100k, F rose4.78372377e-6 to4.01480020e-5
(8.3926x),42/50 increments positive; TP fell13.7024 to10.4070.
From70k onward27/30 increments were positive; from90k all10 were positive.
Descriptive log-fit doubling times about13k–18k steps across tail windows
are not convergence-time forecasts; checkpoints are correlated and
schedules continue changing. This is evidence for a bounded continuation,
distinct from repeating unsuccessful fresh10k sweeps.

Submitted array1888850 with fixed new guard fidelity99_resume_comparison_20260913_01.
Task0 resumes ORIGINAL immutable beta1e9 latest_restart at100k to200k.
Task1 resumes beta1e8 at10k to100k. B65536,rank1,tau40,precision64,
product-state,prefetch,chunk10,metrics every1000,512rows seed12345;
rho2/(k+4)^.6,gamma10/(k+10),beta respective scale/(k+1)^.25.
Two A100 tasks,2CPU/12G/4h caps; about2.5–3h expected per segment.
Preserve momentum/RNG/absolute schedule. Parent metadata step and source
checksums validated in allocation; parent/data hashes saved there.
Native NPZ restarts are consumed directly, never reconstructed from JSON
reports (128-bit PCG state must not pass through JavaScript numbers).
No code/default/noise changes or discarded files. Plan saved locally and
remotely. Primary target remains unmet.

Status2026-09-13 08:07UTC: both array1888850 tasks RUNNING after48m13s
on ruche-gpu15, empty stderr. Parent-step checks passed (100k/10k),
frozen source checksums OK, runner accepted data/configuration and past
schedule prefixes. Direct native NPZ resumes retain exact RNG/momentum.
Baseline at128000steps: F.00010239438644045587 (0.0102394%), TP9.43042;
2.55x fidelity since100k restart. Stronger beta1e8 at38000steps:
F.000010323219924194925 (0.00103232%), TP2.26677. At matched38000
steps this is about6.91x baseline fidelity, with much smaller TP.
Both remain far below99%; continue existing runs with no new submissions,
code edits, or schedule changes during this check. Raw scalar logs and
source/parent/data provenance saved in continuation_status_20260913_0807.json.

Status2026-09-13 08:42UTC: both1888850 tasks RUNNING after1h23m12s,
ruche-gpu15, empty stderr, unchanged source/parents. Baseline148k
F.00018934980395904724 (0.018935%),TP8.89929. Stronger beta1e8 at58k
F.000023781607860965388 (0.00237816%),TP1.83905. At matched58k the
original baseline F7.53718051e-6,TP12.9071: stronger TP has3.155x
fidelity. Its relative advantage is smaller than at38k, although both
absolute fidelities continue increasing. Both remain far below99%; let
the two existing bounded runs continue. No new jobs or scientific edits.
Raw evidence: continuation_status_20260913_0842.json.

USER STOP, 2026-09-13 09:07UTC: user asked to cancel the runs and said
the fidelity was unsalvageably low. Inspected live ownership/name/IDs,
then scancelled only1888850_0 and1888850_1. Both confirmed CANCELLED
after1h47m58s; neither remains in the queue. No files deleted.
Paused native heartbeat reach-99-qpt-fidelity; tool confirmed PAUSED.
Existing datasets, logs, results and rolling checkpoints remain in place.
Last reported metrics precede cancellation and are not final measurements.
The primary99% target was not met; campaign stopped at user request.
No further jobs, resumes or automatic experiments without a new request.
