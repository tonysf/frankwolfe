# Full SHARE long-horizon supplement

The user requested all three datasets on 7 October 2026. This supplement adds
the full 31,890-cell SHARE dataset: FraMES, fresh-gradient entropic SMD, and
theorem-parameter MSBPG, each with three seeds and both original starts, for
18 new 20,000-update runs. Existing BMMC and SNARE runs remain separate.

The input hashes, previously selected parameters, q=128 gradient probes,
initialization seeds, and two 1e-7 inner tolerances are unchanged. The frozen
selection artifact is copied byte-for-byte from the initial long-horizon
campaign. There is no retuning or numerical optimizer change.

Ruche's documented and checked `gpua100` walltime limit is 24 hours. Each
SHARE run uses four A100s, eight CPUs, and 128 GB host memory:

| Method | Runner timeout | Slurm walltime |
|---|---:|---:|
| FraMES | 2 hours | 2 hours 15 minutes |
| Entropic SMD | 12 hours | 12 hours 15 minutes |
| Theorem-parameter MSBPG | 18 hours | 18 hours 15 minutes |

The measured 2,000-update medians extrapolate to roughly 34 minutes, 5.6 hours,
and 9 hours respectively for 20,000 updates. These are estimates; extra
diagnostics and changing inner-solver effort can increase runtimes. The
requests provide headroom and remain within the site limit. Ruche's four-A100
per-user cap serializes these full-dataset jobs. The array throttle is one.

The checkpoints currently written are loss/certificate diagnostics, not
resumable optimizer state. These jobs run each trajectory in one allocation;
they do not yet recover momentum, FP64 log-plans, or RNG state after a timeout.
A timed-out run remains visibly failed, with its available diagnostics
preserved; it is never accepted as a completed 20,000-update run. A resumable
extension would need implementation and equivalence testing if these larger
allocations prove insufficient.

`submit_campaign.sh SOURCE PREPARED ROOT TAG GATE_ID` accepts a new campaign
root and a fresh GPU integration gate. The earlier successful gate (2036886) has
aged out of Slurm’s live controller, so it cannot be reused as a dependency.
It queues one complete seed-810 product-start sentinel per method, then the
other five cells after that sentinel passes all numerical validation. Its
report waits for all submitted jobs to reach a terminal state. Existing
full-size four-A100 numerical validation still applies because numerical
runner files are unchanged. The source package and all configuration/selection
bindings are verified before execution.

The command excludes `ruche-gpu13`, where the preceding submission had three
CUDA-unavailable failures before its first update, along with the already
excluded `ruche-gpu16` and `ruche-gpu19`. Failed evidence is retained separately.

Dense plan storage remains disabled, respecting the measured workdir quota.
Scalar traces, output hashes, independent FP64 final validations, and the
existing iteration checkpoints are retained. No background polling, automatic
resubmission, or new tuning is installed by this supplement.
