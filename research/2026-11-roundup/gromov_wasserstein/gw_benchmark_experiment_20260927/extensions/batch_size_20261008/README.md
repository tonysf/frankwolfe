# FraMES batch-size diagnostic

Compare 128, 1,024, and 2,048 independent Rademacher projection probes per
gradient call. Each probe involves the whole dataset; these counts are not
numbers of sampled cells. The only optimizer parameter varied within a dataset
is the probe count. There is no control variate in this experiment.

| Frozen input | Updates per run | Step prefactor | Momentum prefactor | Smoothing prefactor |
|---|---:|---:|---:|---:|
| Full BMMC, 6,224 cells | 20,000 | 1 | 0.3 | 0.1 |
| Clean synthetic `metric512_s1000`, 512 points | 200,000 | 1 | 0.3 | 10 |

All 18 trajectories use product initialization and development seeds 710–712.
Step, momentum, and smoothing exponents remain 3/4, 1/2, and 1/4. The BMMC
prefactors reproduce the current stalled configuration. The synthetic
prefactors reproduce the earlier stochastic FraMES configuration, with a
horizon covering its roughly 160,000-update stagnation. One synthetic instance
is a diagnostic, not a population-level performance study. These development
runs do not establish a final method ranking or replace the held-out campaigns.

The same seed identifies each batch comparison, but different tensor shapes
consume different random draws: probe streams are not claimed to be nested.
All trajectories finish their predeclared horizons; no loss-based early
stopping, candidate elimination, retuning, or best-iterate selection is used.

## Numerical implementation and reporting

`run_batch.py` adapts only the batch restriction and source hashing of the
existing `published_stochastic_20261002/run_published.py` entry point. It rejects
other methods and unsupported batches. All numerical code, momentum
initialization, schedule indexing, rounding, and independent FP64 output
validation remain in the original runner. Its original files are unchanged.
The controller verifies actual probe work as `batch * completed_updates`.

The report compares full feasible GW loss against completed updates and
recorded solver time plus the repair of that output. It also records raw loss,
column residual, repair distance, total probe work, stage memory, and separate
timing components. Intermediate full-objective diagnostics are float32; final
raw and rounded output values are evaluated independently in float64. Rounded
current-iterate loss is an empirical diagnostic, not the theorem's output
statistic. Time coordinates are observed medians, not interpolation to a common
time budget. A batch's median requires all three valid seeds.

No dense plans are persisted. Hashes and FP64 validation of all four sampled
and final raw/rounded outputs are retained. Scalar checkpoints do not provide
optimizer resume. Each run has a two-hour execution timeout inside a 2h15
allocation, below Ruche's 24-hour limit. A failed or partial run remains failed;
the controller does not silently resume or replace it.

## Submission and provenance

Sources are exported from one reviewed Git commit into a new checksummed
directory. Existing queued sources and results are untouched. The full BMMC
prepared arrays are reused through a link. The small synthetic archive is
exported locally by `prepare_synthetic.py`, preserving every value and dtype;
its manifest and array hashes are frozen in `campaign.json`.

The first Slurm job runs 200 updates at batch 2,048 on both production shapes.
It requires valid finite outputs, at least 10% GPU-memory headroom, and a
runtime estimate that fits the full-run timeout with a factor-two margin.
Passing is unrelated to loss improvement. Two nine-task arrays require this
job to succeed, and every run independently verifies the gate's provenance.
One A100 and two CPUs are requested per run; each array permits two concurrent
tasks, and Ruche's four-GPU user limit remains applicable. The report job runs
after all jobs terminate and exits nonzero if any scientific result is missing
or invalid. No active polling or automatic resubmission is installed.

Submission entry point on Ruche:

```sh
bash extensions/batch_size_20261008/submit_campaign.sh IMMUTABLE_SOURCE NEW_CAMPAIGN_ROOT
```

`submission.tsv` is written after each accepted scheduler operation. Its
exclusive submission marker prevents accidental duplicate submissions.
Before using the entry point, populate the new root's `prepared` directory
and verify the source and data checksums.
