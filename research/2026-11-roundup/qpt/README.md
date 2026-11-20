# QPT experiment collection, 6 November 2026

This collection brings together the small QPT experiment files from Elite and
Ruche that were outside the published `stochastic-frames` branch. The active
package baseline is `4ce1108a0692edc43b47ef465dbc74a42397765e`. Original experiment
files retain their contents, original path, and SHA-256 in `inventory.json`.
No optimizer runs, framework probes, or cluster jobs were launched for this
collection.

## Find the work

| Logical location | Contents |
| --- | --- |
| `elite/qpt_n8_*` and `ruche/frankwolfe/results/qpt/` | Earlier dense/structured and on-demand QPT, eight-qubit continuation, momentum/adaFGD comparisons, and archived experiment recipes/evidence. |
| `elite/qpt_quiroga_*` and `ruche/qpt_quiroga_*` | Reduced-sensing experiments, calibration/validation/followup protocols, failure diagnostics, Gell-Mann checks, and tuning evidence from October. |
| `elite/qpt_frames_*_20261012` | Local FRAMES calibration, followup, four-qubit tuning, and polish sources, plans, tests, scalar evidence, and reports. |
| `elite/qpt_frames_status_20261014` and `elite/qpt_coauthor_update_20261014` | Later status/briefing builders, evidence records, and compact final PDFs. |
| `ruche/frankwolfe/` outside `results/` | Historical dirty checkout source/job-script snapshots. See the comparison below before reusing them. |
| `provenance/` | Local standalone-repository revisions/status, and the Ruche-versus-Elite source comparison. |

The names and dates distinguish historical stages, calibration, validation,
pilots, retries and failed runs. Preserve that distinction when combining
results. Plans, audit JSON and completion manifests contain more precise
scope/validity information than a report title or successful process exit.

## Restore a campaign

Byte-identical evidence is stored once, sometimes at its existing path in the
main repository. `inventory.json` maps every selected logical path either to an
`archive_path` within this directory or to a `repository_path` at the recorded
baseline. Source and launcher files remain explicit even when they are copies.
Use the helper to materialize the complete selected file layout, including
deduplicated configuration and dependency source files:

```bash
python3 restore_snapshot.py --list --prefix elite/qpt_frames_n4_tuning_20261012
python3 restore_snapshot.py \
  --prefix elite/qpt_frames_n4_tuning_20261012 \
  --repository-root /path/to/frankwolfe \
  --destination /path/to/new-qpt-snapshot
```

Run those commands from this collection directory, or give the helper's full
path. The helper validates every source checksum before copying and refuses
unsafe paths, symlinks and overwriting different files. It only restores files;
it does not run experiments. A checkout at the recorded baseline supplies the
canonical repository references. Checksum mismatches on a newer checkout are a
reason to use the baseline rather than silently substituting newer code.

Restoring another prefix or the entire inventory is supported. Original
absolute `/home/tony/` and `/gpfs/workdir/silvetian/` paths are retained in
historical sources, job scripts and evidence; relocating a snapshot does not
rewrite those paths. Review them, imports between sibling campaign directories,
the package revision, environment and data locations before rerunning anything.

## Ruche source state

The main Ruche checkout was at
`ba85bff73c306ee5f12b97aa3a24910851f585ae` with tracked/staged changes and
untracked QPT files. Its 15 inspected structured/on-demand package, test and
documentation files already match the Elite baseline byte for byte. The
distinct additions are the historical `scripts/gen_data.py` and two `.slurm`
launchers; the historical `.gitignore` difference is also recorded in the patch.

`ruche/dirty-frankwolfe-ba85bff.patch` records the tracked/staged changes against
that Ruche HEAD. Untracked files are represented separately in the inventory.
The patch is historical evidence, not an instruction to apply it to the active
package. `provenance/ruche-source-comparison.json` gives the exact hashes and
match status. The Git-bearing pinned Quiroga snapshots inspected on Ruche were
clean at `0586c15` and `c7a8c97`; the other pinned directories were source
exports, so their directory names alone are not a fresh Git verification.

The external baseline is [QPT_BFW](https://github.com/LeNavil/QPT_BFW) at
`48e9aa80250e8da065734de593afe549f05912ce`; only its locally added generation
recipe/launcher is selected here. Its third-party package trees, notebooks and
generated HDF5 dataset are not copied into this collection.

## Reproduction and scientific limits

Use the package's QPT/QPT-JAX optional dependencies and the recorded campaign
environment/launch recipes. Historical scripts may additionally require
Qiskit, the external QPT_BFW checkout, plotting/report tools or specific JAX
hardware builds. Dependency installations and scheduler requests have not been
revalidated by this archive operation. On Ruche, compute and framework tests
belong in Slurm allocations; archived launchers need review against the current
site rules before submission.

The October 12 four-qubit FRAMES tuning evidence records zero joint-gate passes
and skipped confirmation. The polish completion manifest records no recovery
winner. Final heldout channels 1000–1019 remained reserved/unused in those local
campaign records. These are noiseless rank-one nonconvex factor experiments,
not convergence proofs, noise/higher-rank results, or a completed best-for-best
comparison. Quiroga printed-rule and explicitly labelled exact-ray variants
must remain distinguishable. Later status reports do not change these limits.

## What remains outside Git

Raw NPZ/NPY/HDF5 outputs, restart checkpoints, datasets, dependency wheels,
environments, caches and transient logs are excluded, as are redundant previews,
superseded draft media, AppleDouble metadata and individual files above 2 MB.
The inventory records excluded paths/classes; original evidence still records
many checksums of the external artifacts. The earlier
`/home/tony/qpt_sync_20260923_01` full backup is preserved in place rather than
republished.

A separate metadata inventory prepared for the private Elite backup identifies
1,185 older Ruche QPT raw output files (3,631,426,157 bytes), with modification
times from 10 September through 8 October. No QPT jobs were active in the
scheduler snapshot used for this collection. File stability and backup coverage
do not imply successful or scientifically valid runs; failure, smoke and
restart evidence must retain their labels. This public source collection does
not claim to contain or independently validate those raw binaries.
