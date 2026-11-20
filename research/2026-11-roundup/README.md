# FRAMES, Gromov–Wasserstein and QPT experiment roundup

Collected on 6 November 2026 from Elite and Ruche. This directory preserves
experiment sources, configurations, reports, scalar evidence and provenance
that had accumulated outside the published `stochastic-frames` branch.

| Area | Record |
| --- | --- |
| Stochastic FRAMES | [October demos and November factorization, CT and matrix completion](frames/README.md) |
| Gromov–Wasserstein | [Benchmark campaigns, extensions and Ruche source variants](gromov_wasserstein/README.md) |
| Quantum process tomography | [Local followups, Ruche campaigns and checkout differences](qpt/README.md) |

The ordinary package and its supported entry points remain in `frank_wolfe/`
and `paper/experiments/`. These dated snapshots retain the experiments' original
layout and assumptions; they are not all portable application entry points.
Read the area README before attempting a rerun. Original absolute paths and
job scripts describe the recorded execution environment. On Ruche, execute
numerical work in an appropriate scheduler allocation after checking current
resources and dependencies.

## Source history

The published branch was `d62fcf192f27d8fb207ce61a1c466fc5059a7e1f`
(2 October). Elite's clean checkout was
`4ce1108a0692edc43b47ef465dbc74a42397765e` (8 October), with these six
additional commits, all included in the roundup branch:

- `3bf36c4`: paired Quiroga direct comparisons.
- `0586c15`: best-for-best QPT penalty policies.
- `c7a8c97`: failed endpoint diagnostics.
- `0f7857e`: opt-in full-gradient stopping for the Armijo control.
- `966ef74`: explicit Gell-Mann initialization and paired diagnostics.
- `4ce1108`: auditable tuning and an explicit exact-ray step variant.

Separate local repositories and the dirty older Ruche QPT checkout are recorded
in the area inventories. Their source snapshots are retained without replacing
the current package implementation.

## Evidence and storage

Historical reports describe their own data, seeds, numerical precision and
verification status. Collection does not independently reproduce or validate
every numerical result. Failed attempts, superseded protocols and incomplete
campaigns retain their labels. In particular, Ruche GW jobs `2297800` and
`2297801` were running and pending at inspection; their source is preserved,
but this collection does not claim a completed result for either job.

Large datasets, checkpoint arrays, environments, caches and downloaded vendor
trees remain outside Git. The area inventories identify retained and excluded
artifacts, source paths and available hashes. A separate Elite backup collects
selected Ruche QPT arrays; see [backup record](backup_record.json) for the
actual transferred scope and integrity checks. Original experiment directories
and remote job sources are preserved.

## Checking the collection

From this directory, run:

```sh
python3 verify.py
```

This verifies the recorded size and SHA-256 of every indexed file and checks
for missing or unexpected files. It does not execute experiment code or load
datasets. [Validation](validation.json) records the CPU tests of the six
unpublished QPT commits and the static checks applied to the collection.
