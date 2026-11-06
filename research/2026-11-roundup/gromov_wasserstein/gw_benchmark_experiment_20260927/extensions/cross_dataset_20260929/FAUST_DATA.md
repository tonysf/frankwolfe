# FAUST preparation, frozen 2026-09-29

FAUST is pending a user-supplied licensed training archive or an accessible path. The user confirmed that SNARE-seq and BMMC should proceed while FAUST waits. A bounded local filename search found no existing FAUST archive. No account was created, license accepted, download initiated, or FAUST computation submitted.

The [official homepage](https://faust-leaderboard.is.tuebingen.mpg.de/) requires signup/login to download. A user with access can use the [login page](https://faust-leaderboard.is.tuebingen.mpg.de/accounts/login/) and download the training archive containing `training/registrations/tr_reg_000.ply` through `tr_reg_099.ply`. The [data license](https://faust-leaderboard.is.tuebingen.mpg.de/license) governs download/use and redistribution. Supply the existing licensed archive path; do not send account credentials. The preparation program has no network access and records the supplied archive SHA256 and license URL.

The [official overview](https://faust-leaderboard.is.tuebingen.mpg.de/overview) describes 100 training scans, ten poses per subject, and a common watertight registration template with 6,890 vertices. We use registered meshes, not high-resolution raw scans or hidden test-set evaluation. Registered template indices define our held-out correspondence; their accuracy is not the official raw-scan FAUST challenge score.

The frozen within-person pairs follow the training registration numbering convention. Before release, inspect the supplied archive documentation to confirm this numbering; a mismatch must block this stage rather than silently replace files.

| Instance | Source registration | Target registration | Vertices per view |
|---|---|---|---:|
| `faust_subject0_pose0_1` | `tr_reg_000.ply` | `tr_reg_001.ply` | 6,890 |
| `faust_subject1_pose0_1` | `tr_reg_010.ply` | `tr_reg_011.ply` | 6,890 |
| `faust_subject2_pose0_1` | `tr_reg_020.ply` | `tr_reg_021.ply` | 6,890 |

The pair list is fixed before any loss is measured. All vertices are retained, giving 47,472,100 coupling variables per pair. Source and target template orders receive independent fixed permutations, with seeds `2026092900 + 2*pair_index` and `2026092901 + 2*pair_index`. Pair index is zero-based in the table above.

Distances are shortest paths on the undirected mesh-edge graph, weighted by Euclidean edge lengths. These approximate surface geodesics and are not exact continuous-surface geodesics. Each matrix is divided by its own maximum, stored in float32, and paired with uniform float64 marginals using the frozen GW instance writer. Member hashes, archive hash, distance scales, writer hash, reader hash, package versions, and Slurm job ID bind the preparation. Identical registration face-index arrays and 6,890 vertices are required. Invalid face indices, repeated vertices within a face, zero-length edges, nonfinite coordinates, and disconnected graphs are rejected.

`prepare_faust.py` reads only six selected mesh members from the supplied ZIP or TAR archive without extracting files. It rejects unsafe paths, special/link entries, duplicate selected members, and oversized mesh files. ASCII, little-endian binary, and big-endian binary triangular PLY formats are supported without adding a mesh dependency. Tiny tests exercise parsing, distance semantics, independent-order correspondences, malformed input, safe archive handling, and the scheduler guard.

Actual processing must run within a scheduled CPU allocation. The program invocation inside that allocation is:

```sh
python prepare_faust.py --archive /path/to/licensed-training-archive.zip \
  --out /path/to/new-faust-output-directory --runtime /path/to/frozen-runtime
```

The output directory must be new or empty. `faust_preparation.json` contains an `instances` mapping with each instance's `n`, `filename`, `sha256`, metadata binding, and correspondence binding. Optimizer NPZ files contain only `A`, `B`, `a`, and `b`. Evaluation-only `.correspondence.npz` files separately hold `forward`, `inverse`, and the two population-index permutations. The optimizer must not open these files for gradients, tuning, initialization, or hyperparameter selection; metrics are computed from current feasible outputs after optimization updates.

FAUST computation is not yet released or end-to-end tested against the licensed archive. The preparation code is ready for that archive's format validation; this pending access stage must remain visible in the experiment report.
