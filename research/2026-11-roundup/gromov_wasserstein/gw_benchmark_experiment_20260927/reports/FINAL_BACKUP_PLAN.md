# Final backup plan — historical preparation

This is the historical local inventory and transfer plan. The actual outcome is recorded separately in [FINAL_BACKUP.md](FINAL_BACKUP.md). No archive, bundle, SSH call, or transfer was performed for this review. Freeze the final artifacts after the remaining replay reporting, cross-archive audit, and allocation accounting complete; then refresh this inventory against the final committed source revision.

## Audited source increment

The already backed-up remote prerequisite is `b5851980646be55a148b083a89cfbf6153dac77e`. The local snapshot inspected at 2026-09-28 08:10 UTC was `986770d53b28e3dc4cfd54da10d8881452c11375`:

| New reachable object type | Count | Uncompressed bytes |
|---|---:|---:|
| Commits | 6 | 1,532 |
| Trees | 28 | 44,892 |
| Blobs | 68 | 3,203,434 |
| Total | 102 | 3,249,858 |

The endpoint tree diff contains 65 paths. New blobs belong to experiment source/tests, explicitly tracked collection/report scripts, configuration, validation evidence, and reports. No new blob path is under data, raw data, environments, attachments, manuscript provenance, or credentials. A supplemental scan of all 68 new blob contents found no private-key, GitHub-token, AWS-access-key, or OpenAI-key pattern matches; pattern checks are not a guarantee that arbitrary secrets are absent. The reachable object range was inspected, including intermediate versions, rather than relying only on the final tree diff.

Use only the narrow increment `b5851980646be55a148b083a89cfbf6153dac77e..HEAD` after pinning and recording the final HEAD. Re-run the path/object inventory and content review for any subsequently added commits. The earlier rejected full-history bundle must never be retried; do not use `--all`, a full-history bundle, or an unrestricted repository archive. This increment has the existing remote base as a prerequisite and is not a standalone copy of earlier history.

After final source review, the local preparation is:

```bash
base=b5851980646be55a148b083a89cfbf6153dac77e
tip=$(git rev-parse HEAD)
git rev-list --objects "$base..$tip"
git diff --name-status "$base" "$tip"
# Record tip and the reviewed object inventory before creating this fresh file.
git bundle create /ABSOLUTE/FRESH/STAGING/source-increment.bundle "$base..HEAD"
test "$(git rev-parse HEAD)" = "$tip"
git bundle verify /ABSOLUTE/FRESH/STAGING/source-increment.bundle
git bundle list-heads /ABSOLUTE/FRESH/STAGING/source-increment.bundle
```

The commands above are prospective. Record the base, tip, bundle SHA-256/size, object inventory, and verification output in the eventual backup receipt. A changed HEAD requires rebuilding and reviewing the increment. Untracked accounting helper/tests/reports existed at this snapshot; their later inclusion requires the same final review.

## Generated review artifacts

Git ignores most PNG/PDF files and dense generated JSON/CSV, so the source increment alone is insufficient. Build a separate artifact archive from an explicit reviewed file manifest. Current counts below include only regular, non-symlink files with extensions `.md`, `.json`, `.csv`, `.jsonl`, `.png`, `.pdf`, `.xml`, or `.txt` in the named locations. They are inventory estimates before final closeout, not an immutable payload declaration.

| Explicit report location | Files | Bytes |
|---|---:|---:|
| Top-level `reports/` files, without recursive traversal | 78 | 3,569,371 |
| `reports/a100_diagnostics/` | 18 | 2,450,956 |
| `reports/a100_practical/` | 8 | 4,349,843 |
| `reports/a100_theory/` | 8 | 3,654,983 |
| `reports/stage_c_tuning/` | 6 | 16,227,006 |
| `reports/stage_c_final/`, including feasible references | 51 | 23,082,832 |
| `reports/stage_c_final_before_plot_margin_fix/` | 51 | 23,112,385 |
| `reports/final_manifest_audit/`, generated review files only | 3 | 12,826 |
| Current total | 223 | 76,460,202 |

This includes 60 PNG and 60 PDF figures, 49 JSON, five CSV, three JSONL, 40 Markdown, four XML, and two text files. Retain the predecessor final report and its figures/references because the margin-fix audit and failure audit bind that earlier report. The current final report SHA-256 is `39730329a43e14f0ce3af063f8263bb5b6b45e6acc45696ccbff164813d9086c`; it remains subject to explicit later report revisions.

Add these required closeout artifacts to the final manifest when ready:

- All regular generated review files in `reports/momentum_replay_completion/`, including `report.json`, `report.md`, `runs.csv`, `curve_summaries.csv`, and every PNG/PDF figure. This is the combined 796-record report, not the earlier tuning-only audit.
- `reports/replay_original_bindings_final.json` and any accompanying Markdown audit, after all required original/archive bindings have been checked.
- Final `reports/COMPUTE_ACCOUNTING.json` and `.md`, plus the specific immutable `reports/compute_accounting_snapshots/<snapshot>/` evidence files referenced by the final report, including `sacct.psv`, query metadata, ledger, and stderr. Do not omit PSV/stderr merely because they fall outside the report extension list above. Retain the preliminary snapshot if the final narrative cites it, clearly labeled preliminary.
- Updated `reports/EXPERIMENT_REPORT.md`, `reports/PROTOCOL_COVERAGE.md`, `reports/completed_artifact_index.json`, `reports/job_ledger.json`, and this plan/final transfer receipt. Reconcile their pending statements with the final completion state.

The archive can duplicate tracked report text for reviewer convenience. Source/configuration/tests are available through the audited Git base plus increment. Do not recursively include parser-smoke or constructed-fixture directories as scientific results; their explicitly tracked validation source/evidence remains in Git.

## Scope and packaging controls

Exclude `.venv/`, `data/`, `raw/`, `runs/`, `ruche_results/`, `wheelhouse/`, the full `logs/` tree, `.git/`, manuscript or other supplied attachments, archive copies of earlier source history, credentials, home configuration, and unrelated files. Explicitly tracked collection/report scripts are part of the reviewed source increment; that does not authorize copying every log. The report archive contains no coupling NPZ arrays or raw datasets. Original outputs and frozen inputs already retained in the experiment's remote archives remain separate; this deliverable is a review/source backup, not an independent raw-data replication package.

For every artifact, record its safe repository-relative path, byte length, and SHA-256 in `artifact_manifest.json` and a checksum list. Reject absolute paths, `..`, duplicate entries, nonregular files, and symlinks. Materialize the archive using exactly that file list, without broad recursive globs. Check source files against their manifest both before and after archiving; abort and rebuild if any report changes. Hash the completed archive and manifest, and record total files/bytes. Include a small README binding the final source tip, prerequisite base, principal report hashes, artifact scope, and remaining archive locations.

## Destination and verification

Use a fresh delivery directory, for example `/gpfs/workdir/silvetian/gw_benchmark_experiment_20260927/deliverables/<UTC-stamp>_<final-tip>/`, after the root agent confirms the established remote experiment path. Place the source increment, report archive, manifests/checksums, and backup receipt there. Never overwrite a running or frozen release, existing result directory, or earlier delivery; no deletion/synchronization cleanup is needed.

Before transfer, require final replay execution coverage and provenance validation, a completed cross-archive binding audit, and terminal allocation accounting. Preserve explicit failed/unverified scientific records rather than inventing successful diagnostics. After transfer, verify every delivered file checksum remotely and verify the bundle against the already present prerequisite objects. If unpacking for review, extract into a new directory after checking the exact member list and paths; leave source releases and original outputs untouched. Record the destination, final source revision, all payload hashes, verification results, and any intentionally pending item in the receipt. Only then report the backup complete.
