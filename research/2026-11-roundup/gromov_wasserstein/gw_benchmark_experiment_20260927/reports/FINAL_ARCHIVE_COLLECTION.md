# Original final archive collection

**Verified:** all 723 original trials across five phases, 7,839 files and 2,147,932,319 bytes, plus 56 Slurm logs. All corresponding scheduler tasks completed with exit code 0:0; all prescribed receipts had no infrastructure error.

Every phase passed remote before/after SHA-256 stability checks and complete local file-hash verification. The combined receipt review also rechecked file lists/sizes, all 723 metadata/receipt statuses, frozen manifest bindings and all 56 task indices.

| Phase | Trials | Files | Bytes | Scientific failures |
|---|---:|---:|---:|---:|
| final_product | 253 | 3043 | 874014418 | 5 |
| final_randomized | 330 | 3348 | 919343868 | 0 |
| small_pg | 30 | 308 | 21888320 | 0 |
| momentum_product | 55 | 582 | 181447768 | 0 |
| momentum_randomized | 55 | 558 | 151237945 | 0 |

Statuses retained: {'completed': 397, 'completed_with_inner_failures': 271, 'completed_with_warnings': 50, 'failed': 5}. The five native Sampled GW failures and their earlier checkpoint traces are preserved. Inner-solver stopping failures and warnings remain visible; collection integrity does not make those trials scientifically successful.

Local archives are `ruche_results/{phase}_af7b7c5`; each sibling `{phase}_af7b7c5_collection` contains immutable before/after inventories, scheduler accounting, Slurm logs and `collection_verification.json`. Full receipt/inventory hashes and the five failure identities are recorded in `FINAL_ARCHIVE_COLLECTION.json`.

No numerical run or rerun, remote write, scheduler submission or cancellation occurred.
