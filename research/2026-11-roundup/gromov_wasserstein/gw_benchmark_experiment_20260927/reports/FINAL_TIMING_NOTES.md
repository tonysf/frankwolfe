# Final campaign startup and diagnostic timing

A read-only progress check of array 1961580 tasks 4–6 confirmed that missing live CSV/metadata files do not imply a stalled solver: the runner serializes those files at termination. All three tasks completed their first selected run successfully and continued.

Tasks 4/5 launched their shard process at 2026-09-27 21:59:55 UTC, roughly four minutes after allocation; task 6 launched at 22:01:25 UTC, roughly five minutes after allocation. No cancellation, resubmission, runtime change or new numerical test was performed.

| Selected smoothing trial | Solver time (s) | Initial cumulative diagnostic time (s) | Final cumulative diagnostic time (s) | Later diagnostic increment (s) |
|---|---:|---:|---:|---:|
| metric512_s2000 / seed104 | approximately 300.001 | 96.5469576380 | 97.2298648081 | 0.6829071702 |
| metric512_s2001 / seed100 | approximately 300.000 | 96.4353551780 | 97.1519060160 | 0.7165508381 |

Both traces contain all six prescribed checkpoint opportunities. Their approximately 416-second launcher-process durations include solver work, initial diagnostics, preparation, validation, serialization and process overhead. Startup before the shard launcher is additionally visible in scheduler accounting. These are distinct clocks and should not be added together indiscriminately.

Inspection of the unchanged [runner](../run_benchmark.py) confirms synchronization before and after solver work, including in the update failure path. Output repair and diagnostics are timed separately. The initial diagnostic includes the first full-transport assignment check and may include lazy library setup; these records do not isolate the cause of the unusually long initial segment. They do establish that the large diagnostic time is concentrated in initialization rather than the six later checkpoints.

The primary current-output clock remains solver time plus the cost of producing that output. Report the complete measured diagnostic, instrumented and scheduler/process costs separately; do not present the solver clock as cold-start end-to-end latency. No timing field, budget, schedule or selected parameter was changed after inspecting these records.
