# Eight-qubit QPT at sigma=.05

User reopened experimentation specifically at n=8 with original noise sigma=.05 and 99% factor-fidelity target. Original n10 campaign remains stopped. No recurring automation activated.

Matched beta0=1e7, rho=2/(k+4)^.6, gamma=10/(k+10), rank1, tau20, precision64, product-state, fixed independent512-row metrics. Existing n8 dataset and initial seed reused.

Submitted array1889113 with two bounded A100 tasks,2CPU/8G/45min each. Task0 resumes validated native B8192 state at30k to200k total; task1 uses fresh sameU0 with B65536 to100k after a100-step smoke. Both stop at.99 and independently verify final factors with NumPy inside the allocation. Native resume verifies data/config and past schedule prefixes; old source, datasets and parent checkpoints preserved. Frozen validated v3 source only; no algorithm/default or main checkout edits.

Both1889113 tasks failed in CUDA initialization on ruche-gpu12 before any optimizer step. Preserved failed guard/logs. Submitted identical numerical comparison as1889115 in new guarded remote n8_fidelity99_20260913_02, restricted to ruche-gpu15 (previous successful runs). No cancellation or environment changes.

1889115_0 running on gpu15:50k F.7156572,TP1.03225. Task1 pending for resources. Increased only pending1889115_1 cap to90min based on CPU sampling cost at B65536; scontrol confirmed. No scientific setting changed. Scheduler reports existing default Account=fwllm.

Continuation reached200k: F=0.9088526491387247, TP=0.4834645018647628. Independent NumPy coefficient/matrix fidelity and TP verified, sigma=.05. Target not yet reached. Local summary/provenance/verification preserved.

Matched1,638,400,000 sampled observations: B8192/200k verifiedF.9088526491,TP.48346450; B65536/25k checkpointF.9415843912,TP.51533122 (11:17UTC). Fresh large batch uses same U0 and data, same per-iteration schedules; stronger sample efficiency at this point. Larger-batch job remains running;99% not yet reached.

11:48UTC: B65536/78k F.9802687681,TP.28197690. Original sigma=.05 remains unchanged. Prepared optional run_continuation.py and continuation_plan.json for validated100k->200k resume if needed after successful completion. Optional allocated chunk10/1 comparison requires bitwise factor/momentum/metric and sample-hash equivalence plus5% speed improvement before selecting chunk1. No new source/default changes or further submissions yet. Native n10 heartbeat remains paused.

1889115_1 COMPLETED0 in58m44s. At100k: independent NumPy F.984856510677257,TP.24452562605011202. Optimizer820.20s; runner3487.28s; host preparation3457.15s overlaps GPU. User queue inspected empty after both completed. Submitted continuation1889210 (array0) in new guarded remote n8_fidelity99_20260913_03: validated native100k->200k total, same B65536/sigma=.05/data/rho/beta/gamma/radius, stop at.99. Short chunk10/1 comparison precedes production. Parent restarts preserved. All source files unchanged. Final report independently verifies complete factors.

Chunk profile completed on A100: both repeats have bitwise-equal full factors, momentum, sampled metrics and sample hashes for chunk10 versus1. Mean optimizer+blocking transfer per100steps3.54375s ->2.50404s (29.34% reduction). Chose existing chunk1 setting; no source or algorithm change. Native resume accepted data/config/past schedule prefixes at100k. At116k F.9871835832,TP.22789030; continuation running. Local profile/provenance/remote plan copied.

TARGET VERIFIED:1889210_0 COMPLETED0 in21m20s. Stopped at148000 total steps. Independent NumPy fidelity.9900265254378854 (99.00265254%), matrix fidelity.9900265254378857, TP.1990072803857349; sigma=.05,n8. Winning optimizer total1299.8249589s, runner total4699.3940992s; batch elapsed4804s includes profile/setup. Full final/restart/data copied locally with SHA256 matching remote originals and verification. Scoped accounting successful and user queue empty. No further submissions or automation activations. N8 goal completed; old N10 target remains unmet/stopped.
