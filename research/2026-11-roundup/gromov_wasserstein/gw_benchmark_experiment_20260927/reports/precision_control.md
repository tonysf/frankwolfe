# CPU precision control

Fixed frozen n=128 random-metric instance, data seed1000, noise0. Four 2,000-update controls use product initialization and the same schedule: gamma0=rho0=1, exponents (3/4,1/2,1/4), mu0=1/128. Sketch runs use 128 fresh Rademacher probes per oracle call, seed0. This is one problem and one stochastic seed, not a performance comparison or an independent-replicate study.

All metrics below are evaluated in float64 from actual solver outputs. Rounded outputs were repaired in their original solver precision and then promoted unchanged. Canonical costs were stored in float32 and promoted for the float64 solver; this isolates arithmetic precision on the same frozen problem.

| Oracle | Solver precision | Point k | Raw row L1 | Raw column L1 | Raw column L2 | True smoothed gap | Rounded loss | Momentum error |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| exact | float32 | 0 | 0 | 0 | 0 | 0.0063603 | 0.050015142 | 6.70361e-05 |
| exact | float32 | 200 | 2.58121e-08 | 0.578541 | 0.0604855 | 4.75171 | 0.049986443 | 0.548999 |
| exact | float32 | 1000 | 9.94442e-08 | 0.179655 | 0.0183697 | 2.05706 | 0.050015094 | 0.214756 |
| exact | float32 | 2000 | 1.50249e-08 | 0.106982 | 0.0109264 | 1.44027 | 0.050015144 | unavailable |
| exact | float64 | 0 | 0 | 0 | 0 | 0.0063603 | 0.050015142 | 0 |
| exact | float64 | 200 | 4.59702e-17 | 0.576238 | 0.0604537 | 4.79772 | 0.049991378 | 0.518673 |
| exact | float64 | 1000 | 6.93889e-17 | 0.179629 | 0.0183685 | 2.04956 | 0.050015122 | 0.203097 |
| exact | float64 | 2000 | 1.35308e-16 | 0.106981 | 0.0109264 | 1.43938 | 0.050015142 | unavailable |
| sketch | float32 | 0 | 0 | 0 | 0 | 0.0063603 | 0.050015142 | 172.05 |
| sketch | float32 | 200 | 2.34086e-08 | 0.565747 | 0.059181 | 4.81978 | 0.050007949 | 28.2478 |
| sketch | float32 | 1000 | 1.12148e-07 | 0.178783 | 0.0183632 | 1.967 | 0.05001156 | 16.3175 |
| sketch | float32 | 2000 | 1.84318e-08 | 0.110597 | 0.0113146 | 1.5575 | 0.05000454 | unavailable |
| sketch | float64 | 0 | 0 | 0 | 0 | 0.0063603 | 0.050015142 | 172.05 |
| sketch | float64 | 200 | 9.62772e-17 | 0.565747 | 0.059181 | 4.81978 | 0.050007948 | 28.2478 |
| sketch | float64 | 1000 | 1.58727e-16 | 0.186185 | 0.0190091 | 2.17548 | 0.050015603 | 16.3606 |
| sketch | float64 | 2000 | 5.29091e-17 | 0.111127 | 0.0113365 | 1.61201 | 0.050009111 | unavailable |

At k=200 and k=1000, the point and its same-point momentum are retained from the next update record (201/1001 updates completed when observed). At k=2000, no same-point momentum was sampled: d1999 belongs to P1999 and is not used for a P2000 error.

The true gap is the raw row-domain smoothed FW gap using the exact gradient and mu_k. CSV/JSON also preserve raw row L2, rounded marginal errors, original FW gaps at the rounded points, fresh estimator errors, and repair distances. Small signed numerical losses/gaps are not clipped.

## Interpretation

- exact: final raw column L1 is 0.106982 in float32 and 0.106981 in float64 (float64/float32=1). The actual rounded column L1 is 9.46e-08 and 1.47e-16, respectively.
- sketch: final raw column L1 is 0.110597 in float32 and 0.111127 in float64 (float64/float32=1.005). The actual rounded column L1 is 6.97e-08 and 1.54e-16, respectively.

Substantial raw column infeasibility remains with float64 in both controls. Thus these finite-update residuals cannot be explained solely by float32 roundoff. The much smaller residuals of the actual rounded outputs distinguish arithmetic-scale repair error from the raw solver residual. This supports a finite-time algorithmic explanation for this control; it does not identify a unique cause or establish an asymptotic floor.

No all-iterate averages were collected. These four checkpoints establish neither a convergence rate nor a last-iterate theorem claim, nor a numerical asymptotic floor. Matching discrete probes does not require equal optimization paths: rounding can change argmin decisions.

Matched-probe check: every one of 2000 independently replayed float32/float64 probe batches was exactly equal after promotion; actual solver generator states matched the replay after every draw. Exact-gradient runs never advanced their independent seed0 generator. Probe stream SHA256: `2619600ad47537c4f6f3058777134ae1934ed733a45fdb85803dd693dd51cc97`.

Git revision: `aaeb5fa67fe8cd43ae05fc7aad9fa529188f8caf`. The run retains executed source snapshots, exact hashes, dirty status, fixed configuration, raw/rounded checkpoint arrays and per-draw RNG-state hashes.
