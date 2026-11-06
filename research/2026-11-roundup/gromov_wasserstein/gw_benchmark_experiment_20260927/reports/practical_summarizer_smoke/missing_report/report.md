# Initial 30-second A100 practical comparison

Three development instances and one stochastic seed per method, using prespecified settings and a common solver-time checkpoint grid. This is an initial practical pilot, not a replicated final study or a completed parameter-selection exercise.

Panel status: **partial**. Successful run records: 0/18. Missing checkpoint observations: 72.

All points are current rounded outputs at measured times. Missing windows remain missing. A terminal native early stop appears only at its actual time; no trajectory is extended to the nominal budget.

| Instance / method | Status | Stop reason / issues |
|---|---|---|
| metric512_s1001 / smoothing-fw | missing | Missing metadata.json, trace.csv |
| metric512_s1001 / smoothing-fw-exact | missing | Missing metadata.json, trace.csv |
| metric512_s1001 / full-kl | missing | Missing metadata.json, trace.csv |
| metric512_s1001 / stochastic-kl | missing | Missing metadata.json, trace.csv |
| metric512_s1001 / native-sampled-gw | missing | Missing metadata.json, trace.csv |
| metric512_s1001 / fw | missing | Missing metadata.json, trace.csv |
| metric512_noisy_s1001 / smoothing-fw | missing | Missing metadata.json, trace.csv |
| metric512_noisy_s1001 / smoothing-fw-exact | missing | Missing metadata.json, trace.csv |
| metric512_noisy_s1001 / full-kl | missing | Missing metadata.json, trace.csv |
| metric512_noisy_s1001 / stochastic-kl | missing | Missing metadata.json, trace.csv |
| metric512_noisy_s1001 / native-sampled-gw | missing | Missing metadata.json, trace.csv |
| metric512_noisy_s1001 / fw | missing | Missing metadata.json, trace.csv |
| bmmc1000 / smoothing-fw | missing | Missing metadata.json, trace.csv |
| bmmc1000 / smoothing-fw-exact | missing | Missing metadata.json, trace.csv |
| bmmc1000 / full-kl | missing | Missing metadata.json, trace.csv |
| bmmc1000 / stochastic-kl | missing | Missing metadata.json, trace.csv |
| bmmc1000 / native-sampled-gw | missing | Missing metadata.json, trace.csv |
| bmmc1000 / fw | missing | Missing metadata.json, trace.csv |

Checkpoint opportunities use one common requested grid. Actual observation times are reported without moving an improvement backward to a target time.

| Instance / method | Target 1s | Target 3s | Target 10s | Target 30s | Terminal solver s |
|---|---|---|---|---|---|
| metric512_s1001 / smoothing-fw | missing run evidence | missing run evidence | missing run evidence | missing run evidence | — |
| metric512_s1001 / smoothing-fw-exact | missing run evidence | missing run evidence | missing run evidence | missing run evidence | — |
| metric512_s1001 / full-kl | missing run evidence | missing run evidence | missing run evidence | missing run evidence | — |
| metric512_s1001 / stochastic-kl | missing run evidence | missing run evidence | missing run evidence | missing run evidence | — |
| metric512_s1001 / native-sampled-gw | missing run evidence | missing run evidence | missing run evidence | missing run evidence | — |
| metric512_s1001 / fw | missing run evidence | missing run evidence | missing run evidence | missing run evidence | — |
| metric512_noisy_s1001 / smoothing-fw | missing run evidence | missing run evidence | missing run evidence | missing run evidence | — |
| metric512_noisy_s1001 / smoothing-fw-exact | missing run evidence | missing run evidence | missing run evidence | missing run evidence | — |
| metric512_noisy_s1001 / full-kl | missing run evidence | missing run evidence | missing run evidence | missing run evidence | — |
| metric512_noisy_s1001 / stochastic-kl | missing run evidence | missing run evidence | missing run evidence | missing run evidence | — |
| metric512_noisy_s1001 / native-sampled-gw | missing run evidence | missing run evidence | missing run evidence | missing run evidence | — |
| metric512_noisy_s1001 / fw | missing run evidence | missing run evidence | missing run evidence | missing run evidence | — |
| bmmc1000 / smoothing-fw | missing run evidence | missing run evidence | missing run evidence | missing run evidence | — |
| bmmc1000 / smoothing-fw-exact | missing run evidence | missing run evidence | missing run evidence | missing run evidence | — |
| bmmc1000 / full-kl | missing run evidence | missing run evidence | missing run evidence | missing run evidence | — |
| bmmc1000 / stochastic-kl | missing run evidence | missing run evidence | missing run evidence | missing run evidence | — |
| bmmc1000 / native-sampled-gw | missing run evidence | missing run evidence | missing run evidence | missing run evidence | — |
| bmmc1000 / fw | missing run evidence | missing run evidence | missing run evidence | missing run evidence | — |

No failure/warning events were present in the available run records; missing run records cannot establish absence of failures.

This single-seed development pilot supplies no error bars, fitted rates, tuned algorithm ranking, global-optimality claim, or GPU/CPU speedup conclusion. Current feasible quality and raw feasibility remain separate. The BMMC instance receives no biological interpretation. No winner is selected.

Figures:

No valid recorded checkpoints were available; no curves were generated.
