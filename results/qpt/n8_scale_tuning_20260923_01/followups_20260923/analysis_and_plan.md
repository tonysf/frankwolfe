# Late-trajectory analysis and proposed next experiment

Status: analysis and proposal only. No new job submitted, no remote changes, no message sent.

The first choice is an unchanged continuation of step scale 0.5 and momentum scale 0.6 from 148,000 to 200,000 total steps. The requested exponents 3/4,1/4,1/2 remain fixed.

| Step scale | Momentum scale | Final fidelity | Gain 100 k-120 k (pp/10 k) | Gain 120 k-138 k (pp/10 k) | Gain 138 k-148 k (pp/10 k) |
|---:|---:|---:|---:|---:|---:|
| 0.5 | 0.6 | 98.7689% | 0.2528 | 0.1667 | 0.1137 |
| 0.5 | 2 | 93.3581% | 0.7692 | 0.5864 | 0.4851 |
| 1 | 0.6 | 98.3634% | 0.0889 | 0.0751 | 0.0459 |
| 1 | 2 | 97.7448% | 0.3563 | 0.2793 | 0.2342 |

The best run gained at every saved 1,000-step checkpoint after 100 k. Its fidelity rose 97.8494%→98.7689% over 100 k→148 k while TP fell 0.238721→0.197681 (17.2%). Its rate is slowing; a flat loss trace is not evidence that fidelity has stopped improving. Changing smoothing now would mix more-iteration effects with a new penalty schedule.

The scale effects interact: at r=0.6, a=0.5 wins over a=1 by 0.4055 pp;  at r=2, a=1 wins by 4.3867 pp. Lower rho means a smaller new-gradient weight and more averaging, consistent with reduced late stochastic variation, but these runs do not measure gradient noise or isolate a unique mechanism.

Descriptive forecasts from 100 k-148 k predict 99% at 156.5 k (linear F),163.8 k (exponential infidelity),170.3 k (power-law infidelity). Recent-window power fits give roughly 170 k-175 k; this is not a confidence interval. Earlier linear forecasts substantially overpredicted held-out fidelity, including impossible values above 100%. Power fits were closer in two retrospective checks (errors-0.0474 pp and+0.0139 pp), but a nonzero fidelity-error floor remains possible. Use these scenarios to set a modest test budget, not to claim 99% in advance.

Proposed continuation:52,000 additional updates,3,407,872,000 additional row draws, one A 100/2 CPU/12 G with 30-minute cap. Measured-cost extrapolation is about 11.4 minutes runner and 6.9 minutes optimizer; phase times overlap, and queueing is separate. Preserve factor, momentum, RNG state and absolute schedule index. No new dataset, noise realization, initialization, batch, scale, offset or source change. Resume is explicit rather than another fresh run.

Primary outcome: independently verified F>=99% at 200 k. Also report the first saved crossing, all ten final checkpoints, sampled loss and TP. If it misses, stop at 200 k and reassess; no automatic extra sweep. Source, dataset, resume hashes, exact argv, resources and validation requirements are in proposed_experiment.json.

Fidelity is complete-factor overlap, separate from f (measurement least-squares) and g (TP indicator). One seed limits generalization. The unchanged continuation is the most direct and cheapest test of whether the observed shortfall is simply finite-iteration error.
