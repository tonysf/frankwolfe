# Quiroga-style QPT campaign summary

Campaign started 2026-10-02 and completed on 2026-10-03.  Code revision:
`d62fcf192f27d8fb207ce61a1c466fc5059a7e1f` (`stochastic-frames`).

## Validation and provenance

- New campaign runner and hardening tests: 88 targeted tests passed.
- Full repository suite: 517 passed, 55 skipped.
- Ruche smoke job `1992000`: completed, exit 0, empty stderr.
- Ruche arrays `1992001`, `1992028`, `1992340`, `1992357`, `1992362`,
  `1992370`, and `1992462`: every task completed with exit 0.
- The 248 synchronized result, script, and log files (before this local-only
  summary was added) have identical SHA-256 manifests on Ruche and Elite.
- Each JSON/NPZ result records row/observation/initialization hashes, source
  file hashes, software versions, schedules, optimizer domain, timing, and
  failure status.  Failed numerical trajectories were retained.

## n=2 calibration gate

The corresponding local artifacts are in
`/home/tony/qpt_quiroga_campaign_n2_20261002_01`.

At 2,000 effective row epochs on calibration seed 0:

| Design | Method | Fidelity | RMSE | normalized TP |
|---|---|---:|---:|---:|
| full 128 | Armijo FGD | 0.999999999951 | 1.94e-7 | 9.61e-8 |
| full 128 | printed adaFGD | 0.329718 | 4.52e-2 | 4.10e-4 |
| full 128 | FRAMES | 0.946621 | 7.33e-3 | 1.74e-2 |
| subset 96 | Armijo FGD | 0.999999999777 | 4.22e-7 | 1.35e-7 |
| subset 96 | printed adaFGD | 0.214868 | 6.15e-2 | 5.70e-1 |
| subset 96 | FRAMES | 0.998182 | 1.60e-3 | 7.10e-3 |

Full-row Armijo also recovered on held-out seeds 1--3.  The 96-row problem was
seed-sensitive: Armijo recovered on seeds 0, 2, and 3 but not seed 1.  Printed
adaFGD was poor on all held-out seeds and overflowed on full-row seed 3.

## n=4 full-design results

The design has 8,192 rows.  The shared initialization fidelity on seed 0 was
0.0158792.

### Predeclared coarse grid

- Printed adaFGD: 9 of 18 settings failed numerically.  None of the finite
  settings improved fidelity.  Roughly 94% of its runtime was spent in the
  dense adaptive spectral-norm numerator.
- Armijo FGD: the original `beta <= 100` range was finite but did not recover
  within 400 or 2,000 epochs.
- FRAMES: `beta=100` improved fidelity to 0.03993 at 400 epochs and 0.05549 at
  2,000 epochs.

### Explicit boundary refinement

Because every best point lay at the weakest TP penalty, `beta=1000` and
`beta=10000` were added as exploratory refinements.

At 2,000 effective row epochs:

| Method and selected setting | Fidelity | RMSE | normalized TP |
|---|---:|---:|---:|
| printed adaFGD, beta=100, eta_scale=0.003 | 0.009211 | 4.52e-3 | 4.03e-4 |
| Armijo FGD, beta=10000 | 0.425345 | 1.83e-3 | 2.10e-3 |
| FRAMES, beta=1000 | 0.164678 | 2.45e-3 | 2.44e-3 |

The printed update still did not improve on the common initialization.

### Held-out selected comparison

The selected printed-adaFGD and FRAMES settings were then run on held-out
seeds 1--3 at the same 2,000-effective-row-epoch budget:

| Seed | Method and setting | Fidelity | RMSE | normalized TP |
|---:|---|---:|---:|---:|
| 1 | printed adaFGD, beta=100, eta_scale=0.003 | 0.009849 | 4.56e-3 | 3.56e-4 |
| 2 | printed adaFGD, beta=100, eta_scale=0.003 | 0.011640 | 4.75e-3 | 3.84e-4 |
| 3 | printed adaFGD, beta=100, eta_scale=0.003 | 0.000789 | 4.77e-3 | 3.90e-4 |
| 1 | FRAMES, beta=1000 | 0.077822 | 2.94e-3 | 3.49e-3 |
| 2 | FRAMES, beta=1000 | 0.012927 | 3.26e-3 | 3.47e-3 |
| 3 | FRAMES, beta=1000 | 0.038065 | 3.09e-3 | 3.54e-3 |

All six tasks in Ruche array `1992462` completed with exit 0 and empty
stderr.  None passes the declared recovery gate.  Printed adaFGD satisfies
the TP threshold but fails fidelity and RMSE; FRAMES fails all three
thresholds on these held-out seeds.

### Recovery check

An explicitly exploratory 20,000-step Armijo run established that the n=4
inverse problem itself is recoverable:

| Seed | Fidelity | RMSE | normalized TP |
|---:|---:|---:|---:|
| 0 | 0.9999999923 | 1.52e-7 | 1.10e-7 |
| 1 | 0.9999999157 | 5.13e-7 | 1.96e-7 |
| 2 | 0.9999998359 | 7.17e-7 | 2.94e-7 |
| 3 | 0.9999997548 | 8.66e-7 | 3.23e-7 |

All four pass the declared noiseless gate (fidelity at least 0.99, RMSE at
most 1e-6, normalized TP at most 1e-3).  Thus failure of the printed adaptive
rule is not evidence that the sensing instance or ordinary factor GD is
broken.

## Interpretation limits

This is a Quiroga-style benchmark, not a strict reproduction: the higher-qubit
global probes/POVM are an explicit extension, the POVM constants and optimizer
constants are project choices where the paper is underspecified, and the
printed adaptive rule is compared exactly as implemented rather than silently
replaced by the cited line-search rule.  FRAMES is constrained to an
operator-norm ball, while both FGD methods are unconstrained; archives disclose
that domain difference.

The dense realization of the adaptive numerator is feasible at n=4 but
dominates runtime.  It is not feasible at n=8; the committed backend switches
to a matrix-free eigensolve there, which preserves the formula but not the low
cost seen in small systems.

No campaign jobs remain queued or running.
