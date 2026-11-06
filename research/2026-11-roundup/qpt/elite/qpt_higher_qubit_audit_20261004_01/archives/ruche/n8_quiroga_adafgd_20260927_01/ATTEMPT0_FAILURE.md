# Attempt 0 failure

Slurm job `1961084` failed before optimization on 2026-09-27. The shared
FRAMES runner correctly rejected the large adaFGD proportionality constant
because its generic step schedule accepts only values in `[0,1]`. All 16
candidates therefore stopped during schedule materialization and produced no
scientific results. Jobs `1961085` and `1961086` had unsatisfied dependencies.

Retry 1 passes a unit placeholder schedule through the shared validation layer
and applies the calibrated multiplier only inside the custom adaFGD update.
A two-update allocated GPU smoke test gates the full calibration.
