# Quiroga adaFGD baseline at n=8

This campaign implements the paper's Burer--Monteiro rank-one QPT objective
with a fixed trace-preserving penalty and an adaptive FGD step. The paper
specifies the step only up to proportionality and does not report the fixed TP
weight. At n=8, its matrix-space adjoint residual would be a dense
65536-by-65536 matrix, so the implementation uses the Frobenius norm of its
action on the current factor as a documented structured proxy. The denominator
is the published measurement-prediction norm.

On calibration seed 0, a logarithmic 4-by-4 grid tests TP weights
`1e-5, 1e-3, 1e-1, 1e1` and proportionality constants
`1e2, 1e4, 1e6, 1e8` for 2,000 updates. Every candidate uses the identical
initial factor and identical fixed set of 65,536 noisy measurements. Finite
candidates with terminal TP violation at most 1 and fidelity at least `1e-5`
are ranked by fidelity, then TP violation, then the smaller proportionality
constant. The winner must pass a 20,000-update seed-0 confirmation.

After freezing both constants, held-out sampling seeds 1--3 run for 200,000
updates each, serialized to one A100. Each run reuses one fixed seeded
measurement set, matching the paper's fixed-data optimization protocol. FRAMES
uses resampled minibatches, so the two methods share initialization and update
budget but are not claimed to have paired minibatches.
