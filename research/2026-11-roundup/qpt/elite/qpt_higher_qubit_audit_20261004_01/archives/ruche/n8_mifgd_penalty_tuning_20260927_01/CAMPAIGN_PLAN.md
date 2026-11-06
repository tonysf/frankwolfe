# Coupled fixed-penalty MiFGD calibration and comparison

Seed 0 tests `(TP weight, eta)` pairs `(0.1,0.01)`, `(0.01,0.1)`, and
`(0.001,1.0)`. Their product remains `0.001`, preserving the approximate
initial TP correction scale while increasing the measurement-gradient step.

A candidate must be finite, reach TP violation at most 1.0, and fidelity at
least 0.1 after 20,000 updates. The highest-fidelity admissible candidate is
frozen. Plain FGD (`mu=0`) is then checked on seed 0. If all gates pass,
MiFGD (`mu=0.75`) and FGD are each run for 200,000 updates on held-out
sampling seeds 1--3, serialized to one A100. Existing FRAMES results are
reused only after exact initial-factor and paired-minibatch hashes match.
