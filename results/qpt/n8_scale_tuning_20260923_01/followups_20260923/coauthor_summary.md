Subject: 8-qubit QPT update with the paper's exponents

We completed four fresh 8-qubit runs with the usual Gaussian noise σ=0.05, batch 65,536 and 148,000 iterations, retaining step/smoothing/momentum exponents 3/4, 1/4, 1/2. Only the step and momentum scales changed; the dataset, initialization, row sequence and smoothing schedule were fixed.

The best pair was step scale 0.5 and momentum scale 0.6: 98.7689% factor fidelity and TP residual 0.19768, compared with 90.7749% and 0.35670 for the untuned requested-exponent baseline. The other scale pairs reached 93.3581%, 98.3634% and 97.7448%. None of the new runs reached 99%; the earlier schedule reached 99.0027% at the same iteration budget. These are one-channel, one-seed comparisons.

The best trajectory was still improving at every saved late checkpoint, although gains were slowing. Our proposed next test is to resume it unchanged to 200,000 total steps, preserving momentum, RNG state and schedule index. This is 52,000 additional updates on one A100, with an estimated runner time of about 12 minutes. It has not been submitted. Extrapolation suggests a crossing is plausible, but does not rule out a plateau below 99%.

We also removed redundant CPU row decoding/re-encoding while preserving exact sampled symbols and fixed noise. Host preparation improved 4.10×; three alternating pairs of 1,000-step trials gave 1.786× median runner speedup with bitwise-equal factors, momentum and diagnostics. Production host preparation fell from about 61.5 to 16.6-16.8 minutes; it overlaps GPU work. Validation used standalone exact checks; no pytest suite was run.

Fidelity here is |c†u|²/(||c||²||u||²), not the measurement-loss term f, and it does not certify TP. The report includes solid fidelity/loss/TP curves, exact schedules, timing definitions and full provenance.
