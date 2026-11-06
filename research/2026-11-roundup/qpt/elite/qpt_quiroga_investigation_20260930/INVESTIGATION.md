# Quiroga adaFGD investigation — 2026-09-30

## Finding

The failed n=4 campaign is an implementation of Quiroga et al.'s published
objective/update rule on the QPT_BFW benchmark. It is not a reproduction of
the paper's experiment, because the sensing ensemble and initialization
conventions differ and the paper does not disclose enough constants to make
its run uniquely reproducible.

## Published experiment

Source: David A. Quiroga and Anastasios Kyrillidis, “Using non-convex
optimization in quantum process tomography: Factored gradient descent is
tough to beat,” arXiv:2312.01311.

The paper constructs dense sensing matrices D_ij from:

- global generic probe states (computational states and pairwise real/imaginary
  superpositions);
- a Gell–Mann operator basis; and
- a special 2d-element pure-state-informationally-complete POVM.

Its noisy adaFGD figures are for n=2, with 128 full measurements and 96
underdetermined measurements. The n=3 result in the paper is a noiseless
measurement-count study, not the noisy benchmark shown for n=2.

The paper states a raw random complex initialization with uniform real and
imaginary parts, but does not state normalization. It gives the adaptive step
only up to proportionality and does not report the proportionality constant or
the TP penalty weight lambda. The POVM constants a and b are also not reported.
The project page's GitHub link points to the separate MiFGD quantum-state-
tomography repository, not a QPT implementation. No public QPT repository was
found in the first author's public GitHub account.

## Our experiment

Our data comes from QPT_BFW, a separate Boosted Frank–Wolfe experiment. It uses
all tensor products of four local input states, three local Pauli axes, and two
outcomes: 24^n rows. The n=4 baseline takes 8,192 uniformly selected rows from
24^4=331,776 rows. Its basis is the normalized [I, X, -iY, Z] tensor basis.
The initial factor is uniform complex and then trace-normalized.

Thus only the following were ported from Quiroga:

- the summed half-squared measurement objective;
- the exact TP penalty;
- the matrix-space gradient times U update; and
- the adaptive spectral-norm step formula.

## Scale/noise issue

Both experiments add fixed-scale Gaussian noise, but the typical Born
probability shrinks from about 1/4 at n=2 to about 1/16 at n=4. Holding sigma at
0.05 therefore makes the n=4 observations substantially lower-SNR than the
paper's noisy n=2 experiments. The paper does not establish noisy n=4 behavior.

At truth, the expected n=4 summed half-squared loss from Gaussian noise alone
is approximately 0.5 * 8192 * 0.05^2 = 10.24. The best-loss calibration point
remained around 32.47, so the run reduced loss but did not reach the truth's
noise floor. Strong TP weighting reached TP violation 0.0411 while collapsing
fidelity, indicating convergence into a different low-fidelity basin rather
than a sign or nonfinite-gradient failure.

## Diagnostic job

Ruche job 1980562 tests the audited update on QPT_BFW sensing at n=2 and n=3,
using the paper row counts 128 and 1,024. It crosses:

- noiseless versus Gaussian sigma=0.05;
- trace-normalized versus raw-uniform initialization;
- step multiplier 1e-8 through 1; and
- TP weight 0.1, 1, 10, 100.

This test can diagnose scale/noise/initialization effects inside our sensing
model. It still cannot validate the paper's different sensing model.
