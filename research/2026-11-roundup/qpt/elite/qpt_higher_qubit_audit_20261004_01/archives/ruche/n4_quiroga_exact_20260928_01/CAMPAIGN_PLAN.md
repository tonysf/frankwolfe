# Exact Quiroga adaFGD versus FRAMES at n=4

The n=8 experiment establishes a scalability boundary: the published adaFGD
step requires the spectral norm of a 65536-by-65536 matrix-space gradient.
The scalable factor-gradient proxy is not presented as the published method.

At n=4 the process dimension is 256, so the matrix-space gradient is only
256-by-256 and its Hermitian spectral norm is computed exactly with `eigvalsh`
at every update. The TP penalty and its factor gradient are also exact.

Both methods use the same deterministic fixed subset of 8,192 stored noisy
measurements per seed. This is the natural continuation of the paper's
full-measurement coefficient from n=2 and n=3: `C_n=2^(n+1)` and
`m=C_n*4^n`, giving `m=8192` at n=4. Quiroga adaFGD uses the whole fixed set
at every update; FRAMES draws size-32 stochastic minibatches from that same
set. Both use the same benchmark initialization.

Seed 0 is reserved for calibration because the paper does not report either
the adaptive-step proportionality constant or the fixed TP weight. Held-out
seeds 1--3 will be compared only after those constants and the FRAMES beta
choice are frozen. Fidelity, TP violation, measurement loss, optimizer time,
and all subset hashes are retained.

Calibration tests the Cartesian grid `eta_scale in {0.1,1,10}` and
`tp_weight in {0.01,0.1,1}` for 100 exact full-subset updates. Finite
candidates satisfying TP violation at most 1 are ranked by fidelity. If the
short grid has no admissible point, its lowest-TP finite point may proceed only
to a 2,000-update confirmation; held-out runs remain gated on confirmation TP
at most 1 and fidelity at least 0.1. FRAMES beta values
`{10,100,1000,10000}` are independently calibrated on seed 0. The frozen
methods then receive 10,000 updates on each held-out seed 1--3, serialized to
one campaign GPU.

## Paper-normalized retry (`paper_sum_retry2`)

The first calibration above is retained as an audit artifact but is not used
for comparison: it implemented the measurement objective as a mean over the
fixed subset and used the real-Frobenius factor gradient. Quiroga et al.'s
published equations instead use the summed half-squared residual and the
matrix-space gradient applied to the factor. This changes both the gradient
and the adaptive numerator by factors proportional to the measurement count.

The retry implements the published update as
`U <- U - eta * (A^dagger(A(UU^H)-f) U + lambda grad_chi(H) U)` with
`eta = c ||A^dagger(A(UU^H)-f)||_2 / ||A(UU^H)||_2`. The Hermitian spectral
norm and TP gradient are evaluated exactly. A numerical formulation audit
checks the mapping against the existing mean-loss real-factor gradient and a
finite-difference TP derivative before calibration starts.

Because the paper specifies neither the proportionality constant `c` nor the
fixed TP weight `lambda`, seed 0 tunes
`c in {1e-8,1e-7,1e-6,1e-5,1e-4}` and
`lambda in {0.1,1,10,100}` for 500 updates. A selected setting must then pass
a separate 5,000-update seed-0 gate (finite, TP violation at most 1, fidelity
at least 0.1). Only after that gate do held-out paired seeds 1--3 run for
10,000 updates. FRAMES remains frozen at its previously calibrated seed-0
choice `beta0=100`; it is not retuned on held-out seeds.

The paper reports n=2 and n=3 rather than n=4. Thus this is a faithful
implementation of the reported objective and update rule, but the n=4 data
volume and tuned constants are explicitly an out-of-paper extrapolation.
