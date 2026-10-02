# Quiroga-style global-probe QPT sensing

`paper/experiments/qpt_quiroga_sensing.py` adds the reduced sensing design of
Quiroga and Kyrillidis, "Using non-convex optimization in quantum process
tomography: Factored gradient descent is tough to beat" (arXiv:2312.01311,
cited below as [32], its number in the boosting manuscript). It is a separate
backend.
The local-Pauli benchmark, its data formats, runners and defaults are
unchanged, and it remains the default QPT experiment.
`paper/experiments/qpt_quiroga_frames.py` connects the new rows to stochastic
FRAMES and to the adaptive update printed in [32].

## Two benchmarks

| | Local-Pauli benchmark (retained) | Reduced global-probe design (new) |
|---|---|---|
| Inputs | 4^n product states from `{\|0>, \|1>, \|+>, \|+i>}^(x)n` | d^2 = 4^n global probes `\|k>`, `(\|k>+\|l>)/sqrt2`, `(\|k>+i\|l>)/sqrt2` |
| Measurement | 3^n local Pauli settings, 2^n outcomes each | one 2d-outcome POVM, informationally complete for pure states |
| Rows | 24^n | 2 d^3 = 2 * 8^n (128 at n = 2) |
| Code | `qpt_structured_*`, `qpt_generate_data` | `qpt_quiroga_sensing`, `qpt_quiroga_frames` |
| Coordinates of `chi = U U^H` | normalized `[I, X, -iY, Z]` tensor basis | Choi (matrix-unit) coordinates |
| Row value for truth `V` | upstream convention: Born probabilities of `conj(V)` | Born probabilities `Tr(E_j V rho_p V^H)` |

For n >= 2 the global probes are not local preparations: four of the sixteen
two-qubit probes are entangled. [32] remarks that its probes "can be
implemented through the Pauli preparation basis"; that is literally true only
for one qubit, although both input sets span the d-by-d operator space. [32]
mentions the `{X, Y, Z}^(x)n` settings only as the traditional 12^n-circuit
alternative to its POVM. Its experiments use the reduced design: noisy n = 2
results with all 128 rows and with 96 rows, plus a noiseless n = 3
measurement-count study.

## Conventions

[32] restates the POVM but reports neither its constants nor code, so the
following are explicit choices.

- **Choi coordinates.** A rank-r factor `U` has shape `(d**2, r)` with
  `U[i*d + o, a] = K_a[o, i]`, input index major. Then
  `J = U U^H = sum_ij |i><j| (x) E(|i><j|)` is the Choi matrix of Baldwin,
  Kalev and Deutsch (Phys. Rev. A 90, 012110 (2014), Eq. (7)); `Tr J = d` for
  a trace-preserving map.
- **Rows.** Row `s = 2*d*p + j` senses
  `f_s(U) = Tr(D_s^H U U^H) = Tr(E_j E(rho_p))` with
  `D_s^H = D_s = rho_p^T (x) E_j` (Baldwin et al., Eq. (12)). Tests compare
  every row at n = 1, 2 with Born probabilities of explicit Kraus operators.
- **Probes.** Index order `|k>` for `k < d`, then `(|k> + |l>)/sqrt2` and then
  `(|k> + i|l>)/sqrt2`, each over `k < l` in lexicographic order (the order
  of the list in [32] and Baldwin et al. Eq. (9)).
- **POVM** (Baldwin et al. Eq. (18), after Flammia, Silberfarb and Caves,
  arXiv:quant-ph/0404137): outcome `0` is `a|0><0|`; outcome `m` in `[1, d)`
  is `b(I + |0><m| + |m><0|)`; outcome `d - 1 + m` is
  `b(I + i|0><m| - i|m><0|)`; outcome `2d - 1` is the throw-away element
  `I - sum(others)`. Flammia et al. use the opposite sign for the imaginary
  elements; either choice is pure-state complete.
- **Constants.** By default `a = b = 2 / (4d - 1 + sqrt(8d - 7))`: the largest
  common value for which the throw-away element satisfies `E >= b I`, which is
  `1 - b` times the positive-semidefinite limit. This gives `b = 0.2, 0.1,
  0.0519, 0.0270, 0.0140, 0.00721, 0.00368, 0.00187` for n = 1, ..., 8.
  Custom constants are passed together (`povm_a`, `povm_b`) and validated with
  the closed-form smallest eigenvalue
  `1 - 2b(d-1) - (a + sqrt(a**2 + 8 b**2 (d-1)))/2`; the other elements have
  spectra `{a, 0}` and `b{0, 1, 2}`. Tests check positivity, completeness,
  Hermiticity and linear independence numerically for n = 1, ..., 6, and
  Flammia's explicit pure-state inversion.
- **Operator basis.** [32] writes `chi` in a Gell-Mann/Pauli basis. Any
  orthonormal operator basis gives `chi' = W chi W^H` for a unitary `W`, with
  the same row values and TP penalty; factored-gradient iterates correspond
  under `U -> W U`. Only basis-dependent choices, such as an entrywise random
  initialization, differ.

Every real combination `sum_j w_j E_j` has the form
`alpha I + beta|0><0| + |0><u| + |u><0|`. Hence predictions and factor
gradients cost `O(r d)` per sampled row and `O(r d**3)` for all rows, in
bounded probe chunks, and no `D_s`, `J` or other `d**2`-by-`d**2` matrix is
formed. The exact TP map `Tr_out(U U^H) = (sum_a K_a^H K_a)^T`, its residual,
penalty gradient `2 (R (x) I) U` and Jacobian adjoint also cost
`O(r d**3)`. The legacy runners' TP map is the transpose of this one in the
matrix-unit basis, with the same residual norm and factor gradient. Passing
`xp=jax.numpy` gives JIT-compatible operators.

## Data, noise and sampling

`generate_quiroga_sensing_data(n, channel_seed=...)` uses the Haar unitary of
`qpt_generate_data` for the same seed, as a rank-one Choi truth. Modes:

- `noiseless`: exact probabilities, computed on demand.
- `gaussian`: row `s` receives `fixed_row_noise(s, noise_seed, noise_std)`,
  the random-access `splitmix64_box_muller_row_v1` recipe of the on-demand
  local-Pauli mode, indexed by this design's row number. Values are invariant
  to batching, order and repetition, and are not clipped.
- `shots`: each probe independently draws `shots` outcomes of the full POVM
  from a multinomial seeded by
  `SeedSequence((shot_seed, 0x51504F56), spawn_key=(probe,))`, and reports
  frequencies. This requires a CPTP truth.
- `stored`: a complete observation vector, for example from `materialize()`.

`noise_std` is accepted only in `gaussian` mode and `shots` only in `shots`
mode, so a requested noise level is never silently ignored. `sample_rows`
draws uniform rows with replacement; `fixed_row_subset(size, seed)` draws a
sorted subset without replacement, such as the 96 of 128 rows of the
underdetermined n = 2 setting of [32].

## FRAMES and adaFGD

`run_quiroga_sensing_stochastic_frames(data, ...)` has the defaults,
initialization, operator-norm LMO, TP prox, schedules, sampling stream and
checkpoint alignment of `run_qpt_stochastic_frames`. It never forms a dense
process matrix; checkpoint loss and gaps are exact over the active rows, and
fidelity uses the rank-one truth. `rows=` selects a fixed subset. Choose
`tau >= sqrt(2**n)`: the default `tau=10` contains the initializer and the
rank-one truth only up to n = 6.

```python
import numpy as np

from paper.experiments.qpt_quiroga_frames import run_quiroga_sensing_stochastic_frames
from paper.experiments.qpt_quiroga_sensing import generate_quiroga_sensing_data
from paper.experiments.quantum_process_tomography import PowerSchedule

data = generate_quiroga_sensing_data(
    4, channel_seed=0, observation_mode="gaussian", noise_std=0.01, noise_seed=0)
result = run_quiroga_sensing_stochastic_frames(
    data, rows=data.fixed_row_subset(4096, seed=0),  # rows=None uses all 2*8**n
    n_steps=10_000, tau=np.sqrt(data.d) + 1.0, batch_size=32,
    metrics_frequency=1000, show_progress=False,
    rho_schedule=PowerSchedule(2.0, 4.0, 0.6, cap=1.0),
    smoothing_schedule=PowerSchedule(10.0, 1.0, 0.25),
    step_size_schedule=PowerSchedule(2.0, 2.0, 1.0, cap=1.0))
print(result.process_fidelity[-1], result.tp_violation[-1])
```

For n <= 3, `data.to_dense_qpt_data(rows)` builds a `QPTData` with
`A_basis` set to the matrix-unit basis. Tests at n = 1 and 2 show that the
unmodified dense NumPy and JAX runners then reproduce the structured runner's
trajectory and metrics.

`quiroga_adafgd_step(data, U, eta_scale=c, tp_weight=lam)` implements the
adaptive update printed in [32],
`U <- U - eta (A^H(A(UU^H) - f) U + lam grad_chi H(UU^H) U)`, with the summed
loss, the unhalved `H = ||Tr_out(UU^H) - I||_F**2`, and
`eta = c ||A^H(A(UU^H) - f)||_2 / ||A(UU^H)||_2`. [32] reports neither `c`
nor `lam`, so both are required arguments. Pass `observations=` when looping
so targets are not regenerated at every update. A zero or numerically
negligible denominator raises because this printed quotient is undefined.

## Matrix-free adaptive numerator

The numerator is the spectral norm of the Hermitian operator
`A^H(r) = sum_s r_s D_s`. It does **not** require materializing its
`d**2`-by-`d**2` matrix: `design.full_adjoint(v, r)` applies that operator
to a vector in `O(d**3)` time and `O(d**2)` working memory. By default,
`quiroga_adafgd_step` uses a dense Hermitian eigensolve as a small-system
oracle through n = 5, then a deterministic, tolerance-controlled ARPACK
Lanczos solve using this exact matrix-vector action. Tests compare the
matrix-free and dense norms at n = 1, 2 and 3. The iterative numerical
eigensolve is approximate to its requested tolerance, just as a dense
floating-point eigensolve is numerical; it does not change the operator whose
norm is requested.

One matrix-free norm evaluation costs multiple full operator passes, with the
number determined by spectral convergence. It therefore remains substantially
more expensive per iteration than FRAMES even though the dense-memory and
`O(d**6)` dense-eigensolver barriers are gone. The optional dense oracle
would occupy `16 d**4` raw bytes (256 MiB at n = 6, 4 GiB at n = 7 and
64 GiB at n = 8), plus eigensolver workspace.

The printed step in [32] also needs a scientific qualification. [32] describes
it as coming from its reference [47], but [47] gives an exact line-search rule
`mu = ||P_S grad f||_F**2 / ||A P_S grad f||_2**2`, not the printed
spectral-norm quotient above. The printed quotient is not scale-covariant:
rescaling the sensing operator and observations changes its effective update
by the wrong power, and rescaling the factor and observations exposes the
same problem. It also tends to zero with the residual. This implementation is
useful for auditing the literal printed rule, but its constants must be
calibrated per sensing design; poor results from it are not evidence that
fixed-step or line-search FGD fails.

## Scaling, n = 2 to 8

Counts are exact. Timings are for rank one on an AMD Ryzen 5 PRO 3400G
(4 cores, 8 threads) with NumPy 2.4.6 and OpenBLAS. Kernel timings come from
`python -m paper.experiments.qpt_quiroga_sensing --min-qubits 2 --max-qubits 8
--time`: each kernel runs in its own loop, and the reported value is the
median of three runs after one warm-up. The FRAMES column is optimizer time
per step, without checkpoint metrics, averaged over 200 steps at batch size
32. Each size ran in a fresh process.

| n | Probes | Outcomes | Rows `2*8**n` | Local-Pauli `24**n` | float64 targets | Full loss + gradient | Exact TP | FRAMES step | Optional dense adjoint | Optional dense eigensolver `d**6` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 16 | 8 | 128 | 576 | 1 KiB | 0.19 ms | 0.021 ms | | 4 KiB | 4.1e3 |
| 3 | 64 | 16 | 1,024 | 13,824 | 8 KiB | 0.23 ms | 0.021 ms | | 64 KiB | 2.6e5 |
| 4 | 256 | 32 | 8,192 | 331,776 | 64 KiB | 0.44 ms | 0.024 ms | | 1 MiB | 1.7e7 |
| 5 | 1,024 | 64 | 65,536 | 7,962,624 | 512 KiB | 4.6 ms | 0.046 ms | | 16 MiB | 1.1e9 |
| 6 | 4,096 | 128 | 524,288 | 191,102,976 | 4 MiB | 42 ms | 0.13 ms | 1.1 ms | 256 MiB | 6.9e10 |
| 7 | 16,384 | 256 | 4,194,304 | 4,586,471,424 | 32 MiB | 0.30 s | 0.82 ms | 3.0 ms | 4 GiB | 4.4e12 |
| 8 | 65,536 | 512 | 33,554,432 | 110,075,314,176 | 256 MiB | 2.4 s | 4.0 ms | 12.5 ms | 64 GiB | 2.8e14 |

Full passes grow by about eight per qubit, as `O(r d**3)` predicts. The
n = 2 to 8 timing process peaked at 541 MiB RSS; the FRAMES processes peaked
at 133 MiB, 376 MiB and 1.07 GiB for n = 6, 7, 8. A rank-one factor occupies
`16 d**2` bytes (1 MiB at n = 8). The optional dense adaFGD adjoint needs
additional eigensolver workspace beyond the raw matrix sizes in the table
(the measured peak at n = 5 was about 50.7 MiB versus a 16 MiB raw matrix).
Dense `D_s` tensors would need
`16 * 2 d**3 * d**4` bytes, already 8 GiB at n = 4, so `to_dense_qpt_data`
and `dense_sensing_matrices` refuse above `2**26` entries. In an n = 8
FRAMES step, about 40% of the time goes to the exact TP map and Jacobian
adjoint (BLAS products), a quarter to the shared operator-norm LMO, and 14% to
the 32-row stochastic gradient. These are CPU measurements of NumPy kernels;
this backend has no GPU scan runner.

## Signal level and fair comparison

Flammia et al. call this POVM "a poor candidate indeed for an actual
tomographic procedure": the identity in the middle elements makes their
outcomes nearly equiprobable. At a Haar truth (channel seed 0, all rows),
the mean row probability is `1/(2d)`. The amplitude-dependent part `p - b`
of the middle outcomes has root-mean-square `b sqrt(2/(d(d+1)))`, matching
the Haar prediction to within 3%: 0.031, 0.0089, 0.0023, 6.1e-4, 1.6e-4,
4.1e-5 and 1.0e-5 for n = 2, ..., 8. Local-Pauli probabilities have mean and
standard deviation of about `1/d` (standard deviation 0.195, 0.108 and
0.058 for n = 2, 3, 4).

With all rows, the Jacobian with respect to the `2 d**2` real factor
coordinates at the truth has rank `2 d**2 - 1` for both designs: only the
global phase is unobservable. Its smallest nonzero singular value nevertheless
halves with each qubit for the reduced design (0.35, 0.15, 0.080, 0.039 for
n = 1, ..., 4), whereas it grows for the local-Pauli design (1.28, 2.09, 3.35
for n = 1, 2, 3). With equal per-row Gaussian noise `sigma`, `sigma / s_min`
sets the error in the worst direction. One fixed `sigma` across designs and
qubit counts is therefore not a matched signal-to-noise comparison. Report
the per-row signal with `sigma`, and label fixed-`sigma`, rescaled-`sigma`
and shot-noise runs separately.

The hooks are not tuned for this design. A local noiseless n = 2 check with
all rows illustrates this. L-BFGS on the summed loss recovers the truth to
fidelity 1 (twelve decimals) from twelve of twelve random starts. Both
methods below started from the default initializer, at fidelity 0.25.
FRAMES with the local-Pauli power schedules above (5,000 steps, batch 32)
reduced the loss about sixfold and the TP violation to 0.0013, but ended at
fidelity 0.09. The literal printed adaFGD rule ran 3,000 steps for each
`(eta_scale, tp_weight)` in `{0.03, 0.1, 0.3, 1} x {0.1, 1, 10}`. Four of
the twelve runs diverged, and the rest ended between fidelity 0.125 and
0.29. This diagnoses that uncalibrated printed rule, not FGD generally:
fixed-step FGD on the same smooth objective can recover the target after
calibration. Calibrate both methods on this design before comparing them.

## Validation

```bash
python -m pytest tests/test_qpt_quiroga_sensing.py tests/test_qpt_quiroga_frames.py -q
```

`tests/test_qpt_quiroga_sensing.py` checks the backend against independent
dense references at n = 1 and 2. These references enumerate every row: they
build `D_s` from Kronecker products of the listed probes and the Eq. (18)
elements, and compute Born probabilities from Kraus operators. The tests
cover predictions, mean and summed losses, sampled, full and chunked
gradients, adjoints, central finite differences, the TP map and Jacobian
adjoint, the POVM and probe properties above, local identifiability,
observation-model determinism, noise scale, sampling, input validation, the
scaling CLI and, when JAX is installed, JIT parity with NumPy.

`tests/test_qpt_quiroga_frames.py` checks the objective against the existing
`QPTMeasurementObjective` on `to_dense_qpt_data`, for losses, gradients,
sampled batches and the TP/Moreau terms. It checks that the runner reproduces
`run_qpt_stochastic_frames` with explicit and default schedules, and the JAX
runner when installed, and that both runners share defaults. It also runs the
backend at n = 5, where dense sensing tensors are refused, checks one
printed-rule adaFGD update against the dense formula, compares the matrix-free
and dense spectral norms, and exercises automatic matrix-free selection.

## Limitations

- The FRAMES hook is NumPy on CPU. The JAX operators exist, but no fused GPU
  runner, CLI or archive format has been written for this design.
- Only Haar-unitary rank-one truths are generated. Other truths can be passed
  as `truth_factor`. The coherent, depolarizing and incoherent channels of
  [32], and preparation or measurement errors, are not implemented.
- The POVM constants, adaFGD constants and initialization normalization of
  [32] are unreported. The choices here are documented, but results are not a
  reproduction of the figures in [32].

## Manuscript correction

Section 5.2 of the boosting manuscript (arXiv:2605.25255v1) says that "by
using Pauli bases [32]" there are 4^n inputs, 3^n settings and 2^n outcomes,
for 24^n rows. It also says that the loss functions are implemented
"faithfully as in [32]". The functional form is related, but the normalization
is not: [32] defines `F = 0.5 sum_s r_s**2`, whereas the project runners use
`0.5 mean_s r_s**2`. With the same coefficient on `H`, converting the
implemented objective to sum normalization multiplies the effective TP weight
by the number of active rows. The 24^n sensing design also differs: it is the
traditional local-Pauli design, which [32] replaces with a reduced one.
Suggested replacement:

> We use a local Pauli design: product input states from
> {|0⟩, |1⟩, |+⟩, |+i⟩}^⊗ñ, product Pauli measurement settings from
> {X, Y, Z}^⊗ñ and 2^ñ outcomes per setting, giving 4^ñ·3^ñ·2^ñ = 24^ñ
> sensing pairs (A_s, f_s). We use the same least-squares residual form,
> trace-preservation penalty H, Burer–Monteiro factorization and Haar-unitary
> ground truth as [32], but normalize the measurement loss by the number of
> active rows and do not use its measurement design. Thus the TP coefficient
> must be rescaled before comparing it with a coefficient multiplying [32]'s
> summed loss. [32]
> probes the process with d² = 4^ñ global states (computational-basis states
> and their pairwise real and imaginary superpositions) and measures a single
> 2d-outcome POVM that is informationally complete for pure states, giving
> 2·8^ñ sensing rows (128 for ñ = 2). Our results therefore do not reproduce
> the experiments of [32].

Also replace "implement the loss functions faithfully as in [32]" with
"use the residual and TP-penalty forms of [32], with a mean rather than summed
measurement loss." Qualify "full-measurements" as all 24^ñ
local-Pauli rows; in [32] the term means all 2·8^ñ rows of the reduced
design. If results on the reduced design are added, label them as a second
benchmark that follows [32]. State the POVM constants used, here
a = b = 2/(4d − 1 + √(8d − 7)). Also note that equal ξ does not mean equal
per-row signal-to-noise ratio in the two designs.
