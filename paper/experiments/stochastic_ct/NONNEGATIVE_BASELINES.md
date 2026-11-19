# CT baselines without spectral constraints

This campaign compares the existing SFRAMES results with two methods for

\[
  \min_{U,V\geq0}\;\frac{1}{2Nd}\|A(UV^\top)-y\|_F^2.
\]

Here `A` stacks the selected fan-beam views. The rank, data preprocessing,
training/held-out split, and warm-start factors are unchanged. These baselines
have **no spectral bounds**. The previously used radii are recorded only to
measure whether the new factors leave the old feasible set. Spectral norms do
not enter their updates, stopping rules, losses, or feasibility repair.

SFRAMES still optimizes over its product of spectral balls and smooths the
nonnegativity indicator. The comparison therefore concerns application quality
at a common data-processing budget across different optimization domains; it
does not compare solvers with identical feasible sets. Historical repaired
SFRAMES images remain valid nonnegative candidates, although their repair also
enforces the spectral bounds. New-baseline images receive no such rescaling.

## Alternating nonnegative least squares

Fixing either factor gives a convex nonnegative least-squares subproblem. We
solve it approximately with SciPy's bound-constrained **L-BFGS-B**, then switch
factors. This is inexact alternating NNLS, not a reproduction of Kim and Park's
active-set implementation. No dense tomography matrix or block Hessian is
formed. Loss and analytic gradient evaluations use the existing forward and
adjoint operators.

The implementation uses SciPy **1.18.1**, the version in the CT experiment's
existing environment. Reverse communication exposes every function/gradient
request and accepted iterate. Checkpoints retain the trial point, accepted
factors, limited-memory workspace, task flags, integer ABI, and counters. A
checkpoint cannot be resumed with a different SciPy version or configuration.

Fixed inner settings are ten memory corrections, relative projected-gradient
tolerance `1e-3`, function-decrease tolerance `1e-9`, and at most twenty
line-search steps. The accepted-iteration limit per block is selected from
`{5, 10, 20}` on the two existing development slices at both training-view
counts, with 34,560 views per direction. Selection uses training loss relative
to initialization, with equal slice and view-count weight; ties choose the
smaller iteration limit. All twelve development runs are retained. The chosen
setting is frozen before any of the evaluation runs.

Every accepted inner step is recorded. Block terminations distinguish the
projected-gradient criterion, relative function decrease, iteration limit, and
failed line search. The latter retains the last accepted point rather than
publishing an unaccepted trial as an iterate. Numerical failures are retained
and halt the campaign; evaluation settings are not changed after seeing results.

References: [alternating NNLS for NMF](https://faculty.cc.gatech.edu/~hpark/papers/simax-nmf.pdf)
and [SciPy's L-BFGS-B options](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html).

## Multiplicative updates for the signed measurements

The measured log-transmission data include negative entries. Neither those
data nor their adjoint are clipped. For fixed `V`, define

\[
 b_U=\frac{A^*y}{Nd}V,\qquad
 H_U[U]=\frac{A^*A(UV^\top)}{Nd}V.
\]

The alternating update is

\[
 U\leftarrow U\odot\frac{(b_U)_+}{H_U[U]+(b_U)_-},
\]

followed by the analogous update for `V` using the updated `U`. Products and
quotients here are entrywise. This is an operator-aware signed-right-hand-side
extension of the [classical multiplicative-update construction](https://www.cs.cornell.edu/courses/cs6241/2020sp/readings/Lee-Seung-2001-NMF.pdf),
not the ordinary observed-matrix NMF formula applied to a reconstructed image.

For a fixed factor, the Hessian is entrywise nonnegative. At a positive vector
`x`, the diagonal majorizer is `diag((H x + b_minus) / x)`; subtracting `H`
gives a positive semidefinite matrix. Minimizing this quadratic majorizer gives
the stated update. This explains why splitting the signed linear term preserves
the original least-squares objective.

The shared initialization is strictly positive. There is no epsilon denominator,
factor floor, balancing transform, regularizer, or step-size tuning. A zero
numerator with a positive denominator sets an entry to zero; a `0/0` entry is
left unchanged. Existing zero entries stay zero (the usual MU zero-locking
limitation). A positive numerator with a zero denominator raises an error.

## Work, observations, and cohort

The evaluation contains 14 slices from six acquisition groups and both
`N=360` and `N=1080`: 28 runs per deterministic method, **56 physical runs** in
total. There are no duplicate seed repetitions for these deterministic methods.
Evaluation calibration seed 201 and development calibration seed 110 identify
the already-frozen shared factors; neither method resamples a new initialization.

Each direction has a separate cap of **69,120 processed views**, matching the
existing SFRAMES budget. All new-solver operator work counts, including rejected
NNLS line-search trials and the one-time MU adjoint `A* y`. MU subsequently uses
one forward/adjoint pair per factor update, so its directional totals differ by
the initial `N` adjoint views. The displayed work coordinate is the larger of
the cumulative forward and adjoint totals. Both must fit their caps.

One oracle call means a completed forward/adjoint pair; the standalone MU
initial adjoint also counts as one call. Literal forward and adjoint call counts
are recorded separately. For the previously published methods, their gradient
work is reused and curvature-estimation costs remain separate, as in the prior
comparison. Shared data preparation and loss evaluations used only for reporting
are outside optimizer work. Optimizer time is not reported.

Record initialization, every accepted block update, and the terminal charged-work
state. An unsuccessful trial can spend views without changing the accepted
factors; the terminal row then repeats the accepted iteration number rather than
inventing a new update. Completed blocks and alternating sweeps are also stored.
All plotted work trajectories use native observations, never interpolated losses.
The report retains its two figures, with means and one descriptive standard
deviation, equal acquisition-group weights, and both new deterministic methods.

## Execution and preservation

`nonnegative_campaign.py` prepares and verifies the plans and inputs, performs
development selection, runs the evaluation, validates every result, and invokes
the report publisher only after successful completion. It runs from a frozen
copy of the Git sources in a separate runtime directory. `nonnegative_run.py`
records each run; `nonnegative_solvers.py` implements the two methods.
`validate_nonnegative_real.py` performs bounded checks on the real CT projector
before launch. Its results are smoke tests, not evaluation results.

The exact runtime paths, hashes, selected parameters, CPU assignments, and
completion/publication status are stored in the campaign protocol and manifests.
Raw data, calibration arrays, environments, and optimizer checkpoints remain
outside Git. The current paper is replaced only after all 56 runs validate;
completed historical campaigns and their numerical sources remain unchanged.
