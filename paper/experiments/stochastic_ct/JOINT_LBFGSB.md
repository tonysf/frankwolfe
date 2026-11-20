# Joint L-BFGS-B for nonnegative CT factorization

This campaign solves the same nonnegative factorized least-squares problem as
the alternating NNLS and multiplicative-update baselines:

\[
\min_{U,V\geq0}\frac{1}{2Nd}\|A(UV^\top)-y\|_F^2.
\]

L-BFGS-B optimizes the concatenated factor vector jointly. Its limited-memory
curvature approximation persists across accepted iterations. There is no block
alternation, spectral bound, factor balancing, positive floor, or additional
regularizer. Signed measurements are preserved. The existing calibrated factors
are used byte-for-byte.

For a trial pair, form the residual `r = A(U V.T) - y` and backproject
`Z = A* r / (N d)`. The analytic gradient is the concatenation of `Z V` and
`Z.T U`. Both factors therefore share one forward/adjoint pair. An accepted
iteration means one accepted joint L-BFGS-B step, regardless of the number of
line-search trials needed to obtain it.

## Parameters and development selection

Use SciPy 1.18.1's L-BFGS-B implementation with lower bounds zero. The maximum
number of line-search steps is 20 and the relative function-decrease tolerance
is `1e-9`. The projected-gradient stopping threshold is `1e-3` times the norm at
initialization, using SciPy's lower-bound gradient mapping
`max(abs(min(x, gradient)))`. Function-decrease termination is reported
separately from satisfying this projected-gradient criterion.

Choose the memory size from `{5, 10, 20}` on the two existing development
slices, at both 360 and 1,080 training views, with 34,560 processed views per
direction. These twelve development runs use the already-frozen development
initializations. For each memory size, take the median endpoint-to-initial raw
loss ratio across development slices at each view count, then the median of
the two view-count scores. Choose the smallest score, breaking an exact tie
by the smaller memory size. Freeze the result before evaluation.

## Evaluation and work accounting

Run one deterministic trajectory for each of the 14 evaluation slices and
two training-view counts: **28 evaluation runs**. The shared initialization,
six specimen groups, rank 64, image grid 512, data preprocessing, and held-out
views match the existing comparison. Development calibration seed 110 and
evaluation calibration seed 201 refer to the existing factor arrays, not new
random initializations.

Each evaluation run has separate forward and adjoint caps of **69,120 views**.
Every objective/gradient trial counts, including rejected line-search trials.
The fresh initial function/gradient evaluation also counts. The work coordinate
is the larger directional total; an oracle call is one completed
forward/adjoint pair. Diagnostic losses, recorded after every accepted joint
update, are separate from optimizer work under the existing protocol.
Optimizer time is not compared.

Retain the last accepted factors when a budget prevents the next trial. A
terminal observation records any charged work without inventing an accepted
iteration. Raw and evaluated losses are equal because the factors are already
nonnegative; no spectral rescaling is applied. Historical spectral-radius ratios
are diagnostic only. The existing SFRAMES repair convention remains documented
in the report, so its evaluated loss uses a valid nonnegative candidate while
also enforcing its stronger spectral constraints.

The report reuses all 504 earlier physical trajectories and adds the 28 new
ones. Its two figures show mean trajectories with one descriptive standard
deviation, and its existing endpoint table includes the new method. Native
iteration costs differ between joint L-BFGS-B, alternating NNLS, and SFRAMES;
processed views are the primary work comparison.

## Execution and preservation

Run from separately frozen sources with the existing CT numerical environment,
four single-thread workers on CPUs 0, 2, 4, and 6, and persistent user-service
execution. Preserve the L-BFGS-B reverse-communication arrays, pending trial,
accepted factors, and counters in checkpoints. Checkpoint compatibility pins
the source bytes, inputs, SciPy version, and integer ABI.

Numerical errors or abnormal line-search termination retain their evidence and
halt the campaign for review. They are not silently restarted or excluded from
selection. Successful function-decrease termination is not called KKT
convergence. A completed comparison is published only after validating every
development and evaluation result.

The `ct_joint_lbfgsb` result archive stores the final protocol, selection,
metadata, native traces, and report evidence. Raw data, calibration arrays,
optimizer checkpoints, and environments stay outside Git. Earlier result
archives and numerical sources remain intact.

Reference: [SciPy L-BFGS-B options](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html).
