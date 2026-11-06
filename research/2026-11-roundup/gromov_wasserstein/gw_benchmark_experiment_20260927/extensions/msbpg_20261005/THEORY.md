# MSBPG specialization for the GW benchmark

This document specifies the real-arithmetic method and parameter bounds used by
this extension. The reference is Ding, Li, and Toh, *Nonconvex Stochastic Bregman
Proximal Gradient Method with Application to Deep Learning*, JMLR 26 (2025),
Section 3.2, equations (8)–(9), Theorem 24 and Remark 25:
<https://www.jmlr.org/papers/volume26/23-0657/23-0657.pdf>.

The diminishing schedule below specializes Theorem 24 and Remark 25. It does
not claim the fixed-horizon parameter prescription or accuracy-dependent oracle
budget of Corollary 26. Numerical inner solves require the qualification in the
last section.

## Objective and constraints

Distances A and B are finite symmetric matrices of sizes n by n and m by m.
The positive marginals a and b each have total mass one. Define

    U = {P >= 0 : P 1_m = a, P^T 1_n = b},
    F(P) = sum_(i,k,j,l) (A_ik - B_jl)^2 P_ij P_kl,
    R(P) = indicator_U(P).

The smooth objective is the original ambient GW quadratic, including its
marginal-dependent terms. There is no replacement by a restriction to the
transport tangent space, objective rescaling, or altered stochastic estimator.
F is smooth everywhere and nonnegative on U; U is nonempty, compact, and convex.
Its positive product coupling supplies an interior feasible point. The kernel
open domain is the positive orthant, not the transport polytope.

The real-arithmetic construction applies to the distance and marginal values
loaded by the optimizer. In FP32 runs these are FP32 casts of the frozen input
arrays. The bound is not a certificate for an unrounded FP64 input instance,
and FP32 products/reductions introduce further numerical error in the oracle.
The unchanged-objective statement concerns the GW formula and constraints,
without an analytical replacement or additional regularizer; it does not
assert bitwise equivalence of computations at different precisions. Final
FP64 evaluation against the frozen arrays is a separate empirical validation.

## Scaled entropy extension and the theoretical step parameter

Let

    M = min(max_i a_i, max_j b_j) > 0.

Every feasible entry satisfies 0 <= P_ij <= M. For equal uniform marginals on a
square problem, M = 1/n. Define the extended scalar entropy

    psi_M(t) = t log(t) - t,                              0 <= t <= M,
               M log(M) - M + log(M)(t-M)
                   + (t-M)^2/(2M),                      t >= M,
               +infinity,                               t < 0,

where 0 log(0) = 0, and use the **scaled** kernel

    phi(P) = M sum_(i,j) psi_M(P_ij).

The first two derivatives match at M. Its scalar second derivative is M/t for
0 < t <= M and 1 for t >= M. Consequently phi is globally 1-strongly convex in
Frobenius norm, so the paper's strong-convexity constant is **mu = 1**. Its
gradient diverges at the zero boundary, and its quadratic tail is supercoercive.
It satisfies the kernel and Legendre requirements in Definitions 2 and 6.

For feasible Q and strictly positive feasible P,

    D_phi(Q,P) = M sum_(i,j) [Q_ij log(Q_ij/P_ij) - Q_ij + P_ij]
               = M D_KL(Q || P).

Thus scaling and extending the kernel does not change the family of exact KL
proximal updates. It distinguishes two step parameters:

    alpha_k = the step parameter in Theorem 24,
    eta_k   = alpha_k / M = the step multiplying momentum in the KL kernel.

In particular, the proximal objective is

    <v_k,Q> + D_phi(Q,P_k)/alpha_k
      = <v_k,Q> + D_KL(Q || P_k)/eta_k.

The normalized displacement in Theorem 24 uses alpha_k, not eta_k. Both
parameters must be recorded. The extension introduces no lower bound away from
zero on feasible entries and adds no regularization to F + R.

## A computable global Hessian bound

Vectorize P and introduce W only as a mathematical object:

    W_((i,j),(k,l)) = (A_ik - B_jl)^2.

Symmetry of A and B implies W = W^T, and its entries are nonnegative. The
Hessian of F is 2W. Its Euclidean operator norm therefore obeys

    ||Hessian F||_2 <= ||2W||_infinity
                    = 2 max_(i,j) r_ij,

where

    s_i = sum_k A_ik,        u_i = sum_k A_ik^2,
    t_j = sum_l B_jl,        w_j = sum_l B_jl^2,
    r_ij = m u_i + n w_j - 2 s_i t_j.

Only row statistics and a blocked n-by-m maximum are required. No matrix of
size (nm)-by-(nm) is formed. The right side is a global bound for the unchanged
ambient objective. Denote a positive upper bound obtained this way by L.

The implementation computes these statistics in FP64 and records the observed
maximum separately from its numerical inflation. Cancellation in r_ij means
that a guard must scale with the positive terms, rather than r_ij alone. It
computes the following scale in the same row blocks:

    S = max_(i,j) [m computed_u_i + n computed_w_j
                       + abs(2 computed_s_i computed_t_j)].

The chosen conservative guard is

    u_round = eps64/2,
    K = 8(n+m) + 64,
    gamma = K u_round / (1 - K u_round),
    guard = 2 gamma S / (1 - gamma),
    raw_bound = 2 max_(i,j) computed_r_ij,
    L = nextafter(raw_bound + guard, +infinity).

The implementation rejects K u_round >= 1/2. Under the ordinary IEEE FP64
rounding model, standard gamma_q error bounds for sums and products bound
absolute error by gamma times the corresponding sum of absolute terms. K
conservatively covers the row sums, squares, products, scaling, and final
subtraction. Replacing that exact positive sum by its computed value adds the
factor 1/(1-gamma); the factor 2 accounts for Hessian F = 2W. This is a
conservative floating-point safeguard, not a formal directed-rounding interval
certificate. It assumes finite inputs and no exceptional overflow or underflow
in the statistics. The logged guard and input validation make its scope
explicit. If S is zero, the computed distance inputs vanish and the
implementation uses the valid positive bound L = 1 by convention. This zero
case assumes nonzero squares were not lost through underflow; benchmark
distance scales are far from that regime.

Because D_phi(Q,P) >= ||Q-P||_F^2/2, the following are valid constants globally:

    mu    = 1,
    L_F   = L,
    kappa = 2 L^2.

Indeed, the quadratic Taylor remainder is at most
L ||Q-P||_F^2/2 <= L D_phi(Q,P), giving Assumption 1(A2). Also

    ||gradient F(Q) - gradient F(P)||_F^2
        <= L^2 ||Q-P||_F^2 <= 2 L^2 D_phi(Q,P),

which gives Assumption 3. These are sufficient, potentially conservative bounds,
not claims that L is the smallest possible constant.

## Stochastic oracle

Use the same fresh average of 128 independent Rademacher-probe gradient
estimators as in the benchmark. Conditional expectation equals the exact
ambient gradient of F. Finite probe support and continuity on compact U imply
a finite uniform single-probe variance bound sigma^2. Independence therefore
bounds the minibatch variance by sigma^2/128, as required by Assumption 2.

The fresh estimator is unbiased; the moving average need not be. The method
requires one such fresh stochastic gradient per update, and no growing
minibatch or exact-gradient call. Initialization reuses the first update's
stochastic gradient.

## Coupled schedule and indexing

Theorem 24 prescribes alpha_k = c mu beta_(k+1), with

    0 < c <= 1/(2 sqrt(mu kappa)).

For the constants above, define

    c_max = 1/(2 sqrt(2) L),
    c = lambda c_max,                 0 < lambda <= 1,
    beta_(k+1) = theta/sqrt(k+1),      0 < theta <= 1,
    alpha_k = c beta_(k+1),
    eta_k = alpha_k/M.

The only candidate prefactors allowed by this specification are admissible
lambda and theta. The exponent 1/2, coupling, and index shift stay fixed. No
wall-clock-dependent schedule is used. These are explicit sufficient choices
within the theorem; the paper does not select a unique empirical prefactor.

The update at zero-based index k is

    Ghat_k = a fresh q=128 stochastic gradient at P_k,
    v_0 = Ghat_0,
    v_k = (1-beta_k) v_(k-1) + beta_k Ghat_k,     k >= 1,
    P_(k+1) = argmin_(Q in U)
                  {<v_k,Q> + D_KL(Q || P_k)/eta_k}.

In particular, for k >= 1, the moving average uses beta_k = theta/sqrt(k),
whereas the same update's step uses beta_(k+1) = theta/sqrt(k+1).

The initial momentum error is covered by the initial-error term of Theorem 24.
For theta = 1, beta_1 = 1; the momentum-recursion inequality remains valid at
this endpoint. Later coefficients are strictly less than one.

The descent argument in Theorem 24 drops a term having coefficient
1/alpha_k - L_F. Our schedule satisfies

    alpha_k L_F = lambda theta/(2 sqrt(2) sqrt(k+1)) < 1,

so that step restriction is satisfied as well. With fixed minibatch size,
sum alpha_k grows as sqrt(N) and sum alpha_k^2 grows as log(N), giving the
logarithmic-over-square-root bound described in Remark 25. This does not assert
that the squared steps are summable for the exponent 1/2.

## Output statistic and comparison scope

Fix the number N of updates before a run. Independently of the stochastic
optimization draws, choose J in {0,...,N-1} with

    Pr[J=k] = alpha_k / sum_(r=0)^(N-1) alpha_r
            = (k+1)^(-1/2) / sum_(r=0)^(N-1) (r+1)^(-1/2).

Theorem 24 controls the expected squared normalized stochastic displacement

    ||(P_J - P_(J+1))/alpha_J||_F^2.

An independently presampled J is permissible. An alpha-weighted average of
these squared displacements along the run is the conditional expectation over
J and may be reported as an additional diagnostic. Reporting the last iterate's
objective remains useful empirically, but is not the theorem's output
statistic. The statistic also differs from the Bregman forward-backward
envelope used by the published fresh-gradient SMD comparator.

The kernel gradient is not uniformly Lipschitz near zero, so this result alone
must not be promoted to ordinary boundary KKT convergence. Scaling phi changes
the units of the normalized displacement; report the definition together with
the number, and do not compare raw residual magnitudes across differently
scaled kernels as if they were the same stationarity measure.

## Stable evaluation of the same proximal certificate

`stable_prox.py` subclasses the existing certified Sinkhorn solver and changes
only certificate arithmetic. It preserves the same numerical problem

    J(Q) = <v,Q> + KL(Q || exp(logP))/eta,

and the same normalized, unscaled primal-dual-gap tolerance 1e-7. It does not
multiply the objective or tolerance by eta. Sinkhorn iterations, the returned
plan, feasibility checks, and construction of the FP64 feasible primal witness
from the returned (normally FP32) plan are inherited unchanged.

For dual potentials u and w, form the small increment directly:

    delta_ij = -eta v_ij + u_i + w_j,
    reference_ij = exp(logP_ij),
    dual_plan_ij = reference_ij exp(delta_ij).

Do not form delta by subtracting logP from a separately computed large
logarithm. The dual objective is evaluated as

    D(u,w) = [<a,u> + <b,w>
                   - sum_ij reference_ij expm1(delta_ij)] / eta.

This is algebraically the original dual objective, with the mass difference
computed directly rather than by subtracting two nearly equal unit masses.
The certificate independently evaluates J and D, and cross-checks their
difference against the identity

    J(Qwitness) - D(u,w)
       = [KL(Qwitness || dual_plan)
           + <u,Qwitness 1-a> + <w,Qwitness^T 1-b>] / eta.

For nearby entries, use z = log(Qwitness/reference), calculated with
`log1p((Qwitness-reference)/reference)` when appropriate, and the identity

    KL_entry(Qwitness || reference)
        = Qwitness [exp(-z) - 1 + z].

A degree-12 Taylor polynomial evaluates the bracket when |z| <= 0.125;
elsewhere the evaluator uses noncancelling expressions. The omitted Taylor
remainder in that interval is below the adopted arithmetic allowance. The
independent gap KL uses z-delta. Zero witness entries use the usual continuous
KL convention. Thus the numerical method does not hide quantization error by
substituting the FP64 dual candidate for the original primal witness.

The error allowance includes magnitudes of the terms actually summed, errors
in the direct delta calculation, and errors in logarithm/exponential
arithmetic. For a near-zero KL remainder, perturbing z by e changes its value
by at most a first-order term proportional to |Qwitness-reference| e and a
second-order term proportional to reference exp(e) e^2/2. This avoids charging
a unit-mass error divided by eta when the KL function and its first derivative
both vanish. Far-apart and zero entries retain the appropriate absolute error
terms. Dual error includes the sensitivity dual_plan times the error in delta.
The constraint correction includes errors in the computed witness marginals.

Reduction errors use gamma_q = q*u_round/(1-q*u_round), with

    q = max(min(block_rows,n)*m, n, m)
          + ceil(n/block_rows) + world_size + 128.

A serial block reduction bounds the depth of an ordinary parallel reduction
tree, with further allowance for block and rank accumulation, vector sums, and
scalar assembly. This does not assume a particular CUDA reduction tree. The
bound is rejected when q*u_round >= 1/4. Conservative FP64 transcendental-error
allowances are also included; this remains a numerical certificate, not formal
interval arithmetic. The report stores the error components and reduction
factor. A primal-dual discrepancy exceeding their combined allowance fails the
same existing acceptance audit.

Tests compare the defining primal and dual formulas with independent
120-digit Decimal calculations, including eta=1e-10, reference entries near
1e-9, zero witness entries, general dual potentials, and deliberately poor dual
candidates. A separate case verifies that a genuinely inaccurate FP32-derived
witness still fails when its actual normalized gap exceeds 1e-7. Stable
arithmetic removes avoidable cancellation; it does not guarantee that every
small-step FP32 inner solve passes.

## Finite precision and inner solves

The mathematical update above has an exact proximal minimizer. The experiment
uses the benchmark's Sinkhorn-based numerical solver, with marginal and
normalized primal-dual-gap tolerances 1e-7, FP64 logarithmic/certificate
arithmetic, and an FP32 returned coupling. These finite tolerances do not
establish an inexact-solve version of Theorem 24. Report the method as a
numerical implementation of the published MSBPG update with the stated
admissible parameters and numerical tolerances.

Any displacement recovered from retained FP64 log plans is a numerical
statistic for those plans; a statistic from rounded FP32 iterates is a statistic
for the actual stored trajectory. Label the representation used. Neither
representation removes the exact-proximal-solve qualification. Inner solver
failures and violated acceptance tolerances remain validation failures.
Underflow in the returned FP32 plan is explicitly counted; retained finite
FP64 logarithms may represent positive entries that round to zero in that
plan. Such a numerical trajectory is not an exact positive-iterate theorem
certificate, even if its numerical feasibility and gap tests pass.

These runs are new MSBPG runs. Historical momentum-KL runs with a constant KL
step and decaying momentum did not obey this coupling and cannot be relabeled
as theorem-matched MSBPG runs.
