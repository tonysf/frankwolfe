# Protocol: stochastic Moreau-smoothed Frank–Wolfe for fixed GW optimization

Prepared 25 September 2026. This is a proposed experiment, not a report of solver
performance. Biological interpretation is outside its scope.

## 1. Experimental question and fixed problem

The question is whether a stochastic, single-loop optimizer with a rowwise LMO
and asymptotic enforcement of the other marginal produces good **feasible outputs**
more cheaply than optimization methods with full transport projections or LMOs.
A second, distinct question is whether its measured stationarity and feasibility
statistics behave as predicted by the authors' theorem.

Use only unregularized, balanced, squared-loss GW in the initial study:

\[
f(P)=\sum_{i,k=1}^{n}\sum_{j,l=1}^{m}
(A_{ik}-B_{jl})^2 P_{ij}P_{kl},\qquad
\min_{P\in\mathcal U(a,b)} f(P),
\]
\[
\mathcal U(a,b)=\{P\ge0:P1_m=a,\ P^T1_n=b\}.
\]

No factor of 1/2, square root of the objective, entropy penalty, fused feature cost,
low-rank restriction, or changing geometry is included. Begin with n=m and uniform
positive marginals a=b=1/n. This also permits exact assignment LMOs.

For the proposed algorithm use
\[
C=\{P\ge0:P1_m=a\},\quad T(P)=P^T1_n,\quad D=\{b\}.
\]
Its smoothed objective is
\[
f_\mu(P)=f(P)+\frac{\|P^T1_n-b\|_2^2}{2\mu}.
\]
C is compact and convex; D is closed and convex; ab^T proves joint feasibility.
Positive b also lies in the relative interior of T(C), the probability simplex.
The objective is a polynomial with constant Hessian. On this matrix space, the
operator norm of T is sqrt(n), so the penalty gradient has Lipschitz constant n/mu.
This matters when interpreting dimension-dependent constants.

The problem really is nonconvex. For A=B=[[0,1],[1,0]], and
P(t)=[[t,1/2-t],[1/2-t,t]], the objective is 4t-8t^2 on 0<=t<=1/2.

Do not include partial transport in the first experiment. If total mass is left
in the smoothed constraint, the homogeneous quadratic can reward deficient mass.
A partial-transport extension should preserve total mass in C and be designed and
benchmarked separately.

## 2. Freeze the environment and mathematical instances

Use one A100 per run and the other A100s for independent runs, not distributed
training initially. Record card model/VRAM, driver, CUDA, Python, package versions,
CPU model, thread counts, precision, deterministic settings and source commit.

Use float64 on CPU for correctness tests. Begin GPU experiments in float32 without
AMP and with one explicitly documented matrix-multiplication precision policy for
all methods. Validate representative checkpoints in float64. A small negative GW
value from cancellation must be distinguished from a substantially incorrect loss;
do not silently clamp it in stored results.

Install a CUDA-compatible PyTorch build for the host, then the other dependencies.
Do not blindly replace an existing working CUDA build with a CPU-only build.
`requirements.txt` is a bootstrap list, not a pinned publication environment.
After tests succeed, run `python -m pip freeze > environment.lock.txt` and archive it.

Save each mathematical instance once as an NPZ containing A,B,a,b, plus a JSON
manifest. Hash A and B. Save subset indices, distance choice, scaling rule and
random seeds. All solvers must load these same arrays. Caching A**2 and B**2 is
allowed for every method; count the associated time and memory consistently.

## 3. Build the controlled synthetic family

Draw the upper-triangular entries of A independently from Uniform[0.5,1], reflect
across the diagonal, and set the diagonal to zero. This is a metric: every distance
is at most 1 and the sum of two nonzero distances is at least 1.

Choose a random permutation p and set B=A[p,p]. If `B=A[np.ix_(p,p)]`, source index
i matches target index `argsort(p)[i]`. The scaled permutation Pstar with mass 1/n
on those positions has f(Pstar)=0. Since f is nonnegative on nonnegative matrices,
the global optimum is known. Keep this permutation out of optimizer code.

For a noisy family, draw another independent metric E and set
B_noise=(1-noise)*B+noise*E. Use noise=0 and noise=0.1. A convex combination of these
metrics is a metric. Do not claim a known zero optimum when noise>0.

Use sizes 128,512,2048,8192 after toy tests at sizes 4–16. Use development problem
seeds 1000–1002, and held-out problem seeds 2000–2004. Generate each size as a frozen
instance; do not demand nested synthetic samples. Additional seeds can be added
only under a prespecified replication rule, not because a particular result is bad.

Commands:

```bash
python make_instances.py synthetic --n 128 --seed 2000 --noise 0 --out data/metric128_s2000.npz
python make_instances.py synthetic --n 512 --seed 2000 --noise 0.1 --out data/metric512_noisy_s2000.npz
```

The generator stores float32 distances as the canonical problem data; float64
validation promotes those stored numbers rather than regenerating a subtly different
problem. It stores the planted permutation only for diagnostic tests.

## 4. Build one real-data family, without biological evaluation

Use the processed moscot BMMC files. The maintained loader documents 6,224 rows per
view and points to these files [1]. The tutorial documents the RNA PCA and ATAC LSI
representations, and rowwise L2 normalization of the latter [2].

```bash
mkdir -p raw data
curl -L --fail --retry 3 https://figshare.com/ndownloader/files/40195114 -o raw/rna.h5ad
curl -L --fail --retry 3 https://figshare.com/ndownloader/files/41013551 -o raw/atac.h5ad
sha256sum raw/rna.h5ad raw/atac.h5ad > raw/SHA256SUMS

python make_instances.py bmmc --rna raw/rna.h5ad --atac raw/atac.h5ad --n 1000 --out data/bmmc1000.npz
python make_instances.py bmmc --rna raw/rna.h5ad --atac raw/atac.h5ad --n 3000 --out data/bmmc3000.npz
python make_instances.py bmmc --rna raw/rna.h5ad --atac raw/atac.h5ad --n 6224 --out data/bmmc6224.npz
```

The specified preprocessing is:

- RNA coordinates: `rna.obsm['GEX_X_pca']`, with no extra normalization.
- ATAC coordinates: `atac.obsm['ATAC_lsi_red']`, each row divided by its L2 norm.
- Independently fixed permutations of each view, using seeds 2026 and 2027; take
  prefixes to form nested subsets within each view. No labels or pair identities
  enter the preparation. These subsets are not claimed to be fully paired.
- A and B: ordinary Euclidean pairwise distances, **not squared Euclidean distances**.
- Scale each distance matrix by its own maximum. Its entries then lie in [0,1].
- Uniform marginals; no biology-derived weights or downstream metrics.

This is the proposed optimization instance, not a claimed reproduction of moscot's
entire preprocessing/solver pipeline. Freeze the output before tuning. A zero-norm
ATAC row, missing representation key or nonfinite input should trigger an explicit
preprocessing decision, not a silent fallback.

Use the 1,000-point instance for development, the 3,000-point instance for the first
larger pilot, and the full instance only after successful validation. Its coupling
has 38,738,176 variables. Do not fetch the full 69k-cell collection at this stage.

## 5. Implement and validate the objective at infeasible iterates

Let r=P1 and c=P^T1. For symmetric A,B,
\[
f(P)=r^T A^{\odot2}r+c^T B^{\odot2}c-2\langle APB,P\rangle,
\]
\[
\nabla f(P)=2[(A^{\odot2}r)1^T+1(B^{\odot2}c)^T-2APB].
\]
Use the **actual** marginals r,c. Even though the row condition is invariant for
your method, its column condition is not. Do not use a library's feasible-only
loss shortcut to score raw iterates. The original indicator-valued objective is
infinite at infeasible points; call f(P) the 'raw smooth loss'.

Validate before optimization:

1. On a 4-by-3 instance, compare the matrix loss and gradient with the direct
   four-index formula and autodifferentiation. Use nonnegative row-feasible P
   whose column marginal is genuinely wrong.
2. Verify the two-point negative-curvature example and the planted f(Pstar)=0.
3. Verify row LMO value, nonnegativity and preservation of row masses after an FW step.
4. Verify stochastic-oracle expectations by exact enumeration on tiny instances.
5. Verify output rounding, feasible fixed points and the stated perturbation bound.
6. Verify KL projection and the small Euclidean projection against their tolerances.

`python -m pytest -q` runs 12 deterministic tests for the provided kernels. Add
checks for the exact estimator and update in your own implementation.

## 6. Choose a stochastic oracle explicitly

The optimizer must sample gradients of the frozen full problem, not solve random
smaller GW problems. Two reference oracles are provided.

### 6.1 Transparent scalar finite-sum oracle

For K^{kl}_{ij}=(A_ik-B_jl)^2, let (k,l) be uniform over nm anchors and define
\[
\phi_{kl}(P)=nm P_{kl}\langle K^{kl},P\rangle.
\]
Then E phi=f and
\[
\nabla\phi_{kl}(P)=nm[P_{kl}K^{kl}+\langle K^{kl},P\rangle E_{kl}].
\]
Batch averages are unbiased. Do not omit the E_kl derivative term and still call
this the gradient of phi_kl. `anchor_gradient` implements the full expression.

This estimator is mainly a correctness reference. Uniform entry sampling can
be very noisy near concentrated plans. A simple Python loop over anchors should
not be used as the final high-performance implementation.

### 6.2 Optional fixed-distribution trace-sketch oracle

A candidate that avoids a rare-nonzero-entry issue is to draw independent
Rademacher vectors z in {-1,+1}^m. Because E[zz^T]=I, define
\[
\psi_z(P)=r^TA^{\odot2}r+c^TB^{\odot2}c
-2(Pz)^T A(PBz).
\]
Then E psi_z=f, and
\[
\nabla\psi_z(P)=2[(A^{\odot2}r)1^T+1(B^{\odot2}c)^T]
-2[(APBz)z^T+(APz)(Bz)^T].
\]
Every sample is a smooth quadratic under a fixed finite-support distribution.
Thus the usual unbiasedness and finite-variance-on-C conditions hold. Check any
additional assumptions in your theorem, including individual component conditions
and sample reuse rules. Sampled losses need not be nonnegative.

For batch matrix Z with q unscaled +/-1 columns, average the cross terms by q.
Compute BZ, PZ, P(BZ), then the two multiplications by A. **Do not form APB.**
The dense cost is O(q(n^2+m^2+nm)+n^2+m^2), compared with
O(n^2m+nm^2) for a full gradient. This is only an operation-count comparison;
benchmark the kernels because full matrix products can use hardware efficiently.

This oracle is derived here for the protocol; it is not attributed to the
published Sampled GW algorithm. It is a Monte Carlo identity for the same fixed
objective, not optimization of a frozen low-rank approximation. Draw fresh probes
according to your theorem; never freeze one sketch for the entire run and pretend
the objective is unchanged.

Before choosing the default oracle, measure error and wall time at three kinds of
P: ab^T, a randomized feasible plan, and a concentrated feasible plan. For each
batch size 8,32,128, compare 100 independent oracle evaluations with the true
gradient on n=128 or 512. Log mean error, variance, time, and FW decision error:
\[
\langle G,S_{\widehat G}\rangle-\min_{S\in C}\langle G,S\rangle.
\]
This decision error is useful because row-constant gradient components do not
affect a rowwise LMO. Report absolute as well as relative estimator error.

Select an oracle on development instances, then freeze it. Use precisely that
oracle in an oracle-matched stochastic KL-prox baseline. Published Sampled GW
keeps its own native estimator and sample-count meaning [4].

## 7. Connect your method without changing its theorem

For ordinary exponential momentum the structural update is
\[
d_k=(1-\rho_k)d_{k-1}+\rho_k\widehat{\nabla f}(P_k),
\quad H_k=d_k+\mu_k^{-1}1(c_k-b)^T,
\]
\[
j_i\in\arg\min_j(H_k)_{ij},\qquad
(P_{k+1})_{ij}=(1-\gamma_k)(P_k)_{ij}
+\gamma_k a_i\mathbf1\{j=j_i\}.
\]
Only the smooth gradient belongs in the momentum estimator. The current exact
smoothing gradient is added afterward. A dense atom S need not be allocated;
scale P and scatter-add one entry per row. Tie breaking must be deterministic
and documented.

This skeleton is **not** a substitute for your actual estimator. If your theorem
uses variance reduction, extra gradients, coupled samples, or a different momentum
initialization, implement those exact details and count their oracle calls.

Prepare a `theory_config` containing the theorem identifier, schedule formulas,
exponents, constants, offset/indexing convention, batch schedule, momentum
initialization, and proved output statistic. Copy these from your manuscript.
No numerical exponents are prescribed here because your stochastic theorem has
not been supplied. Do not copy deterministic FRAMES rates into a stochastic run.

In a separate practical-tuning track, vary allowed constants and batch sizes, not
unannounced theory assumptions. To make initial smoothing choices intelligible
across sizes, report both mu_0 and m*mu_0. With uniform b, m*||c-b||_2^2 is a
relative marginal-discrepancy scale. Development candidates m*mu_0 in {0.1,1,10}
are reasonable search points, **not a theorem or a claim of good conditioning**.
Keep T unscaled and record the actual mu_0 used. Also remember ||T||^2=n when
comparing smoothness constants.

Never repair the coupling between main iterations, silently truncate its support,
rescale its columns, or replace the exact penalty gradient by a sampled version.
Any of those creates a different algorithm and belongs in a named ablation.

## 8. Implement optimization baselines on the same problem

### A. Full-gradient KL/Bregman proximal gradient

Use
\[
P_{k+1}\approx\arg\min_{P\in\mathcal U(a,b)}
\langle\nabla f(P_k),P\rangle+\eta_k^{-1}KL(P\Vert P_k).
\]
Equivalently scale the log-kernel log(P_k)-eta_k*grad to the target marginals.
The proximal KL term does not add entropy to the target objective. Retain log-state
to avoid permanently deleting support through exponentiation underflow.

The provided `kl_prox` is an inspectable log-domain reference. Begin with eta in
{0.1,1,10,100} and actual summed marginal-L1 tolerances in {1e-4,1e-6,1e-8} during
development. Use a cap on inner iterations and log when it is reached. Do not
assume the requested tolerance was attained. Underflow, NaNs and support loss
must be reported.

POT's `entropic_gromov_wasserstein(..., solver='PPA')` targets the unregularized
problem; the default `solver='PGD'` targets an entropy-regularized problem [3].
Here the proximal weight `epsilon` corresponds to 1/eta. For a CPU reference:

```python
P = ot.gromov.entropic_gromov_wasserstein(
    A, B, a, b, loss_fun='square_loss', symmetric=True,
    G0=P0, solver='PPA', epsilon=0.1, max_iter=500,
    tol=1e-12, numItermax=1000, stopThr=1e-9,
)
```

Those are smoke-test settings, not chosen winners. POT's `tol`, `stopThr` and your
externally measured marginal-L1 residual are different quantities. Check the
installed signature and archive its version. Native logs may assume prescribed
marginals; use the independent evaluator for every returned P.

### B. Oracle-matched stochastic KL-prox

Replace the exact gradient in A by precisely your selected stochastic oracle.
Use the same family of samples and comparable seed handling. Initially omit
momentum; add a separately named momentum-matched variant to isolate constraint
handling from momentum. Tune a constant or diminishing eta schedule on development
instances. A chosen stochastic proximal schedule is not automatically covered by
the theorem for your own method.

### C. Published Sampled GW

Use `ot.gromov.sampled_gromov_wasserstein` [4], with square loss and positive
proximal epsilon. Start with batches 8,32,128 and epsilon 0.01,0.1,1 as development
candidates. Its native code normalizes the sampled cost and uses its own conditional
sampling; do not equate epsilon or a 'sample' numerically with those of B.

```python
P = ot.gromov.sampled_gromov_wasserstein(
    A, B, a, b, loss_fun=lambda x, y: (x-y)**2,
    nb_samples_grad=32, epsilon=0.1, max_iter=500,
    random_state=100, log=False,
)
```

The documented implementation starts at ab^T and has no G0 argument. Thus the
native comparison must use product initialization for the other methods too.
For randomized-start experiments, instrument a clearly identified fork or leave
this native solver out of that particular comparison. It also lacks a generic
per-iteration callback; add read-only logging in an archived copy rather than
resetting the solver repeatedly to fake a single trajectory.

Do not compare an unoptimized CPU Python sampled loop against your GPU code and
call the ratio an algorithmic speedup. Use a CPU reference panel and a same-device,
same-backend matched comparison; label any faithful port and verify it.

### D. Standard FW over the full transport polytope

At uniform square marginals, its LMO is exactly a linear assignment problem:
\[
\min_{\pi}\frac1n\sum_i G_{i,\pi(i)}.
\]
Use `scipy.optimize.linear_sum_assignment` as an exact CPU reference [5], or POT's
unregularized `gromov_wasserstein`. Count gradient transfers to/from CPU when using
SciPy alongside GPU gradients. This is not a global GW optimizer.

Use exact line search for the quadratic. With D=S-P, let
ell=<G,D> and q=f(S)-f(P)-ell. Minimize ell*gamma+q*gamma^2 on [0,1]; if q>0, clip
-ell/(2q); otherwise compare endpoints. Since S is a scaled permutation,
f(S)=n^{-2}||A-B[pi,pi]||_F^2, and f(P)=<G,P>/2. This avoids another dense GW
matrix product solely for line search. The reference driver uses this shortcut.

### E. Euclidean projected gradient and smoothing ablations

For small instances include P_next=Proj_U(P-eta*G). Dykstra over row and column
simplices is a transparent reference. **Feasibility alone does not prove that a
point is close to the Euclidean projection.** The included routine uses a
primal-dual projection gap with a feasible candidate, not just marginal residuals.
Reset Dykstra corrections for each new projection unless a mathematically valid
warm-start scheme is implemented.

For a more research-oriented Euclidean comparator, examine Liang and Yang's
May 2026 inexact-PG paper [6]. Its particular dual construction makes a feasibility
residual a valid inexactness test; do not transfer that claim to an arbitrary
projection procedure. The provided Dykstra solver is not a reproduction of iPG.

Also compare your method with the same smoothing/momentum scheme but a row-simplex
projection instead of the FW step. This tests whether the row LMO itself helps.
Use a separately tuned step size and label the ablation clearly.

BAPG is an optional second-stage inexact-feasibility comparator [7], not a
replacement for A–D. Record whether marginal-dependent terms are retained, and
always evaluate the full original quadratic externally. Its relaxation should not
be presented as an exact solution guarantee for your original problem.

## 9. Output rounding and feasible evaluation

Keep raw iterates untouched. At evaluation checkpoints repair a copy using the
transport-rounding construction of Altschuler, Weed and Rigollet [8]. For your
row-feasible P, first scale down excess columns:
\[
c=P^T1,\ d_j=\min(1,b_j/c_j),\ Q=P\operatorname{diag}(d).
\]
Take d_j=1 if c_j=0. Set r=a-Q1, s=b-Q^T1 and delta=sum(r)=sum(s). Return
\[
\widehat P=Q+rs^T/\delta
\]
if delta>0, and Q otherwise. This costs O(nm); it is not Euclidean projection.
For nonnegative baseline outputs that violate both marginals, first scale down
excess rows, then columns, then fill deficits. `round_transport` implements this
general version. Reject genuinely negative or nonfinite inputs rather than silently
pretending they were feasible outputs.

Verify numerical marginals after repair. For exactly row-feasible inputs,
\[
\|\widehat P-P\|_1\le \|P^T1-b\|_1=:v_1.
\]
If kappa=max(A_ik-B_jl)^2, then
\[
|f(\widehat P)-f(P)|\le2\kappa v_1.
\]
Here norms on matrices are entrywise L1. These are elementary specializations for
this formulation; they are not performance claims. With the normalization above,
kappa<=1. In floating point, record actual displacement and residuals too.

Use exactly this output opportunity for every inexact method. Do not give your
method a hidden extra optimization phase. A polished variant would be a separate
algorithm and its entire cost would count.

## 10. Metrics and logs

For every run, save instance hash, solver/version/config, initialization seed,
stochastic seed, dtype, device, outer iteration, smoothing and step parameters,
probe/anchor counts, exact-gradient calls, LMO/projection calls, inner iterations,
inner stopping failures, and peak memory.

At evaluation checkpoints record:

- Raw smooth loss f(P), never presented as a feasible objective.
- Row-L1 residual; target-marginal L1 and L2 residuals; total mass error; min entry.
- True smoothed FW gap, when a full gradient is affordable:
  G_mu,C(P)=<H,P>-sum_i a_i min_j H_ij, with H=grad f(P)+1(c-b)^T/mu.
- Momentum-gradient error at the same point; a gap formed with the noisy estimator
  is only a diagnostic, not the exact theoretical certificate.
- Rounded loss f(Phat), rounded feasibility, and ||Phat-P||_1.
- On n<=512 initially, the original FW gap
  G_U(Phat)=<grad f(Phat),Phat-S>, using an exact assignment LMO.

For this formulation one can derive the additional check
\[
G_U(\widehat P)\le G_{\mu,C}(P)+6\kappa v_1.
\]
It follows from the nonnegative penalty inner product against any feasible S,
||grad f(P)-grad f(Q)||_infty<=2*kappa*||P-Q||_1, and the rounding bound. This
relationship concerns true gradients, not raw momentum estimates.

Report both L1 and L2 feasibility. Across 10,000 target points, deviations of size
1e-5 can give L2=1e-3 but L1=0.1. Rounding then reallocates 5% of total mass.
A small L2 number alone can be misleading.

## 11. Timing and checkpoint rules

Track separately: one-time preparation/cache time, solver time, output-repair time,
exact diagnostic time, serialization time and total elapsed time. GPU operations
are asynchronous, so synchronize around timing regions or use appropriate CUDA
events [9]. Include transfers and all inner solves performed by the algorithm.

For theory plots on small instances, evaluate every iterate when your theorem
requires an average over every iterate. For an expensive large run, use sparse
checkpoints and do not call an average over those checkpoints the theorem's
full-trajectory average.

For practical comparisons, use a common elapsed-solver-time checkpoint grid, for
example 1,3,10,30,100,300 seconds in the pilot. Record the first completed iterate
at or after a target time and its actual time; do not interpolate an improvement
backward across a slow iteration. A missing time window is a missing checkpoint.

The main time-to-output axis is solver time plus the cost of producing that
checkpoint's feasible rounded output. If a running best output is selected using
objective evaluations, the selection evaluations become algorithmic cost and must
be counted. Do not repeatedly repair/evaluate outputs for free to select a winner.
Also report total instrumented time including all diagnostics.

Use the same checkpoint opportunities for incumbent plots. The primary plot should
show current feasible output quality; best-observed quality is supplementary.
Evaluation random samples and solver samples must use separate random streams.

Do not evaluate a full objective after every stochastic update at n=8192. For
large instances use infrequent exact evaluations. If an independent MC evaluation
is necessary, show uncertainty and perform exact final evaluations where feasible;
do not represent a noisy estimate as an exact objective or exact FW gap.

## 12. Initialization, tuning and replications

The native all-method panel uses P0=ab^T. A secondary panel uses strictly positive,
uninformative feasible starts
P0=0.9*ab^T+0.1*R/n, where R is an independent random permutation. Do not use the
planted permutation or labels. Match initializations across the methods included
in that panel. Store separate initialization and stochastic seeds in final runs.

Use development data seeds 1000–1002 and stochastic seeds 0–2 for tuning. Limit
each tunable method to the same prespecified number of candidate configurations,
for example 12, selected from its own reasonable log-spaced parameter ranges.
FW with exact line search need not consume artificial tuning trials.

Select practical settings using feasible rounded objective at fixed development
budgets, with residuals and failures reported. Keep a dedicated theory-valid
configuration even if a heuristic setting performs better. Do not optimize a
parameter on a final instance and call that instance held out.

For final synthetic results use problem seeds 2000–2004 and stochastic seeds
100–104. For fixed BMMC matrices use stochastic/initialization replications but do
not pretend these are independent datasets. Theory-rate plots should use more
stochastic repeats where affordable, for example 10. A deterministic solver does
not obtain an artificial error bar by rerunning the identical initialization.

Report medians and interquartile ranges for runtime/objective plots, all runs or
failure counts, and the statistic actually covered by the theorem for theory
plots. Empirical slopes over a finite range are consistency checks, not a proof or
a claim of a tight asymptotic rate. Choose the fitted range before inspecting which
range makes the result attractive.

## 13. Staged execution

### Stage A: correctness

Run the 12 tests in float64. Smoke-test every baseline on n=16. Confirm the loss
and factor conventions against POT on small feasible matrices. Connect your own
method and test its row invariant over repeated updates. No performance conclusion
is drawn here.

### Stage B: oracle and constraint diagnostics

At n=128 and 512, measure oracle error/time at diffuse and concentrated couplings.
Run your method with exact gradients as an ablation, then with stochastic gradients.
Use the exact theorem schedules and output statistics. Plot v1,v2,true smoothed
gap and estimator error. Check whether feasibility stagnation is a schedule issue,
a floating-point effect or the expected finite-time trade-off.

### Stage C: the first real comparison

Use synthetic n=512 and BMMC n=1000. Compare your method, full-gradient KL-prox,
oracle-matched stochastic KL-prox, native Sampled GW and standard full-polytope FW.
Include small Euclidean PG and the row-projection ablation where practical.
Run matched budgets such as 30 and 300 solver seconds with actual times recorded.
These numbers are experiment budgets, not runtime predictions.

### Stage D: scale only after Stage C is informative

Use synthetic n=2048 and 8192; BMMC n=3000 and 6224. Start on one A100, using the
other devices for independent runs. Add larger dimensions only after profiling
shows a sampling benefit and memory headroom. There is no need to start on H100s.

At n=8192, one float32 dense array is about 0.268 GB; at n=6224 it is about 0.155 GB.
Multiply by the actual simultaneous arrays, including cost squares, momentum,
log-kernels and scratch space. Do not claim a memory benefit merely because the
LMO atom is sparse; the maintained state and estimator can remain dense.

## 14. Figures and decision rules

The initial report should contain separate plots for feasible rounded loss vs
time; raw marginal violation vs time; true smoothed gap vs iteration/oracle work;
and original feasible FW gap vs time on smaller instances. Put time breakdowns,
peak memory and failure counts in tables. Do not combine changing-mu penalized
objectives on one 'same objective' convergence curve.

For noise=0 synthetic cases, fstar=0 is known. Else use the best independently
evaluated feasible value only as a reference f_ref, never as a certified optimum.
A negative f-f_ref from a new run means the reference must be updated, not clipped.

Proceed to larger runs only when the kernels are correct, the oracle gives a
useful cost/error trade-off, and at least one practical configuration gives
competitive rounded outputs. If exact-gradient smoothing works but stochastic
smoothing fails, investigate estimator variance or the momentum scheme. If raw
loss improves but rounded loss does not, constraint handling is the limiting factor.
If exact full gradients are already faster at equal quality, do not force a
stochastic-necessity story for that size.

An appropriate positive finding is improved time to a good feasible stationary
output, alongside behavior consistent with the proved guarantees. No biological
finding is required. No outcome is promised by this protocol.

## 15. Delivery scope and reproducibility

The starter includes no implementation of your unpublished method, no downloaded
BMMC files and no claimed GPU benchmark results. Twelve kernel tests and tiny CPU
smoke runs for the four provided reference methods passed before delivery. Record
your own GPU, data and POT validation before scaling. The reference driver is not
a production-grade timing harness and deliberately performs expensive exact
checks often; the protocol above governs publishable comparisons.

## Sources

[1] Moscot dataset loader (file identifiers and sizes):
https://moscot.readthedocs.io/en/stable/_modules/moscot/datasets.html

[2] Moscot multiomics translation tutorial (representations and normalization):
https://moscot.readthedocs.io/en/stable/notebooks/tutorials/600_tutorial_translation.html

[3] POT GW API and Bregman implementation (PPA versus PGD, tolerances):
https://pythonot.github.io/gen_modules/ot.gromov.html
https://pythonot.github.io/_modules/ot/gromov/_bregman.html

[4] POT stochastic GW implementation; references Kerdoncuff, Emonet and Sebban,
Sampled Gromov Wasserstein (2021):
https://pythonot.github.io/_modules/ot/gromov/_estimators.html

[5] SciPy exact linear assignment API:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html

[6] Liang and Yang, A Provably Convergent and Practical Algorithm for
Gromov–Wasserstein Optimal Transport, May 2026 preprint:
https://arxiv.org/html/2605.04175v1

[7] Li et al., A Convergent Single-Loop Algorithm for Relaxation of
Gromov-Wasserstein in Graph Data, ICLR 2023:
https://arxiv.org/abs/2303.06595

[8] Altschuler, Weed and Rigollet, Near-linear time approximation algorithms for
optimal transport via Sinkhorn iteration, NeurIPS 2017:
https://arxiv.org/abs/1705.09634

[9] PyTorch CUDA semantics, especially asynchronous execution and precision:
https://docs.pytorch.org/docs/stable/notes/cuda.html

[10] Context only, not a substitute for the user's stochastic theorem:
Silveti-Falls, Molinari and Woodstock, Frank-Wolfe with Moreau Envelope Smoothing
for Nonsmooth Nonconvex Problems (2026):
https://arxiv.org/html/2606.00853v1
