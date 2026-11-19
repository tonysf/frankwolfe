# SPRING–SARAH with current minibatch curvature

This comparison replaces the archived fixed-initial-curvature SPRING steps with the practical rule in Section 5.1 of [the SPRING paper](https://doi.org/10.1137/20M1387213), pp. 23–24 of arXiv v3. It retains the paper's formal loopless SARAH estimator and the existing refresh parameter $q=16$. No new parameter sweep or endpoint-loss eligibility screen is used.

The previous arm used fixed steps $(2.3/L_U^0,2.3/L_V^0)$. Its data and publication records remain historical files; the fresh local-curvature results have separate provenance. The problem, spectral-ball/orthant projections, shared calibrated factors, evaluation cohort, other methods, and joint SFRAMES tune are unchanged.

## Step rule

For the normalized batch loss

$$
 f_B(U,V)=\frac{1}{2|B|d}\sum_{i\in B}\|P_i(UV^\top)-y_i\|^2,
 \qquad |B|=64,\quad d=956,
$$

the exact-arithmetic partial Hessian actions are

$$
 H_{U,B}(V)[Z]=\frac{P_B^*P_B(ZV^\top)V}{|B|d},\qquad
 H_{V,B}(U)[Z]=\frac{[P_B^*P_B(UZ^\top)]^\top U}{|B|d}.
$$

At iteration $k$, estimate $H_{U,B_U}(V_k)$, take the first projected block step, then estimate $H_{V,B_V}(U_{k+1})$ before the second projected block step. For each estimate, independently draw a fresh Gaussian factor-shaped $Z_0$ and normalize it in Frobenius norm. Perform five normalized power iterations,

$$
 Z_{j+1}=H[Z_j]/\|H[Z_j]\|_F,\quad j=0,\ldots,4,
 \qquad \widetilde L=\|H[Z_5]\|_F,
 \qquad \eta=1/(2\widetilde L).
$$

The final norm requires a sixth Hessian application. It is an operator norm estimate, not a Rayleigh quotient or a certified upper bound. There is no warm start of the power vector, running maximum of estimates, step cap, backtracking, or additional damping. Zero or nonfinite curvature stops the run and is recorded as a failure.

The curvature batch contains 64 uniformly sampled views without replacement on every block update, including full-gradient refreshes. On recursive updates it is the same batch used in that block's SARAH difference. On full-gradient refreshes a separate 64-view curvature batch is sampled. A separate reproducible random stream supplies curvature-only batches and Gaussian power starts, so this additional randomness does not alter the existing gradient-sampling stream.

## Estimator and update order

The first estimate of each block is a full gradient. Later, each block independently refreshes with probability $1/16$. Otherwise, using the same batch for both terms,

$$
 v^U_k=v^U_{k-1}+\nabla_U f_{B_U}(U_k,V_k)
                         -\nabla_U f_{B_U}(U_{k-1},V_{k-1}),
$$

$$
 v^V_k=v^V_{k-1}+\nabla_V f_{B_V}(U_{k+1},V_k)
                         -\nabla_V f_{B_V}(U_k,V_{k-1}).
$$

The full-gradient branch uses the corresponding current query point. Both block steps project onto that factor's spectral ball intersected with the nonnegative orthant.

The paper defines this loopless estimator on p. 3, with refresh parameter $p$ described as a tuning parameter generally of order $n$. Section 5.1 does not prescribe a numerical refresh probability. Retaining $q=16$ is an inherited experimental setting, not a claim that the paper mandates it.

The authors' MATLAB SNMF driver is a different practical estimator schedule: it uses an initial SGD epoch, then periodic full-gradient block steps at epoch boundaries. It uses full curvature on those full-gradient steps. We implement the paper's formal loopless estimator with Section 5.1's minibatch curvature prescription, rather than copying that driver schedule. The algorithm is therefore described as the published practical step rule adapted to CT, not a literal reproduction of the MATLAB experiment.

## Work accounting

One Hessian application uses one forward and one adjoint projection over its batch. Each complete two-block iteration has 12 curvature-operator calls and $12\times64=768$ curvature views per direction. Gradient work is additional and includes snapshots, refreshes, and both terms of recursive differences.

- `gradient_calls`: actual joint or block gradient evaluations only.
- `curvature_operator_calls`: Hessian applications, including the sixth application in each estimate.
- `oracle_calls`: the sum of the preceding two counters.
- `forward_views`, `backward_views`: total optimization views, including curvature.
- `curvature_forward_views`, `curvature_backward_views`: the curvature portion of those totals.

Consequently, an iteration with two recursive blocks costs 16 oracle calls and 1,024 views per direction. Two full-gradient blocks cost 14 oracle calls and $2N+768$ views per direction. A mixed iteration costs 15 oracle calls and $N+896$ views per direction. Diagnostic evaluation and shared initialization remain outside optimizer work.

The evaluation cap is 69,120 total forward and adjoint views per direction. Every iterate is recorded. The report uses the last native observation within the cap, excluding a final overshooting update. Historical methods have no curvature-operator calls, so their existing gradient counts are their oracle counts. The two existing figures retain their iterations, oracle calls, and processed-views axes.

## Source evidence and records

The locally archived source is `provenance/spring_original_code/stochastic-proximal-alternating-minimization-master/` in the factorization experiment provenance. The relevant files are:

- `auxiliary/power_method.m`, lines 10–24: fresh normalized Gaussian start, five normalized applications when called with `pn=5`, then a sixth application for the norm.
- `algorithms/SNMF_SPRING_SARAH_e.m`, lines 29, 124–145 and 161–181: five power iterations, the current sampled block, coefficient $1/2$, and sequential updated factors. The `max(...)` alternative at line 128 is commented out.
- The same file, lines 69–104 and 135–140: periodic full-gradient steps and first-epoch SGD in the separate MATLAB driver.
- arXiv v3, p. 3 and Section 5.1, pp. 23–24: formal loopless estimator and the minibatch practical step rule used here.

`specification.json` and `evaluation_plan.json` freeze the method and matched cases before evaluation. `implementation_manifest.json` records the additional local-curvature solver and recording engine separately from the unchanged numerical components. `validation_summary.json` validates all 84 fresh native trajectories. `report_update.json` records staged validation, publication hashes, and the exact pre-replacement report/audit backup. The feasibility-retune summary and its selection remain unchanged; its publication audit links this separate correction.
