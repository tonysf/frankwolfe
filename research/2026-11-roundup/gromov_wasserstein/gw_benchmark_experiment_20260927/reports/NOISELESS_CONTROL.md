# Why the noiseless control can be solved in one Frank–Wolfe step

This is an analytical control interpretation, not a claim about general nonconvex GW optimization or GPU performance. The independent check uses only the six frozen noiseless development instances at n=128/512, seeds 1000–1002. It does not read held-out arrays or planted correspondences, change any optimizer, or select parameters.

For symmetric costs and equal uniform marginals, let x=Aa and y=Bb be the weighted row means. At the product coupling P₀=abᵀ, the protocol's unregularized squared-loss gradient is

\[
\nabla F(P_0)=2(A^{\circ2}a)\mathbf1^\top
 +2\mathbf1(B^{\circ2}b)^\top-4xy^\top.
\]

Every feasible coupling has fixed marginals, so the first two terms contribute the same value to every feasible linear-oracle objective. The full Frank–Wolfe assignment oracle therefore maximizes xᵀQy. For Q equal to a permutation coupling, this is a scaled dot product of matched weighted row means.

Suppose B[j,l]=A[p(j),p(l)] for a permutation p. Then y[j]=x[p(j)]. The rearrangement inequality matches the sorted values of x and y. If all weighted row means are distinct, the assignment is unique and pairs i with the j satisfying p(j)=i. The resulting coupling S has

\[
F(S)=\frac1{n^2}\sum_{i,k}
  \bigl(A_{ik}-B_{j(i),j(k)}\bigr)^2=0.
\]

Squared GW is nonnegative over nonnegative couplings. Thus the exact global quadratic line search on (1−γ)P₀+γS has a global minimizer at γ=1. If the initial product loss is positive, the endpoint improves it strictly. Repeated weighted row means invalidate the unique-identification argument; this is a sufficient condition for this particular control, not a general GW recovery theorem.

The independent NumPy/SciPy float64 check constructs the full product gradient, calls SciPy's assignment solver, verifies the recovered cost matrices directly, and computes the segment coefficients without importing benchmark kernels. All six full-gradient assignments match the sorted-degree assignment and recover the cost matrix exactly. Direct nonnegative evaluation of every permutation loss is exactly 0, and every global segment minimizer is γ=1. The original files contain float32 costs, promoted to float64 for this check; uniform masses are exactly representable here.

| Development instance | Minimum float64 degree spacing | Equal neighboring float64 degrees | Duplicate NumPy float32 degrees |
|---|---:|---:|---:|
| metric128_s1000 | 1.1944212e-6 | 0 | 0 |
| metric128_s1001 | 1.0388903e-6 | 0 | 0 |
| metric128_s1002 | 4.9686059e-7 | 0 | 0 |
| metric512_s1000 | 9.8720193e-8 | 0 | 0 |
| metric512_s1001 | 1.6763806e-7 | 0 | 0 |
| metric512_s1002 | 9.8953024e-9 | 0 | 1 |

Sorted source/target row means agree exactly in float64 in all six cases. NumPy float32 averaging merges one degree pair for n=512, seed1002. Its reduction order is not a replica of a CUDA reduction: finite-precision product gradients may therefore introduce ties or perturb the assignment, and the exact arithmetic argument does not promise that every float32 implementation recovers the permutation in one step. The near-zero noiseless FW result in the earlier pilot is consistent with this control structure; it does not establish behavior on noisy or unrelated costs.

Reproduce with `.venv/bin/python validate_noiseless_control.py` from the experiment folder. The script limits BLAS to one thread and has a strict six-file development whitelist. Source: [validate_noiseless_control.py](../validate_noiseless_control.py). Full values, library versions, script hash, input-file hashes and segment coefficients: [noiseless_control.json](noiseless_control.json). The independent six-instance check completed in under one second on the local CPU; this duration is not a solver benchmark.
