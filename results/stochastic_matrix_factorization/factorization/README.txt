Spectrally constrained factorization on Elite -- 2026-11-04

Data: official ORL64 (4096 pixels x 400 images, rank25) and Extended Yale-B32
(1024 pixels x2414 images, rank49). Each image is normalized to Euclidean
norm one, matching the original SPRING author's intended preprocessing.
Raw files, checksums, source commit, and separate global-scaled variants are
preserved in data/ and provenance/. The main runs use *_unitnorm.npz only.

Main problem:
 f(U,V) = ||UV^T-A||_F^2/(2*m*n)
 C = {||U||op<=RU, ||V||op<=RV}
 D = {U>=0, entrywise ||V||1<=s}; V is signed.
 RU=RV=1.25*sqrt(sigma1(A)); s=.35*sqrt(n*rank)*RV.
These data-only choices were frozen before optimization. This is a declared
convex-sparsity/spectral adaptation, not the original paper's l0 model.

Execution:
  /home/tony/zofw_experiments/.venv/bin/python campaign.py
The campaign acquires an exclusive lock and resumes its own checkpoints.
It first runs the author-code SPRING-SARAH controls, then26 development runs
(12 SFRAMES prefactor configurations +1 SPRING-SARAH per dataset), then12
confirmation runs (two methods x three fresh seeds x two datasets).
Development:20000 iterations, seed110. Confirmation:50000 iterations,
seeds201-203. No development configuration is dropped before its horizon.
The original-code controls use1000 reported epochs forORL and250 forYale-B.
Each must finish and reduce its own objective by20% before the main stage.

Resource use: four independent one-thread workers on four physical CPU cores
for the main campaign. Original controls run in parallel with two cores each.
No GPU or Slurm job is used. The reused Python environment is not modified.

SFRAMES: gamma0/(k+1)^.75, rho0/(k+1)^.5, mu0/(k+1)^.25;
gamma0 in{.3,1},rho0 in{.1,1},mu0 in{.1,1,10}/L_full_bound.
The first sampled gradient initializes momentum. Only the smooth stochastic
gradient enters momentum; the exact squared-distance penalty is added after.
Uniform image-column minibatches contain ceil(.05*n) images, sampled without
replacement within each batch. All pixels of each chosen image contribute.

SPRING-SARAH: Corollary3.3 constants, loopless independent block refreshes
with probability1/n, step1/(2*L_component*sqrt(30*n)). Block projections onto
the intersections use exact single-set reductions where possible and Dykstra
otherwise. Numerical primal-dual gaps and floating-point floors are recorded.
The step prescription matches the theorem; finite-precision prox/LMO errors
are not claimed to be literally exact or covered by a proved inexact theorem.

Mathematical global smoothness for SFRAMES is obtained by a C2 cutoff of each
quartic component outside a neighbourhood of C. See provenance/sframes_audit.json.
On C the extended and original objectives and gradients agree exactly.

Diagnostics: raw and feasibly repaired loss, relative reconstruction error,
constraint violations, and true smoothed FW gap at saved probes. The repair
is never used by an optimizer. The SVD rank-r reference is an unconstrained
lower bound, not a certified constrained optimum. Probe gaps are not the
theorem's average over every iteration. Initializations are shared by seed.

Timing: optimizer seconds include sampling, estimators, LMO/prox and updates;
diagnostics, initialization and checkpoint IO are excluded. Image-component
gradient counts include full refreshes and both evaluations in SARAH differences.
Concurrent CPU timings are not an isolated throughput benchmark.

Checkpoints save factors, momentum/recursive estimators, exact iteration and
RNG state every250 iterations or120 seconds. Progress is in progress.json
and runs/.../progress.json. Plots are generated automatically after development
and confirmation. Manual partial plots: python build_plots.py.

The independent author-code control has different constraints and is kept
separate. Its numerical port and known source departures are documented in
provenance/spring_original_settings.json; it is not a theorem-prescribed run.
