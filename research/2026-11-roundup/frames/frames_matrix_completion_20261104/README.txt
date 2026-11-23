SFRAMES and SHCGM schedules: small convex matrix-completion experiment
Created 4 November 2026 on Elite (local CPU).

Main files
  data.npz                 Frozen 50x50 synthetic data and constraints.
  reference.json           Numerical primal/dual optimum bracket.
  provenance/algorithm_audit.json
                           Assumptions, algorithms, and theorem mapping.
  campaign_config.json     Frozen tuning/confirmation protocol.
  selection.json           Selection using development seeds only.
  completed.json           Completion and common sample-stream validation.
  runs/                    Per-run trajectories and saved final states.
  build_report.py          Scientific plots and LaTeX report builder.

Execution environment
  /home/tony/zofw_experiments/.venv/bin/python
  Exact versions are recorded in provenance/environment.json and requirements.txt.
  No existing environment was modified. A separate environment can be recreated
  from requirements.txt if the shared local interpreter is unavailable.

Run from this directory with OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1 and
MKL_NUM_THREADS=1. Commands, using the interpreter above:
  python -m pytest -q test_core.py
  python campaign.py
  python build_report.py

The campaign uses four workers pinned to physical cores 0,2,4,6. Completed runs
are reused only when configurations, frozen data and numerical source hashes
match. Interrupted individual runs are preserved rather than silently restarted.

Scientific scope
  Both prescriptions share the momentum, Moreau smoothing, LMO and convex-
  combination update. This convex experiment compares schedules and prefactors.
  It demonstrates neither algorithmic superiority nor the nonconvex contribution.
  SFRAMES tuned step/momentum/smoothing; SHCGM tuned smoothing only, retaining
  the paper constants. Equal candidate counts do not imply equal tuning freedom.
  f(X) = sum_observed (Xij-Yij)^2 / (2 * number_observed).
  C is the nuclear-norm ball of radius40. D is the entrywise box[-1,1].
  T is the identity, g is the indicator of D. Zero is strictly feasible.
  Uniform observed-entry sampling is unbiased. Both methods share sample-index
  streams for matched batch/seed, while evaluating at their own iterates.
  An exact dense SVD supplies each LMO. No approximate power-method accuracy
  assumption is used for this small experiment.

  SHCGM uses Algorithm1 of Locatello et al., NeurIPS2019, retaining the exact
  step/momentum constants and indexing. Its positive smoothing prefactor is tuned.
  SFRAMES retains exponents3/4,1/2,1/4. Both tuned prefactors and a separate
  unit-step/unit-momentum-prefactor control are evaluated.
  Each method has16 tuning configurations and2 development seeds. The selected
  configurations are tested on5 fresh seeds at each batch size16,64,256.
  Selection uses final feasible training loss at20,000updates, with batch64;
  selected prefactors are reused unchanged for the other confirmation batches.

  The diagnostic feasible output is X/max(1,max(abs(X))). This radial contraction
  preserves the nuclear ball as well as enforcing the box. It is never fed back
  to optimization. Raw loss and violation are retained separately.
  Exact true smoothed gaps are evaluated at EVERY pre-update iterate. Their
  cumulative average is the theorem statistic; sparse loss samples are not
  substituted into this average. Last-iterate objective curves are descriptive.
  Optimization time excludes diagnostics, repair and reference-solver work.
  Timings come from concurrent workers on separate physical cores and are pilot
  measurements, not a controlled standalone hardware speed benchmark.

  The reference solve is used only for evaluation. The first proposed radius20
  made the box inactive; radius40 was selected by reference constraint activity
  before running either optimizer. Evidence is in design_checks/selection.json.

  This is a new synthetic comparison, not a reproduction of published MovieLens
  losses. The problem is convex and therefore a special case of the SFRAMES
  setting. The earlier nonconvex row-sampling example is reused for illustrating
  the nonconvex theorem, with its engineered geometry and residual objective
  error explicitly disclosed.
