# PALM and ProxSARAH with paper parameter rules

This campaign replaces the archived custom-parameter PALM and ProxSARAH arms. It keeps the data, view split, product of spectral balls, nonnegative-factor constraints, projection routines, calibrated initial factors, and evaluation cohort. It runs after the completed SPRING local-curvature correction and retains that corrected arm and all SFRAMES results. Historical optimizer runs and summaries remain unchanged.

## PALM

Use the full-data practical PALM comparator from Section 5.1 of [the SPRING paper](https://doi.org/10.1137/20M1387213): at each block update, use step $1/\widehat L$, where $\widehat L$ estimates that block's current full-data Hessian norm. Start from a fresh normalized Gaussian direction, perform five normalized Hessian applications, and take the norm of a sixth application. The second block uses the updated first factor. There is no added damping, line search, or power-vector warm start.

The block-Hessian actions are

$$
H_U(V)[Z]=\frac{P_{\mathcal I}^*P_{\mathcal I}(ZV^\top)V}{Nd},\qquad
H_V(U)[Z]=\frac{[P_{\mathcal I}^*P_{\mathcal I}(UZ^\top)]^\top U}{Nd}.
$$

The [original PALM manuscript](https://bolte.perso.math.cnrs.fr/BST2013.pdf), Remark 3.2(iii–iv), permits inverse-Lipschitz steps for convex nonsmooth block terms. Our constraint indicators satisfy that condition. Finite power estimates remain numerical estimates, rather than certified upper bounds.

Fresh random power starts make this numerical implementation random: run all three seeds for every slice/view-count pair, giving 84 physical PALM runs. Each two-block update costs twelve full-data Hessian applications and two full block gradients, or $14N$ views per direction. At 69,120 views, 13 complete updates fit for $N=360$ and four for $N=1080$.

## ProxSARAH

Use the named constant-parameter **ProxSARAH-v3** numerical variant in Section 5 of [the paper](https://www.jmlr.org/papers/v21/19-248.html), with its normalized $L=1$ recipe applied to $f/\widehat L$:

$$
\gamma=0.99,\quad C=\frac{2}{3\gamma^2},\quad
m=\lfloor\sqrt N\rfloor,\quad b=\lfloor\sqrt N/C\rfloor,\qquad
\eta=\frac{2}{(4+\gamma)\widehat L}.
$$

Thus $(b,m)=(27,18)$ for 360 views and $(48,32)$ for 1,080 views; the original-loss proximal step is $0.4008016032064128/\widehat L$. Every epoch begins with a full $N$-view snapshot and continues with $m$ recursive updates. Both gradient terms in a recursive difference use the same sampled batch. Each joint projected candidate is relaxed by the fixed factor $\gamma$.

Only the fixed curvature scale $\widehat L=c_L(L_U^0+L_V^0)$ is selected on development data, starting from $c_L\in\{2^{-5},2^{-4},\ldots,2^2\}$. Use the same two development slices, two view counts, three seeds, and 34,560-view cap per direction. Nonfinite settings and settings ending above initial repaired loss are ineligible; selection minimizes the nested median repaired-to-initial loss. A boundary winner extends the grid by a factor of two on that side, within guards $[2^{-8},2^5]$; if no candidate is eligible, extend toward larger scales. A boundary winner at a guard, or no eligible setting at the upper guard, requires review before evaluation. The selection record lists the full tested grid. The normalized step and relaxation prescribed by v3 are not independently tuned. The initial block-curvature sum is a practical scaling proxy, not a certified joint average-smoothness bound. Selection precedes all 84 evaluation runs.

## Recording and comparison

The evaluation cap is 69,120 forward and 69,120 adjoint views. Charge all full snapshots, both recursive-difference gradients, and every curvature application. `gradient_calls` counts actual gradients; `curvature_operator_calls` counts Hessian actions; their sum is `oracle_calls`. Diagnostic loss evaluations and the common initialization are outside optimizer work. Every update is saved. Endpoint and first-hit comparisons use only native iterates within the cap, excluding an overshooting update. Optimizer timing is not compared.

There are 168 new physical evaluation runs and 84 logical cases per method. With the retained three stochastic arms and the 28 deterministic full-batch SFRAMES runs, the two existing trajectory figures contain 448 physical traces. Figures retain solid colored lines, with iterations, inclusive oracle calls, and views processed as their three horizontal axes.

## Publication and provenance

`protocol.json`, `selection.json`, `evaluation_plan.json`, and `implementation_manifest.json` freeze the campaign; `validation_summary.json` audits completion. `update_report.py` refuses to publish before both predecessor campaigns and all new evaluations have completed and validated.

The starting all-method summary is the completed SPRING publication's audited `corrected_summary`, never the older fixed-step comparison. A new derived summary replaces only PALM and ProxSARAH. A second derived summary recomputes the joint SFRAMES comparison against corrected ProxSARAH using the pure aggregation helpers in `feasibility_retune/analysis.py`; the frozen feasibility summary stays unchanged.

Publication acquires locks in this order: this campaign, SPRING correction, feasibility retune. It stages the same two figures and one table, compiles the paper, preserves exact backups, and journals replacement of report assets, appendix, and both predecessor publication audits. Those audits retain their original scientific anchors and gain current hashes and a successor pointer. The historical marked feasibility table stays explicitly historical so rerunning its upstream publisher cannot restore an obsolete current comparison. The appendix adds the corrected comparison separately. Interrupted publication accepts only recorded old or new file versions; unrelated edits stop replay.

`report_update.json` records the derived summaries, predecessor backups, asset hashes, and completed transaction. Archival source papers and parameter audits are under `provenance/`.
