# CT experiments: reproducibility details

This supplement specifies the data, numerical settings, and reporting rules for the SFRAMES low-dose CT experiment. The paper describes the optimization problem and main comparisons.

The current paper adds nonnegative-only baselines at the processed-view budget described in the final section below. Earlier total-view comparisons are retained as historical records. The data, parameter choices, initialization and evaluation cohort are unchanged.

## Data and view splits

The measurements are Mode 1 (low dose) from [2DeteCT](https://doi.org/10.1038/s41597-023-02484-6). The image reference is the dataset's published Mode 2 high-dose reconstruction of the same slice. It is used for evaluation only.

| Use | Acquisition group | Slice IDs | Detector |
| --- | --- | --- | --- |
| Development | Mix 2, Mix 3 | `03150`, `04300`, respectively | 2 |
| Evaluation | Mix 1 | `00601`, `01201` | 1 |
| Evaluation | Mix 2 | `03041`, `03381`, `03611` | 2 |
| Evaluation | Mix 3 | `03791`, `04131`, `04931` | 2 |
| Evaluation | Almond (OOD Pure) | `05581`, `05611` | 1 |
| Evaluation | Walnut (OOD Pure) | `05731`, `05761` | 1 |
| Evaluation | Lava stone (OOD Pure) | `05831`, `05861` | 1 |

Evaluation slices were selected before the expanded evaluation. The [dataset paper's acquisition table](https://arxiv.org/html/2306.05907) gives the full slice-to-detector mapping. Slices within an acquisition group are repeated sections; the six groups are the units of the descriptive group summaries. The development and evaluation **slices** are disjoint, but Mix 2 and Mix 3 occur in both sets.

Download the named archive members `sliceNNNNN/mode1/{sinogram,dark,flat1,flat2}.tif` and `sliceNNNNN/mode2/reconstruction.tif` from the corresponding records:

| Slice block | Measurements | Reference reconstructions |
| --- | --- | --- |
| 1–1000 | [8014758](https://zenodo.org/records/8014758) | [8017583](https://zenodo.org/records/8017583) |
| 1001–2000 | [8014766](https://zenodo.org/records/8014766) | [8017604](https://zenodo.org/records/8017604) |
| 3001–4000 | [8014829](https://zenodo.org/records/8014829) | [8017618](https://zenodo.org/records/8017618) |
| 4001–5000 | [8014874](https://zenodo.org/records/8014874) | [8017624](https://zenodo.org/records/8017624) |
| OOD | [8014907](https://zenodo.org/records/8014907) | [8017653](https://zenodo.org/records/8017653) |

1. Read the sinogram with angles on the first axis (transpose if needed). Drop its repeated last view, leaving 3,600 distinct angles. Sum adjacent pairs of detector cells, leaving 956 bins, and apply the same binning to the dark field and to the average of the two flat fields.
2. Form transmission as `(binned_sinogram - binned_dark) / (binned_flat - binned_dark)`. For detector 1, evaluate the linear interpolant at detector indices shifted by **+1**, extrapolating at the edge, as in the [2DeteCT authors' preprocessing](https://github.com/mbkiss/2DeteCTcodes/blob/main/Reconstructions_2DeteCT.py). For detector 2 the shift is zero.
3. Floor transmission at $10^{-6}$, take its negative logarithm, and reverse the detector axis following the [DeepInverse angle and detector convention](https://deepinv.org/auto_examples/external-libraries/demo_astra_2detect.html). The floor changes 12 sinogram entries in slice `05861` and none in the other evaluation slices.
4. Use a $512\times512$ image grid and the [ASTRA](https://astra-toolbox.com/docs/proj2d.html#fan-beam) CPU `line_fanflat` ray model. Source-to-origin distance is $431.019989$ mm and source-to-detector distance is $529.000488$ mm. Detector pitch is $0.1496$ mm. The square field width is $w=956\cdot0.1496\cdot431.019989/529.000488$ mm, with center $(-w/1024,-w/1024)$. View $i$ has angle $\pi-2\pi i/3600$.
5. Vertically flip the Mode 2 image, divide the native $1024\times1024$ pixel values by the native voxel width $w/1024$ to match the projector's physical path lengths, and average nonoverlapping $2\times2$ blocks.

View indices are zero based. For $N=360$, train on $i\bmod 10=0$; for $N=1080$, train on $i\bmod 10\in\{0,3,6\}$. Both use $i\bmod 10=9$ for the 360 held-out views and $i\bmod 30=0$ for the 120 calibration views. Calibration views are part of the training set.

## Initialization, oracle, and randomness

All methods use the same rank-64 factors for a slice. Use the calibration view set $J$ to form a sensitivity-normalized backprojection:

$$
X_{\mathrm{init}} =
\frac{(P_J^*y_J)_+}{\max\{P_J^*P_J\mathbf{1},\,10^{-3}\max(P_J^*P_J\mathbf{1})\}},
$$

where division and the outer maximum are entrywise and the inner maximum is over pixels. Compute a thin SVD of this image. For each of the leading 64 singular triplets $(a_k,\sigma_k,b_k)$, choose $(p_k,q_k)$ from $((a_k)_+,(b_k)_+)$ and $((-a_k)_+,(-b_k)_+)$ to maximize $\|p_k\|_2\|q_k\|_2$, choosing the first pair on a tie. Set

$$
s_k=\sqrt{\sigma_k\|p_k\|_2\|q_k\|_2},\qquad
U_{:,k}=s_k\frac{p_k}{\|p_k\|_2},\qquad
V_{:,k}=s_k\frac{q_k}{\|q_k\|_2},
$$

with zero columns if either norm is zero. Add independent uniform $[0,1)$ perturbations scaled by $10^{-3}$ times the RMS of the respective factor. For $Z=P_J(UV^\top)$, compute $a=\max\{\langle Z,y_J\rangle/\|Z\|_F^2,10^{-6}\}$ and multiply each factor by $\sqrt a$.

Set $R_U=2\|U_0\|_{\mathrm{op}}$, $R_V=2\|V_0\|_{\mathrm{op}}$. From an all-ones unit-Frobenius initial direction, apply 10 power iterations to $P_J^*P_J$. Divide the final image norm before renormalization by $120\cdot956$ to obtain $\widehat\lambda$. Use the curvature scales $L_U=\widehat\lambda\|V_0\|_{\mathrm{op}}^2$ and $L_V=\widehat\lambda\|U_0\|_{\mathrm{op}}^2$ for SFRAMES and the fixed-step baselines. The revised SPRING baseline recomputes its block curvatures during optimization as specified below.

These numerical curvature estimates are fixed for the run; they are not certified global Lipschitz bounds over the constraint set.

Each stochastic gradient uses a uniform sample of 64 training views without replacement within the batch, except where an algorithm specifies a snapshot or full refresh. Each view includes all 956 detector bins. SARAH difference terms share the sampled views. The recorded environment uses ASTRA 2.5.0 (FP32 projection/backprojection), NumPy 2.5.4, and FP64 factors and loss accumulation.

| Stage | Calibration seed | Oracle seed labels |
| --- | ---: | --- |
| Original SFRAMES joint search | 110 | 110 |
| Baseline development and joint loss/feasibility retune | 110 | 110, 111, 112 |
| Evaluation | 201 | 202, 203, 204 |

Calibration uses `numpy.random.default_rng(calibration_seed)`. Optimizers use `numpy.random.default_rng(100_000 + oracle_seed_label)`. Pair runs by slice, training-view count, and oracle seed label; methods with different estimators can consume random draws differently. Full-batch SFRAMES and the archived fixed-step PALM are deterministic given the calibration, so each runs once per slice and view count and is shared across seed pairings.

## Solvers and numerical parameters

The baselines use the cited update structures with parameters tuned for this CT problem. Their steps, refresh frequencies, and snapshot sizes below are practical choices rather than theorem-prescribed constants.

- **SFRAMES:** $\beta_0=c_\beta/\max(L_U,L_V)$, $\beta_k=\beta_0(k+1)^{-1/4}$, $\gamma_k=\gamma_0(k+1)^{-3/4}$, and momentum weight $\alpha_k=(k+1)^{-1/2}$. Initialize the momentum with the first gradient. Add $(x-\Pi_Dx)/\beta_k$ to the momentum before the LMO; $\Pi_D$ clips both factors to the nonnegative orthant. Minibatch SFRAMES uses 64 views and full-batch SFRAMES uses all $N$. Each spectral-ball LMO uses a thin SVD of the corresponding block direction $G$. It returns $-R$ times the product of the left and right singular vectors for retained singular values; discard singular values at or below $\epsilon_{\rm mach}\max(\operatorname{shape}(G))\sigma_{\max}(G)$. The resulting linear objective gap is bounded by $R$ times the sum of the discarded values.
- **[SPRING–SARAH](https://doi.org/10.1137/20M1387213):** Alternating block steps $1/(2\widetilde L_{U,k})$ and $1/(2\widetilde L_{V,k})$, with current minibatch curvature estimates. The first gradient estimate is full for each block. Thereafter each block independently refreshes from the full gradient with probability $1/q$ and otherwise uses a SARAH difference.
- **Archived custom [ProxSARAH](https://www.jmlr.org/papers/v21/19-248.html):** Joint step $a/(L_U+L_V)$, one snapshot update followed by $m$ recursive updates, and relaxation $0.8$. A minibatch snapshot samples $\min(N,256)$ views without replacement; a full snapshot uses all $N$.
- **Archived custom [PALM](https://doi.org/10.1007/s10107-013-0701-9):** Alternating full block gradients and steps $a/L_U$ and $a/L_V$; the second gradient uses the updated first block.

Baseline proximal steps numerically approximate the Euclidean projection onto the intersection of each factor's spectral ball and its nonnegative orthant. A single-set projection that meets the other constraint can be accepted directly; otherwise Dykstra's algorithm uses orthant clipping and spectral projection by singular-value clipping. For a feasible candidate $z$ and a primal–dual gap $\Delta$, the exact-arithmetic distance bound is $\|z-\Pi_K(v)\|_F\le\sqrt{2\Delta}$ for the projection problem on $K$. The implementation adds a conservative rounding allowance and requests distance $10^{-8}(k+1)^{-1.1}$, checking on the first and every fifth inner iteration and on the last, with a cap of 2,000 inner iterations. It records when the rounding allowance limits the achievable accuracy. These numerical checks are not interval-arithmetic certificates, and the exact-prox convergence theorems do not automatically apply to a fixed floating-point accuracy floor. Final inward scaling enforces the spectral bound to numerical tolerance and preserves nonnegativity.

The exact spectral LMO for a block direction $G=A\operatorname{diag}(\sigma)B^\top$ returns $-R A_+B_+^\top$, where the columns correspond to positive singular values. Its objective is $-R\|G\|_*$ by spectral–nuclear norm duality. The implementation's reported support gap accounts for the discarded singular values; it does not include a separate certified bound for rounding errors in the SVD. Each loss component is a polynomial in the two factors and has a Lipschitz gradient on a bounded neighborhood of the compact product of spectral balls. A smooth cutoff equal to one on that neighborhood gives a globally Lipschitz-gradient extension if needed for the theoretical assumptions, without changing any values or gradients queried by the solvers. Strictly positive factors of sufficiently small norm lie in the interior of both constraints.

## Development searches

Each row describes a Cartesian product, with the stated exception for the additional anchor pair. Development uses both training-view counts on the two designated slices.

| Method | Candidates | Selected setting |
| --- | --- | --- |
| Original minibatch and full-batch SFRAMES, separately | $c_\beta\in\{0.25,1,4,16,64\}$; $\gamma_0\in\{0.05,0.1,0.16,0.25,0.5,0.8\}$ | $(16,0.1)$ for both |
| SPRING–SARAH, archived fixed-step run | $a\in\{0.7,1.1,1.6,2.3,3.4,5\}$; $q\in\{8,16,32\}$ | $(a,q)=(2.3,16)$ |
| SPRING–SARAH, revised paper | Published practical rule; no new search | $1/(2\widetilde L)$ per block; $q=16$, batch 64, five power iterations plus a final norm evaluation |
| ProxSARAH, archived custom parameters | $a\in\{0.9,1.5,2.3,3.4,5\}$; $m\in\{6,12,24,48,96,192\}$; both snapshot policies | $(a,m,s)=(2.3,96,\text{minibatch})$ |
| PALM, archived custom parameters | $a\in\{0.6,1.1,1.6,2.3,3.4,5\}$ | $a=2.3$ |
| Minibatch SFRAMES, joint loss/feasibility retune | $c_\beta\in\{0.25,1,2,4\}$; $\gamma_0\in\{0.05,0.075,0.1\}$; also $(16,0.1)$ | See retune status below |

The original SFRAMES search used 8,640 views per direction, one oracle seed label, and 240 development runs. It minimized the median repaired-to-initial loss ratio at the last iterate within budget over the four slice/view-count cases, breaking ties by smaller $c_\beta$ and then smaller $\gamma_0$. See the [original development score heatmap](../../../results/stochastic_matrix_factorization/ct/spectral_balls/expanded/reports/paper_section/joint_selection.png).

Historical fixed-step baseline retuning used 34,560 views per direction and 960 development runs. ProxSARAH started with $m\in\{6,12,24\}$, then doubled the upper edge when selected there; the table shows the final grids. Search guards were $a\in[0.05,80]$, $q\in[1,128]$, and $m\in[3,256]$. All selected numerical baseline parameters were interior to their final grids. Selection minimized the median repaired-to-initial loss ratio at the last iterate within budget, first taking seed medians within each slice/view-count case and then a median across the four cases. Ties favored smaller step multiplier and then smaller secondary parameter. Nonfinite candidates were ineligible; operational or projection failures stopped selection. No evaluation slices, held-out views, or reference images entered parameter selection.

### Joint loss/feasibility retune

At 34,560 views per direction (540 minibatch updates), rank the 13 SFRAMES pairs by the count of development runs meeting **both** repaired training loss at most 30% of initialization and normalized raw orthant distance at most 1% at the final iterate. There are 12 cases per setting (two slices, two view counts, three seeds). Break ties by lower median repaired-to-initial loss, then lower median normalized raw distance, then lexicographically by $(c_\beta,\gamma_0)$. If no candidate hits both thresholds, first minimize the median of the larger of the two ratios to their thresholds, then apply those tie-breaks. Compute all medians first over seeds and then over the four slice/view-count combinations. The 0.1% distance and 40% loss thresholds are sensitivity checks and do not enter selection.

Lock one setting before running it on the existing 84 paired evaluation cases (14 slices, two view counts, three seeds). Use the same 69,120 views per direction as the original minibatch SFRAMES and ProxSARAH references. The evaluation cohort is reused to make exact pairwise comparisons; it is excluded from parameter selection.

The following marked joint-tune table retains its original ProxSARAH reference for the predecessor publication audit. The current comparison uses the corrected reference below.

<!-- CT_FEASIBILITY_RETUNE_RESULT_START -->
The joint sweep selected **(c_beta, gamma_0) = (4,0.1)**. It met the primary endpoint target in 12/12 development runs.
The [development tradeoff plot](../../../results/stochastic_matrix_factorization/ct/spectral_balls/expanded/reports/paper_section/revised/joint_selection.png) shows all 13 settings.

Each entry gives the number of final iterates meeting both targets, followed by the number of runs meeting both at any recorded iterate. There are 42 paired cases per training-view count.

| Training views | Repaired loss / initial loss | Relative raw distance | Joint tune | Original SFRAMES | ProxSARAH |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 360 | 30% | 1% | 39 / 39 | 0 / 0 | 39 / 39 |
| 360 | 30% | 0.1% | 0 / 0 | 0 / 0 | 39 / 39 |
| 360 | 40% | 1% | 39 / 39 | 0 / 0 | 42 / 42 |
| 360 | 40% | 0.1% | 0 / 0 | 0 / 0 | 42 / 42 |
| 1,080 | 30% | 1% | 39 / 39 | 0 / 3 | 39 / 39 |
| 1,080 | 30% | 0.1% | 0 / 0 | 0 / 0 | 39 / 39 |
| 1,080 | 40% | 1% | 39 / 39 | 0 / 3 | 42 / 42 |
| 1,080 | 40% | 0.1% | 0 / 0 | 0 / 0 | 42 / 42 |

<!-- CT_FEASIBILITY_RETUNE_RESULT_END -->

## Metrics and aggregation

A complete update of both factors is one iteration. Each block or joint gradient evaluation and each block-Hessian operator application counts as one oracle call. The separate `gradient_calls` and `curvature_operator_calls` counters sum to `oracle_calls`; historical methods have zero curvature-operator calls. If the calls use view batches $B_1,\dots,B_G$, the processed-view count per direction is $\sum_{\ell=1}^G|B_\ell|$. A view includes one forward and one adjoint operation over all detector bins. Count snapshots, refreshes, both terms in each recursive difference, and all six operator applications in each block-curvature estimate. Exclude calibration and diagnostic projections from the optimizer work axes. No wall-clock comparison is reported.

Record the initial state and every evaluation update. With a cap of 69,120 forward and 69,120 adjoint views (the earlier cap was 17,280 per direction), use the last recorded iterate at or below the cap. A method with variable work per iteration may finish an update beyond the cap; that update is excluded from the budget-matched report. The matched retune comparison requires all three arms to reach the cap exactly. To extend an existing run, continue from its saved optimizer and RNG state only after verifying matching data, initialization, solver settings, and numerical code; otherwise start from the common calibrated factors.

Raw training loss evaluates the original factors. To obtain repaired factors, clip each factor to its nonnegative entries, then scale it down if needed to meet its spectral radius. Use the repaired image to measure repaired training loss, held-out loss, and relative Frobenius image error against the aligned Mode 2 reconstruction. Repair does not affect solver iterates. The raw orthant distance and its normalized version are

$$
d_D(x)=\sqrt{\|\min(U,0)\|_F^2+\|\min(V,0)\|_F^2},
\qquad
\bar d_D(x)=\frac{d_D(x)}{\sqrt{\|U_0\|_F^2+\|V_0\|_F^2}}.
$$

For a joint target $(\rho,\varepsilon)$ require both $f(\mathcal R(x_k))/f(x_0)\le\rho$ and $\bar d_D(x_k)\le\varepsilon$ at the same saved iterate, where $\mathcal R$ is the repair. The original experiment evaluates all combinations of $\rho\in\{0.4,0.3\}$ and $\varepsilon\in\{0.01,0.001\}$. First-hit work uses only recorded iterates, without interpolating crossings. A target never reached within the cap is right censored. Ratios of first-hit work are defined only on pairs where both methods reached the target and are reported with their coverage. The retuned comparison prioritizes attainment at the endpoint; earlier hits are a separate diagnostic.

Aggregate endpoint metrics and paired percentage changes by taking medians over seeds within slices, over slices within groups, and then over groups. The paper displays the three loss columns multiplied by $10^3$ and raw infeasibility as a percentage of the initial factor-pair norm. Normalize the distance within each run before aggregation; dividing aggregated distances by an aggregated norm would produce a different statistic. For a paired percentage change, subtract the comparator and divide by its value before aggregation. For attainment, take means at the same three levels. Trajectory curves show arithmetic means, with shaded bands at one descriptive standard deviation above and below the mean for every method. Acquisition groups receive equal weight; slices within a group and physical repetitions within a slice receive equal weight. Full-batch SFRAMES contributes its single deterministic run per slice. Iteration plots align at the same integer iteration. Gradient-call and gradient-view plots use the latest native iterate at or below each work value, with step curves and bands. Each method stops at the smallest final observed work value (or iteration) among its runs, keeping the cohort fixed; endpoint tables still use each run's last iterate within the full view cap. No individual trajectories are drawn.

### Original first-hit and paired summaries

The following two tables retain the archived fixed-step SPRING comparison with $(a,q)=(2.3,16)$. They are historical results, superseded for SPRING in the revised paper by the local-curvature comparison below. The other arms are unchanged.

The original evaluation has 308 runs (84 each for minibatch SFRAMES, SPRING–SARAH, and ProxSARAH; 28 each for full-batch SFRAMES and PALM). Specimen-balanced attainment percentages for the original five arms, averaged over the two view counts, are:

| Method | $(0.4,0.01)$ | $(0.4,0.001)$ | $(0.3,0.01)$ | $(0.3,0.001)$ |
| --- | ---: | ---: | ---: | ---: |
| Minibatch SFRAMES | 3 | 0 | 3 | 0 |
| Full-batch SFRAMES | 0 | 0 | 0 | 0 |
| SPRING–SARAH | 88 | 88 | 72 | 72 |
| ProxSARAH | 100 | 100 | 94 | 94 |
| PALM | 94 | 94 | 67 | 67 |

Of 32 comparator/target/view-count combinations, only five had paired first hits with minibatch SFRAMES. Within those combinations, 2–3 of 42 pairs from one group reached the target with both methods; the median ratio of minibatch SFRAMES work to comparator work ranged from 1.06 to 6.68. At the cap, every original SFRAMES endpoint exceeded 1% normalized raw orthant distance. The largest absolute raw versus repaired endpoint loss gap was 20.19% of repaired loss.

| Comparator | Minibatch SFRAMES paired change in repaired loss, $N=360$ | $N=1080$ |
| --- | ---: | ---: |
| Full-batch SFRAMES | −7.9% | −20.8% |
| SPRING–SARAH | −44.3% | −40.2% |
| ProxSARAH | −19.4% | −18.9% |
| PALM | −33.4% | −53.7% |

Minibatch SFRAMES had lower repaired loss in all six group summaries in each comparator/view-count combination. The individual group changes ranged from −63.6% to −5.8%. These are descriptive comparisons across acquisition groups.

## Experiment records

The data-preparation and cohort manifests record source archives, file hashes, slice groups, and view splits. The original SFRAMES and baseline selection records store all development candidates and their scores. The joint retune has its own protocol, development and evaluation plans, and selection record. Per-run `metadata.json`, `trace.json`, and `result.json` provide configuration, native observations, and validated endpoints. The summary JSON records pairings, group summaries, first-hit coverage, and work budgets. These roles are distinct: the development selections define configurations, and the evaluation traces determine the tables and trajectory plots.

<!-- CT_BASELINE_STEP_AUDIT_START -->
## PALM and ProxSARAH step-size audit — 12 November 2026

The archived PALM and ProxSARAH results use custom tuned parameters. They do not reproduce the papers' prescribed parameter choices. This audit is resolved by the completed paper-parameter reruns below. The old runs and their parameter audit remain historical evidence; the corrected SPRING arm is retained.

**PALM.** The archived steps are fixed at $2.3/L_U^0$ and $2.3/L_V^0$. The practical comparator in SPRING Section 5.1 instead recomputes full-data block curvature at each update and uses $1/\widehat L$. Five normalized power iterations plus the final norm evaluation require six Hessian applications per block. The second estimate uses the newly updated first factor. The [original PALM paper](https://bolte.perso.math.cnrs.fr/BST2013.pdf), Remark 3.2(iii–iv), permits inverse-Lipschitz steps for convex block constraints such as ours; a finite power estimate remains an uncertified approximation. Charging all curvature work gives $14N$ views per direction per iteration.

**ProxSARAH.** Fixed steps are permitted, and the implemented estimator/proximal/relaxation structure matches Algorithm 1. However, the independently chosen proximal step $2.3/(L_U^0+L_V^0)$ and relaxation $0.8$ do not follow the [paper's](https://www.jmlr.org/papers/volume21/19-248/19-248.pdf) coupled constant rule:

$$
\omega=\frac{3(N-b)}{2b(N-1)},\qquad
\gamma=\frac{1}{L\sqrt{\omega m}},\qquad
\eta=\frac{2\sqrt{\omega m}}{4\sqrt{\omega m}+1}.
$$

The relaxation must be admissible ($\gamma\le1$). The dynamic alternative precomputes a relaxation schedule from fixed $L$; it does not update curvature online. General ProxSARAH permits minibatch snapshots, but the main finite-sum guarantees and author examples use full snapshots. Our fixed 256-view snapshots retain a variance term. Also, the sum of initial block-curvature proxies does not certify the required joint average-smoothness constant: it omits residual cross-Hessian terms.

The [audit record](../../../results/stochastic_matrix_factorization/ct/spectral_balls/expanded/baseline_step_rules/audit.json) and [primary-source audit](../../../results/stochastic_matrix_factorization/ct/spectral_balls/expanded/baseline_step_rules/provenance/proxsarah_step_audit.json) retain the checked formulas, qualifications, and source hashes. Historical results are retained as evidence; they should not be described as runs with the published step-size prescriptions.
<!-- CT_BASELINE_STEP_AUDIT_END -->

<!-- CT_SPRING_LOCALCURVATURE_CORRECTION_START -->
## SPRING with the published practical step rule

This section records the predecessor publication. Its corrected SPRING results are retained; the later correction below replaces its PALM and ProxSARAH references.

The archived SPRING runs used fixed steps $2.3/L_U^0,2.3/L_V^0$ from initial curvature scales. Those steps differ from Section 5.1 of the [SPRING paper](https://doi.org/10.1137/20M1387213), which estimates current minibatch block curvature at every update and uses $1/(2\widetilde L)$. The revised paper replaces that arm with 84 fresh runs using this published practical rule. It retains the formal loopless SARAH estimator and $q=16$; no additional step tuning or endpoint-loss screening is used. All other original arms and the completed joint SFRAMES tune are retained.

For a batch $B$ of 64 training views and $d=956$ detector bins, define the block Hessian actions

$$
H_{U,B}(V)[Z]=\frac{P_B^*P_B(ZV^\top)V}{|B|d},\qquad
H_{V,B}(U)[Z]=\frac{[P_B^*P_B(UZ^\top)]^\top U}{|B|d}.
$$

The first block uses $V_k$; the second uses the just-projected $U_{k+1}$. These are Hessians of the same mean squared batch loss used by the gradient. They are not a product bound based on an initial image-operator norm and a factor norm.

For each block, draw a fresh Gaussian factor-shaped direction and normalize it in Frobenius norm. Apply $Z_{j+1}=H[Z_j]/\|H[Z_j]\|_F$ five times, then use the additional application $\widetilde L=\|H[Z_5]\|_F$ and step $1/(2\widetilde L)$. Thus five normalized power iterations require **six** Hessian applications. The curvature batch is always 64 views: reuse the SARAH-difference batch on recursive updates, and draw a separate curvature batch on full-gradient refreshes. A separate seeded random stream supplies curvature-only batches and power directions, preserving the gradient sampling stream. There is no power-vector warm start, running maximum of curvature estimates, step cap, backtracking, or added damping. A zero or nonfinite estimate stops the run instead of silently substituting a fixed step. The estimate is numerical, not a certified Lipschitz upper bound.

The full gradient initializes each block estimate; later each block refreshes independently with probability $1/16$. A recursive first-block difference compares $(U_k,V_k)$ with $(U_{k-1},V_{k-1})$; the second-block difference compares $(U_{k+1},V_k)$ with $(U_k,V_{k-1})$, using the same batch for both terms. The paper's formal estimator and practical step rule are combined here. The authors' separate SNMF MATLAB driver instead uses an initial SGD epoch and periodic full refreshes; those driver choices are not imported into this CT comparison.

Each Hessian action uses one forward and one adjoint projection on its batch. Curvature therefore adds 12 operator calls and 768 views **per direction per complete two-block iteration**. These views are included in the 69,120-view cap, and the operators are included in the oracle-call axis. Gradient evaluations remain separately recorded. Endpoints use the last native iterate within the inclusive work cap; an overshooting final iterate is excluded. Every iteration is recorded. The spectral-ball/orthant projections, problem data, and calibrated initial factors are unchanged.

Corrected paired changes in repaired loss (original minibatch SFRAMES minus SPRING, divided by SPRING) are:

| Training views | Median paired change across groups |
| ---: | ---: |
| 360 | -61.1% |
| 1,080 | -61.1% |

The [frozen specification](../../../results/stochastic_matrix_factorization/ct/spectral_balls/expanded/spring_localcurvature/specification.json), [implementation notes](../../../results/stochastic_matrix_factorization/ct/spectral_balls/expanded/spring_localcurvature/README.md), [completion audit](../../../results/stochastic_matrix_factorization/ct/spectral_balls/expanded/spring_localcurvature/validation_summary.json), and [publication audit](../../../results/stochastic_matrix_factorization/ct/spectral_balls/expanded/spring_localcurvature/report_update.json) record the method and result provenance. The publication audit links the derived comparison summary and exact pre-replacement report/audit backup. Historical files and their manifests were not rewritten.
<!-- CT_SPRING_LOCALCURVATURE_CORRECTION_END -->

<!-- CT_BASELINE_PAPER_PARAMETERS_START -->
## Completed PALM and ProxSARAH paper-parameter comparison

The paper replaces both archived custom-parameter arms with 84 fresh PALM and 84 fresh ProxSARAH runs. The corrected local-curvature SPRING arm, both original SFRAMES arms, and the completed joint SFRAMES tune are retained. The data, split, projections, and shared starting factors are unchanged.

**PALM.** At each block update, estimate full-data block curvature with a fresh normalized Gaussian direction, five normalized Hessian applications and a sixth application for the norm. Use $1/\widehat L$; the second block uses the updated first factor. Fresh power directions make the numerical implementation random, so all three seed repetitions are executed instead of sharing one deterministic reference. Twelve curvature applications plus two block gradients cost $14N$ views per direction per complete iteration: 13 updates fit the cap for $N=360$ and four for $N=1080$.

**ProxSARAH.** Use the constant-parameter v3 experiment in Section 5 of the paper, with $\gamma=0.99$, $C=2/(3\gamma^2)$, $m=\lfloor\sqrt N\rfloor$ and $b=\lfloor\sqrt N/C\rfloor$. Thus $(b,m)=(27,18)$ or $(48,32)$. Full snapshots use all $N$ training views; recursive differences use the same batch at the two points. Apply the normalized-$L=1$ paper rule to $f/\widehat L$, giving proximal step $[2/(4+0.99)]/\widehat L$ and fixed relaxation $0.99$. The frozen scale is $\widehat L=0.25(L_U^0+L_V^0)$. Only $c_L$ is selected, starting from $2^{-5},2^{-4},\ldots,2^2$ on the two development slices, both view counts and three seeds, using 34,560 views per direction. Nonfinite settings and settings ending above initial repaired loss are ineligible; the nested median repaired-to-initial loss determines selection. A boundary winner extends the grid by a factor of two on that side, within guards $[2^{-8},2^5]$. A boundary winner at a guard requires review before evaluation. The selection record lists every tested scale. The paper's relaxation and normalized step are not independently tuned. This is a practical curvature scale, not a certified joint average-smoothness bound.

Both methods record every native iterate, and all snapshot, difference, and curvature views count toward 69,120 views per direction. An overshooting step is excluded from reported endpoints and first-hit comparisons. `gradient_calls` counts actual gradients; `oracle_calls` additionally counts Hessian applications. The same two figures and one table now use the corrected native traces.

The joint SFRAMES comparison has been recomputed against the corrected ProxSARAH reference. Entries below are endpoint hits / first hits among 42 paired cases per view count.

| Training views | Relative repaired loss | Relative raw distance | Joint SFRAMES | Original SFRAMES | ProxSARAH |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 360 | 30% | 1.0% | 39 / 39 | 0 / 0 | 42 / 42 |
| 360 | 30% | 0.1% | 0 / 0 | 0 / 0 | 42 / 42 |
| 360 | 40% | 1.0% | 39 / 39 | 0 / 0 | 42 / 42 |
| 360 | 40% | 0.1% | 0 / 0 | 0 / 0 | 42 / 42 |
| 1,080 | 30% | 1.0% | 39 / 39 | 0 / 3 | 39 / 39 |
| 1,080 | 30% | 0.1% | 0 / 0 | 0 / 0 | 39 / 39 |
| 1,080 | 40% | 1.0% | 39 / 39 | 0 / 3 | 42 / 42 |
| 1,080 | 40% | 0.1% | 0 / 0 | 0 / 0 | 42 / 42 |

The [method notes](../../../results/stochastic_matrix_factorization/ct/spectral_balls/expanded/baseline_step_rules/README.md), [selection](../../../results/stochastic_matrix_factorization/ct/spectral_balls/expanded/baseline_step_rules/selection.json), [completion audit](../../../results/stochastic_matrix_factorization/ct/spectral_balls/expanded/baseline_step_rules/validation_summary.json), and [publication audit](../../../results/stochastic_matrix_factorization/ct/spectral_balls/expanded/baseline_step_rules/report_update.json) record all configurations and checks. The publication audit links the derived all-method and joint-comparison summaries, exact predecessor backups, and the publication chain. Frozen historical summaries are unchanged.
<!-- CT_BASELINE_PAPER_PARAMETERS_END -->


<!-- CT_TRAJECTORY_MEAN_SD_START -->
### Mean and standard-deviation trajectories

The plots use the same 448 physical evaluation traces and the same charged work counters as the corrected baseline comparison. At a plotted coordinate, the weight for repetition $r$ of slice $s$ in group $g$ is $w_{gsr}=1/(G S_g R_s)$, where $G=6$, $S_g$ counts the group's slices, and $R_s$ counts physical repetitions. The plotted arithmetic mean is $\mu=\sum w_{gsr}y_{gsr}$ and the descriptive standard deviation is $\sigma=\sqrt{\sum w_{gsr}(y_{gsr}-\mu)^2}$. Bands show $\mu\pm\sigma$, not standard errors or confidence intervals. They include variation across slices and repetitions; they are not the standard deviation of six group averages. Statistics are computed in the original metric units before axis transforms. Signed arithmetic band limits are retained; an axis uses a symmetric-log scale if needed to display nonpositive limits.

Work grids are unions of native recorded work values on the common observed range of each method. The last recorded iterate is held until its next update, without linear interpolation. Overshooting updates are excluded. The existing endpoint table and numerical comparisons retain their group-balanced median summaries.

The [plotting code](../../../results/stochastic_matrix_factorization/ct/spectral_balls/expanded/trajectory_mean_sd/mean_sd_plots.py) and [publication audit](../../../results/stochastic_matrix_factorization/ct/spectral_balls/expanded/trajectory_mean_sd/report_update.json) record this presentation change and link the per-figure mean/standard-deviation arrays. Frozen optimization traces, selections, and comparison summaries are unchanged.
<!-- CT_TRAJECTORY_MEAN_SD_END -->


<!-- CT_GRADIENT_BUDGET_START -->
## Gradient-budget continuation

The current endpoint table and both existing mean/standard-deviation figures use a common cap of 69,120 gradient forward views and 69,120 gradient adjoint views. For every native row, gradient views equal total views minus curvature views. Full snapshots and both recursive-difference gradient evaluations count toward this cap. Hessian applications remain included in total operator views and `oracle_calls`, but are excluded from the displayed `gradient_calls` and gradient-view axes. This comparison does not impose equal total operator work.

PALM and SPRING–SARAH continue their 84 published runs each from the final saved optimizer state, preserving random-generator states, recursive estimates, factors and complete recorded prefixes. Their steps, power iterations, sampling and refresh rules are unchanged. The old total-budget overshooting update is a valid prefix of the extended trajectory. New endpoints exclude any update exceeding the gradient cap. Loss and feasibility are recorded at every native iteration. ProxSARAH and all SFRAMES traces are unchanged; no new tuning is performed.

The 448 physical trajectories retain equal group/slice/repetition weighting for their arithmetic means and descriptive ±1 SD bands. Native step alignment and each method's common observed support are unchanged. The endpoint table still uses successive medians over repetitions, slices within groups and groups; all best-value bolding is recomputed. First-hit results are recomputed using gradient work and retain actual total and curvature work at each hit. The original inclusive-budget comparison remains frozen and is linked from the new derived summary.

### Effect of extending the alternating methods

Changes are paired within slice and seed before the same median aggregation. Costs are directional; forward and adjoint totals agree.

| Training views | Method | Old repaired loss | Extended repaired loss | Paired change (%) | Gradient views | Curvature views | Total / gradient |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 360 | PALM | 0.0228596 | 0.0136021 | -36.14 | 69,120 | 414,720 | 7.000 |
| 360 | SPRING–SARAH | 0.0198952 | 0.0127788 | -32.27 | 68,984 | 180,480 | 3.617 |
| 1080 | PALM | 0.0288127 | 0.0200139 | -25.52 | 69,120 | 414,720 | 7.000 |
| 1080 | SPRING–SARAH | 0.0202407 | 0.0142686 | -25.59 | 68,968 | 129,792 | 2.882 |

The [continuation notes](../../../results/stochastic_matrix_factorization/ct/spectral_balls/expanded/gradient_budget/README.md), [completion audit](../../../results/stochastic_matrix_factorization/ct/spectral_balls/expanded/gradient_budget/validation_summary.json), and [publication audit](../../../results/stochastic_matrix_factorization/ct/spectral_balls/expanded/gradient_budget/report_update.json) specify frozen inputs, checkpoint provenance, work accounting and the derived comparison summary. Historical trace files, selections and summaries remain unchanged.
<!-- CT_GRADIENT_BUDGET_END -->

<!-- CT_NONNEGATIVE_BASELINES_START -->
## Nonnegative-only baselines

The existing six methods and all their recorded values are retained. Alternating NNLS and multiplicative updates minimize the same training loss over nonnegative factors without spectral bounds. Their actual factors are used for raw/evaluated training loss, held-out loss and image-reference error. Former spectral radii are diagnostics only. For the original methods, the established clipping plus optional radial spectral repair remains the evaluation convention: it produces a valid nonnegative candidate with the additional spectral restriction. Raw loss and raw orthant distance always use actual factors. Thus this is an application comparison across different feasible domains, not an isolated solver comparison on the same domain.

There are 504 physical evaluation runs: the existing 448 plus 28 deterministic NNLS and 28 deterministic multiplicative runs (14 slices at each view count). The same calibrated initial factors are reused. Deterministic runs are not duplicated to manufacture stochastic repetitions. Each group, its slices, and each slice's physical repetitions retain equal nested weights for means and descriptive standard deviations; endpoint summaries retain nested medians.

### Algorithms and selection

Alternating NNLS uses SciPy L-BFGS-B for each convex factor subproblem, with nonnegative bounds, memory 10, relative projected-KKT tolerance 1e-3, objective tolerance 1e-9, and at most 20 line-search trials. The accepted-step cap per block is selected from 5, 10, 20 using only the two development slices and both view counts, with 34,560 processed views per direction. Selection minimizes nested-median raw training loss relative to initialization; held-out loss and reference images do not enter selection. The exact selection record is linked below.

Multiplicative updates use the signed-right-hand-side majorization rule and alternate the two factors. They have no step-size tuning. Initialization, positivity safeguards, normalizations and stopping details are in the frozen method implementation and campaign notes.

### Work and recorded states

The cap is 69,120 forward views and 69,120 adjoint views per evaluation run. Every new solver application counts, including NNLS inner evaluations, line-search trials, and the cached adjoint of data for multiplicative updates. Evaluation diagnostics are excluded. Previously reported methods retain gradient views with curvature-estimation work recorded separately; this comparison therefore does not equalize total operator work across all eight methods. The displayed processed-view coordinate is the maximum of the two directional charged counts. Oracle calls count paired forward/adjoint evaluations plus the multiplicative method's standalone startup adjoint; prior arms use their gradient-call count, excluding curvature calls.

Native iterations count accepted updates: an accepted inner factor step for NNLS, a factor update for multiplicative updates, and the original update convention for each retained method. This axis does not assign equal computational meaning to an iteration. Cost plots retain all native charged-work observations, including a terminal budget-stop record when unsuccessful trials spend additional work without changing factors. At duplicate coordinates the latest state is used. Mean/SD summaries use last-observation step alignment on fixed method cohorts; no metric interpolation or extrapolation is performed.

| Training views | Method | Domain | Raw loss | Evaluated loss | Held-out loss | Raw orthant distance |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 360 | Alternating NNLS | Nonnegative only | 0.007329927 | 0.007329927 | 0.007446206 | 0 |
| 360 | Multiplicative updates | Nonnegative only | 0.01002637 | 0.01002637 | 0.01004065 | 0 |
| 1080 | Alternating NNLS | Nonnegative only | 0.01074588 | 0.01074588 | 0.01073077 | 0 |
| 1080 | Multiplicative updates | Nonnegative only | 0.01404035 | 0.01404035 | 0.0140912 | 0 |

The [campaign notes](../../../results/stochastic_matrix_factorization/ct_nonnegative/README.md), [selection](../../../results/stochastic_matrix_factorization/ct_nonnegative/selection.json), [completion audit](../../../results/stochastic_matrix_factorization/ct_nonnegative/validation_summary.json), and [publication audit](../../../results/stochastic_matrix_factorization/ct_nonnegative/report_update.json) record all configurations, hashes, work counters and comparison statistics. The preceding gradient-budget publication is preserved in the publication backup; all earlier numerical traces and summaries remain unchanged.
<!-- CT_NONNEGATIVE_BASELINES_END -->

<!-- CT_JOINT_LBFGSB_START -->
## Joint nonnegative L-BFGS-B

The current comparison contains 532 physical trajectories: all previous 504 plus 28 deterministic joint L-BFGS-B runs, one for each of the 14 evaluation slices and two view counts. Its feasible set is nonnegative factors only. The same frozen rank-64 initial factors, data and view split are reused; there is no spectral bound, radial repair, factor balancing or regularizer. All prior numerical evidence remains unchanged.

SciPy 1.18.1 L-BFGS-B operates on the concatenated U,V vector and retains its limited-memory state across joint accepted updates. This differs from alternating NNLS, whose optimizer is restarted for each factor subproblem. Each joint evaluation uses one shared residual/backprojection to obtain both factor gradients. Every trial forward/adjoint pair counts, including unsuccessful line-search trials. Loss and feasibility are recorded at every accepted state, with terminal unaccepted work retained as a repeated native iteration. Checkpoints preserve the complete reverse-communication state.

The memory is selected from {5,10,20} by twelve development runs (two slices, two view counts) at 34,560 views per direction. The selection criterion is the median relative raw endpoint loss across slices at each view count, then the median over view counts; exact ties favor lower memory. Fixed settings are relative projected-gradient tolerance 1e-3, objective tolerance 1e-9, and at most 20 line-search trials. The resulting setting is frozen before evaluation; held-out images are not used for selection.

Evaluation stops within 69,120 views separately in the forward and adjoint directions. Only numerical stopping conditions or the budget can end the joint solve; there is no inner-iteration restart cap. Diagnostics remain outside solver work. Earlier spectral-constrained methods retain their gradient-only budget and separate curvature accounting. Consequently the comparison does not equalize total operator work across all nine methods. Native joint iterations are accepted updates of both factors; oracle calls and views retain their actual charged costs. The existing group-balanced means ±1 descriptive SD and nested-median endpoint statistics are unchanged.

| Training views | Raw/evaluated loss | Held-out loss | Relative orthant distance |
| ---: | ---: | ---: | ---: |
| 360 | 0.005606946 | 0.006330238 | 0 |
| 1080 | 0.006823715 | 0.006873106 | 0 |

The [campaign notes](../../../results/stochastic_matrix_factorization/ct_joint_lbfgsb/README.md), [selection](../../../results/stochastic_matrix_factorization/ct_joint_lbfgsb/selection.json), [completion audit](../../../results/stochastic_matrix_factorization/ct_joint_lbfgsb/validation_summary.json), and [publication audit](../../../results/stochastic_matrix_factorization/ct_joint_lbfgsb/report_update.json) retain the exact configuration and results. The previous scientific publication and the subsequent gradient-estimator editorial revision are preserved in the predecessor records and document backup.
<!-- CT_JOINT_LBFGSB_END -->


## Compact paper presentation (23 November 2026)

The paper section occupies two pages, including the table and two figures. It retains the mathematical model and sampling estimator, data source, essential settings, and results with interpretations. Development grids, screening stages, exact seed values, numerical tolerances and full work accounting remain in the sections above and in the [expanded protocol](ROBUSTNESS_EVALUATION.md).

The current endpoint table contains all nine methods. The figures emphasize loss-tuned and joint-tuned SFRAMES, ProxSARAH, SPRING-SARAH and joint L-BFGS-B, showing raw loss and raw orthant distance against iterations, gradient-oracle calls and processed views. The plotted means, standard deviations and observation alignment are unchanged. Full evaluated-loss and held-out-loss curves, and the other method curves, remain available in the frozen [360-view figure](https://github.com/tonysf/frankwolfe/blob/3da4834c42c928a2488a721a04a142efb5bc5719/paper/experiments/stochastic_ct/report/trajectories_360.png) and [1,080-view figure](https://github.com/tonysf/frankwolfe/blob/3da4834c42c928a2488a721a04a142efb5bc5719/paper/experiments/stochastic_ct/report/trajectories_1080.png).

The larger campaign remains in progress. Its completed results will use the same two-page limit, replacing the second comparison figure with the prespecified stochastic-mechanism panel. No pending experimental result is presented as completed in the current paper.
