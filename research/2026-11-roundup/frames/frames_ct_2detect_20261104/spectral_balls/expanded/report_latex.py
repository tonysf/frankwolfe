"""Build the native LaTeX report from the completed expanded comparison."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent
ANCHOR = "sframes"
METHODS = {
    "sframes": "Minibatch SFRAMES",
    "sframes_full": "Full-batch SFRAMES",
    "spring_sarah": "SPRING--SARAH",
    "prox_sarah": "ProxSARAH",
    "palm": "PALM",
}
STEM = "expanded_report"


def read(path):
    return json.loads(path.read_text())


def count(value):
    return f"{value:,}".replace(",", "{,}")


def metric(value, digits):
    if value is None:
        return "---"
    decimal, has_exponent, exponent = f"{value:.{digits}g}".partition("e")
    return rf"${decimal}\times10^{{{int(exponent)}}}$" if has_exponent else f"${decimal}$"


def percent(value):
    return "---" if value is None else rf"${value:+.2f}\%$"


def ids(values):
    return ", ".join(str(value).removeprefix("slice") for value in values)


def table(columns, heading, rows, caption, label=None):
    end = r" \\"
    return "\n".join([
        r"\begin{table}[!htbp]", r"\centering", rf"\caption{{{caption}}}",
        *([rf"\label{{{label}}}"] if label else []),
        r"\small", r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{@{{}}{columns}@{{}}}}", r"\toprule",
        " & ".join(heading) + end, r"\midrule",
        *(" & ".join(row) + end for row in rows),
        r"\bottomrule", r"\end{tabular}", r"\end{table}",
    ])


def require_terminal(summary, progress):
    coverage = summary["coverage"]
    expected = coverage["expected_physical_arms"]
    evaluation = {key: row for key, row in progress["runs"].items() if key.startswith("evaluation_")}
    if len(evaluation) != expected or any(
        row["status"] not in ("completed", "failed") for row in evaluation.values()
    ):
        raise ValueError("The expanded report requires terminal evaluation runs")
    completed = sum(row["status"] == "completed" for row in evaluation.values())
    if coverage["physical_arm_status_counts"].get("completed", 0) != completed:
        raise ValueError("The saved analysis does not match the completed evaluation runs")
    failed = sum(row["status"] == "failed" for row in evaluation.values())
    return completed, failed


def protocol(design, selection, root):
    def setting(trial):
        if "refresh_period" in trial:
            return f"({trial['rate']:g}, {trial['refresh_period']})"
        if "inner_steps" in trial:
            return f"({trial['rate']:g}, {trial['inner_steps']})"
        return f"{trial['rate']:g}"

    rows = []
    upper_edge, lower_edge = [], []
    for variant, trials in design["development_trials"].items():
        chosen = trials[selection["selected"][variant]]
        rows.append((METHODS[variant], ", ".join(map(setting, trials)), setting(chosen)))
        rates = {trial["rate"] for trial in trials}
        if len(rates) > 1 and chosen["rate"] == max(rates):
            upper_edge.append(METHODS[variant])
        if len(rates) > 1 and chosen["rate"] == min(rates):
            lower_edge.append(METHODS[variant])
    edge_notes = []
    for side, methods in (("upper", upper_edge), ("lower", lower_edge)):
        if methods:
            edge_notes.append(f"The selected rate lies at the {side} grid edge for {', '.join(methods)}.")
    if edge_notes:
        edge_notes.append("The comparison reflects the settings covered by these grids.")
    grid = table(
        "lrl", ("Variant", "Candidates", "Selected"), rows,
        r"Development grid and frozen evaluation settings. Pairs list the step "
        r"multiplier followed by the refresh period for SPRING--SARAH or the "
        r"inner-loop length for ProxSARAH.",
    )
    eval_rows = ((mix, ids(sid for sid, value in design["evaluation_slice_mixes"].items() if value == mix))
                 for mix in sorted(set(design["evaluation_slice_mixes"].values())))
    slice_table = table(
        "ll", ("Sample mix", "Evaluation slices"), eval_rows,
        "Fresh evaluation slices. All belong to the post-replacement detector regime.",
        label="tab:sample-slices",
    )
    calibration_rows = []
    for phase in ("development", "evaluation"):
        seed = design[f"{phase}_calibration_seed"]
        for sid in design[f"{phase}_slices"]:
            metadata = next((root / "runs").glob(f"{phase}_{sid}_360_*/metadata.json"))
            saved = read(metadata)
            if saved["config"]["calibration_seed"] != seed:
                raise ValueError(f"Unexpected calibration seed in {metadata}")
            cap_u, cap_v = saved["caps"]
            lu, lv = saved["curvature"]
            calibration_rows.append((ids([sid]), metric(cap_u, 5), metric(cap_v, 5),
                                     metric(lu, 5), metric(lv, 5), metric(1 / max(lu, lv), 5)))
    calibration_table = table(
        "lrrrrr", ("Slice", r"$R_U$", r"$R_V$", r"$L_U$", r"$L_V$", r"$\mu_0$"),
        calibration_rows,
        "Calibrated spectral caps and empirical curvature scales. Values are rounded for display. "
        "Each slice shares these settings across methods, seeds, and training-view counts.",
        label="tab:calibration",
    )
    return rf"""
\section{{Data and evaluation protocol}}
The inputs are measured fan-beam sinograms from 2DeteCT \cite{{detect}}.
A sinogram stacks the detector readings over the angular views of one slice.
We fit the Mode~1 low-dose measurements. The image reference is the published
Mode~2 reconstruction from a separate high-dose acquisition of the same slice.
The two sample mixes label different specimen tubes, with three evaluation
cross sections from each tube.

Each raw Mode~1 sinogram has \(3{{,}}601\) views and \(1{{,}}912\) detector cells.
We discard the repeated final view and sum adjacent detector cells. A dark
field measures detector offset and two flat fields measure unobstructed
illumination. After the same detector binning, let \(S_i,D_{{\rm dark}}\)
and \(F_{{\rm flat}}\) denote the measurement, dark field and mean flat field.
The processed measurement before detector reversal is
\[
  -\log\!\left(\max\!\left\{{
    \frac{{S_i-D_{{\rm dark}}}}{{F_{{\rm flat}}-D_{{\rm dark}}}},10^{{-6}}
  \right\}}\right).
\]
We reverse the detector axis to match the DeepInverse convention
\cite{{deepinv}}. All eight selected acquisitions have zero transmission
clipping counts. Each processed sinogram has \(3{{,}}600\times956\) entries.

ASTRA 2.5.0 applies the CPU \texttt{{line\_fanflat}} operator in millimeters
\cite{{astra}}. The source-to-origin distance is \(431.019989\)~mm, the
origin-to-detector distance is \(97.980499\)~mm and the binned detector
spacing is \(0.1496\)~mm. The square field has width \(116.528143\)~mm and
center \((-0.113797,-0.113797)\)~mm. View \(i\) has angle
\(\pi-2\pi i/3600\). The factors use FP64 arithmetic and the projector
uses FP32. To align the published \(1024\times1024\) reference with the
fitting grid, we flip it vertically, divide by its native voxel width
\(0.113797\)~mm, then average \(2\times2\) pixel blocks.

Views have indices \(0,\ldots,3599\). The training set uses residue zero modulo
ten for \(360\) views and residues zero, three, and six for \(1{{,}}080\)
views. Both settings reserve residue nine for \(360\) held-out views. The
\(120\) indices divisible by \(30\) calibrate the same warm start for both
training sizes. Calibration views belong to both training sets. Held-out
views are disjoint from training and calibration. Increasing the number of
training views changes the finite sum while preserving the image and factor
dimensions.

The development slices are {ids(design["development_slices"])}.
Table~\ref{{tab:sample-slices}} lists the evaluation slices. Each slice is
reconstructed separately. The stochastic methods are paired at seed labels
{ids(design["evaluation_seeds"])}. Development uses oracle seed label
{design["development_oracle_seed"]}. Each oracle generator
uses its seed label plus \(100{{,}}000\). Calibration uses seed
{design["development_calibration_seed"]} in development and
{design["evaluation_calibration_seed"]} in evaluation. It is fixed within a
slice across methods, oracle seeds, and view counts. Full-batch SFRAMES and PALM
each run once per slice and training size and provide shared references for
the stochastic pairs.

{slice_table}

\section{{Calibration and initialization}}
Let \(P_J\) stack the \(120\) calibration views. We divide
\((P_J^\ast y_J)_+\) entrywise by \(P_J^\ast P_J\mathbf1\), floored at
\(10^{{-3}}\) times its maximum. From a thin SVD of this image, each
rank-\(64\) factor column uses the positive or negative singular-vector
parts with the larger product of norms, with the corresponding singular-value
scaling. We add independent uniform \([0,1)\) entries multiplied by
\(10^{{-3}}\) times the RMS of each factor. Finally, we fit a scalar
amplitude by least squares on the calibration views, with lower bound
\(10^{{-6}}\), and multiply both factors by its square root.

Write the resulting factors as \(U_{{\rm cal}},V_{{\rm cal}}\).
The spectral radii are
\(R_U=2\|U_{{\rm cal}}\|_{{\rm op}}\) and
\(R_V=2\|V_{{\rm cal}}\|_{{\rm op}}\).
All expanded evaluation runs start at these common feasible factors in
\(C\cap D\). The earlier LMO initialization outside \(D\) belongs to the
separate pilot ablation. The expanded trajectories therefore begin at
\(d_D=0\), and the first SFRAMES update leaves \(D\).

Starting with a normalized all-ones image, ten power iterations on
\(P_J^\ast P_J\) estimate its largest eigenvalue. Divide this estimate by
\(120d\) to obtain \(\widehat\lambda\). The empirical curvature scales
and smoothing prefactor are
\[
  L_U=\widehat\lambda\|V_{{\rm cal}}\|_{{\rm op}}^2,\qquad
  L_V=\widehat\lambda\|U_{{\rm cal}}\|_{{\rm op}}^2,\qquad
  \mu_0=1/\max(L_U,L_V).
\]
These scales remain fixed during optimization.

{calibration_table}

\section{{Optimizers and rate selection}}
For \(k\ge0\), both SFRAMES variants use
\[
  \gamma_k=\gamma_0/(k+1)^{{3/4}},\qquad
  \rho_k=1/(k+1)^{{1/2}},\qquad
  \mu_k=\mu_0/(k+1)^{{1/4}}.
\]
Minibatch SFRAMES draws {design["stochastic_batch"]}
training views uniformly without replacement at each step, independently across
steps. Its full-batch
counterpart uses every training view and tunes its own \(\gamma_0\). The
momentum starts with the first gradient and subsequently satisfies
\[
  m_k=(1-\rho_k)m_{{k-1}}+\rho_k\widehat\nabla f_{{B_k}}(x_k),\qquad k\ge1.
\]
Both add the orthant Moreau gradient after this averaging and update by
\[
  s_k=\operatorname{{LMO}}_C\!\left(m_k+
       \frac{{x_k-\Pi_D(x_k)}}{{\mu_k}}\right),\qquad
  x_{{k+1}}=(1-\gamma_k)x_k+\gamma_k s_k.
\]
Here \(\Pi_D\) clips negative entries, so the Moreau term is the negative
part of each factor divided by \(\mu_k\) \cite{{sframes}}.
For a block direction with thin singular
value decomposition \(G=A\operatorname{{diag}}(\sigma)B^\top\), the exact
oracle is \(-RA_+B_+^\top\), with singular vectors restricted to positive
singular values. Its optimal value is \(-R\|G\|_*\), where the star denotes
the nuclear norm. The FP64 implementation retains singular values exceeding
\(\varepsilon_{{\rm mach}}\max(\operatorname{{shape}}G)\sigma_{{\max}}(G)\)
and returns zero for a zero direction. Discarding small singular values gives
an oracle support-gap bound of \(R\) times their sum. Each SFRAMES update
uses two \(512\times64\) thin SVDs, one for each factor.

SPRING--SARAH updates \(U\), then \(V\) using the new \(U\), with steps
\(a/L_U\) and \(a/L_V\). Each block independently refreshes its estimator
with probability \(1/q\) using all training views. Both blocks begin with
full gradients at their respective query points. Between refreshes, a
\(64\)-view SARAH difference updates each estimator \cite{{spring}}.
ProxSARAH uses a joint step \(a/(L_U+L_V)\), a \(256\)-view snapshot,
then \(m\) recursive updates with \(64\)-view differences before the next
snapshot. Its relaxation is \(0.8\) \cite{{proxsarah}}. Each recursive
difference evaluates the same sampled views at its two query points. All
subset samples are uniform without replacement within a batch.
PALM uses every training view per alternating block step with the same
curvature scaling as SPRING--SARAH \cite{{palm}}. Proximal block steps
project onto the spectral-ball and orthant intersection using a single-set
shortcut when feasible and Dykstra iterations with a numerical certificate otherwise.
The individual projections clip negative entries or clip singular values at
the radius. Dykstra uses a requested distance tolerance
\(10^{{-8}}/(k+1)^{{1.1}}\), at most \(2{{,}}000\) inner iterations,
and certificate checks after the first iteration, every five iterations, and
at the final iteration. The certificate uses a primal--dual gap with a
roundoff allowance. A reported floating-point floor can terminate the
projection before the requested tolerance is attained.
The work count includes both evaluations in each recursive difference and all
views used for refreshes and snapshots.

{grid}

We select each setting by the smallest median ratio of repaired training loss
to initial loss across both development slices and both training sizes. The
ratio uses the last iterate within \({count(design["development_view_budget"])}\)
forward and adjoint views in each direction. This gives
\(88\) development runs across two slices, two training sizes and
\(22\) candidate settings. Held-out loss and reference-image error are
reporting metrics.
{" ".join(edge_notes)}
"""


def results(summary, completed, failed):
    view = count(summary["budgets"]["forward_view_cap"])
    work_rows, metric_rows, paired_rows = [], [], []
    for n_train, cohort in summary["cohorts"].items():
        current = cohort["view_cap"]
        marginal = current["matched_all_methods"]
        for variant, row in marginal["median_of_slice_medians"].items():
            work_rows.append((count(int(n_train)), METHODS[variant],
                              metric(row["iteration"], 6), metric(row["gradient_calls"], 6),
                              metric(row["forward_views"], 6), metric(row["backward_views"], 6)))
            metric_rows.append((count(int(n_train)), METHODS[variant],
                                metric(row["raw_loss"], 4), metric(row["feasible_loss"], 4),
                                metric(row["validation_loss"], 4), metric(row["reference_relative_error"], 4),
                                metric(row["c_violation"], 3), metric(row["d_distance"], 3)))
        for variant, result in current["sframes_vs"].items():
            metrics = result["metrics"]
            training = metrics["feasible_loss"]["percent_difference"]
            heldout = metrics["validation_loss"]["percent_difference"]
            paired_rows.append((count(int(n_train)), METHODS[variant], str(result["seed_pairs"]),
                                percent(training["median_across_slices"]),
                                percent(heldout["median_across_slices"]),
                                f'{training["slices_favoring_sframes"]}/{training["slices"]}'))
    full_batch_effects = []
    for n_train, cohort in summary["cohorts"].items():
        effect = cohort["view_cap"]["sframes_vs"]["sframes_full"]["metrics"]["feasible_loss"]
        reduction = -effect["percent_difference"]["median_across_slices"]
        full_batch_effects.append(rf"\({reduction:.1f}\%\) with \({count(int(n_train))}\) training views")
    work_table = table("llrrrr", (r"$N$", "Variant", "Iterations", "Gradient calls", "Forward views", "Adjoint views"),
        work_rows,
        rf"Work at the projection budget of \({view}\) views per direction. Values are medians across slice medians. The last saved iterate can fall below the cap when the next complete update would exceed it.",
        label="tab:work")
    metrics_table = table("llrrrrrr", (r"$N$", "Variant", "Raw", "Repaired", "Held-out", "Ref. error", r"$d_C$", r"$d_D$"),
        metric_rows,
        r"Losses, image error and feasibility at the saved within-budget iterates in Table~\ref{tab:work}. All losses use the same detector and view normalization. Values are medians across slice medians.",
        label="tab:metrics")
    paired_table = table("llrrrr", (r"$N$", "Comparator", "Pairs", "Repaired", "Held-out", "Slices won"),
        paired_rows,
        rf"Median paired percent differences for minibatch SFRAMES at \({view}\) views per direction. Negative values favor minibatch SFRAMES. Slices won counts slices with lower median paired repaired loss.",
        label="tab:paired")
    blocks = [rf"""
\section{{Evaluation results}}
{completed} of {completed + failed} evaluation runs completed. The comparison
uses the last saved iterate with at most \({view}\) forward views and
\({view}\) adjoint views. A run stops after the first complete update that
reaches the work budget. When an update crosses the budget, the previous
iterate supplies the within-budget comparison. The tables use saved probes
without interpolation.

An iteration completes one update of both factors. A gradient call applies
one forward operator and one adjoint to the requested batch and returns either
a joint gradient or one block gradient. Both evaluations in each SARAH
difference and all refresh and snapshot views contribute to the work count.
Gradient-call counts therefore describe the implementation, while the two
view counts measure how much projection data the calls process.

Raw loss uses the actual factors. Repaired loss, held-out loss and reference
error use the image formed after taking each factor's positive part and
scaling it down if its operator norm exceeds its cap. Spectral violation is
\(d_C=\max\{{(\|U\|_{{\rm op}}-R_U)_+,
(\|V\|_{{\rm op}}-R_V)_+\}}\). Distance to \(D\) is
\(d_D=(\|\min(U,0)\|_F^2+\|\min(V,0)\|_F^2)^{{1/2}}\).
Reference error is the Frobenius error relative to the aligned Mode~2 image,
divided by that image's Frobenius norm.

For absolute metrics we take the median over seeds within each slice and then
the median across slices at each training size. Percent differences first
compare paired runs using the comparator in the denominator and then apply
the same aggregation. Each training size has \(18\) paired seed cases from
six slices. Full-batch SFRAMES and PALM provide shared deterministic
references. These summaries are descriptive because the three slices from
each sample mix share a sample tube.

{work_table}

{metrics_table}

\clearpage
{paired_table}

At the common work cap, minibatch SFRAMES completes \(270\) updates at both
training sizes. Full-batch SFRAMES completes \(48\) updates with \(360\)
training views and \(16\) with \(1{{,}}080\) views. The median paired reduction
in repaired training loss relative to full-batch SFRAMES is
{" and ".join(full_batch_effects)}.
Minibatch SFRAMES has lower median paired repaired loss than each comparator
on all six evaluation slices at both training sizes. Its held-out loss also
improves relative to every comparator in the overall slice summaries.
Figure~\ref{{fig:paired-effects}} shows the individual slice effects so that
the variability across slices remains visible.

Figures~\ref{{fig:training-trajectories}} and~\ref{{fig:quality-trajectories}}
show the saved loss and feasibility trajectories. These expanded runs use
the shared feasible warm start described above. Both SFRAMES variants leave
\(D\) after the first update and subsequently reduce their orthant distance.
Their loss after feasibility repair remains close to their raw loss at the
work cap. Every saved evaluation probe has zero spectral violation, and the
projection baselines retain nonnegative factors. At every evaluation baseline
projection, clipping to the orthant already satisfies the spectral cap, so
the single-set shortcut applies. The separately
tuned rates and the boundary selections in the development grid limit the
comparison to these practical settings.
"""]
    return "\n\n".join(blocks)


OPENING = r"""
\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[letterpaper,margin=0.80in]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx}
\usepackage[hidelinks]{hyperref}
\setlength{\emergencystretch}{2em}
\makeatletter
\setlength{\@fptop}{0pt}
\setlength{\@fpsep}{18pt}
\makeatother
\hypersetup{pdftitle={Spectrally constrained stochastic factorization for low-dose CT}}
\begin{document}
\begin{center}
  \Large\bfseries Spectrally constrained stochastic factorization for low-dose CT
\end{center}

\section{Low-dose CT reconstruction model}
We reconstruct a two-dimensional attenuation image from X-ray measurements
taken at several angles. Attenuation describes how strongly material reduces
the transmitted intensity per unit distance. At one angular position, called
a view, an X-ray source illuminates a line of detector bins. The source-to-bin
rays spread out in a fan. Under the Beer--Lambert model, the negative logarithm
of intensity relative to unobstructed illumination approximates a line integral
of attenuation \cite{detect}.

Treat \(X_{pq}\) as the constant attenuation in pixel \((p,q)\) on a
\(512\times512\) grid. If \(\ell_{ij,pq}\) is the length in millimeters
of ray \(j\) in view \(i\) within that pixel, the calibrated forward map is
\[
  P_i\colon\mathbb R^{512\times512}\longrightarrow\mathbb R^d,\qquad
  [P_i(X)]_j=\sum_{p,q}\ell_{ij,pq}X_{pq},\qquad d=956.
\]
The fixed source and detector geometry determines these weights. ASTRA traces
the rays to apply this map to selected views \cite{astra}.

We write the image as \(X=UV^\top\) with \(U,V\in\mathbb R^{512\times64}\),
restricting its rank to at most \(64\). All views constrain the same factors.
For measured view data \(y_i\in\mathbb R^d\), the unweighted loss is
\[
  f(U,V) = \frac{1}{N}\sum_{i\in\mathcal I_{\rm train}}
    \frac{\|P_i(UV^\top)-y_i\|_2^2}{2d},
  \qquad N=|\mathcal I_{\rm train}|.
\]
The loss is quadratic in the image and nonconvex in the factors. To bound
factor scale, use spectral-norm caps
\(C=\{(U,V):\|U\|_{\rm op}\le R_U,\ \|V\|_{\rm op}\le R_V\}\).
Each radius is twice the operator norm of the corresponding calibrated warm
factor. Nonnegative factors give a nonnegative attenuation image, so define
\(D=\{(U,V):U\ge0,\ V\ge0\}\) entrywise. Write \(g=\iota_D\) for the
function that is zero on \(D\) and infinite elsewhere. Since \(g\) acts
directly on the factors, \(T\) is the identity. With \(x=(U,V)\), the
problem has the SFRAMES form
\[
  \min_{x\in C}\{f(x)+g(Tx)\}.
\]
The component gradients are Lipschitz on the compact region \(C\). Strictly
positive sufficiently small factors lie in the interiors of both \(C\) and
\(D\).

The adjoint \(P_i^\ast\), called backprojection, distributes each detector
residual to the pixels crossed by its ray using the same ray-length weights.
For a uniform minibatch \(B\) of training views, define
\[
  E_B=\frac{1}{|B|d}\sum_{i\in B}P_i^\ast\bigl(P_i(UV^\top)-y_i\bigr),
  \qquad \widehat\nabla f_B(U,V)=(E_BV,E_B^\top U).
\]
This estimator is unbiased conditional on the current factors. One joint
gradient invocation applies one forward projection and one adjoint over
\(B\). An alternating method requests one of the two factor gradients at
its block query point.
"""


ENDING = r"""
\clearpage
\section{Optimization trajectories}
Figures~\ref{fig:training-trajectories} and~\ref{fig:quality-trajectories}
show all five methods at both training-view counts. For display, each saved
seed trajectory is linearly interpolated between diagnostic probes on a common
work grid. We take the pointwise median over seeds within each slice, then the
median across the six slices. Full-batch SFRAMES and PALM contribute one run
per slice and training size. Shading shows the middle \(50\%\) of slice
medians for the two SFRAMES variants. These are descriptive trajectories,
with no extrapolation beyond recorded work. Numerical tables use actual
saved within-budget iterates, so a displayed interpolated endpoint can differ
slightly from a table entry.

Probes were scheduled every \(4{,}320\) forward views in evaluation and
every \(2{,}160\) in development, with extra probes at the first update and
around budget crossings. The spacing limits what can be inferred about
behavior between probes. Forward and adjoint view counts agree at every
saved probe.

\begin{figure}[!htbp]
  \centering
  \includegraphics[width=\linewidth]{trajectory_training.pdf}
  \caption{Raw and repaired training loss versus processed views in each
  projection direction. Each loss is divided by its own slice and training
  split's common initial loss before aggregation. Rows compare \(360\) and
  \(1{,}080\) training views.}
  \label{fig:training-trajectories}
\end{figure}
\clearpage
\begin{figure}[!htbp]
  \centering
  \includegraphics[width=\linewidth]{trajectory_quality.pdf}
  \caption{Held-out loss after factor repair and distance of the raw factors
  to the nonnegative orthant \(D\), versus projection work. The feasibility
  axis uses a symmetric logarithmic scale with a linear region around zero.
  The shared warm start has zero distance to \(D\). SFRAMES leaves \(D\)
  after its first update. The three projection baselines coincide at zero
  distance throughout.}
  \label{fig:quality-trajectories}
\end{figure}
\clearpage
\begin{figure}[!htbp]
  \centering
  \includegraphics[width=\linewidth]{comparison.pdf}
  \caption{Paired effects for minibatch SFRAMES at the saved within-budget
  iterates. Small marks give the median paired effect over seeds for each
  slice. Circles and triangles distinguish the two sample mixes, and diamonds
  give the median across slices. Negative values favor minibatch SFRAMES.}
  \label{fig:paired-effects}
\end{figure}
\clearpage
\section{Reproduction and scope}
The expanded campaign contains \(88\) development runs and \(132\) fresh
evaluation runs. Per-run configurations and original source, data and
calibration hashes are stored under \texttt{runs/}. The files
\texttt{manifest.json}, \texttt{selection.json} and
\texttt{calibration.json} record the design, frozen rate choices and
calibration checksums. Data sidecars \texttt{data/slice*.json} record the
Zenodo sources, preprocessing conventions and original TIFF checksums.
\texttt{reports/summary.json} records work-based endpoint and paired
summaries. \texttt{analyze.py}, \texttt{plot\_trajectories.py} and
\texttt{report\_latex.py} regenerate the analysis, figures and this document
from saved results.

The comparisons measure optimization progress under the recorded operator
work budget. Concurrent execution on the host does not provide a controlled
timing benchmark. This report uses iterations, gradient-oracle calls and
projection work to characterize the completed runs. The problem size, rank,
initialization, smoothing schedule and projection settings are fixed in this
campaign. Training-view count, stochastic seed and the development rate and
refresh grids supply the reported variations.

\begin{thebibliography}{9}
\bibitem{sframes}
\emph{Improved convergence and stochastic estimation for the Frank--Wolfe
with Moreau Envelope Smoothing (FraMES) algorithm}. Local manuscript,
27 September 2026. Corollary~2.10 and Proposition~2.11.
\bibitem{detect}
M.~B.~Kiss et al.
\emph{2DeteCT: A large 2D expandable, trainable, experimental Computed Tomography
dataset for machine learning}. Scientific Data 10, article 576, 2023.
\href{https://doi.org/10.1038/s41597-023-02484-6}{DOI}.
\bibitem{deepinv}
DeepInverse.
\href{https://deepinv.org/auto_examples/external-libraries/demo_astra_2detect.html}
{Reconstruct real CT sinograms with the 2DeteCT benchmark}.
\bibitem{astra}
ASTRA Toolbox. \href{https://astra-toolbox.com/docs/proj2d.html#fan-beam}{2D Projectors}.
\bibitem{spring}
D.~Driggs, J.~Tang, J.~Liang, M.~E.~Davies, and C.-B.~Sch\"onlieb.
\emph{A Stochastic Proximal Alternating Method for Nonsmooth and Nonconvex Optimization}.
SIAM Journal on Imaging Sciences 14(4), 1932--1970, 2021.
\href{https://doi.org/10.1137/20M1387213}{DOI}.
\bibitem{proxsarah}
N.~H.~Pham, L.~M.~Nguyen, D.~T.~Phan, and Q.~Tran-Dinh.
\emph{ProxSARAH: An Efficient Algorithmic Framework for Stochastic Composite Nonconvex Optimization}.
Journal of Machine Learning Research 21(110), 1--48, 2020.
\href{https://www.jmlr.org/papers/v21/19-248.html}{Article}.
\bibitem{palm}
J.~Bolte, S.~Sabach, and M.~Teboulle.
\emph{Proximal alternating linearized minimization for nonconvex and nonsmooth problems}.
Mathematical Programming 146, 459--494, 2014.
\href{https://doi.org/10.1007/s10107-013-0701-9}{DOI}.
\end{thebibliography}
\end{document}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "reports"
    subprocess.run([sys.executable, str(ROOT / "analyze.py"), "--root", str(root)], check=True)
    subprocess.run([sys.executable, str(ROOT / "plot_trajectories.py"), "--root", str(root)], check=True)
    summary = read(output / "summary.json")
    completed, failed = require_terminal(summary, read(root / "progress.json"))
    document = "\n".join((OPENING, protocol(summary["design"], summary["selection"], root),
                          results(summary, completed, failed), ENDING))
    with TemporaryDirectory(prefix="ct-expanded-latex-") as temp:
        stage = Path(temp)
        source = stage / f"{STEM}.tex"
        source.write_text(document)
        for name in ("comparison.pdf", "trajectory_training.pdf", "trajectory_quality.pdf"):
            shutil.copyfile(output / name, stage / name)
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", source.name],
                       cwd=stage, check=True)
        for suffix in ("tex", "pdf"):
            name = f"{STEM}.{suffix}"
            pending = output / f"{name}.tmp"
            shutil.copyfile(stage / name, pending)
            pending.replace(output / name)
    print(output / f"{STEM}.pdf")


if __name__ == "__main__":
    main()
