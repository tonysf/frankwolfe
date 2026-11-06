"""Render the completed spectral-ball CT comparison and signed-start pairs."""

import argparse
from collections import defaultdict
import importlib.util
import json
from pathlib import Path
import shutil
from statistics import median
import subprocess
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages


HERE = Path(__file__).resolve().parent
WORK = ("iteration", "forward_views", "backward_views", "gradient_calls")
METRICS = ("raw_loss", "feasible_loss", "validation_loss", "reference_relative_error")


def parent_report():
    spec = importlib.util.spec_from_file_location("ct_plotting", HERE.parent / "report.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path):
    return json.loads(path.read_text())


def load_runs(root, base):
    progress_path = root / "progress.json"
    progress = read_json(progress_path)["runs"] if progress_path.exists() else {}
    failures = {run_id: item for run_id, item in progress.items() if item["status"] == "failed"}
    runs = []
    for run in base.read_runs(root):
        if run["run_id"] in failures:
            continue
        directory = Path(run["path"])
        if run["status"] != "completed":
            raise ValueError(f"Run {run['run_id']} is {run['status']}")
        run["metadata"] = read_json(directory / "metadata.json")
        saved_rows = read_json(directory / "trace.json")
        for probe, row in zip(run["rows"], saved_rows, strict=True):
            probe.update(row)
        runs.append(run)
    return runs, failures


def short_id(slice_id):
    return slice_id.removeprefix("slice")


def number(n):
    return f"{n:,}".replace(",", "{,}")


def scientific(value):
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def run_pairs(runs):
    by_id = {run["run_id"]: run for run in runs}
    pairs = []
    for cold in runs:
        config = cold["config"]
        if config["initialization"] != "gaussian_spectral_lmo":
            continue
        warm = by_id.get(config["paired_run_id"])
        if warm is None:
            continue
        for field in ("optimizer", "caps", "curvature", "data_sha256",
                      "train_view_ids", "validation_view_ids"):
            if cold["metadata"][field] != warm["metadata"][field]:
                raise ValueError(f"Paired runs differ in {field} for {warm['run_id']}")
        if cold["rows"][0]["d_distance"] <= 0 or warm["rows"][0]["d_distance"] > 1e-10:
            raise ValueError(f"Unexpected initial D distances for {warm['run_id']}")
        for warm_row, cold_row in zip(warm["rows"], cold["rows"], strict=True):
            if any(warm_row[field] != cold_row[field] for field in WORK):
                raise ValueError(f"Paired work differs for {warm['run_id']}")
        pairs.append((warm, cold))
    if len({warm["run_id"] for warm, _ in pairs}) != len(pairs):
        raise ValueError("Duplicate SFRAMES signed-start pairs")
    return sorted(pairs, key=lambda pair: (
        pair[0]["config"]["phase"], pair[0]["config"]["slice_id"],
        pair[0]["config"]["n_train"], pair[0]["config"]["rate"]))


def select_probe(run, key, cap):
    return next(row for row in reversed(run["rows"]) if row[key] <= cap)


def calibration_rows(evaluation, development):
    by_slice = {}
    for run in development + evaluation:
        by_slice.setdefault(run["config"]["slice_id"], run["metadata"])
    rows = []
    for slice_id, metadata in sorted(by_slice.items()):
        cap_u, cap_v = metadata["caps"]
        curvature_u, curvature_v = metadata["curvature"]
        mu0 = 1.0 / max(curvature_u, curvature_v)
        values = (cap_u, cap_v, curvature_u, curvature_v, mu0)
        rows.append(short_id(slice_id) + " & " + " & ".join(f"{x:.8f}" for x in values) + r" \\")
    return "\n".join(rows)


def rate_rows(evaluation, candidate_rates, methods):
    step_rules = {
        "sframes": r"\gamma_0=a,\ \rho_0=1",
        "spring_sarah": r"(\eta_U,\eta_V)=(a/L_U,a/L_V)",
        "prox_sarah": r"\eta=a/(L_U+L_V)",
        "palm": r"(\eta_U,\eta_V)=(a/L_U,a/L_V)",
    }
    rows = []
    for method, (label, _) in methods.items():
        candidates = candidate_rates[method]
        selected, = {r["config"]["rate"] for r in evaluation if r["config"]["method"] == method}
        rates = r",\ ".join(f"{rate:.2f}" for rate in candidates)
        rows.append(rf"{label} & \({rates}\) & \({selected:.2f}\) & \({step_rules[method]}\) \\")
    return "\n".join(rows)


def result_table(summary, methods, view_cap):
    rows = []
    for method, (label, _) in methods.items():
        data = summary[method]
        values = [f"{data[key]:.6f}" for key in METRICS[:-1]]
        values.append(rf"{100*data[METRICS[-1]]:.2f}\%")
        rows.append(" & ".join([label, *values]) + r" \\")
    return (
        r"\begin{table}[htbp]" "\n"
        r"\centering" "\n"
        rf"\caption{{Medians at the last saved probe within \({number(view_cap)}\) forward "
        "views and the same number of backprojected views.}\n"
        r"\label{tab:matched-methods}" "\n"
        r"\begin{tabular}{@{}lrrrr@{}}" "\n"
        r"\toprule" "\n"
        r"Method & Raw training & Repaired training & Held-out & Reference error \\" "\n"
        r"\midrule" "\n" + "\n".join(rows) + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )


def ablation_table(pairs):
    rows = []
    for warm, cold in pairs:
        config = warm["config"]
        a, b = warm["rows"][-1], cold["rows"][-1]
        values = [short_id(config["slice_id"]), f"{config['n_train']:,}", f"{config['rate']:.2f}",
                  f"{cold['rows'][0]['d_distance']:.4f}", f"{a['feasible_loss']:.6f}",
                  f"{b['feasible_loss']:.6f}", f"{a['d_distance']:.6f}", f"{b['d_distance']:.6f}"]
        rows.append(" & ".join(values) + r" \\")
    return (
        r"\begin{table}[htbp]" "\n"
        r"\centering" "\n"
        r"\caption{Paired SFRAMES endpoints. The original feasible initialization has zero distance to \(D\).}" "\n"
        r"\label{tab:lmo-start}" "\n"
        r"\small" "\n"
        r"\begin{tabular}{@{}lrrrrrrr@{}}" "\n"
        r"\toprule" "\n"
        r" & & & & \multicolumn{2}{c}{Repaired loss} & \multicolumn{2}{c}{Final distance to \(D\)} \\" "\n"
        r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}" "\n"
        r"Slice & Views & \(\gamma_0\) & LMO initial \(D\) dist. & Warm & LMO & Warm & LMO \\" "\n"
        r"\midrule" "\n" + "\n".join(rows) + "\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}"
    )


def results_text(cohorts, development, pairs, failures, view_cap, methods):
    evaluation = [run for cohort in cohorts.values() for run in cohort]
    view_probes = {r["run_id"]: select_probe(r, "forward_views", view_cap) for r in evaluation}
    time_cap = min(100.0, min(r["rows"][-1]["optimizer_seconds"] for r in evaluation))
    time_probes = {r["run_id"]: select_probe(r, "optimizer_seconds", time_cap) for r in evaluation}
    medians = {}
    time_medians = {}
    for method in methods:
        ids = [r["run_id"] for r in evaluation if r["config"]["method"] == method]
        medians[method] = {field: median(view_probes[i][field] for i in ids) for field in METRICS}
        time_medians[method] = median(time_probes[i]["feasible_loss"] for i in ids)

    def wins(probes):
        return sum(
            next(probes[r["run_id"]]["feasible_loss"] for r in cohort
                 if r["config"]["method"] == "sframes")
            == min(probes[r["run_id"]]["feasible_loss"] for r in cohort)
            for cohort in cohorts.values()
        )

    pair_changes = [(cold["rows"][-1]["feasible_loss"] / warm["rows"][-1]["feasible_loss"] - 1) * 100
                    for warm, cold in pairs if warm["config"]["phase"] == "evaluation"]
    ends = [r["rows"][-1]["optimizer_seconds"] for r in evaluation]
    blocks = [
        r"\section{Results}\label{sec:results}",
        r"\subsection{Four methods with a common feasible start}",
        f"{len(development)} development runs and {len(evaluation)} evaluation runs "
        "completed. Each comparison uses the last saved probe within its budget. "
        f"SFRAMES attained the lowest repaired training loss in {wins(view_probes)} of "
        f"{len(cohorts)} cases at the projection budget and in {wins(time_probes)} of "
        rf"{len(cohorts)} at \({time_cap:.0f}\) optimizer seconds. Table~\ref{{tab:matched-methods}} "
        r"summarizes the view-budget comparison. Appendix~\ref{app:four_method} gives the saved trajectories.",
        result_table(medians, methods, view_cap),
        rf"At \({time_cap:.0f}\) optimizer seconds, median repaired losses were "
        + ", ".join(rf"\({value:.6f}\) for {methods[m][0]}" for m, value in time_medians.items())
        + rf". Evaluation runs reached their stopping budget in \({min(ends):.1f}\) to "
          rf"\({max(ends):.1f}\) optimizer seconds.",
    ]
    projection_rows = [r["rows"][-1] for r in evaluation if r["config"]["method"] != "sframes"]
    projection_fields = ("projection_calls", "projection_dykstra_calls", "projection_floor_count",
                         "projection_seconds", "max_projection_distance_bound")
    if all(set(projection_fields) <= row.keys() for row in projection_rows):
        projection_count = sum(row["projection_calls"] for row in projection_rows)
        dykstra_count = sum(row["projection_dykstra_calls"] for row in projection_rows)
        floor_count = sum(row["projection_floor_count"] for row in projection_rows)
        median_seconds = median(row["projection_seconds"] for row in projection_rows)
        max_distance = max(row["max_projection_distance_bound"] for row in projection_rows)
        blocks.append(
            f"The proximal evaluation arms made {projection_count:,} factor projections. "
            f"Of these, {dykstra_count:,} entered Dykstra iterations and {floor_count:,} "
            f"terminated at the numerical floor. Median recorded projection time was "
            rf"{median_seconds:.2f} seconds per arm. The largest reported numerical distance "
            rf"bound was \({scientific(max_distance)}\).")
    if failures:
        counts = {phase: sum(name.startswith(phase) and not name.endswith("_lmo") for name in failures)
                  for phase in ("development", "evaluation")}
        cold_failed = len(failures) - sum(counts.values())
        blocks.append(
            r"\paragraph{Failed attempts.} "
            f"{counts['development']} development, {counts['evaluation']} evaluation, and "
            f"{cold_failed} signed-start attempts failed. The tables summarize completed "
            "comparisons. The accompanying summary records each failure.")
    if pairs:
        cap_excess = max(row["c_violation"] for _, cold in pairs for row in cold["rows"])
        blocks.extend([
            r"\subsection{Paired SFRAMES initialization ablation}",
            "The Gaussian-direction spectral LMO starts are outside the nonnegative orthant. "
            f"All {len(pairs)} completed pairs matched work at each saved probe. "
            rf"The maximum observed spectral-cap excess for the signed-start runs was \({scientific(cap_excess)}\). "
            r"Table~\ref{tab:lmo-start} compares their endpoints.",
            ablation_table(pairs),
        ])
        if pair_changes:
            blocks.append(
                r"The change in repaired loss at the matched evaluation endpoints, computed as "
                rf"LMO-start loss divided by warm-start loss minus one, ranged from \({min(pair_changes):+.2f}\%\) "
                rf"to \({max(pair_changes):+.2f}\%\).")
        blocks.append("The signed LMO atoms change both initial feasibility and the information "
                      "available in the starting factors.")
    blocks.append(
        r"\paragraph{Scope.} The pilot uses one development slice, two evaluation slices, and "
        "one evaluation seed. The four-method results use the same feasible warm start. "
        "The initialization ablation compares SFRAMES starts at frozen settings.")
    return "\n\n".join(blocks), {
        "view_cap": view_cap, "time_cap": time_cap, "view_budget_medians": medians,
        "time_budget_median_repaired_loss": time_medians, "case_probes_by_work": view_probes,
        "case_probes_by_time": time_probes,
    }


def plot_figures(path, directory, cohorts, base):
    lines = []
    with PdfPages(path) as pdf:
        for page, ((slice_id, n_train), runs) in enumerate(sorted(cohorts.items()), start=1):
            base.plot_cohort(pdf, directory, (slice_id, n_train), runs)
            case = f"Slice {short_id(slice_id)} with {n_train:,} training views."
            lines.extend([
                rf"\ctfigure{{{2*page-1}}}{{fig:work_{page}}}{{{case} "
                "Raw and repaired training loss by update count, optimizer time, and projection work.}",
                rf"\ctfigure{{{2*page}}}{{fig:quality_{page}}}{{{case} "
                "Held-out loss, reference-image error, and raw feasibility.}",
            ])
    return "\n".join(lines)


def render(template, values):
    for key, value in values.items():
        token = f"@@{key}@@"
        if token not in template:
            raise ValueError(f"Template is missing {token}")
        template = template.replace(token, value)
    if "@@" in template:
        raise ValueError("Unfilled LaTeX template fields")
    return template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / "reports"
    manifest = read_json(root / "manifest.json")
    base = parent_report()
    runs, failures = load_runs(root, base)
    pairs = run_pairs(runs)
    development = [r for r in runs if r["config"]["phase"] == "development"]
    evaluation = [r for r in runs if r["config"]["phase"] == "evaluation"]
    cohorts = defaultdict(list)
    for run in evaluation:
        config = run["config"]
        cohorts[config["slice_id"], config["n_train"]].append(run)
    design = manifest["design"]
    expected_cases = {(s, n) for s in design["evaluation_slices"] for n in design["evaluation_training_views"]}
    if set(cohorts) != expected_cases or any(
        {r["config"]["method"] for r in case} != set(base.METHODS) or len(case) != len(base.METHODS)
        for case in cohorts.values()
    ):
        raise ValueError("Expected each evaluation slice and view count to have four completed methods")
    dev_budget = design["development_view_budget"]
    eval_budget = design["evaluation_view_budget"]
    dev_stride, = {r["config"]["probe_every"] for r in development}
    eval_stride, = {r["config"]["probe_every"] for r in evaluation}
    result, metrics = results_text(cohorts, development, pairs, failures, eval_budget, base.METHODS)
    template = (HERE / "report_template.tex").read_text()
    names = ("ct_sframes_report.tex", "ct_sframes_report.pdf", "ct_sframes_figures.pdf", "summary.json")
    with TemporaryDirectory(prefix="ct-spectral-report-") as temp:
        stage = Path(temp)
        figures = plot_figures(stage / names[2], stage, cohorts, base)
        text = render(template, {
            "CAP_CALIBRATION": r"Set \(R_U=2\|U_{\rm cal}\|_{\rm op}\) and \(R_V=2\|V_{\rm cal}\|_{\rm op}\).",
            "CALIBRATION_ROWS": calibration_rows(evaluation, development),
            "RATE_ROWS": rate_rows(evaluation, design["development_rates"], base.METHODS),
            "PROJECTION_SETTINGS":
                r"The requested projection-distance tolerance is \(10^{-8}/(k+1)^{1.1}\), "
                "with a check every five Dykstra iterations and at the first and last iterations. "
                "The limit is 2,000 iterations per factor projection.",
            "ABLATION_SETTINGS":
                "The initialization ablation repeats all SFRAMES development rates and the "
                "frozen evaluation rate without selecting a new rate.",
            "WORK_BUDGETS":
                rf"Each run stops after the first completed update that reaches its target of "
                rf"\({number(dev_budget)}\) forward views in development or \({number(eval_budget)}\) "
                r"in evaluation. Save probes at iteration zero, after the first update, upon "
                rf"crossing multiples of \({number(dev_stride)}\) or \({number(eval_stride)}\) views "
                "respectively, and at the endpoint. Forward and backprojected views are recorded "
                "separately. Counts include refreshes and both evaluations in a SARAH difference.",
            "DIAGNOSTIC_REPAIR":
                "For reporting, clip negative entries and scale the clipped block toward zero "
                "when its operator norm exceeds its cap.",
            "RESULTS": result,
            "FIGURES": figures,
        })
        (stage / names[0]).write_text(text)
        (stage / names[3]).write_text(json.dumps(
            {**metrics, "failures": failures,
             "runs": [base.run_summary({k: v for k, v in r.items() if k != "metadata"})
                                for r in runs],
             "pairs": [{"warm": warm["run_id"], "lmo": cold["run_id"], "warm_final": warm["rows"][-1],
                        "lmo_initial": cold["rows"][0], "lmo_final": cold["rows"][-1]}
                       for warm, cold in pairs]}, indent=2, allow_nan=False) + "\n")
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", names[0]],
                       cwd=stage, check=True)
        output.mkdir(parents=True, exist_ok=True)
        for name in names:
            pending = output / (name + ".tmp")
            shutil.copyfile(stage / name, pending)
            pending.replace(output / name)
    print(output.resolve())


if __name__ == "__main__":
    main()
