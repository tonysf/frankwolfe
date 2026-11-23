"""Summarize the initial, single-seed A100 diagnostic candidate panel.

Consumes existing run outputs; never fabricates missing runs or fits slopes.
Use --scope parser-smoke for explicitly labeled CPU parser checks.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLANS = ("product", "mixed_permutation", "concentrated_permutation")
ORACLES = ("anchor", "trace_sketch")
TIMING_KEYS = ("preparation_cache_seconds", "solver_seconds", "repair_seconds", "diagnostic_seconds",
               "serialization_seconds", "final_float64_validation_seconds")
COLORS = {"exact": "#252525", "sketch_q8": "#0072B2", "sketch_q32": "#D55E00", "sketch_q128": "#009E73"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def number(value):
    if value in (None, ""):
        return None
    output = float(value)
    if not math.isfinite(output):
        raise ValueError("Nonfinite numeric result encountered.")
    return output


def fmt(value):
    return "—" if value is None else f"{value:.5g}"


def hashes(paths):
    return {str(path): sha256(path) for path in paths if path.exists()}


def validate_device(device, gpu, scope):
    if scope == "a100-pilot" and (not str(device).startswith("cuda") or "A100" not in str(gpu)):
        raise ValueError(f"Expected recorded CUDA/A100 execution; metadata says device={device}, GPU={gpu}.")


def load_smoothing(root, n, method, expected_steps, scope):
    path = root / "smoothing" / f"n{n}_{method}"
    entry = {"name": f"n{n}_{method}", "n": n, "method": method, "path": str(path), "status": "missing", "issues": []}
    required = [path/name for name in ("config.json", "summary.json", "trace.csv")]
    entry["source_sha256"] = hashes(required + [path/"failure.json"])
    if (path/"failure.json").exists():
        entry.update(status="failed", failure=read_json(path/"failure.json"))
        return entry, None
    missing = [p.name for p in required if not p.exists()]
    if missing:
        entry["issues"].append("Missing " + ", ".join(missing))
        entry["status"] = "missing" if len(missing) == 3 else "incomplete"
        return entry, None
    try:
        config, summary = (read_json(path/name) for name in ("config.json", "summary.json"))
        with (path/"trace.csv").open() as handle:
            trace = list(csv.DictReader(handle))
        arguments, environment = config["arguments"], config["environment"]
        gpu = environment.get("gpu", {}).get("name")
        validate_device(arguments["device"], gpu, scope)
        if config["shape"] != [n, n]:
            raise ValueError("Recorded instance shape differs from directory name.")
        if arguments["steps"] != expected_steps or summary["completed_updates"] != expected_steps:
            raise ValueError("Requested/completed steps differ from the expected pilot cap.")
        if summary["status"] != "completed":
            raise ValueError("Run summary does not report completion.")
        if arguments["seed"] != 0 or arguments["mu_scaled"] != 1.:
            raise ValueError("Pilot requires stochastic seed 0 and m*mu0=1.")
        if config["instance_manifest"]["seed"] != 1000:
            raise ValueError("Pilot requires development problem seed 1000.")
        if arguments["update"] != "fw":
            raise ValueError("Primary diagnostic panel requires FW updates.")
        if method == "exact":
            if arguments["oracle"] != "exact":
                raise ValueError("Exact-gradient directory contains a different oracle.")
        elif arguments["oracle"] != "sketch" or arguments["batch"] != int(method.split("q")[-1]):
            raise ValueError("Sketch directory and recorded oracle/batch disagree.")
        if scope == "a100-pilot" and arguments["dtype"] != "float32":
            raise ValueError("Initial A100 panel requires float32 solver arithmetic.")
        if len(trace) != expected_steps+1:
            raise ValueError("Trace must contain every pre-update point and one terminal point.")
        if [int(row["point_iteration"]) for row in trace] != list(range(expected_steps+1)):
            raise ValueError("Trace iteration indices are missing or duplicated.")
        if any(row["trajectory_average_included"] != "True" for row in trace[:-1]) or trace[-1]["trajectory_average_included"] != "False":
            raise ValueError("Trajectory-average inclusion flags are inconsistent.")
        for key, value in summary["trajectory_means"].items():
            recomputed = float(np.mean([number(row[key]) for row in trace[:-1]]))
            if not np.isclose(recomputed, value, rtol=2e-12, atol=1e-12):
                raise ValueError(f"Saved full-trajectory mean for {key} differs from trace.")
        running = np.cumsum([number(row["true_smoothed_gap"]) for row in trace[:-1]])/np.arange(1, expected_steps+1)
        if not np.allclose(running, [number(row["mean_true_smoothed_gap"]) for row in trace[:-1]], rtol=2e-12, atol=1e-12):
            raise ValueError("Running true-gap mean does not include every pre-update iterate.")
        final = summary["final_float64"]
        actual64 = "rounded_source" in final
        if not actual64:
            message = "Legacy final_float64 fields describe fresh float64 repair, not verified actual saved rounded output."
            if scope == "a100-pilot":
                raise ValueError(message)
            entry["issues"].append(message)
        entry.update(status="completed", recorded_device=arguments["device"], recorded_gpu=gpu,
                     dtype=arguments["dtype"], stochastic_seed=arguments["seed"], problem_seed=config["instance_manifest"]["seed"],
                     batch=arguments["batch"] if method != "exact" else None,
                     instance_hashes={key: config[key] for key in ("A_sha256", "B_sha256")},
                     source_revision=environment.get("git_revision"), code_sha256=config["source_sha256"],
                     completed_updates=expected_steps, final_float64_actual_output_available=actual64,
                     final_float64=final, timing=summary["timing"], memory=summary["memory"],
                     solver_counts=summary["solver_counts"], diagnostic_counts=summary["trajectory_diagnostic_counts"],
                     terminal_checkpoint=summary["final_checkpoint"], trajectory_means=summary["trajectory_means"],
                     total_instrumented_seconds=summary["total_instrumented_seconds_before_summary_write"],
                     initialization_solver_seconds=summary["initialization_solver_seconds"],
                     max_row_l1=summary["max_row_l1_in_preupdate_trajectory"],
                     max_certificate_excess=summary.get("max_certificate_excess"))
        return entry, trace
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        entry.update(status="invalid", issues=entry["issues"]+[str(exc)])
        return entry, None


def load_oracles(root, n, batches, repetitions, scope):
    path = root/"reports"/f"oracle_a100_n{n}_s1000.json"
    entry = {"n": n, "path": str(path), "status": "missing", "issues": []}
    if not path.exists():
        entry["issues"].append("Expected oracle report is missing.")
        return entry
    entry["source_sha256"] = sha256(path)
    try:
        report = read_json(path)
        settings, env, manifest = report["settings"], report["environment"], report["instance"]["manifest"]
        validate_device(settings["device"], env.get("gpu_name"), scope)
        if not report["status"].startswith("completed_"):
            raise ValueError("Oracle report does not state completion.")
        if manifest["n"] != n or manifest["m"] != n or manifest["seed"] != 1000:
            raise ValueError("Oracle instance does not match its expected shape/problem seed.")
        if settings["stochastic_seed"] != 0 or settings["evaluations_per_configuration"] != repetitions:
            raise ValueError("Oracle seed/evaluation count differs from expected pilot settings.")
        if scope == "a100-pilot" and settings["dtype"] != "float32":
            raise ValueError("Initial A100 oracle panel requires float32.")
        expected = {(plan, oracle, q) for plan in PLANS for oracle in ORACLES for q in batches}
        found = {(row["plan"], row["oracle"], row["batch_size"]) for row in report["configurations"]}
        if found != expected or len(report["configurations"]) != len(expected):
            raise ValueError("Oracle configuration panel has missing, unexpected, or duplicated entries.")
        if any(row["evaluations"] != repetitions for row in report["configurations"]):
            raise ValueError("One or more oracle configurations has incomplete repeated evaluations.")
        if any(report["exact_gradient"][plan]["evaluations"] != repetitions for plan in PLANS):
            raise ValueError("Exact-gradient timing repetitions are incomplete.")
        entry.update(status="completed", settings=settings, recorded_device=settings["device"], recorded_gpu=env.get("gpu_name"),
                     configurations=report["configurations"], exact_gradient=report["exact_gradient"],
                     time_breakdown_seconds=report["time_breakdown_seconds"],
                     peak_process_rss_mib=report["peak_process_rss_mib"],
                     peak_cuda_allocated_bytes=report["peak_cuda_allocated_bytes"],
                     peak_cuda_reserved_bytes=report["peak_cuda_reserved_bytes"],
                     instance_hashes=report["instance"]["verified_array_sha256"], source=report["source"],
                     methodology=report["methodology"])
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        entry.update(status="invalid", issues=[str(exc)])
    return entry


def label(entry):
    return "Exact gradient + same momentum" if entry["method"] == "exact" else "Trace sketch q="+str(entry["batch"])


def style(ax, xlabel, ylabel, title):
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(alpha=.22)


def save_figure(fig, out, stem, title, paths):
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94))
    for extension in ("png", "pdf"):
        path = out/(stem+"."+extension)
        fig.savefig(path, dpi=180)
        paths.append(path.name)
    plt.close(fig)


def make_smoothing_figures(runs, traces, out, stamp):
    valid = [entry for entry in runs if entry["status"] == "completed"]
    sizes = sorted({entry["n"] for entry in valid})
    paths = []
    if not valid:
        return paths
    specifications = (
        ("current_rounded_objective_vs_output_time", "time_to_this_feasible_output", "rounded_feasible_loss", "Solver + this output's repair time (s)", "Current rounded objective", False, False),
        ("true_smoothed_gap_vs_iteration", "point_iteration", "true_smoothed_gap", "Pre-update iterate k", "True smoothed FW gap", True, False),
        ("true_smoothed_gap_vs_probes", "solver_sketch_probes", "true_smoothed_gap", "Cumulative sketch probes (algorithm only)", "True smoothed FW gap", True, True),
        ("full_trajectory_mean_gap", "trajectory_average_points", "mean_true_smoothed_gap", "Number of pre-update iterates averaged", "Full-trajectory arithmetic mean of true gap", True, False),
        ("rounded_original_gap_vs_output_time", "time_to_this_feasible_output", "original_fw_gap_at_rounded", "Solver + this output's repair time (s)", "Original FW gap of current rounded output", False, False),
    )
    for stem, xkey, ykey, xlabel, ylabel, preonly, sketches_only in specifications:
        fig, axes = plt.subplots(1, len(sizes), figsize=(6.5*len(sizes), 4.4), squeeze=False)
        plotted = False
        for ax, n in zip(axes[0], sizes):
            for entry in valid:
                if entry["n"] != n or (sketches_only and entry["method"] == "exact"):
                    continue
                rows = traces[entry["name"]][:-1] if preonly else traces[entry["name"]]
                pairs = [(number(row[xkey]), number(row[ykey])) for row in rows]
                pairs = [(x,y) for x,y in pairs if x is not None and y is not None]
                if not pairs:
                    continue
                x, y = zip(*pairs)
                kwargs = dict(label=label(entry), color=COLORS.get(entry["method"]))
                if "time" in xkey:
                    ax.scatter(x, y, s=10, alpha=.7, **kwargs)
                else:
                    ax.plot(x, y, linewidth=1.2, **kwargs)
                plotted = True
            style(ax, xlabel, ylabel, f"n={n}")
            if ax.lines or ax.collections:
                ax.legend(fontsize=8)
            if "gap" in stem:
                ax.set_yscale("symlog", linthresh=1e-8)
        if plotted:
            extra = " · exact-gradient runs have no sketch probes" if sketches_only else ""
            save_figure(fig, out, stem, stamp+extra, paths)
        else:
            plt.close(fig)
    fig, axes = plt.subplots(2, len(sizes), figsize=(6.5*len(sizes), 7.2), squeeze=False)
    for column, n in enumerate(sizes):
        for row_id, metric in enumerate(("col_l1", "col_l2")):
            ax = axes[row_id,column]
            for entry in valid:
                if entry["n"] == n:
                    trace = traces[entry["name"]]
                    ax.scatter([number(row["solver_seconds_cumulative"]) for row in trace],
                               [number(row[metric]) for row in trace], s=9, alpha=.7,
                               label=label(entry), color=COLORS.get(entry["method"]))
            style(ax, "Cumulative solver time (s)", "Raw target-marginal "+("L1" if metric == "col_l1" else "L2"), f"n={n}")
            ax.legend(fontsize=8)
    save_figure(fig, out, "raw_feasibility_vs_solver_time", stamp, paths)
    return paths


def make_oracle_figures(oracles, out, stamp):
    valid = [entry for entry in oracles if entry["status"] == "completed"]
    paths = []
    if not valid:
        return paths
    for metric, ylabel, stem in (("relative_error_frobenius", "Relative Frobenius error", "oracle_time_vs_relative_error"),
                                  ("fw_decision_error", "True-gradient row-LMO decision regret", "oracle_time_vs_decision_regret")):
        fig, axes = plt.subplots(len(valid), 3, figsize=(16, 4.2*len(valid)), squeeze=False)
        for row_id, report in enumerate(valid):
            for col_id, plan in enumerate(PLANS):
                ax = axes[row_id,col_id]
                for oracle, color, marker in (("anchor", "#0072B2", "o"), ("trace_sketch", "#D55E00", "s")):
                    group = sorted((row for row in report["configurations"] if row["plan"] == plan and row["oracle"] == oracle), key=lambda row: row["batch_size"])
                    for i, row in enumerate(group):
                        timing, error = row["sampling_inclusive_seconds"], row[metric]
                        x, y = 1000*timing["median"], error["median"]
                        ax.errorbar(x, y, xerr=[[1000*(timing["median"]-timing["q25"])] ,[1000*(timing["q75"]-timing["median"])]],
                                    yerr=[[max(0., error["median"]-error["q25"])], [max(0., error["q75"]-error["median"])]],
                                    color=color, marker=marker, linestyle="none", capsize=2, label=oracle if i == 0 else None)
                        ax.annotate("q="+str(row["batch_size"]), (x,y), xytext=(5,4), textcoords="offset points", fontsize=7)
                exact = report["exact_gradient"][plan]["seconds"]
                ax.scatter(1000*exact["median"], 0., color="black", marker="*", s=85, label="Full gradient (zero error by definition)")
                ax.set_xscale("log")
                ax.set_yscale("symlog", linthresh=1e-8)
                style(ax, "Sampling-inclusive oracle time (ms)", ylabel, f"n={report['n']} · {plan.replace('_',' ')}")
                ax.legend(fontsize=7)
        save_figure(fig, out, stem, stamp+" · medians and IQR of independent oracle evaluations", paths)
    return paths


def make_markdown(report):
    lines = ["# "+report["title"], "", report["scope_statement"], "",
             f"Panel status: **{report['status']}**. Completed smoothing runs: {report['completed_smoothing_runs']}/{len(report['smoothing_runs'])}; completed oracle reports: {report['completed_oracle_reports']}/{len(report['oracle_reports'])}.", "",
             "| Expected run/report | Status | Missing evidence or issue |", "|---|---|---|"]
    for row in report["smoothing_runs"]+report["oracle_reports"]:
        name = row.get("name", f"oracle n={row['n']}")
        issue = "; ".join(row.get("issues", [])) or ("See failure record in report.json" if row["status"] == "failed" else "—")
        lines.append(f"| {name} | {row['status']} | {issue.replace('|','/')} |")
    valid = [row for row in report["smoothing_runs"] if row["status"] == "completed"]
    if valid:
        lines += ["", "Final outputs, reevaluated in CPU float64 using the actual saved rounded plan. Raw loss is the smooth quadratic at an infeasible point; rounded output feasibility is reported separately.", "",
                  "| Run | Raw smooth loss | Actual rounded loss | Raw column L1 | Raw column L2 | Rounded marginal L1 | Original gap at actual rounded output | Full-trajectory mean true gap |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for row in valid:
            f = row["final_float64"]
            actual = row["final_float64_actual_output_available"]
            values = [f["raw_smooth_loss"], f["rounded_feasible_loss"] if actual else None,
                      f["raw_residuals"]["col_l1"], f["raw_residuals"]["col_l2"],
                      f["rounded_residuals"]["both_l1"] if actual else None,
                      f["original_fw_gap_at_rounded"] if actual else None, row["trajectory_means"]["true_smoothed_gap"]]
            lines.append("| "+row["name"]+" | "+" | ".join(fmt(value) for value in values)+" |")
        lines += ["", "Time components in seconds. The plotted time-to-output is cumulative solver time plus that checkpoint's repair; diagnostic/serialization overhead is excluded from that axis and shown here. Total elapsed includes the instrumented run. Values are one observed run, not runtime estimates across repetitions.", "",
                  "| Run | Preparation/cache | Solver | Repairs, all checkpoints | Exact diagnostics | Serialization | Final float64 check | Instrumented total |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in valid:
            values = [row["timing"][key] for key in TIMING_KEYS]+[row["total_instrumented_seconds"]]
            lines.append("| "+row["name"]+" | "+" | ".join(fmt(value) for value in values)+" |")
        lines += ["", "Peak memory includes diagnostics and terminal validation; it is not a solver-only memory measurement.", "",
                  "| Run | Peak process RSS MiB | Peak GPU allocated MiB | Peak GPU reserved MiB | Algorithm oracle calls | Exact-gradient calls | Sketch probes | Row LMO calls | Diagnostic gradients |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in valid:
            memory, count = row["memory"], row["solver_counts"]
            values = [memory["process_peak_rss_bytes"]/2**20,
                      memory.get("cuda_peak_allocated_bytes",0)/2**20 if memory.get("cuda_peak_allocated_bytes") is not None else None,
                      memory.get("cuda_peak_reserved_bytes",0)/2**20 if memory.get("cuda_peak_reserved_bytes") is not None else None,
                      count["oracle_evaluations"], count["exact_gradient_calls"], count["sketch_probes"], count["row_lmo_calls"], row["diagnostic_counts"]["exact_gradient_calls"]]
            lines.append("| "+row["name"]+" | "+" | ".join(fmt(value) for value in values)+" |")
    oracle_valid = [row for row in report["oracle_reports"] if row["status"] == "completed"]
    if oracle_valid:
        lines += ["", "Fixed-plan oracle measurements compare batches at the same plan and true full gradient. Timing/error medians and IQR describe independent oracle evaluations; they are separate from optimizer-seed uncertainty. Anchor timings use the supplied Python-loop implementation. These measurements do not freeze an oracle or establish algorithmic speedups.", "",
                  "| n | Plan | Oracle | q | Evaluations | Mean absolute error | Mean relative error | Frobenius variance | Median row-LMO regret | Median oracle ms | Median full-gradient ms |", "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for report_row in oracle_valid:
            for row in report_row["configurations"]:
                values = [report_row["n"], row["plan"], row["oracle"], row["batch_size"], row["evaluations"],
                          row["absolute_error_frobenius"]["mean"], row["relative_error_frobenius"]["mean"], row["sample_variance_frobenius"],
                          row["fw_decision_error"]["median"], row["sampling_inclusive_seconds"]["median"]*1000,
                          report_row["exact_gradient"][row["plan"]]["seconds"]["median"]*1000]
                lines.append("| "+" | ".join(value if isinstance(value,str) else fmt(value) for value in values)+" |")
    if oracle_valid:
        lines += ["", "Oracle measurement cost and memory (diagnostics included). Timed stochastic evaluations and timed exact gradients are counted separately from warmup and initial reference evaluations.", "",
                  "| n | Stochastic evaluations | Timed exact gradients | Preparation s | Warmup s | Sampled-oracle s | Exact-gradient s | Diagnostics s | Serialization s | Instrumented total s | RSS MiB | GPU allocated MiB |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in oracle_valid:
            timing = row["time_breakdown_seconds"]
            values = [row["n"], sum(item["evaluations"] for item in row["configurations"]),
                      sum(item["evaluations"] for item in row["exact_gradient"].values()),
                      timing["preparation_and_initial_exact_diagnostics"], timing["warmup"],
                      timing["sampling_inclusive_oracles"], timing["timed_exact_gradients"],
                      timing["error_and_moment_diagnostics"], timing["raw_serialization"],
                      timing["total_before_summary_serialization"], row["peak_process_rss_mib"],
                      row["peak_cuda_allocated_bytes"]/2**20 if row["peak_cuda_allocated_bytes"] is not None else None]
            lines.append("| "+" | ".join(fmt(value) for value in values)+" |")
    lines += ["", "Figure interpretation: plotted objective values are current rounded outputs, never best-selected values. Time plots show only recorded checkpoints. True-gap means include every pre-update iterate and exclude the extra terminal point. With one optimization seed, these are realized trajectory statistics, not estimates of theorem expectations. The exact-gradient ablation retains the same momentum schedule. Signed numerical values are preserved; gap plots use a symmetric logarithmic scale near zero. No slopes are fitted, and no GPU/CPU algorithmic speedup conclusion is drawn.", ""]
    if report["figures"]:
        lines += ["Figures (PNG and PDF):", ""]
        for name in report["figures"]:
            lines.append(f"- [{name}]({name})")
    else:
        lines.append("No valid recorded trajectories or oracle panels were available; no curves were generated.")
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scope", choices=("a100-pilot", "parser-smoke"), default="a100-pilot")
    parser.add_argument("--expected-sizes", nargs="+", type=int, default=[128,512])
    parser.add_argument("--sketch-batches", nargs="+", type=int, default=[8,32,128])
    parser.add_argument("--expected-steps", type=int, default=200)
    parser.add_argument("--oracle-repetitions", type=int, default=100)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.expected_steps < 1 or any(n<2 for n in args.expected_sizes) or any(q<1 for q in args.sketch_batches):
        parser.error("Positive steps/batches and instance sizes at least 2 are required.")
    root, out = args.root.resolve(), args.out.resolve()
    if (out/"report.json").exists():
        parser.error("Report already exists; choose a new output directory to preserve it.")
    out.mkdir(parents=True, exist_ok=True)
    runs, traces = [], {}
    for n in args.expected_sizes:
        for method in ["exact"]+[f"sketch_q{q}" for q in args.sketch_batches]:
            entry, trace = load_smoothing(root, n, method, args.expected_steps, args.scope)
            runs.append(entry)
            if trace is not None:
                traces[entry["name"]] = trace
    oracles = [load_oracles(root,n,args.sketch_batches,args.oracle_repetitions,args.scope) for n in args.expected_sizes]
    # Shared mathematical instances are mandatory within each size, including the oracle panel.
    for n in args.expected_sizes:
        group = [entry for entry in runs if entry["n"]==n and entry["status"]=="completed"]
        distinct = {(entry["instance_hashes"]["A_sha256"],entry["instance_hashes"]["B_sha256"]) for entry in group}
        oracle = next(row for row in oracles if row["n"]==n)
        if oracle["status"]=="completed":
            distinct.add((oracle["instance_hashes"]["A"],oracle["instance_hashes"]["B"]))
        if len(distinct)>1:
            for entry in group+[oracle]:
                if entry["status"]=="completed":
                    entry["status"]="invalid"
                    entry["issues"].append("Cost-array hashes differ across supposedly matched runs for this size.")
    complete = all(row["status"]=="completed" for row in runs+oracles)
    if args.scope == "a100-pilot":
        title = "Initial A100 diagnostic candidate pilot"
        stamp = "A100 candidate pilot · problem seed 1000 · optimizer seed 0 · no replicate uncertainty"
        scope = "One development instance per size, one optimization seed, and prespecified candidate batches. This is a diagnostic candidate pilot, not a completed tuning exercise, held-out final study, convergence proof, or algorithmic speed comparison."
    else:
        title = "Parser smoke using existing local outputs"
        stamp = "PARSER SMOKE · existing local CPU outputs · not A100 evidence"
        scope = "Parser/plotting smoke test using existing recorded local outputs. Directory layout is only a test fixture; these are not A100 measurements or experiment results beyond the original CPU checks."
    figures = make_smoothing_figures(runs,traces,out,stamp)+make_oracle_figures(oracles,out,stamp)
    report = {"title":title,"scope":args.scope,"scope_statement":scope,
              "status":"complete_initial_candidate_panel" if complete and args.scope=="a100-pilot" else ("complete_parser_fixture" if complete else "incomplete"),
              "generated_utc":datetime.now(timezone.utc).isoformat(),"input_root":str(root),"command":sys.argv,
              "expected": {"sizes":args.expected_sizes,"sketch_batches":args.sketch_batches,"steps":args.expected_steps,"oracle_repetitions":args.oracle_repetitions,"problem_seed":1000,"stochastic_seed":0,"m_mu0":1},
              "completed_smoothing_runs":sum(row["status"]=="completed" for row in runs),
              "completed_oracle_reports":sum(row["status"]=="completed" for row in oracles),
              "smoothing_runs":runs,"oracle_reports":oracles,"figures":figures,
              "full_study_complete":False,"oracle_frozen":False,"fitted_slopes":None,"speedup_claim":None,
              "script_sha256":sha256(Path(__file__)),"plot_library":"matplotlib " + matplotlib.__version__}
    (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    (out/"report.md").write_text(make_markdown(report))
    print(json.dumps({"report":str(out/"report.md"),"status":report["status"],"completed_smoothing":report["completed_smoothing_runs"],"completed_oracle_reports":report["completed_oracle_reports"],"figures":len(figures)},indent=2))
    return 1 if args.require_complete and not complete else 0


if __name__ == "__main__":
    raise SystemExit(main())
