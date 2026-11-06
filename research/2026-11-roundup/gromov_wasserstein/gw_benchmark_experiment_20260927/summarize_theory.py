"""Validate and summarize the prospectively specified stochastic replications.

Primary statistic: arithmetic mean across runs of each run's full prefix average
of the exact smoothed FW gap. All runs use one fixed mathematical instance.
IQR describes stochastic run dispersion, not a confidence interval or independent
dataset variation. No slope fit, rate proof, or last-iterate guarantee is made.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_variable] = "1"
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PRE_METRICS = ("true_smoothed_gap", "row_l1", "col_l1", "col_l2", "col_l2_squared", "momentum_error_l2", "oracle_error_l2")
OUTPUT_METRICS = ("rounded_feasible_loss", "original_fw_gap_at_rounded")
TIMING_KEYS = ("preparation_cache_seconds", "solver_seconds", "repair_seconds", "diagnostic_seconds", "serialization_seconds", "final_float64_validation_seconds")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def number(value):
    if value is None or value == "":
        raise ValueError("Required numeric metric is missing.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Required numeric metric is nonfinite.")
    return value


def close(actual, expected, name, rtol=2e-11, atol=2e-12):
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise ValueError(f"{name} differs from the prospectively required or independently recomputed value.")


def residuals(P, a, b):
    er, ec = P.sum(1)-a, P.sum(0)-b
    return {"row_l1": float(abs(er).sum()), "col_l1": float(abs(ec).sum()),
            "col_l2": float(np.linalg.norm(ec)), "both_l1": float(abs(er).sum()+abs(ec).sum()),
            "mass_error": abs(float(P.sum())-1), "min_entry": float(P.min())}


def independent_final_audit(path, canonical, recorded, mu):
    """Score the saved actual rounded output; never replace it by a fresh repair."""
    with np.load(path, allow_pickle=False) as data:
        raw = np.asarray(data["P_raw"], dtype=np.float64)
        actual = np.asarray(data["P_rounded"], dtype=np.float64)
        recorded_dtype = str(data["P_rounded"].dtype)
    A, B, a, b = (canonical[key] for key in ("A", "B", "a", "b"))
    if raw.shape != (len(a), len(b)) or actual.shape != raw.shape:
        raise ValueError("Saved terminal coupling shapes disagree with the frozen instance.")
    if not np.isfinite(raw).all() or not np.isfinite(actual).all() or np.any(raw < 0) or np.any(actual < 0):
        raise ValueError("Saved terminal plans contain negative or nonfinite entries.")
    def evaluate(P):
        r, c = P.sum(1), P.sum(0)
        APB = A @ P @ B
        loss = float(r @ (A**2 @ r) + c @ (B**2 @ c) - 2*(APB*P).sum())
        G = 2*((A**2 @ r)[:, None] + (B**2 @ c)[None, :] - 2*APB)
        return loss, G
    raw_loss, raw_G = evaluate(raw)
    actual_loss, actual_G = evaluate(actual)
    row, col = linear_sum_assignment(actual_G)
    actual_gap = float((actual_G*actual).sum()-actual_G[row,col].sum()/len(a))
    H = raw_G + (raw.sum(0)-b)[None,:]/mu
    actual_residuals = residuals(actual,a,b)
    raw_residuals = residuals(raw,a,b)
    audit = {"raw_smooth_loss":raw_loss, "rounded_feasible_loss":actual_loss,
             "original_fw_gap_at_rounded":actual_gap,
             "true_smoothed_gap":float((H*raw).sum()-(a*H.min(1)).sum()),
             "repair_entrywise_l1":float(abs(actual-raw).sum()),
             "raw_residuals":raw_residuals, "rounded_residuals":actual_residuals,
             "rounded_source_dtype":recorded_dtype,
             "scope":"Independent CPU float64 evaluation of actual saved P_rounded and P_raw, without rerounding"}
    if "Actual" not in recorded.get("rounded_source", "") or "P_rounded" not in recorded.get("rounded_source", ""):
        raise ValueError("Final float64 report does not explicitly identify the actual saved rounded output.")
    if recorded.get("rounded_source_dtype") != recorded_dtype:
        raise ValueError("Final float64 report and saved rounded output dtype differ.")
    for key in ("raw_smooth_loss", "rounded_feasible_loss", "original_fw_gap_at_rounded", "true_smoothed_gap", "repair_entrywise_l1"):
        close(audit[key], number(recorded[key]), "terminal actual-output "+key, rtol=2e-9, atol=2e-10)
    for group in ("raw_residuals", "rounded_residuals"):
        for key, value in audit[group].items():
            close(value, number(recorded[group][key]), "terminal "+group+" "+key, rtol=2e-9, atol=2e-10)
    if actual_residuals["both_l1"] > number(recorded["rounded_feasibility_tolerance"]):
        raise ValueError("Actual saved rounded output exceeds its stated numerical feasibility tolerance.")
    return audit


def load_run(root, seed, spec, schedules, canonical, expected_hashes, cpu_smoke):
    path = root/f"seed{seed}"
    result = {"stochastic_seed":seed,"path":str(path),"status":"missing","issues":[]}
    required = ("config.json","summary.json","trace.csv","final.npz","final_float64_evaluation.json")
    result["input_sha256"] = {name:sha256(path/name) for name in required if (path/name).exists()}
    if (path/"failure.json").exists():
        result.update(status="failed", failure=read_json(path/"failure.json"))
        return result,None
    missing=[name for name in required if not (path/name).exists()]
    if missing:
        result["status"]="missing" if len(missing)==len(required) else "incomplete"
        result["issues"]=["Missing "+", ".join(missing)]
        return result,None
    try:
        config,summary=(read_json(path/name) for name in ("config.json","summary.json"))
        with (path/"trace.csv").open() as stream:
            trace=list(csv.DictReader(stream))
        args,env,recorded_schedule=config["arguments"],config["environment"],config["schedules"]
        T=spec["outer_steps"]
        n=len(canonical["a"])
        if summary["status"]!="completed" or args["steps"]!=T or summary["completed_updates"]!=T:
            raise ValueError("Requested/completed updates do not match the prospective cap.")
        if args["seed"]!=seed or args["oracle"]!=spec["oracle"] or args["batch"]!=spec["batch"] or args["update"]!="fw":
            raise ValueError("Seed, oracle, batch, or update does not match the prospective configuration.")
        if config["shape"]!=[n,n] or Path(config["instance"]).name!=spec["instance"] or config["instance_manifest"]["seed"]!=spec["problem_seed"]:
            raise ValueError("Run uses the wrong frozen mathematical instance.")
        if config["instance_manifest"].get("noise")!=0:
            raise ValueError("Theory repeats require the specified noiseless instance.")
        for key,value in expected_hashes.items():
            if config[key]!=value:
                raise ValueError("Frozen instance "+key+" differs from the prospective data.")
        gpu=env.get("gpu",{}).get("name")
        if cpu_smoke:
            if args["device"]!="cpu":
                raise ValueError("CPU parser-smoke mode requires recorded CPU execution.")
        elif not str(args["device"]).startswith("cuda") or "A100" not in str(gpu) or args["dtype"]!="float32":
            raise ValueError("Production replication report requires recorded float32 CUDA/A100 execution.")
        if not env["deterministic_algorithms"] or env["float32_matmul_precision"]!="highest" or env["cuda_matmul_allow_tf32"] or env["cudnn_allow_tf32"] or env["amp"]:
            raise ValueError("Precision/determinism policy differs from the prescribed run policy.")
        if env["torch_threads"]!=1 or env["torch_interop_threads"]!=1:
            raise ValueError("Replication panel requires the common one-thread CPU policy.")
        for key in ("alpha","beta","delta","gamma0","rho0"):
            close(number(recorded_schedule[key]),number(schedules[key]),"schedule "+key)
        close(number(args["mu_scaled"]),spec["m_mu0"],"m*mu0")
        close(number(recorded_schedule["mu0"]),spec["m_mu0"]/n,"mu0")
        if recorded_schedule["oracle"]!=spec["oracle"] or recorded_schedule["batch"]!=spec["batch"] or recorded_schedule["update"]!="fw":
            raise ValueError("Recorded solver schedule oracle/batch/update differs from arguments/specification.")
        if recorded_schedule["initialization"]!="P0=outer(a,b)" or "first smooth gradient" not in recorded_schedule["momentum_initialization"] or "2.11" not in recorded_schedule["theory_id"]:
            raise ValueError("Initialization or theorem-statistic metadata disagrees with the specification.")
        if len(trace)!=T+1 or [int(r["point_iteration"]) for r in trace]!=list(range(T+1)):
            raise ValueError("Trace must contain each P0..P(T-1) exactly once plus one terminal PT.")
        if [r["trajectory_average_included"] for r in trace]!=["True"]*T+["False"]:
            raise ValueError("Terminal/pre-update trajectory inclusion flags are incorrect.")
        if [r["checkpoint"] for r in trace]!=["pre_update"]*T+["terminal"]:
            raise ValueError("Trace checkpoint types are inconsistent.")
        if [int(r["completed_updates_at_checkpoint"]) for r in trace]!=list(range(1,T+1))+[T]:
            raise ValueError("Checkpoint completed-update counts are incorrect.")
        if [int(r["trajectory_average_points"]) for r in trace]!=list(range(1,T+1))+[T]:
            raise ValueError("Trajectory prefix lengths are incorrect.")
        if summary["trajectory_average_points"]!=T or summary["trajectory_average_index_range"]!=[0,T-1]:
            raise ValueError("Summary includes a missing/extra point in the theoretical statistic.")
        k=np.arange(T,dtype=np.float64)
        mu0=spec["m_mu0"]/n
        for column,expected in (("gamma",schedules["gamma0"]/(k+1)**schedules["alpha"]),
                                ("rho",schedules["rho0"]/(k+1)**schedules["beta"]),
                                ("mu",mu0/(k+1)**schedules["delta"]),
                                ("beta_theorem",mu0/n/(k+1)**schedules["delta"])):
            close([number(r[column]) for r in trace[:-1]],expected,"every-iterate "+column)
        close(number(trace[-1]["mu"]),mu0/(T+1)**schedules["delta"],"terminal mu")
        curves={key:np.array([number(r[key]) for r in trace[:-1]]) for key in PRE_METRICS}
        for key,values in curves.items():
            prefix=np.cumsum(values)/np.arange(1,T+1)
            close([number(r["mean_"+key]) for r in trace[:-1]],prefix,"every-prefix mean "+key)
            close(number(summary["trajectory_means"][key]),prefix[-1],"summary full-trajectory mean "+key)
            close(number(trace[-1]["mean_"+key]),prefix[-1],"terminal-excluded mean "+key)
        if trace[-1]["momentum_error_l2"]!="" or trace[-1]["oracle_error_l2"]!="":
            raise ValueError("Terminal estimator errors must remain absent because no terminal oracle was sampled.")
        curves["prefix_mean_true_smoothed_gap"]=np.cumsum(curves["true_smoothed_gap"])/np.arange(1,T+1)
        for key in OUTPUT_METRICS:
            curves[key]=np.array([number(r[key]) for r in trace])
        curves["time_to_this_feasible_output"]=np.array([number(r["time_to_this_feasible_output"]) for r in trace])
        expected_counts={"oracle_evaluations":T,"exact_gradient_calls":0,"stochastic_gradient_calls":T,
                         "sketch_probes":T*spec["batch"],"anchor_samples":0,"initialization_oracle_calls":1,
                         "row_lmo_calls":T,"row_projection_calls":0}
        if summary["solver_counts"]!=expected_counts:
            raise ValueError("Solver oracle/initialization/LMO work counts do not match every-iterate sampling.")
        if any(int(r["solver_sketch_probes"])!=(i+1)*spec["batch"] for i,r in enumerate(trace[:-1])):
            raise ValueError("Per-point sketch work counters are inconsistent.")
        expected_diagnostics={"exact_gradient_calls":2*(T+1),"full_loss_calls":2*(T+1),
                              "assignment_lmo_calls":T+1,"row_gap_evaluations":T+1,"rounding_calls":T+1}
        if summary["trajectory_diagnostic_counts"]!=expected_diagnostics:
            raise ValueError("Exact every-iterate diagnostic counts are incomplete.")
        for key in ("solver_seconds_cumulative","diagnostic_seconds_cumulative","repair_seconds_cumulative"):
            values=np.array([number(r[key]) for r in trace])
            if np.any(np.diff(values)<-1e-12) or np.any(values<0):
                raise ValueError("Cumulative timing is decreasing or negative: "+key)
        for row in trace:
            close(number(row["time_to_this_feasible_output"]),number(row["solver_seconds_cumulative"])+number(row["repair_seconds_current"]),"feasible-output timing")
        final=summary["final_float64"]
        separate_final=read_json(path/"final_float64_evaluation.json")
        if final!=separate_final:
            raise ValueError("Summary final float64 metrics differ from the separate final report.")
        audit=independent_final_audit(path/"final.npz",canonical,final,mu0/(T+1)**schedules["delta"])
        for key in TIMING_KEYS:
            if number(summary["timing"][key])<0:
                raise ValueError("Negative timing component: "+key)
        if number(summary["total_instrumented_seconds_before_summary_write"])<0:
            raise ValueError("Negative total instrumented time.")
        if number(summary["memory"]["process_peak_rss_bytes"])<0:
            raise ValueError("Negative peak process memory.")
        if not cpu_smoke:
            for key in ("cuda_peak_allocated_bytes","cuda_peak_reserved_bytes"):
                if number(summary["memory"][key])<0:
                    raise ValueError("Negative peak CUDA memory.")
        code_hashes=config["source_sha256"]
        if not all(key in code_hashes and len(code_hashes[key])==64 for key in ("gw_core.py","smoothing_fw.py","run_smoothing.py","PROTOCOL.md")):
            raise ValueError("Source provenance hashes are incomplete.")
        result.update(status="completed",device=args["device"],gpu=gpu,dtype=args["dtype"],
                      problem_seed=spec["problem_seed"],shape=[n,n],source_revision=env.get("git_revision"),
                      source_sha256=code_hashes,instance_hashes=expected_hashes,
                      completed_updates=T,trajectory_average_points=T,
                      trajectory_means=summary["trajectory_means"],final_float64_actual_output=final,
                      independent_final_audit=audit,timing=summary["timing"],memory=summary["memory"],
                      total_instrumented_seconds=summary["total_instrumented_seconds_before_summary_write"],
                      solver_counts=summary["solver_counts"],diagnostic_counts=summary["trajectory_diagnostic_counts"],
                      max_row_l1=summary["max_row_l1_in_preupdate_trajectory"],
                      max_certificate_excess=summary.get("max_certificate_excess"))
        return result,curves
    except (ValueError,KeyError,TypeError,OSError,AssertionError,IndexError) as error:
        result.update(status="invalid",issues=[str(error)])
        return result,None


def ensemble(values):
    values=np.asarray(values,dtype=np.float64)
    return {"mean":values.mean(axis=0).tolist(),"median":np.median(values,axis=0).tolist(),
            "q25":np.quantile(values,.25,axis=0).tolist(),"q75":np.quantile(values,.75,axis=0).tolist(),
            "minimum":values.min(axis=0).tolist(),"maximum":values.max(axis=0).tolist()}


def save_figure(fig,out,stem,stamp,figures):
    fig.suptitle(stamp,fontsize=10)
    fig.tight_layout(rect=(0,0,1,.95))
    for extension in ("png","pdf"):
        path=out/(stem+"."+extension)
        fig.savefig(path,dpi=170)
        figures.append(path.name)
    plt.close(fig)


def make_figures(curves,aggregates,out,stamp,steps):
    figures=[]
    seeds=sorted(curves)
    def panel(ax,key,x,label,individual=True,points=False):
        for seed in seeds:
            y=curves[seed][key]
            if points:
                ax.scatter(x,y,s=3,alpha=.13,rasterized=True,color="#777777")
            elif individual:
                ax.plot(x,y,lw=.6,alpha=.3,color="#777777")
        stats=aggregates[key]
        ax.fill_between(x,stats["q25"],stats["q75"],alpha=.22,color="#0072B2",label="IQR across stochastic runs")
        ax.plot(x,stats["mean"],lw=1.7,color="#0072B2",label="Arithmetic mean across runs")
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
        ax.set_yscale("symlog",linthresh=1e-8)
    fig,ax=plt.subplots(figsize=(8,4.6))
    panel(ax,"prefix_mean_true_smoothed_gap",np.arange(1,steps+1),"Mean of each run's full-prefix true gap average")
    ax.set_xlabel("Number t of pre-update points P0,…,P(t−1) averaged")
    ax.legend(fontsize=8)
    save_figure(fig,out,"theory_prefix_gap_mean",stamp+" · fixed instance · no slope fit",figures)
    fig,axes=plt.subplots(2,2,figsize=(11,7.1))
    for ax,key,label in zip(axes.flat,("col_l1","col_l2","momentum_error_l2","oracle_error_l2"),
                           ("Raw target-marginal L1 residual","Raw target-marginal L2 residual","Momentum error ||d_k−∇f(P_k)||F","Fresh oracle error ||Ghat_k−∇f(P_k)||F")):
        panel(ax,key,np.arange(steps),label)
        ax.set_xlabel("Pre-update iterate k")
    axes[0,0].legend(fontsize=7)
    save_figure(fig,out,"theory_raw_feasibility_estimator_errors",stamp+" · pointwise descriptive statistics",figures)
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for ax,key,label in zip(axes,OUTPUT_METRICS,("Current rounded objective","Original FW gap at current rounded output")):
        panel(ax,key,np.arange(steps+1),label,points=True)
        ax.axvline(steps,color="#999999",ls=":",lw=.8)
        ax.set_xlabel("Evaluated iterate k (terminal PT shown separately at right)")
    axes[0].legend(fontsize=7)
    save_figure(fig,out,"theory_current_rounded_outputs",stamp+" · separate output metrics; no last-iterate guarantee",figures)
    return figures


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True,help="Parent of seed0,...,seed9 run directories")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--spec",type=Path,default=ROOT/"configs/theory_replicates.json")
    parser.add_argument("--cpu-smoke",action="store_true")
    parser.add_argument("--expected-steps",type=int)
    parser.add_argument("--expected-seeds",type=int,nargs="+")
    args=parser.parse_args()
    if (args.expected_steps is not None or args.expected_seeds is not None) and not args.cpu_smoke:
        parser.error("Prospective A100 settings cannot be overridden; step/seed overrides are only for --cpu-smoke.")
    if args.cpu_smoke and (args.expected_steps is None or args.expected_seeds is None):
        parser.error("--cpu-smoke requires explicit --expected-steps and --expected-seeds.")
    spec=read_json(args.spec)
    original_spec=json.loads(json.dumps(spec))
    if args.cpu_smoke:
        spec["outer_steps"]=args.expected_steps
        spec["stochastic_seeds"]=args.expected_seeds
        spec["number_of_replicates"]=len(args.expected_seeds)
    if spec["outer_steps"]<1 or len(set(spec["stochastic_seeds"]))!=spec["number_of_replicates"] or len(spec["stochastic_seeds"])!=spec["number_of_replicates"]:
        parser.error("Expected steps/seeds/replication count are invalid or duplicated.")
    schedules_path=ROOT/spec["schedules"]
    schedules=read_json(schedules_path)
    instance_path=ROOT/"data"/spec["instance"]
    with np.load(instance_path,allow_pickle=False) as frozen:
        arrays={key:frozen[key].copy() for key in ("A","B","a","b")}
    canonical={key:np.asarray(value,dtype=np.float64) for key,value in arrays.items()}
    expected_hashes={key+"_sha256":hashlib.sha256(arrays[key].tobytes()).hexdigest() for key in ("A","B")}
    expected_hashes["instance_sha256"]=sha256(instance_path)
    root,out=args.root.resolve(),args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    runs,curves=[],{}
    for seed in spec["stochastic_seeds"]:
        entry,trajectory=load_run(root,seed,spec,schedules,canonical,expected_hashes,args.cpu_smoke)
        runs.append(entry)
        if trajectory is not None:
            curves[seed]=trajectory
    issues=[]
    observed_dirs={path.name for path in root.glob("seed*") if path.is_dir()} if root.exists() else set()
    unexpected=sorted(observed_dirs-{f"seed{s}" for s in spec["stochastic_seeds"]})
    if unexpected:
        issues.append("Unprespecified run directories present (not pooled): "+", ".join(unexpected))
    code_groups={json.dumps(run["source_sha256"],sort_keys=True) for run in runs if run["status"]=="completed"}
    if len(code_groups)>1:
        issues.append("Valid runs have different source hashes; refusing to pool mixed implementations.")
        curves={}
    complete=len(curves)==spec["number_of_replicates"] and not issues
    scope="CPU PARSER SMOKE — no A100 outcomes" if args.cpu_smoke else "A100 fixed-instance stochastic replications"
    stamp=f"{scope} · {len(curves)}/{spec['number_of_replicates']} runs"+(" · INCOMPLETE" if not complete else "")
    aggregates={}
    if curves:
        for key in next(iter(curves.values())):
            aggregates[key]=ensemble([curves[seed][key] for seed in sorted(curves)])
    figures=make_figures(curves,aggregates,out,stamp,spec["outer_steps"]) if curves else []
    timing_summary={}
    valid=[run for run in runs if run["status"]=="completed"]
    if valid:
        timing_summary={key:ensemble([number(run["timing"][key]) for run in valid]) for key in TIMING_KEYS}
    report={"status":"completed" if complete else "incomplete","scope":scope,
            "a100_outcomes":not args.cpu_smoke and bool(curves),
            "created_utc":datetime.now(timezone.utc).isoformat(),"input_root":str(root),
            "prospective_specification":original_spec,"effective_parser_specification":spec,
            "spec_sha256":sha256(args.spec),"schedule_sha256":sha256(schedules_path),
            "summarizer_sha256":sha256(Path(__file__)),"expected_instance_hashes":expected_hashes,
            "expected_runs":spec["number_of_replicates"],"valid_runs":len(curves),"issues":issues,
            "runs":runs,"figures":figures,"timing_across_completed_runs":timing_summary,
            "primary_statistic":{"definition":"For each t, mean over stochastic seeds of (1/t)*sum_{k=0}^{t-1} true_smoothed_gap(seed,k)",
                                 "complete_prespecified_replication_set":complete,
                                 "prefix_lengths":list(range(1,spec["outer_steps"]+1)),
                                 "per_seed_prefix_means":{str(seed):data["prefix_mean_true_smoothed_gap"].tolist() for seed,data in curves.items()},
                                 "ensemble":aggregates.get("prefix_mean_true_smoothed_gap")},
            "pointwise_ensembles":{key:value for key,value in aggregates.items() if key!="prefix_mean_true_smoothed_gap"},
            "interpretation":{"randomness":"Stochastic repetitions on one fixed n128 problem, not independent datasets.",
                              "uncertainty":"IQR is empirical run-to-run dispersion, not a confidence interval for the expectation.",
                              "expectation":"The mean across the prespecified stochastic repetitions estimates the theorem's expected full-trajectory mean; incomplete subsets remain explicitly incomplete.",
                              "terminal":"Terminal PT is excluded from every prefix/theory mean. Its actual saved rounded output is independently audited in CPU float64 and reported separately.",
                              "secondary_curves":"Raw feasibility, estimator errors, and current rounded output metrics are descriptive pointwise statistics; no last-iterate theorem guarantee is claimed.",
                              "slope_fitting":"None. No fitted exponent, guide slope, or finite-range rate claim.",
                              "timing_memory":"Per-run solver, repair, diagnostics, serialization, total instrumented time, and peak process/GPU memory are reported; peaks include diagnostics."}}
    (out/"theory_replicates.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    def fmt(value):
        return "—" if value is None else f"{value:.6g}"
    lines=["# "+scope,"",f"Status: **{report['status']}**; {len(curves)}/{spec['number_of_replicates']} valid prescribed runs; {spec['outer_steps']} pre-update points per run.","",
           "The primary curve averages each run's full-prefix arithmetic mean of the true smoothed FW gap. All runs use the same frozen instance. The IQR shows stochastic run dispersion; it is not an expectation confidence interval or independent-dataset uncertainty. Terminal points are excluded from these averages.","",
           "No slope is fitted. Feasibility, estimator-error and current-output curves are descriptive; they do not establish a last-iterate guarantee.","",
           "| Seed | Status | Full-trajectory mean true gap | Terminal actual rounded loss (float64) | Terminal original FW gap (float64) | Raw col L1 | Raw col L2 | Rounded both-L1 |","|---:|---|---:|---:|---:|---:|---:|---:|"]
    for run in runs:
        if run["status"]!="completed":
            lines.append(f"| {run['stochastic_seed']} | {run['status']} | — | — | — | — | — | — |")
            continue
        final=run["independent_final_audit"]
        lines.append(f"| {run['stochastic_seed']} | completed | {fmt(run['trajectory_means']['true_smoothed_gap'])} | {fmt(final['rounded_feasible_loss'])} | {fmt(final['original_fw_gap_at_rounded'])} | {fmt(final['raw_residuals']['col_l1'])} | {fmt(final['raw_residuals']['col_l2'])} | {fmt(final['rounded_residuals']['both_l1'])} |")
    if curves:
        primary=aggregates["prefix_mean_true_smoothed_gap"]
        lines += ["",f"At t={spec['outer_steps']}, the mean across the {len(curves)} included stochastic runs is {fmt(primary['mean'][-1])}; the run-level IQR is [{fmt(primary['q25'][-1])}, {fmt(primary['q75'][-1])}]."+(" The prescribed replication set is incomplete." if not complete else "")]
    lines += ["","| Seed | Solver s | Repair s | Diagnostics s | Serialization s | Float64 audit s | Total instrumented s | CPU peak MiB | GPU allocated peak MiB | GPU reserved peak MiB |","|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for run in valid:
        timing,memory=run["timing"],run["memory"]
        memory_value=lambda key: fmt(memory[key]/2**20) if key in memory else "—"
        lines.append(f"| {run['stochastic_seed']} | {fmt(timing['solver_seconds'])} | {fmt(timing['repair_seconds'])} | {fmt(timing['diagnostic_seconds'])} | {fmt(timing['serialization_seconds'])} | {fmt(timing['final_float64_validation_seconds'])} | {fmt(run['total_instrumented_seconds'])} | {memory_value('process_peak_rss_bytes')} | {memory_value('cuda_peak_allocated_bytes')} | {memory_value('cuda_peak_reserved_bytes')} |")
    all_issues=issues+[f"seed{run['stochastic_seed']}: "+"; ".join(run["issues"]) for run in runs if run["issues"]]
    if all_issues:
        lines += ["","Validation issues:",""]+["- "+issue for issue in all_issues]
    lines += ["","The JSON preserves per-seed prefix means, pointwise ensemble statistics, all source/input hashes, validation status, actual-output audits, timings and memory. PNG and PDF figures accompany this report. Missing or failed runs are retained in the status table."]
    (out/"theory_replicates.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":report["status"],"valid_runs":len(curves),"expected_runs":spec["number_of_replicates"],"output":str(out),"scope":scope},indent=2))


if __name__=="__main__":
    main()
