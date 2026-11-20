"""Aggregate recorded Stage C fixed-budget outputs without selecting a winner.

Reads metadata.json, trace.csv and final.npz under ROOT/<instance>/<method>.
Missing checkpoint windows and native early stops remain missing observations.
Use --scope parser-smoke with a separate fixture plan for existing CPU test data.
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

HERE = Path(__file__).resolve().parent
COLORS = {"smoothing-fw":"#0072B2", "smoothing-fw-exact":"#56B4E9", "full-kl":"#D55E00",
          "stochastic-kl":"#E69F00", "native-sampled-gw":"#009E73", "fw":"#252525"}
MARKERS = {"smoothing-fw":"o", "smoothing-fw-exact":"s", "full-kl":"^",
           "stochastic-kl":"v", "native-sampled-gw":"D", "fw":"P"}
LABELS = {"smoothing-fw":"Smoothing FW (sketch)", "smoothing-fw-exact":"Smoothing FW (exact gradient)",
          "full-kl":"Full-gradient KL prox", "stochastic-kl":"Oracle-matched stochastic KL",
          "native-sampled-gw":"Native Sampled GW", "fw":"Full-polytope FW"}
TIME_KEYS = ("preparation_seconds", "native_instrumentation_preparation_seconds", "solver_seconds", "repair_seconds", "exact_diagnostic_seconds",
             "final_float64_validation_seconds", "serialization_seconds", "unclassified_python_overhead_seconds",
             "total_elapsed_seconds_before_final_metadata_write")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def scalar(value):
    if value is None or value == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Nonfinite stored numeric value.")
    return result


def fmt(value):
    return "—" if value is None else f"{value:.5g}"


def close(actual, expected, name):
    if actual is None or not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"Recorded {name}={actual} differs from frozen value {expected}.")


def parameter_summary(method, metadata):
    config = metadata.get("config", {})
    smooth = metadata.get("smoothing_config")
    if smooth:
        return {"oracle":smooth["oracle"], "batch":smooth["batch"] if smooth["oracle"]!="exact" else None,
                "mu0":metadata.get("mu0", smooth["mu0"]), "m_mu0":metadata.get("m_mu0"),
                "gamma0":smooth["gamma0"], "rho0":smooth["rho0"],
                "step_exponent":smooth["alpha"], "momentum_exponent":smooth["beta"], "smoothing_exponent":smooth["delta"]}
    if method == "native-sampled-gw":
        return {"native_samples":config.get("native_samples", config.get("nb_samples_grad", config.get("batch"))),
                "epsilon":config.get("epsilon"), "max_iter":config.get("max_iter"),
                "sample_semantics":"native conditional samples; not comparable to trace-sketch probes",
                "native_inner_settings_unchanged":True}
    if method == "fw":
        return {"lmo":"exact SciPy assignment", "line_search":"exact quadratic"}
    return {"oracle":config.get("oracle") if method=="stochastic-kl" else "exact",
            "batch":config.get("batch") if method=="stochastic-kl" else None,
            "eta":config.get("eta"), "eta_power":config.get("eta_power"),
            "tolerance_marginal_l1":config.get("inner_tol"), "inner_max":config.get("inner_max")}


def read_run(root, instance_name, method, plan, scope):
    stem = Path(instance_name).stem
    directory = root/stem/method
    result = {"instance":stem,"method":method,"path":str(directory),"status":"missing","issues":[],"failure_events":[],"plot_eligible":False}
    required = [directory/name for name in ("metadata.json","trace.csv")]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        result["issues"].append("Missing "+", ".join(missing))
        result["status"] = "missing" if len(missing)==2 else "incomplete"
        return result, []
    try:
        metadata = load_json(directory/"metadata.json")
        with (directory/"trace.csv").open() as handle:
            trace = list(csv.DictReader(handle))
        config, environment = metadata["config"], metadata["environment"]
        manifest = metadata.get("instance_manifest") or {}
        counters = metadata.get("counters", {})
        raw_status = metadata.get("status", "unknown")
        result.update(recorded_status=raw_status, n=manifest.get("n"), m=manifest.get("m"),
                      recorded_device=environment.get("device",config.get("device")), recorded_gpu=environment.get("gpu",{}).get("name"),
                      dtype=config.get("dtype"), environment=environment, config=config,
                      parameters=parameter_summary(method,metadata), counters=counters,
                      timing=metadata.get("timing",{}), memory=metadata.get("memory",{}),
                      array_sha256=metadata.get("array_sha256",{}),
                      diagnostic_counts=metadata.get("diagnostic_counts",{}),
                      final_float64_validation=metadata.get("final_float64_validation"),
                      stop_reason=metadata.get("stop_reason",metadata.get("native_stop_reason")),
                      source_commit=metadata.get("source_commit"), source_file_sha256=metadata.get("source_file_sha256"),
                      source_artifact_sha256={path.name:sha256(path) for path in required},
                      numerical_policy=metadata.get("numerical_policy"),
                      checkpoint_policy=metadata.get("checkpoint_policy"),
                      native_instrumentation=metadata.get("instrumentation"),
                      native_warnings=metadata.get("warnings", []), timing_scope=metadata.get("timing_scope"))
        if method == "native-sampled-gw" and result["native_instrumentation"]:
            native = result["native_instrumentation"]
            result["stop_reason"] = ("native early stopping rule" if native.get("native_early_stop") else
                                     "native maximum iterations" if native.get("max_iter_exhausted") else
                                     "requested solver budget reached" if native.get("time_budget_reached") else
                                     "native solver returned; see instrumentation record")
        failures_path = directory/"failures.json"
        if failures_path.exists():
            result["failure_events"] = load_json(failures_path)
            result["source_artifact_sha256"]["failures.json"] = sha256(failures_path)
        warnings_path = directory/"warnings.json"
        if warnings_path.exists():
            result["native_warnings"] = load_json(warnings_path)
            result["source_artifact_sha256"]["warnings.json"] = sha256(warnings_path)
        if scope == "a100-pilot":
            if not str(result["recorded_device"]).startswith("cuda") or "A100" not in str(result["recorded_gpu"]):
                raise ValueError("Recorded execution is not CUDA/A100; CPU output cannot enter this panel.")
            if config.get("dtype") != plan["dtype"]:
                raise ValueError("Recorded dtype differs from the frozen plan.")
        if Path(metadata["instance"]).stem != stem:
            raise ValueError("Recorded instance name differs from output directory.")
        close(config.get("budget"),plan["budget_solver_seconds"],"solver budget")
        close(metadata.get("stochastic_seed",config.get("seed")),plan["stochastic_seed"],"stochastic seed")
        close(metadata.get("initialization_seed",config.get("init_seed")),plan["initialization_seed"],"initialization seed")
        close(config.get("threads"),plan["threads"],"thread count")
        expected_method = "smoothing-fw" if method=="smoothing-fw-exact" else method
        if config.get("method",method) != expected_method:
            raise ValueError("Recorded method differs from requested method directory.")
        expected_parameters = plan["methods"][method]
        actual_parameters = result["parameters"]
        for key in ("eta","batch","m_mu0","gamma0","rho0","step_exponent","momentum_exponent","smoothing_exponent","tolerance_marginal_l1","inner_max","native_samples","epsilon"):
            if key in expected_parameters:
                close(actual_parameters.get(key),expected_parameters[key],key)
        if "oracle" in expected_parameters and actual_parameters.get("oracle") != expected_parameters["oracle"]:
            raise ValueError("Recorded oracle differs from frozen method settings.")
        observed, missed, point_rows = {}, {}, []
        last_time = -1.
        for row in trace:
            time_value = scalar(row["actual_solver_seconds"])
            if time_value < last_time-1e-9:
                raise ValueError("Trace solver time decreases.")
            last_time = time_value
            target = scalar(row.get("target_solver_seconds"))
            if row["kind"] == "missed":
                if target is None or target in observed or target in missed:
                    raise ValueError("Missing-window record has duplicate or absent target.")
                missed[target] = {"reason":row.get("reason","unspecified"),"actual_solver_seconds":time_value,"iteration":int(row["iteration"])}
                if scalar(row.get("rounded_loss")) is not None:
                    raise ValueError("A missed checkpoint must not contain a fabricated objective.")
                continue
            if scalar(row.get("rounded_loss")) is None:
                continue
            for key in ("raw_loss","rounded_loss","row_l1","col_l1","col_l2","rounded_both_l1","repair_seconds_current","time_to_this_feasible_output"):
                scalar(row[key])
            if not math.isclose(float(row["time_to_this_feasible_output"]), time_value+float(row["repair_seconds_current"]), rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("Time-to-output does not equal recorded solver plus checkpoint repair.")
            if target is not None:
                if time_value+1e-9 < target or target in observed or target in missed:
                    raise ValueError("Checkpoint is backdated or duplicates a target.")
                observed[target] = {"actual_solver_seconds":time_value,"iteration":int(row["iteration"]),
                                    "time_to_output":float(row["time_to_this_feasible_output"])}
            point_rows.append(row)
        targets = [float(value) for value in plan["checkpoints_seconds"]]
        recorded_targets = set(observed)|set(missed)
        if not recorded_targets.issubset(set(targets)):
            raise ValueError("Trace has checkpoint targets outside the frozen common grid.")
        for target in targets:
            if target not in recorded_targets:
                missed[target] = {"reason":"checkpoint_target_not_accounted_in_trace","actual_solver_seconds":None,"iteration":None}
                result["issues"].append(f"Target {target:g}s lacks an observed or explicit missed-window record.")
        result.update(observed_checkpoints=observed,missed_checkpoints=missed,
                      last_observed_point=point_rows[-1] if point_rows else None,
                      terminal_solver_seconds=metadata.get("timing",{}).get("solver_seconds"),
                      ended_before_budget=metadata.get("timing",{}).get("solver_seconds",0)<plan["budget_solver_seconds"],
                      plot_eligible=True)
        if raw_status == "failed":
            result["status"] = "failed"
            result["issues"].append(metadata.get("error") or "Run failed; valid prefix checkpoints are retained.")
            return result, point_rows
        if not raw_status.startswith("completed"):
            result["status"] = "incomplete"
            result["issues"].append("Metadata does not record successful or native-stopped completion.")
            return result, point_rows
        final = result["final_float64_validation"]
        if not final:
            raise ValueError("Completed run lacks float64 validation of its actual saved outputs.")
        for key in ("raw_loss","rounded_loss"):
            scalar(final[key])
        for name in ("raw","rounded"):
            for key in ("row_l1","col_l1","col_l2","both_l1","min_entry"):
                scalar(final[name][key])
        final_path = directory/"final.npz"
        if not final_path.exists():
            raise ValueError("Completed run lacks final.npz.")
        with np.load(final_path,allow_pickle=False) as arrays:
            for key in ("P_raw","P_rounded"):
                P=arrays[key]
                if not np.isfinite(P).all() or (P<0).any():
                    raise ValueError(f"Saved {key} is nonfinite or negative.")
                if P.shape != (result["n"],result["m"]):
                    raise ValueError("Saved coupling shape differs from instance manifest.")
        result["source_artifact_sha256"]["final.npz"] = sha256(final_path)
        if not point_rows:
            raise ValueError("Completed run has no recorded feasible output.")
        result["status"] = raw_status
        if result["ended_before_budget"] and method!="native-sampled-gw":
            raise ValueError("Non-native run claims completion before its solver budget.")
        return result, point_rows
    except (KeyError,ValueError,TypeError,OSError,json.JSONDecodeError) as exc:
        result.update(status="invalid",plot_eligible=False)
        result["issues"].append(str(exc))
        return result, []


def save(fig,out,stem,stamp,files):
    fig.suptitle(stamp,fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94))
    for extension in ("png","pdf"):
        path=out/(stem+"."+extension)
        fig.savefig(path,dpi=180)
        files.append(path.name)
    plt.close(fig)


def plot_panel(runs,traces,instances,out,stamp):
    files=[]
    any_points=any(entry.get("plot_eligible") and traces.get((entry["instance"],entry["method"])) for entry in runs)
    if not any_points:
        return files
    for ykey,ylabel,stem in (("rounded_loss","Current rounded objective","current_rounded_objective"),
                             ("original_fw_gap","Original FW gap at current rounded output","rounded_original_fw_gap")):
        fig,axes=plt.subplots(1,len(instances),figsize=(6*len(instances),4.7),squeeze=False)
        for ax,instance in zip(axes[0],instances):
            plotted=False
            for entry in runs:
                if entry["instance"]!=instance or not entry.get("plot_eligible"):
                    continue
                pairs=[(scalar(row["time_to_this_feasible_output"]),scalar(row.get(ykey))) for row in traces.get((instance,entry["method"]),[])]
                pairs=[(x,y) for x,y in pairs if x is not None and y is not None]
                if not pairs:
                    continue
                x,y=zip(*pairs)
                name=LABELS.get(entry["method"],entry["method"])+( " [failed prefix]" if entry["status"]=="failed" else "")
                ax.scatter(x,y,s=30,label=name,color=COLORS.get(entry["method"]),marker=MARKERS.get(entry["method"],"o"))
                if entry.get("ended_before_budget") and entry["method"]=="native-sampled-gw":
                    ax.annotate("native stop",(x[-1],y[-1]),xytext=(4,5),textcoords="offset points",fontsize=7)
                plotted=True
            ax.set(title=instance,xlabel="Actual solver + checkpoint repair time (s)",ylabel=ylabel)
            ax.grid(alpha=.22)
            if plotted:
                ax.legend(fontsize=7)
                if ykey=="original_fw_gap":
                    ax.set_yscale("symlog",linthresh=1e-8)
            else:
                ax.text(.5,.5,"No recorded original FW gap" if ykey=="original_fw_gap" else "No valid recorded outputs",transform=ax.transAxes,ha="center")
        save(fig,out,stem,stamp,files)
    fig,axes=plt.subplots(2,len(instances),figsize=(6*len(instances),7.7),squeeze=False)
    for column,instance in enumerate(instances):
        for row_index,key in enumerate(("col_l1","col_l2")):
            ax=axes[row_index,column]
            for entry in runs:
                if entry["instance"]!=instance or not entry.get("plot_eligible"):
                    continue
                rows=traces.get((instance,entry["method"]),[])
                if not rows:
                    continue
                ax.scatter([scalar(row["actual_solver_seconds"]) for row in rows],[scalar(row[key]) for row in rows],s=26,
                           label=LABELS.get(entry["method"],entry["method"]),color=COLORS.get(entry["method"]),marker=MARKERS.get(entry["method"],"o"))
            ax.set(title=instance,xlabel="Actual cumulative solver time (s)",ylabel="Raw target-marginal "+("L1" if key=="col_l1" else "L2"))
            ax.grid(alpha=.22)
            if ax.collections:
                ax.legend(fontsize=7)
    save(fig,out,"raw_feasibility",stamp,files)
    return files


def table(lines,headers,rows):
    lines += ["", "| "+" | ".join(headers)+" |", "|"+"|".join("---" for _ in headers)+"|"]
    for row in rows:
        lines.append("| "+" | ".join(str(value).replace("|","/") if isinstance(value,str) else fmt(value) for value in row)+" |")


def markdown(report):
    lines=["# "+report["title"],"",report["scope_statement"],"",
           f"Panel status: **{report['status']}**. Successful run records: {report['successful_runs']}/{len(report['runs'])}. Missing checkpoint observations: {report['missing_checkpoint_observations']}.","",
           "All points are current rounded outputs at measured times. Missing windows remain missing. A terminal native early stop appears only at its actual time; no trajectory is extended to the nominal budget."]
    table(lines,["Instance / method","Status","Stop reason / issues"],[
        [row["instance"]+" / "+row["method"],row["status"],"; ".join(row["issues"]+[str(row["stop_reason"]) if row.get("stop_reason") else ""]).strip("; ") or "—"] for row in report["runs"]])
    targets=report["plan"]["checkpoints_seconds"]
    rows=[]
    for entry in report["runs"]:
        values=[entry["instance"]+" / "+entry["method"]]
        observed=entry.get("observed_checkpoints",{})
        missed=entry.get("missed_checkpoints",{})
        for target in targets:
            key=float(target)
            if key in observed:
                point=observed[key]
                values.append(f"observed at {point['actual_solver_seconds']:.5g}s, iteration {point['iteration']}")
            elif key in missed:
                values.append("MISSED: "+missed[key]["reason"])
            else:
                values.append("missing run evidence")
        values.append(entry.get("terminal_solver_seconds"))
        rows.append(values)
    lines += ["", "Checkpoint opportunities use one common requested grid. Actual observation times are reported without moving an improvement backward to a target time."]
    table(lines,["Instance / method"]+[f"Target {target:g}s" for target in targets]+["Terminal solver s"],rows)
    finished=[row for row in report["runs"] if row["status"].startswith("completed")]
    if finished:
        lines += ["", "Final losses use the actual saved raw and rounded plans promoted to float64, without a second repair. Raw smooth loss is not a feasible objective. Feasibility is reported for the rounded output to arithmetic tolerance."]
        values=[]
        for row in finished:
            f=row["final_float64_validation"]
            values.append([row["instance"]+" / "+row["method"],f["raw_loss"],f["rounded_loss"],f["raw"]["row_l1"],f["raw"]["col_l1"],f["raw"]["col_l2"],f["rounded"]["row_l1"],f["rounded"]["col_l1"]])
        table(lines,["Instance / method","Raw loss64","Rounded loss64","Raw row L1","Raw column L1","Raw column L2","Rounded row L1","Rounded column L1"],values)
    recorded=[row for row in report["runs"] if "config" in row]
    if recorded:
        lines += ["", "Recorded method parameters are development settings, not selected winners. Native Sampled GW epsilon and conditional-sample counts have their own meanings; they are not equivalent to KL step sizes or trace-sketch probe counts."]
        table(lines,["Instance / method","Parameters"],[[row["instance"]+" / "+row["method"],json.dumps(row["parameters"],sort_keys=True)] for row in recorded])
        lines += ["", "Time components in seconds. The quality axis includes solver work plus the repair for that output. Diagnostics, float64 validation, serialization, and process overhead are reported separately; no uncounted best-output selection is performed."]
        table(lines,["Instance / method","Prepare","Native instrumentation prep","Solver","All repairs","Diagnostics","Validation64","Serialization","Other overhead","Total instrumented"],
              [[row["instance"]+" / "+row["method"]]+[row["timing"].get(key, row["timing"].get("unclassified_instrumentation_and_python_seconds") if key=="unclassified_python_overhead_seconds" else None) for key in TIME_KEYS] for row in recorded])
        lines += ["", "Native snapshot/callback instrumentation is outside its solver clock. The separate native callback duration overlaps repairs and diagnostics and must not be added to those components again; the full instrumentation record is preserved in report.json."]
        lines += ["", "Peak memory covers the instrumented process, including diagnostic and validation work. Unavailable native counters are blank rather than inferred from another method's sample definition."]
        rows=[]
        for row in recorded:
            memory,counters=row["memory"],row["counters"]
            rows.append([row["instance"]+" / "+row["method"],memory.get("cpu_process_peak_rss_kib",0)/1024 if memory.get("cpu_process_peak_rss_kib") is not None else None,
                         memory.get("cuda_peak_allocated_bytes",0)/2**20 if memory.get("cuda_peak_allocated_bytes") is not None else None,
                         memory.get("cuda_peak_reserved_bytes",0)/2**20 if memory.get("cuda_peak_reserved_bytes") is not None else None,
                         counters.get("completed_updates"),counters.get("oracle_evaluations"),counters.get("sketch_probes"),
                         counters.get("inner_iterations_total",counters.get("inner_iterations")),counters.get("inner_stopping_failures"),counters.get("underflow_steps"),counters.get("underflow_entries_max")])
        table(lines,["Instance / method","RSS MiB","GPU allocated MiB","GPU reserved MiB","Updates","Oracle calls","Sketch probes","Inner iterations","Inner failures","Underflow steps","Max underflow entries"],rows)
    failure_rows=[]
    for row in report["runs"]:
        groups={}
        for event in row.get("failure_events",[]):
            kind=event.get("type","native warning")
            detail=event.get("error","")
            groups.setdefault((kind,detail),[]).append(event.get("iteration",event.get("completed_iterations")))
        for warning in row.get("native_warnings",[]):
            detail=(warning.get("category","")+": "+warning.get("message","")).strip(": ") if isinstance(warning,dict) else str(warning)
            groups.setdefault(("native warning",detail),[]).append(warning.get("iteration") if isinstance(warning,dict) else None)
        for (kind,detail),iterations in groups.items():
            present=[value for value in iterations if value is not None]
            failure_rows.append([row["instance"]+" / "+row["method"],kind,len(iterations),min(present) if present else None,max(present) if present else None,detail or "Full event records are preserved in report.json"])
    if failure_rows:
        lines += ["", "Every recorded failure/warning event is preserved in report.json, including events between checkpoints. The table groups identical event types/messages to keep this report readable."]
        table(lines,["Instance / method","Event","Count","First iteration","Last iteration","Details"],failure_rows)
    else:
        lines += ["", "No failure/warning events were present in the available run records; missing run records cannot establish absence of failures."]
    lines += ["", "This single-seed development pilot supplies no error bars, fitted rates, tuned algorithm ranking, global-optimality claim, or GPU/CPU speedup conclusion. Current feasible quality and raw feasibility remain separate. The BMMC instance receives no biological interpretation. No winner is selected.","", "Figures:",""]
    if report["figures"]:
        lines += [f"- [{name}]({name})" for name in report["figures"]]
    else:
        lines.append("No valid recorded checkpoints were available; no curves were generated.")
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--plan",type=Path,default=HERE/"configs/stage_c_pilot.json")
    parser.add_argument("--scope",choices=("a100-pilot","parser-smoke"),default="a100-pilot")
    parser.add_argument("--require-complete",action="store_true")
    args=parser.parse_args()
    root,out=args.root.resolve(),args.out.resolve()
    if (out/"report.json").exists():
        parser.error("Report exists; choose a new output directory to preserve it.")
    out.mkdir(parents=True,exist_ok=True)
    plan=load_json(args.plan)
    runs,traces=[],{}
    for instance in plan["instances"]:
        for method in plan["methods"]:
            entry,rows=read_run(root,instance,method,plan,args.scope)
            runs.append(entry)
            traces[(entry["instance"],method)]=rows
    for instance in plan["instances"]:
        stem=Path(instance).stem
        group=[row for row in runs if row["instance"]==stem and row.get("plot_eligible")]
        distinct={tuple(row["array_sha256"].get(key) for key in ("A","B","a","b")) for row in group}
        if len(distinct)>1:
            for row in group:
                row.update(status="invalid",plot_eligible=False)
                row["issues"].append("Frozen mathematical arrays differ across methods on this instance.")
    successful=sum(row["status"].startswith("completed") for row in runs)
    missed=sum(len(row.get("missed_checkpoints",{})) for row in runs)
    unavailable=sum(len(plan["checkpoints_seconds"]) for row in runs if "missed_checkpoints" not in row)
    complete=(successful==len(runs) and missed+unavailable==0)
    numerical_issues=any(row["status"]=="completed_with_inner_failures" or row.get("failure_events") or row.get("native_warnings") for row in runs)
    status=("complete_with_numerical_issues" if numerical_issues else "complete_initial_pilot") if complete else "partial"
    if args.scope=="parser-smoke":
        if complete:
            status="complete_parser_fixture_with_numerical_issues" if numerical_issues else "complete_parser_fixture"
        title="Practical-run parser smoke using preserved CPU test outputs"
        stamp="PARSER SMOKE · preserved CPU records · not A100 evidence"
        scope="This is an output-parser/plotting test using preserved small CPU runs and a separate fixture plan. These records are not A100 Stage C results."
    else:
        title="Initial 30-second A100 practical comparison"
        stamp="A100 development pilot · stochastic seed 0 · current outputs · no replicate uncertainty"
        scope="Three development instances and one stochastic seed per method, using prespecified settings and a common solver-time checkpoint grid. This is an initial practical pilot, not a replicated final study or a completed parameter-selection exercise."
    figures=plot_panel(runs,traces,[Path(name).stem for name in plan["instances"]],out,stamp)
    report={"title":title,"scope":args.scope,"scope_statement":scope,"status":status,
            "generated_utc":datetime.now(timezone.utc).isoformat(),"input_root":str(root),"command":sys.argv,
            "plan":plan,"plan_sha256":sha256(args.plan),"runs":runs,"successful_runs":successful,
            "all_run_records_successful":successful==len(runs),"missing_checkpoint_observations":missed+unavailable,
            "full_study_complete":False,"winner_selected":None,"fitted_slopes":None,"speedup_claim":None,
            "figures":figures,"script_sha256":sha256(__file__),"plot_library":"matplotlib "+matplotlib.__version__}
    (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    (out/"report.md").write_text(markdown(report))
    print(json.dumps({"report":str(out/"report.md"),"status":status,"successful_runs":successful,"expected_runs":len(runs),"missing_checkpoint_observations":missed+unavailable,"figures":len(figures)},indent=2))
    return 1 if args.require_complete and not complete else 0


if __name__=="__main__":
    raise SystemExit(main())
