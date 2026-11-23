"""Manifest-driven progress, selection audit, and replicated GW result summaries.

Inputs are immutable manifests plus recorded run outputs. Reports may be refreshed
while jobs are running. Missing/failed runs never acquire invented checkpoints.
Tuning uses strict candidate eligibility and a flat median of normalized losses.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from method_names import method_name

COLORS={"smoothing-fw":"#0072B2","smoothing-fw-exact":"#56B4E9","full-kl":"#D55E00","stochastic-kl":"#E69F00","stochastic-kl-momentum":"#AA3377","smoothing-row-projection":"#882255","native-sampled-gw":"#009E73","fw":"#252525","pg":"#CC79A7"}
ELIGIBLE_STATUSES={"completed","completed_with_warnings","completed_with_inner_failures"}
TERMINAL=ELIGIBLE_STATUSES | {"failed","infrastructure_failure","invalid"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def safe_json(obj):
    if isinstance(obj,float) and not math.isfinite(obj):
        if math.isnan(obj):
            raise ValueError("NaN is not a valid report statistic.")
        return "+inf" if obj>0 else "-inf"
    if isinstance(obj,dict):
        return {str(key):safe_json(value) for key,value in obj.items()}
    if isinstance(obj,(list,tuple)):
        return [safe_json(value) for value in obj]
    return obj


def atomic_json(path,obj):
    temporary=path.with_name(path.name+f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(safe_json(obj),indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def value(x):
    if x in (None,""):
        return None
    result=float(x)
    if not math.isfinite(result):
        raise ValueError("Nonfinite recorded metric.")
    return result


def fmt(x):
    return "—" if x is None else x if isinstance(x,str) else f"{x:.5g}"


def stats(samples):
    samples=[float(x) for x in samples if x is not None]
    if not samples:
        return {"count":0,"median":None,"q25":None,"q75":None}
    a=np.asarray(samples)
    return {"count":len(samples),"median":float(np.median(a)),"q25":float(np.quantile(a,.25)),"q75":float(np.quantile(a,.75))}


def initialization(run):
    return "product" if run.get("initialization","product")=="product" else "randomized"


def replicate_key(run):
    mode=initialization(run)
    if run.get("deterministic",False):
        return ("deterministic",run.get("init_seed",run.get("initialization_seed"))) if mode!="product" else ("deterministic_product",)
    return (run.get("seed",run.get("stochastic_seed")),run.get("init_seed",run.get("initialization_seed")) if mode!="product" else None)


def resolve_run(roots,run_id):
    relative=Path(run_id)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Manifest run_id must be a safe relative path.")
    paths=[root/relative for root in roots if (root/relative).exists()]
    if len(paths)>1:
        raise ValueError("Run exists in multiple roots; select one explicit source instead of silently choosing a rerun.")
    return paths[0] if paths else None


def manifest_cli(argv):
    """Read the explicit key/value options frozen by the manifest generator."""
    options={}
    index=1
    while index<len(argv):
        key=argv[index]
        if not key.startswith("--") or index+1>=len(argv):
            raise ValueError("Manifest argv must contain explicit option/value pairs.")
        options[key[2:].replace("-","_")]=argv[index+1]
        index+=2
    return options


def source_ledgers(roots):
    index=defaultdict(list)
    for root in roots:
        for path in sorted((root/"_campaign").glob("shard_*.json")):
            ledger=read_json(path)
            for receipt in ledger.get("records",[]):
                index[(str(root),receipt["run_id"])].append({"path":str(path),"sha256":digest(path),
                    "manifest_sha256":ledger.get("manifest_sha256"),"source_commit":ledger.get("source_commit"),"receipt":receipt})
    return index


def saved_residuals(P):
    P=np.asarray(P,dtype=np.float64)
    n,m=P.shape
    er=P.sum(axis=1)-1/n
    ec=P.sum(axis=0)-1/m
    return {"row_l1":float(np.abs(er).sum()),"col_l1":float(np.abs(ec).sum()),"col_l2":float(np.linalg.norm(ec)),
            "both_l1":float(np.abs(er).sum()+np.abs(ec).sum()),"mass_error":float(abs(P.sum()-1)),"min_entry":float(P.min())}


def initialization_audit(runs):
    groups=defaultdict(list)
    for run in runs:
        if run.get("initialization_plan_sha256"):
            groups[(run.get("instance_sha256"),run["initialization"],run["init_seed"] if run["initialization"]!="product" else None,run.get("solver_dtype"))].append(run)
    errors=[]
    for key,members in groups.items():
        if len({run["initialization_plan_sha256"] for run in members})>1:
            errors.append({"group":key,"run_ids":[r["run_id"] for r in members],"error":"Initial plan hashes differ for the same instance, initialization and dtype."})
            for run in members:
                run.update(status="invalid",eligible_final_output=False,provenance_valid=False,failure_penalty=float("inf"))
                run["issues"].append(errors[-1]["error"])
                run["observations"]={}
    return errors


def inspect_run(spec,roots,phase,manifest_hash,scope,ledger_index=None):
    result={"run_id":spec["run_id"],"phase":phase,"method":spec["method"],"candidate_id":spec.get("candidate_id",spec.get("config_id")),
            "instance":spec["instance"],"family":spec.get("family"),"n":spec.get("n"),"noise":spec.get("noise"),"problem_seed":spec.get("problem_seed"),
            "seed":spec.get("seed",spec.get("stochastic_seed")),"init_seed":spec.get("init_seed",spec.get("initialization_seed")),
            "initialization":initialization(spec),"deterministic":spec.get("deterministic",False),"budget":float(spec["budget"]),
            "checkpoints":[float(x) for x in spec["checkpoints"]],"parameters":spec.get("parameters",{}),
            "comparison_phase":spec.get("comparison_phase",phase),"source_method_for_selection":spec.get("source_method_for_selection"),
            "source_candidate_id":spec.get("source_candidate_id"),"momentum_source_method_for_selection":spec.get("momentum_source_method_for_selection"),
            "momentum_source_candidate_id":spec.get("momentum_source_candidate_id"),"parent_run_id":spec.get("parent_run_id"),
            "parent_manifest_sha256":spec.get("parent_manifest_sha256"),"expected_instance_sha256":spec.get("instance_sha256",spec.get("expected_npz_sha256")),
            "expected_array_sha256":spec.get("expected_array_sha256",{}),"status":"missing","attempted":False,"terminal":False,"execution_terminal":False,"provenance_valid":False,"eligible_final_output":False,"failure_penalty":float("inf"),
            "issues":[],"observations":{},"terminal_observation":None,"source_manifest_sha256":manifest_hash}
    try:
        path=resolve_run(roots,spec["run_id"])
        if path is None:
            return result
        result["path"]=str(path)
        receipt_path=path/"campaign_receipt.json"
        receipt=read_json(receipt_path) if receipt_path.exists() else None
        if receipt:
            result["receipt_status"]=receipt.get("status")
            result["attempted"]=True
            result["receipt_sha256"]=digest(receipt_path)
            if receipt.get("manifest_sha256")!=manifest_hash:
                raise ValueError("Campaign receipt refers to a different frozen manifest.")
            if receipt.get("run_spec")!=spec or receipt.get("run_id")!=spec["run_id"]:
                raise ValueError("Campaign receipt run specification differs from the frozen manifest.")
            result["execution_terminal"]=receipt.get("status") in TERMINAL
            if receipt.get("status")=="infrastructure_failure":
                result.update(status="infrastructure_failure",terminal=True)
                result["issues"].append(receipt.get("infrastructure_error") or "Infrastructure failure recorded by shard launcher.")
                return result
        metadata_path=path/"metadata.json"
        if not metadata_path.exists():
            result["status"]="pending" if receipt is None else "missing_metadata"
            return result
        if receipt is None and scope!="parser-smoke":
            result["status"]="pending_receipt"
            result["issues"].append("Solver metadata exists, but launcher completion receipt is not yet available.")
            return result
        metadata=read_json(metadata_path)
        if receipt and receipt.get("status")!=metadata.get("status"):
            result["provenance_valid"]=False
            raise ValueError("Launcher receipt status disagrees with solver metadata.")
        if scope=="parser-smoke":
            result["execution_terminal"]=metadata.get("status") in TERMINAL
        else:
            root=next(root for root in roots if path==root/Path(spec["run_id"]))
            matching=(ledger_index or {}).get((str(root),spec["run_id"]),[])
            if not matching:
                result.update(status="pending_ledger",provenance_valid=False)
                result["issues"].append("Completion receipt exists, but its shard source ledger is not yet available.")
                return result
            if len(matching)!=1 or matching[0]["manifest_sha256"]!=manifest_hash or matching[0]["source_commit"]!=metadata.get("source_commit"):
                result["provenance_valid"]=False
                raise ValueError("Solver source commit or manifest differs from its unique shard ledger.")
            if metadata.get("source_dirty") is not False:
                result["provenance_valid"]=False
                raise ValueError("Campaign solver source was not recorded as clean.")
            result["source_ledger"]={key:item for key,item in matching[0].items() if key!="receipt"}
        result["attempted"]=True
        config=metadata["config"]
        result.update(recorded_status=metadata["status"],counters=metadata.get("counters",{}),timing=metadata.get("timing",{}),memory=metadata.get("memory",{}),
                      environment=metadata.get("environment",{}),source_commit=metadata.get("source_commit"),metadata_sha256=digest(metadata_path),
                      array_sha256=metadata.get("array_sha256",{}),instance_sha256=metadata.get("instance_sha256"),solver_dtype=config.get("dtype"),
                      initialization_plan_sha256=metadata.get("initialization_plan_sha256"),initialization_details=metadata.get("initialization_details"),
                      mu0=metadata.get("mu0"),m_mu0=metadata.get("m_mu0"),smoothing_config=metadata.get("smoothing_config"),momentum_config=metadata.get("momentum_config"),failures_file=str(path/"failures.json") if (path/"failures.json").exists() else None)
        expected_npz=spec.get("instance_sha256",spec.get("expected_npz_sha256"))
        if expected_npz and metadata.get("instance_sha256")!=expected_npz:
            raise ValueError("Frozen instance NPZ hash differs from manifest.")
        for key,expected in spec.get("expected_array_sha256",{}).items():
            if metadata.get("array_sha256",{}).get(key)!=expected:
                raise ValueError("Frozen "+key+" array hash differs from manifest.")
        if Path(metadata["instance"]).name!=spec["instance"]:
            raise ValueError("Recorded instance differs from manifest.")
        for actual,expected,name in ((config["budget"],spec["budget"],"budget"),
                                     (metadata.get("stochastic_seed",config.get("seed")),result["seed"],"seed"),
                                     (metadata.get("initialization_seed",config.get("init_seed")),result["init_seed"],"initialization seed")):
            if expected is not None and actual!=expected:
                raise ValueError("Recorded "+name+" differs from manifest.")
        actual_mode="product" if config.get("initialization","product")=="product" else "randomized"
        if actual_mode!=result["initialization"]:
            raise ValueError("Recorded initialization mode differs from manifest.")
        expected_method=spec.get("implementation",spec["method"])
        if metadata.get("method",config.get("method"))!=expected_method:
            raise ValueError("Recorded implementation differs from manifest.")
        checked=[]
        for key,expected in manifest_cli(spec.get("argv",[])).items():
            if key not in config:
                raise ValueError("Recorded config lacks frozen CLI parameter "+key+".")
            actual=config[key]
            if key=="device":
                matches=str(actual).split(":")[0]==str(expected).split(":")[0] and (":" not in expected or actual==expected)
            elif key=="checkpoints":
                matches=[float(x) for x in str(actual).split(",")]==[float(x) for x in str(expected).split(",")]
            elif isinstance(actual,(int,float)) and not isinstance(actual,bool):
                matches=float(actual)==float(expected)
            else:
                matches=str(actual)==str(expected)
            if not matches:
                raise ValueError("Recorded CLI parameter "+key+" differs from frozen manifest.")
            checked.append(key)
        if scope!="parser-smoke":
            env=metadata["environment"]
            if not str(env.get("device","")).startswith("cuda") or "A100" not in env.get("gpu",{}).get("name",""):
                raise ValueError("Campaign output was not recorded on the prescribed CUDA A100 device.")
            policy={"float32_matmul_precision":"highest","cuda_matmul_allow_tf32":False,"cudnn_allow_tf32":False,"amp":False,"deterministic_algorithms":True}
            if any(env.get(key)!=expected for key,expected in policy.items()) or env.get("torch_num_threads")!=config.get("threads"):
                raise ValueError("Recorded precision/determinism/thread policy differs from the campaign.")
            if not metadata.get("source_file_sha256"):
                raise ValueError("Missing solver source file hashes.")
            for filename,expected in metadata["source_file_sha256"].items():
                snapshot=path/"source_snapshot"/filename
                if not snapshot.exists() or digest(snapshot)!=expected:
                    result["provenance_valid"]=False
                    raise ValueError("Archived solver source hash mismatch for "+filename+".")
        for key,expected in spec.get("parameters",{}).items():
            if key not in config:
                continue  # Formula/meaning annotations accompany their concrete CLI parameters.
            actual=config[key]
            matches=math.isclose(float(actual),float(expected),rel_tol=1e-12,abs_tol=1e-15) if isinstance(expected,(int,float)) and actual is not None else actual==expected
            if not matches:
                raise ValueError("Recorded parameter "+key+" differs from frozen manifest.")
            checked.append(key)
        if expected_method=="stochastic-kl-momentum" and (metadata.get("momentum_config") or {}).get("beta")!=spec.get("parameters",{}).get("momentum_beta",0.5):
            raise ValueError("Recorded KL momentum exponent differs from its frozen supplemental specification.")
        result["parameter_keys_checked"]=checked
        native=metadata.get("instrumentation") or {}
        if scope!="parser-smoke" and native:
            native_archives=native.get("provenance",{}).get("archive_sha256",{})
            if not native_archives:
                raise ValueError("Missing native instrumentation archive hashes.")
            for filename,expected in native_archives.items():
                archived=path/"native_source"/filename
                if not archived.is_file() or digest(archived)!=expected:
                    result["provenance_valid"]=False
                    raise ValueError("Native archived source hash mismatch for "+filename+".")
        result["provenance_valid"]=True
        warnings=metadata.get("warnings",[])
        result["warning_counts"]=[{"category":category,"message":message,"count":count} for (category,message),count in Counter((w.get("category",""),w.get("message","")) for w in warnings).items()]
        result["warning_count"]=len(warnings)
        result["native_early_stop"]=bool(native.get("native_early_stop"))
        result["native_max_iter_exhausted"]=bool(native.get("max_iter_exhausted"))
        result["native_time_budget_reached"]=native.get("time_budget_reached")
        result["native_stop_semantics"]="native stopping event; no convergence claim, particularly when warnings are present" if native else None
        actual_solver=value(result["timing"].get("solver_seconds",0))
        if actual_solver is None or actual_solver<0:
            raise ValueError("Invalid final solver clock.")
        result["actual_solver_seconds"]=actual_solver
        result["budget_met"]=actual_solver+1e-10>=result["budget"]
        result["ended_before_budget"]=not result["budget_met"]
        trace_path=path/"trace.csv"
        if trace_path.exists():
            with trace_path.open() as handle:
                rows=list(csv.DictReader(handle))
            result["trace_sha256"]=digest(trace_path)
            result["missed_checkpoints"]=[]
            seen=set();seen_iterations=set();previous_clock=-1.;previous_iteration=-1
            for row in rows:
                target=value(row.get("target_solver_seconds"))
                if row.get("kind")=="missed":
                    result["missed_checkpoints"].append({"target":target,"reason":row.get("reason"),"actual_solver_seconds":value(row.get("actual_solver_seconds"))})
                    continue
                if value(row.get("rounded_loss")) is None:
                    continue
                point={"target":target,"kind":row.get("kind"),"iteration":int(row["iteration"]),
                       "actual_solver_seconds":value(row["actual_solver_seconds"]),"time_to_output":value(row["time_to_this_feasible_output"]),
                       "rounded_loss":value(row["rounded_loss"]),"raw_loss":value(row["raw_loss"]),
                       "row_l1":value(row["row_l1"]),"col_l1":value(row["col_l1"]),"col_l2":value(row["col_l2"]),
                       "rounded_both_l1":value(row["rounded_both_l1"]),"original_fw_gap":value(row.get("original_fw_gap")),
                       **{key:value(row.get(key)) for key in ("repair_l1","mass_error","min_entry","true_smoothed_gap","mu",
                           "cumulative_oracle_evaluations","cumulative_sketch_probes","cumulative_exact_gradient_calls")}}
                clock=point["actual_solver_seconds"]
                if clock is None or clock<0 or clock+1e-10<previous_clock or clock>actual_solver+1e-8 or point["iteration"]<previous_iteration:
                    raise ValueError("Trace has a backward or impossible solver clock/iteration.")
                if point["time_to_output"] is None or point["time_to_output"]+1e-10<clock:
                    raise ValueError("Time to feasible output precedes its solver time.")
                repair=value(row.get("repair_seconds_current"))
                if repair is not None and (repair<0 or not math.isclose(point["time_to_output"],clock+repair,rel_tol=1e-9,abs_tol=1e-8)):
                    raise ValueError("Time to feasible output is not actual solver plus current repair time.")
                previous_clock=clock;previous_iteration=point["iteration"]
                if target is not None:
                    if point["iteration"] in seen_iterations:
                        raise ValueError("Same completed iteration was reused for multiple checkpoint windows.")
                    seen_iterations.add(point["iteration"])
                    if target in seen or point["actual_solver_seconds"]+1e-10<target or target not in result["checkpoints"]:
                        raise ValueError("Trace checkpoint is duplicated, backdated, or outside the prescribed grid.")
                    seen.add(target)
                    result["observations"][str(target)]=point
                result["terminal_observation"]=point
        result["missing_checkpoint_targets"]=[target for target in result["checkpoints"] if str(target) not in result["observations"]]
        status=metadata["status"]
        result.update(status=status,terminal=status in TERMINAL)
        if status=="failed":
            result["issues"].append(metadata.get("error") or "Solver failed; any valid earlier checkpoint remains explicitly observed.")
            return result
        if status not in ELIGIBLE_STATUSES:
            raise ValueError("Unrecognized completed/terminal status is not eligible.")
        if scope!="parser-smoke" and expected_method!="native-sampled-gw":
            if not metadata.get("initialization_plan_sha256"):
                result["provenance_valid"]=False
                raise ValueError("Missing initial coupling hash.")
            details=metadata.get("initialization_details") or {}
            if details.get("mode")!=result["initialization"]:
                result["provenance_valid"]=False
                raise ValueError("Initial plan details disagree with prescribed initialization.")
            if result["initialization"]=="randomized" and details.get("independent_permutation")!=np.random.default_rng(result["init_seed"]).permutation(result["n"]).tolist():
                result["provenance_valid"]=False
                raise ValueError("Initial permutation differs from the independent prescribed initialization seed.")
        if not trace_path.exists():
            raise ValueError("Completed output lacks its observed checkpoint trace.")
        accounted={float(key) for key in result["observations"]} | {point["target"] for point in result["missed_checkpoints"]}
        if accounted!=set(result["checkpoints"]):
            raise ValueError("Completed trace does not account for every prescribed checkpoint.")
        final=metadata.get("final_float64_validation")
        if not final or final.get("dtype")!="float64":
            raise ValueError("No float64 validation of the actual final output.")
        loss=value(final["rounded_loss"])
        raw_loss=value(final["raw_loss"])
        if loss < -1e-10 or raw_loss < -1e-10:
            raise ValueError("Substantially negative squared GW loss; small cancellation values remain unclamped.")
        if not result["budget_met"] and spec["method"]!="native-sampled-gw":
            raise ValueError("Non-native completed run ended before its prescribed solver budget.")
        if not result["budget_met"] and spec["method"]=="native-sampled-gw" and not (result["native_early_stop"] or result["native_max_iter_exhausted"]):
            raise ValueError("Sub-budget native output lacks an explicit native stopping reason.")
        dtype=np.dtype(config["dtype"])
        tolerance=100*np.finfo(dtype).eps
        if value(final["rounded"]["both_l1"])>tolerance or value(final["rounded"]["min_entry"])<0:
            raise ValueError("Actual final rounded output fails feasibility/nonnegativity validation.")
        final_path=path/"final.npz"
        result["final_npz_sha256"]=digest(final_path)
        independent={}
        with np.load(final_path,allow_pickle=False) as plans:
            for key,record_key in (("P_raw","raw"),("P_rounded","rounded")):
                P=plans[key]
                if P.shape!=(result["n"],spec.get("m",result["n"])) or P.dtype!=dtype or not np.isfinite(P).all() or (P<0).any():
                    raise ValueError("Saved final coupling shape, dtype, finiteness or nonnegativity is invalid.")
                independent[record_key]=saved_residuals(P)
                for metric,actual in independent[record_key].items():
                    recorded=value(final[record_key][metric])
                    if recorded is None or not math.isclose(actual,recorded,rel_tol=1e-6,abs_tol=1e-11):
                        raise ValueError("Saved "+key+" "+metric+" differs from its float64 validation metadata.")
            result["repair_l1_float64"]=float(np.abs(np.asarray(plans["P_rounded"],dtype=np.float64)-np.asarray(plans["P_raw"],dtype=np.float64)).sum())
        if independent["rounded"]["both_l1"]>tolerance:
            raise ValueError("Independently summed saved rounded output is infeasible.")
        result["independent_saved_plan_residuals"]=independent
        result["saved_plan_validation"]="NumPy float64 sums versus exact uniform marginals; original losses retained from runner float64 evaluator, not recomputed by summarizer"
        product=value(spec.get("product_loss_float64"))
        if product is None or product<=0:
            raise ValueError("Manifest must supply a positive finite float64 product-plan loss.")
        result.update(eligible_final_output=True,failure_penalty=0.,final_time_to_output=(result["terminal_observation"] or {}).get("time_to_output"),rounded_loss_float64=loss,raw_loss_float64=raw_loss,
                      normalized_loss=loss/product,product_loss_float64=product,final_raw_residuals=final["raw"],
                      final_rounded_residuals=final["rounded"],final_validation_plan=final.get("plan"))
        return result
    except (KeyError,ValueError,TypeError,OSError,EOFError,zipfile.BadZipFile,json.JSONDecodeError) as exc:
        result.update(status="invalid",terminal=True,eligible_final_output=False)
        result["issues"].append(str(exc))
        result["observations"]={}
        return result


def candidate_audit(runs,selection):
    groups=defaultdict(list)
    for run in runs:
        if run["phase"]=="tuning":
            groups[(run["method"],run["candidate_id"])].append(run)
    candidates=[]
    for (method,candidate),members in sorted(groups.items()):
        eligible=all(run["eligible_final_output"] for run in members)
        score=float(np.median([run["normalized_loss"] for run in members])) if eligible else float("inf")
        candidates.append({"method":method,"candidate_id":candidate,"eligible":eligible,"score":score,
                           "prescribed_runs":len(members),"valid_runs":sum(run["eligible_final_output"] for run in members),
                           "disqualifying_run_ids":[run["run_id"] for run in members if not run["eligible_final_output"]],
                           "warning_count":sum(run.get("warning_count",0) for run in members),
                           "inner_stopping_failures":sum(run.get("counters",{}).get("inner_stopping_failures") or 0 for run in members) if all(run.get("counters",{}).get("inner_stopping_failures") is not None for run in members) else None,
                           "known_inner_stopping_failures":sum(run.get("counters",{}).get("inner_stopping_failures") or 0 for run in members),
                           "runs_with_unknown_inner_counts":sum(run.get("counters",{}).get("inner_stopping_failures") is None for run in members)})
    audits=[]
    for method in sorted({row["method"] for row in candidates}):
        eligible=sorted((row for row in candidates if row["method"]==method and row["eligible"]),key=lambda row:(row["score"],row["candidate_id"]))
        authoritative=(selection or {}).get("selections",{}).get(method)
        expected=eligible[0]["candidate_id"] if eligible else None
        audit={"method":method,"lowest_eligible_candidate_by_frozen_rule":expected,"authoritative_selection":authoritative,
               "selection_audit":"not_supplied","pending_runs":sum(not run["terminal"] for run in runs if run["phase"]=="tuning" and run["method"]==method)}
        if authoritative:
            chosen=authoritative.get("candidate_id")
            valid_status=(authoritative.get("status")=="selected" if expected is not None else authoritative.get("status")=="no_valid_setting")
            score_ok=(authoritative.get("score") is None if expected is None else isinstance(authoritative.get("score"),(int,float)) and math.isfinite(authoritative["score"]) and authoritative["score"]==eligible[0]["score"])
            audit["selection_audit"]="matches" if chosen==expected and valid_status and score_ok else "MISMATCH"
        audits.append(audit)
    return candidates,audits


def build_selection(candidates,tuning_record,tuning_manifest):
    if not tuning_record.get("campaign_config_sha256"):
        raise ValueError("Selection requires a campaign configuration hash in the tuning manifest.")
    methods=list(tuning_manifest.get("candidate_catalog",{})) or sorted({row["method"] for row in candidates})
    selections={}
    for method in methods:
        eligible=sorted((row for row in candidates if row["method"]==method and row["eligible"]),key=lambda row:(row["score"],row["candidate_id"]))
        chosen=eligible[0] if eligible else None
        selections[method]={"status":"selected" if chosen else "no_valid_setting", "candidate_id":chosen["candidate_id"] if chosen else None,
                            "score":chosen["score"] if chosen else None,"eligible_candidate_count":len(eligible),
                            "rule":"flat median over prescribed normalized final losses; any invalid/missing run disqualifies candidate; exact-score ties use lexicographic candidate ID"}
    return {"schema_version":1,"campaign_config_sha256":tuning_record["campaign_config_sha256"],
            "tuning_manifest_sha256":tuning_record["sha256"],"selections":selections,"candidate_scores":candidates,
            "created_utc":datetime.now(timezone.utc).isoformat(),"selection_source":"tuning phase only; actual final rounded plans evaluated in float64",
            "failed_run_penalty":"+inf; whole candidate ineligible","warnings_policy":"completed_with_inner_failures and completed_with_warnings remain eligible when the final plan is valid",
            "no_valid_setting_policy":"blocked dependent work; no fallback candidate"}


def aggregate(runs):
    fixed_groups=defaultdict(list)
    for run in runs:
        if run["phase"]!="tuning":
            key=(run["phase"],run["initialization"],run["family"],run["n"],run["noise"],run["method"],run["candidate_id"],run["instance"])
            fixed_groups[key].append(run)
    fixed=[]
    duplicates=[]
    metrics=("rounded_loss","raw_loss","repair_l1","row_l1","col_l1","col_l2","mass_error","min_entry","original_fw_gap","time_to_output","actual_solver_seconds","iteration","true_smoothed_gap","mu","cumulative_oracle_evaluations","cumulative_sketch_probes","cumulative_exact_gradient_calls")
    for key,members in fixed_groups.items():
        unique={}
        for run in members:
            unit=replicate_key(run)
            if unit in unique:
                duplicates.append({"run_id":run["run_id"],"same_scientific_unit_as":unique[unit]["run_id"],"reason":"identical deterministic/product or duplicated seed/initialization unit; not an independent error-bar replicate"})
            else:
                unique[unit]=run
        representatives=list(unique.values())
        variable_init=len({r["init_seed"] for r in representatives})>1 and key[1]!="product"
        variable_stochastic=any(not r["deterministic"] for r in representatives) and len({r["seed"] for r in representatives})>1
        axis="stochastic and initialization repeats" if variable_init and variable_stochastic else "initialization repeats" if variable_init else "stochastic repeats" if variable_stochastic else "single realization; no replication error bar"
        points=[]
        for target in sorted({x for r in representatives for x in r["checkpoints"]}):
            observed=[r["observations"].get(str(target)) for r in representatives]
            observed=[x for x in observed if x]
            points.append({"target":target,"prescribed_units":len(representatives),"observed_units":len(observed),
                           "coverage_complete":len(observed)==len(representatives),"metrics":{metric:stats([point.get(metric) for point in observed]) for metric in metrics}})
        fixed.append({"phase":key[0],"initialization":key[1],"family":key[2],"n":key[3],"noise":key[4],"method":key[5],"candidate_id":key[6],"instance":key[7],
                      "comparison_phase":members[0].get("comparison_phase",key[0]),"uncertainty_axis":axis,"unique_units":len(representatives),"prescribed_runs":len(members),
                      "valid_final_units":sum(r["eligible_final_output"] for r in representatives),"points":points})
    broad_groups=defaultdict(list)
    for row in fixed:
        broad_groups[(row["phase"],row["initialization"],row["family"],row["n"],row["noise"],row["method"],row["candidate_id"])].append(row)
    broad=[]
    for key,problems in broad_groups.items():
        if len(problems)==1:
            row=dict(problems[0]);row["problem_count"]=1;row["aggregation"]="within one fixed mathematical instance"
            broad.append(row);continue
        points=[]
        for target in sorted({p["target"] for row in problems for p in row["points"]}):
            problem_points=[next((point for point in row["points"] if point["target"]==target),None) for row in problems]
            available=[point for point in problem_points if point and point["observed_units"]]
            points.append({"target":target,"prescribed_units":sum(point["prescribed_units"] for point in problem_points if point),
                           "observed_units":sum(point["observed_units"] for point in available),"observed_problems":len(available),
                           "coverage_complete":len(available)==len(problems) and all(point["coverage_complete"] for point in available),
                           "metrics":{metric:stats([point["metrics"][metric]["median"] for point in available]) for metric in metrics}})
        broad.append({"phase":key[0],"initialization":key[1],"family":key[2],"n":key[3],"noise":key[4],"method":key[5],"candidate_id":key[6],
                      "comparison_phase":problems[0].get("comparison_phase",key[0]),"problem_count":len(problems),"uncertainty_axis":"across independent problem instances (one within-instance median per instance)",
                      "aggregation":"equal weight per mathematical instance, after within-instance replicate medians","points":points})
    return fixed,broad,duplicates


def save_figure(fig,out,name,files,title):
    fig.suptitle(title,fontsize=10)
    # Keep explicit method names outside the data area. One shared legend serves
    # both feasibility axes, preserving identical colors and plotted values.
    legend = {}
    for ax in fig.axes:
        handles, labels = ax.get_legend_handles_labels()
        legend.update(zip(labels, handles))
    bottom = 0
    if legend:
        width, height = fig.get_size_inches()
        fig.set_size_inches(max(width, 11), height + 1.25)
        fig.legend(legend.values(), legend.keys(), loc="lower center", ncol=2,
                   fontsize=8, frameon=True)
        bottom = .20
    fig.tight_layout(rect=(0,bottom,1,.92))
    for extension in ("png","pdf"):
        path=out/(name+"."+extension)
        fig.savefig(path,dpi=170)
        files.append(path.name)
    plt.close(fig)


def time_axis(ax,curves,metric):
    positions=[point["metrics"][metric]["median"] for row in curves for point in row["points"]
               if point["metrics"][metric]["median"] is not None]
    if positions and min(positions)>0 and max(positions)/min(positions)>=30:
        ax.set_xscale("log")


def figures(aggregates,candidates,out,scope):
    files=[]
    groups=defaultdict(list)
    for row in aggregates:
        groups[(row.get("comparison_phase",row["phase"]),row["initialization"],row["family"],row["n"],row["noise"])].append(row)
    prefix="PARSER SMOKE; preserved CPU outputs" if scope=="parser-smoke" else "Recorded campaign outputs; descriptive median/IQR, not confidence intervals"
    for index,(group,curves) in enumerate(sorted(groups.items(),key=lambda item:str(item[0]))):
        if not any(point["observed_units"] for row in curves for point in row["points"]):
            continue
        description=f"{group[0]} · {group[2]} n={group[3]} noise={group[4]} · {group[1]}"
        uncertainty="Across-problem median/IQR of within-instance medians" if any(row.get("problem_count",1)>1 for row in curves) else "Fixed-instance replicate median/IQR; single runs have no error bars"
        title=prefix+"\n"+description+"\n"+uncertainty+" · open markers: incomplete coverage"
        for metric,ylabel,stem in (("rounded_loss","Current rounded objective","rounded_objective"),
                                   ("original_fw_gap","Original FW gap at rounded output","original_gap")):
            if metric=="original_fw_gap" and group[3]>512:
                continue
            fig,ax=plt.subplots(figsize=(8.5,5.3))
            for row in curves:
                labelled=False
                for point in row["points"]:
                    x,y=point["metrics"]["time_to_output"],point["metrics"][metric]
                    if x["median"] is None or y["median"] is None:
                        continue
                    color=COLORS.get(row["method"],"#777777")
                    label=method_name(row["method"]) if not labelled else None
                    err=[[y["median"]-y["q25"]],[y["q75"]-y["median"]]] if y["count"]>1 else None
                    ax.errorbar(x["median"],y["median"],yerr=err,xerr=[[x["median"]-x["q25"]],[x["q75"]-x["median"]]] if x["count"]>1 else None,fmt="o",color=color,
                                markerfacecolor=color if point["coverage_complete"] else "none",capsize=3,label=label)
                    labelled=True
            ax.set(xlabel="Median actual solver + current-output repair time (s)",ylabel=ylabel)
            time_axis(ax,curves,"time_to_output")
            if metric=="original_fw_gap":
                ax.set_yscale("symlog",linthresh=1e-8)
                # Recompute padding in the transformed coordinate system. A
                # symlog change can otherwise retain the prior linear limits.
                ax.autoscale(enable=True,axis="y")
                ax.margins(y=.05)
            ax.grid(alpha=.2)

            save_figure(fig,out,f"panel_{index:02d}_{stem}",files,title)
        fig,axes=plt.subplots(1,2,figsize=(13,5))
        for ax,metric in zip(axes,("col_l1","col_l2")):
            for row in curves:
                labelled=False
                for point in row["points"]:
                    x,y=point["metrics"]["actual_solver_seconds"],point["metrics"][metric]
                    if x["median"] is None or y["median"] is None:continue
                    color=COLORS.get(row["method"],"#777777")
                    err=[[y["median"]-y["q25"]],[y["q75"]-y["median"]]] if y["count"]>1 else None
                    ax.errorbar(x["median"],y["median"],yerr=err,xerr=[[x["median"]-x["q25"]],[x["q75"]-x["median"]]] if x["count"]>1 else None,fmt="o",color=color,markerfacecolor=color if point["coverage_complete"] else "none",
                                capsize=3,label=method_name(row["method"]) if not labelled else None)
                    labelled=True
            ax.set(xlabel="Median actual solver time (s)",ylabel="Raw target-marginal "+("L1" if metric=="col_l1" else "L2"))
            time_axis(ax,curves,"actual_solver_seconds")
            ax.grid(alpha=.2)

        save_figure(fig,out,f"panel_{index:02d}_raw_feasibility",files,title)
    methods=sorted({row["method"] for row in candidates})
    if methods and any(row["eligible"] for row in candidates):
        fig,axes=plt.subplots(len(methods),1,figsize=(9,3*len(methods)),squeeze=False)
        for ax,method in zip(axes[:,0],methods):
            rows=[row for row in candidates if row["method"]==method]
            for index,row in enumerate(rows):
                if row["eligible"]:ax.scatter(index,row["score"],color=COLORS.get(method,"black"))
            ax.set_xticks(range(len(rows)),[row["candidate_id"]+(" ×" if not row["eligible"] else "") for row in rows])
            ax.set(title=method_name(method),ylabel="Flat median loss/product loss")
            ax.grid(alpha=.2)
        save_figure(fig,out,"tuning_candidate_scores",files,prefix+"\nTuning only; × means ineligible (+infinite score), not an omitted failure")
    return files


def selection_provenance(manifests,runs,selection,selection_sha,scope):
    """Bind final and supplemental settings to the already frozen tuning choice."""
    errors=[]
    finals=[row for row in manifests if row['phase']!='tuning']
    strict=scope=='campaign'
    if strict and finals and not selection:
        return ['Final-phase reporting requires --selection with the frozen tuning choice.']
    if not selection:
        return errors
    if selection.get('schema_version')!=1:
        errors.append('Unsupported selection schema.')
    for row in manifests:
        if row.get('campaign_config_sha256')!=selection.get('campaign_config_sha256'):
            errors.append(row['phase']+': campaign configuration differs from frozen selection.')
        expected_tuning=row['sha256'] if row['phase']=='tuning' else row.get('tuning_manifest_sha256')
        if (strict or expected_tuning) and expected_tuning!=selection.get('tuning_manifest_sha256'):
            errors.append(row['phase']+': tuning manifest lineage differs from frozen selection.')
        if row['phase']!='tuning' and (strict or row.get('selection_sha256')) and row.get('selection_sha256')!=selection_sha:
            errors.append(row['phase']+': selection file hash differs from final manifest.')
        if row.get('supplemental_config_sha256'):
            if row.get('parent_campaign_config_sha256')!=selection.get('campaign_config_sha256'):
                errors.append(row['phase']+': supplemental parent campaign hash differs from frozen selection.')
            expected_phase={'momentum_product':'final_product','momentum_randomized':'final_randomized'}.get(row['phase'])
            base=[item for item in manifests if item['phase']==expected_phase]
            if row.get('comparison_phase')!=expected_phase or not row.get('base_manifest_sha256'):
                errors.append(row['phase']+': supplemental comparison/base-manifest lineage is missing or inconsistent.')
            if strict and len(base)!=1:
                errors.append(row['phase']+': supplemental comparisons require exactly one supplied baseline manifest.')
            if base and any(item['sha256']!=row['base_manifest_sha256'] for item in base):
                errors.append(row['phase']+': supplied baseline manifest differs from the supplemental parent manifest.')
    for run in runs:
        if run['phase']=='tuning' or not strict:
            continue
        method=run['method']
        if method=='fw':
            if run['candidate_id']!='untuned':
                errors.append(run['run_id']+': parameter-free FW must retain its untuned identifier.')
            continue
        parent=run.get('source_method_for_selection') or ('smoothing-fw' if method=='smoothing-fw-exact' else method)
        chosen=selection.get('selections',{}).get(parent,{})
        score=chosen.get('score')
        if chosen.get('status')!='selected' or isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score):
            errors.append(run['run_id']+': final settings lack a finite selected tuning parent.')
            continue
        if method=='stochastic-kl-momentum':
            enclosing=[row for row in manifests if row['sha256']==run['source_manifest_sha256']]
            parent_runs=[row for row in runs if row['run_id']==run.get('parent_run_id')]
            paired_ok=len(enclosing)==1 and len(parent_runs)==1
            if paired_ok:
                enclosing=enclosing[0];base_run=parent_runs[0]
                paired_ok=(enclosing.get('supplemental_config_sha256') is not None
                    and run['phase']==enclosing['phase']
                    and run.get('comparison_phase')==enclosing.get('comparison_phase')
                    and run.get('parent_manifest_sha256')==enclosing.get('base_manifest_sha256')
                    and base_run['source_manifest_sha256']==enclosing.get('base_manifest_sha256')
                    and base_run['phase']==run.get('comparison_phase') and base_run['method']=='stochastic-kl'
                    and base_run['candidate_id']==chosen.get('candidate_id')
                    and all(run.get(key)==base_run.get(key) for key in ('instance','n','noise','family','problem_seed','seed','init_seed','initialization','budget','checkpoints','expected_instance_sha256','expected_array_sha256'))
                    and all(run.get('parameters',{}).get(key)==val for key,val in base_run.get('parameters',{}).items()))
            if not paired_ok:
                errors.append(run['run_id']+': supplemental run is not bound to its exact matched baseline run and comparison phase.')
            other=selection.get('selections',{}).get(run.get('momentum_source_method_for_selection'),{})
            other_score=other.get('score')
            expected='kl_'+str(chosen.get('candidate_id'))+'_rho_'+str(other.get('candidate_id'))
            if (parent!='stochastic-kl' or run.get('momentum_source_method_for_selection')!='smoothing-fw'
                or other.get('status')!='selected' or isinstance(other_score,bool) or not isinstance(other_score,(int,float))
                or not math.isfinite(other_score) or run['candidate_id']!=expected
                or run.get('source_candidate_id')!=chosen.get('candidate_id')
                or run.get('momentum_source_candidate_id')!=other.get('candidate_id')):
                errors.append(run['run_id']+': momentum supplement does not inherit both frozen tuning parents.')
        elif run['candidate_id']!=chosen.get('candidate_id'):
            errors.append(run['run_id']+': final candidate differs from frozen tuning choice.')
    return list(dict.fromkeys(errors))


def _summary_counts(runs):
    attempted=[r for r in runs if r['attempted']]
    return {'expected_units':len(runs),'attempted_units':len(attempted),
            'terminal_units':sum(r.get('execution_terminal',False) for r in runs),
            'valid_final_units':sum(r['eligible_final_output'] for r in runs),
            'failed_or_invalid_units':sum(r['status'] in ('failed','invalid','infrastructure_failure') for r in runs),
            'missing_or_pending_units':sum(not r.get('execution_terminal',False) for r in runs),
            'valid_budget_met_units':sum(r['eligible_final_output'] and r.get('budget_met',False) for r in runs),
            'native_early_or_max_iter_units':sum(r.get('native_early_stop',False) or r.get('native_max_iter_exhausted',False) for r in runs),
            'missing_checkpoint_observations':sum(len(r.get('missing_checkpoint_targets',r['checkpoints'])) for r in runs),
            'warning_count':sum(r.get('warning_count',0) for r in runs),
            'known_inner_stopping_failures':sum(r.get('counters',{}).get('inner_stopping_failures') or 0 for r in runs),
            'attempted_units_with_unknown_inner_counts':sum(r.get('counters',{}).get('inner_stopping_failures') is None for r in attempted),
            'underflow_steps':sum(r.get('counters',{}).get('underflow_steps') or 0 for r in runs)}


def endpoint_aggregates(runs):
    """Keep actual terminal endpoints separate from prescribed-window curves."""
    grouped=defaultdict(list)
    for run in runs:
        if run['phase']!='tuning':
            grouped[(run['phase'],run['initialization'],run['family'],run['n'],run['noise'],run['method'],run['candidate_id'],run['instance'])].append(run)
    fixed=[]
    timing_keys=('preparation_seconds','solver_seconds','repair_seconds','exact_diagnostic_seconds',
                 'final_float64_validation_seconds','serialization_seconds','total_elapsed_seconds_before_final_metadata_write')
    memory_keys=('cpu_process_peak_rss_kib','cuda_peak_allocated_bytes','cuda_peak_reserved_bytes')
    for key,members in grouped.items():
        units={}
        for run in members:units.setdefault(replicate_key(run),run)
        units=list(units.values())
        valid=[r for r in units if r['eligible_final_output']]
        timed=[r for r in units if r.get('execution_terminal') and r.get('provenance_valid')]
        endpoint={metric:stats([r.get(metric) for r in valid]) for metric in
                  ('rounded_loss_float64','raw_loss_float64','repair_l1_float64','actual_solver_seconds','final_time_to_output')}
        for source,prefix in (('final_raw_residuals','raw_'),('final_rounded_residuals','rounded_')):
            endpoint.update({prefix+metric:stats([r.get(source,{}).get(metric) for r in valid]) for metric in
                             ('row_l1','col_l1','col_l2','both_l1','mass_error','min_entry')})
        resources={'timing_'+metric:stats([r.get('timing',{}).get(metric) for r in timed]) for metric in timing_keys}
        resources.update({'memory_'+metric:stats([r.get('memory',{}).get(metric) for r in timed]) for metric in memory_keys})
        maxima={metric:max((r.get('memory',{}).get(metric) for r in timed if r.get('memory',{}).get(metric) is not None),default=None) for metric in memory_keys}
        fixed.append(dict(zip(('phase','initialization','family','n','noise','method','candidate_id','instance'),key)) |
                     {'comparison_phase':members[0].get('comparison_phase',key[0]),'prescribed_runs':len(members),
                      'budgets':sorted({r['budget'] for r in members}),'counts':_summary_counts(units),'metrics':endpoint,'resources':resources,
                      'resource_units':len(timed),'memory_maxima':maxima,'problem_count':1,
                      'uncertainty_axis':'within-instance stochastic/initialization dispersion; unique deterministic product start contributes once'})
    broad_groups=defaultdict(list)
    for row in fixed:broad_groups[tuple(row[k] for k in ('phase','initialization','family','n','noise','method','candidate_id'))].append(row)
    broad=[]
    for key,rows in broad_groups.items():
        if len(rows)==1:
            broad.append(dict(rows[0]));continue
        row=dict(zip(('phase','initialization','family','n','noise','method','candidate_id'),key))
        row.update(comparison_phase=rows[0]['comparison_phase'],problem_count=len(rows),
                   prescribed_runs=sum(x['prescribed_runs'] for x in rows),budgets=sorted({b for x in rows for b in x['budgets']}),
                   counts={name:sum(x['counts'][name] for x in rows) for name in rows[0]['counts']},
                   resource_units=sum(x['resource_units'] for x in rows),
                   uncertainty_axis='across independent instances after within-instance medians; no independent-dataset claim for fixed BMMC')
        for group in ('metrics','resources'):
            row[group]={metric:stats([x[group][metric]['median'] for x in rows]) for metric in rows[0][group]}
        row['memory_maxima']={metric:max((x['memory_maxima'][metric] for x in rows if x['memory_maxima'][metric] is not None),default=None) for metric in memory_keys}
        broad.append(row)
    return fixed,broad


def evidence_inventory(out,phase_counts,manifests):
    root=Path(__file__).resolve().parent
    artifacts={
        'protocol_coverage':root/'reports/PROTOCOL_COVERAGE.md',
        'oracle_diagnostics':root/'reports/a100_diagnostics/report.json',
        'true_gap_vs_iteration':root/'reports/a100_diagnostics/true_smoothed_gap_vs_iteration.png',
        'true_gap_vs_oracle_work':root/'reports/a100_diagnostics/true_smoothed_gap_vs_probes.png',
        'theory_replicates':root/'reports/a100_theory/theory_replicates.json'}
    sources={key:{'available':path.is_file(),'path':str(path),'relative_path':os.path.relpath(path,out),
                  'sha256':digest(path) if path.is_file() else None} for key,path in artifacts.items()}
    required={key:None for key in ('final_product','final_randomized','small_pg')}
    campaign_path=root/'configs/stage_c_campaign.json'
    if campaign_path.is_file() and any(r.get('campaign_config_sha256')==digest(campaign_path) for r in manifests):
        campaign=read_json(campaign_path)
        required={key:campaign.get('expected_counts',{}).get(key) for key in required}
        supplement_path=root/'configs/momentum_ablation.json'
        if supplement_path.is_file():
            supplement=read_json(supplement_path)
            if supplement.get('parent_campaign_config_sha256')==digest(campaign_path):
                required.update({key:value['run_count'] for key,value in supplement.get('phases',{}).items()})
    coverage={key:{'prescribed_count':count,'provided_count':phase_counts.get(key,{}).get('expected',0),
                   'terminal_count':phase_counts.get(key,{}).get('terminal',0)} for key,count in required.items()}
    return {'decision':'not_automatically_opened','required_stage_c_phase_coverage':coverage,'sources':sources,
            'quality_evidence':'Current rounded checkpoint curves and independently checked final float64 endpoint summaries, with failures and native early termination retained.',
            'oracle_evidence':'Use the dedicated same-device cost/error/regret diagnostics. Kernel call counts are not interchangeable with native sample counts.',
            'memory_evidence':'Tables report observed instrumented process peaks, including preparation/diagnostics. Small-instance peaks do not establish capacity or speed at larger sizes.',
            'decision_rule':'Complete the prescribed Stage C evidence; require correct kernels, a useful measured oracle cost/error trade-off, competitive feasible outputs, and memory headroom. No arbitrary quality threshold or automatic winner is introduced.',
            'theory_caveat':'Dedicated figures use true gradients. Practical checkpoint gaps are sparse diagnostics, not a full-trajectory theorem average. Tuned constants and row projection are not new theorem guarantees.'}


def stat_text(item):
    if item['median'] is None:return '—'
    return fmt(item['median'])+' ['+fmt(item['q25'])+', '+fmt(item['q75'])+']'


def final_tables(report):
    rows=report['final_endpoint_aggregates']
    if not rows:return []
    lines=['','Final saved-output summaries (median [Q1, Q3]). These are actual terminal outputs, not outputs moved to the nominal budget. Native early outputs retain their actual times; missing later checkpoint windows stay missing. Quality statistics exclude invalid finals, whose counts remain explicit. Timing and memory include provenance-verified terminal failures.','',
           '| Phase / data | Method | Valid / prescribed units | Budget (s) | Rounded loss, float64 | Raw loss, float64 | Actual solver (s) | Final output time (s) |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    def label(r):return f"{r['phase']} / {r['family']} n={r['n']} noise={r['noise']}"
    for r in rows:
        m=r['metrics'];c=r['counts']
        lines.append(f"| {label(r)} | {method_name(r['method'])} | {c['valid_final_units']}/{c['expected_units']} | {','.join(map(fmt,r['budgets']))} | "+' | '.join(stat_text(m[k]) for k in ('rounded_loss_float64','raw_loss_float64','actual_solver_seconds','final_time_to_output'))+' |')
    lines += ['','| Phase / data | Method | Raw row L1 | Raw column L1 | Raw column L2 | Rounded both L1 | Repair displacement L1 |',
              '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {label(r)} | {method_name(r['method'])} | "+' | '.join(stat_text(r['metrics'][k]) for k in ('raw_row_l1','raw_col_l1','raw_col_l2','rounded_both_l1','repair_l1_float64'))+' |')
    lines += ['','Time components in seconds. Repair and diagnostic columns are cumulative instrumentation costs; the time-to-output axis adds only the current output repair. Native callback totals overlap these components and are not added twice. Additional Python/native preparation overhead remains in the detailed timing dictionary and total elapsed time.','',
              '| Phase / data | Method | Preparation | Solver | Repairs | Diagnostics | Final validation | Serialization | Total instrumented |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {label(r)} | {method_name(r['method'])} | "+' | '.join(stat_text(r['resources']['timing_'+k]) for k in ('preparation_seconds','solver_seconds','repair_seconds','exact_diagnostic_seconds','final_float64_validation_seconds','serialization_seconds','total_elapsed_seconds_before_final_metadata_write'))+' |')
    lines += ['','Peak memory is the maximum observed process peak across the summarized runs, including diagnostics; it is not isolated solver workspace.','',
              '| Phase / data | Method | GPU allocated max (GiB) | GPU reserved max (GiB) | CPU RSS max (MiB) | Failed/invalid | Pending | Native early/max-iter | Warnings | Underflow steps | Inner failures / unknown units |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        c=r['counts'];mem=r['memory_maxima']
        vals=[fmt(mem[k]/scale) if mem[k] is not None else '—' for k,scale in (('cuda_peak_allocated_bytes',2**30),('cuda_peak_reserved_bytes',2**30),('cpu_process_peak_rss_kib',1024))]
        lines.append(f"| {label(r)} | {method_name(r['method'])} | "+' | '.join(vals)+f" | {c['failed_or_invalid_units']} | {c['missing_or_pending_units']} | {c['native_early_or_max_iter_units']} | {c['warning_count']} | {c['underflow_steps']} | {c['known_inner_stopping_failures']} known / {c['attempted_units_with_unknown_inner_counts']} unknown |")
    return lines


def make_markdown(report):
    lines=["# Campaign results and progress","",report["scope_statement"],"",
           f"Status: **{report['status']}**. Expected runs: {len(report['runs'])}; attempted: {report['attempted_runs']}; terminal records: {report['terminal_runs']}; valid final outputs: {report['valid_final_outputs']}.","",
           "Execution completion includes failed terminal trials; output validity and selection readiness are separate report fields. Saved raw/rounded plan marginals are independently recomputed in NumPy float64 and compared to runner validation; final objective values come from the runner float64 evaluator. Checkpoint curves retain the trace evaluations in the recorded solver precision (float32 in the A100 campaign); they are distinct from final float64 validation. Missing and failed runs carry +infinite tuning penalties. Any such run makes its whole candidate ineligible. Eligible candidate scores are the flat median of prescribed final rounded losses divided by their fixed product-plan losses. Warnings and inner stopping failures remain visible and do not alone disqualify a valid output.","",
           "| Phase | Expected | Attempted | Valid final | Failed/invalid/infrastructure | Missing/pending | Early native stop/max-iter | Missing checkpoint observations |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for phase,row in report["phase_counts"].items():
        lines.append("| "+phase+" | "+" | ".join(str(row[key]) for key in ("expected","attempted","valid_final","failures","pending","native_early","missing_checkpoints"))+" |")
    lines += final_tables(report)
    if report["candidate_scores"]:
        lines += ["","| Method | Candidate | Eligible | Normalized flat-median score | Valid / prescribed runs | Inner failures | Warnings |","|---|---|---|---:|---:|---:|---:|"]
        for row in report["candidate_scores"]:
            lines.append(f"| {method_name(row['method'])} | {row['candidate_id']} | {row['eligible']} | {fmt(row['score'])} | {row['valid_runs']}/{row['prescribed_runs']} | {row['known_inner_stopping_failures']} known; {row['runs_with_unknown_inner_counts']} unknown runs | {row['warning_count']} |")
    if report["selection_audit"]:
        lines += ["","Authoritative selection audit (no selection is made from held-out data):","","| Method | Supplied choice | Audit | Pending tuning runs |","|---|---|---|---:|"]
        for row in report["selection_audit"]:
            selected=row["authoritative_selection"]
            lines.append(f"| {method_name(row['method'])} | {selected.get('candidate_id') if selected else 'not supplied'} | {row['selection_audit']} | {row['pending_runs']} |")
    lines += ["","Every run, including failures and unavailable final metrics, is listed in [runs.csv](runs.csv) and report.json. The CSV includes actual rounded float64 loss, raw row/column residuals, true solver-budget coverage, timing, memory, inner failures, warnings, and failure penalties. Native stopping flags never imply convergence; native conditional samples are not converted into sketch-probe counts.","",
              "Replicate statistics use unique stochastic/initialization units at a fixed instance. Identical deterministic product starts contribute once, so they never acquire artificial error bars. Across different problem instances, plots use one within-instance median per problem and show the median/IQR across those problem values. A single fixed BMMC instance instead shows stochastic/initialization variation. IQRs describe dispersion, not confidence intervals or theorem expectations.","",
              "Checkpoint summaries contain observed outputs only; missing windows are never filled or interpolated. Horizontal bars show IQR of actual observation times at a common requested window, never an assigned nominal timestamp. Open plot markers indicate incomplete coverage. Native terminal outputs produced before the budget remain recorded at their actual time in the run table and are not extended to later checkpoints. Final failure penalties and all missing-coverage counts remain separate from descriptive statistics over available checkpoints.","",
              "No fitted slopes, best-so-far output selection, deterministic duplicate error bars, or GPU/CPU algorithmic speedup claims are produced.",""]
    bad=[row for row in report["runs"] if row["issues"] or row.get("warning_count",0) or (row.get("counters",{}).get("inner_stopping_failures") or 0)]
    if bad:
        lines += ["Run issues and numerical events:","","| Run | Status | Inner failures | Warnings | Issue / source |","|---|---|---:|---:|---|"]
        for row in bad:
            issue="; ".join(row["issues"]) or row.get("failures_file") or "Native warnings summarized in report.json"
            lines.append(f"| {row['run_id']} | {row['status']} | {fmt(row.get('counters',{}).get('inner_stopping_failures'))} | {row.get('warning_count',0)} | {issue.replace('|','/')} |")
    evidence=report["stage_d_evidence"]
    lines += ["", "Stage D assessment remains an evidence review; this report does not automatically open the scale-up gate.", "",
              "| Requirement | Evidence and limitation |", "|---|---|",
              "| Feasible quality | "+evidence["quality_evidence"]+" |",
              "| Oracle trade-off | "+evidence["oracle_evidence"]+" |",
              "| Memory | "+evidence["memory_evidence"]+" |", "", evidence["decision_rule"], "", evidence["theory_caveat"]]
    lines += ["", "| Required final phase | Prescribed runs | Supplied runs | Terminal receipts |", "|---|---:|---:|---:|"]
    for phase,coverage in evidence["required_stage_c_phase_coverage"].items():
        lines.append(f"| {phase} | {fmt(coverage['prescribed_count'])} | {coverage['provided_count']} | {coverage['terminal_count']} |")
    lines.append("")
    for name,source in evidence["sources"].items():
        lines.append("- ["+name.replace("_"," ")+"]("+source["relative_path"]+")" if source["available"] else "- "+name.replace("_"," ")+": evidence not available in this workspace.")
    lines += ["", "Noiseless synthetic instances have known f*=0; noisy synthetic and BMMC objectives are reported as absolute feasible losses, without a certified optimum or clipped reference gaps. Randomized starts pair stochastic and initialization seeds, so their variation is not attributed to initialization alone. The row-projection ablation is separately tuned and cannot isolate the row LMO while ignoring changed constants. Supplemental stochastic KL momentum inherits its selected KL step/tolerance and selected smoothing momentum constant without extra tuning; it is an empirical ablation, not a new theorem guarantee."]
    if report.get("selection_provenance_errors"):
        lines += ["", "Selection provenance errors (comparison figures withheld): "+"; ".join(report["selection_provenance_errors"])]
    if report.get("selection_blocked_methods"):
        lines += ["", "No eligible setting was selected for: "+", ".join(report["selection_blocked_methods"])+". Dependent work is blocked; no fallback is chosen."]
    if report["manifest_omissions"]:
        lines += ["","Manifest omissions or blocked dependent work:","",json.dumps(report["manifest_omissions"],indent=2)]
    lines += ["","Figures:",""]
    lines += [f"- [{name}]({name})" for name in report["figures"]] or ["No valid observed panel or eligible tuning scores are available yet."]
    return "\n".join(lines)+"\n"


def write_flat_csv(path,runs):
    rows=[]
    for run in runs:
        counts=run.get("counters",{});timing=run.get("timing",{});memory=run.get("memory",{})
        raw=run.get("final_raw_residuals",{});rounded=run.get("final_rounded_residuals",{})
        rows.append({key:run.get(key) for key in ("run_id","phase","method","candidate_id","instance","family","n","noise","problem_seed","seed","init_seed","initialization","deterministic","status","attempted","terminal","eligible_final_output","failure_penalty","budget","budget_met","actual_solver_seconds","native_early_stop","native_max_iter_exhausted","rounded_loss_float64","raw_loss_float64","normalized_loss","warning_count","mu0","m_mu0","repair_l1_float64","final_time_to_output","comparison_phase")} |
                    {"raw_row_l1":raw.get("row_l1"),"raw_col_l1":raw.get("col_l1"),"raw_col_l2":raw.get("col_l2"),"rounded_both_l1":rounded.get("both_l1"),
                     "missing_checkpoints":len(run.get("missing_checkpoint_targets",run["checkpoints"])),
                     "completed_updates":counts.get("completed_updates"),"sketch_probes":counts.get("sketch_probes"),"inner_stopping_failures":counts.get("inner_stopping_failures"),
                     "underflow_steps":counts.get("underflow_steps"),"underflow_entries_max":counts.get("underflow_entries_max"),
                     "repair_seconds":timing.get("repair_seconds"),"diagnostic_seconds":timing.get("exact_diagnostic_seconds"),"validation_seconds":timing.get("final_float64_validation_seconds"),
                     "serialization_seconds":timing.get("serialization_seconds"),"total_seconds":timing.get("total_elapsed_seconds_before_final_metadata_write"),
                     "peak_cpu_process_rss_kib":memory.get("cpu_process_peak_rss_kib"),"peak_gpu_allocated_bytes":memory.get("cuda_peak_allocated_bytes"),"peak_gpu_reserved_bytes":memory.get("cuda_peak_reserved_bytes"),
                     "issues":"; ".join(run["issues"])})
    with path.open("w",newline="") as handle:
        if rows:
            writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(safe_json(rows))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,action="append",required=True)
    parser.add_argument("--root",type=Path,action="append",required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--selection",type=Path)
    parser.add_argument("--write-selection",type=Path,help="Explicitly freeze a provenance-bound choice using tuning manifests only.")
    parser.add_argument("--scope",choices=("campaign","parser-smoke"),default="campaign")
    parser.add_argument("--require-complete",action="store_true")
    args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    previous=read_json(out/"report.json") if (out/"report.json").exists() else None
    roots=[root.resolve() for root in args.root]
    records=[];manifest_records=[];omissions=[];seen=set();tuning_inputs=[]
    ledger_index=source_ledgers(roots)
    for manifest_path in args.manifest:
        manifest=read_json(manifest_path)
        if manifest["schema_version"]!=1:raise ValueError("Unsupported manifest schema.")
        sha=digest(manifest_path);phase=manifest["phase"]
        manifest_records.append({"path":str(manifest_path.resolve()),"sha256":sha,"phase":phase,"expected_runs":len(manifest["runs"]),
                                 "campaign_config_sha256":manifest.get("campaign_config_sha256"),"selection_sha256":manifest.get("selection_sha256"),
                                 **{key:manifest.get(key) for key in ("tuning_manifest_sha256","comparison_phase","supplemental_config_sha256",
                                     "supplemental_config_path","parent_campaign_config_sha256","base_manifest_sha256","base_manifest_path")}})
        if phase=="tuning":tuning_inputs.append((manifest_records[-1],manifest))
        omissions.extend(manifest.get("omissions",[]))
        for spec in manifest["runs"]:
            if spec["run_id"] in seen:raise ValueError("Duplicate run_id across input manifests.")
            seen.add(spec["run_id"])
            records.append(inspect_run(spec,roots,phase,sha,args.scope,ledger_index))
    if args.selection and args.write_selection:
        parser.error("Use --selection to inspect an existing choice or --write-selection to create one, not both.")
    initialization_errors=initialization_audit(records)
    selection=read_json(args.selection) if args.selection else None
    candidates,audits=candidate_audit(records,selection)
    selection_path=args.selection
    if args.write_selection:
        if args.scope=="parser-smoke":
            parser.error("Parser-smoke reports cannot freeze an authoritative campaign selection.")
        if len(tuning_inputs)!=1:
            parser.error("Selection writing requires exactly one complete frozen tuning manifest input.")
        if not all(row["execution_terminal"] and row["provenance_valid"] for row in records if row["phase"]=="tuning"):
            parser.error("Selection requires terminal, provenance-verified records for every prescribed tuning run; no provisional choice is allowed.")
        if args.write_selection.exists():
            parser.error("Selection file already exists; use a new filename to preserve the frozen choice.")
        selection=build_selection(candidates,*tuning_inputs[0])
        args.write_selection.parent.mkdir(parents=True,exist_ok=True)
        atomic_json(args.write_selection,selection)
        selection_path=args.write_selection
        candidates,audits=candidate_audit(records,selection)
    selection_provenance_errors=selection_provenance(manifest_records,records,selection,digest(selection_path) if selection_path else None,args.scope)
    fixed,broad,duplicates=aggregate(records)
    fixed_endpoints,endpoints=endpoint_aggregates(records)
    phase_counts={}
    for phase in sorted({row["phase"] for row in records}):
        group=[row for row in records if row["phase"]==phase]
        phase_counts[phase]={"expected":len(group),"attempted":sum(row["attempted"] for row in group),
                             "valid_final":sum(row["eligible_final_output"] for row in group),"terminal":sum(row["execution_terminal"] for row in group),
                             "failures":sum(row["status"] in ("failed","invalid","infrastructure_failure") for row in group),
                             "pending":sum(not row["terminal"] for row in group),
                             "native_early":sum(row.get("native_early_stop",False) or row.get("native_max_iter_exhausted",False) for row in group),
                             "missing_checkpoints":sum(len(row.get("missing_checkpoint_targets",row["checkpoints"])) for row in group)}
    execution_complete=bool(records) and all(row["execution_terminal"] for row in records)
    provenance_valid=all(row["provenance_valid"] for row in records) and not selection_provenance_errors and not any(row["selection_audit"]=="MISMATCH" for row in audits)
    all_outputs_valid=bool(records) and all(row["eligible_final_output"] for row in records)
    complete=execution_complete and provenance_valid and not omissions
    selection_ready=args.scope=="campaign" and bool(tuning_inputs) and all(row["execution_terminal"] and row["provenance_valid"] for row in records if row["phase"]=="tuning")
    issues=any(row.get("warning_count",0) or row.get("counters",{}).get("inner_stopping_failures") or row.get("counters",{}).get("underflow_steps") for row in records)
    status=("execution_complete_with_failed_or_invalid_outputs" if not all_outputs_valid else "complete_provided_manifests_with_numerical_issues" if issues else "complete_provided_manifests") if complete else "partial"
    generated=figures(broad,candidates,out,args.scope) if not selection_provenance_errors else []
    scope="Parser/aggregation smoke test using preserved CPU outputs; not A100 campaign evidence." if args.scope=="parser-smoke" else "Manifest-driven report for the supplied campaign phases. Completion refers only to these manifests; it does not assert that every protocol phase or the full study is complete. Statistics are descriptive and retain failed-run and missing-checkpoint coverage."
    report={"schema_version":1,"generated_utc":datetime.now(timezone.utc).isoformat(),"scope":args.scope,"scope_statement":scope,"status":status,
            "execution_complete":execution_complete,"provenance_valid":provenance_valid,"all_outputs_valid":all_outputs_valid,"selection_ready":selection_ready,
            "initialization_hash_audit_errors":initialization_errors,"manifests":manifest_records,"manifest_omissions":omissions,"input_roots":[str(x) for x in roots],"runs":records,
            "attempted_runs":sum(row["attempted"] for row in records),"terminal_runs":sum(row["terminal"] for row in records),"valid_final_outputs":sum(row["eligible_final_output"] for row in records),
            "phase_counts":phase_counts,"candidate_scores":candidates,"selection_audit":audits,"authoritative_selection":selection,
            "selection_sha256":digest(selection_path) if selection_path else None,"selection_path":str(selection_path) if selection_path else None,
            "selection_provenance_errors":selection_provenance_errors,
            "selection_blocked_methods":[method for method,item in (selection or {}).get("selections",{}).items() if item["status"]=="no_valid_setting"],"fixed_problem_aggregates":fixed,"across_problem_aggregates":broad,
            "fixed_problem_final_endpoints":fixed_endpoints,"final_endpoint_aggregates":endpoints,
            "stage_d_evidence":evidence_inventory(out,phase_counts,manifest_records),"duplicate_scientific_units":duplicates,"figures":generated,"full_study_complete":False,"fitted_slopes":None,"speedup_claim":None,
            "script_sha256":digest(__file__),"command":sys.argv,"plot_library":"matplotlib "+matplotlib.__version__}
    write_flat_csv(out/"runs.csv",records)
    (out/"report.md").write_text(make_markdown(safe_json(report)))
    atomic_json(out/"report.json",report)
    if previous:
        for name in set(previous.get("figures",[]))-set(generated):
            path=out/name
            if path.parent==out and path.suffix in (".png",".pdf") and path.is_file():path.unlink()
    with (out/"report_history.jsonl").open("a") as handle:
        handle.write(json.dumps({"generated_utc":report["generated_utc"],"status":status,"valid_final_outputs":report["valid_final_outputs"],"expected_runs":len(records),"report_sha256":digest(out/"report.json")})+"\n")
    print(json.dumps({"report":str(out/"report.md"),"status":status,"expected_runs":len(records),"valid_final_outputs":report["valid_final_outputs"],"figures":len(generated)},indent=2))
    return 1 if args.require_complete and not complete else 0


if __name__=="__main__":
    raise SystemExit(main())
