"""Bounded report fixture: synthetic traces, real layout, no optimization/publication."""

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("baseline_report_fixture_target", ROOT / "update_report.py")
U = importlib.util.module_from_spec(spec)
spec.loader.exec_module(U)


def main():
    stage = Path(tempfile.mkdtemp(prefix="report_fixture_", dir=ROOT))
    report = U.module("fixture_base_report", U.FEASIBILITY / "report.py")
    analysis = U.module("fixture_analysis", report.LEGACY / "analyze.py")
    helpers = U.module("fixture_joint_helpers", U.FEASIBILITY / "analysis.py")
    plotting = U.module("fixture_plotting", U.FEASIBILITY / "report_trajectories.py")
    previous = U.read(report.LEGACY / "reports" / "summary.json")
    previous["spring_correction"] = {"fixture_only": True, "marker": "must remain identical"}
    legacy, _, _ = report.archived_inputs()
    live = {p: U.digest(p) for p in (U.PAPER.parent / report.APPENDIX,
                                    U.PAPER / (report.STEM + ".pdf"),
                                    report.LEGACY / "reports" / "summary.json")}
    original_runs, corrected = {}, {}
    cases = []
    targets = previous["targets"]["targets"]

    def synthetic(case, method, corrected=False):
        arm = case["arms"][method if method != "tuned_sframes" else "sframes"]
        meta = U.read(Path(arm["source_directory"]) / "metadata.json")
        seed = case["seed"] if corrected or method not in ("palm", "sframes_full") else 202
        name = f"{case['slice_id']}_{case['n_train']}_{seed}_{method}_{'new' if corrected else 'old'}"
        folder = stage / "synthetic_runs" / name
        folder.mkdir(parents=True, exist_ok=True)
        meta = copy.deepcopy(meta)
        meta["config"].update(run_id=name, method=method, seed=seed)
        first = U.read(Path(arm["source_directory"]) / "trace.json")[0]
        costs = [0, 12000, 68000, 71000]
        if corrected and method == "palm":
            step = 14 * case["n_train"]
            costs = list(range(0, U.CAP + step, step))
        trace = []
        for i, cost in enumerate(costs):
            row = dict(first, iteration=i, forward_views=cost, backward_views=cost,
                       gradient_calls=i * 2, curvature_operator_calls=i * 12 if method == "palm" and corrected else 0)
            row["oracle_calls"] = row["gradient_calls"] + row["curvature_operator_calls"]
            for metric in ("raw_loss", "feasible_loss", "validation_loss", "reference_relative_error"):
                row[metric] = first[metric] * (1 if i == 0 else .55 if cost <= U.CAP else .01)
            if method in ("palm", "prox_sarah", "spring_sarah"):
                row["d_distance"] = 0
            else:
                row["d_distance"] = .01 if i else 0
            trace.append(row)
        # Joint SFRAMES plotting requires its exact cap as in the production protocol.
        if method == "tuned_sframes":
            trace = trace[:3]
            trace[-1].update(forward_views=U.CAP, backward_views=U.CAP)
        U.write(folder / "metadata.json", meta)
        U.write(folder / "trace.json", trace)
        U.write(folder / "result.json", {"fixture_only": True})
        return dict(config=meta["config"], meta=meta, trace=trace, source_directory=str(folder), status="completed",
                    canonical_numerical_source_sha256=arm["canonical_numerical_source_sha256"])

    for case in previous["cases"]:
        records = {}
        for method in analysis.VARIANTS:
            run = synthetic(case, method)
            if method not in ("palm", "sframes_full") or case["seed"] == 202:
                original_runs[(*U.key(case), method)] = dict(run, trace=run["trace"][:-1])
            if method == "spring_sarah":
                case["arms"][method]["source_directory"] = run["source_directory"]
            if method in U.METHODS:
                corrected[(*U.key(case), method)] = synthetic(case, method, corrected=True)
            if method in ("sframes", "prox_sarah"):
                eligible = dict(run, trace=run["trace"][:-1])
                records[method] = helpers.record_run(eligible, Path(run["source_directory"]), analysis, targets, U.CAP)
            case["arms"][method]["probes"] = analysis.saved_probes(
                run["trace"], U.CAP, previous["budgets"]["original_forward_view_cap"])
        run = synthetic(case, "tuned_sframes")
        records["tuned_sframes"] = helpers.record_run(run, Path(run["source_directory"]), analysis, targets, U.CAP)
        cases.append(dict(slice_id=case["slice_id"], seed=case["seed"], n_train=case["n_train"],
                          group=case["group"], runs=records))
    frozen = dict(cases=cases, cohort=previous["design"], targets=targets, by_n={"360": {}, "1080": {}},
                  status="completed_validated", view_budget_per_direction=U.CAP, fixture_only=True)
    before = copy.deepcopy(previous)
    old_joint = copy.deepcopy(frozen)
    updated = U.replace_summary(previous, corrected, analysis, {"fixture": True})
    joint = U.joint_summary(frozen, corrected, analysis, helpers, {"fixture": True})
    assert previous == before and frozen == old_joint
    assert updated["spring_correction"] == previous["spring_correction"]
    assert "palm" not in updated["design"]["deterministic_variants"]
    assert updated["historical_design"] == previous["design"]
    for case in updated["cases"]:
        for method in U.METHODS:
            endpoint = case["arms"][method]["probes"]["view_cap"]
            assert endpoint["forward_views"] <= U.CAP
            assert endpoint["raw_loss"] > case["arms"][method]["probes"]["endpoint"]["raw_loss"]
    for case in joint["cases"]:
        endpoint = case["runs"]["prox_sarah"]["endpoint"]
        assert endpoint["forward_views"] == 68000 and endpoint["relative_raw_loss"] == .55
    tune = U.read(U.FEASIBILITY / "selection.json")
    selection = {"selected": {"c_L": .5}}
    source = U.source_text(report, updated, legacy, tune, joint, selection)
    assert "(27,18)" in source and "(48,32)" in source and "0.400802" in source
    assert "in all six group summaries" not in source
    (stage / report.BODY).write_text(source)
    for name in (report.STEM + ".tex", report.LINK, "joint_selection.pdf", "joint_selection.png"):
        shutil.copyfile(U.PAPER / name, stage / name)
    appendix = U.appendix_text((U.PAPER.parent / report.APPENDIX).read_text(), ROOT, U.PAPER, joint, selection)
    assert "not yet available" not in appendix and U.START in appendix
    (stage / report.APPENDIX).write_text(appendix)
    # The predecessor's idempotent publisher still regenerates its marked table.
    # Keep that table historical so doing so cannot revert this correction.
    chain = stage / "appendix_chain"
    (chain / "paper").mkdir(parents=True)
    (chain / report.APPENDIX).write_text((U.PAPER.parent / report.APPENDIX).read_text())
    report.supplement_results(chain / "paper", tune, joint)
    corrected_appendix = U.appendix_text((chain / report.APPENDIX).read_text(), ROOT, U.PAPER, joint, selection)
    (chain / report.APPENDIX).write_text(corrected_appendix)
    report.supplement_results(chain / "paper", tune, joint)
    assert (chain / report.APPENDIX).read_text() == corrected_appendix
    slices = sorted({c["slice_id"] for c in previous["cases"]})
    spring_runs = {k: v for k, v in original_runs.items() if k[3] == "spring_sarah"}
    count, observations = U.plot(report, plotting, analysis, original_runs, corrected, previous, slices,
                                 (360, 1080), joint, stage, spring_runs)
    assert count == 448 and observations < 4000
    pages = U.compile_pdf(stage, report)

    # Journal replay, foreign-edit refusal, and rollback without any live destination.
    transaction = stage / "transaction"
    transaction.mkdir()
    payload = transaction / "report_stage_fixture"
    backup = transaction / "report_previous"
    payload.mkdir()
    backup.mkdir()
    destination = transaction / "live"
    destination.mkdir()
    names = ("paper.txt", "spring_audit.json", "feasibility_audit.json")
    dests = {name: destination / name for name in names}
    for name, path in dests.items():
        path.write_text("old " + name)
        shutil.copyfile(path, backup / name)
        (payload / name).write_text("new " + name)
    audit = dict(stage=str(payload), backup=str(backup),
                 original_sha256={name: U.digest(path) for name, path in dests.items()},
                 published_sha256={name: U.digest(payload / name) for name in names})
    pending = transaction / "report_update.pending.json"
    U.write(pending, audit)
    dests[names[0]].write_text("foreign edit")
    try:
        U.finish(transaction, report, audit, dests, pending)
        raise AssertionError("Foreign edit was accepted")
    except ValueError as exc:
        assert "Unrelated live edit" in str(exc)
    shutil.copyfile(backup / names[0], dests[names[0]])
    original_copy = report.atomic_copy
    calls = 0
    def fail_once(source, target):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected failure")
        original_copy(source, target)
    report.atomic_copy = fail_once
    try:
        U.finish(transaction, report, audit, dests, pending)
        raise AssertionError("Injected failure did not occur")
    except RuntimeError:
        assert all(U.digest(path) == audit["original_sha256"][name] for name, path in dests.items())
    finally:
        report.atomic_copy = original_copy
    shutil.copyfile(payload / names[0], dests[names[0]])  # interrupted transaction: mixed old/new
    U.finish(transaction, report, audit, dests, pending)
    assert not pending.exists() and all(U.digest(path) == audit["published_sha256"][name] for name, path in dests.items())
    assert all(U.digest(path) == sha for path, sha in live.items())
    result = dict(status="passed", fixture_only=True, no_optimization=True, no_live_publication=True,
                  stage=str(stage), pages=pages, figures=2, tables=1, physical_traces=count,
                  synthetic_observations=observations, checks=["input immutability", "corrected SPRING preserved",
                  "both baseline arms replaced", "joint comparator recomputed", "overshoot excluded",
                  "two figures and one table compiled", "upstream appendix regeneration unchanged",
                  "foreign edit rejected", "rollback", "mixed-state replay"],
                  updater_sha256=U.digest(ROOT / "update_report.py"), validator_sha256=U.digest(Path(__file__)))
    U.write(ROOT / "report_fixture_validation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
