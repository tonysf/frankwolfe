"""Tiny numerical archive checks only; no local tuning sweep."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as c
import audit_results as audit
import run_tuning as runner


def trial(**changes):
    result = dict(id="tiny", channel=0, initialization_seed=200000, method="printed-adafgd",
                  alpha=5e-4, eta_scale=1., n_steps=2, metrics_frequency=1)
    result.update(changes)
    return result


def plan(t=None):
    return dict(schema_version=1, campaign_id="tiny", phase="calibration", gate=c.GATE,
                source=dict(repo=str(c.REPO), commit="a" * 40), resources=dict(minimum_seconds_before_new_trial=120),
                trials=[t or trial()])


@pytest.mark.parametrize("changes", [{"channel": 100}, {"channel": -1}, {"initialization_seed": True},
                                    {"eta_scale": 0}, {"alpha": float("nan")}, {"n_steps": 0},
                                    {"id": "../bad"}, {"method": "armijo-fgd"}])
def test_reject_unsafe_trial(changes):
    with pytest.raises(ValueError):
        c.validate_plan(plan(trial(**changes)))


def test_validation_requires_frozen_configuration():
    p = plan(trial(channel=100))
    p["phase"] = "validation"
    p["selection"] = {"calibration_audit_sha256": "b" * 64,
                      "configuration": {k: p["trials"][0][k] for k in c.SELECTION_KEYS}}
    c.validate_plan(p)
    p["trials"][0]["eta_scale"] = 2.
    with pytest.raises(ValueError, match="retunes"):
        c.validate_plan(p)


def test_no_overwrite_atomic_json(tmp_path):
    path = tmp_path / "inventory.json"
    c.write_json(path, {"one": 1})
    with pytest.raises(FileExistsError):
        c.write_json(path, {"two": 2})
    assert json.loads(path.read_text()) == {"one": 1}
    c.write_json(path, {"three": 3}, True)
    assert json.loads(path.read_text()) == {"three": 3}
    assert not path.with_name("inventory.json.pending").exists()


def test_validation_checks_actual_calibration_audit(tmp_path):
    p = plan(trial(channel=100))
    p["phase"] = "validation"
    p["selection"] = {"configuration": {k: p["trials"][0][k] for k in c.SELECTION_KEYS}}
    report = dict(kind="qpt_quiroga_tuning_audit", audit_passed=True, partial_mode=False,
                  missing_trial_ids=[], trial_ids_without_inventory_checksum=[], provenance={"phase": "calibration"},
                  cases=[dict(trial=trial(), channel=0, status="complete", gate={"pass": True})])
    path = tmp_path / "calibration.json"
    c.write_json(path, report)
    p["selection"]["calibration_audit_sha256"] = c.base.file_hash(path)
    c.check_selection(p, path)
    report["cases"][0]["gate"]["pass"] = False
    c.write_json(path, report, True)
    with pytest.raises(ValueError, match="checksum"):
        c.check_selection(p, path)
    p["selection"]["calibration_audit_sha256"] = c.base.file_hash(path)
    with pytest.raises(ValueError, match="did not pass"):
        c.check_selection(p, path)


@pytest.mark.parametrize("method,eta", [("printed-adafgd", 1.), ("printed-adafgd", 1e300),
                                       ("exact-linesearch-fgd", 1.)])
def test_tiny_native_archive_roundtrip(tmp_path, method, eta):
    t = trial(method=method, eta_scale=eta)
    p = plan(t)
    provenance = dict(campaign_id="tiny", phase="calibration", plan_sha256="b" * 64,
                      source_commit="a" * 40,
                      source_sha256={str(path.relative_to(c.REPO)): c.base.file_hash(path)
                                     for path in (c.REPO / "paper").rglob("*.py")})
    record = runner.execute(p, t, tmp_path, provenance)
    path = tmp_path / record["json"]
    case = audit.audit_case(path, p, t, provenance, np)
    assert case["trial"] == t
    assert case["status"] == ("failed" if eta == 1e300 else "complete")
    assert record["json_sha256"] == c.base.file_hash(path)
    assert case["gate"]["pass"] is False
    assert case["work_accounting"]["verified_rows"] >= 128
    with pytest.raises(FileExistsError):
        runner.execute(p, t, tmp_path, provenance)
    damaged = copy.deepcopy(p)
    damaged_trial = dict(t, alpha=t["alpha"] * 2)
    with pytest.raises(ValueError, match="configuration"):
        audit.validate_archive(path, damaged, damaged_trial, provenance, np)


def test_printed_work_rejects_unexplained_row_charge():
    arrays = {"printed_adafgd_checkpoint_steps": np.array([0, 1]),
              "printed_adafgd_cumulative_data_row_accesses": np.array([0, 256]),
              "printed_adafgd_optimizer_seconds": np.array([0., 1.])}
    with pytest.raises(ValueError, match="failed-attempt"):
        audit.check_work(arrays, "printed_adafgd", {"status": "complete"}, np)


def test_complete_inventory_audit_and_tamper(tmp_path, monkeypatch):
    t, p = trial(), plan()
    provenance = dict(campaign_id="tiny", phase="calibration", plan_sha256="b" * 64, source_commit="a" * 40,
                      source_sha256={str(path.relative_to(c.REPO)): c.base.file_hash(path)
                                     for path in (c.REPO / "paper").rglob("*.py")})
    runtime_provenance = dict(provenance, slurm_array_task_id=0, slurm_job_id="unit-test", hostname="unit-test")
    record = runner.execute(p, t, tmp_path, runtime_provenance)
    inventory = dict(schema_version=1, kind="qpt_quiroga_tuning_inventory", **runtime_provenance,
                     channel=0, state="complete", trials=[dict(trial=t, **record)])
    c.write_json(tmp_path / "inventory.json", inventory)
    c.write_json(tmp_path / "plan.json", p)
    monkeypatch.setattr(c, "provenance", lambda *args: provenance)
    monkeypatch.setattr(c, "source_hashes", lambda *args: provenance["source_sha256"])
    args = ["--results-dir", str(tmp_path), "--plan", str(tmp_path / "plan.json"), "--plan-sha256", "b" * 64]
    assert audit.main(args + ["--output", str(tmp_path / "audit.json")]) == 0
    report = json.loads((tmp_path / "audit.json").read_text())
    assert report["audited_trials"] == 1 and report["missing_trial_ids"] == []
    inventory["trials"][0]["npz_sha256"] = "0" * 64
    c.write_json(tmp_path / "inventory.json", inventory, True)
    with pytest.raises(ValueError, match="path/hash"):
        audit.main(args + ["--output", str(tmp_path / "audit_bad.json")])
    assert not (tmp_path / "audit_bad.json").exists()
