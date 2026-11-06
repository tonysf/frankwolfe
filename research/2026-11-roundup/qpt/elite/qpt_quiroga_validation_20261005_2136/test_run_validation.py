"""Synthetic inventory/control-flow tests; no optimization is performed."""
import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("validation_runner", Path(__file__).with_name("run_validation.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture
def frozen(tmp_path):
    shared = copy.deepcopy(runner.SHARED)
    shared["row_epochs"] = 2000
    manifest = {"schema_version": 1, "kind": "qpt_method_native_selection_manifest",
                "regime_id": "n2_full_128", "selection_rule_id": "gate_first_common_physical_metrics_v1",
                "source_commit": runner.COMMIT,
                "printed_config_id": "n2f-q-a5em1-e1em2-v1", "frames_config_id": "n2f-f-a5em5-p025-v1",
                "selected_candidates": {
                    "printed-adafgd": {"id": "n2f-q-a5em1-e1em2-v1", "alpha": 0.5, "eta_scale": 0.01},
                    "frames": {"id": "n2f-f-a5em5-p025-v1", "alpha0": 0.00005}},
                "shared_calibration_configuration": shared}
    path = tmp_path / "selection_manifest.json"
    runner.write_json(path, manifest)
    plan = runner.frozen_plan(path)
    return plan, path, runner.make_inventory(plan, path)


def test_frozen_inventory_has_disjoint_seeds_and_exact_counts(frozen):
    plan, path, inventory = frozen
    assert len(inventory) == 24
    assert len({r["trial_id"] for r in inventory}) == 24
    assert {r["channel"] for r in inventory} == set(range(100, 108))
    assert {r["initialization_seed"] for r in inventory} == set(range(201000, 201071, 10))
    assert len({r["sampling_seed"] for r in inventory}) == 24
    assert len({r["cohort_id"] for r in inventory}) == 1
    for row in inventory:
        assert row["array_index"] == row["channel"] - 100
        assert row["sampling_seed"] == 300000 + 100 * row["channel"] + row["sampling_index"]
    assert plan["selection"]["manifest_sha256"] == runner.file_hash(path)
    assert plan["matched_budget"]["nominal_row_accesses_per_method"] == 1280000


def test_all_commands_match_pinned_parser_and_selection(frozen):
    from paper.experiments.qpt_quiroga_direct_comparison import _build_parser
    plan, _, inventory = frozen
    parser = _build_parser()
    for trial in inventory:
        args = parser.parse_args(runner.command_for(plan, trial, Path("/not-written"))[3:])
        assert args.split == "validation"
        assert args.selection_manifest_sha256 == plan["selection"]["manifest_sha256"]
        assert args.printed_config_id == trial["printed_config_id"]
        assert args.frames_config_id == trial["frames_config_id"]
        assert args.printed_tp_weight_per_row == 0.5
        assert args.frames_initial_tp_weight_per_row == 0.00005
        assert args.eta_scale == 0.01
        assert args.row_epochs == 10000 and args.checkpoint_epochs == 100
        assert args.channel_seed == trial["channel"]
        assert args.initialization_seed == trial["initialization_seed"]
        assert args.sampling_seed == trial["sampling_seed"]
        assert args.require_clean and args.fail_on_method_failure


def test_changed_manifest_or_selected_parameters_are_rejected(frozen):
    plan, path, _ = frozen
    changed = copy.deepcopy(plan)
    changed["selection"]["selected"]["frames"]["alpha0"] = 0.005
    with pytest.raises(ValueError, match="Frozen parameters"):
        runner.make_inventory(changed, path)
    with path.open("ab") as stream:
        stream.write(b"\n")
    with pytest.raises(ValueError, match="manifest hash"):
        runner.make_inventory(plan, path)


@pytest.mark.parametrize("field,value", [("channels", list(range(8))),
                                        ("sampling_indices", [0]),
                                        ("shared", {"row_epochs": 2000}),
                                        ("matched_budget", {}),
                                        ("expected_counts", {})])
def test_changed_protocol_or_replication_is_rejected(frozen, field, value):
    plan, path, _ = frozen
    altered = copy.deepcopy(plan)
    altered[field] = value
    with pytest.raises(ValueError):
        runner.make_inventory(altered, path)


def test_manifest_ids_and_calibration_settings_must_agree(frozen):
    _, path, _ = frozen
    manifest = json.loads(path.read_text())
    manifest["printed_config_id"] = "different"
    with pytest.raises(ValueError, match="IDs disagree"):
        runner.selected_configs(manifest)
    manifest = json.loads(path.read_text())
    manifest["shared_calibration_configuration"]["tau"] = 4
    with pytest.raises(ValueError, match="shared settings differ"):
        runner.selected_configs(manifest)


def test_plan_writer_refuses_overwrite(tmp_path):
    path = tmp_path / "frozen.json"
    runner.write_json(path, {"a": 1})
    with pytest.raises(FileExistsError):
        runner.write_json(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 1}


def fake_execution(monkeypatch, frozen, tmp_path, *, bad_archive=False):
    plan, manifest, inventory = frozen
    out = tmp_path / "channel-100"
    out.mkdir()
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic-test")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID", "synthetic-test")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setattr(runner.subprocess, "check_output",
                        lambda command, **kw: runner.COMMIT + "\n" if command[1] == "rev-parse" else "")
    calls = []
    def fake_run(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=2 if len(calls) == 1 else 0)
    def fake_validate(*args, **kwargs):
        if bad_archive:
            raise ValueError("synthetic invalid archive")
        return {"synthetic": True}
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "validate_archive", fake_validate)
    args = SimpleNamespace(array_index=0, deadline_epoch=runner.time.time() + 1000, out=out)
    code = runner.run_channel(args, plan, inventory, json.dumps(plan).encode(), manifest.read_bytes())
    return code, calls, json.loads((out / "inventory.json").read_text())


def test_valid_method_failure_is_retained_and_other_trials_continue(monkeypatch, frozen, tmp_path):
    code, calls, inventory = fake_execution(monkeypatch, frozen, tmp_path)
    assert code == 2 and len(calls) == 3
    assert inventory["status"] == "all_trials_archived_with_method_failures"
    assert inventory["trial_status_counts"] == {"method_failure_archived": 1, "complete": 2}


def test_invalid_archive_stops_but_keeps_pending_inventory(monkeypatch, frozen, tmp_path):
    code, calls, inventory = fake_execution(monkeypatch, frozen, tmp_path, bad_archive=True)
    assert code == 3 and len(calls) == 1
    assert inventory["status"] == "incomplete_infrastructure_or_invalid_archive"
    assert inventory["trial_status_counts"] == {"infrastructure_failure_or_invalid_archive": 1,
                                                 "pending_not_run": 2}


def test_launcher_destinations_match_runner_constants():
    import re
    text = Path(__file__).with_name("validation.sbatch").read_text()
    commit = re.search(r"^commit=(\w+)$", text, re.M).group(1)
    assert commit == runner.COMMIT
    output = re.search(r'^output_root="([^"]+)"$', text, re.M).group(1).replace("${commit}", commit)
    assert output == runner.OUTPUT_ROOT
    staged = re.search(r'^staged="([^"]+)"$', text, re.M).group(1).replace("${output_root}", output)
    assert staged == str(Path(output) / runner.STAGED)
    assert "#SBATCH --array=0-7%1" in text
    assert "#SBATCH --time=00:20:00" in text
