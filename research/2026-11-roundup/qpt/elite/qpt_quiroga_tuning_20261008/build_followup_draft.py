"""Freeze a selected calibration setting before additional channel evaluation."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KEYS = {"method", "alpha", "eta_scale", "n_steps", "metrics_frequency"}
parser = argparse.ArgumentParser()
parser.add_argument("--phase", choices=("confirmation", "revalidation"), required=True)
parser.add_argument("--method", default="exact-linesearch-fgd")
args = parser.parse_args()

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

selection_path = ROOT / "selection_screen.json"
selection = json.loads(selection_path.read_text())
screen_path = ROOT / "audit_screen.json"
assert digest(screen_path) == selection["audit_sha256"]
screen = json.loads(screen_path.read_text())
assert screen["audit_passed"] and not screen["partial_mode"]
assert not screen["missing_trial_ids"] and not screen["trial_ids_without_inventory_checksum"]
chosen = selection["methods"][args.method]["selected_configuration"]
assert chosen is not None and set(chosen) == KEYS
screen_cases = [c for c in screen["cases"] if {k: c["trial"][k] for k in KEYS} == chosen]
assert len(screen_cases) == 2 and {c["channel"] for c in screen_cases} == {0, 1}
assert all(c["gate"]["pass"] for c in screen_cases)

plan = copy.deepcopy(json.loads((ROOT / "stage/tuning_plan.json").read_text()))
plan.pop("frozen_utc", None)
plan["source"].pop("sha256", None)
plan.pop("preserved_prior_audit", None)
plan.pop("selection_rule", None)
plan["campaign_id"] = "qpt_quiroga_tuning_20261008_" + args.phase
plan["phase"] = "calibration" if args.phase == "confirmation" else "validation"
plan["frozen_screen_selection"] = {
    "screen_audit_sha256": digest(screen_path),
    "selection_file_sha256": digest(selection_path),
    "configuration": chosen,
    "no_retuning": True,
}
if args.phase == "confirmation":
    channels = range(2, 8)
    plan["qualification"] = (
        "Additional calibration confirmation on channels2..7, using exactly the setting selected "
        "on channels0/1. No retuning. Exact factor-ray line search is not printed Quiroga adaFGD. "
        "These channels are additional calibration, not claimed globally unseen."
    )
else:
    calibration_path = ROOT / "audit_confirmation.json"
    calibration = json.loads(calibration_path.read_text())
    assert calibration["audit_passed"] and not calibration["partial_mode"]
    assert not calibration["missing_trial_ids"] and not calibration["trial_ids_without_inventory_checksum"]
    assert len(calibration["cases"]) == 6 and {c["channel"] for c in calibration["cases"]} == set(range(2, 8))
    assert all(c["gate"]["pass"] and {k: c["trial"][k] for k in KEYS} == chosen for c in calibration["cases"])
    plan["selection"] = {"calibration_audit_sha256": digest(calibration_path), "configuration": chosen}
    channels = range(100, 108)
    plan["qualification"] = (
        "Historical revalidation on channels100..107, already used in earlier campaigns, hence "
        "NOT unseen validation. Setting frozen before confirmation and unchanged afterwards. "
        "Exact factor-ray line search is not printed Quiroga adaFGD. Finalheldout1000..1019 untouched."
    )
plan["trials"] = [dict(id=f"c{channel}-frozen-{args.method}", channel=channel,
                       initialization_seed=200000 + 10*channel, **chosen) for channel in channels]
stage = ROOT / (args.phase + "_stage")
stage.mkdir(exist_ok=True)
draft_path = stage / "draft.json"
with draft_path.open("x") as stream:
    json.dump(plan, stream, indent=2, sort_keys=True, allow_nan=False)
    stream.write("\n")
print(draft_path)
