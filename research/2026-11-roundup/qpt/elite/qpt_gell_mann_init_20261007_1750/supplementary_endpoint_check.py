#!/usr/bin/env python3
"""Cross-check native endpoint and final trace metrics against archived factors.

This supplements, rather than changes, the immutable eight-case GM audit.
No optimizer is run and no source or archived result is modified. Failed cases
whose factor has undefined metrics remain explicitly unverified for those
metrics; they are never relabeled successful. Run --self-test for pure checks.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

RTOL = 1e-10
ATOL = 1e-13
METRICS = ("measurement_loss_sum", "measurement_loss_mean", "rmse",
           "tp_violation", "normalized_tp_violation", "process_fidelity")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def compare_metric(value, native, trace, status):
    """Compare finite metrics; do not pretend undefined failures were verified."""
    # JSON stores nonfinite final traces as null in the method endpoint.
    trace_value = trace if finite(trace) else None
    require(native == trace_value, "Native endpoint and final trace disagree")
    if value is None:
        require(status == "failed", "A successful endpoint metric is undefined")
        return {"state": "undefined_failed_factor_not_comparable", "recomputed": None,
                "native_endpoint": native, "trace_final": trace_value,
                "native_absolute_error": None, "trace_absolute_error": None}
    require(finite(value) and finite(native) and finite(trace), "Finite recomputation has nonfinite archived metric")
    for label, archived in (("native endpoint", native), ("final trace", trace)):
        require(abs(archived - value) <= ATOL + RTOL * abs(value),
                f"{label} differs from recomputation: {archived!r} versus {value!r}")
    return {"state": "verified", "recomputed": value, "native_endpoint": native,
            "trace_final": trace, "native_absolute_error": abs(native - value),
            "trace_absolute_error": abs(trace - value)}


def run(audit_path, repo, output):
    require(not output.exists(), "Refusing to overwrite supplementary report")
    audit_hash = sha(audit_path)
    audit = json.loads(audit_path.read_text())
    require(audit["kind"] == "qpt_gell_mann_audit"
            and audit["audit_passed"] is True and audit["expected_trials"] == audit["audited_trials"] == 8
            and not audit["missing_trial_ids"] and not audit["partial_mode"]
            and not audit["trial_ids_without_inventory_checksum"], "A complete verified 8/8 audit is required")
    cases = audit["cases"]
    require(len(cases) == len({c["trial_id"] for c in cases}) == 8, "Wrong or duplicate trial count")
    provenance = audit["provenance"]
    require(subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
            == provenance["source_commit"], "Source commit differs")
    require(not subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip(),
            "Source checkout is dirty")
    for relative, expected_hash in provenance["source_sha256"].items():
        require(sha(repo / relative) == expected_hash, "Source hash differs: " + relative)
    sys.path.insert(0, str(repo))
    import numpy as np
    from paper.experiments.qpt_quiroga_campaign import evaluate_common_metrics
    from paper.experiments.qpt_quiroga_sensing import QuirogaSensingData, QuirogaSensingDesign
    for function in (evaluate_common_metrics, QuirogaSensingData, QuirogaSensingDesign):
        require(Path(sys.modules[function.__module__].__file__).resolve().is_relative_to(repo.resolve()),
                "Imported code is outside the audited source checkout")
    checked = []
    for case in cases:
        path = Path(case["json_path"])
        require(sha(path) == case["json_sha256"] and sha(path.with_suffix(".npz")) == case["npz_sha256"],
                "Artifact differs from full audit: " + case["trial_id"])
        metadata = json.loads(path.read_text())
        method = case["method"]
        prefix = method.replace("-", "_")
        endpoint = metadata["methods"][method]
        require(endpoint["status"] == case["status"], "Audited/native status differs")
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as archive:
            require(json.loads(str(archive["metadata_json"].item())) == metadata, "JSON/NPZ metadata differs")
            require(np.array_equal(archive["active_rows"], np.arange(128)), "Not the full 128-row design")
            data = QuirogaSensingData(QuirogaSensingDesign(2), archive["truth_factor"], "noiseless")
            require(np.allclose(data.all_observations(), archive["active_observations"], rtol=1e-12, atol=1e-14),
                    "Targets do not match archived truth")
            error = None
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    metrics = evaluate_common_metrics(data, archive[prefix + "_final_factor"],
                                                      archive["active_rows"], archive["active_observations"], 0.)
                recomputed = {key: float(metrics[key]) for key in METRICS}
                if not all(finite(v) for v in recomputed.values()):
                    raise FloatingPointError("Nonfinite recomputed metric")
            except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
                require(endpoint["status"] == "failed", "Successful factor metric recomputation failed")
                recomputed = dict.fromkeys(METRICS)
                error = f"{type(exc).__name__}: {exc}"
            comparisons = {}
            for key in METRICS:
                try:
                    comparisons[key] = compare_metric(recomputed[key], endpoint[key],
                                                      float(archive[prefix + "_" + key][-1]), endpoint["status"])
                except ValueError as exc:
                    raise ValueError(f"{case['trial_id']} / {key}: {exc}") from exc
        checked.append({"trial_id": case["trial_id"], "method": method, "channel": case["channel"],
                        "status": endpoint["status"], "comparisons": comparisons,
                        "recomputation_error": error, "json_sha256": case["json_sha256"],
                        "npz_sha256": case["npz_sha256"],
                        "fully_verified": all(c["state"] == "verified" for c in comparisons.values())})
    require(sha(audit_path) == audit_hash, "Full audit changed during supplementary check")
    report = {"schema_version": 1, "kind": "qpt_gell_mann_native_endpoint_supplement",
              "all_comparable_checks_passed": True, "checked_trials": len(checked),
              "fully_verified_trials": sum(c["fully_verified"] for c in checked),
              "undefined_failed_trial_ids": [c["trial_id"] for c in checked if not c["fully_verified"]],
              "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "tolerance": {"rtol": RTOL, "atol": ATOL}, "metrics": list(METRICS),
              "audit_path": str(audit_path.resolve()), "audit_sha256": audit_hash,
              "source_commit": provenance["source_commit"], "script_sha256": sha(__file__),
              "cases": checked,
              "qualification": "Native endpoint and final checkpoint values compared with physical metrics recomputed from final factors on the full sensing design. No optimizer runs, source changes, endpoint relabeling, or objective-score comparison."}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("all_comparable_checks_passed", "checked_trials",
                                                  "fully_verified_trials", "undefined_failed_trial_ids")}))


class ComparisonTests(unittest.TestCase):
    def test_finite_match_and_roundoff(self):
        self.assertEqual(compare_metric(1., 1. + 1e-12, 1. + 1e-12, "complete")["state"], "verified")

    def test_recomputation_mismatch(self):
        with self.assertRaises(ValueError):
            compare_metric(1., 1.1, 1.1, "complete")

    def test_native_trace_mismatch(self):
        with self.assertRaises(ValueError):
            compare_metric(1., 1., 1.1, "complete")

    def test_failed_undefined_not_verified(self):
        self.assertEqual(compare_metric(None, None, float("nan"), "failed")["state"],
                         "undefined_failed_factor_not_comparable")
        with self.assertRaises(ValueError):
            compare_metric(None, None, float("nan"), "complete")

    def test_finite_recompute_rejects_null(self):
        with self.assertRaises(ValueError):
            compare_metric(1., None, float("nan"), "failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ComparisonTests))
        return 0 if result.wasSuccessful() else 1
    require(args.audit is not None and args.repo is not None and args.output is not None,
            "--audit, --repo, and --output are required")
    run(args.audit, args.repo, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
