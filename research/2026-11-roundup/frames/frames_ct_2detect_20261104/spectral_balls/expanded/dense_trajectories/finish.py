"""Validate the dense cohort and publish the paper's observed trajectories."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
EXPANDED = ROOT.parent
PAPER = EXPANDED / "reports" / "paper_section"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    while True:
        state = read(ROOT / "progress.json")
        if state["status"] != "running":
            break
        if not args.wait:
            raise RuntimeError("Dense evaluation is still running")
        time.sleep(30)
    if state["status"] != "completed" or len(state["runs"]) != 132:
        raise RuntimeError(f"Dense cohort is incomplete: {state['status']}")
    if any(row.get("exact_original_probe_match") is not True for row in state["runs"].values()):
        raise ValueError("Not every dense run passed original-probe verification")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(EXPANDED.parent.parent / "deps")
    subprocess.run([sys.executable, str(EXPANDED / "analyze.py"),
                    "--root", str(ROOT), "--no-plots"], env=env, check=True)
    old = read(EXPANDED / "reports" / "summary.json")
    new = read(ROOT / "reports" / "summary.json")
    if new["coverage"]["physical_arm_status_counts"] != {"completed": 132}:
        raise ValueError("Dense analysis is missing completed runs")
    if new["cohorts"] != old["cohorts"]:
        raise ValueError("Dense and original endpoint summaries differ")
    source = PAPER / "ct_experiments.tex"
    text = source.read_text()
    if r"d_C" in text or text.count(r"\textbf{") != 14:
        raise ValueError("The requested table formatting has changed")
    old_pdf_hash = digest(PAPER / "sframes_ct_section.pdf")
    subprocess.run([sys.executable, str(PAPER / "plot_trajectories.py"), "--root", str(ROOT)],
                   env=env, check=True)
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                    "sframes_ct_section.tex"], cwd=PAPER, check=True)
    audits = [read(ROOT / "runs" / name / "validation.json") for name in state["runs"]]
    completion = dict(
        completed_utc=datetime.now(timezone.utc).isoformat(), runs=len(audits),
        diagnostic_observations=sum(row["logged_iterations"] for row in audits),
        exact_original_probe_match=all(row["exact_original_probe_match"] for row in audits),
        identical_endpoint_summaries=True,
        plot_observations="Every iteration, without resampling or interpolation",
        previous_paper_pdf_sha256=old_pdf_hash,
        paper_pdf=str(PAPER / "sframes_ct_section.pdf"),
        paper_pdf_sha256=digest(PAPER / "sframes_ct_section.pdf"),
        paper_source_sha256=digest(source),
        plotting_source_sha256=digest(PAPER / "plot_trajectories.py"),
    )
    pending = ROOT / "completion.json.tmp"
    pending.write_text(json.dumps(completion, indent=2) + "\n")
    pending.replace(ROOT / "completion.json")
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
