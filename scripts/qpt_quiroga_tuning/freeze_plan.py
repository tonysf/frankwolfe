#!/usr/bin/env python3
"""Add reviewed clean-commit file hashes to a draft, publishing without overwrite."""
import argparse
import json
from pathlib import Path
import common as c


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-audit", type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(args.draft.read_text())
    c.validate_plan(plan)
    c.check_selection(plan, args.calibration_audit)
    c.require("sha256" not in plan["source"], "Draft already frozen")
    plan["source"]["sha256"] = c.source_hashes(c.REPO, plan["source"]["commit"])
    plan["frozen_utc"] = c.base.now()
    c.write_json(args.output, plan)
    print(c.base.file_hash(args.output))


if __name__ == "__main__":
    raise SystemExit(main())
