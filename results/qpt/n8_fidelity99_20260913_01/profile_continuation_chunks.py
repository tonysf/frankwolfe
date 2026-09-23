"""Optional continuation preparation; run only within an A100 allocation.

Compare existing chunk settings, requiring identical final factors, momentum,
sample hashes and metrics before selecting the faster setting. No source edits.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--provenance", type=Path, required=True)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir()
parent = json.loads(args.provenance.read_text())
command = parent["command"]
common = command[:command.index("--steps")]
common[0] = sys.executable
cases = []
reference = None
equivalent = True
for repeat in range(2):
    for chunk in (10, 1):
        run = list(common)
        run[run.index("--chunk-steps") + 1] = str(chunk)
        save = args.output / f"chunk{chunk}_repeat{repeat}.npz"
        run += ["--steps", "100", "--save", str(save), "--quiet"]
        subprocess.run(run, cwd=args.source, check=True)
        with np.load(save, allow_pickle=False) as z:
            metadata = json.loads(str(z["metadata_json"].item()))
            arrays = {key: z[key].copy() for key in (
                "final_factor", "gradient_estimate", "process_fidelity_proxy",
                "tp_violation", "measurement_loss", "smoothed_gap")}
        if reference is None:
            reference = (metadata["batch_symbols_sha256"], arrays)
        equal = (metadata["batch_symbols_sha256"] == reference[0]
                 and all(np.array_equal(value, reference[1][key]) for key, value in arrays.items()))
        equivalent = equivalent and equal
        steady = metadata["optimizer_seconds"] + metadata["sampling_transfer_seconds"]
        cases.append(dict(chunk_steps=chunk, repeat=repeat, equivalent=equal,
            optimizer_seconds=metadata["optimizer_seconds"],
            sampling_transfer_seconds=metadata["sampling_transfer_seconds"],
            host_preparation_seconds=metadata["host_preparation_seconds"],
            wall_seconds=metadata["wall_seconds"], steady_seconds=steady,
            compiled_memory_estimate_bytes=metadata["compiled_memory_estimate_bytes"],
            result=str(save)))
means = {chunk: float(np.mean([c["steady_seconds"] for c in cases if c["chunk_steps"] == chunk]))
         for chunk in (10, 1)}
chosen = 1 if equivalent and means[1] < .95 * means[10] else 10
report = dict(cases=cases, equivalent=equivalent, selected_chunk_steps=chosen,
    mean_optimizer_plus_blocking_transfer_seconds=means,
    timing_scope="100 steps; host preparation overlaps GPU work and is not added",
    decision="Require bitwise equivalence and at least 5% measured improvement to choose chunk1.")
(args.output / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2), flush=True)
