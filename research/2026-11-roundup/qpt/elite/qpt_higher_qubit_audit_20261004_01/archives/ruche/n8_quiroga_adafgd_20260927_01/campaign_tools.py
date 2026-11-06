"""Select and summarize the Quiroga-style adaFGD baseline campaign."""
import argparse
import json
from pathlib import Path

import numpy as np


WEIGHTS = (1e-5, 1e-3, 1e-1, 1e1)
SCALES = (1e2, 1e4, 1e6, 1e8)
CANDIDATES = tuple(
    (f"w{wi}_s{si}", weight, scale)
    for wi, weight in enumerate(WEIGHTS)
    for si, scale in enumerate(SCALES)
)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata_json"].item()))
        values = dict(
            steps=z["checkpoint_steps"].copy(),
            fidelity=z["process_fidelity_proxy"].copy(),
            loss=z["measurement_loss"].copy(),
            tp=z["tp_violation"].copy(),
            rates=z["estimated_gaps"].copy(),
            factor=z["final_factor"].copy(),
        )
    return meta, values


def select(args):
    rows, init_hashes, measurement_hashes, noise_hashes = [], set(), set(), set()
    for label, weight, scale in CANDIDATES:
        path = args.root / label / "final.npz"
        if not path.exists():
            rows.append(dict(label=label, tp_weight=weight, eta_scale=scale, status="failed"))
            continue
        meta, values = load(path)
        if meta.get("optimizer") != "quiroga_adafgd_fixed_tp" or meta["sampling_seed"] != 0:
            raise RuntimeError(f"invalid metadata in {path}")
        finite = all(np.all(np.isfinite(value)) for value in values.values())
        init_hashes.add(meta["initial_factor_sha256"])
        measurement_hashes.add(meta["fixed_measurement_symbols_sha256"])
        noise_hashes.add(meta["fixed_measurement_noise_sha256"])
        rows.append(dict(
            label=label,
            tp_weight=weight,
            eta_scale=scale,
            status="finite" if finite else "nonfinite",
            final_fidelity=float(values["fidelity"][-1]),
            final_measurement_loss=float(values["loss"][-1]),
            final_tp_violation=float(values["tp"][-1]),
            eta_first=float(values["rates"][0]),
            eta_last=float(values["rates"][-1]),
            final_factor_norm=float(np.linalg.norm(values["factor"])),
            optimizer_seconds=meta["optimizer_seconds"],
        ))
    if not init_hashes or len(init_hashes) != 1:
        raise RuntimeError("completed candidates did not use one identical U0")
    if len(measurement_hashes) != 1 or len(noise_hashes) != 1:
        raise RuntimeError("candidates did not use one identical fixed data set")
    admissible = [
        row for row in rows
        if row["status"] == "finite"
        and row["final_tp_violation"] <= 1.0
        and row["final_fidelity"] >= 1e-5
    ]
    chosen = min(
        admissible,
        key=lambda row: (-row["final_fidelity"], row["final_tp_violation"], row["eta_scale"]),
    ) if admissible else None
    report = dict(
        status="selected" if chosen else "no_admissible_candidate",
        selection_seed=0,
        steps=2000,
        candidates=[dict(label=l, tp_weight=w, eta_scale=s) for l, w, s in CANDIDATES],
        admissibility="finite, terminal TP <= 1.0, terminal fidelity >= 1e-5",
        selection="maximum terminal fidelity; ties by TP then eta scale",
        selected_label=None if chosen is None else chosen["label"],
        selected_tp_weight=None if chosen is None else chosen["tp_weight"],
        selected_eta_scale=None if chosen is None else chosen["eta_scale"],
        initial_factor_sha256=next(iter(init_hashes)),
        fixed_measurement_symbols_sha256=next(iter(measurement_hashes)),
        fixed_measurement_noise_sha256=next(iter(noise_hashes)),
        runs=rows,
    )
    args.save.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    if chosen is None:
        raise SystemExit(42)


def summarize(args):
    selection = json.loads(args.selection.read_text())
    rows, initial = [], set()
    for seed in (1, 2, 3):
        directory = args.results / f"quiroga_seed{seed}"
        meta, values = load(directory / "final.npz")
        if meta["n_steps"] != 200000 or meta["sampling_seed"] != seed:
            raise RuntimeError(f"wrong budget/seed: {directory}")
        initial.add(meta["initial_factor_sha256"])
        rows.append(dict(
            method="quiroga_adafgd",
            sampling_seed=seed,
            final_fidelity=float(values["fidelity"][-1]),
            final_measurement_loss=float(values["loss"][-1]),
            final_tp_violation=float(values["tp"][-1]),
            eta_first=float(values["rates"][0]),
            eta_last=float(values["rates"][-1]),
            optimizer_seconds=meta["optimizer_seconds"],
            fixed_measurement_symbols_sha256=meta["fixed_measurement_symbols_sha256"],
            fixed_measurement_noise_sha256=meta["fixed_measurement_noise_sha256"],
            result=str(directory / "final.npz"),
        ))
    if len(initial) != 1 or next(iter(initial)) != selection["initial_factor_sha256"]:
        raise RuntimeError("held-out initial factors differ from calibration")
    keys = ("final_fidelity", "final_measurement_loss", "final_tp_violation", "optimizer_seconds")
    aggregate = {
        key: dict(
            mean=float(np.mean([row[key] for row in rows])),
            std_ddof1=float(np.std([row[key] for row in rows], ddof=1)),
        ) for key in keys
    }
    report = dict(
        status="complete",
        selected_eta_scale=selection["selected_eta_scale"],
        selected_tp_weight=selection["selected_tp_weight"],
        protocol="one fixed seeded measurement set reused for every update within each run",
        comparison_note="FRAMES uses resampled minibatches; results share initialization and budget but are not minibatch-paired",
        paired_minibatches_verified=False,
        rows=rows,
        aggregate=aggregate,
    )
    args.save.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    choose = sub.add_parser("select")
    choose.add_argument("--root", type=Path, required=True)
    choose.add_argument("--save", type=Path, required=True)
    choose.set_defaults(func=select)
    final = sub.add_parser("summarize")
    final.add_argument("--selection", type=Path, required=True)
    final.add_argument("--results", type=Path, required=True)
    final.add_argument("--save", type=Path, required=True)
    final.set_defaults(func=summarize)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
