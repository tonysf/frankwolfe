"""Calibration and aggregation for exact-norm Quiroga adaFGD at n=4."""
import argparse
import json
from pathlib import Path

import numpy as np


QUIROGA = tuple(
    (f"c{ci}_w{wi}", scale, weight)
    for ci, scale in enumerate((0.1, 1.0, 10.0))
    for wi, weight in enumerate((0.01, 0.1, 1.0))
)
BETAS = (10.0, 100.0, 1000.0, 10000.0)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata_json"].item()))
        values = dict(
            steps=z["checkpoint_steps"].copy(),
            fidelity=z["process_fidelity_proxy"].copy(),
            loss=z["measurement_loss"].copy(),
            tp=z["tp_violation"].copy(),
            factor=z["final_factor"].copy(),
        )
    return meta, values


def choose(args):
    rows, initial, subsets = [], set(), set()
    for label, scale, weight in QUIROGA:
        path = args.root / "quiroga" / label / "final.npz"
        if not path.exists():
            rows.append(dict(label=label, eta_scale=scale, tp_weight=weight, status="failed"))
            continue
        meta, values = load(path)
        finite = all(np.all(np.isfinite(v)) for v in values.values())
        initial.add(meta["initial_factor_sha256"]); subsets.add(meta["subset_symbols_sha256"])
        rows.append(dict(
            label=label, eta_scale=scale, tp_weight=weight,
            status="finite" if finite else "nonfinite",
            initial_fidelity=float(values["fidelity"][0]),
            final_fidelity=float(values["fidelity"][-1]),
            final_measurement_loss=float(values["loss"][-1]),
            final_tp_violation=float(values["tp"][-1]),
            optimizer_seconds=meta["optimizer_seconds"],
        ))
    if len(initial) != 1 or len(subsets) != 1:
        raise RuntimeError("Quiroga candidates did not share initialization and subset")
    finite = [r for r in rows if r["status"] == "finite"]
    admissible = [r for r in finite if r["final_tp_violation"] <= 1.0]
    if admissible:
        selected = min(admissible, key=lambda r: (-r["final_fidelity"], r["final_tp_violation"]))
        rule = "maximum fidelity among finite candidates with terminal TP <= 1"
    else:
        selected = min(finite, key=lambda r: (r["final_tp_violation"], -r["final_fidelity"]))
        rule = "fallback lowest terminal TP among finite candidates; confirmation gate still required"

    frame_rows = []
    for beta in BETAS:
        label = f"beta{int(beta)}"
        meta, values = load(args.root / "frames" / label / "final.npz")
        if meta["initial_factor_sha256"] not in initial or meta["subset_symbols_sha256"] not in subsets:
            raise RuntimeError(f"FRAMES mismatch in {label}")
        frame_rows.append(dict(
            label=label, beta0=beta, final_fidelity=float(values["fidelity"][-1]),
            final_measurement_loss=float(values["loss"][-1]),
            final_tp_violation=float(values["tp"][-1]), optimizer_seconds=meta["optimizer_seconds"],
        ))
    frame_ok = [r for r in frame_rows if r["final_tp_violation"] <= 1.0]
    if not frame_ok:
        raise RuntimeError("no admissible FRAMES beta")
    selected_frame = min(frame_ok, key=lambda r: (-r["final_fidelity"], r["final_tp_violation"]))
    report = dict(
        status="selected", selection_seed=0, subset_size=8192,
        quiroga_steps=100, quiroga_selection_rule=rule,
        selected_eta_scale=selected["eta_scale"], selected_tp_weight=selected["tp_weight"],
        selected_quiroga_label=selected["label"], selected_beta0=selected_frame["beta0"],
        initial_factor_sha256=next(iter(initial)), subset_symbols_sha256=next(iter(subsets)),
        quiroga_runs=rows, frames_runs=frame_rows,
    )
    args.save.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


def summarize(args):
    selection = json.loads(args.selection.read_text())
    rows, by_key = [], {}
    for method in ("quiroga", "frames"):
        for seed in (1, 2, 3):
            directory = args.results / f"{method}_seed{seed}"
            meta, values = load(directory / "final.npz")
            if meta["n_steps"] != 10000 or meta["subset_seed"] != seed:
                raise RuntimeError(f"wrong budget/seed in {directory}")
            row = dict(
                method=method, subset_seed=seed,
                final_fidelity=float(values["fidelity"][-1]),
                final_measurement_loss=float(values["loss"][-1]),
                final_tp_violation=float(values["tp"][-1]),
                optimizer_seconds=float(meta["optimizer_seconds"]),
                subset_symbols_sha256=meta["subset_symbols_sha256"],
                initial_factor_sha256=meta["initial_factor_sha256"],
                result=str(directory / "final.npz"),
            )
            rows.append(row); by_key[(method, seed)] = row
    for seed in (1, 2, 3):
        if by_key[("quiroga", seed)]["subset_symbols_sha256"] != by_key[("frames", seed)]["subset_symbols_sha256"]:
            raise RuntimeError(f"subset mismatch for seed {seed}")
        if by_key[("quiroga", seed)]["initial_factor_sha256"] != by_key[("frames", seed)]["initial_factor_sha256"]:
            raise RuntimeError(f"initialization mismatch for seed {seed}")
    aggregates = {}
    for method in ("quiroga", "frames"):
        subset = [r for r in rows if r["method"] == method]
        aggregates[method] = {
            key: dict(mean=float(np.mean([r[key] for r in subset])),
                      std_ddof1=float(np.std([r[key] for r in subset], ddof=1)))
            for key in ("final_fidelity", "final_measurement_loss", "final_tp_violation", "optimizer_seconds")
        }
    report = dict(status="complete", selected_eta_scale=selection["selected_eta_scale"],
                  selected_tp_weight=selection["selected_tp_weight"],
                  selected_beta0=selection["selected_beta0"], paired_fixed_subsets_verified=True,
                  rows=rows, aggregates=aggregates)
    args.save.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select"); s.add_argument("--root", type=Path, required=True); s.add_argument("--save", type=Path, required=True); s.set_defaults(func=choose)
    s = sub.add_parser("summarize"); s.add_argument("--selection", type=Path, required=True); s.add_argument("--results", type=Path, required=True); s.add_argument("--save", type=Path, required=True); s.set_defaults(func=summarize)
    a = p.parse_args(); a.func(a)


if __name__ == "__main__":
    main()
