"""Select and summarize the paper-normalized exact Quiroga campaign."""
import argparse
import json
from pathlib import Path

import numpy as np


SCALES = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4)
WEIGHTS = (0.1, 1.0, 10.0, 100.0)
CANDIDATES = tuple(
    (f"s{si}_w{wi}", scale, weight)
    for si, scale in enumerate(SCALES)
    for wi, weight in enumerate(WEIGHTS)
)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata_json"].item()))
        values = dict(
            fidelity=z["process_fidelity_proxy"].copy(),
            loss=z["measurement_loss"].copy(), tp=z["tp_violation"].copy(),
            factor=z["final_factor"].copy(), steps=z["checkpoint_steps"].copy(),
        )
    return meta, values


def select(args):
    rows, initial, subsets = [], set(), set()
    for label, scale, weight in CANDIDATES:
        path = args.root / label / "final.npz"
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
            final_measurement_loss_sum=float(values["loss"][-1]),
            final_tp_violation=float(values["tp"][-1]),
            optimizer_seconds=meta["optimizer_seconds"],
        ))
    if len(initial) != 1 or len(subsets) != 1:
        raise RuntimeError("completed candidates did not share U0 and fixed subset")
    finite = [r for r in rows if r["status"] == "finite"]
    admissible = [r for r in finite if r["final_tp_violation"] <= 1.0 and r["final_fidelity"] >= 0.01]
    if admissible:
        chosen = min(admissible, key=lambda r: (-r["final_fidelity"], r["final_tp_violation"]))
        rule = "maximum fidelity among finite runs with TP<=1 and fidelity>=0.01"
    else:
        bridge = [r for r in finite if r["final_tp_violation"] <= 5.0 and r["final_fidelity"] > r["initial_fidelity"]]
        if not bridge:
            report = dict(status="no_scientifically_promising_candidate", candidates=rows)
            args.save.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps(report, indent=2, allow_nan=False)); raise SystemExit(42)
        chosen = min(bridge, key=lambda r: (-r["final_fidelity"], r["final_tp_violation"]))
        rule = "fallback maximum fidelity among improving finite runs with TP<=5; confirmation gate required"
    report = dict(
        status="selected", selection_seed=0, steps=500, subset_size=8192,
        formulation="Quiroga sum loss, exact spectral numerator, published matrix-gradient-times-U update",
        selection_rule=rule, selected_label=chosen["label"],
        selected_eta_scale=chosen["eta_scale"], selected_tp_weight=chosen["tp_weight"],
        selected_beta0=100.0, initial_factor_sha256=next(iter(initial)),
        subset_symbols_sha256=next(iter(subsets)), candidates=rows,
    )
    args.save.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


def summarize(args):
    selection = json.loads(args.selection.read_text())
    rows, keyed = [], {}
    for method in ("quiroga", "frames"):
        for seed in (1, 2, 3):
            directory = args.results / f"{method}_seed{seed}"
            meta, values = load(directory / "final.npz")
            verification = json.loads((directory / "verification.json").read_text())
            if meta["n_steps"] != 10000 or meta["subset_seed"] != seed:
                raise RuntimeError(f"wrong budget/seed in {directory}")
            row = dict(method=method, subset_seed=seed,
                       final_fidelity=float(values["fidelity"][-1]),
                       final_measurement_loss_sum=float(verification["fixed_subset_measurement_loss_sum"]),
                       native_saved_measurement_loss=float(values["loss"][-1]),
                       final_tp_violation=float(values["tp"][-1]),
                       optimizer_seconds=float(meta["optimizer_seconds"]),
                       subset_symbols_sha256=meta["subset_symbols_sha256"],
                       initial_factor_sha256=meta["initial_factor_sha256"],
                       result=str(directory / "final.npz"))
            rows.append(row); keyed[(method, seed)] = row
    for seed in (1, 2, 3):
        for key in ("subset_symbols_sha256", "initial_factor_sha256"):
            if keyed[("quiroga", seed)][key] != keyed[("frames", seed)][key]:
                raise RuntimeError(f"{key} mismatch for seed {seed}")
    aggregates = {}
    for method in ("quiroga", "frames"):
        subset = [r for r in rows if r["method"] == method]
        aggregates[method] = {
            key: dict(mean=float(np.mean([r[key] for r in subset])),
                      std_ddof1=float(np.std([r[key] for r in subset], ddof=1)))
            for key in ("final_fidelity", "final_measurement_loss_sum", "final_tp_violation", "optimizer_seconds")
        }
    report = dict(status="complete", formulation=selection["formulation"],
                  selected_eta_scale=selection["selected_eta_scale"],
                  selected_tp_weight=selection["selected_tp_weight"], selected_beta0=100.0,
                  paired_fixed_subsets_verified=True, rows=rows, aggregates=aggregates)
    args.save.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
    s=sub.add_parser("select"); s.add_argument("--root",type=Path,required=True); s.add_argument("--save",type=Path,required=True); s.set_defaults(func=select)
    s=sub.add_parser("summarize"); s.add_argument("--selection",type=Path,required=True); s.add_argument("--results",type=Path,required=True); s.add_argument("--save",type=Path,required=True); s.set_defaults(func=summarize)
    a=p.parse_args(); a.func(a)


if __name__ == "__main__":
    main()
