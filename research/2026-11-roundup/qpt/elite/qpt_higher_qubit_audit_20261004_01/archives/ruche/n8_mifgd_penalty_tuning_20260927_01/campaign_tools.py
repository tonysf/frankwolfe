"""Selection and aggregation for coupled fixed-penalty MiFGD tuning."""

import argparse
import json
from pathlib import Path

import numpy as np


CANDIDATES = (
    ("w0p1_e0p01", 0.1, 0.01),
    ("w0p01_e0p1", 0.01, 0.1),
    ("w0p001_e1", 0.001, 1.0),
)


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


def select(args):
    rows, init_hashes, batch_hashes = [], set(), set()
    for label, weight, eta in CANDIDATES:
        path = args.root / label / "final.npz"
        if not path.exists():
            rows.append(dict(label=label, tp_weight=weight, eta=eta, status="failed"))
            continue
        meta, values = load(path)
        if meta.get("optimizer") != "mifgd_fixed_tp" or meta["sampling_seed"] != 0:
            raise RuntimeError(f"invalid metadata in {path}")
        finite = all(np.all(np.isfinite(v)) for v in values.values())
        init_hashes.add(meta["initial_factor_sha256"])
        batch_hashes.add(meta["batch_symbols_sha256"])
        rows.append(dict(
            label=label, tp_weight=weight, eta=eta,
            status="finite" if finite else "nonfinite",
            final_fidelity=float(values["fidelity"][-1]),
            final_measurement_loss=float(values["loss"][-1]),
            final_tp_violation=float(values["tp"][-1]),
            final_factor_norm=float(np.linalg.norm(values["factor"])),
            optimizer_seconds=meta["optimizer_seconds"],
        ))
    if len(init_hashes) != 1 or len(batch_hashes) != 1:
        raise RuntimeError("candidates did not use identical U0 and minibatches")
    admissible = [r for r in rows if r["status"] == "finite"
                  and r["final_tp_violation"] <= 1.0
                  and r["final_fidelity"] >= 0.1]
    chosen = min(admissible, key=lambda r: (-r["final_fidelity"], r["final_tp_violation"], r["eta"])) if admissible else None
    report = dict(
        status="selected" if chosen else "no_admissible_candidate",
        selection_seed=0, steps=20000, mu=0.75,
        candidates=[dict(label=l, tp_weight=w, eta=e) for l, w, e in CANDIDATES],
        admissibility="finite, terminal TP <= 1.0, terminal fidelity >= 0.1",
        selection="maximum terminal fidelity; ties by TP then eta",
        selected_label=None if chosen is None else chosen["label"],
        selected_tp_weight=None if chosen is None else chosen["tp_weight"],
        selected_eta=None if chosen is None else chosen["eta"],
        initial_factor_sha256=next(iter(init_hashes)),
        batch_symbols_sha256=next(iter(batch_hashes)), runs=rows,
    )
    args.save.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    if chosen is None:
        raise SystemExit(42)


def summarize(args):
    selection = json.loads(args.selection.read_text())
    rows, by_key, initial = [], {}, set()
    sources = [("frames", args.frames_root)] + [
        ("fgd", args.results), ("mifgd", args.results)]
    for method, root in sources:
        for seed in (1, 2, 3):
            directory = root / f"{method}_seed{seed}"
            meta, values = load(directory / "final.npz")
            if meta["n_steps"] != 200000 or meta["sampling_seed"] != seed:
                raise RuntimeError(f"wrong budget/seed: {directory}")
            initial.add(meta["initial_factor_sha256"])
            row = dict(
                method=method, sampling_seed=seed,
                final_fidelity=float(values["fidelity"][-1]),
                final_measurement_loss=float(values["loss"][-1]),
                final_tp_violation=float(values["tp"][-1]),
                optimizer_seconds=meta["optimizer_seconds"],
                batch_symbols_sha256=meta["batch_symbols_sha256"],
                result=str(directory / "final.npz"),
            )
            rows.append(row); by_key[(method, seed)] = row
    if len(initial) != 1:
        raise RuntimeError("initial factors differ")
    for seed in (1, 2, 3):
        hashes = {by_key[(m, seed)]["batch_symbols_sha256"] for m in ("frames", "fgd", "mifgd")}
        if len(hashes) != 1:
            raise RuntimeError(f"minibatch mismatch seed {seed}")
    aggregates = {}
    for method in ("frames", "fgd", "mifgd"):
        subset = [r for r in rows if r["method"] == method]
        aggregates[method] = {
            key: dict(mean=float(np.mean([r[key] for r in subset])),
                      std_ddof1=float(np.std([r[key] for r in subset], ddof=1)))
            for key in ("final_fidelity", "final_measurement_loss", "final_tp_violation", "optimizer_seconds")
        }
    report = dict(status="complete", selected_eta=selection["selected_eta"],
                  selected_tp_weight=selection["selected_tp_weight"], mu_mifgd=0.75,
                  paired_minibatches_verified=True, rows=rows, aggregates=aggregates)
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
    final.add_argument("--frames-root", type=Path, required=True)
    final.add_argument("--save", type=Path, required=True)
    final.set_defaults(func=summarize)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
