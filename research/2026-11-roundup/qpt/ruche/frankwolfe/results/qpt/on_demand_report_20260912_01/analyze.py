"""Build the experiment report from preserved JSON and scalar trajectories."""
import json
from pathlib import Path
import statistics
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from paper.experiments.qpt_observation_noise import fixed_row_noise

profiles = []
for directory in ("v100_profile", "a100_profile", "batch_profile"):
    for path in sorted((HERE / directory).glob("*_benchmark.json")):
        report = json.loads(path.read_text())
        if report["status"] != "complete":
            continue
        timings = [r for r in report["runs"] if r["purpose"] == "timing" and r["status"] == "success"]
        memory = next((r["memory"] for r in report["runs"] if r["purpose"] == "memory" and r["status"] == "success"), None)
        cfg = report["configuration"]
        seconds = statistics.median(r["optimizer_seconds"] for r in timings)
        entry = dict(n=report["n_qubits"], device=timings[0]["device_kind"],
                     backend=cfg["measurement_backend"], batch=cfg["batch_size"], steps=cfg["steps"],
                     optimizer_seconds=seconds, runner_seconds=statistics.median(r["runner_wall_seconds"] for r in timings),
                     observations_per_second=cfg["batch_size"]*cfg["steps"]/seconds,
                     input_observation_bytes=report["memory_estimates"]["structured_host_observation_bytes"],
                     source=str(path.relative_to(HERE)))
        if memory:
            entry.update(cpu_rss_mib=memory["cpu_process_peak_rss_bytes"]/2**20,
                         jax_peak_mib=memory["jax_allocator"]["stats"]["peak_bytes_in_use"]/2**20,
                         sampled_vram_mib=memory["external_sampler"]["gpu"]["sampled_peak_process_bytes"]/2**20)
        profiles.append(entry)
profiles.sort(key=lambda row: ("A100" in row["device"], row["n"], row["batch"], row["backend"]))

traces = json.loads((HERE / "traces.json").read_text()) if (HERE / "traces.json").exists() else []
recovery = []
for record in traces:
    if "profile" in record["folder"] or "a100_20260912_02" in record["folder"]:
        continue
    cfg, meta, trace = record["configuration"], record["source_metadata"], record["trace"]
    if cfg["steps"] <= 1000:
        continue
    fidelity = np.asarray(trace["process_fidelity_proxy"])
    best = int(np.argmax(fidelity))
    ids = np.random.default_rng(cfg["metric_seed"]).integers(0, 24**record["n"], size=cfg["metric_samples"])
    noise = fixed_row_noise(ids, meta["noise_seed"], meta["noise_std"])
    recovery.append(dict(n=record["n"], device=record["device"], batch=cfg["batch_size"], steps=cfg["steps"],
                         sigma=meta["noise_std"], beta0=cfg["smoothing_scale"], tau=cfg["tau"],
                         final_fidelity=float(fidelity[-1]), best_fidelity=float(fidelity[best]),
                         best_step=trace["checkpoint_steps"][best],
                         loss=trace["measurement_loss"][-1], tp=trace["tp_violation"][-1],
                         fixed_metric_sample_truth_loss=float(np.mean(noise**2)/2),
                         name=record["name"], folder=record["folder"]))
recovery.sort(key=lambda row: (row["n"], row["batch"], row["beta0"], -row["sigma"]))

summary = dict(profiles=profiles, recovery=recovery)
(HERE / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")

lines = ["# On-demand stochastic-FRAMES QPT: measured scaling through 10 qubits", "",
         "The implementation now executes noisy 10-qubit experiments without an observation table. "
         "High-fidelity 10-qubit reconstruction remains unresolved: the best tested final fidelity was 0.000502%. "
         "The best 8-qubit run reached 17.9% after 50,000 steps. All submitted jobs are terminal; none remains queued or running.", "",
         "All runs use complex128/float64, rank one, exact TP gradients, fixed row-indexed Gaussian noise, "
         "and disabled GPU autotuning. Source is the preserved v4 snapshot based on ba85bff; no commit or push was made. "
         "The new noise recipe is not the old sequential PCG64 realization.", "",
         "Validation: 470 tests passed on Ruche, plus 7 on-demand tests with GPU trajectory checks on each of V100 and A100. "
         "The full local suite passed 414 tests and skipped 56 for unavailable optional dependencies.", "",
         "**Timing and memory**", "",
         "Times are medians of two fresh-process timing runs. Memory comes from a separate monitored run. "
         "Optimizer time includes on-device target generation and excludes compilation, host preparation/transfers, and metrics. "
         "RSS, JAX allocator memory, and sampled driver VRAM measure different things; sampled VRAM can miss brief peaks.", "",
         "| GPU | n | Backend | Batch | Steps | Optimizer s | Host RSS MiB | JAX peak MiB | Sampled VRAM MiB |",
         "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
for row in profiles:
    gpu = "A100 40GB" if "A100" in row["device"] else "V100 32GB"
    lines.append(f"| {gpu} | {row['n']} | {row['backend']} | {row['batch']} | {row['steps']} | "
                 f"{row['optimizer_seconds']:.3f} | {row.get('cpu_rss_mib',float('nan')):.1f} | "
                 f"{row.get('jax_peak_mib',float('nan')):.1f} | {row.get('sampled_vram_mib',float('nan')):.0f} |")
lines += ["", "At n=10/batch32, rank-one takes 14.170 s/1,000 steps on V100 and 6.127 s on A100 (2.31x faster). "
          "Tensor takes 181.206 s and 75.147 s respectively. Rank-one is about 13x faster than tensor on V100.", "",
          "At n=10 on V100, batches 32/128/512 process approximately 2,258/3,926/4,417 sampled rows per optimizer second. "
          "Batch128 achieves 89% of batch512's throughput with a 4.10 GiB allocator peak instead of 16.11 GiB. "
          "Batch32 profiles used beta0=10000 and metrics every 100 steps; batch128/512 profiles used beta0=10240000 "
          "and metrics every 1000 steps. Optimizer timing excludes those metric evaluations. "
          "At batch512, a 10,000-step run took about 19.3 optimizer minutes on V100 and 9.46 minutes on A100; "
          "the A100 recovery runs used a different penalty scale, so this is not the matched hardware comparison above.", "",
          "The n=10 truth factor occupies 16 MiB and took 3.716 seconds to generate. No observation vector is stored; "
          "enumerating its 63,403,380,965,376 float64 observations would require 472,392 GiB (461.32 TiB) before overhead.", "",
          "**Where time goes**", "",
          "On V100 at n=10/batch32, isolated median calls took 6.228 ms for generated targets plus the rank-one measurement gradient "
          "and 8.114 ms for exact TP. With tensor measurements the first figure was 173.962 ms and TP was 8.003 ms. "
          "These include dispatch/synchronization and do not add up exactly to fused-scan time. "
          "Host noise preparation took about 1.95 ms per 100-step chunk; CPU prefetch was not justified by that measurement.", "",
          "**Recovery experiments**", "",
          "All are fresh runs with initialization/sampling/channel/noise seeds zero. Metrics use the same fixed 512-row sample "
          "within each qubit size; fidelity and TP use complete factors. The relative-noise controls use sigma=0.05*2**(5-n), "
          "preserving the n=5 ratio of noise standard deviation to mean outcome probability. "
          "Batch sizes, penalty scales, step counts, and devices differ as shown; these are exploratory configurations, "
          "not isolated causal tests or universal optima.", "",
          "| n | GPU | sigma | Batch | Steps | beta0 | Final fidelity % | TP | Sampled loss | Truth loss on same sample |",
          "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for row in recovery:
    gpu = "A100" if "A100" in row["device"] else "V100"
    lines.append(f"| {row['n']} | {gpu} | {row['sigma']:.7g} | {row['batch']} | {row['steps']} | {row['beta0']:.7g} | "
                 f"{100*row['final_fidelity']:.6g} | {row['tp']:.6g} | {row['loss']:.7g} | {row['fixed_metric_sample_truth_loss']:.7g} |")
lines += ["", "Truth loss is computed from the exact fixed metric-sample noise recipe, up to floating-point rounding. "
          "It is not a convergence certificate. Smoothed objectives and gaps are not compared across penalty schedules.", "",
          "An allocated CPU probe measured U0 gradient norms using a bounded streamed batch. At n=10/batch512, the unscaled TP "
          "gradient norm was 42,560; measurement norms were 0.000174 at sigma=.05 and 0.000142 at sigma=.0015625. "
          "Thus beta0=10,240,000 made TP approximately 24–29x larger at initialization. The subsequent beta0=1e9 tests "
          "reduce that initial ratio to roughly 0.25–0.30. This is a sampled-gradient observation at U0, not a statement "
          "about all iterations. Increasing beta0 weakens TP throughout the schedule.", "",
          "**Interpretation and next experiment**", "",
          "Generating observations on demand removes the 24**n storage bottleneck. Exact TP remains a substantial compute cost, "
          "but neither GPU memory nor target generation prevented n=10 execution. Increasing the GPU speed alone does not "
          "resolve the observed recovery failure. Fixed sigma=.05 also becomes much larger relative to the signal as n grows; "
          "the smaller-noise controls are distinct experiments and must not be presented as recovery at the original noise level.", "",
          "The original n=5 runs used 320,000 sampled rows for 1,024 complex factor entries. The n=10 batch512/10k runs "
          "use 5.12 million sampled rows for 1,048,576 entries: 64x fewer draws per entry. Repeated samples and optimizer "
          "weighting matter, so this ratio is descriptive, not a sample-complexity bound. Matching that ratio would take "
          "640,000 batch512 steps, roughly 10.1 A100 optimizer hours at the measured rate, without guaranteeing recovery. "
          "A useful next investigation is convergence versus sample budget, step schedule, and TP schedule, calibrated first "
          "on the improving n=8 configuration before a long n=10 run. Repeating short near-zero-fidelity n=10 runs across seeds "
          "would not yet validate a promising recovery setting.", "",
          "![Saved fidelity trajectories for the relative-noise controls](recovery.png)", "",
          "**Reproducibility and files**", "",
          "Raw reports are in the adjacent profile directories; complete scalar traces are in traces.json, and normalized "
          "tables are in summary.json. WORK_LOG.md records all job IDs, failed/canceled attempts, output locations, and "
          "environment handling. The reviewed uncommitted changes are preserved in source.patch. "
          "Ruche's original datasets/results, staged scripts/gen_data.py, and .gitignore edits were preserved.", ""]
(HERE / "report.md").write_text("\n".join(lines))

if recovery:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for n, ax in zip((8, 10), axes):
        for record in traces:
            cfg, meta, trace = record["configuration"], record["source_metadata"], record["trace"]
            if record["n"] != n or cfg["steps"] <= 1000 or meta["noise_std"] == .05:
                continue
            gpu = "A100" if "A100" in record["device"] else "V100"
            label = f"{gpu}, B={cfg['batch_size']}, beta0={cfg['smoothing_scale']:.3g}"
            ax.plot(trace["checkpoint_steps"], np.maximum(100*np.asarray(trace["process_fidelity_proxy"]),1e-10), label=label)
        ax.set(title=f"{n} qubits · relative-noise control", xlabel="Optimizer steps", ylabel="Fidelity proxy (%)", yscale="log")
        ax.set_ylim(1e-6, 100)
        ax.grid(alpha=.2, which="both")
        if ax.lines:
            ax.legend(fontsize=8, loc="upper left")
    fig.savefig(HERE / "recovery.png", dpi=180)
    plt.close(fig)
print(f"Wrote report with {len(profiles)} profiles and {len(recovery)} recovery runs")
