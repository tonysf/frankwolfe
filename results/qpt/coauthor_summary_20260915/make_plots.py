"""Plot existing scalar traces only. Run with Python, NumPy and Matplotlib.

No JAX import, simulation, optimization, remote access, or dataset generation.
Paths are relative to this file so the sharing package is self-contained.
"""
from pathlib import Path
import hashlib
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qpt-coauthor-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)


def read(name):
    return json.loads((ROOT / name).read_text())


def join_segments(*segments):
    rows = {}
    for segment in segments:
        for row in segment["trajectory"]:
            k = row["step"]
            if k in rows:
                for key in ("fidelity", "tp_violation", "measurement_loss"):
                    assert np.isclose(rows[k][key], row[key], rtol=1e-12, atol=1e-14), (k, key)
            rows[k] = row
    result = [rows[k] for k in sorted(rows)]
    assert all(np.isfinite(row[key]) for row in result for key in row)
    return result


small = read("n8/small_batch_continuation.json")
first = read("n8/first_100k_trace.json")
last = read("n8/final_segment_trace.json")
verification = read("n8/final_verification.json")
small_verification = read("n8/small_batch_verification.json")
provenance = read("n8/provenance.json")
small_provenance = read("n8/small_batch_provenance.json")
for segment in (small, first, last):
    m = segment["metadata"]
    assert m["n_qubits"] == 8 and m["noise_std"] == .05
    assert m["observation_count"] == 24**8
    assert m["data_metadata"]["truth_factor_sha256"] == first["metadata"]["data_metadata"]["truth_factor_sha256"]
for prefix, expected in (("rho", (2, 4, .6)), ("smoothing", (1e7, 1, .25)), ("step", (10, 10, 1))):
    for prov in (provenance, small_provenance):
        cmd = prov["command"]
        actual = tuple(float(cmd[cmd.index(f"--{prefix}-{field}") + 1]) for field in ("scale", "offset", "exponent"))
        assert actual == expected
assert provenance["data_sha256"] == small_provenance["data_sha256"]

large_rows = join_segments(first, last)
small_rows = join_segments(small)
assert large_rows[-1]["step"] == 148000 and small_rows[0]["step"] == 30000
assert large_rows[-1]["fidelity"] >= .99
assert max(r["fidelity"] for r in large_rows[:-1]) < .99
for rows, verified in ((large_rows, verification), (small_rows, small_verification)):
    assert np.isclose(rows[-1]["fidelity"], verified["fidelity_numpy"], rtol=0, atol=1e-13)

BLUE = "#176B9A"
ORANGE = "#BF651D"
INK = "#253343"
GREY = "#687483"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK,
    "ytick.color": INK, "axes.edgecolor": "#BBC4CC", "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "#E4E9EE", "grid.linewidth": .7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
})


def arrays(rows):
    return (np.array([r[k] for r in rows]) for k in ("step", "fidelity", "tp_violation"))


sx, sf, stp = arrays(small_rows)
lx, lf, ltp = arrays(large_rows)


def decorate(ax):
    ax.grid(True, axis="both")
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=190)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.4))
fig.subplots_adjust(left=.068, right=.974, top=.77, bottom=.24, wspace=.24)
fig.suptitle("Eight-qubit QPT: recovery at the original noise level", x=.068, y=.962, ha="left", fontsize=18, weight="bold")
fig.text(.068, .905, r"Gaussian noise $\sigma=0.05$ · same truth and initial factor · one dataset / seed", fontsize=11.5, color=GREY)
ax, zoom = axes
ax.plot(lx/1000, lf*100, color=BLUE, lw=2.6, label="Batch 65,536")
ax.plot(sx/1000, sf*100, color=ORANGE, lw=2.6, label="Batch 8,192 (saved trace from 30k)")
ax.set(xlim=(0, 207), ylim=(0, 103), xlabel="Completed iterations (thousands)", ylabel="Factor fidelity (%)", title="A  Full recovery trajectories")
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.yaxis.set_major_locator(MultipleLocator(20))
ax.axhline(99, color=GREY, ls=(0,(4,3)), lw=1)
ax.scatter([148,200], [lf[-1]*100,sf[-1]*100], c=[BLUE,ORANGE], s=38, zorder=5)
ax.scatter([30], [sf[0]*100], facecolor="white", edgecolor=ORANGE, s=42, zorder=5)
ax.annotate("90.8853% at 200k", xy=(200,sf[-1]*100), xytext=(115,73), color=ORANGE,
            arrowprops={"arrowstyle":"-", "color":ORANGE}, fontsize=10.5)
ax.text(10, 7, "Small-batch iterations 0–30k\nare not reconstructed.", color=GREY, fontsize=9)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.061,.874), ncol=2, frameon=False, fontsize=10.5)
mask = lx >= 20000
zoom.plot(lx[mask]/1000, lf[mask]*100, color=BLUE, lw=2.6)
zoom.axhline(99, color=GREY, ls=(0,(4,3)), lw=1)
zoom.text(22,99.07,"99% target", color=GREY, fontsize=9)
zoom.set(xlim=(20,157), ylim=(94.7,99.45), xlabel="Completed iterations (thousands)", ylabel="Factor fidelity (%)", title="B  Batch 65,536: approach to 99%")
zoom.xaxis.set_major_locator(MultipleLocator(25))
zoom.yaxis.set_major_locator(MultipleLocator(1))
zoom.scatter([100,148], [lf[lx==100000][0]*100,lf[-1]*100], color=BLUE, s=38, zorder=5)
zoom.annotate("98.48565%\n100k checkpoint / resume", xy=(100,lf[lx==100000][0]*100), xytext=(70,96.8),
              arrowprops={"arrowstyle":"-", "color":BLUE}, fontsize=9.5, color=BLUE)
zoom.annotate("99.00265% at 148k\nIndependently verified", xy=(148,lf[-1]*100), xytext=(103,95.45),
              arrowprops={"arrowstyle":"-", "color":BLUE}, fontsize=10.5, color=BLUE, weight="bold")
for axis in axes: decorate(axis)
fig.text(.068,.102,r"Both runs: $\gamma_k=10/(k+10)$; $\beta_k=10^7/(k+1)^{0.25}$; $\rho_k=2/(k+4)^{0.6}$; rank 1; $\tau=20$.", fontsize=11)
fig.text(.068,.047,"Lines join saved checkpoints; no fitted or smoothed curves. Fidelity uses complete factors. Trace preservation remains approximate.", fontsize=9.5, color=GREY)
save(fig, "n8_fidelity_iterations")

fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.4))
fig.subplots_adjust(left=.076, right=.974, top=.77, bottom=.24, wspace=.27)
fig.suptitle("Eight-qubit QPT: sampling budget and TP residual", x=.076, y=.962, ha="left", fontsize=18, weight="bold")
fig.text(.076,.905,"Same two runs, Gaussian noise σ = 0.05. Row counts include repeated samples.", fontsize=11.5,color=GREY)
ax, tp = axes
for x,f,t,b,c,label in ((lx,lf,ltp,65536,BLUE,"Batch 65,536"),(sx,sf,stp,8192,ORANGE,"Batch 8,192")):
    ax.plot(x*b/1e9,f*100,color=c,lw=2.6,label=label)
    tp.plot(x/1000,t,color=c,lw=2.6,label=label)
ax.axhline(99,color=GREY,ls=(0,(4,3)),lw=1)
budget=200000*8192/1e9
matched=next(r for r in large_rows if r["step"]==25000)
ax.axvline(budget,color=GREY,ls=":",lw=1)
ax.scatter([budget,budget], [100*matched["fidelity"],100*sf[-1]], c=[BLUE,ORANGE],s=45,zorder=5)
ax.annotate("At 1.6384 billion draws:\n94.1584% (batch 65,536, 25k steps)\n90.8853% (batch 8,192, 200k steps)",
            xy=(budget,100*sf[-1]), xytext=(2.7,42),fontsize=9.5,
            arrowprops={"arrowstyle":"-","color":GREY})
ax.set(xlim=(0,10),ylim=(0,103),xlabel="Cumulative sampled row draws (billions)",ylabel="Factor fidelity (%)",title="A  Comparison by sampled row draws")
tp.set(xlim=(0,207),yscale="log",ylim=(.15,300),xlabel="Completed iterations (thousands)",ylabel=r"TP residual $\|T(uu^\dagger)-I\|_F$",title="B  Exact factor-based TP residual")
tp.xaxis.set_major_locator(MultipleLocator(50))
tp.scatter([148,200],[ltp[-1],stp[-1]],c=[BLUE,ORANGE],s=38,zorder=5)
tp.annotate("0.1990 at 148k",xy=(148,ltp[-1]),xytext=(80,1.8),color=BLUE,fontsize=10,
            arrowprops={"arrowstyle":"-","color":BLUE})
tp.annotate("0.4835 at 200k",xy=(200,stp[-1]),xytext=(118,8),color=ORANGE,fontsize=10,
            arrowprops={"arrowstyle":"-","color":ORANGE})
for axis in axes: decorate(axis)
handles,labels=ax.get_legend_handles_labels()
fig.legend(handles,labels,loc="upper left",bbox_to_anchor=(.07,.874),ncol=2,frameon=False)
fig.text(.076,.101,"Full virtual table: 24⁸ = 110,075,314,176 rows. Successful run: 9.6993 billion draws = 8.8115% of table size.",fontsize=10.5)
fig.text(.076,.047,"Draw counts are not unique-row coverage. Schedules depend on iterations, so their values differ at a matched draw budget.",fontsize=9.5,color=GREY)
save(fig,"n8_sampling_and_tp")

inputs=["n8/small_batch_continuation.json","n8/first_100k_trace.json","n8/final_segment_trace.json",
        "n8/final_verification.json","n8/small_batch_verification.json","n8/provenance.json","n8/small_batch_provenance.json"]
data={"scope":"Existing n8 sigma=.05 scalar traces only; no new computation of the optimization problem.",
      "schedule_indexing":"k=0 is the first update; a checkpoint t is after t updates. Last applied k=t-1.",
      "schedules":{"step":{"scale":10,"offset":10,"exponent":1,"cap":1},
                   "smoothing":{"scale":1e7,"offset":1,"exponent":.25,"cap":None},
                   "rho":{"scale":2,"offset":4,"exponent":.6,"cap":1}},
      "virtual_observations":24**8,"stored_observations":0,"sampling":"uniform_with_replacement_fixed_row_noise",
      "runs":[],"source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in inputs}}
for batch,rows in ((8192,small_rows),(65536,large_rows)):
    steps=rows[-1]["step"]
    data["runs"].append({"batch_size":batch,"first_plotted_step":rows[0]["step"],"total_steps":steps,
                         "batch_fraction_of_table":batch/24**8,"total_row_draws":batch*steps,
                         "draws_divided_by_table_size":batch*steps/24**8,"trajectory":rows})
(ROOT/"plot_data.json").write_text(json.dumps(data,indent=2)+"\n")
print(json.dumps({"figures":[str(p.relative_to(ROOT)) for p in sorted(OUT.glob('*'))],
                  "checkpoints":{"batch8192":len(small_rows),"batch65536":len(large_rows)},
                  "successful_final_fidelity_percent":100*lf[-1]},indent=2))
