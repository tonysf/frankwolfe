"""Plot observed median feasible GW loss for the completed stochastic campaign.

Run from anywhere with the experiment's .venv/bin/python. No remote reads or
numerical reruns: this uses the checksum-verified, collected report only.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
METHODS = {
    'smoothing-fw': ('FraMES (momentum)', '#1766ad', 'o'),
    'published-smd': ('Entropic SMD (no momentum)', '#c65e12', 's'),
    'published-msbpg': ('MSBPG (theorem parameters)', '#278447', '^'),
}
DATASETS = {'bmmc6224': 'BMMC · 6,224 cells',
            'snare_full': 'SNARE · 10,000 cells',
            'shareseq_full': 'SHARE · 31,890 cells'}
STARTS = ('product', 'randomized')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    receipt = json.loads((ROOT / 'reports/MSBPG_CAMPAIGN.json').read_text())
    source = ROOT / receipt['collected_report']['directory'] / 'report.json'
    assert digest(source) == receipt['collected_report']['verified_sha256']['report.json']
    report = json.loads(source.read_text())
    config_path = ROOT / 'extensions/msbpg_20261005/campaign.json'
    assert digest(config_path) == report['config_sha256']
    config = json.loads(config_path.read_text())
    assert report['status'] == 'completed'
    checkpoints = [0] + config['final_checkpoint_iterations']
    expected_seeds = config['final_seeds']
    finals = [r for r in report['runs'] if r['phase'] == 'final']
    assert len(finals) == 54 and all(r['status'] == 'completed' for r in finals)
    observations, summaries = [], []
    for dataset in DATASETS:
        for start in STARTS:
            for method in METHODS:
                runs = sorted((r for r in finals if (r['dataset'], r['initialization'], r['method'])
                               == (dataset, start, method)), key=lambda r: r['seed'])
                assert [r['seed'] for r in runs] == expected_seeds
                for iteration in checkpoints:
                    values = []
                    kind = ('initial' if iteration == 0 else 'last' if iteration == config['final_iterations']
                            else 'current_diagnostic')
                    for run in runs:
                        points = [p for p in run['trajectory'] if p['iteration'] == iteration and p['kind'] == kind]
                        assert len(points) == 1
                        loss = points[0]['rounded_loss']
                        assert np.isfinite(loss)
                        if kind == 'last':
                            assert loss == run['last_rounded_loss']
                        values.append(loss)
                        observations.append(dict(dataset=dataset, initialization=start, method=method,
                                                 candidate_id=run['candidate_id'], seed=run['seed'],
                                                 iteration=iteration, loss=loss, kind=kind,
                                                 precision='FP64' if kind == 'last' else 'FP32'))
                    q25, median, q75 = np.quantile(values, [.25, .5, .75], method='linear')
                    if kind == 'last':
                        endpoint = next(s for s in report['final_summaries'] if
                                        (s['dataset'], s['initialization'], s['method']) == (dataset, start, method))
                        assert median == endpoint['statistics']['last_rounded_loss']['median']
                    summaries.append(dict(dataset=dataset, initialization=start, method=method,
                                          iteration=iteration, n=len(values), median=float(median),
                                          q25=float(q25), q75=float(q75)))
    assert len(observations) == 270 and len(summaries) == 90
    write_csv('observations.csv', observations)
    write_csv('median_loss_iterations.csv', summaries)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.labelsize': 10, 'axes.titlesize': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(3, 2, figsize=(12.4, 10.2), sharex=True, sharey='row')
    fig.subplots_adjust(left=.085, right=.98, bottom=.14, top=.85, hspace=.37, wspace=.13)
    fig.suptitle('Median feasible GW loss over iterations', fontsize=18, fontweight='bold', y=.975)
    fig.text(.5, .94, 'Three final seeds per method and initialization · Lower loss is better',
             ha='center', color='#424b57', fontsize=11)
    handles = [Line2D([], [], color=color, marker=marker, linestyle='--', linewidth=1.8,
                      markersize=6, label=label) for label, color, marker in METHODS.values()]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .92), ncol=3,
               frameon=False, handlelength=2.6, columnspacing=2)
    for row, (dataset, name) in enumerate(DATASETS.items()):
        limits = []
        for col, start in enumerate(STARTS):
            ax = axes[row, col]
            for method, (_, color, marker) in METHODS.items():
                points = [s for s in summaries if (s['dataset'], s['initialization'], s['method'])
                          == (dataset, start, method)]
                x = np.array([p['iteration'] for p in points])
                y = np.array([p['median'] for p in points])
                low, high = np.array([p['q25'] for p in points]), np.array([p['q75'] for p in points])
                limits.extend(low.tolist() + high.tolist())
                ax.errorbar(x, y, yerr=np.stack([y-low, high-y]), color=color, marker=marker,
                            markersize=5.5, linestyle='--', linewidth=1.7, elinewidth=1.2,
                            capsize=3, capthick=1.1, zorder=3)
            ax.set_title(f'{name} | {start.capitalize()} start', loc='left', pad=9)
            ax.grid(axis='both', alpha=.18, zorder=0)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
            ax.set_xlim(-45, 2060)
            ax.set_xticks([0, 500, 1000, 1500, 2000])
            if col == 0:
                ax.set_ylabel('Feasible GW loss')
            if row == 2:
                ax.set_xlabel('Completed outer updates')
        pad = max(limits) - min(limits)
        axes[row, 0].set_ylim(min(limits)-.11*pad, max(limits)+.15*pad)
    fig.text(.085, .08, 'Markers: medians at saved updates 0, 100, 300, 1,000 and 2,000. Bars: interquartile range (n = 3).',
             fontsize=9, color='#424b57')
    fig.text(.085, .059, 'Dashed lines connect observed medians as visual guides; no unsaved losses are estimated. Y-scales differ by dataset.',
             fontsize=9, color='#424b57')
    fig.text(.085, .038, 'Current rounded outputs; intermediate loss diagnostics use FP32, final losses use FP64. GW loss excludes the KL term.',
             fontsize=9, color='#424b57')
    for suffix in ('png', 'pdf'):
        fig.savefig(OUT / f'median_loss_iterations.{suffix}', dpi=180, facecolor='white')
    plt.close(fig)
    provenance = dict(source_report=str(source.relative_to(ROOT)), source_sha256=digest(source),
                      config_sha256=digest(config_path), final_runs=54, seeds=expected_seeds,
                      checkpoints=checkpoints, observation_rows=len(observations), summary_rows=len(summaries),
                      aggregation='Median and linear IQR across all three final seeds, separately by dataset, start, method.',
                      duplicate_endpoint_policy='Only the FP64 last record is used at 2000; current_diagnostic is excluded.',
                      trajectory_policy='Only common prescribed observed checkpoints; no smoothing, interpolation of seed traces, or best-so-far loss.',
                      precision='FP32 intermediate diagnostics, FP64 final endpoints.',
                      methods={m: label for m, (label, _, _) in METHODS.items()},
                      plotter_sha256=digest(Path(__file__)),
                      files={name: digest(OUT / name) for name in
                             ('observations.csv', 'median_loss_iterations.csv', 'median_loss_iterations.png', 'median_loss_iterations.pdf')})
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(dict(final_runs=54, observations=270, summaries=90, output=str(OUT))))


if __name__ == '__main__':
    main()
