"""Present the completed probe-count study on separate iteration and time axes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator


ROOT = Path(__file__).resolve().parent
DATASETS = [('bmmc6224', 'BMMC · 6,224 cells'),
            ('metric512_s1000', 'Clean synthetic · 512 points')]
STYLES = {128: ('#0072B2', 'o'), 1024: ('#D78300', 's'), 2048: ('#009E73', '^')}


def main():
    report_path = ROOT/'report.json'
    report = json.loads(report_path.read_text())
    assert report['status'] == 'completed' and report['task_counts']['completed'] == 18
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'medium', 'pdf.fonttype': 42})
    out = ROOT/'figures'
    out.mkdir(exist_ok=True)
    artifacts = []
    for axis in ('iterations', 'wall_clock'):
        is_time = axis == 'wall_clock'
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
        fig.subplots_adjust(left=.095, right=.975, top=.77, bottom=.24, wspace=.28)
        fig.suptitle('FraMES: loss versus '+('wall-clock time' if is_time else 'iterations'),
                     fontsize=17, y=.97)
        for ax, (dataset, title) in zip(axes, DATASETS):
            ax.set_title(title, fontsize=13, pad=12)
            for q, (color, marker) in STYLES.items():
                rows = sorted((row for row in report['trajectory_summaries']
                               if row['dataset'] == dataset and row['batch'] == q),
                              key=lambda row: row['iteration'])
                assert all(row['status'] == 'completed' and row['n'] == 3 for row in rows)
                losses = [row['statistics']['rounded_loss'] for row in rows]
                x = ([row['statistics']['time_to_this_feasible_output']['median']/60 for row in rows]
                     if is_time else [row['iteration'] for row in rows])
                y = [value['median'] for value in losses]
                ax.fill_between(x, [value['q25'] for value in losses],
                                [value['q75'] for value in losses], color=color, alpha=.16, linewidth=0)
                ax.plot(x, y, color=color, marker=marker, markersize=5, linewidth=1.7,
                        label=f'{q:,} probes')
                if is_time:
                    times = [row['statistics']['time_to_this_feasible_output'] for row in rows]
                    xerr = [[(value['median']-value['q25'])/60 for value in times],
                            [(value['q75']-value['median'])/60 for value in times]]
                    ax.errorbar(x, y, xerr=xerr, fmt='none', ecolor=color, elinewidth=.8,
                                capsize=2, alpha=.65)
            ax.set_xlabel('Solver wall-clock time + output rounding (minutes)' if is_time else 'Completed updates',
                          fontsize=11, labelpad=9)
            ax.set_ylabel('Feasible squared GW loss', fontsize=11, labelpad=9)
            ax.grid(True, alpha=.20, linewidth=.7)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=10)
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
            if not is_time:
                ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
            if dataset == 'metric512_s1000':
                ax.set_ylim(-.0015, .0465)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .90),
                   ncol=3, frameon=False, fontsize=12, handlelength=2.3, columnspacing=3)
        fig.text(.095, .135, 'Median and shaded IQR across 3 seeds. Markers are saved checkpoints; connecting lines are visual guides.',
                 fontsize=9.2)
        if is_time:
            fig.text(.095, .097, 'Each marker pairs median loss and median recorded time at the same update count; horizontal bars show timing IQR.',
                     fontsize=9.2)
            fig.text(.095, .059, 'Time excludes setup, diagnostic evaluations, earlier output repairs, and final FP64 validation.', fontsize=9.2)
        else:
            fig.text(.095, .097, 'Final losses are independently evaluated in FP64; earlier losses use optimizer precision. Panels use different loss scales.',
                     fontsize=9.2)
        for extension in ('png', 'pdf'):
            path = out/f'loss_vs_{axis}.{extension}'
            fig.savefig(path, dpi=190, facecolor='white')
            artifacts.append(dict(file=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        plt.close(fig)
    receipt = dict(source_report_sha256=hashlib.sha256(report_path.read_bytes()).hexdigest(),
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   artifacts=artifacts,
                   unchanged_results=True,
                   time_axis='Median recorded solver wall time plus repair of the current output, in minutes. Not total end-to-end elapsed time.',
                   lines='Straight visual guides between saved observations; no intermediate values or transition times inferred.')
    (ROOT/'separate_plots.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
