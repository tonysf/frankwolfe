"""Compact CT trajectory figures from unchanged group-balanced statistics.

This module changes presentation only. Existing statistics can be rendered
directly, while ``statistics`` delegates aggregation and native-state alignment
to the established renderer for new completed campaigns.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np

try:
    from . import nonnegative_plots as reference
except ImportError:
    import nonnegative_plots as reference


AXES = reference.AXES
METRICS = (('raw_loss', 'Relative\nraw loss'), ('d_distance', 'Orthant\ndistance'))
CURRENT_VARIANTS = ('sframes', 'tuned_sframes', 'prox_sarah', 'spring_sarah', 'joint_lbfgsb')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def statistics(runs, slices, groups, n, variants):
    """Return native, group-balanced means/SDs for an ordered label/color map."""
    stats = dict(training_views=n,
        renderer_sha256=_digest(__file__),
        aggregation_source_sha256=_digest(reference.__file__),
        center='Arithmetic mean with equal group, within-group slice, and within-slice repetition weights',
        standard_deviation='Descriptive weighted population SD: sqrt(sum(w * (value - mean)^2)); not standard error',
        alignment='At each work value use the last recorded iterate at or below that value; no linear interpolation',
        support='Fixed cohort, truncated at minimum final native coordinate; duplicate coordinates retain latest state',
        band='Unmodified arithmetic mean minus/plus one standard deviation in the original metric units',
        individual_trajectories_drawn=False, methods={}, row_scales={})
    for variant, (label, color) in variants.items():
        chosen, weights = reference.select_and_weight(runs, slices, groups, n, variant)
        entry = dict(label=label, color=color, physical_runs=len(chosen),
            runs=[dict(slice_id=key[0], seed=key[1], group=groups[key[0]], weight=float(weight))
                  for (key, _), weight in zip(chosen, weights)], axes={})
        for axis, _ in AXES:
            metrics, grid = {}, None
            for metric, _ in METRICS:
                coordinates, mean, sd = reference.summarize(runs, slices, groups, n, variant, metric, axis)
                if grid is not None and not np.array_equal(grid, coordinates):
                    raise ValueError('Metrics have different native support')
                grid = coordinates
                metrics[metric] = dict(mean=mean.tolist(), standard_deviation=sd.tolist(),
                    lower_min=float(np.min(mean-sd)), upper_max=float(np.max(mean+sd)))
            entry['axes'][axis] = dict(grid=grid.tolist(), metrics=metrics)
        stats['methods'][variant] = entry
    return stats


def _arrays(entry, axis, metric):
    series = entry['axes'][axis]
    x = np.asarray(series['grid'], dtype=float)
    mean = np.asarray(series['metrics'][metric]['mean'], dtype=float)
    sd = np.asarray(series['metrics'][metric]['standard_deviation'], dtype=float)
    if (x.ndim != 1 or not len(x) or x.shape != mean.shape or x.shape != sd.shape
            or not np.isfinite(x).all() or not np.isfinite(mean).all() or not np.isfinite(sd).all()
            or x[0] != 0 or np.any(np.diff(x) <= 0) or np.any(sd < 0)):
        raise ValueError('Invalid native trajectory summary')
    return x, mean, sd


def _work_ticks(limit):
    ticks = [0]
    if limit < 10:
        return sorted(set(ticks + [1, limit]))
    power = 10
    while power <= limit:
        ticks.append(power)
        power *= 10
    return ticks


def _count(value, _):
    if abs(value) >= 1000:
        return f'{value/1000:.3g}k'
    return f'{value:g}'


def render_statistics(stats, output_stem, variants=None):
    """Render two rows and three work axes without altering any mean or band.

    ``stats`` is a statistics JSON path or the same in-memory mapping. The
    returned JSON-serializable receipt pins the source statistics and exact
    selected arrays; it is also written beside the PDF and PNG.
    """
    source_path = str(Path(stats).resolve()) if isinstance(stats, (str, Path)) else None
    if source_path is not None:
        source_sha = _digest(source_path)
        stats = json.loads(Path(source_path).read_text())
    else:
        source_sha = hashlib.sha256(_canonical(stats)).hexdigest()
    original = deepcopy(stats)
    variants = list(stats['methods']) if variants is None else list(variants)
    if not variants or len(variants) != len(set(variants)):
        raise ValueError('Select distinct methods')
    entries = {name: stats['methods'][name] for name in variants}
    arrays = {(metric, axis, name): _arrays(entry, axis, metric)
              for metric, _ in METRICS for axis, _ in AXES for name, entry in entries.items()}
    selected = {name: dict(label=entry['label'], color=entry['color'], axes={
        axis: dict(grid=entry['axes'][axis]['grid'], metrics={
            metric: entry['axes'][axis]['metrics'][metric] for metric, _ in METRICS})
        for axis, _ in AXES}) for name, entry in entries.items()}
    receipt = dict(schema_version=1, renderer_sha256=_digest(__file__),
        source_statistics=source_path, source_statistics_sha256=source_sha,
        plotted_statistics_sha256=hashlib.sha256(_canonical(selected)).hexdigest(),
        training_views=stats['training_views'], variants=variants,
        metrics=[metric for metric, _ in METRICS], axes=[axis for axis, _ in AXES],
        mean_and_sd_unchanged=True, band='Unmodified mean minus/plus one descriptive SD',
        row_scales={}, individual_trajectories_drawn=False)
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    style = {'font.size': 7.5, 'axes.labelsize': 8, 'xtick.labelsize': 7,
             'ytick.labelsize': 7, 'legend.fontsize': 7, 'pdf.fonttype': 42,
             'ps.fonttype': 42, 'axes.linewidth': .6}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 3, figsize=(6.8, 2.5), sharey='row')
        for row, (metric, ylabel) in enumerate(METRICS):
            values = [v for (m, _, _), v in arrays.items() if m == metric]
            low = min(float(np.min(mean-sd)) for _, mean, sd in values)
            high = max(float(np.max(mean+sd)) for _, mean, sd in values)
            scale = 'log' if metric != 'd_distance' and low > 0 else 'symlog'
            threshold = .001 if metric == 'd_distance' else .01
            receipt['row_scales'][metric] = dict(scale=scale, lower_band_min=low,
                upper_band_max=high, negative_band_preserved=low < 0,
                linthresh=threshold if scale == 'symlog' else None)
            for col, (axis, heading) in enumerate(AXES):
                ax = axes[row, col]
                for name, entry in entries.items():
                    x, mean, sd = arrays[metric, axis, name]
                    ax.fill_between(x, mean-sd, mean+sd, color=entry['color'], alpha=.13,
                        linewidth=0, step=None if axis == 'iteration' else 'post', zorder=1)
                    ax.plot(x, mean, color=entry['color'], linestyle='-', linewidth=1,
                        label=entry['label'], drawstyle='default' if axis == 'iteration' else 'steps-post', zorder=2)
                limit = max(arrays[metric, axis, name][0][-1] for name in entries)
                if axis == 'comparison_views':
                    ax.set_xticks(np.linspace(0, max(limit, 1), 3))
                else:
                    ax.set_xscale('symlog', linthresh=1, linscale=.5)
                    ax.set_xticks(_work_ticks(max(limit, 1)))
                ax.set_xlim(0, max(limit, 1))
                ax.xaxis.set_major_formatter(FuncFormatter(_count))
                ax.xaxis.set_minor_formatter(NullFormatter())
                if row == 0:
                    ax.tick_params(axis='x', labelbottom=False)
                else:
                    ax.set_xlabel(heading, labelpad=2)
                if col == 0:
                    ax.set_ylabel(ylabel, labelpad=3)
                if scale == 'log':
                    ax.set_yscale('log')
                    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 3), numticks=4))
                    ax.set_ylim(low*.94, high*1.06)
                else:
                    ax.set_yscale('symlog', linthresh=threshold, linscale=.7)
                    lower = -threshold*.1 if low >= 0 else low-max(abs(low)*.05, threshold*.1)
                    upper = high+max(abs(high)*.05, threshold*.1)
                    ax.set_ylim(lower, upper)
                    ax.set_yticks(reference._signed_ticks(lower, upper, threshold))
                if metric == 'raw_loss':
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{100*value:g}%'))
                else:
                    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:g}'))
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.tick_params(length=2.5, width=.6, pad=2)
                ax.grid(alpha=.17, linewidth=.4)
                ax.set_axisbelow(True)
                ax.spines[['top', 'right']].set_visible(False)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.52, .995),
            ncol=min(len(variants), 5), frameon=False, handlelength=1.6,
            handletextpad=.4, columnspacing=.9)
        fig.subplots_adjust(left=.11, right=.96, bottom=.17, top=.84, hspace=.20, wspace=.17)
        fig.savefig(stem.with_suffix('.pdf'))
        fig.savefig(stem.with_suffix('.png'), dpi=220)
        plt.close(fig)
    if stats != original:
        raise AssertionError('Rendering modified source statistics')
    receipt['outputs'] = {suffix[1:]: dict(path=str(stem.with_suffix(suffix)),
        sha256=_digest(stem.with_suffix(suffix))) for suffix in ('.pdf', '.png')}
    stem.with_suffix('.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    return receipt
