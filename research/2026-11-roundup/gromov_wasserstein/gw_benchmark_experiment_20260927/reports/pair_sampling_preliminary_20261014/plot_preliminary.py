"""Plot audited, completed pair-oracle runs; never load or rerun an optimizer.

The input is report_campaign.py's report.json. Time coordinates are exactly
its solver + current repair costs, NOT end-to-end times including diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter
import numpy as np

DATASETS = {
    'metric512_s1000': dict(label='Synthetic metric data', n=512, horizon=200000),
    'bmmc6224': dict(label='BMMC', n=6224, horizon=20000),
    'snare_full': dict(label='SNARE-seq', n=10000, horizon=20000),
}
METHODS = {
    'smoothing-fw': dict(label='FraMES', color='#0072B2'),
    'published-smd': dict(label='Entropic SMD (fresh gradient)', color='#D55E00'),
    'published-msbpg': dict(label='MSBPG (theorem parameters)', color='#009E73'),
}
SEEDS = [710, 711, 712]
TIME_LABEL = 'Solver + current repair time (min)'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Invalid finite number: {label}')
    return float(value)


def stats(values):
    values = np.asarray(values, dtype=float)
    return dict(median=float(np.median(values)), minimum=float(values.min()), maximum=float(values.max()))


def validate_runs(report):
    selected, excluded, units = [], [], set()
    for original in report['runs']:
        if original['dataset'] not in DATASETS:
            excluded.append({key: original.get(key) for key in ('dataset', 'method', 'seed', 'status')})
            continue
        dataset, method, seed = (original[key] for key in ('dataset', 'method', 'seed'))
        if method not in METHODS or seed not in SEEDS or original['status'] != 'completed':
            raise ValueError(f'Included dataset has an unexpected or incomplete cell: {(dataset,method,seed)}')
        unit = (dataset, method, seed)
        if unit in units:
            raise ValueError(f'Duplicate scientific cell: {unit}')
        units.add(unit)
        if original['completed_updates'] != DATASETS[dataset]['horizon']:
            raise ValueError(f'Wrong completed horizon: {unit}')
        if original.get('initialization') != 'product' or original.get('oracle') != 'pair-sampling':
            raise ValueError(f'Unexpected initialization or oracle: {unit}')
        if original.get('pairs_per_call') != 1024:
            raise ValueError(f'Unexpected pair budget: {unit}')
        metadata_hash = original.get('metadata_sha256', '')
        if len(metadata_hash) != 64 or any(c not in '0123456789abcdef' for c in metadata_hash):
            raise ValueError(f'Missing audited metadata hash: {unit}')
        indexed, duplicates = {}, []
        for row in original['observed_current_trajectory']:
            iteration = row['iteration']
            if isinstance(iteration, bool) or not isinstance(iteration, int) or not 0 <= iteration <= original['completed_updates']:
                raise ValueError(f'Invalid logged iteration: {unit}')
            loss = finite(row['rounded_loss'], 'online rounded loss')
            seconds = finite(row['time_to_this_feasible_output'], 'solver + current repair time')
            if seconds < 0:
                raise ValueError(f'Negative output time: {unit}')
            if iteration in indexed:
                previous = indexed[iteration]
                if previous['rounded_loss'] != loss:
                    raise ValueError(f'Duplicate iteration has conflicting logged losses: {unit}, iteration {iteration}')
                duplicates.append(dict(iteration=iteration, previous=previous, retained=dict(row)))
            # The trace ends with the explicit terminal measurement; retain it
            # when the same P_T also has a scheduled checkpoint measurement.
            indexed[iteration] = dict(iteration=iteration, rounded_loss=loss,
                                      time_to_this_feasible_output=seconds)
        if 0 not in indexed or original['completed_updates'] not in indexed:
            raise ValueError(f'Missing initial/final observed checkpoint: {unit}')
        observations = [indexed[k] for k in sorted(indexed)]
        times = [r['time_to_this_feasible_output'] for r in observations]
        if any(t2 <= t1 for t1, t2 in zip(times, times[1:])):
            raise ValueError(f'Nonmonotone logged solver + repair costs; cannot interpolate: {unit}')
        initial = observations[0]['rounded_loss']
        if initial <= 0:
            raise ValueError(f'Nonpositive baseline prevents relative reduction: {unit}')
        final64 = finite(original['last_rounded_loss'], 'independently reevaluated final float64 loss')
        solver = finite(original['solver_seconds'], 'total solver seconds')
        final_time = finite(original['last_feasible_output_seconds'], 'terminal output seconds')
        elapsed = finite(original['total_elapsed_seconds'], 'total elapsed seconds')
        if min(solver, final_time, elapsed) < 0 or final_time != times[-1]:
            raise ValueError(f'Inconsistent terminal timing: {unit}')
        selected.append(dict(original, observations=observations, initial_loss=initial,
            final_float64_loss=final64, terminal_online_loss=observations[-1]['rounded_loss'],
            float64_minus_online=final64-observations[-1]['rounded_loss'],
            relative_reduction_final_float64_percent=100*(initial-final64)/initial,
            duplicate_iteration_measurements=duplicates))
    expected = {(d,m,s) for d in DATASETS for m in METHODS for s in SEEDS}
    if units != expected:
        raise ValueError(f'Incomplete plotting population: {len(units)}/{len(expected)} cells; missing {sorted(expected-units)}')
    return selected, excluded


def aggregate(runs):
    curves, groups = [], []
    for dataset in DATASETS:
        dataset_runs = [r for r in runs if r['dataset'] == dataset]
        # Every time coordinate is an actual logged time from this dataset,
        # plus explicit support boundaries. No artificial fine-resolution grid.
        dataset_time_knots = sorted({p['time_to_this_feasible_output'] for r in dataset_runs for p in r['observations']})
        for method in METHODS:
            group = sorted((r for r in dataset_runs if r['method'] == method), key=lambda r:r['seed'])
            per_seed = [{p['iteration']:p for p in r['observations']} for r in group]
            common_iterations = sorted(set.intersection(*(set(p) for p in per_seed)))
            for k in common_iterations:
                losses = [p[k]['rounded_loss'] for p in per_seed]
                reduction = [100*(r['initial_loss']-f)/r['initial_loss'] for r,f in zip(group,losses)]
                loss_stat, reduction_stat = stats(losses), stats(reduction)
                curves.append(dict(dataset=dataset,method=method,axis='iteration',x=k,x_unit='completed updates',
                    seed_count=3,source='observed at common logged iteration',
                    **{'loss_'+key:value for key,value in loss_stat.items()},
                    **{'reduction_percent_'+key:value for key,value in reduction_stat.items()}))
            lower = max(r['observations'][0]['time_to_this_feasible_output'] for r in group)
            upper = min(r['observations'][-1]['time_to_this_feasible_output'] for r in group)
            if upper <= lower:
                raise ValueError(f'No common time support: {dataset}/{method}')
            time_grid = sorted({lower,upper,*[t for t in dataset_time_knots if lower <= t <= upper]})
            interpolated = np.asarray([np.interp(time_grid,
                [p['time_to_this_feasible_output'] for p in r['observations']],
                [p['rounded_loss'] for p in r['observations']]) for r in group])
            for j,t in enumerate(time_grid):
                losses = interpolated[:,j]
                reduction = [100*(r['initial_loss']-f)/r['initial_loss'] for r,f in zip(group,losses)]
                loss_stat,reduction_stat = stats(losses),stats(reduction)
                curves.append(dict(dataset=dataset,method=method,axis='time',x=t,x_unit='solver + current repair seconds',
                    seed_count=3,source='linear interpolation between logged points; no extrapolation',
                    **{'loss_'+key:value for key,value in loss_stat.items()},
                    **{'reduction_percent_'+key:value for key,value in reduction_stat.items()}))
            groups.append(dict(dataset=dataset,method=method,method_label=METHODS[method]['label'],
                expected_seeds=SEEDS,observed_seeds=[r['seed'] for r in group],coverage_complete=True,
                completed_updates=DATASETS[dataset]['horizon'],common_logged_iterations=common_iterations,
                interpolation_support_seconds=[lower,upper],interpolated_grid_point_count=len(time_grid),
                initial_online_loss=stats([r['initial_loss'] for r in group]),
                terminal_online_loss=stats([r['terminal_online_loss'] for r in group]),
                terminal_float64_loss=stats([r['final_float64_loss'] for r in group]),
                float64_minus_online=stats([r['float64_minus_online'] for r in group]),
                maximum_absolute_float64_minus_online=max(abs(r['float64_minus_online']) for r in group),
                relative_reduction_final_float64_percent=stats([r['relative_reduction_final_float64_percent'] for r in group]),
                solver_seconds=stats([r['solver_seconds'] for r in group]),
                last_feasible_output_seconds=stats([r['last_feasible_output_seconds'] for r in group]),
                total_elapsed_seconds=stats([r['total_elapsed_seconds'] for r in group])))
    return curves,groups


def plot(curves,runs,out,normalized=False,combined_pdf=None):
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':10,
                         'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes = plt.subplots(3,2,figsize=(13.6,12.1),sharey='row')
    prefix = 'reduction_percent_' if normalized else 'loss_'
    for row,(dataset,info) in enumerate(DATASETS.items()):
        for column,axis in enumerate(('iteration','time')):
            ax=axes[row,column]
            for method,style in METHODS.items():
                selected=[c for c in curves if c['dataset']==dataset and c['method']==method and c['axis']==axis]
                x=np.array([c['x'] for c in selected],dtype=float)/(60 if axis=='time' else 1)
                ax.fill_between(x,[c[prefix+'minimum'] for c in selected],[c[prefix+'maximum'] for c in selected],
                                color=style['color'],alpha=.13,linewidth=0)
                ax.plot(x,[c[prefix+'median'] for c in selected],color=style['color'],lw=2.1,
                        marker='o' if axis=='iteration' else None,markersize=3,markeredgewidth=0)
                members=[r for r in runs if r['dataset']==dataset and r['method']==method]
                # Unconnected final FP64 markers preserve the distinction from
                # the online objective. On the time panels these also expose
                # individual endpoints beyond the three-seed support window.
                terminal_x=[r['completed_updates'] if axis=='iteration' else r['last_feasible_output_seconds']/60 for r in members]
                terminal_y=[r['relative_reduction_final_float64_percent'] if normalized else r['final_float64_loss'] for r in members]
                ax.scatter(terminal_x,terminal_y,facecolors='none',edgecolors=style['color'],s=32,linewidths=1,alpha=.6,zorder=4)
            ax.set_title(f"{info['label']} · n = {info['n']:,}",loc='left',fontweight='bold')
            ax.grid(True,alpha=.20,linewidth=.6)
            if axis=='iteration':
                ax.set_xscale('symlog',linthresh=1,linscale=.5)
                ax.set_xlabel('Completed updates (log scale above 1)')
            else:
                ax.set_xlabel(TIME_LABEL)
                ax.set_xlim(left=0)
            formatter=ScalarFormatter(useOffset=False)
            formatter.set_powerlimits((-3,4))
            ax.yaxis.set_major_formatter(formatter)
            ax.ticklabel_format(axis='y',style='plain',useOffset=False)
            ax.margins(y=.08)
        axes[row,0].set_ylabel('Loss reduction from product start (%)' if normalized else 'Rounded feasible GW loss')
    handles=[Line2D([0],[0],color=style['color'],lw=2.5,label=style['label']) for style in METHODS.values()]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.952),ncol=3,frameon=False,fontsize=11)
    fig.suptitle('Pair-sampling comparison: relative loss reduction' if normalized else 'Pair-sampling comparison: feasible GW loss',
                 fontsize=17,fontweight='bold',y=.988)
    fig.text(.5,.957,'Preliminary · 3 stochastic seeds per method · 1 A100 per run · 1,024 pairs per gradient',
             ha='center',va='bottom',fontsize=10)
    caption=('Lines: median; shaded bands: full range over three seeds (not confidence intervals). Hollow circles: independently reevaluated FP64 final outputs.\n'
             'Time curves linearly interpolate logged checkpoints within each method’s three-seed overlap; sparse logs do not resolve within-interval changes.\n'
             'Time excludes diagnostics, logging, preparation and final validation. All curves use current outputs; no best-so-far selection. SHARE-seq is excluded.')
    if normalized:
        caption += '\nReduction = 100 × (initial online loss − current loss) / initial online loss, separately for each seed.'
    fig.text(.06,.018,caption,ha='left',va='bottom',fontsize=8.7,linespacing=1.4)
    fig.subplots_adjust(left=.085,right=.98,top=.89,bottom=.12 if not normalized else .135,hspace=.38,wspace=.19)
    stem='relative_loss_reduction' if normalized else 'feasible_gw_loss'
    for suffix in ('png','pdf'):
        fig.savefig(out/f'{stem}.{suffix}',dpi=200)
    if combined_pdf is not None:
        combined_pdf.savefig(fig)
    plt.close(fig)
    return [f'{stem}.png',f'{stem}.pdf']


def write_csv(path,rows):
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    source=json.loads(args.report.read_text())
    runs,excluded=validate_runs(source)
    curves,groups=aggregate(runs)
    args.out.mkdir(parents=True,exist_ok=True)
    with PdfPages(args.out/'pair_sampling_preliminary.pdf') as combined:
        combined.infodict()['Title']='Preliminary pair-sampling comparison: feasible GW loss and percentage reduction'
        artifacts=plot(curves,runs,args.out,combined_pdf=combined)+plot(curves,runs,args.out,normalized=True,combined_pdf=combined)
    artifacts += ['pair_sampling_preliminary.pdf']
    write_csv(args.out/'curve_data.csv',curves)
    observations=[];endpoints=[]
    for r in runs:
        common={key:r[key] for key in ('dataset','method','seed','metadata_sha256')}
        observations += [dict(common,**p,reduction_percent=100*(r['initial_loss']-p['rounded_loss'])/r['initial_loss']) for p in r['observations']]
        endpoints.append(dict(common,**{key:r[key] for key in ('completed_updates','initial_loss','terminal_online_loss','final_float64_loss',
            'float64_minus_online','relative_reduction_final_float64_percent','solver_seconds','last_feasible_output_seconds','total_elapsed_seconds')}))
    write_csv(args.out/'observations.csv',observations)
    write_csv(args.out/'endpoints.csv',endpoints)
    summary=dict(status='complete_for_three_included_datasets',overall_campaign_status=source.get('status'),
        generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),input_report=str(args.report.resolve()),input_report_sha256=digest(args.report),
        campaign_config_sha256=source.get('config_sha256'),script_sha256=digest(__file__),
        matplotlib_version=matplotlib.__version__,numpy_version=np.__version__,
        included_run_count=len(runs),expected_included_run_count=27,excluded_runs=excluded,
        methods=METHODS,datasets=DATASETS,seeds=SEEDS,groups=groups,
        run_bindings=[dict(dataset=r['dataset'],method=r['method'],seed=r['seed'],metadata_sha256=r['metadata_sha256'],
                           duplicate_iteration_measurements=r['duplicate_iteration_measurements']) for r in runs],
        uncertainty='Median and full min-max range across exactly three stochastic seeds; not a confidence interval.',
        clock='Recorded solver seconds + current output repair seconds. Excludes diagnostic, preparation, logging, serialization and final-validation time.',
        time_interpolation='Piecewise linear at the union of logged dataset times, restricted to the intersection of the three seed supports separately for each method; no extrapolation.',
        precision='Online full-objective checkpoint values are preserved. Hollow markers and endpoint tables use independent final FP64 evaluations; all discrepancies are retained.',
        interpretation=source.get('interpretation'),scope=source.get('scope'),
        artifacts={name:digest(args.out/name) for name in artifacts+['curve_data.csv','observations.csv','endpoints.csv']})
    (args.out/'plot_summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True,allow_nan=False)+'\n')
    lines=['# Preliminary pair-sampling plots','',
           'Three completed datasets, three methods and three stochastic seeds: **27/27 included runs**. SHARE-seq remains excluded.',
           '', '[Download both figures as a two-page PDF](pair_sampling_preliminary.pdf)',
           '', '[Feasible GW loss](feasible_gw_loss.png) · [PDF](feasible_gw_loss.pdf) · [Relative loss reduction](relative_loss_reduction.png) · [PDF](relative_loss_reduction.pdf)',
           '', 'Lines show the median and bands the full three-seed range, not confidence intervals. Curves preserve current logged outputs, without best-so-far selection.',
           '', '**Clock:** solver time plus the current repair only. Diagnostics, preparation, logging and final validation are excluded. Actual final total elapsed time is listed below. Time curves linearly interpolate sparse checkpoints within each method’s three-seed overlap; they do not identify when changes occurred between checkpoints.',
           '', '**Precision:** curves retain the online full-objective values. Hollow endpoints and this table use the independent final FP64 reevaluation. The endpoint CSV records every signed FP64-minus-online difference.',
           '', '| Dataset | Method | Final FP64 loss, median [min, max] | Reduction from start, median (%) | Final output time, median (min) | Actual total elapsed, median (min) |',
           '|---|---|---:|---:|---:|---:|']
    for g in groups:
        f=g['terminal_float64_loss']
        lines.append(f"| {DATASETS[g['dataset']]['label']} | {g['method_label']} | {f['median']:.8g} [{f['minimum']:.8g}, {f['maximum']:.8g}] | {g['relative_reduction_final_float64_percent']['median']:.4g} | {g['last_feasible_output_seconds']['median']/60:.4g} | {g['total_elapsed_seconds']['median']/60:.4g} |")
    lines += ['',source.get('scope',''),source.get('interpretation',''),'',
              f"Input report SHA256: `{digest(args.report)}`.",
              '', 'Data: [curves](curve_data.csv), [observations](observations.csv), [endpoints](endpoints.csv), [provenance and summary](plot_summary.json).']
    (args.out/'PLOTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=summary['status'],included_runs=len(runs),output=str(args.out.resolve()),figures=artifacts)))


if __name__=='__main__':
    main()
