"""Publish mean +/- one SD trajectories without changing experiment results."""
import argparse
import copy
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import tempfile

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'baseline_step_rules'
SPRING = ROOT.parent / 'spring_localcurvature'
FEASIBILITY = ROOT.parent / 'feasibility_retune'
PAPER = ROOT.parent / 'reports/paper_section/revised'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def targets(report, baseline):
    return {**baseline.destinations(report, PAPER, SPRING, FEASIBILITY),
            'baseline_report_update.json': BASE / 'report_update.json'}


def revised_source(source):
    old = r'''\caption{Trajectories for \(N=360\). Thin curves show individual
 runs at their recorded iterates; bold curves summarize work and metrics
 at the same iteration by successive medians over repetitions, slices,
 and groups. SFRAMES bands span the middle \(50\%\) of group medians.
 Training losses are relative to their initial values.
 Vertical markers indicate one quarter of the total view budget.}'''
    new = r'''\caption{Mean trajectories for \(N=360\), with shaded bands showing
 \(\pm1\) standard deviation over runs and slices, weighting acquisition
 groups equally. Work-axis summaries use the latest recorded iterate
 within each budget; curves stop at the range shared by all runs of a method.
 Training losses are relative to initialization. Vertical markers indicate
 one quarter of the total view budget.}'''
    require(source.count(old) == 1, 'The published trajectory caption changed')
    source = source.replace(old, new)
    source = source.replace(r'\caption{Trajectories for \(N=1{,}080\), summarized as in',
                            r'\caption{Mean trajectories and \(\pm1\) standard-deviation bands for \(N=1{,}080\), as in')
    require('Thin curves' not in source and 'middle \\(50' not in source,
            'A stale trajectory description remains')
    return source


def revised_appendix(source):
    old = ('Summary trajectories align by native iteration within each method: at a given iteration, '
           'summarize the work coordinate and metric separately with the same nested medians, using '
           'the common iteration range for that method. Thin lines show individual recorded runs; '
           'bands for SFRAMES span the middle half of group medians.')
    new = ('Trajectory curves show arithmetic means, with shaded bands at one descriptive standard '
           'deviation above and below the mean for every method. Acquisition groups receive equal '
           'weight; slices within a group and physical repetitions within a slice receive equal '
           'weight. Full-batch SFRAMES contributes its single deterministic run per slice. '
           'Iteration plots align at the same integer iteration. Oracle-call and processed-view '
           'plots use the latest native iterate at or below each work value, with step curves and '
           'bands. Each method stops at the smallest final observed work value (or iteration) '
           'among its runs, keeping the cohort fixed; endpoint tables still use each run\'s last '
           'iterate within the full view cap. No individual trajectories are drawn.')
    require(source.count(old) == 1, 'The previous appendix trajectory definition changed')
    updated = source.replace(old, new)
    pattern = r'<!-- CT_FEASIBILITY_RETUNE_RESULT_START -->.*?<!-- CT_FEASIBILITY_RETUNE_RESULT_END -->'
    require(re.findall(pattern, source, re.S) == re.findall(pattern, updated, re.S),
            'Historical feasibility-result block changed')
    updated += r'''

<!-- CT_TRAJECTORY_MEAN_SD_START -->
### Mean and standard-deviation trajectories

The plots use the same 448 physical evaluation traces and the same charged work counters as the corrected baseline comparison. At a plotted coordinate, the weight for repetition $r$ of slice $s$ in group $g$ is $w_{gsr}=1/(G S_g R_s)$, where $G=6$, $S_g$ counts the group's slices, and $R_s$ counts physical repetitions. The plotted arithmetic mean is $\mu=\sum w_{gsr}y_{gsr}$ and the descriptive standard deviation is $\sigma=\sqrt{\sum w_{gsr}(y_{gsr}-\mu)^2}$. Bands show $\mu\pm\sigma$, not standard errors or confidence intervals. They include variation across slices and repetitions; they are not the standard deviation of six group averages. Statistics are computed in the original metric units before axis transforms. Signed arithmetic band limits are retained; an axis uses a symmetric-log scale if needed to display nonpositive limits.

Work grids are unions of native recorded work values on the common observed range of each method. The last recorded iterate is held until its next update, without linear interpolation. Overshooting updates are excluded. The existing endpoint table and numerical comparisons retain their group-balanced median summaries.

The [plotting code](../../trajectory_mean_sd/mean_sd_plots.py) and [publication audit](../../trajectory_mean_sd/report_update.json) record this presentation change and link the per-figure mean/standard-deviation arrays. Frozen optimization traces, selections, and comparison summaries are unchanged.
<!-- CT_TRAJECTORY_MEAN_SD_END -->
'''
    return updated


def stage(report, baseline):
    dest = targets(report, baseline)
    prior_b = read(BASE / 'report_update.json')
    require(prior_b['status'] == 'completed_published', 'Baseline publication is incomplete')
    require(all(digest(path) == prior_b['published_sha256'][name]
                for name, path in dest.items() if name != 'baseline_report_update.json'),
            'Published baseline assets changed')
    before = {name: digest(path) for name, path in dest.items()}
    anchors = {'plotting_source': {'path': str(ROOT / 'mean_sd_plots.py'), 'sha256': digest(ROOT / 'mean_sd_plots.py')},
               'publisher_source': {'path': str(Path(__file__)), 'sha256': digest(Path(__file__))}}
    for name in ('corrected_summary', 'corrected_joint_summary'):
        require(digest(prior_b[name]) == prior_b[name + '_sha256'], 'Comparison summary changed')
        anchors[name] = {'path': prior_b[name], 'sha256': prior_b[name + '_sha256']}
    validation, plan, previous, joint, tune, legacy, predecessor, old_anchors = baseline.inputs(
        BASE, SPRING, FEASIBILITY, report)
    anchors.update({f'baseline_input_{k}': v for k, v in old_anchors.items()})
    plotting = module('mean_sd_original_loader', FEASIBILITY / 'report_trajectories.py')
    new_plotting = module('mean_sd_plotting', ROOT / 'mean_sd_plots.py')
    analysis, runs, slices, sizes = plotting.evaluation_traces(report.LEGACY, previous['design'], baseline.CAP)
    corrected = baseline.corrected_runs(BASE, validation, plan, previous, analysis)
    spring_runs = baseline.retained_spring_runs(previous, SPRING)
    output = Path(tempfile.mkdtemp(prefix='report_stage_', dir=ROOT))
    for name in report.ASSETS:
        shutil.copyfile(PAPER / name, output / name)
    original_source = (PAPER / report.BODY).read_text()
    source = revised_source(original_source)
    require(re.findall(r'\\begin\{table\}.*?\\end\{table\}', source, re.S) ==
            re.findall(r'\\begin\{table\}.*?\\end\{table\}', original_source, re.S), 'Endpoint table changed')
    (output / report.BODY).write_text(source)
    (output / report.APPENDIX).write_text(revised_appendix((PAPER.parent / report.APPENDIX).read_text()))
    plotting.plot_size = new_plotting.plot_size
    count, observations = baseline.plot(report, plotting, analysis, runs, corrected, previous, slices,
                                        sizes, joint, output, spring_runs)
    pages = baseline.compile_pdf(output, report)
    require(source.count(r'\begin{figure}') == 2 and source.count(r'\begin{table}') == 1,
            'Figure/table count changed')
    assets = {name: digest(output / name) for name in report.ASSETS}
    successor = {'publication_audit': str(ROOT / 'report_update.json'),
                 'scope': 'Mean plus/minus one descriptive SD for every method; no individual curves; numerical results unchanged.'}
    prior_f, prior_s = read(FEASIBILITY / 'report_update.json'), read(SPRING / 'report_update.json')
    next_f = copy.deepcopy(prior_f)
    next_f.update(plot_successor=successor, pages=pages, staged_sha256=assets, published_sha256=assets)
    write(output / 'feasibility_report_update.json', next_f)
    next_s = copy.deepcopy(prior_s)
    next_s.update(plot_successor=successor, pages=pages,
                  published_sha256={name: digest(output / name) for name in prior_s['published_sha256']})
    write(output / 'spring_report_update.json', next_s)
    next_b = copy.deepcopy(prior_b)
    next_b.update(plot_successor=successor, pages=pages,
                  published_sha256={name: digest(output / name) for name in prior_b['published_sha256']})
    write(output / 'baseline_report_update.json', next_b)
    statistics_files = {p.name: digest(p) for p in sorted(output.glob('*statistics*.json'))}
    require(len(statistics_files) == 2, 'Missing mean/SD arrays')
    audit = dict(status='staged_validated', generated_utc=datetime.now(timezone.utc).isoformat(),
                 stage=str(output), backup=str(ROOT / 'report_previous'), report_pdf=str(PAPER / f'{report.STEM}.pdf'),
                 anchors=anchors, figures=2, tables=1, pages=pages, physical_runs=count,
                 native_iteration_observations=observations, statistics=statistics_files,
                 original_sha256=before, published_sha256={name: digest(output / name) for name in dest},
                 scope=successor['scope'])
    require(all(digest(path) == before[name] for name, path in dest.items()), 'Live assets changed during staging')
    write(output / 'report_validation.json', audit)
    write(ROOT / 'staged_report.json', audit)
    return audit


def publish(report, baseline):
    dest = targets(report, baseline)
    final, pending = ROOT / 'report_update.json', ROOT / 'report_update.pending.json'
    if final.exists():
        audit = read(final)
        require(all(digest(path) == audit['published_sha256'][name] for name, path in dest.items()),
                'Published mean/SD report changed')
        return audit
    audit = read(pending if pending.exists() else ROOT / 'staged_report.json')
    output, backup = Path(audit['stage']), Path(audit['backup'])
    require(output.parent == ROOT and output.name.startswith('report_stage_') and backup == ROOT / 'report_previous',
            'Unexpected publication paths')
    require(all(digest(v['path']) == v['sha256'] for v in audit['anchors'].values()), 'Publication inputs changed')
    require(all(digest(output / name) == sha for name, sha in audit['statistics'].items()), 'Plot statistics changed')
    for name, path in dest.items():
        require(digest(output / name) == audit['published_sha256'][name], f'Staged asset changed: {name}')
        require(digest(path) in (audit['original_sha256'][name], audit['published_sha256'][name]),
                f'Unrelated live edit: {name}')
    backup.mkdir(exist_ok=True)
    for name, path in dest.items():
        if not (backup / name).exists():
            require(digest(path) == audit['original_sha256'][name], 'Cannot reconstruct an original backup')
            report.atomic_copy(path, backup / name)
        require(digest(backup / name) == audit['original_sha256'][name], 'Backup changed')
    audit.update(status='completed_published', published_utc=datetime.now(timezone.utc).isoformat())
    write(pending, audit)
    try:
        for name, path in dest.items():
            report.atomic_copy(output / name, path)
        require(all(digest(path) == audit['published_sha256'][name] for name, path in dest.items()), 'Publication mismatch')
        report.atomic_copy(pending, final)
    except BaseException:
        for name, path in dest.items():
            report.atomic_copy(backup / name, path)
        raise
    pending.unlink()
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    with ExitStack() as stack:
        for folder in (ROOT, BASE, SPRING, FEASIBILITY):
            lock = stack.enter_context((folder / 'report_update.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = module('mean_sd_base_report', FEASIBILITY / 'report.py')
        baseline = module('mean_sd_baseline_report', BASE / 'update_report.py')
        audit = publish(report, baseline) if args.publish else stage(report, baseline)
        print(json.dumps({k: audit[k] for k in ('status', 'stage', 'report_pdf', 'figures', 'tables', 'pages')}, indent=2))


if __name__ == '__main__':
    main()
