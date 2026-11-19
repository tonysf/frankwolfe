"""Publish the matched gradient-budget continuation, retaining all earlier results."""
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
TRAJECTORIES = ROOT.parent / 'trajectory_mean_sd'
PAPER = ROOT.parent / 'reports/paper_section/revised'
METHODS = ('palm', 'spring_sarah')
CAP, COUNT = 69120, 168


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


def key(case):
    return case['slice_id'], int(case['seed']), int(case['n_train'])


def gradient_views(row, direction='forward'):
    return row[f'{direction}_views'] - row.get(f'curvature_{direction}_views', 0)


def with_gradient_counts(row):
    value = dict(row)
    for direction in ('forward', 'backward'):
        name = f'gradient_{direction}_views'
        derived = gradient_views(row, direction)
        require(name not in row or row[name] == derived, 'Gradient work counter differs')
        value[name] = derived
        value.setdefault(f'curvature_{direction}_views', 0)
    value.setdefault('curvature_operator_calls', 0)
    value.setdefault('oracle_calls', value['gradient_calls'] + value['curvature_operator_calls'])
    return value


def within_gradient_cap(trace, cap=CAP):
    return [with_gradient_counts(row) for row in trace
            if max(gradient_views(row), gradient_views(row, 'backward')) <= cap]


def probe(trace, cap, gradient=True):
    within = within_gradient_cap(trace, cap) if gradient else [with_gradient_counts(row) for row in trace
        if max(row['forward_views'], row['backward_views']) <= cap]
    require(within, 'Missing within-budget native observation')
    return within[-1]


def target_hit(run, target, analysis):
    trace = within_gradient_cap(run['trace'])
    norm = analysis.initial_factor_norm(run['meta'])
    initial = trace[0]['feasible_loss']
    selected = next((row for row in trace if row['feasible_loss']/initial <= target['relative_repaired_loss']
        and row['d_distance']/norm <= target['max_relative_d_distance']), None)
    if selected is None:
        return dict(status='right_censored', reached=False, view_cap=CAP,
                    first_views=None, lower_bound_views=CAP, initial_factor_frobenius_norm=norm,
                    budget_component='gradient_views')
    return dict(status='reached', reached=True, first_views=selected['gradient_forward_views'],
        first_total_views=selected['forward_views'], first_curvature_views=selected['curvature_forward_views'],
        iteration=selected['iteration'], gradient_calls=selected['gradient_calls'],
        oracle_calls=selected['oracle_calls'], relative_repaired_loss=selected['feasible_loss']/initial,
        relative_d_distance=selected['d_distance']/norm, initial_factor_frobenius_norm=norm,
        budget_component='gradient_views')


def grouped(cases, groups, analysis, getter):
    values = {}
    for case in cases:
        values.setdefault(case['slice_id'], []).append(getter(case))
    return analysis.grouped_stat(values, groups)


def replace_summary(previous, extended, analysis, anchors):
    """Select by gradient work, but retain actual total counters in every probe."""
    updated = copy.deepcopy(previous)
    groups = analysis.group_map(previous['design'])
    for case in updated['cases']:
        for method in METHODS:
            run = extended[(*key(case), method)]
            old = case['arms'][method]
            arm = copy.deepcopy(old)
            arm.update(run_id=run['config']['run_id'], source_directory=run['source_directory'],
                origin='continued_from_published_checkpoint', continuation=run['meta'].get('continuation'),
                imported_checkpoint=run['meta'].get('imported_checkpoint'),
                optimizer=run['meta']['optimizer'], checkpoint_iteration=run['trace'][-1]['iteration'],
                source_sha256=run['meta']['source_sha256'], superseded_run_id=old['run_id'],
                inclusive_budget_probes=copy.deepcopy(old['probes']))
            arm['canonical_source_meaning'] = 'Unchanged original numerical core; continuation engine sources are recorded separately in source_sha256.'
            arm['probes'] = {'original_cap': probe(run['trace'], CAP//4),
                             'view_cap': probe(run['trace'], CAP), 'endpoint': probe(run['trace'], CAP)}
            arm['last_recorded_observation'] = with_gradient_counts(run['trace'][-1])
            if updated.get('targets'):
                arm['target_hits'] = {t['name']: target_hit(run, t, analysis)
                                      for t in updated['targets']['targets']}
            # The original inclusive-cap endpoint must remain exactly reproducible.
            retained = probe(run['trace'], CAP, gradient=False)
            require(all(retained[k] == v for k, v in old['probes']['view_cap'].items()),
                    'Continuation changed the original inclusive-cap endpoint')
            case['arms'][method] = arm
    attribution = {}
    old_cases = {key(c): c for c in previous['cases']}
    for n, cohort in updated['cohorts'].items():
        subset = [c for c in updated['cases'] if c['n_train'] == int(n)]
        for budget in ('original_cap', 'view_cap', 'endpoint'):
            cohort[budget] = dict(matched_all_methods=analysis.marginal_summary(subset, budget, groups),
                sframes_vs={m: analysis.paired_summary(subset, m, budget, groups)
                            for m in analysis.VARIANTS if m != analysis.ANCHOR})
        if updated.get('targets'):
            cohort['accuracy_targets'] = {t['name']: analysis.target_summary(subset, t['name'], groups, CAP)
                                          for t in updated['targets']['targets']}
        attribution[n] = {}
        for method in METHODS:
            def endpoint(c):
                return c['arms'][method]['probes']['view_cap']
            def original(c):
                return old_cases[key(c)]['arms'][method]['probes']['view_cap']
            fields = {
                'paired_repaired_loss_change_percent': lambda c: 100*(endpoint(c)['feasible_loss']/original(c)['feasible_loss']-1),
                'paired_raw_loss_change_percent': lambda c: 100*(endpoint(c)['raw_loss']/original(c)['raw_loss']-1),
                'old_repaired_loss': lambda c: original(c)['feasible_loss'],
                'new_repaired_loss': lambda c: endpoint(c)['feasible_loss'],
                'old_iterations': lambda c: original(c)['iteration'],
                'new_iterations': lambda c: endpoint(c)['iteration'],
                'gradient_views': lambda c: endpoint(c)['gradient_forward_views'],
                'curvature_views': lambda c: endpoint(c)['curvature_forward_views'],
                'total_views': lambda c: endpoint(c)['forward_views'],
                'total_to_gradient_ratio': lambda c: endpoint(c)['forward_views']/endpoint(c)['gradient_forward_views'],
                'curvature_fraction': lambda c: endpoint(c)['curvature_forward_views']/endpoint(c)['forward_views'],
            }
            attribution[n][method] = {field: grouped(subset, groups, analysis, getter) for field, getter in fields.items()}
    for before, after in zip(previous['cases'], updated['cases']):
        require(key(before) == key(after) and all(before['arms'][m] == arm for m, arm in after['arms'].items()
                if m not in METHODS), 'An unaffected arm changed')
    updated['generated_utc'] = datetime.now(timezone.utc).isoformat()
    updated['inclusive_reference_budgets'] = copy.deepcopy(previous['budgets'])
    updated['budgets'] = dict(gradient_forward_view_cap=CAP, gradient_backward_view_cap=CAP,
        original_gradient_forward_view_cap=CAP//4, inclusive_reference_view_cap=CAP,
        probe_budget_component='gradient_views', recorded_forward_backward_views='actual total including curvature')
    updated['gradient_budget_extension'] = dict(anchors=anchors, replaced_methods=list(METHODS),
        physical_continuations=COUNT, original_inclusive_summary=anchors['previous_summary'],
        attribution=attribution, scope='Continue fixed methods; compare native endpoints at equal gradient-only view caps.')
    return updated


def attribution_text(updated):
    result = []
    stats = updated['gradient_budget_extension']['attribution']
    for method, label in (('palm', 'PALM'), ('spring_sarah', 'SPRING--SARAH')):
        changes = [stats[n][method]['paired_repaired_loss_change_percent']['median_across_groups'] for n in ('360', '1080')]
        ratios = [stats[n][method]['total_to_gradient_ratio']['median_across_groups'] for n in ('360', '1080')]
        result.append(rf'\(({changes[0]:+.1f}\%,{changes[1]:+.1f}\%)\) for {label}')
        if method == 'palm': palm_ratio = ratios
        else: spring_ratio = ratios
    return (r'Relative to the former cap on total views, the paired repaired-loss changes are '
        + ' and '.join(result) + r' at \(N=360,1{,}080\), respectively. '
        + rf'Including curvature, total views are \({palm_ratio[0]:.1f}\times\) gradient views for PALM '
        + rf'and \({spring_ratio[0]:.1f}\times,{spring_ratio[1]:.1f}\times\) for SPRING--SARAH. '
        + 'All proximal baselines remain nonnegative. This comparison uses equal gradient work.')


def source_text(source, updated, joint, report, legacy):
    require(source.count('equal projection budgets.') == 1, 'The experiment opening changed')
    source = source.replace('equal projection budgets.', 'equal gradient-view budgets.')
    old = r'''We compare the last iterate within \(69{,}120\) forward and adjoint
views per direction, including snapshot, recursive-gradient, and curvature work.
An oracle call is one batch gradient or one block-Hessian application.'''
    new = r'''We compare the last iterate within \(69{,}120\) gradient forward and
adjoint views per direction, including snapshots and recursive differences.
Curvature-estimation views are recorded separately and excluded from this cap.
A gradient call evaluates one joint or block gradient on a batch.'''
    require(source.count(old) == 1, 'The previous budget definition changed')
    source = source.replace(old, new)
    old = ('The proximal baselines remain nonnegative throughout. All curvature and\n'
           'snapshot work is charged to the view budget; PALM completes 13 and 4\n'
           'two-block updates at the two view counts, respectively.')
    require(source.count(old) == 1, 'The previous cost comparison changed')
    source = source.replace(old, attribution_text(updated))
    source = source.replace('show progress versus iterations, oracle calls,\nand processed views.',
                            'show progress versus iterations, gradient calls,\nand gradient views.')
    report.LABELS['prox_sarah'] = 'ProxSARAH-v3'
    table = report.quality_table(legacy, updated, joint).replace('at a common view budget.',
        'at a common gradient-view budget; curvature work is excluded from this cap.')
    source, number = re.subn(r'\\begin\{table\}.*?\\end\{table\}', lambda _: table, source, count=1, flags=re.S)
    require(number == 1, 'The endpoint table is missing')
    source = source.replace('Both estimate current block curvature by five power iterations followed',
        'Both retain their original steps and sampling rules during continuation.\n'
        'They estimate current block curvature by five power iterations followed')
    source = source.replace('groups equally. Work-axis summaries use the latest recorded iterate',
        'groups equally. Gradient-work summaries use the latest recorded iterate')
    source = source.replace('one quarter of the total view budget.', 'one quarter of the gradient-view budget.')
    require(source.count(r'\begin{figure}') == 2 and source.count(r'\begin{table}') == 1,
            'The comparison added a figure or table')
    return source


def appendix_text(source, updated):
    require('<!-- CT_GRADIENT_BUDGET_START -->' not in source, 'Gradient-budget appendix already exists')
    notice = ('\n\nThe current paper uses the gradient-only view budget described in the final '
        'section below. Earlier total-view comparisons are retained as historical records. '
        'The data, parameter choices, initialization and evaluation cohort are unchanged.')
    split = source.index('\n\n## Data and view splits')
    updated_source = source[:split] + notice + source[split:]
    updated_source = updated_source.replace('Oracle-call and processed-view plots use the latest native iterate',
        'Gradient-call and gradient-view plots use the latest native iterate')
    updated_source += r'''

<!-- CT_GRADIENT_BUDGET_START -->
## Gradient-budget continuation

The current endpoint table and both existing mean/standard-deviation figures use a common cap of 69,120 gradient forward views and 69,120 gradient adjoint views. For every native row, gradient views equal total views minus curvature views. Full snapshots and both recursive-difference gradient evaluations count toward this cap. Hessian applications remain included in total operator views and `oracle_calls`, but are excluded from the displayed `gradient_calls` and gradient-view axes. This comparison does not impose equal total operator work.

PALM and SPRING–SARAH continue their 84 published runs each from the final saved optimizer state, preserving random-generator states, recursive estimates, factors and complete recorded prefixes. Their steps, power iterations, sampling and refresh rules are unchanged. The old total-budget overshooting update is a valid prefix of the extended trajectory. New endpoints exclude any update exceeding the gradient cap. Loss and feasibility are recorded at every native iteration. ProxSARAH and all SFRAMES traces are unchanged; no new tuning is performed.

The 448 physical trajectories retain equal group/slice/repetition weighting for their arithmetic means and descriptive ±1 SD bands. Native step alignment and each method's common observed support are unchanged. The endpoint table still uses successive medians over repetitions, slices within groups and groups; all best-value bolding is recomputed. First-hit results are recomputed using gradient work and retain actual total and curvature work at each hit. The original inclusive-budget comparison remains frozen and is linked from the new derived summary.

### Effect of extending the alternating methods

Changes are paired within slice and seed before the same median aggregation. Costs are directional; forward and adjoint totals agree.

| Training views | Method | Old repaired loss | Extended repaired loss | Paired change (%) | Gradient views | Curvature views | Total / gradient |
| --- | --- | --- | --- | --- | --- | --- | --- |
'''
    for n in ('360', '1080'):
        for method, label in (('palm', 'PALM'), ('spring_sarah', 'SPRING–SARAH')):
            s = updated['gradient_budget_extension']['attribution'][n][method]
            val = lambda name: s[name]['median_across_groups']
            updated_source += (f"| {n} | {label} | {val('old_repaired_loss'):.6g} | {val('new_repaired_loss'):.6g} | "
                f"{val('paired_repaired_loss_change_percent'):+.2f} | {val('gradient_views'):,.0f} | "
                f"{val('curvature_views'):,.0f} | {val('total_to_gradient_ratio'):.3f} |\n")
    updated_source += ('\nThe [continuation notes](../../gradient_budget/README.md), '
        '[completion audit](../../gradient_budget/validation_summary.json), and '
        '[publication audit](../../gradient_budget/report_update.json) specify frozen inputs, '
        'checkpoint provenance, work accounting and the derived comparison summary. '
        'Historical trace files, selections and summaries remain unchanged.\n'
        '<!-- CT_GRADIENT_BUDGET_END -->\n')
    marker = r'<!-- CT_FEASIBILITY_RETUNE_RESULT_START -->.*?<!-- CT_FEASIBILITY_RETUNE_RESULT_END -->'
    require(re.findall(marker, source, re.S) == re.findall(marker, updated_source, re.S),
            'Historical feasibility marker changed')
    return updated_source


def targets(report, baseline):
    return {**baseline.destinations(report, PAPER, SPRING, FEASIBILITY),
            'baseline_report_update.json': BASE/'report_update.json',
            'trajectory_report_update.json': TRAJECTORIES/'report_update.json'}


def inputs(report, baseline):
    fixture = read(ROOT/'report_fixture_validation.json')
    require(fixture['status'] == 'passed' and fixture['updater_sha256'] == digest(Path(__file__))
        and fixture['validator_sha256'] == digest(ROOT/'validate_report.py')
        and fixture['plotting_sha256'] == digest(ROOT/'plot_trajectories.py'), 'Report fixture is stale')
    progress, validation = read(ROOT/'progress.json'), read(ROOT/'validation_summary.json')
    plan, protocol = read(ROOT/'evaluation_plan.json'), read(ROOT/'protocol.json')
    require((progress['status'], progress['phase']) in (('evaluated','evaluation_validated'), ('completed','report_published'))
        and validation['status'] == 'completed_validated' and len(validation['cases']) == len(plan['runs']) == COUNT,
        'Publication requires all 168 validated continuations')
    for name in ('protocol', 'evaluation_plan'):
        require(validation[name+'_sha256'] == digest(ROOT/(name+'.json')), 'Continuation validation anchor changed')
    current = read(TRAJECTORIES/'report_update.json')
    require(current['status'] == 'completed_published', 'Mean/SD predecessor is not published')
    require(current == read(ROOT/'predecessor_publication.json'),
            'The prepared mean/SD predecessor changed during continuation')
    dest = targets(report, baseline)
    require(all(digest(path) == current['published_sha256'][name] for name,path in dest.items()
                if name != 'trajectory_report_update.json'), 'Predecessor publication differs from live files')
    prior_b = read(BASE/'report_update.json')
    require(prior_b == read(ROOT/'predecessor_baseline_publication.json'),
            'The prepared baseline comparison changed during continuation')
    previous, joint = (read(prior_b[name]) for name in ('corrected_summary', 'corrected_joint_summary'))
    anchors = {'previous_summary': {'path':prior_b['corrected_summary'], 'sha256':prior_b['corrected_summary_sha256']},
        'unchanged_joint_summary': {'path':prior_b['corrected_joint_summary'], 'sha256':prior_b['corrected_joint_summary_sha256']}}
    for name,path in dict(validation=ROOT/'validation_summary.json',plan=ROOT/'evaluation_plan.json',
        protocol=ROOT/'protocol.json',report_builder=Path(__file__),report_validator=ROOT/'validate_report.py',
        report_fixture=ROOT/'report_fixture_validation.json',plotting=ROOT/'plot_trajectories.py',
        reference_plotting=TRAJECTORIES/'mean_sd_plots.py',baseline_helper=BASE/'update_report.py',
        feasibility_helper=FEASIBILITY/'report.py',legacy_analysis=report.LEGACY/'analyze.py',
        base_plot_loader=FEASIBILITY/'report_trajectories.py',campaign=ROOT/'campaign.py',readme=ROOT/'README.md',
        predecessor_publication=ROOT/'predecessor_publication.json',
        predecessor_baseline_publication=ROOT/'predecessor_baseline_publication.json').items():
        anchors[name] = {'path':str(path),'sha256':digest(path)}
    require(all(digest(v['path']) == v['sha256'] for v in anchors.values()), 'Scientific/report input changed')
    return validation,plan,protocol,previous,joint,current,anchors


def extended_runs(validation, plan, protocol):
    campaign = module('gradient_report_campaign', ROOT/'campaign.py')
    campaign.verify_prepared()
    configs = {c['run_id']:c for c in plan['runs']}
    extended = {}
    for case in validation['cases']:
        cfg = configs[case['run_id']]
        checked = campaign.validate_result(cfg, plan['assignments'][case['run_id']], plan, protocol)
        require(checked == case, 'A continuation changed after validation')
        folder = Path(case['source_directory'])
        meta, trace = read(folder/'metadata.json'), read(folder/'trace.json')
        require(digest(folder/'trace.json') == case['trace_sha256'], 'Continuation trace hash changed')
        k = (*key(case),case['method'])
        require(k not in extended, 'Duplicate continuation case')
        extended[k] = dict(config=cfg, meta=meta, trace=trace, source_directory=str(folder))
    require(len(extended) == COUNT, 'Missing continuation')
    return extended


def load_retained_runs(report, baseline):
    old_validation, old_plan, previous_spring, joint, tune, legacy, predecessor, anchors = baseline.inputs(
        BASE, SPRING, FEASIBILITY, report)
    plotting = module('gradient_retained_loader', FEASIBILITY/'report_trajectories.py')
    analysis,runs,slices,sizes = plotting.evaluation_traces(report.LEGACY,previous_spring['design'],CAP)
    summary_variants = dict(analysis.VARIANTS)
    corrected = baseline.corrected_runs(BASE,old_validation,old_plan,previous_spring,analysis)
    spring = baseline.retained_spring_runs(previous_spring,SPRING)
    plotting.plot_size = lambda *args,**kwargs: None
    baseline.plot(report,plotting,analysis,runs,corrected,previous_spring,slices,sizes,joint,ROOT,spring)
    analysis.plot_variants = dict(analysis.VARIANTS)
    analysis.VARIANTS = summary_variants
    return analysis,runs,slices,sizes,legacy,anchors


def render_plots(analysis,runs,extended,slices,sizes,groups,output):
    plotting = module('gradient_budget_plots',ROOT/'plot_trajectories.py')
    analysis.VARIANTS = analysis.plot_variants
    for k,run in extended.items():
        require(k in runs, 'Continuation does not match the displayed cohort')
        runs[k] = run
    for k,run in list(runs.items()):
        runs[k] = dict(run,trace=within_gradient_cap(run['trace']))
    require(len(runs) == 448, 'Wrong physical figure cohort')
    plotting.plt.rcParams.update({'font.size':8,'axes.titlesize':8,'axes.labelsize':7.8,
        'xtick.labelsize':6.5,'ytick.labelsize':7,'axes.edgecolor':'0.55','axes.linewidth':.6,'pdf.fonttype':42})
    for size in sizes:
        plotting.plot_size(analysis,runs,slices,groups,size,CAP,CAP//4,output)
    return len(runs),sum(len(run['trace']) for run in runs.values())


def chain_audits(output,report,successor,pages):
    assets={name:digest(output/name) for name in report.ASSETS}
    audit_paths=[('feasibility_report_update.json',FEASIBILITY),('spring_report_update.json',SPRING),
                 ('baseline_report_update.json',BASE),('trajectory_report_update.json',TRAJECTORIES)]
    for name,folder in audit_paths:
        predecessor=copy.deepcopy(read(folder/'report_update.json'))
        predecessor.update(gradient_budget_successor=successor,pages=pages)
        predecessor['published_sha256']={key:digest(output/key) for key in predecessor['published_sha256']}
        if folder==FEASIBILITY:predecessor['staged_sha256']=assets
        write(output/name,predecessor)


def stage(report, baseline):
    validation,plan,protocol,previous,joint,prior_t,anchors = inputs(report,baseline)
    dest = targets(report,baseline)
    before = {name:digest(path) for name,path in dest.items()}
    analysis,runs,slices,sizes,legacy,retained_anchors = load_retained_runs(report,baseline)
    anchors.update({f'retained_{k}':v for k,v in retained_anchors.items()})
    extended = extended_runs(validation,plan,protocol)
    updated = replace_summary(previous,extended,analysis,anchors)
    output = Path(tempfile.mkdtemp(prefix='report_stage_',dir=ROOT))
    for name in report.ASSETS:
        shutil.copyfile(PAPER/name,output/name)
    write(output/'gradient_comparison_summary.json',updated)
    (output/report.BODY).write_text(source_text((PAPER/report.BODY).read_text(),updated,joint,report,legacy))
    (output/report.APPENDIX).write_text(appendix_text((PAPER.parent/report.APPENDIX).read_text(),updated))
    count,observations = render_plots(analysis,runs,extended,slices,sizes,analysis.group_map(previous['design']),output)
    pages = baseline.compile_pdf(output,report)
    successor = dict(publication_audit=str(ROOT/'report_update.json'),gradient_comparison_summary=str(output/'gradient_comparison_summary.json'),
        gradient_comparison_summary_sha256=digest(output/'gradient_comparison_summary.json'),
        scope='Extend PALM and SPRING at equal gradient-only view budgets; preserve curvature and total work separately.')
    chain_audits(output,report,successor,pages)
    stats = {p.name:digest(p) for p in output.glob('statistics_*.json')}
    require(len(stats)==2,'Missing mean/SD statistics')
    audit = dict(status='staged_validated',generated_utc=datetime.now(timezone.utc).isoformat(),
        stage=str(output),backup=str(ROOT/'report_previous'),report_pdf=str(PAPER/(report.STEM+'.pdf')),
        figures=2,tables=1,pages=pages,physical_runs=count,continued_runs=COUNT,native_iteration_observations=observations,
        anchors=anchors,statistics=stats,original_sha256=before,
        published_sha256={name:digest(output/name) for name in dest},**{k:v for k,v in successor.items() if k!='publication_audit'})
    require(all(digest(path)==before[name] for name,path in dest.items()),'Live publication changed while staging')
    require(all(digest(v['path'])==v['sha256'] for v in anchors.values()),'Inputs changed while staging')
    write(output/'report_validation.json',audit)
    write(ROOT/'staged_report.json',audit)
    return audit


def finish(root,report,audit,dest,pending):
    output,backup = Path(audit['stage']),Path(audit['backup'])
    require(output.parent==root and output.name.startswith('report_stage_') and backup==root/'report_previous','Unexpected publication path')
    for name,path in dest.items():
        require(digest(output/name)==audit['published_sha256'][name] and digest(backup/name)==audit['original_sha256'][name],
                'Stage or backup changed')
        require(digest(path) in (audit['original_sha256'][name],audit['published_sha256'][name]),'Unrelated live edit prevents recovery')
    try:
        for name,path in dest.items():report.atomic_copy(output/name,path)
        require(all(digest(path)==audit['published_sha256'][name] for name,path in dest.items()),'Publication mismatch')
        report.atomic_copy(pending,root/'report_update.json')
    except BaseException:
        for name,path in dest.items():report.atomic_copy(backup/name,path)
        raise
    pending.unlink(missing_ok=True)
    return audit


def publish(report,baseline):
    dest = targets(report,baseline)
    final,pending = ROOT/'report_update.json',ROOT/'report_update.pending.json'
    if final.exists():
        audit=read(final)
        require(all(digest(path)==audit['published_sha256'][name] for name,path in dest.items()),'Published report changed')
        require(digest(audit['gradient_comparison_summary'])==audit['gradient_comparison_summary_sha256'],'Derived summary changed')
        return audit
    if pending.exists():
        return finish(ROOT,report,read(pending),dest,pending)
    audit=stage(report,baseline)
    require(all(digest(v['path'])==v['sha256'] for v in audit['anchors'].values()),'Publication input changed')
    backup=ROOT/'report_previous';backup.mkdir(exist_ok=True)
    for name,path in dest.items():
        if not (backup/name).exists():report.atomic_copy(path,backup/name)
        require(digest(backup/name)==audit['original_sha256'][name],'Backup belongs to another publication')
    audit.update(status='completed_published',published_utc=datetime.now(timezone.utc).isoformat())
    write(pending,audit)
    return finish(ROOT,report,audit,dest,pending)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    with ExitStack() as stack:
        for folder in (ROOT,TRAJECTORIES,BASE,SPRING,FEASIBILITY):
            lock=stack.enter_context((folder/'report_update.lock').open('a'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        report=module('gradient_base_report',FEASIBILITY/'report.py')
        baseline=module('gradient_baseline_report',BASE/'update_report.py')
        audit=publish(report,baseline) if args.publish else stage(report,baseline)
        print(json.dumps({k:audit[k] for k in ('status','stage','report_pdf','figures','tables','pages')},indent=2))


if __name__=='__main__':main()
