"""Two-page CT experiment section; detailed evidence stays in the Git supplement."""
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

try:
    from . import robustness_report as evidence
    from . import compact_plots
except ImportError:
    import robustness_report as evidence
    import compact_plots

read, write, digest = evidence.read, evidence.write, evidence.digest
ASSETS = ('ct_experiments.tex', 'sframes_ct_section.tex', 'ct_reproducibility_link.tex',
          'trajectories_360.pdf', 'trajectories_360.png', 'trajectories_1080.pdf',
          'trajectories_1080.png', 'sframes_ct_section.pdf')
LEGACY_LABELS = {'sframes': 'SFRAMES (loss)', 'tuned_sframes': 'SFRAMES (joint)',
    'sframes_full': 'SFRAMES (full)', 'spring_sarah': 'SPRING--SARAH',
    'prox_sarah': 'ProxSARAH', 'palm': 'PALM', 'anls': 'Alternating NNLS',
    'mu': 'Multiplicative', 'joint_lbfgsb': 'L-BFGS-B'}


def driver(include_palm=True):
    source = r'''\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage[letterpaper,margin=0.85in]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx}
\usepackage[hidelinks]{hyperref}
\setlength{\emergencystretch}{2em}
\setlength{\textfloatsep}{9pt}
\setlength{\intextsep}{9pt}
\setlength{\abovecaptionskip}{4pt}
\setcounter{topnumber}{3}
\renewcommand{\topfraction}{0.95}
\renewcommand{\textfraction}{0.05}
\hypersetup{pdftitle={Numerical experiments}}
\input{ct_reproducibility_link.tex}
\begin{document}
\input{ct_experiments.tex}
\begingroup\small
\begin{thebibliography}{9}\setlength{\itemsep}{0pt}
\bibitem{detect} M.~B.~Kiss et al. 2DeteCT. \emph{Scientific Data} 10:576, 2023.
\bibitem{spring} D.~Driggs et al. SPRING. \emph{SIAM J. Imaging Sci.} 14:1932--1970, 2021.
\bibitem{proxsarah} N.~H.~Pham et al. ProxSARAH. \emph{JMLR} 21(110):1--48, 2020.
\bibitem{palm} J.~Bolte, S.~Sabach, and M.~Teboulle. PALM. \emph{Math. Program.} 146:459--494, 2014.
\end{thebibliography}\endgroup
\end{document}
'''
    if not include_palm:
        source = source.replace(
            r'\bibitem{palm} J.~Bolte, S.~Sabach, and M.~Teboulle. PALM. \emph{Math. Program.} 146:459--494, 2014.'+'\n', '')
    return source


def setup(expanded=False, selection=None):
    views = r'120,360,1080,2160' if expanded else r'360,1080'
    joint, loss = (4., .1), (16., .1)
    if expanded:
        s = selection['global_selection']['sframes_b64']
        joint = tuple(s['joint']['parameters'][k] for k in ('beta_scale', 'gamma0'))
        loss = tuple(s['loss']['parameters'][k] for k in ('beta_scale', 'gamma0'))
    return r'''\section{Numerical experiments}
\label{sec:ct-experiments}
\paragraph{Low-rank CT reconstruction.}
We use 14 low-dose CT slices from six acquisition groups in 2DeteCT~\cite{detect}.
Training indices \(\mathcal I\) comprise \(N\in\{''' + views + r'''\}\) views, with 360 views held out.
Each measured view \(y_i\) contains the rays at one source angle. The linear map
\(P_i\) sums pixels weighted by ray path lengths; \(P_i^\ast\) backprojects residuals.
For a \(512\times512\) image \(UV^\top\), with
\(U,V\in\mathbb R^{512\times64}\), the data-fitting loss is
\begin{equation}
 f(U,V)=\frac{1}{2Nd}\sum_{i\in\mathcal I}
       \|P_i(UV^\top)-y_i\|_2^2,\qquad d=956.
 \label{eq:ct-loss}
\end{equation}
Nonnegative factors ensure nonnegative attenuation; spectral caps give
SFRAMES a bounded domain. Define
\[
 C=\{(U,V):\|U\|_{\rm op}\le R_U,\ \|V\|_{\rm op}\le R_V\},
 \qquad D=\{(U,V):U,V\ge0\}.
\]
This is \(\min_{x\in C}f(x)+g(Tx)\), with \(x=(U,V)\),
\(T=\operatorname{Id}\), and \(g=\iota_D\), the indicator of \(D\).

\paragraph{Methods.}
SFRAMES samples \(b=64\) views uniformly without replacement, independently
at each iteration. Its unbiased minibatch gradient is
\[
 Z_k=\frac1{bd}\sum_{i\in B_k}P_i^\ast(P_i(U_kV_k^\top)-y_i),
 \qquad \widehat G_k=(Z_kV_k,Z_k^\top U_k).
\]
It averages these gradients with weight \((k+1)^{-1/2}\), starting from
\(\widehat G_0\), and adds the exact penalty gradient
\((x_k-(x_k)_+)/\beta_k\) before the spectral LMO.
''' + (r'''We compare with deterministic FraMES, full-batch SFRAMES,
SPRING--SARAH~\cite{spring}, ProxSARAH~\cite{proxsarah}, and joint L-BFGS-B;
the latter optimizes over \(D\) alone. Separate controls vary batch size,
averaging, rank, radius and initialization, with six fresh confirmation slices.
''' if expanded else r'''We compare with full-batch SFRAMES, SPRING--SARAH~\cite{spring},
ProxSARAH~\cite{proxsarah}, PALM~\cite{palm}, and nonnegative alternating
NNLS, multiplicative updates and joint L-BFGS-B. The last three optimize
over \(D\) alone.
''') + r'''
All methods share a nonnegative SVD initialization from training-view
backprojection, with \(R_U=2\|U_0\|_{\rm op}\) and
\(R_V=2\|V_0\|_{\rm op}\). SFRAMES uses
\(\gamma_k=\gamma_0(k+1)^{-3/4}\), \(\beta_k=\beta_0(k+1)^{-1/4}\), and
\(\beta_0=c_\beta/\max(L_U^0,L_V^0)\), using initial block-curvature estimates.
Separate development slices select
''' + (rf'\((c_\beta,\gamma_0)=({loss[0]:g},{loss[1]:g})\) for loss and '
       rf'\(({joint[0]:g},{joint[1]:g})\) for joint loss--feasibility performance.'
       '\n') + r'''The proximal baselines use their papers' step rules, with development-only
tuning. Complete settings and screening procedures are in the \CTReproducibility{}.

We compare at 69,120 gradient/solver views per direction, including all
line-search trials; curvature work is recorded separately.
Raw feasibility is measured by \(d_D(x)=\|x-x_+\|_F\).
Evaluated loss clips factors to nonnegativity and radially restores their
spectral bounds; nonnegative-only methods use their actual factors.
Reference error compares this image with the high-dose reconstruction.
Means and SDs over ''' + ('five' if expanded else 'three') + r''' repetitions and slices, and endpoint medians,
balance acquisition groups.
'''


def legacy_table(summary):
    lines = [r'\begin{table}[!htbp]\centering\small',
        r'\caption{Group-balanced endpoint medians: evaluated loss ($\times10^3$), reference error, and raw infeasibility ($100d_D(x)/\|x_0\|_F$). Bold marks the best value per column.}',
        r'\label{tab:ct-results}\setlength{\tabcolsep}{5pt}',
        r'\begin{tabular}{@{}lrrrrrr@{}}\toprule',
        r'& \multicolumn{3}{c}{$N=360$} & \multicolumn{3}{c}{$N=1080$} \\',
        r'Method & Loss & Error & Infeas. & Loss & Error & Infeas. \\\midrule']
    for key, label in LEGACY_LABELS.items():
        values = []
        for n in ('360', '1080'):
            methods = summary['cohorts'][n]['methods']
            for metric, scale, precision in [('feasible_loss',1000,2),('reference_relative_error',1,3),('relative_d_distance',100,2)]:
                value = methods[key][metric]['median_across_groups']
                best = min(methods[m][metric]['median_across_groups'] for m in LEGACY_LABELS)
                number = f'{scale*value:.{precision}f}'
                values.append('$'+(r'\mathbf{'+number+'}' if abs(value-best)<1e-12 else number)+'$')
        lines.append(label + ' & ' + ' & '.join(values) + r' \\')
    return '\n'.join(lines + [r'\bottomrule\end{tabular}\end{table}'])


def figure(name, caption):
    return ('\n\\begin{figure}[!htbp]\n\\centering\n'
            '\\includegraphics[width=\\linewidth]{trajectories_'+str(name)+'.pdf}\n'
            '\\caption{'+caption+'}\n\\label{fig:ct-trajectories-'+str(name)+'}\n\\end{figure}\n')


def legacy_body(summary):
    return setup() + r'''
\paragraph{Results.}
Minibatch SFRAMES attains lower evaluated loss than full-batch SFRAMES and
ProxSARAH at both view counts (Table~\ref{tab:ct-results}). Joint tuning
reduces raw relative infeasibility from about \(2\%\) to \(0.67\%\), with a
modest loss increase. At \(N=360\), L-BFGS-B improves evaluated loss over
loss-tuned SFRAMES but worsens reference error on all 14 slices: better data
fitting need not mean better reconstruction. At \(N=1080\), SFRAMES improves
both quantities on 13 of 14 slices. L-BFGS-B remained within the spectral
bounds, so this does not establish a benefit from removing them.
''' + legacy_table(summary) + figure(360,
        r'Raw loss relative to initialization and raw nonnegativity distance at $N=360$. Solid curves show means; shading is $\pm1$ SD. Work axes use recorded iterates without interpolation. Iteration costs differ between methods.') + figure(1080,
        r'The same comparison at $N=1080$. Proximal methods and L-BFGS-B remain nonnegative; SFRAMES trades loss against feasibility. Complete curves and settings are in the linked repository.')


def compile_report(report):
    report = Path(report)
    with (report/'build.log').open('w') as log:
        subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','sframes_ct_section.tex'],
                       cwd=report,stdout=log,stderr=subprocess.STDOUT,check=True)
    text = (report/'sframes_ct_section.log').read_text()
    if 'Overfull \\hbox' in text or 'undefined references' in text:
        raise ValueError('Compact report has overflow or unresolved references')
    info = subprocess.check_output(['pdfinfo',str(report/'sframes_ct_section.pdf')],text=True)
    pages = int(re.search(r'Pages:\s+(\d+)',info).group(1))
    if pages > 2:
        raise ValueError(f'Experimental section exceeds two pages: {pages}')
    return pages


def write_sources(report, body, predecessor):
    report = Path(report); report.mkdir(parents=True,exist_ok=True)
    (report/'ct_experiments.tex').write_text(body)
    (report/'sframes_ct_section.tex').write_text(driver(include_palm=r'\cite{palm}' in body))
    shutil.copyfile(Path(predecessor)/'report/ct_reproducibility_link.tex',report/'ct_reproducibility_link.tex')


def expanded_body(summary, selection):
    cells = evidence.main_cells(summary)
    sf = cells[1080, 'sframes_joint']['metrics']
    comparisons = []
    for other, label in [('frames', 'deterministic FraMES'), ('prox_sarah', 'ProxSARAH')]:
        change = 100*(sf['feasible_loss']['median_across_groups'] /
                      cells[1080, other]['metrics']['feasible_loss']['median_across_groups'] - 1)
        comparisons.append(rf'\({abs(change):.1f}\%\) ' + ('lower' if change < 0 else 'higher') + ' than ' + label)
    batch = next(c for c in summary['studies']['batch'] if c['variant']=='sframes_b64')
    noavg = next(c for c in summary['studies']['averaging_control'] if c['variant']=='sframes_noavg_paired')
    ratio = noavg['metrics']['feasible_loss']['median_across_groups']/batch['metrics']['feasible_loss']['median_across_groups']
    violation = [100*c['metrics']['relative_d_distance']['median_across_groups'] for c in (batch,noavg)]
    # Validate all required ablation settings/cohorts even though the paper only
    # highlights the matched averaging control; the full account stays online.
    evidence.ablation_results(summary)
    table = evidence.quality_table(summary)
    start, end = table.index(r'\caption{'), table.index('\n', table.index(r'\caption{'))
    table = table[:start] + r'\caption{Evaluated loss ($\times10^3$) / raw relative infeasibility (\%). Bold marks the best metric per view count. L-BFGS-B targets $D$; other methods target $C\cap D$.}' + table[end:]
    return setup(True, selection) + '\n\\paragraph{Results.}\n' + (
        'At \\(N=1080\\), joint-tuned SFRAMES achieves group-median evaluated loss '
        + ' and '.join(comparisons) + '. '
        'The endpoint comparison across view counts appears in Table~\\ref{tab:ct-results}; '
        'Figure~\\ref{fig:ct-trajectories-360} shows the loss--feasibility trajectories. '
        rf'With matched parameters and sampling, removing averaging multiplies evaluated loss by \({ratio:.2f}\), '
        rf'and changes raw relative infeasibility from \({violation[0]:.2f}\%\) to \({violation[1]:.2f}\%\). '
        'Figure~\\ref{fig:ct-trajectories-1080} separates this control from the separately tuned batch-size comparisons. '
        'Fresh-slice, domain-sensitivity and longer stationarity results are reported in the repository.\n'
    ) + table + figure(360,
        r'Method comparison at $N=1080$: raw loss relative to initialization and raw feasibility. Solid means and $\pm1$ SD use equal acquisition-group weights; work axes use recorded states. Iteration costs differ.') + figure(1080,
        r'Stochastic controls on six slices at $N=1080$. Batch sizes are tuned separately; the no-averaging control shares the $b=64$ parameters and samples. Full-batch SFRAMES retains averaging; deterministic FraMES uses current gradients and its own schedules.')


def publish_expanded(root):
    """Publish a separately validated document overlay, never mutate live runs."""
    root = Path(root).resolve()
    try:
        from . import robustness_export
    except ImportError:
        import robustness_export
    if (root/'staged_report.json').exists():
        return robustness_export.export(root)
    completion = read(root/'completion.json')
    if completion.get('status') != 'completed' or not completion.get('publication_overlay'):
        raise ValueError('Compact publication requires a completed numerical-evidence overlay')
    summary = evidence.summarize(root, 'completed')
    configs = evidence.load_configs(root)
    if summary['pending_runs'] or any(configs[f['run_id']]['phase']!='development' for f in summary['failed_runs']):
        raise ValueError('Compact publication requires all evaluation cases')
    cohort, selection, publication = [read(root/name) for name in ('cohort.json','selection.json','publication_config.json')]
    cells = evidence.main_cells(summary)
    paired = {}
    for n in (120,360,1080,2160):
        for other in ('frames','prox_sarah','joint_lbfgsb'):
            paired[f'{n}:sframes_joint:{other}'] = {
                metric:evidence.paired_groups(cells[n,'sframes_joint']['metrics'][metric]['by_slice'],
                    cells[n,other]['metrics'][metric]['by_slice'],cohort['evaluation_slice_groups'])
                for metric in ('feasible_loss','reference_relative_error')}
    summary['paired_main_comparisons'] = paired
    write(root/'report_summary.json',summary)
    predecessor = root/'predecessor_bundle'
    for name, sha in publication['predecessor_files'].items():
        if digest(predecessor/name) != sha:
            raise ValueError('Compact predecessor changed: '+name)
    main_labels = {k:evidence.LABELS[k] for k in ('sframes_joint','frames','prox_sarah','spring_sarah','joint_lbfgsb')}
    mechanism_labels = {'sframes_b16':('SFRAMES b=16','#56B4E9'),
        'sframes_b64':('SFRAMES b=64','#333333'), 'sframes_b256':('SFRAMES b=256','#0072B2'),
        'sframes_full':('SFRAMES full','#E69F00'), 'frames':('FraMES','#CC79A7'),
        'sframes_noavg_paired':('No averaging','#D55E00')}
    main, mechanism = {}, {}
    for cfg in configs.values():
        if cfg['n_train'] != 1080 or cfg['phase'] != 'evaluation':
            continue
        main_names = [m['variant'] for m in evidence.memberships(cfg) if m['study']=='main' and m['variant'] in main_labels]
        control_names = [m['variant'] for m in evidence.memberships(cfg)
            if m['study'] in ('batch','averaging_control') and m['variant'] in mechanism_labels]
        if not main_names and not control_names:
            continue
        run = evidence.read_run(root,cfg)
        for target, names in ((main,main_names),(mechanism,control_names)):
            for name in names:
                key = (cfg['slice_id'],cfg['seed'],1080,name)
                if key in target:
                    raise ValueError('Duplicate compact plotted case')
                target[key] = run
    stage = Path(tempfile.mkdtemp(prefix='report_stage_',dir=root)); report = stage/'report'
    report.mkdir()
    groups = cohort['evaluation_slice_groups']
    stats = {360:compact_plots.statistics(main,cohort['evaluation_slices'],groups,1080,main_labels),
             1080:compact_plots.statistics(mechanism,cohort['panel_slices'],groups,1080,mechanism_labels)}
    for name, stat in stats.items():
        write(report/f'statistics_{name}.json',stat)
        compact_plots.render_statistics(stat,report/f'trajectories_{name}')
    write_sources(report,expanded_body(summary,selection),predecessor)
    pages = compile_report(report)
    appendix = (predecessor/'CT_REPRODUCIBILITY.md').read_text()
    appendix += '\n\n## Completed expanded evaluation\n\nSee [complete results](ROBUSTNESS_RESULTS.md) and [the full protocol](ROBUSTNESS_EVALUATION.md). The paper section is limited to two pages; complete settings and secondary analyses remain here.\n'
    (stage/'CT_REPRODUCIBILITY.md').write_text(appendix)
    import json
    (stage/'ROBUSTNESS_RESULTS.md').write_text(evidence.full_appendix(summary,selection)
        + '\n## Paired group comparisons\n\n```json\n' + json.dumps(paired,indent=2) + '\n```\n')
    protocol_text = root/'frozen_sources/publication/ROBUSTNESS_EVALUATION.md'
    if not protocol_text.exists():
        protocol_text = root/'frozen_sources/ROBUSTNESS_EVALUATION.md'
    shutil.copyfile(protocol_text,stage/'ROBUSTNESS_EVALUATION.md')
    audit = dict(status='completed_published',generated_utc=datetime.now(timezone.utc).isoformat(),
        stage=str(stage),pages=pages,figures=2,tables=1,physical_runs=summary['completed_physical_runs'],
        evaluation_runs=sum(c['phase']=='evaluation' for c in configs.values()),
        development_runs=sum(c['phase']=='development' for c in configs.values()),
        failed_development_runs=len(summary['failed_runs']),
        reused_physical_runs=summary['reused_physical_runs'],
        statistics={f'statistics_{n}.json':digest(report/f'statistics_{n}.json') for n in stats},
        protocol_sha256=digest(root/'protocol.json'),summary_sha256=digest(root/'report_summary.json'),
        supplement_sha256={name:digest(stage/name) for name in ('CT_REPRODUCIBILITY.md','ROBUSTNESS_RESULTS.md','ROBUSTNESS_EVALUATION.md')},
        published_sha256={name:digest(report/name) for name in ASSETS},
        figure_roles={'trajectories_360':'main comparison at N1080','trajectories_1080':'mechanism controls at N1080'},
        publication_overlay=True, maximum_pages=2)
    write(root/'staged_report.json',audit)
    return robustness_export.export(root,staged=audit)
