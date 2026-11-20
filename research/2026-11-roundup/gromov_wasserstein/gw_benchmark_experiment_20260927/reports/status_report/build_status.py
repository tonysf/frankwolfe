"""Build status PDFs: shared GW and oracle pages, then one dataset's experiment.

The two reference pages are separate, stable LaTeX inputs. This builder regenerates
only completed probe-study experiment pages and charts from the verified report.
New experiments should reuse gw_problem.tex, gw_stochastic_oracle.tex and report_style.tex unchanged.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REPORT = ROOT/'reports/batch_size_20261008/report.json'
REPORT_SHA = '6ce0b0198875649d7a524abceef5d8c5e4d904c7711bd8d22949c28534808640'
PROBE_REPORT = ROOT/'reports/probe8192_20261009/report.json'
PROBE_REPORT_SHA = '969f9f3b225c832a493c29cea509603a2fea59bd40aca00eded8c0b6c87e1952'
STYLES = {128: ('#0072B2', 'o'), 1024: ('#D78300', 's'), 2048: ('#009E73', '^'),
          8192: ('#9B359B', 'o')}
DETAILS = {
    'metric512_s1000': dict(stem='synthetic_probe_count_status', title='Synthetic data: increasing the probe count',
        verified='11 October 2026',
        data=r'''\textbf{Data.} Two $512\times512$ distance matrices, giving 262,144 coupling
variables. The source is a symmetric random metric with off-diagonal distances
uniform in $[0.5,1)$ and zero diagonal; the target is an exact permutation of it
(seed 1000, no noise or rescaling). Uniform masses are used. The planted matching
has \textbf{GW loss zero}, but the optimizer is not given that matching.''',
        finding=r'''\textbf{Assessment.} All three 8,192-probe runs first show at least 99\% loss
reduction at 40,000--42,000 updates, versus a 50,000--100,000 checkpoint bracket
with 2,048. At 200,000 updates, the median final loss is 41.5\% lower, while solver
time rises from 12.6 to 17.1 minutes. The higher probe count brings the loss drop
forward; 128 probes still makes little progress.''',
        followup=r'''\textbf{Scope.} The 8,192-probe runs record loss every 1,000 updates, with
additional early checkpoints. Smoothing, prefactors, seeds, start and horizon are
unchanged; GPU trajectory-parity checks passed for the added diagnostics. This
supports a probe-count effect on this clean instance, without establishing that
initial smoothing caused the delay or predicting performance on real data.''',
        provenance=r'''Frozen instance \texttt{metric512\_s1000}; run sources \texttt{8d11e2a}
(128--2,048 probes) and \texttt{49911ec} (8,192). Validated report SHA256 prefixes:
\texttt{6ce0b0198875649d7} and \texttt{969f9f3b225c832a}.'''),
    'bmmc6224': dict(stem='bmmc_probe_count_status', title='BMMC data: increasing the probe count',
        verified='9 October 2026',
        data=r'''\textbf{Data.} Bone-marrow mononuclear cells (BMMC): all 6,224 cells in each of
the downloaded processed moscot RNA and ATAC files, with no additional cell
subsampling (38,738,176 coupling variables). RNA uses 50 provided PCA coordinates;
ATAC uses 38 provided LSI coordinates, normalized rowwise to unit Euclidean norm.
Within-view Euclidean distances are divided by each matrix's own maximum.
The views are independently reordered; masses are uniform and labels are unused.''',
        finding=r'''\textbf{Assessment.} Increasing the probe count improves the final feasible
loss at 20,000 updates, but the decrease remains modest and solver cost rises.
The 2,048-probe median reduction is 1.471\%, versus 0.268\% with 128 probes.
These are three random seeds on one fixed processed dataset, not independent
biological replicates. The result does not establish that gradient noise is the
only remaining limitation; biological matching accuracy was not evaluated.''',
        followup=r'''\textbf{Scope.} This comparison changes only the probe count within the stated
configuration. It is not a comparison against another optimization method, nor a
claim that these prefactors are the best ones for BMMC. The processed files used
here are not the entire larger original BMMC collection.''',
        provenance=r'''Frozen instance \texttt{bmmc6224}; run source \texttt{8d11e2a};
completed probe-study report SHA256 prefix \texttt{6ce0b0198875649d7}.'''),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def combined_report():
    assert digest(REPORT)==REPORT_SHA, 'Original report differs from verified snapshot.'
    assert digest(PROBE_REPORT)==PROBE_REPORT_SHA, '8,192-probe report differs from verified snapshot.'
    report=json.loads(REPORT.read_text())
    extra=json.loads(PROBE_REPORT.read_text())
    assert report['status']=='completed' and report['task_counts']['completed']==18
    assert extra['status']=='completed' and extra['task_counts']['completed']==extra['expected_runs']==3
    assert extra['baseline']['status']=='validated' and extra['baseline']['report_sha256']==REPORT_SHA
    assert extra['development_seeds']==report['development_seeds']==[710,711,712]
    assert extra['initialization']==report['initialization']=='product'
    dataset='metric512_s1000'
    for key in ('n','world_size','array_files_sha256','instance_manifest_sha256','prefactors','final_iterations'):
        assert extra['instances'][dataset][key]==report['instances'][dataset][key], key
    for run in extra['runs']:
        assert run['status']=='completed' and run['dataset']==dataset and run['batch']==8192
        assert run['completed_updates']==200000 and run['sketch_probes']==8192*200000
    # Retain original scientific summaries; append only the new probe group.
    for key in ('runs','final_summaries','trajectory_summaries'):
        report[key].extend(extra[key])
    return report


def plot(report, dataset, stem):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.52))
    fig.subplots_adjust(left=.095, right=.985, bottom=.24, top=.80, wspace=.34)
    for ax, is_time in zip(axes, (False, True)):
        for q, (color, marker) in STYLES.items():
            points = sorted((r for r in report['trajectory_summaries']
                if r['dataset'] == dataset and r['batch'] == q), key=lambda r:r['iteration'])
            if not points: continue
            assert all(r['status'] == 'completed' and r['n'] == 3 for r in points)
            x = [r['statistics']['actual_solver_seconds']['median']/60 if is_time else r['iteration'] for r in points]
            stats = [r['statistics']['rounded_loss'] for r in points]
            ax.plot(x, [s['median'] for s in stats], color=color, marker=marker,
                    markersize=1.5 if q==8192 else 3, linewidth=1.25, label=f'{q:,} probes')
            ax.fill_between(x, [s['q25'] for s in stats], [s['q75'] for s in stats], color=color, alpha=.16, linewidth=0)
        ax.set_xlabel('Solver wall-clock time (minutes)' if is_time else 'Completed updates', fontsize=9)
        ax.set_ylabel('Feasible GW loss', fontsize=9)
        ax.grid(alpha=.18)
        ax.ticklabel_format(axis='y', useOffset=False, style='plain')
        ax.xaxis.set_major_locator(MaxNLocator(4))
        if not is_time:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:g}k' if x else '0'))
        if dataset == 'metric512_s1000': ax.set_ylim(-.001, .046)
        else: ax.set_ylim(.15845,.16125)
    handles, labels=axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(labels),
               frameon=False, bbox_to_anchor=(.55,1.02), fontsize=9 if len(labels)>3 else 10)
    for suffix in ('pdf', 'png'):
        fig.savefig(OUT/(stem+'_curves.'+suffix), dpi=180, facecolor='white')
    plt.close(fig)


def page(report, dataset, settings):
    instance = report['instances'][dataset]
    rows = []
    summaries = sorted((r for r in report['final_summaries'] if r['dataset']==dataset), key=lambda r:r['batch'])
    for row in summaries:
        assert row['status']=='completed' and row['completed_seeds']==[710,711,712]
        s=row['statistics']
        rows.append(f"{row['batch']:,} & {s['last_rounded_loss']['median']:.8f} & "
            f"{s['loss_reduction_percent']['median']:.4f}\\% & {s['solver_seconds']['median']/60:.2f} \\\\")
    f0=summaries[0]['statistics']['initial_rounded_loss']['median']
    probe_labels=[f"{r['batch']:,}" for r in summaries]
    probes=', '.join(probe_labels[:-1])+' or '+probe_labels[-1]
    count=3*len(summaries)
    pref=instance['prefactors']; horizon=instance['final_iterations']
    text=r'''{\color{ink}\fontsize{18}{21}\selectfont\bfseries TITLE\par}
{\small\color{muted}Experiment results \enspace / \enspace verified DATE \hfill COUNT of COUNT runs completed\par}
\vspace{4pt}
DATA

\textbf{Design.} Stochastic FraMES with PROBES random-sign probes per
gradient; three development seeds (710--712), HORIZON updates each, product start
$P_0=ab^\top$, and one Ruche A100 per run. One gradient estimate is formed per
update, with $d_0=\widehat G_0$. All other settings are fixed across probe counts:
\[
 \gamma_k=(k+1)^{-3/4},\qquad
 \rho_k=0.3(k+1)^{-1/2}\ (k\geq1),\qquad
 \mu_k=\frac{MU}{N}(k+1)^{-1/4}.
\]
\vspace{-5pt}
\begin{center}
\small\renewcommand{\arraystretch}{1.16}\setlength{\tabcolsep}{11pt}
\begin{tabular}{@{}rrrr@{}}
\toprule
Probes & Final feasible loss $\downarrow$ & Decrease from $P_0$ & Solver minutes\\
\midrule
TABLE
\bottomrule
\end{tabular}
\end{center}
\vspace{-5pt}
{\small Medians across three seeds; initial feasible loss $f(P_0)=INITIAL$.
Percentage decreases are computed per seed before taking the median.}

\includegraphics[width=\linewidth]{FIGURE}
\vspace{-5pt}
{\footnotesize\color{muted}Markers are saved checkpoints; lines are visual guides. Shading shows the three-seed IQR.
The time plot places each common-iteration median loss at its median solver time; it does not interpolate to a common time grid.}

FINDING

FOLLOWUP

\vfill
{\footnotesize\color{muted}\textbf{Measurement.} Final-iterate loss after transport repair, evaluated in FP64;
initial and intermediate losses use FP32. This is neither a best-checkpoint loss
nor a theorem-selected output. Solver time includes initialization and updates;
it excludes preparation, repairs, loss diagnostics, logging, snapshots and final
FP64 validation. It is not end-to-end elapsed time.\\[3pt]
\textbf{Provenance.} PROVENANCE}
'''
    replacements={'TITLE':settings['title'],'DATA':settings['data'],'HORIZON':f'{horizon:,}',
        'MU':f"{pref['mu_scale']:g}",'N':str(instance['n']),'TABLE':'\n'.join(rows),
        'INITIAL':f'{f0:.8f}','FIGURE':settings['stem']+'_curves.pdf','FINDING':settings['finding'],
        'FOLLOWUP':settings['followup'],'PROVENANCE':settings['provenance'],
        'DATE':settings['verified'],'COUNT':str(count),'PROBES':probes}
    # Replace complete tokens in one pass, so token names never alter prose.
    import re
    return re.sub(r'\b(?:'+ '|'.join(sorted(replacements,key=len,reverse=True))+r')\b',
                  lambda m:replacements[m.group()],text)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--latex',type=Path,default=Path('/tmp/gw_latex_tools/tectonic'))
    args=p.parse_args()
    report=combined_report()
    shared_before={name:digest(OUT/name) for name in ('gw_problem.tex','gw_stochastic_oracle.tex')}
    outputs=[]
    for dataset,settings in DETAILS.items():
        stem=settings['stem']; plot(report,dataset,stem)
        (OUT/(stem+'_experiment.tex')).write_text(page(report,dataset,settings))
        document='\\input{report_style.tex}\n'+r'\hypersetup{pdftitle={GW status: '+settings['title']+'}}\n'+r'\begin{document}'+'\n'+r'\input{gw_problem.tex}'+'\n'+r'\clearpage'+'\n'+r'\input{gw_stochastic_oracle.tex}'+'\n'+r'\clearpage'+'\n'+r'\input{'+stem+'_experiment.tex}'+'\n'+r'\end{document}'+'\n'
        (OUT/(stem+'.tex')).write_text(document)
        result=subprocess.run([str(args.latex),'--only-cached','--keep-logs',stem+'.tex'],cwd=OUT,capture_output=True,text=True)
        (OUT/(stem+'_build.txt')).write_text(result.stdout+result.stderr)
        if result.returncode: raise RuntimeError(result.stdout+result.stderr)
        pdf=OUT/(stem+'.pdf')
        info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
        assert re_page_count(info)==3, info
        log=(OUT/(stem+'.log')).read_text()
        assert 'Overfull \\vbox' not in log and 'Overfull \\hbox' not in log, 'Inspect LaTeX overflow: '+stem
        outputs.append(dict(dataset=dataset,pdf=str(pdf.relative_to(ROOT)),pdf_sha256=digest(pdf),pages=3))
    assert all(digest(OUT/name)==value for name,value in shared_before.items())
    receipt=dict(layout='Shared GW problem page, shared stochastic oracle page, then one data/settings/results page per experiment.',
        reference_pages_sha256={'reports/status_report/'+name:value for name,value in shared_before.items()},
        source_reports=[dict(file=str(path.relative_to(ROOT)),sha256=sha)
            for path,sha in ((REPORT,REPORT_SHA),(PROBE_REPORT,PROBE_REPORT_SHA))],
        builder_sha256=digest(__file__),outputs=outputs,
        status_scope='Synthetic: 12 completed runs, including three q8192 runs verified 11 October 2026. BMMC: unchanged nine-run snapshot verified 9 October. No remote status query during this document update.')
    (OUT/'build_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


def re_page_count(info):
    import re
    return int(re.search(r'^Pages:\s+(\d+)',info,re.M).group(1))


if __name__=='__main__': main()
