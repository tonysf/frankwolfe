"""Read preserved campaign evidence and generate a dated, non-experimental status PDF."""
import datetime
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
TUNING = Path('/home/tony/qpt_frames_n4_tuning_20261012')
POLISH = Path('/home/tony/qpt_frames_n4_polish_20261012')
PDF = ROOT/'FRAMES_N4_STATUS_20261014.pdf'
EVIDENCE = ROOT/'status_evidence.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not PDF.exists() and not EVIDENCE.exists()
    manifests = []
    for folder in (TUNING, POLISH):
        manifest = json.loads((folder/'completion_manifest.json').read_text())
        assert manifest['status'] == 'complete'
        assert not manifest['ruche_used'] and not manifest['final_heldout_used']
        for name, sha in manifest['evidence_sha256'].items():
            assert digest(folder/name) == sha
        for name, sha in manifest['script_sha256'].items():
            assert digest(folder/name) == sha
        manifests.append(manifest)
    audit = json.loads((POLISH/'audit_calibration.json').read_text())
    assert audit['phase_complete_and_valid'] and audit['prefix_checks_verified'] == 2
    rows = sorted(audit['trials'], key=lambda r: r['channel'])
    assert len(rows) == 2 and all(r['valid'] and r['status'] == 'complete' and not r['gate_pass'] for r in rows)
    for row in rows:
        for name, key in [('started.json','started_sha256'), ('summary.json','summary_sha256'),
                          ('result.json','json_sha256'), ('result.npz','npz_sha256')]:
            assert digest(POLISH/'calibration'/row['id']/name) == row[key]
    fig = plt.figure(figsize=(8.27, 11.69), facecolor='white')
    navy = '#18334d'
    fig.text(.075,.94,'FRAMES n=4: status check', fontsize=22, weight='bold', color=navy)
    fig.text(.075,.91,'14 October 2026 | Elite local CPU | No Ruche contact', fontsize=11, color='#51677d')
    fig.text(.075,.856,'Unchanged since the completed 12 October runs.', fontsize=15, weight='bold', color=navy)
    fig.text(.075,.815,'No active local FRAMES runner matched the host process check.\n'
             'No newer files were found in the four local FRAMES campaign roots.\n'
             'The longer precision phase was proposed; it has not been launched.',
             fontsize=11, linespacing=1.6, va='top')
    fig.text(.075,.715,'Best measured calibration endpoints', fontsize=14, weight='bold', color=navy)
    ax = fig.add_axes([.075,.585,.85,.11]); ax.axis('off')
    data = [[str(r['channel']), f"{r['metrics']['rmse']:.3e}",
             f"{r['metrics']['process_fidelity']:.10f}",
             f"{r['metrics']['normalized_tp_violation']:.3e}"] for r in rows]
    table = ax.table(cellText=data, colLabels=['Channel','RMSE','Fidelity','TP residual'],
                     cellLoc='center', loc='center', bbox=[0,0,1,1], colWidths=[.15,.26,.33,.26])
    table.auto_set_font_size(False); table.set_fontsize(10)
    for (r,c), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_facecolor(navy if r == 0 else '#edf3f7')
        if r == 0: cell.set_text_props(color='white', weight='bold')
    fig.text(.075,.55,'Fidelity and TP pass. RMSE remains 1.69–1.80 times above its limit.', fontsize=11, color=navy)
    fig.text(.075,.522,'Unchanged gates: fidelity ≥ 0.99; RMSE ≤ 10⁻⁶; normalized TP ≤ 10⁻³.', fontsize=10)
    fig.text(.075,.467,'What is complete', fontsize=14, weight='bold', color=navy)
    fig.text(.075,.436,'12-setting screen across penalty, step/momentum schedules and batch size;\n'
             'two long-budget finalists; then one separately declared late-step follow-up.\n'
             '33 archived runs including 3 resource pilots; no numerical failures.\n'
             'Independent audits and exact 48k-prefix checks passed.\n'
             'Confirmation was skipped because neither calibration channel passed\n'
             'the joint gate. Prior results and final held-out channels remain preserved.',
             fontsize=10.7, linespacing=1.5, va='top')
    fig.text(.075,.268,'Next step — pending, not running', fontsize=14, weight='bold', color=navy)
    fig.text(.075,.237,'Test a longer precision phase with the current configuration fixed.\n'
             'Only after both calibration channels pass: run fresh-channel confirmation.\n'
             'This remains noiseless rank-one n=4 calibration, not a completed\n'
             'best-for-best comparison with Quiroga or a higher-qubit result.',
             fontsize=10.7, linespacing=1.5, va='top')
    fig.text(.075,.105,'Latest completed experiment: 12 October 2026, 20:45 Europe/Paris.\n'
             'Full loss plots: QPT_FRAMES_N4_POLISH_20261012.pdf.\n'
             'No experiment was launched by this status check.', fontsize=9, color='#51677d', linespacing=1.5)
    fig.savefig(PDF, format='pdf'); plt.close(fig)
    evidence = dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='read-only experiment inspection plus status artifact generation',
        host_process_check='ps -u tony filtered to qpt_frames/run_polish.py/run_tuning.py/run_batch.py: no matches',
        newer_campaign_files_since='2026-10-12T18:46:00Z: none in four existing local FRAMES roots',
        inputs={str(p):digest(p) for p in (TUNING/'completion_manifest.json', POLISH/'completion_manifest.json', POLISH/'audit_calibration.json')},
        trial_ids=[r['id'] for r in rows], new_experiments_launched=False, ruche_contacted=False,
        script_sha256=digest(__file__), pdf_sha256=digest(PDF))
    with EVIDENCE.open('x') as stream: json.dump(evidence, stream, indent=2)
    print(json.dumps({'pdf':str(PDF), 'pdf_sha256':digest(PDF)}))


if __name__ == '__main__': main()
