"""Dated coauthor briefing from preserved evidence; no optimizer execution."""
import datetime
import hashlib
import json
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent
Q = Path('/home/tony/qpt_quiroga_tuning_20261008')
F = Path('/home/tony/qpt_frames_n4_polish_20261012')
T = Path('/home/tony/qpt_frames_n4_tuning_20261012')
G = Path('/home/tony/qpt_gell_mann_init_20261007_1750/audit_complete.json')
DOC = Path('/home/tony/frankwolfe/docs/qpt_quiroga_sensing.md')
OUT = ROOT/'QUIROGA_AND_FRAMES_COAUTHOR_UPDATE_20261014.pdf'
inputs = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p, expected=None):
    p = Path(p); actual = sha(p)
    if expected is not None: assert actual == expected, str(p)
    inputs[str(p)] = actual
    return json.loads(p.read_text())


def check_case(c):
    p = Path(c['json_path'])
    assert sha(p) == c['json_sha256'] and sha(p.with_suffix('.npz')) == c['npz_sha256']
    inputs[str(p)] = c['json_sha256']; inputs[str(p.with_suffix('.npz'))] = c['npz_sha256']


class Page:
    def __init__(self, title, subtitle, number):
        self.fig = plt.figure(figsize=(11.69,8.27), facecolor='white')
        self.text(.93,title,22,weight='bold')
        self.text(.875,subtitle,10.5,color='#526878')
        self.text(.043,'Coauthor update | 14 October 2026 | Local evidence review; no new experiment or remote contact',8,color='#526878')
        self.fig.text(.95,.043,str(number),fontsize=8,ha='right',color='#526878')
    def text(self,y,s,size=11,color='#163044',**kw):
        return self.fig.text(.06,y,s,va='top',fontsize=size,color=color,**kw)
    def para(self,y,s,size=11,width=132,color='#163044'):
        lines=textwrap.wrap(s,width,break_long_words=False,break_on_hyphens=False)
        self.text(y,'\n'.join(lines),size,color,linespacing=1.4)
        return y-len(lines)*size*1.4/(8.27*72)-.018
    def table(self,box,headers,rows,widths):
        ax=self.fig.add_axes(box);ax.axis('off')
        tab=ax.table(cellText=rows,colLabels=headers,colWidths=widths,cellLoc='left',bbox=[0,0,1,1])
        tab.auto_set_font_size(False);tab.set_fontsize(10)
        for (r,c),cell in tab.get_celld().items():
            cell.set_facecolor('#163044' if r==0 else '#edf3f7' if r%2 else '#f7f9fb')
            cell.set_edgecolor('white');cell.PAD=.055
            if r==0:cell.set_text_props(color='white',weight='bold')
    def check(self):
        self.fig.canvas.draw(); renderer=self.fig.canvas.get_renderer()
        for t in self.fig.texts:
            b=t.get_window_extent(renderer).transformed(self.fig.transFigure.inverted())
            assert b.x0>.01 and b.x1<.99 and b.y0>.01 and b.y1<.99, t.get_text()


def main():
    assert not OUT.exists() and not (ROOT/'briefing_evidence.json').exists()
    manifest=read(Q/'completion_manifest.json')
    reports={k:read(Q/v['path'],v['sha256']) for k,v in manifest['audits'].items()}
    assert all(a['audit_passed'] for a in reports.values())
    screen=reports['screen']['cases']
    printed=[c for c in screen if c['method']=='printed-adafgd']
    assert len(printed)==44 and sum(c['gate']['pass'] for c in printed)==0
    assert sum(c['status']=='complete' for c in printed)==14 and sum(c['status']=='failed' for c in printed)==30
    frozen=[c for c in screen if c['method']=='exact-linesearch-fgd' and c['trial']['alpha']==5e-5]
    frozen+=reports['confirmation']['cases']+reports['historical_revalidation']['cases']
    assert len(frozen)==16 and all(c['gate']['pass'] for c in frozen)
    for a in reports.values():
        for c in a['cases']:check_case(c)
    worst_rmse=max(c['endpoint_physical']['rmse'] for c in frozen)
    worst_tp=max(c['endpoint_physical']['normalized_tp_violation'] for c in frozen)
    old_evidence=read(Q/'coauthor_briefing_evidence.json')
    long_audit=read(G,old_evidence['inputs'][str(G)])
    long_printed=[c for c in long_audit['cases'] if c['method']=='printed-adafgd']
    assert len(long_printed)==4 and all(c['completed_steps']==100000 and not c['gate']['pass'] for c in long_printed)
    for c in long_printed:check_case(c)
    n4=[c for c in old_evidence['n4_native_archive_checks'] if c['method']=='armijo-fgd']
    assert len(n4)==4 and all(c['gate_pass'] for c in n4)
    for c in n4:
        p=Path(c['path']); assert sha(p)==c['json_sha256'] and sha(p.with_suffix('.npz'))==c['npz_sha256']
        inputs[str(p)]=c['json_sha256'];inputs[str(p.with_suffix('.npz'))]=c['npz_sha256']
    fm=read(F/'completion_manifest.json');tm=read(T/'completion_manifest.json')
    for root,m in ((F,fm),(T,tm)):
        assert m['status']=='complete' and not m['ruche_used'] and not m['final_heldout_used']
        for name,h in m['evidence_sha256'].items():assert sha(root/name)==h
    frames=read(F/'audit_calibration.json',fm['evidence_sha256']['audit_calibration.json'])['trials']
    assert len(frames)==2 and all(c['valid'] and not c['gate_pass'] for c in frames)
    for c in frames:
        for name,key in [('result.json','json_sha256'),('result.npz','npz_sha256')]:
            p=F/'calibration'/c['id']/name; assert sha(p)==c[key];inputs[str(p)]=c[key]
    inputs[str(DOC)]=sha(DOC)
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42})
    pages=[]
    p=Page('Did Quiroga’s method ever work for us?',
           'Answer: the printed adaptive rule is not reproduced; modified fixed-penalty FGD does recover accurately.',1)
    pages.append(p)
    p.para(.814,'“Never worked” is too broad. Our implementation of the printed adaFGD step has not passed the full recovery gate in the audited campaigns. That is different from saying that fixed-penalty factorized gradient descent fails, or that the paper is wrong.',12,width=123)
    p.table([.06,.38,.88,.30],['Method / protocol','Qubits','Observed result'],[
        ['Printed-rule adaFGD: latest screen','2','0/44 pass at 10k updates; 14 finite misses, 30 numerical failures'],
        ['Printed-rule adaFGD: earlier long tests','2','0/4 pass at 100k updates; fidelity 99.53–99.87%'],
        ['Our exact factor-ray line-search FGD','2','16/16 pass with one frozen fixed-penalty setting'],
        ['Earlier Armijo-backtracking FGD','4','4/4 pass; different historical initialization and tuning'],
        ['Latest independently tuned FRAMES','4','0/2 joint passes; fidelity and TP pass, RMSE narrowly misses'],
    ],[.39,.08,.53])
    p.para(.343,'The successful FGD variants retain a fixed TP penalty but replace step selection. Exact-ray FGD minimizes the objective along the factor-gradient direction; Armijo FGD uses sufficient-decrease backtracking. Neither should be labelled “published Quiroga adaFGD”.',11,width=136)
    p.para(.241,'Our joint gate requires fidelity ≥ 0.99, full-design RMSE ≤ 10⁻⁶ and normalized TP residual ≤ 10⁻³. The long printed-rule runs exceeded 99.5% fidelity but still failed both data fit and TP. Thus “no recovery at all” is also misleading.',11,width=136)
    p.para(.139,'All results above are noiseless rank-one tests. Different dimensions, budgets and protocols prevent a direct speed or superiority ranking from this table.',10.5,width=140,color='#526878')

    p=Page('Meeting update: what changed over the last week',
           'Evidence through 12 October, checked 14 October 2026. No new experiments launched for this briefing.',2)
    pages.append(p)
    p.text(.806,'8 October — a working modified fixed-penalty baseline',13,weight='bold')
    p.para(.768,f'Exact-ray FGD passed 2 calibration + 6 confirmation + 8 historical revalidation channels at n=2. Worst RMSE: {worst_rmse:.2e}; worst normalized TP: {worst_tp:.2e}; fidelity is essentially one. The eight historical channels are not an unseen-test set. Printed-rule reproduction remained unresolved.',11,width=133)
    p.text(.644,'12 October — substantial FRAMES improvement at n=4',13,weight='bold')
    p.para(.607,'Completed a 12-setting penalty/step/sampling screen, long-budget checks and a separately declared two-phase step test, all locally. Latest paired RMSE: 1.688e−6 and 1.795e−6; fidelity >0.99999999; normalized TP: 5.04e−5 and 5.37e−5. These miss only the 1e−6 RMSE gate. Confirmation was skipped; the proposed longer precision run has not been launched.',11,width=133)
    p.text(.453,'What we can safely say to the coauthors',13,weight='bold')
    p.para(.414,'“We can solve the corrected physical sensing problem accurately with fixed-penalty FGD when we modify the step selection. We have not yet reproduced the published adaptive-rule results under our full recovery criterion. FRAMES tuning has substantially improved four-qubit recovery, but the strict gate and the final best-for-best comparison remain unfinished.”',12,width=122)
    p.text(.252,'Open items and cautions',13,weight='bold')
    p.para(.216,'Obtain an author reference implementation/configuration to resolve the printed-rule discrepancy; do not interpret our negative runs as disproving the paper. Keep historical n=4 Armijo evidence separate from the current protocol. Legacy high-qubit local-Pauli results remain a separate benchmark. A fixed-setting longer FRAMES precision test is the next proposed local experiment.',10.5,width=141)
    p.para(.112,'Evidence: Oct8 audit_screen/confirmation/revalidation and coauthor briefing; Oct7 Gell-Mann audit; Oct12 FRAMES manifests and calibration audit. Archived sources and hashes are listed in the accompanying evidence JSON.',8.5,width=174,color='#526878')
    with PdfPages(OUT) as pdf:
        for p in pages:p.check();pdf.savefig(p.fig);plt.close(p.fig)
    evidence=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),inputs=inputs,
        pdf_sha256=sha(OUT),script_sha256=sha(__file__),new_experiments=False,remote_contact=False,
        printed_new_trials=len(printed),printed_new_gate_passes=0,modified_frozen_passes=len(frozen),
        qualification='Published-rule implementation and modified step-selection baselines are explicitly separated; different-qubit results are not a head-to-head comparison.')
    with (ROOT/'briefing_evidence.json').open('x') as stream:json.dump(evidence,stream,indent=2)
    print(json.dumps({'pdf':str(OUT),'sha256':sha(OUT),'pages':2}))


if __name__=='__main__':main()
