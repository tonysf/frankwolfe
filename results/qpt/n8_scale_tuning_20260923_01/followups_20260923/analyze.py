"""Analyze saved scalar trajectories and prepare an unsubmitted continuation proposal."""
import hashlib
import json
import numpy as np
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CAMPAIGN=ROOT.parent
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(name,data):(ROOT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
data=read(CAMPAIGN/'comparison_data.json')
rows=[]
for run in data['runs']:
    case=run['case'];trace=run['new'];points={p['step']:p for p in trace}
    t=np.array([p['step'] for p in trace]);f=np.array([p['fidelity'] for p in trace])
    windows=[]
    for start,end in [(100000,120000),(120000,138000),(138000,148000)]:
        gain=100*(points[end]['fidelity']-points[start]['fidelity'])
        windows.append(dict(start=start,end=end,gain_pp=gain,gain_pp_per_10000=gain*10000/(end-start)))
    rows.append(dict(name=case['name'],step_scale=case['step_scale'],rho_scale=case['rho_scale'],
        endpoint=trace[-1],windows=windows,tail_fidelity_decreases=int(np.count_nonzero(np.diff(f[t>=100000])<0)),
        tp_reduction_fraction_100k_to_148k=1-points[148000]['tp_violation']/points[100000]['tp_violation']))

best=next(r for r in data['runs'] if r['case']['name']=='n8_step0p5_rho0p6')
t=np.array([p['step'] for p in best['new']]);f=np.array([p['fidelity'] for p in best['new']])
def fit(model,start,end,predict_step):
    keep=(t>=start)&(t<=end)
    x=np.log(t[keep]) if model=='power_infidelity' else t[keep]/1000
    y=f[keep] if model=='linear_fidelity' else np.log1p(-f[keep])
    slope,intercept=np.polyfit(x,y,1)
    xfuture=np.log(predict_step) if model=='power_infidelity' else predict_step/1000
    predicted=slope*xfuture+intercept
    if model!='linear_fidelity':predicted=1-np.exp(predicted)
    transformed_hit=(.99-intercept)/slope if model=='linear_fidelity' else (np.log(.01)-intercept)/slope
    hit=np.exp(transformed_hit) if model=='power_infidelity' else transformed_hit*1000
    return dict(model=model,start=start,end=end,predict_step=predict_step,predicted_fidelity=float(predicted),
        estimated_99_step=float(hit),slope=float(slope),intercept=float(intercept))

models=('linear_fidelity','exponential_infidelity','power_infidelity')
scenarios=[fit(model,start,148000,200000) for start in (100000,120000,128000,138000) for model in models]
backtests=[]
for start,end in [(60000,100000),(80000,120000)]:
    for model in models:
        value=fit(model,start,end,148000)
        value['actual_fidelity']=float(f[-1]);value['prediction_error_pp']=100*(value['predicted_fidelity']-f[-1])
        backtests.append(value)
analysis=dict(scope='Local scalar analysis only; no QPT execution, remote operation, or new experiment.',
    observed=rows,scenarios=scenarios,retrospective_checks=backtests,
    fit_definitions={'linear_fidelity':'OLS F(t)=a*t+b','exponential_infidelity':'OLS log(1-F(t))=a*t+b',
        'power_infidelity':'OLS log(1-F(t))=a*log(t)+b'},
    caveat='Descriptive extrapolations, not confidence intervals or convergence certificates. All assume their fitted trend continues; none identifies a nonzero asymptotic error floor.',
    input_sha256={'comparison_data.json':sha(CAMPAIGN/'comparison_data.json')})
save('late_analysis.json',analysis)

source=read(CAMPAIGN/'preflight_retry1/selection.json')['source']
case=best['case'];parent=CAMPAIGN/case['name'];prov=read(parent/'provenance.json')
remote_parent=Path(data['plan']['remote_directory'])/case['name']
remote_output='/gpfs/workdir/silvetian/frankwolfe/results/qpt/n8_continue_200k_20260923_01'
with np.load(parent/'latest_restart.npz',allow_pickle=False) as z:
    restart_metadata=json.loads(str(z['metadata_json'].item()))
assert restart_metadata['completed_steps']==148000
assert restart_metadata['fidelity']==best['new'][-1]['fidelity']
assert sha(parent/'latest_restart.npz')==best['validation']['restart_sha256']
argv=list(prov['command'])
for flag,value in [('--steps','200000'),('--restart-path',remote_output+'/latest_restart.npz'),('--save',remote_output+'/final.npz')]:
    argv[argv.index(flag)+1]=value
argv+=['--resume',str(remote_parent/'latest_restart.npz')]
assert '--fidelity-target' not in argv
proposal=dict(status='draft_not_submitted',question='Does the best requested-exponent trajectory reach 99% with more iterations alone?',
    type='exact_continuation',parent_case=case['name'],resume=str(remote_parent/'latest_restart.npz'),
    resume_archive_sha256=best['validation']['restart_sha256'],parent_initial_factor_sha256=case['initial_factor_sha256'],
    dataset=case['data'],dataset_archive_sha256=case['data_sha256'],
    dataset_internal_digest=restart_metadata['configuration']['data_sha256'],
    source=source,source_sha256=prov['source_sha256'],remote_output_directory=remote_output,
    start_step=148000,total_steps=200000,additional_steps=52000,batch_size=65536,
    extra_optimizer_row_draws=52000*65536,total_optimizer_row_draws=200000*65536,
    rank=1,tau=20,noise_std=.05,exponents={'step':.75,'smoothing':.25,'rho':.5},
    scales={'step':.5,'smoothing':1e7,'rho':.6},offsets={'step':10,'smoothing':1,'rho':4},
    caps={'step':1,'rho':1,'smoothing':None},precision='64',measurement_backend='product-state',prefetch=True,
    metric_samples=512,metric_seed=12345,metrics_every=1000,chunk_steps=1,fidelity_early_stopping=False,
    scheduler={'partition':'gpua100','nodelist':'ruche-gpu15','account':'fwllm','nodes':1,'tasks':1,'cpus':2,'gpus':1,'memory':'12G','time':'00:30:00','max_campaign_concurrent_gpus':1},
    command=argv,
    runner_minutes_point_estimate=best['metadata']['wall_seconds']/148000*52000/60,
    optimizer_minutes_point_estimate=best['metadata']['optimizer_seconds']/148000*52000/60,
    estimate_note='Linear budget scaling of one measured run; excludes queueing and separate setup/verification. Allow 12-15 minutes elapsed as a planning estimate, 30-minute allocation cap.',
    success={'primary':'Independently verified endpoint fidelity >=0.99 at total step200000.',
        'persistence':'Also report whether all ten saved checkpoints191000..200000 are >=0.99; do not stop at a transient crossing.',
        'tp':'Report endpoint TP against the parent0.1976805392 and the full trajectory; fidelity is not a TP certificate.'},
    execution_requirements=[
        'Not authorized for submission by this planning follow-up. Keep the campaign automation paused.',
        'Before any later submission, inspect user-scoped campaign scheduler state; allow at most one campaign GPU.',
        'Use a fixed new output directory created with mkdir without -p in a set -e subshell before sbatch. Retain guard directory on failure.',
        'Supply sbatch through a quoted heredoc; no Slurm file in Git; no switch/pull/commit/push.',
        'Use --export=NONE --propagate=NONE. Capture QPT_OUTPUT=$(pwd -P), cd "$SLURM_SUBMIT_DIR", then execute from the selected frozen source.',
        'Load anaconda3/2023.09-0/none-none and activate qpt within the allocation; XLA_FLAGS=--xla_gpu_autotune_level=0; preserve CUDA_VISIBLE_DEVICES.',
        'Inside the allocation verify source/dataset/restart hashes before optimization; preserve factor, momentum, sampling RNG state and global schedule index.',
        'The CLI --steps200000 denotes total steps, not 200000 additional steps. Never overwrite the parent restart.',
        'Validate the initial resumed metrics against the parent; metadata initial-factor hash now denotes the resumed factor, with U0 provenance through the parent.',
        'Run independent full-factor verification in the allocation; verify final step/schedules, scheduler exit and local archive checksums before accepting a result.',
        'If the fixed continuation misses99%, stop and report; no automatic extension or fallback sweep.'
    ])
save('proposed_experiment.json',proposal)

lines=['# Late-trajectory analysis and proposed next experiment','',
 'Status: analysis and proposal only. No new job submitted, no remote changes, no message sent.','',
 'The first choice is an unchanged continuation of step scale0.5 and momentum scale0.6 from148,000 to200,000 total steps. The requested exponents3/4,1/4,1/2 remain fixed.','',
 '| Step scale | Momentum scale | Final fidelity | Gain100k-120k (pp/10k) | Gain120k-138k (pp/10k) | Gain138k-148k (pp/10k) |',
 '|---:|---:|---:|---:|---:|---:|']
for row in rows:
    lines.append(f"| {row['step_scale']:g} | {row['rho_scale']:g} | {100*row['endpoint']['fidelity']:.4f}% | "+' | '.join(f"{w['gain_pp_per_10000']:.4f}" for w in row['windows'])+' |')
lines+=['',
 'The best run gained at every saved1,000-step checkpoint after100k. Its fidelity rose97.8494%→98.7689% over100k→148k while TP fell0.238721→0.197681 (17.2%). Its rate is slowing; a flat loss trace is not evidence that fidelity has stopped improving. Changing smoothing now would mix more-iteration effects with a new penalty schedule.',
 '',
 'The scale effects interact: at r=0.6, a=0.5 wins over a=1 by0.4055pp; at r=2, a=1 wins by4.3867pp. Lower rho means a smaller new-gradient weight and more averaging, consistent with reduced late stochastic variation, but these runs do not measure gradient noise or isolate a unique mechanism.',
 '',
 'Descriptive forecasts from100k-148k predict99% at156.5k(linear F),163.8k(exponential infidelity),170.3k(power-law infidelity). Recent-window power fits give roughly170k-175k; this is not a confidence interval. Earlier linear forecasts substantially overpredicted held-out fidelity, including impossible values above100%. Power fits were closer in two retrospective checks (errors-0.0474pp and+0.0139pp), but a nonzero fidelity-error floor remains possible. Use these scenarios to set a modest test budget, not to claim99% in advance.',
 '',
 'Proposed continuation:52,000 additional updates,3,407,872,000 additional row draws, one A100/2CPU/12G with30-minute cap. Measured-cost extrapolation is about11.4minutes runner and6.9minutes optimizer; phase times overlap, and queueing is separate. Preserve factor, momentum, RNG state and absolute schedule index. No new dataset, noise realization, initialization, batch, scale, offset or source change. Resume is explicit rather than another fresh run.',
 '',
 'Primary outcome: independently verified F>=99% at200k. Also report the first saved crossing, all ten final checkpoints, sampled loss and TP. If it misses, stop at200k and reassess; no automatic extra sweep. Source, dataset, resume hashes, exact argv, resources and validation requirements are in proposed_experiment.json.',
 '',
 'Fidelity is complete-factor overlap, separate from f (measurement least-squares) and g (TP indicator). One seed limits generalization. The unchanged continuation is the most direct and cheapest test of whether the observed shortfall is simply finite-iteration error.'
]
import re
text='\n'.join(lines)+'\n'
text=re.sub(r'(?<=[A-Za-z])(?=\d)', ' ', text)
text=re.sub(r'(?<=\d)(?=[A-Za-z])', ' ', text)
text=text.replace('k(', 'k (').replace('pp;', 'pp; ')
(ROOT/'analysis_and_plan.md').write_text(text)
(ROOT/'coauthor_summary.md').write_text("Subject: 8-qubit QPT update with the paper's exponents\n\nWe completed four fresh 8-qubit runs with the usual Gaussian noise σ=0.05, batch 65,536 and 148,000 iterations, retaining step/smoothing/momentum exponents 3/4, 1/4, 1/2. Only the step and momentum scales changed; the dataset, initialization, row sequence and smoothing schedule were fixed.\n\nThe best pair was step scale 0.5 and momentum scale 0.6: 98.7689% factor fidelity and TP residual 0.19768, compared with 90.7749% and 0.35670 for the untuned requested-exponent baseline. The other scale pairs reached 93.3581%, 98.3634% and 97.7448%. None of the new runs reached 99%; the earlier schedule reached 99.0027% at the same iteration budget. These are one-channel, one-seed comparisons.\n\nThe best trajectory was still improving at every saved late checkpoint, although gains were slowing. Our proposed next test is to resume it unchanged to 200,000 total steps, preserving momentum, RNG state and schedule index. This is 52,000 additional updates on one A100, with an estimated runner time of about 12 minutes. It has not been submitted. Extrapolation suggests a crossing is plausible, but does not rule out a plateau below 99%.\n\nWe also removed redundant CPU row decoding/re-encoding while preserving exact sampled symbols and fixed noise. Host preparation improved 4.10×; three alternating pairs of 1,000-step trials gave 1.786× median runner speedup with bitwise-equal factors, momentum and diagnostics. Production host preparation fell from about 61.5 to 16.6-16.8 minutes; it overlaps GPU work. Validation used standalone exact checks; no pytest suite was run.\n\nFidelity here is |c†u|²/(||c||²||u||²), not the measurement-loss term f, and it does not certify TP. The report includes solid fidelity/loss/TP curves, exact schedules, timing definitions and full provenance.\n")
print('Saved late_analysis.json, proposed_experiment.json, analysis_and_plan.md and coauthor_summary.md. No job submitted.')
