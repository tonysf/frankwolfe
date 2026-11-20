"""Plot native saved scalar histories; no optimizer execution."""
from pathlib import Path
import hashlib
import json
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import ScalarFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = Path('/home/tony/qpt_gell_mann_init_20261007_1750')
MOM = Path('/home/tony/qpt_higher_qubit_audit_20261004_01/archives/ruche/n8_momentum_sgd_20260924_01')
N8 = Path('/home/tony/qpt_n8_continuation_20260923/n8_continue_200k_20260923_01')
BASE = Path('/home/tony/frankwolfe/results/qpt/n8_scale_tuning_20260923_01/n8_step0p5_rho0p6/summary.json')
OUT = ROOT / 'QPT_LOSS_PLOTS_20261008.pdf'
assert not OUT.exists()
inputs = {}

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    inputs[str(p)] = sha(p)
    return json.loads(p.read_text())

def archive(p, expected):
    inputs[str(p)] = sha(p)
    assert inputs[str(p)] == expected, p
    with np.load(p, allow_pickle=False) as z:
        return {k:z[k] for k in z.files if not k.endswith('final_factor') and k not in ('gradient_estimate', 'checkpoint_factors')}

screen = read(ROOT / 'audit_screen.json')
prior = read(OLD / 'audit_complete.json')
assert screen['audit_passed'] and prior['audit_passed']
series = []
for channel in (0, 1):
    chosen = []
    for method, alpha, scale, label, color, style in [
        ('exact-linesearch-fgd', 5e-5, 1., 'Exact-ray FGD (modified), λ=0.0064', '#087e8b', '-'),
        ('printed-adafgd', 5e-5, 2., 'Printed rule c=2, λ=0.0064 (10k)', '#d9791f', '--'),
    ]:
        c = next(c for c in screen['cases'] if c['channel']==channel and c['method']==method
                 and c['trial']['alpha']==alpha and c['trial']['eta_scale']==scale)
        chosen.append((c, label, color, style))
    for method, alpha, label, color, style in [
        ('armijo-fgd', 5e-4, 'Armijo FGD (control), λ=0.064', '#3969ac', '-'),
        ('frames', None, 'FRAMES, λ starts at 0.0064 and rises', '#a63d71', '-'),
        ('printed-adafgd', 5e-4, 'Earlier printed c=1, λ=0.064 (100k)', '#936636', '-.'),
    ]:
        c = next(c for c in prior['cases'] if c['channel']==channel and c['method']==method
                 and (alpha is None or c['trial']['alpha']==alpha))
        chosen.append((c, label, color, style))
    shared = None
    for c, label, color, style in chosen:
        p = Path(c['json_path'])
        assert sha(p) == c['json_sha256']
        inputs[str(p)] = sha(p)
        z = archive(p.with_suffix('.npz'), c['npz_sha256'])
        keys = ('active_rows', 'active_observations', 'initial_factor', 'truth_factor')
        if shared is None:
            shared = [z[k] for k in keys]
        else:
            assert all(np.array_equal(z[k], v) for k,v in zip(keys,shared))
        prefix = c['method'].replace('-', '_')
        loss = z[prefix+'_measurement_loss_mean']
        assert np.all(np.isfinite(loss)) and np.all(loss>0)
        assert np.allclose(loss, .5*z[prefix+'_rmse']**2, rtol=2e-13, atol=0)
        assert np.allclose(128*loss, z[prefix+'_measurement_loss_sum'], rtol=2e-13, atol=0)
        assert np.isclose(loss[-1], c['endpoint_physical']['measurement_loss_mean'], rtol=1e-12, atol=0)
        series.append(dict(channel=channel, label=label, color=color, style=style, loss=loss,
                           rows=z[prefix+'_cumulative_data_row_accesses'], steps=z[prefix+'_checkpoint_steps']))

legacy = []
symbols = None
for method, color in [('frames','#087e8b'), ('momentum_sgd','#d9791f')]:
    for seed, style in [(1,'-'),(2,'--'),(3,':')]:
        p = MOM / f'results/{method}_seed{seed}'
        v = read(p/'verification.json')
        assert v['status']=='verified'
        z = archive(p/'final.npz', v['result_sha256'])
        m = json.loads(str(z['metadata_json'].item()))
        assert m['metric_samples']==512 and m['metric_mode']=='sampled' and m['metric_seed']==12345
        assert m['n_qubits']==8 and m['noise_std']==.05 and m['batch_size']==65536
        if symbols is None:
            symbols = z['metric_symbols']
        else:
            assert np.array_equal(symbols,z['metric_symbols'])
        assert np.isclose(z['process_fidelity_proxy'][-1],v['fidelity_numpy'],rtol=1e-10)
        legacy.append(dict(label=f"{'FRAMES' if method=='frames' else 'Momentum-SGD'} s{seed}",
                           steps=z['checkpoint_steps'], loss=z['measurement_loss'], fidelity=z['process_fidelity_proxy'],
                           color=color,style=style))
v = read(N8/'verification.json')
z = archive(N8/'final.npz',v['result_sha256'])
b = read(BASE)
assert b['metadata']['metric_seed']==12345 and np.array_equal(symbols,z['metric_symbols'])
pre = b['trajectory']
assert pre[-1]['step']==int(z['checkpoint_steps'][0])==148000
assert pre[-1]['measurement_loss']==float(z['measurement_loss'][0])
assert np.isclose(pre[-1]['fidelity'],z['process_fidelity_proxy'][0],rtol=1e-13)
legacy.insert(0,dict(label='FRAMES s0 (continued)', color='#172f44', style='-',
    steps=np.r_[[c['step'] for c in pre],z['checkpoint_steps'][1:]],
    loss=np.r_[[c['measurement_loss'] for c in pre],z['measurement_loss'][1:]],
    fidelity=np.r_[[c['fidelity'] for c in pre],z['process_fidelity_proxy'][1:]]))

plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'font.size':11,'axes.titlesize':14,
                     'axes.labelsize':12,'axes.spines.top':False,'axes.spines.right':False})

def page(title, subtitle, number):
    f=plt.figure(figsize=(13.333,7.5),facecolor='white')
    f.text(.07,.95,'QPT · SAVED LOSS HISTORIES',fontsize=10,color='#087e8b',weight='bold')
    f.text(.07,.891,title,fontsize=23,color='#163044',weight='bold')
    f.text(.07,.846,subtitle,fontsize=11.5,color='#526878')
    axes=[f.add_axes([.085,.31,.395,.40]), f.add_axes([.57,.31,.395,.40])]
    for ax in axes:ax.grid(True,which='major',alpha=.2)
    f.text(.07,.034,'8 October 2026 · native saved checkpoints · no new optimization or smoothing of curves',fontsize=9,color='#526878')
    f.text(.955,.034,str(number),fontsize=9,color='#526878')
    return f,axes

def notes(f, lines):
    y=.215
    for line in lines:
        wrapped=textwrap.wrap(line, width=155,break_long_words=False)
        f.text(.07,y,'\n'.join(wrapped),va='top',fontsize=9.5,color='#526878',linespacing=1.25)
        y-=len(wrapped)*9.5*1.25/540+.010
    assert y>.05

with PdfPages(OUT,metadata={'Title':'QPT loss plots: current and legacy benchmarks','Author':'QPT project'}) as pdf:
    f,axes=page('Current Quiroga-style setup: measurement loss',
                 'Two qubits · noiseless · 128 rows · shared target, data and initialization within each channel',1)
    for channel,ax in enumerate(axes):
        for c in [c for c in series if c['channel']==channel]:
            mask=c['rows']>0
            ax.loglog(c['rows'][mask],c['loss'][mask],label=c['label'],color=c['color'],linestyle=c['style'],lw=2)
            ax.plot(c['rows'][-1],c['loss'][-1],'o',ms=4,color=c['color'])
        ax.axhline(5e-13,ls=':',lw=1.4,color='#656565')
        ax.set(title=f'Channel {channel}', xlabel='Charged optimizer row accesses',
               ylabel='Mean measurement loss = RMSE² / 2',ylim=(1e-22,2e-2),xlim=(1e4,2e7))
    handles,labels=axes[0].get_legend_handles_labels()
    f.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.81),ncol=3,fontsize=9.4,frameon=False)
    notes(f,[
        'Dotted line: measurement-loss threshold 5×10⁻¹³, equivalent to RMSE ≤10⁻⁶. Passing this line alone does not establish the joint fidelity/TP gate. Initial zero-work checkpoints are omitted on the logarithmic x axis.',
        'Loss excludes TP penalties. Penalties and update horizons differ; these are paired diagnostics, not a final best-for-best comparison. Exact-ray and Armijo are modified/control methods, not the printed adaFGD rule.',
        'Charged rows include line-search evaluations but exclude spectral-norm, TP and checkpoint work. This is not a wall-time comparison.'
    ])
    pdf.savefig(f);f.savefig(ROOT/'QPT_LOSS_CURRENT_20261008.png',dpi=150);plt.close(f)

    f,axes=page('Earlier local-Pauli setup: eight-qubit loss',
                 'Gaussian observation noise σ=0.05 · one channel / initialization · sampling streams s0–s3',2)
    for ax in axes:
        box=ax.get_position()
        ax.set_position([box.x0,box.y0,box.width,.36])
    for c in legacy:
        axes[0].plot(c['steps']/1000,c['loss'],label=c['label'],color=c['color'],ls=c['style'],lw=1.6)
        axes[1].plot(c['steps']/1000,100*c['fidelity'],color=c['color'],ls=c['style'],lw=1.6)
    axes[0].set(title='Fixed 512-row measurement sample',xlabel='Completed updates (thousands)',
                ylabel='Sampled mean measurement loss',xlim=(0,200))
    axes[0].ticklabel_format(axis='y',style='sci',scilimits=(0,0),useOffset=False)
    axes[1].set(title='Full-factor fidelity: context for the loss',xlabel='Completed updates (thousands)',
                ylabel='Normalized factor fidelity (%)',xlim=(0,200),ylim=(-2,102))
    axes[1].axhline(99,color='#656565',ls=':',lw=1)
    handles,labels=axes[0].get_legend_handles_labels()
    f.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.813),ncol=4,fontsize=9.8,frameon=False)
    notes(f,[
        'All loss curves use the same fixed 512-row diagnostic sample, not the full virtual dataset. The y axis is zoomed and does not start at zero. Noise-dominated loss values can be close while recovered channels differ greatly.',
        'FRAMES ends at 98.92–99.22% fidelity with normalized TP ≈0.0107. Momentum-SGD stays near 0.000873% fidelity in these tested settings. These are sampling replicates, not independent target channels.',
        'The s0 curve joins saved history at 148,000 updates to its verified continuation. Same batch size 65,536 throughout; no new runs. This noisy local-Pauli benchmark is separate from the first page.'
    ])
    pdf.savefig(f);f.savefig(ROOT/'QPT_LOSS_LEGACY_20261008.png',dpi=150);plt.close(f)

(ROOT/'loss_plot_evidence.json').write_text(json.dumps({'inputs_sha256':inputs,'pdf_sha256':sha(OUT),
    'builder_sha256':sha(Path(__file__)),'current_endpoints':[{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in c.items() if k not in ('color','style')} for c in series],
    'notes':'Native scalar histories; no optimization; legacy loss is sampled, current loss full-design.'},indent=2)+'\n')
print(OUT)
