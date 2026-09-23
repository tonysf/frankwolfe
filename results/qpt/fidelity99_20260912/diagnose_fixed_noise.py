"""Fixed-factor gradient diagnostics only; no optimization or dataset changes."""
import argparse, hashlib, json, os, socket, sys
from pathlib import Path
from time import perf_counter
parser=argparse.ArgumentParser()
parser.add_argument("--source",type=Path,required=True)
parser.add_argument("--data",type=Path,required=True)
parser.add_argument("--fixed-result",type=Path,required=True)
parser.add_argument("--relative-result",type=Path,required=True)
parser.add_argument("--save",type=Path,required=True)
parser.add_argument("--batches",type=int,default=128)
parser.add_argument("--batch-size",type=int,default=65536)
args=parser.parse_args()
if args.save.exists(): raise FileExistsError(args.save)
sys.path.insert(0,str(args.source))
import numpy as np
from paper.experiments.qpt_structured_data import StructuredQPTData
from paper.experiments.quantum_process_tomography import make_factor_initial_point,unpack_factor
from paper.experiments.quantum_process_tomography_jax import _require_jax,_configure_jax_precision,select_jax_device
from paper.experiments.qpt_structured_operators import product_state_measurement_bank,product_state_measurement_values,product_state_measurement_loss_and_gradient,trace_preserving_loss_and_gradient
jax=_require_jax()
_configure_jax_precision(jax,"64")
import jax.numpy as jnp
device=select_jax_device("gpu")
data=StructuredQPTData.load_npz(args.data)
assert data.n_qubits==10 and data.metadata["noise_std"]==.05
def put(x): return jax.device_put(x,device)
truth=put(data.truth_factor)
bank=tuple(put(x) for x in product_state_measurement_bank(data.local_measurements,data.local_basis))
basis=put(data.local_basis)
@jax.jit
def evaluate(u,s,eps):
    target=product_state_measurement_values(truth,s,bank,xp=jnp)
    clean_loss,clean=product_state_measurement_loss_and_gradient(u,s,target,bank,xp=jnp)
    noisy_loss,noisy=product_state_measurement_loss_and_gradient(u,s,target+eps,bank,xp=jnp)
    return clean,noisy-clean,jnp.stack((clean_loss,noisy_loss,jnp.mean(target),jnp.mean(target**2),jnp.mean(eps**2)))
@jax.jit
def tp_eval(u): return trace_preserving_loss_and_gradient(u,basis,xp=jnp)
@jax.jit
def summaries(g,noise,u,df):
    un=jnp.vdot(u,u).real
    tangent=lambda x:x-u*(jnp.vdot(u,x)/un)
    clean_t=tangent(g);noise_t=tangent(noise)
    norm=lambda x:jnp.linalg.norm(x)
    return jnp.stack((norm(g),norm(noise),norm(clean_t),norm(noise_t),
                     jnp.vdot(df,-g).real,jnp.vdot(df,-noise).real,
                     jnp.vdot(clean_t,noise_t).real))
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(2**20),b""):h.update(b)
    return h.hexdigest()
cases=[("initialization",unpack_factor(make_factor_initial_point(data,1,0),data.process_dimension,1),None,0)]
for name,path in (("fixed_final",args.fixed_result),("relative_final",args.relative_result)):
    with np.load(path,allow_pickle=False) as z:
        cases.append((name,z["final_factor"].copy(),z["gradient_estimate"].copy(),int(z["checkpoint_steps"][-1])))
report=dict(status="complete",purpose="fixed-factor diagnostic, truth only used for evaluation; no optimization",
    noise_std=.05,batch_size=args.batch_size,batches=args.batches,
    diagnostic_sampling_seed=24680,
    reference_note="All measurement-gradient means are finite Monte Carlo estimates, not exact full gradients.",
    host=socket.gethostname(),device_kind=device.device_kind,job_id=os.environ.get("SLURM_JOB_ID"),
    source=str(args.source),source_sha256={str(p.relative_to(args.source)):digest(p) for p in
      [args.source/"paper/experiments/qpt_structured_operators.py",args.source/"paper/experiments/quantum_process_tomography_structured_jax.py"]},
    data_sha256=digest(args.data),script_sha256=digest(Path(__file__)),cases=[])
for name,array,momentum,step in cases:
    began=perf_counter();u=put(array);c2=jnp.vdot(truth,truth).real;u2=jnp.vdot(u,u).real
    fidelity=jnp.abs(jnp.vdot(truth,u))**2/(c2*u2)
    df=2*(truth*(jnp.vdot(truth,u)/c2)-fidelity*u)/u2
    tp_loss,tp_gradient=tp_eval(u)
    rng=np.random.default_rng(24680)
    sum_g=jnp.zeros_like(u);sum_noise=jnp.zeros_like(u)
    half_g=[];half_noise=[];stats=[];scalar=[];averages=[]
    for k in range(args.batches):
        symbols=data.sample_symbols(rng,args.batch_size)
        eps=data.noise_for_symbols(symbols)
        g,gn,sc=evaluate(u,put(symbols),put(eps))
        sum_g=sum_g+g;sum_noise=sum_noise+gn
        st=summaries(g,gn,u,df)
        jax.block_until_ready((sum_g,sum_noise,st,sc))
        stats.append(np.asarray(st));scalar.append(np.asarray(sc))
        if k+1==args.batches//2: half_g=sum_g.copy();half_noise=sum_noise.copy()
        if k+1 in (1,4,16,64,args.batches):
            avg=np.asarray(summaries(sum_g/(k+1),sum_noise/(k+1),u,df))
            averages.append(dict(rows=(k+1)*args.batch_size,gradient_summary=avg.tolist()))
    mean_g=sum_g/args.batches;mean_noise=sum_noise/args.batches
    stats=np.asarray(stats);scalar=np.asarray(scalar)
    mean_summary=np.asarray(summaries(mean_g,mean_noise,u,df))
    half_count=args.batches//2
    h1g=half_g/half_count;h2g=(sum_g-half_g)/(args.batches-half_count)
    h1n=half_noise/half_count;h2n=(sum_noise-half_noise)/(args.batches-half_count)
    norm=lambda x:float(jnp.linalg.norm(x))
    cosine=lambda a,b:float(jnp.vdot(a,b).real/(jnp.linalg.norm(a)*jnp.linalg.norm(b)))
    tp_norm=norm(tp_gradient)
    item=dict(name=name,step=step,fidelity=float(fidelity),factor_norm=float(jnp.sqrt(u2)),
      tp_violation=float(jnp.sqrt(2*tp_loss)),raw_tp_gradient_norm=tp_norm,
      single_batch_median_summary=np.median(stats,axis=0).tolist(),
      mean_gradient_summary=mean_summary.tolist(),mean_gradients_by_rows=averages,
      mean_clean_noisy_losses_target_mean_target_rms_noise_rms=[float(scalar[:,0].mean()),float(scalar[:,1].mean()),
        float(scalar[:,2].mean()),float(np.sqrt(scalar[:,3].mean())),float(np.sqrt(scalar[:,4].mean()))],
      split_half_clean_cosine=cosine(h1g,h2g),split_half_noisy_cosine=cosine(h1g+h1n,h2g+h2n),
      clean_reference_half_difference_norm=norm(h1g-h2g),
      penalized_tp_gradient_norms={str(b):tp_norm/(b/(step+1)**.25) for b in (1e7,1e8,1e9,1e10)},
      elapsed_seconds=perf_counter()-began)
    if momentum is not None:
        mom=put(momentum)
        item["momentum_norm"]=norm(mom)
        item["momentum_clean_cosine"]=cosine(mom,mean_g)
        item["momentum_noisy_mean_cosine"]=cosine(mom,mean_g+mean_noise)
        item["saved_momentum_fidelity_directional_derivative"]=float(jnp.vdot(df,-mom).real)
    report["cases"].append(item)
    print(json.dumps(item,allow_nan=False),flush=True)
report["gradient_summary_columns"]=["clean_norm","noise_norm","clean_tangent_norm","noise_tangent_norm","fidelity_derivative_minus_clean","fidelity_derivative_minus_noise","clean_noise_tangent_inner_product"]
with args.save.open("x") as f:json.dump(report,f,indent=2,allow_nan=False)
print("Saved "+str(args.save),flush=True)

