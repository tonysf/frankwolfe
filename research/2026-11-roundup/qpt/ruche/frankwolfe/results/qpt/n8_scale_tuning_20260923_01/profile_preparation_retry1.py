"""Allocated correctness and timing gate for the host preparation candidate."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
from time import perf_counter
from types import SimpleNamespace

if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Run only in a Slurm allocation.')
ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT/'plan.json').read_text())
DEPLOY = json.loads((ROOT/'source_deployment.json').read_text())
OUT = ROOT/'preflight_retry1'
OUT.mkdir()
sys.path.insert(0, PLAN['source'])
import numpy as np
from paper.experiments.qpt_structured_data import StructuredQPTData
from paper.experiments.qpt_observation_noise import fixed_row_noise
from qpt_batch_preparation import decode_rows, prepare_on_demand_batch

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(name, value):
    (OUT/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

for source, hashes in [(PLAN['source'],DEPLOY['original_sha256']),
                       (PLAN['optimized_source'],DEPLOY['optimized_sha256'])]:
    assert {name:sha(Path(source)/name) for name in hashes} == hashes
case = PLAN['cases'][0]
assert sha(case['data']) == case['data_sha256']
data = StructuredQPTData.load_npz(case['data'])
assert data.n_qubits == 8 and data.metadata['noise_std'] == .05

# Boundary/random rows across the supported integer range, including n=13.
checks = []
for n in range(1,14):
    model = SimpleNamespace(n_qubits=n,m=24**n)
    rows = np.r_[np.array([0,1,24**n-2,24**n-1],dtype=np.int64),
                 np.random.default_rng(n).integers(0,24**n,size=2048)]
    np.testing.assert_array_equal(decode_rows(rows,n),StructuredQPTData.indices_to_symbols(model,rows))
    checks.append('decoder_n'+str(n))
for count in [1,7,65536,2*65536]:
    old_rng = np.random.default_rng(0); new_rng = np.random.default_rng(0)
    symbols = data.sample_symbols(old_rng,count)
    noise = data.noise_for_symbols(symbols)
    fast_symbols,fast_noise = prepare_on_demand_batch(data,new_rng,count)
    np.testing.assert_array_equal(symbols,fast_symbols)
    np.testing.assert_array_equal(noise,fast_noise)
    assert old_rng.bit_generator.state == new_rng.bit_generator.state
    checks.append('batch_symbols_noise_rng_'+str(count))
# Verify partitioned calls preserve the row stream and fixed observations.
r1=np.random.default_rng(17);r2=np.random.default_rng(17)
parts=[prepare_on_demand_batch(data,r1,n) for n in [11,65536,9]]
whole=prepare_on_demand_batch(data,r2,65556)
np.testing.assert_array_equal(np.concatenate([x[0] for x in parts]),whole[0])
np.testing.assert_array_equal(np.concatenate([x[1] for x in parts]),whole[1])
assert r1.bit_generator.state==r2.bit_generator.state
checks.append('partition_invariance')

def measure(fn,repeats=15):
    fn()
    samples=[]
    for _ in range(repeats):
        start=perf_counter();fn();samples.append(perf_counter()-start)
    return dict(median_seconds=statistics.median(samples),samples_seconds=samples)

B=case['batch_size']; rows=np.random.default_rng(9).integers(0,data.m,size=B)
symbols=data.indices_to_symbols(rows)
rng=np.random.default_rng(0)
def legacy():
    s=data.sample_symbols(rng,B)
    return s,data.noise_for_symbols(s)
timings={
    'rng_rows':measure(lambda:rng.integers(0,data.m,size=B)),
    'legacy_decode':measure(lambda:data.indices_to_symbols(rows)),
    'fast_decode':measure(lambda:decode_rows(rows,8)),
    'legacy_reencode_and_noise':measure(lambda:data.noise_for_symbols(symbols)),
    'direct_row_noise':measure(lambda:fixed_row_noise(rows,0,.05)),
    'legacy_preparation':measure(legacy),
    'fast_preparation':measure(lambda:prepare_on_demand_batch(data,rng,B)),
}
write('host_profile.json',dict(checks=checks,timings=timings,batch_size=B,
      note='Isolated host timings are not additive to overlapped runner timings.'))
print('HOST_PROFILE',json.dumps({k:v['median_seconds'] for k,v in timings.items()}),flush=True)

# The qpt environment has no pytest. The standalone checks below validate the
# changed path directly; no third-party test runner or installation is needed.
write('test_scope.json',dict(pytest_suite='not run: pytest unavailable in qpt environment',
      standalone_checks=checks,remaining='Three repeated original/fast factor, momentum, scalar and restart comparisons.'))

common=['-m','paper.experiments.quantum_process_tomography_structured_jax',
    '--data',case['data'],'--device','gpu','--precision','64','--measurement-backend','product-state',
    '--rank','1','--tau','20','--batch-size',str(B),'--chunk-steps','1','--metrics-every','1000',
    '--metric-mode','sampled','--metric-samples','512','--metric-batch-size','512','--metric-seed','12345',
    '--initialization-seed','0','--sampling-seed','0','--rho-scale','.6','--rho-offset','4','--rho-exponent','.5',
    '--smoothing-scale','1e7','--smoothing-offset','1','--smoothing-exponent','.25',
    '--step-scale','.5','--step-offset','10','--step-exponent','.75','--prefetch']

def run(source,name,steps=1000,resume=None):
    args=[sys.executable]+common+['--steps',str(steps),'--save',str(OUT/(name+'.npz')),
                               '--restart-path',str(OUT/(name+'_restart.npz'))]
    if resume:args+=['--resume',str(resume)]
    with (OUT/(name+'.log')).open('x') as log:
        subprocess.run(args,cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True)
    with np.load(OUT/(name+'.npz'),allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in z.files if k!='metadata_json'}
        meta=json.loads(str(z['metadata_json'].item()))
    assert meta['device_platform']=='gpu' and meta['n_steps']==steps
    assert meta['noise_std']==.05
    if not resume:
        assert meta['initial_factor_sha256']==case['initial_factor_sha256']
    else:
        assert meta['start_step']==100
    return arrays,meta

runs=[];anchor=None
for repeat in range(3):
    # Alternate order to reduce systematic warm-cache/order bias.
    for kind in (['original','fast'] if repeat%2==0 else ['fast','original']):
        source=PLAN['source'] if kind=='original' else PLAN['optimized_source']
        name=kind+'_'+str(repeat)
        arrays,meta=run(source,name)
        if anchor is None:anchor=(arrays,meta)
        for key in ['final_factor','gradient_estimate','estimated_gaps','checkpoint_steps',
                    'process_fidelity_proxy','measurement_loss','tp_violation','smoothed_gap']:
            np.testing.assert_array_equal(arrays[key],anchor[0][key],err_msg=name+':'+key)
        assert meta['batch_symbols_sha256']==anchor[1]['batch_symbols_sha256']
        runs.append(dict(name=name,kind=kind,runner_seconds=meta['wall_seconds'],
                         optimizer_seconds=meta['optimizer_seconds'],
                         host_preparation_seconds=meta['host_preparation_seconds'],
                         sampling_transfer_seconds=meta['sampling_transfer_seconds'],
                         final_factor_sha256=hashlib.sha256(arrays['final_factor'].tobytes()).hexdigest()))
        print('TIMING',json.dumps(runs[-1]),flush=True)

run(PLAN['source'],'restart_prefix',steps=100)
resumed,meta=run(PLAN['optimized_source'],'restart_fast',resume=OUT/'restart_prefix_restart.npz')
for key in ['final_factor','gradient_estimate']:
    np.testing.assert_array_equal(resumed[key],anchor[0][key])
with np.load(OUT/'restart_fast_restart.npz',allow_pickle=False) as z:
    actual_state=json.loads(str(z['metadata_json'].item()))['sampling_rng_state']
with np.load(OUT/'original_0_restart.npz',allow_pickle=False) as z:
    expected_state=json.loads(str(z['metadata_json'].item()))['sampling_rng_state']
assert actual_state==expected_state

# Separate GPU-decoding prototype: integer symbols only, no change to production.
# Keep fixed noise on NumPy to avoid changing transcendental rounding/noise values.
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
device=jax.devices('gpu')[0]
@jax.jit
def gpu_decode(r):
    inputs=r//(6**8);axes=(r//(2**8))%(3**8);outcomes=r%(2**8)
    return jnp.stack([6*((inputs//(4**q))%4)+2*((axes//(3**q))%3)
                      +((outcomes//(2**(7-q)))%2) for q in range(8)],axis=1).astype(jnp.int32)
device_rows=jax.device_put(rows,device)
np.testing.assert_array_equal(np.asarray(gpu_decode(device_rows)),symbols)
gpu_profile=measure(lambda:jax.block_until_ready(gpu_decode(device_rows)),repeats=50)
transfer_profile=measure(lambda:jax.block_until_ready(gpu_decode(jax.device_put(rows,device))),repeats=50)
medians={kind:statistics.median([r['runner_seconds'] for r in runs if r['kind']==kind]) for kind in ['original','fast']}
use_fast=medians['fast'] < .95*medians['original']
selected='fast' if use_fast else 'original'
selection=dict(status='validated',selected=selected,
    source=PLAN['optimized_source'] if use_fast else PLAN['source'],
    source_sha256=DEPLOY['optimized_sha256'] if use_fast else DEPLOY['original_sha256'],
    reason='All exact equivalence checks passed; select fast only for >5% median runner improvement.',
    runner_medians_seconds=medians,runner_speedup=medians['original']/medians['fast'],
    checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    host=socket.gethostname(),slurm_job_id=os.environ['SLURM_JOB_ID'],
    gpu_kind=device.device_kind,checks=checks+['three_repeat_full_factor_momentum_scalar_bitwise_equivalence','original_to_fast_restart_bitwise_equivalence'],
    runs=runs,host_timings=timings,gpu_decode_resident_rows=gpu_profile,gpu_decode_including_row_transfer=transfer_profile,
    gpu_decode_note='Isolated prototype, exact integer symbols; not used in production or an end-to-end speedup claim.',
    script_sha256=sha(__file__),jax_version=jax.__version__)
write('selection.json',selection)
write('checksums.json',{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file()})
print('SELECTION',selected,'runner_speedup',selection['runner_speedup'],flush=True)
