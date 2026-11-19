import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
os.sched_setaffinity(0,{4})
import sys, json, pickle, types, traceback, hashlib
from pathlib import Path
import numpy as np

old=Path('/home/tony/frames_ct_2detect_20261104/spectral_balls/expanded/convincing_4x')
out=old.with_name('convincing_4x_recovery');out.mkdir(exist_ok=True)
sys.path.insert(0,str(old))
import methods
from ct_problem import CTProblem
from run import observe, COUNTS

folder=old/'runs/evaluation_slice01201_1080_spring_sarah_seed203_a2p3_q16'
state=pickle.loads((folder/'checkpoint.pkl').read_bytes());meta=state['metadata'];config=meta['config']
problem=CTProblem(old/'data'/f"{config['slice_id']}.npz",config['grid'],config['n_train'])
with np.load(old/'calibration'/f"{config['slice_id']}_seed201.npz") as calibration:
    initial_u,initial_v=calibration['U'],calibration['V']

def solver():
    item=methods.make_solver(config['method'],problem,initial_u,initial_v,meta['caps'],
                             seed=config['seed']+100000,config=meta['optimizer'])
    item.load_state_dict(state['solver'])
    return item

def equal(a,b):
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

report=dict(checkpoint_iteration=state['solver']['k'],checkpoint_sha256=hashlib.sha256((folder/'checkpoint.pkl').read_bytes()).hexdigest())
try:
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        solver().step()
    raise AssertionError('The original strict step unexpectedly succeeded')
except FloatingPointError as exc:
    rendered=traceback.format_exc();(out/'original_failure_traceback.txt').write_text(rendered)
    report['original_failure']=repr(exc)
    tb=exc.__traceback__
    while tb:
        if tb.tb_frame.f_code.co_name=='certify':
            values=tb.tb_frame.f_locals
            report['certificate_failure']=dict(opnorm=float(values['opnorm']),radius=float(values['radius']),
                                               input_finite=bool(np.all(np.isfinite(values['x']))),
                                               input_nonzero=int(np.count_nonzero(values['x'])))
        tb=tb.tb_next

legacy=solver()
with np.errstate(over='ignore',invalid='raise',divide='raise'):
    legacy_step=legacy.step()

text=(old/'spectral_reference_baseline.py').read_text()
before='scale = min(1.0, radius / max(opnorm, np.finfo(float).tiny))'
after='scale = radius / opnorm if opnorm > radius else 1.0'
assert text.count(before)==1
patched=text.replace(before,after)
module=types.ModuleType('ct_guard_fixed_validation');sys.modules[module.__name__]=module
exec(compile(patched,'ct_guard_fixed_validation.py','exec'),module.__dict__)
methods.project_intersection=module.project_intersection
fixed=solver()
with np.errstate(over='raise',invalid='raise',divide='raise'):
    fixed_step=fixed.step()
assert equal(legacy.state_dict(),fixed.state_dict())
assert equal(legacy_step,fixed_step)
report.update(fixed_strict_step_passed=True,new_iteration=fixed.k,
              old_permissive_and_fixed_strict_solver_state_bitwise_equal=True,
              old_permissive_and_fixed_strict_work_and_certificate_equal=True,
              all_factors_finite=bool(np.isfinite(fixed.U).all() and np.isfinite(fixed.V).all()),
              fixed_helper_sha256=hashlib.sha256(patched.encode()).hexdigest())
counters=state['counters'].copy()
for name in COUNTS:counters[name]+=fixed_step.get(name,0)
report['fixed_step_diagnostics']=observe(problem,fixed.U,fixed.V,meta['caps'],counters,fixed.k)
problem.close()
(out/'guard_replay_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
