"""Confirm installed native POT and its callback instrumentation agree on A100."""
import argparse
import json
import os
from pathlib import Path
import warnings
import numpy as np
import ot
import torch
from capture_environment import capture
from native_sampled_gw import sampled_gromov_wasserstein


def main():
    p=argparse.ArgumentParser(); p.add_argument('--instance',required=True); p.add_argument('--out',required=True)
    args=p.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    env=capture('cuda',1)
    data=np.load(args.instance)
    arrays=[torch.tensor(data[k],device='cuda',dtype=torch.float32) for k in ('A','B','a','b')]
    base_rng=np.random.RandomState(0); wrapped_rng=np.random.RandomState(0)
    kwargs=dict(loss_fun=lambda x,y:(x-y)**2,nb_samples_grad=8,epsilon=.1,max_iter=5,log=False,verbose=False)
    callbacks=[]
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        original=ot.gromov.sampled_gromov_wasserstein(*arrays,random_state=base_rng,**kwargs)
        result,info=sampled_gromov_wasserstein(*arrays,random_state=wrapped_rng,callback=lambda T,s:callbacks.append(s),archive_dir=out/'instrumented_source',**kwargs)
    torch.cuda.synchronize()
    assert torch.isfinite(result).all()
    assert torch.equal(original,result),'Instrumentation changed the native output'
    bstate=base_rng.get_state(); wstate=wrapped_rng.get_state()
    assert bstate[0]==wstate[0] and np.array_equal(bstate[1],wstate[1]) and bstate[2:]==wstate[2:],'Instrumentation changed native RNG consumption'
    (out/'validation.json').write_text(json.dumps({'status':'passed','environment':env,'POT_version':ot.__version__,'bitwise_same_coupling':True,'same_rng_state':True,'callback_count':len(callbacks),'warnings':[str(w.message) for w in ws],'instrumentation':info},indent=2,default=str)+'\n')
    print('Native POT CUDA instrumentation: same coupling and RNG state')

if __name__=='__main__': main()
