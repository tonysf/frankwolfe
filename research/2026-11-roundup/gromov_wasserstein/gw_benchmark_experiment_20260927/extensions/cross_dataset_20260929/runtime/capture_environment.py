"""Record actual runtime; invoke CUDA mode only inside a GPU allocation."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import torch


def command(argv):
    result = subprocess.run(argv, text=True, capture_output=True)
    return {'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr}


def capture(device='cpu', threads=1):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(__file__).resolve().parent
    gpu = None
    if device == 'cuda':
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Ruche GPU validation must run inside a Slurm allocation.')
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable in the selected environment.')
        if torch.cuda.device_count()!=1:
            raise RuntimeError('The protocol requires exactly one visible GPU per run.')
        props = torch.cuda.get_device_properties(0)
        if 'A100' not in props.name:
            raise RuntimeError(f'Expected A100, got {props.name}')
        gpu={'name':props.name,'total_memory_bytes':props.total_memory,'capability':torch.cuda.get_device_capability(0),'visible_count':torch.cuda.device_count(), 'driver':command(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'])}
        x=torch.tensor([[1.,2.],[3.,4.]],device='cuda')
        assert torch.allclose(x@x,torch.tensor([[7.,10.],[15.,22.]],device='cuda'))
        torch.cuda.synchronize()
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}
    return {'hostname':platform.node(),'platform':platform.platform(),'python':sys.version,'python_executable':sys.executable,'cpu':command(['lscpu']), 'device':device,'gpu':gpu,'torch_cuda_runtime':torch.version.cuda,'torch_version':torch.__version__,'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','torch','pytest','POT']},'threads':torch.get_num_threads(),'interop_threads':torch.get_num_interop_threads(),'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'matmul_precision':torch.get_float32_matmul_precision(),'allow_tf32':torch.backends.cuda.matmul.allow_tf32,'AMP':False,'cpu_validation_dtype':'float64','gpu_experiment_dtype':'float32','git':command(['git','-C',str(root),'rev-parse','HEAD']),'git_status':command(['git','-C',str(root),'status','--porcelain']), 'source_sha256':files,'job_id':os.environ.get('SLURM_JOB_ID'),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'thread_environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']}}

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    path=Path(args.out); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(capture(args.device,args.threads),indent=2)+'\n')
    print(path)
