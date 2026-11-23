from concurrent.futures import ThreadPoolExecutor
import hashlib,json,subprocess
from pathlib import Path
base=Path('/home/tony/gw_benchmark_experiment_20260927/experiments/large_pot_frames_20261016/archive/shareseq_full_20261001/final_502_stochastic-kl-momentum')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()
def copy(item):
 target,info=item
 if target.exists() and target.stat().st_size==info['bytes'] and sha(target)==info['file_sha256']:return
 partial=target.with_suffix('.npy.preload')
 subprocess.run(['scp','-C','-o','BatchMode=yes','ruche:'+info['file'],str(partial)],check=True)
 assert partial.stat().st_size==info['bytes'] and sha(partial)==info['file_sha256']
 partial.replace(target)
 print('preloaded '+str(target),flush=True)
items=[]
for rank in (3,2,1,0):
 path=base/('rank'+str(rank))
 meta=json.loads((path/'metadata.json').read_text())
 for name in ('P_rounded','P_raw'):items.append((path/(name+'.npy'),meta['saved_plans'][name]))
with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(copy,items))
