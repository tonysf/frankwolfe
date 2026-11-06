import json
from pathlib import Path
import numpy as np
base=Path('/gpfs/workdir/silvetian/frankwolfe/results/qpt')
for dirname in ('fidelity99_n8_control_20260912_01','fidelity99_product_profile_20260912_02'):
 for path in sorted((base/dirname).glob('*.npz')):
  with np.load(path,allow_pickle=False) as z:
   m=json.loads(str(z['metadata_json'].item()))
   if m.get('format')=='qpt_structured_restart':
    row=dict(step=m['completed_steps'],fidelity=m['fidelity'],tp=m['tp_violation'])
   elif 'process_fidelity_proxy' in z.files:
    f=z['process_fidelity_proxy']; steps=z['checkpoint_steps']; j=int(np.argmax(f))
    row=dict(step=int(steps[-1]),fidelity=float(f[-1]),best_fidelity=float(f[j]),best_step=int(steps[j]),tp=float(z['tp_violation'][-1]),loss=float(z['measurement_loss'][-1]))
   else:
    continue
   print(json.dumps(dict(folder=dirname,file=path.name,**row),allow_nan=False))
