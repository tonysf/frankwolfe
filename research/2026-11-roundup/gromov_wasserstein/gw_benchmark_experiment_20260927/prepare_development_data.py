"""Freeze the small protocol instances; no held-out optimization or large stage D work."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent

def run_if_missing(name, args):
    path = ROOT / 'data' / name
    if path.exists():
        arrays = np.load(path)
        meta = json.loads(path.with_suffix('.json').read_text())
        for key in ('A', 'B'):
            assert hashlib.sha256(arrays[key].tobytes()).hexdigest() == meta[key+'_sha256']
        return
    subprocess.run([sys.executable, str(ROOT/'make_instances.py'), *args, '--out', str(path)], check=True)

def main():
    for n in (4, 8, 16):
        run_if_missing(f'toy{n}_s1000.npz', ['synthetic', '--n', str(n), '--seed', '1000', '--noise', '0'])
    for n in (128, 512):
        for seed in (1000,1001,1002,2000,2001,2002,2003,2004):
            for noise in (0, .1):
                suffix = '_noisy' if noise else ''
                run_if_missing(f'metric{n}{suffix}_s{seed}.npz', ['synthetic','--n',str(n),'--seed',str(seed),'--noise',str(noise)])
    run_if_missing('bmmc1000.npz', ['bmmc','--rna',str(ROOT/'raw/rna.h5ad'),'--atac',str(ROOT/'raw/atac.h5ad'),'--n','1000','--seed','2026'])
    entries=[]
    for path in sorted((ROOT/'data').glob('*.npz')):
        meta=json.loads(path.with_suffix('.json').read_text())
        arrays=np.load(path)
        for key in ('A','B'):
            assert hashlib.sha256(arrays[key].tobytes()).hexdigest()==meta[key+'_sha256']
            assert arrays[key].dtype==np.float32
            assert np.isfinite(arrays[key]).all()
        entries.append({'path':str(path.relative_to(ROOT)), 'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(), **meta})
    (ROOT/'reports/instance_inventory.json').write_text(json.dumps(entries,indent=2)+'\n')
    print(f'Validated {len(entries)} frozen instances. Held-out instances generated only; no tuning/evaluation.')

if __name__ == '__main__':
    main()
