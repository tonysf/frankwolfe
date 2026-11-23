"""Repackage frozen arrays without changing geometry, within a CPU allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def prepare_one(instance_id, spec, target):
    target = Path(target)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    source = spec['source_input']
    metadata_path = Path(source['metadata'])
    if digest(metadata_path) != source['metadata_sha256']:
        raise ValueError('Frozen source metadata hash mismatch.')
    metadata = json.loads(metadata_path.read_text())
    n = spec['n']
    if metadata.get('n') != n or metadata.get('m') != n:
        raise ValueError('Frozen source dimensions differ from campaign.')
    target.parent.mkdir(parents=True, exist_ok=True)
    if source['kind'] == 'canonical':
        directory = Path(source['directory'])
        hashes = metadata['array_files_sha256']
        if set(hashes) != {'A.npy','B.npy','a.npy','b.npy'}:
            raise ValueError('Missing canonical arrays.')
        for name, expected in hashes.items():
            if digest(directory/name) != expected:
                raise ValueError('Canonical array checksum mismatch: '+name)
        target.symlink_to(directory, target_is_directory=True)
        return {'kind':'canonical_symlink','directory':str(target),'source':source,
                'manifest_sha256':digest(target/'manifest.json')}
    if source['kind'] != 'npz':
        raise ValueError('Unknown source format.')
    path = Path(source['path'])
    file_hash = digest(path)
    expected_hash = source.get('sha256') or metadata.get('sha256')
    if expected_hash and file_hash != expected_hash:
        raise ValueError('Frozen NPZ checksum mismatch.')
    target.mkdir()
    arrays, value_hashes = {}, {}
    with np.load(path, allow_pickle=False) as archive:
        for name in ('A','B','a','b'):
            value = archive[name]
            shape = (n,n) if name.isupper() else (n,)
            if value.shape != shape or value.dtype.kind != 'f':
                raise ValueError('Invalid canonical array shape/dtype: '+name)
            if name.isupper():
                for row in range(0,n,256):
                    block = value[row:row+256]
                    if not np.isfinite(block).all() or (block<0).any() or not np.array_equal(block,value[:,row:row+256].T):
                        raise ValueError('Invalid symmetric distance matrix: '+name)
                if not np.all(value.diagonal()==0):
                    raise ValueError('Distance diagonal must remain zero.')
            elif not np.allclose(value, np.full(n,1/n), rtol=0, atol=1e-14):
                raise ValueError('Expected frozen uniform probability marginal.')
            actual = hashlib.sha256(value.tobytes()).hexdigest()
            if name.isupper() and actual != metadata.get(name+'_sha256'):
                raise ValueError('Frozen distance value hash mismatch: '+name)
            value_hashes[name] = actual
            np.save(target/(name+'.npy'), value, allow_pickle=False)
            arrays[name] = {'shape':list(value.shape),'dtype':str(value.dtype),
                            'file':name+'.npy','sha256':digest(target/(name+'.npy'))}
    converted = dict(metadata)
    converted.update(id=instance_id, storage='standalone_npy_arrays', arrays=arrays,
        array_files_sha256={name+'.npy':row['sha256'] for name,row in arrays.items()},
        repackaging={'source_npz':str(path),'source_npz_sha256':file_hash,
                    'source_metadata_sha256':digest(metadata_path),
                    'array_value_sha256':value_hashes,'values_changed':False})
    write(target/'manifest.json', converted)
    return {'kind':'npz_repackaged','directory':str(target),'source':source,
            'manifest_sha256':digest(target/'manifest.json')}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Production data repackaging requires a Slurm allocation.')
    config=json.loads(Path(args.config).read_text())
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    instances={name:prepare_one(name,spec,out/name) for name,spec in config['instances'].items()}
    write(out/'preparation.json',{'status':'completed','instances':instances,
        'config_sha256':digest(args.config),'job_id':os.environ['SLURM_JOB_ID']})


if __name__=='__main__':
    main()
