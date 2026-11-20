"""Read only the two existing dense BMMC representations; run in Slurm."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.spatial.distance import pdist, squareform


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(root, out, runtime, n):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Instance preparation requires a Slurm allocation.')
    sys.path.insert(0, str(runtime))
    from make_instances import write_instance
    config = json.loads((Path(__file__).parent / 'pilot.json').read_text())
    for name, expected in config['canonical_n1000_sha256'].items():
        if digest(root / 'data' / name) != expected:
            raise ValueError('Canonical input file hash mismatch: ' + name)
    original = json.loads((root / 'data/bmmc1000.json').read_text())
    representations = []
    for name, key, hash_key in [('rna', 'GEX_X_pca', 'source_file_sha256'),
                                ('atac', 'ATAC_lsi_red', 'target_file_sha256')]:
        path = root / 'raw' / (name + '.h5ad')
        if digest(path) != original[hash_key]:
            raise ValueError('Original raw-file hash mismatch: ' + name)
        with h5py.File(path, 'r') as file:
            dataset = file['obsm'][key]
            if not isinstance(dataset, h5py.Dataset):
                raise TypeError('Expected the frozen dense representation dataset.')
            representations.append(np.asarray(dataset, dtype=np.float64))
    X, Y = representations
    if n > min(len(X), len(Y)) or n < 2:
        raise ValueError('Invalid instance size.')
    if not all(z.ndim == 2 and np.isfinite(z).all() for z in (X, Y)):
        raise ValueError('Invalid representations.')
    norms = np.linalg.norm(Y, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError('ATAC zero row.')
    Y = Y / norms
    ir = np.random.default_rng(2026).permutation(len(X))[:n]
    ia = np.random.default_rng(2027).permutation(len(Y))[:n]
    A = squareform(pdist(X[ir], metric='euclidean'))
    B = squareform(pdist(Y[ia], metric='euclidean'))
    sa, sb = A.max(), B.max()
    if sa <= 0 or sb <= 0:
        raise ValueError('Degenerate distances.')
    A, B = A / sa, B / sb
    if n == 1000:
        # Independent reader must reproduce all canonical scientific arrays.
        array_hashes = {}
        with np.load(root / 'data/bmmc1000.npz', allow_pickle=False) as frozen:
            for key, value in [('A', A.astype(np.float32)), ('B', B.astype(np.float32)),
                               ('a', np.full(n, 1/n)), ('b', np.full(n, 1/n)),
                               ('source_indices', ir), ('target_indices', ia)]:
                expected = frozen[key]
                value_hash = hashlib.sha256(value.tobytes()).hexdigest()
                if (value.dtype != expected.dtype or value.shape != expected.shape or
                        value_hash != hashlib.sha256(expected.tobytes()).hexdigest()):
                    raise ValueError('Canonical n=1000 reconstruction differs: ' + key)
                array_hashes[key] = dict(sha256=value_hash, dtype=str(value.dtype), shape=list(value.shape))
        (out / 'canonical_reconstruction.json').write_text(json.dumps({
            'n': n, 'all_six_arrays_bitwise_equal': True, 'arrays': array_hashes,
            'canonical_file_sha256': digest(root / 'data/bmmc1000.npz'),
            'h5py_version': h5py.__version__, 'reader_sha256': digest(__file__),
            'source_representation_shape': list(X.shape), 'target_representation_shape': list(Y.shape)
        }, indent=2) + '\n')
        return
    metadata = {k: original[k] for k in (
        'family', 'seed', 'source_subset_seed', 'target_subset_seed',
        'source_file_sha256', 'target_file_sha256', 'source_representation',
        'target_representation', 'metric', 'scale', 'subset_rule', 'known_optimum')}
    metadata.update(A_scale=float(sa), B_scale=float(sb),
                    source_file=str(root / 'raw/rna.h5ad'),
                    target_file=str(root / 'raw/atac.h5ad'),
                    preparation_reader='h5py dense representations; canonical n1000 arrays reproduced',
                    preparation_reader_sha256=digest(__file__), h5py_version=h5py.__version__)
    write_instance(out / ('bmmc' + str(n) + '.npz'), A, B, metadata,
                   source_indices=ir, target_indices=ia)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--n', type=int, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    prepare(args.root, args.out, args.runtime, args.n)
