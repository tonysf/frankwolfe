"""Freeze synthetic or BMMC distance matrices; no biological evaluation.

Examples:
  python make_instances.py synthetic --n 128 --seed 2000 --noise 0 --out data/toy.npz
  python make_instances.py bmmc --rna raw/rna.h5ad --atac raw/atac.h5ad --n 1000 --out data/bmmc1000.npz
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import pdist, squareform


def write_instance(path, A, B, metadata, **extras):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_suffix('.json').exists():
        raise FileExistsError(f'Frozen instance already exists: {path}')
    A, B = np.asarray(A, dtype=np.float32), np.asarray(B, dtype=np.float32)
    n, m = len(A), len(B)
    for name, M in [('A', A), ('B', B)]:
        if not np.isfinite(M).all() or (M < 0).any() or not np.allclose(M, M.T):
            raise ValueError(f'{name} is not a finite symmetric nonnegative matrix.')
        np.fill_diagonal(M, 0)
    # Marginals are stored in float64. Cast consistently with costs at load time.
    a, b = np.full(n, 1/n), np.full(m, 1/m)
    metadata.update(n=n, m=m, objective='full_squared_discrepancy_no_half_no_entropy',
                    cost_storage_dtype='float32',
                    A_sha256=hashlib.sha256(A.tobytes()).hexdigest(),
                    B_sha256=hashlib.sha256(B.tobytes()).hexdigest())
    np.savez(path, A=A, B=B, a=a, b=b, **extras)
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    print(f'Wrote {path} with {n*m:,} coupling variables.')


def random_metric(n, rng):
    M = np.zeros((n, n), dtype=np.float32)
    i, j = np.triu_indices(n, 1)
    M[i, j] = rng.uniform(.5, 1., len(i))
    M[j, i] = M[i, j]
    return M


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='kind', required=True)
    s = sub.add_parser('synthetic')
    s.add_argument('--n', type=int, required=True)
    s.add_argument('--noise', type=float, default=0.)
    s.add_argument('--seed', type=int, default=2000)
    s.add_argument('--out', required=True)
    b = sub.add_parser('bmmc')
    b.add_argument('--rna', required=True)
    b.add_argument('--atac', required=True)
    b.add_argument('--n', type=int, default=6224)
    b.add_argument('--seed', type=int, default=2026)
    b.add_argument('--out', required=True)
    args = p.parse_args()
    if args.n < 2:
        raise ValueError('n must be at least 2.')
    rng = np.random.default_rng(args.seed)
    if args.kind == 'synthetic':
        if not 0 <= args.noise <= 1:
            raise ValueError('noise must be in [0,1].')
        A = random_metric(args.n, rng)
        perm = rng.permutation(args.n)
        B0 = A[np.ix_(perm, perm)]
        B = (1-args.noise)*B0 + args.noise*random_metric(args.n, rng)
        write_instance(args.out, A, B,
                       dict(family='random_metric', seed=args.seed, noise=args.noise,
                            scale='native_distances_in_[0.5,1]',
                            known_optimum=0. if args.noise == 0 else None),
                       true_target=np.argsort(perm))
    else:
        import anndata as ad  # optional dependency; synthetic tests do not need it
        rna, atac = ad.read_h5ad(args.rna, backed='r'), ad.read_h5ad(args.atac, backed='r')
        if 'GEX_X_pca' not in rna.obsm or 'ATAC_lsi_red' not in atac.obsm:
            raise KeyError(f'Unexpected representation keys. RNA: {list(rna.obsm)}; ATAC: {list(atac.obsm)}')
        X = np.asarray(rna.obsm['GEX_X_pca'], dtype=np.float64)
        Y = np.asarray(atac.obsm['ATAC_lsi_red'], dtype=np.float64)
        if X.ndim != 2 or Y.ndim != 2 or not np.isfinite(X).all() or not np.isfinite(Y).all():
            raise ValueError('Representations must be finite two-dimensional arrays; choose an explicit preprocessing policy.')
        if args.n > min(len(X), len(Y)):
            raise ValueError('Requested subset larger than available representations.')
        norms = np.linalg.norm(Y, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError('ATAC representation contains a zero row. Choose and document a policy.')
        Y = Y/norms
        # Independent fixed permutations, so no cell correspondence or labels
        # enter the optimizer. Prefixes give nested subsets for each view.
        ir = np.random.default_rng(args.seed).permutation(len(X))[:args.n]
        ia = np.random.default_rng(args.seed+1).permutation(len(Y))[:args.n]
        A, B = squareform(pdist(X[ir], metric='euclidean')), squareform(pdist(Y[ia], metric='euclidean'))
        sa, sb = A.max(), B.max()
        if sa <= 0 or sb <= 0:
            raise ValueError('Degenerate distance matrix.')
        A, B = A/sa, B/sb
        write_instance(args.out, A, B,
                       dict(family='moscot_bmmc_representations', seed=args.seed,
                            source_subset_seed=args.seed, target_subset_seed=args.seed+1,
                            source_file_sha256=hashlib.sha256(Path(args.rna).read_bytes()).hexdigest(),
                            target_file_sha256=hashlib.sha256(Path(args.atac).read_bytes()).hexdigest(),
                            source_representation='GEX_X_pca',
                            target_representation='ATAC_lsi_red_row_l2_normalized',
                            metric='ordinary_euclidean_NOT_squared_euclidean',
                            scale='each_matrix_divided_by_its_own_maximum',
                            A_scale=float(sa), B_scale=float(sb),
                            source_file=str(Path(args.rna).resolve()),
                            target_file=str(Path(args.atac).resolve()),
                            subset_rule='independent_permutation_prefixes_no_labels',
                            known_optimum=None),
                       source_indices=ir, target_indices=ia)
        rna.file.close(); atac.file.close()

if __name__ == '__main__':
    main()
