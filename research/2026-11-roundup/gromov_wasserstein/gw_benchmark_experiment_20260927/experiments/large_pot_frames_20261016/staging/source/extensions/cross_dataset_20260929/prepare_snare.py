"""Freeze paired SNARE-seq metrics; execute full preparation through Slurm.

Usage: python prepare_snare.py --raw-dir RAW --out-dir OUT
Writes preparation.json with `instances[id]` metadata. Pair labels are stored
separately from optimizer-compatible instance NPZ files.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import mmread
from scipy.spatial.distance import pdist, squareform

PREFIX = 'GSE126074_AdBrainCortex_SNAREseq_'
RECIPE = {
    'schema_version': 1,
    'paired_cells': 'exact barcode intersection in lexicographic order; reject duplicate barcodes',
    'joint_qc': {'rna_detected_genes_min': 200, 'atac_detected_peaks_min': 500},
    'representation_fit_population': 'all jointly retained paired cells, before subsampling',
    'rna': {
        'library_size': 10000.0, 'transform': 'natural log1p',
        'selected_genes': 2000, 'selection': 'descending population variance after log1p; ties by original feature index',
        'pca_components': 50, 'pca_center': True, 'pca_scale_features': False,
        'pca_solver': 'randomized', 'pca_iterated_power': 7, 'pca_n_oversamples': 10,
        'pca_seed': 20260929,
    },
    'atac': {
        'binarize': True, 'tf': 'binary count / nonzero peaks in cell',
        'idf': 'number of retained cells / number of retained cells accessing peak',
        'zero_prevalence_peaks': 'discard', 'tfidf': 'natural log1p(10000 * TF * IDF)',
        'svd_components': 50, 'svd_algorithm': 'randomized', 'svd_n_iter': 7,
        'svd_n_oversamples': 10, 'svd_seed': 20260930,
        'discard_components': [0], 'row_l2_normalize_after_discard': True,
    },
    'cell_selection_seed': 20260931,
    'source_order_seed': 20260932, 'target_order_seed': 20260933,
    'cell_selection': 'one permutation prefix per size; both views contain identical cells',
    'view_order': 'independent full-population permutations filtered to selected cells',
    'instance_sizes': {'snare3000': 3000, 'snare_full': 'min(10000, jointly retained count)'},
    'metric': 'ordinary Euclidean distance, not squared Euclidean',
    'numerical_degeneracy': 'reject ATAC retained row norm <=64*eps*max(1,full row norm); distance maximum <=64*eps*max(1,max abs coordinate)*sqrt(dim)',
    'distance_normalization': 'each modality divided by its own maximum, separately for each size',
    'cost_storage_dtype': 'float32', 'marginals': 'uniform float64',
    'objective': 'full_squared_discrepancy_no_half_no_entropy',
    'correspondence': 'separate evaluation-only NPZ; absent from optimizer instance arrays',
}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def verify_sources(raw_dir, sources):
    """Verify every frozen compressed input before loading any count matrix."""
    actual = {}
    required = {PREFIX + suffix for suffix in (
        'cDNA.barcodes.tsv.gz', 'cDNA.counts.mtx.gz', 'cDNA.genes.tsv.gz',
        'chromatin.barcodes.tsv.gz', 'chromatin.counts.mtx.gz', 'chromatin.peaks.tsv.gz')}
    records = sources.get('files', [])
    if len(records) != 6 or {item['filename'] for item in records} != required:
        raise ValueError('Source manifest must contain exactly the six adult SNARE files.')
    for item in records:
        path = Path(raw_dir) / item['filename']
        digest = sha256_file(path)
        if digest != item['sha256'] or path.stat().st_size != item['bytes']:
            raise ValueError(f'Frozen raw input changed: {path.name}')
        actual[path.name] = {'sha256': digest, 'bytes': path.stat().st_size, 'url': item['url']}
    return actual


def read_lines(path):
    with gzip.open(path, 'rt') as handle:
        lines = [line.rstrip('\r\n') for line in handle]
    if not lines or any(not line for line in lines):
        raise ValueError(f'Empty identifier in {path}')
    return np.asarray(lines, dtype=str)


def checked_counts(matrix, *, name):
    matrix = sparse.csr_matrix(matrix, dtype=np.float64)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
        raise ValueError(f'{name} counts must be finite and nonnegative.')
    if not np.equal(matrix.data, np.floor(matrix.data)).all():
        raise ValueError(f'{name} input must contain integer counts.')
    return matrix


def load_modality(raw_dir, modality, feature_suffix):
    barcodes = read_lines(Path(raw_dir) / f'{PREFIX}{modality}.barcodes.tsv.gz')
    features = read_lines(Path(raw_dir) / f'{PREFIX}{modality}.{feature_suffix}.tsv.gz')
    with gzip.open(Path(raw_dir) / f'{PREFIX}{modality}.counts.mtx.gz', 'rb') as handle:
        matrix = checked_counts(mmread(handle).T, name=modality)
    if matrix.shape != (len(barcodes), len(features)):
        raise ValueError(f'{modality} matrix dimensions do not match barcodes/features.')
    return matrix, barcodes, features


def pair_and_qc(rna, atac, rna_ids, atac_ids, *, min_genes=200, min_peaks=500):
    """Align by exact ID and apply one joint mask, preserving original indices."""
    rna, atac = checked_counts(rna, name='RNA'), checked_counts(atac, name='ATAC')
    rna_ids, atac_ids = np.asarray(rna_ids, dtype=str), np.asarray(atac_ids, dtype=str)
    if len(rna_ids) != rna.shape[0] or len(atac_ids) != atac.shape[0]:
        raise ValueError('Barcode count does not match matrix rows.')
    if len(set(rna_ids)) != len(rna_ids) or len(set(atac_ids)) != len(atac_ids):
        raise ValueError('Duplicate barcodes cannot define one-to-one correspondences.')
    if min_genes < 1 or min_peaks < 1:
        raise ValueError('Joint QC thresholds must be positive.')
    common, ir, ia = np.intersect1d(rna_ids, atac_ids, assume_unique=True, return_indices=True)
    nr, na = np.diff(rna.indptr)[ir], np.diff(atac.indptr)[ia]
    keep = (nr >= min_genes) & (na >= min_peaks)
    qc = {'rna_raw_cells': len(rna_ids), 'atac_raw_cells': len(atac_ids),
          'paired_before_qc': len(common), 'retained_paired_cells': int(keep.sum()),
          'paired_failing_rna': int((nr < min_genes).sum()),
          'paired_failing_atac': int((na < min_peaks).sum()),
          'paired_failing_either': int((~keep).sum())}
    return rna[ir[keep]], atac[ia[keep]], common[keep], ir[keep], ia[keep], qc


def representations(rna, atac, *, rna_components=50, atac_components=50,
                    top_genes=2000, rna_seed=20260929, atac_seed=20260930):
    """Fit representations independently; no cross-modality alignment or labels."""
    from sklearn.decomposition import PCA, TruncatedSVD
    rna, atac = checked_counts(rna, name='RNA'), checked_counts(atac, name='ATAC')
    if rna.shape[0] != atac.shape[0]:
        raise ValueError('Representation fitting requires aligned population sizes.')
    rna_totals = np.asarray(rna.sum(axis=1)).ravel()
    if (rna_totals <= 0).any():
        raise ValueError('RNA representation includes a zero-library cell.')
    logged = rna.multiply((10000.0 / rna_totals)[:, None]).tocsr()
    logged.data = np.log1p(logged.data)
    means = np.asarray(logged.mean(axis=0)).ravel()
    variances = np.maximum(np.asarray(logged.power(2).mean(axis=0)).ravel() - means**2, 0)
    chosen = np.lexsort((np.arange(rna.shape[1]), -variances))[:min(top_genes, rna.shape[1])]
    if rna_components > min(len(chosen), rna.shape[0] - 1):
        raise ValueError('Not enough RNA observations/features for requested PCA.')
    pca = PCA(n_components=rna_components, svd_solver='randomized', iterated_power=7,
              n_oversamples=10, random_state=rna_seed)
    X = pca.fit_transform(logged[:, chosen].toarray())

    binary = atac.copy()
    binary.data[:] = 1.0
    prevalence = np.asarray(binary.sum(axis=0)).ravel()
    peaks = np.flatnonzero(prevalence > 0)
    binary = binary[:, peaks].tocsr()
    totals = np.asarray(binary.sum(axis=1)).ravel()
    if (totals <= 0).any():
        raise ValueError('ATAC representation includes a zero-library cell.')
    if atac_components < 2 or atac_components > min(binary.shape[0] - 1, binary.shape[1] - 1):
        raise ValueError('Not enough ATAC observations/features for requested SVD.')
    tfidf = binary.multiply((1.0 / totals)[:, None]).multiply(
        (binary.shape[0] / prevalence[peaks])[None, :]).tocsr()
    tfidf.data = np.log1p(10000.0 * tfidf.data)
    svd = TruncatedSVD(n_components=atac_components, algorithm='randomized', n_iter=7,
                       n_oversamples=10, random_state=atac_seed)
    scores = svd.fit_transform(tfidf)
    Y = scores[:, 1:].copy()
    norms = np.linalg.norm(Y, axis=1)
    tiny = 64 * np.finfo(np.float64).eps * np.maximum(1.0, np.linalg.norm(scores, axis=1))
    if (norms <= tiny).any() or not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError('Degenerate or nonfinite representation.')
    Y /= norms[:, None]
    metadata = {'rna_components': X.shape[1], 'atac_components': Y.shape[1],
                'rna_selected_feature_count': len(chosen), 'atac_nonzero_peak_count': len(peaks),
                'rna_pca_explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'atac_svd_explained_variance_ratio_before_drop': svd.explained_variance_ratio_.tolist(),
                'rna_representation_sha256': array_sha256(X),
                'atac_representation_sha256': array_sha256(Y)}
    return X, Y, chosen, peaks, metadata


def paired_orders(n_population, sizes, *, selection_seed=20260931,
                  source_seed=20260932, target_seed=20260933):
    """Nested cell sets, independent view orders, exact forward/inverse maps."""
    if any(n < 2 or n > n_population for n in sizes.values()):
        raise ValueError('Requested instance size exceeds jointly retained population.')
    selection = np.random.default_rng(selection_seed).permutation(n_population)
    source_priority = np.random.default_rng(source_seed).permutation(n_population)
    target_priority = np.random.default_rng(target_seed).permutation(n_population)
    result = {}
    for name, n in sizes.items():
        selected = selection[:n]
        mask = np.zeros(n_population, dtype=bool)
        mask[selected] = True
        source = source_priority[mask[source_priority]]
        target = target_priority[mask[target_priority]]
        target_lookup = np.full(n_population, -1, dtype=np.int64)
        target_lookup[target] = np.arange(n)
        forward = target_lookup[source]
        inverse = np.argsort(forward)
        result[name] = {'selected': selected, 'source': source, 'target': target,
                        'forward': forward, 'inverse': inverse}
    return result


def normalized_distances(X):
    X = np.asarray(X, dtype=np.float64)
    condensed = pdist(X, metric='euclidean')
    scale = float(condensed.max())
    tiny = 64 * np.finfo(np.float64).eps * max(1.0, float(np.abs(X).max())) * np.sqrt(X.shape[1])
    if not np.isfinite(condensed).all() or scale <= tiny:
        raise ValueError('Degenerate or nonfinite distance matrix.')
    # Normalize in float64 then store only float32 square matrices.
    condensed /= scale
    matrix = squareform(condensed.astype(np.float32))
    return matrix, scale


def freeze_instances(out_dir, X, Y, cell_ids, rna_indices, atac_indices, sizes, binding):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f'{name}{suffix}' for name in sizes
             for suffix in ('.npz', '.json', '.correspondence.npz')]
    if any(path.exists() for path in paths):
        raise FileExistsError('Refusing to overwrite a frozen instance or correspondence.')
    cell_ids = np.asarray(cell_ids, dtype=str)
    rna_indices, atac_indices = np.asarray(rna_indices), np.asarray(atac_indices)
    X, Y = np.asarray(X), np.asarray(Y)
    if len(cell_ids) != len(X) or len(X) != len(Y) or len(X) != len(rna_indices) or len(X) != len(atac_indices):
        raise ValueError('Representations and cell provenance disagree.')
    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError('Frozen correspondences require unique cell identifiers.')
    records = {}
    for name, order in paired_orders(len(cell_ids), sizes).items():
        ir, ia = order['source'], order['target']
        n = len(ir)
        A, sa = normalized_distances(X[ir])
        B, sb = normalized_distances(Y[ia])
        filename, corrfile = f'{name}.npz', f'{name}.correspondence.npz'
        # The instance deliberately excludes forward/inverse maps and barcode IDs.
        np.savez(out_dir / filename, A=A, B=B, a=np.full(n, 1.0/n), b=np.full(n, 1.0/n),
                 source_indices=rna_indices[ir], target_indices=atac_indices[ia])
        np.savez(out_dir / corrfile, forward=order['forward'], inverse=order['inverse'],
                 source_cell_ids=cell_ids[ir], target_cell_ids=cell_ids[ia],
                 source_population_indices=ir, target_population_indices=ia,
                 selected_population_indices=order['selected'])
        record = {
            'id': name, 'n': n, 'm': n, 'filename': filename,
            'sha256': sha256_file(out_dir / filename),
            'correspondence_file': corrfile, 'correspondence_sha256': sha256_file(out_dir / corrfile),
            'metadata_file': f'{name}.json', 'family': 'snareseq_adult_mouse_cortex',
            'objective': RECIPE['objective'], 'metric': RECIPE['metric'],
            'scale': 'each_matrix_divided_by_its_own_maximum', 'A_scale': sa, 'B_scale': sb,
            'A_sha256': array_sha256(A), 'B_sha256': array_sha256(B),
            'cost_storage_dtype': 'float32', 'known_optimum': None,
            'representation_shapes': [list(X.shape), list(Y.shape)],
            'source_representation': 'log_library10000_top2000_variance_centered_pca50',
            'target_representation': 'binary_log1p_tfidf_lsi50_discard_first_row_l2',
            'binding': binding,
        }
        write_json(out_dir / f'{name}.json', record)
        records[name] = {**record, 'metadata_sha256': sha256_file(out_dir / f'{name}.json')}
        del A, B
    return records


def prepare(args):
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError('Preparation requires a new or empty output directory.')
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = json.loads(args.sources.read_text())
    verified = verify_sources(args.raw_dir, sources)
    rna, rna_ids, genes = load_modality(args.raw_dir, 'cDNA', 'genes')
    atac, atac_ids, peaks = load_modality(args.raw_dir, 'chromatin', 'peaks')
    rna, atac, cell_ids, ir, ia, qc = pair_and_qc(rna, atac, rna_ids, atac_ids)
    if len(cell_ids) < 3000:
        raise ValueError(f'Only {len(cell_ids)} jointly valid cells; no prescribed size is possible.')
    X, Y, chosen, active_peaks, representation_metadata = representations(rna, atac)
    del rna, atac
    np.savez(out_dir / 'representations.npz', rna=X, atac=Y, paired_cell_ids=cell_ids,
             source_original_indices=ir, target_original_indices=ia,
             rna_selected_feature_indices=chosen, rna_selected_feature_ids=genes[chosen],
             atac_retained_feature_indices=active_peaks, atac_retained_feature_ids=peaks[active_peaks])
    binding = {
        'recipe_sha256': hashlib.sha256(canonical_json(RECIPE)).hexdigest(),
        'source_manifest_sha256': sha256_file(args.sources),
        'preparation_script_sha256': sha256_file(__file__),
        'representations_file': 'representations.npz',
        'representations_sha256': sha256_file(out_dir / 'representations.npz'),
        'raw_files': verified,
    }
    instances = freeze_instances(out_dir, X, Y, cell_ids, ir, ia,
                                 {'snare3000': 3000, 'snare_full': min(10000, len(cell_ids))}, binding)
    manifest = {'schema_version': 1, 'status': 'completed', 'dataset': sources['dataset'],
                'recipe': RECIPE, 'binding': binding, 'qc': qc,
                'representations': representation_metadata, 'instances': instances,
                'versions': {package: importlib.metadata.version(package)
                             for package in ('numpy', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl')},
                'python': platform.python_version(), 'requested_threads': args.threads}
    write_json(out_dir / 'preparation.json', manifest)
    print(json.dumps({'status': 'completed', 'manifest': str(out_dir / 'preparation.json'),
                      'qc': qc, 'instances': {key: value['n'] for key, value in instances.items()}}, sort_keys=True))


def main():
    from threadpoolctl import threadpool_limits

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--sources', type=Path, default=Path(__file__).with_name('data_sources.json'))
    parser.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    # Import sklearn before limiting, so its OpenMP libraries are covered too.
    import sklearn.decomposition  # noqa: F401
    with threadpool_limits(limits=args.threads):
        prepare(args)


if __name__ == '__main__':
    main()
