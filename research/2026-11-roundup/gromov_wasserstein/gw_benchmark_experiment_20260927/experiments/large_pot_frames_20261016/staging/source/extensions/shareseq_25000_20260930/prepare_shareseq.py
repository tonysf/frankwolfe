"""Freeze SHARE-seq mouse-skin metrics as memory-mapped arrays.

Full preparation must execute through Slurm. This self-contained script reuses
(the copied) SNARE representation/QC/order formulas from prepare_snare.py;
H5AD input and blockwise metric output are the only dataset/storage changes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist

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
    'instance_sizes': {'shareseq25000': 25000},
    'upstream_population': 'scGLUE Ma-2020 raw-count H5ADs: 32231 paired cells, after author Mix-cell exclusion and upstream feature filters',
    'cost_storage': 'uncompressed standalone NPY arrays; float64 tiled distances normalized before float32 storage',
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


def verify_sources(raw_dir, sources):
    """Verify exact upstream bytes before parsing any count arrays."""
    records = sources.get('files', [])
    required = {'Ma-2020-RNA.h5ad', 'Ma-2020-ATAC.h5ad'}
    if len(records) != 2 or {item['filename'] for item in records} != required:
        raise ValueError('Source manifest must contain exactly the two Ma-2020 H5ADs.')
    verified = {}
    for item in records:
        path = Path(raw_dir) / item['filename']
        if path.stat().st_size != item['bytes'] or sha256_file(path) != item['sha256']:
            raise ValueError(f'Frozen raw input changed: {path.name}')
        verified[path.name] = {key: item[key] for key in ('url', 'bytes', 'sha256')}
    return verified


def _decoded(value):
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def _h5_index(group):
    key = _decoded(group.attrs.get('_index', '_index'))
    if key not in group or not isinstance(group[key], h5py.Dataset):
        raise ValueError(f'Missing simple string index for {group.name}.')
    item = group[key]
    if item.ndim != 1 or h5py.check_string_dtype(item.dtype) is None:
        raise ValueError(f'{group.name} index must be a one-dimensional string array.')
    values = np.asarray(item.asstr()[:], dtype=str)
    if len(values) == 0 or any(not value for value in values):
        raise ValueError(f'Empty identifier in {group.name}.')
    if len(set(values)) != len(values):
        raise ValueError(f'Duplicate identifiers in {group.name}.')
    return values


def load_h5ad_counts(path):
    """Read only raw sparse X and observation/feature IDs, without anndata.

    The frozen files use CSR encoding 0.1.0. CSR/CSC versions 0.1.0 and 0.2.0
    share the explicit data/indices/indptr/shape contract. Other encodings fail
    closed: do not guess a layer, rescale values or densify X.
    """
    with h5py.File(path, 'r') as handle:
        if any(key not in handle for key in ('X', 'obs', 'var')):
            raise ValueError('H5AD requires X, obs and var.')
        cells, features = _h5_index(handle['obs']), _h5_index(handle['var'])
        X = handle['X']
        if not isinstance(X, h5py.Group):
            raise ValueError('Only explicitly encoded sparse count X is supported.')
        encoding = _decoded(X.attrs.get('encoding-type', ''))
        version = _decoded(X.attrs.get('encoding-version', ''))
        if encoding not in ('csr_matrix', 'csc_matrix') or version not in ('0.1.0', '0.2.0'):
            raise ValueError('Unsupported sparse H5AD encoding.')
        shape_values = np.asarray(X.attrs.get('shape', []))
        if shape_values.shape != (2,) or not np.issubdtype(shape_values.dtype, np.integer):
            raise ValueError('Sparse shape must have two integer dimensions.')
        shape = tuple(int(i) for i in shape_values)
        if shape != (len(cells), len(features)):
            raise ValueError('Count shape disagrees with cell/feature identifiers.')
        if any(key not in X or not isinstance(X[key], h5py.Dataset)
               or X[key].ndim != 1 for key in ('data', 'indices', 'indptr')):
            raise ValueError('Sparse X requires one-dimensional data, indices and indptr.')
        data, indices, indptr = (X[key][:] for key in ('data', 'indices', 'indptr'))
        if not np.issubdtype(data.dtype, np.number) or np.iscomplexobj(data):
            raise ValueError('Count data must be real numbers.')
        if not np.issubdtype(indices.dtype, np.integer) or not np.issubdtype(indptr.dtype, np.integer):
            raise ValueError('Sparse indices must be integers.')
        major, minor = shape if encoding == 'csr_matrix' else shape[::-1]
        if (len(indices) != len(data) or len(indptr) != major + 1 or indptr[0] != 0
                or indptr[-1] != len(data) or (np.diff(indptr) < 0).any()
                or (indices < 0).any() or (indices >= minor).any()):
            raise ValueError('Invalid sparse array structure.')
        # Validate before merging duplicate coordinates. Chunking avoids a
        # full-sized temporary validity mask for the large ATAC count array.
        for start in range(0, len(data), 1_000_000):
            values = data[start:start + 1_000_000]
            if (not np.isfinite(values).all() or (values < 0).any()
                    or not np.equal(values, np.floor(values)).all()):
                raise ValueError('X must contain finite, nonnegative integer raw counts.')
        constructor = sparse.csr_matrix if encoding == 'csr_matrix' else sparse.csc_matrix
        counts = checked_counts(constructor((data, indices, indptr), shape=shape), name='X')
    return counts, cells, features


def write_normalized_distances(path, coordinates, *, block_rows=512):
    """Two tiled passes preserve float64 normalization before float32 storage."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Refusing to overwrite {path}.')
    X = np.asarray(coordinates, dtype=np.float64)
    if X.ndim != 2 or min(X.shape) < 1 or len(X) < 2 or not np.isfinite(X).all():
        raise ValueError('Invalid metric coordinates.')
    if block_rows < 1:
        raise ValueError('Distance block size must be positive.')
    n, scale = len(X), 0.0
    for i in range(0, n, block_rows):
        for j in range(i, n, block_rows):
            distances = cdist(X[i:i + block_rows], X[j:j + block_rows], metric='euclidean')
            if not np.isfinite(distances).all():
                raise ValueError('Nonfinite distances.')
            scale = max(scale, float(distances.max()))
    tiny = 64 * np.finfo(np.float64).eps * max(1.0, float(np.abs(X).max())) * np.sqrt(X.shape[1])
    if scale <= tiny:
        raise ValueError('Degenerate distance matrix.')
    matrix = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(n, n))
    for i in range(0, n, block_rows):
        for j in range(i, n, block_rows):
            distances = cdist(X[i:i + block_rows], X[j:j + block_rows], metric='euclidean')
            distances /= scale
            block = distances.astype(np.float32)
            matrix[i:i + block_rows, j:j + block_rows] = block
            if i != j:
                matrix[j:j + block_rows, i:i + block_rows] = block.T
    np.fill_diagonal(matrix, 0)
    matrix.flush()
    del matrix
    return scale


def array_file_record(path):
    matrix = np.load(path, mmap_mode='r', allow_pickle=False)
    record = {'file': Path(path).name, 'sha256': sha256_file(path),
              'bytes': Path(path).stat().st_size, 'dtype': str(matrix.dtype),
              'shape': list(matrix.shape)}
    del matrix
    return record


def freeze_instances(out_dir, X, Y, cell_ids, rna_indices, atac_indices, sizes, binding,
                     *, block_rows=512):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if any((out_dir / name).exists() for name in sizes):
        raise FileExistsError('Refusing to overwrite a frozen instance directory.')
    cell_ids = np.asarray(cell_ids, dtype=str)
    X, Y = np.asarray(X), np.asarray(Y)
    rna_indices, atac_indices = np.asarray(rna_indices), np.asarray(atac_indices)
    if (len(X) != len(Y) or len(X) != len(cell_ids) or len(X) != len(rna_indices)
            or len(X) != len(atac_indices) or len(set(cell_ids)) != len(cell_ids)):
        raise ValueError('Representations and unique cell provenance disagree.')
    records = {}
    for name, order in paired_orders(len(cell_ids), sizes).items():
        directory = out_dir / name
        directory.mkdir()
        ir, ia = order['source'], order['target']
        n = len(ir)
        sa = write_normalized_distances(directory / 'A.npy', X[ir], block_rows=block_rows)
        sb = write_normalized_distances(directory / 'B.npy', Y[ia], block_rows=block_rows)
        np.save(directory / 'a.npy', np.full(n, 1.0 / n, dtype=np.float64), allow_pickle=False)
        np.save(directory / 'b.npy', np.full(n, 1.0 / n, dtype=np.float64), allow_pickle=False)
        np.savez(directory / 'correspondence.npz', forward=order['forward'], inverse=order['inverse'],
                 source_cell_ids=cell_ids[ir], target_cell_ids=cell_ids[ia],
                 source_original_indices=rna_indices[ir], target_original_indices=atac_indices[ia],
                 source_population_indices=ir, target_population_indices=ia,
                 selected_population_indices=order['selected'])
        arrays = {key: array_file_record(directory / f'{key}.npy') for key in ('A', 'B', 'a', 'b')}
        record = {
            'schema_version': 1, 'id': name, 'n': n, 'm': n,
            'storage': 'standalone_npy_arrays', 'arrays': arrays,
            'array_files_sha256': {item['file']: item['sha256'] for item in arrays.values()},
            'correspondence_file': 'correspondence.npz',
            'correspondence_sha256': sha256_file(directory / 'correspondence.npz'),
            'family': 'shareseq_mouse_skin', 'objective': RECIPE['objective'],
            'metric': RECIPE['metric'], 'A_scale': sa, 'B_scale': sb,
            'scale': 'each_matrix_divided_by_its_own_maximum', 'cost_storage_dtype': 'float32',
            'known_optimum': None, 'representation_shapes': [list(X.shape), list(Y.shape)],
            'source_representation': 'log_library10000_top2000_variance_centered_pca50',
            'target_representation': 'binary_log1p_tfidf_lsi50_discard_first_row_l2', 'binding': binding,
        }
        write_json(directory / 'manifest.json', record)
        records[name] = {**record, 'directory': name, 'metadata_file': f'{name}/manifest.json',
                         'metadata_sha256': sha256_file(directory / 'manifest.json')}
    return records


def prepare(args):
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError('Preparation requires a new or empty output directory.')
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = json.loads(args.sources.read_text())
    verified = verify_sources(args.raw_dir, sources)
    rna, rna_ids, genes = load_h5ad_counts(args.raw_dir / 'Ma-2020-RNA.h5ad')
    atac, atac_ids, peaks = load_h5ad_counts(args.raw_dir / 'Ma-2020-ATAC.h5ad')
    rna, atac, cell_ids, ir, ia, qc = pair_and_qc(rna, atac, rna_ids, atac_ids)
    if qc['paired_before_qc'] != sources['expected_shared_cells_before_our_qc']:
        raise ValueError('Paired population disagrees with the frozen source manifest.')
    if len(cell_ids) < 25000:
        raise ValueError(f'Only {len(cell_ids)} jointly valid cells; require 25000.')
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
        'representations_sha256': sha256_file(out_dir / 'representations.npz'), 'raw_files': verified,
    }
    instances = freeze_instances(out_dir, X, Y, cell_ids, ir, ia, {'shareseq25000': 25000},
                                 binding, block_rows=args.distance_block_rows)
    manifest = {'schema_version': 1, 'status': 'completed', 'dataset': sources['dataset'],
                'recipe': RECIPE, 'binding': binding, 'qc': qc,
                'representations': representation_metadata, 'instances': instances,
                'versions': {package: importlib.metadata.version(package)
                             for package in ('numpy', 'scipy', 'h5py', 'scikit-learn', 'joblib', 'threadpoolctl')},
                'python': platform.python_version(), 'requested_threads': args.threads,
                'distance_block_rows': args.distance_block_rows}
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
    parser.add_argument('--distance-block-rows', type=int, default=512)
    args = parser.parse_args()
    if min(args.threads, args.distance_block_rows) < 1:
        parser.error('Thread count and distance block size must be positive.')
    import sklearn.decomposition  # noqa: F401; load OpenMP libraries before limiting
    with threadpool_limits(limits=args.threads):
        prepare(args)


if __name__ == '__main__':
    main()
