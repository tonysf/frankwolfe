"""Small scientific and provenance checks; never process real counts locally."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from scipy import sparse
from scipy.spatial.distance import pdist, squareform

import prepare_shareseq as prep


def write_fixture(path, values, ids, features, *, encoding='csr_matrix'):
    ctor = sparse.csr_matrix if encoding == 'csr_matrix' else sparse.csc_matrix
    matrix = ctor(values)
    with h5py.File(path, 'w') as handle:
        X = handle.create_group('X')
        X.attrs['encoding-type'] = encoding
        X.attrs['encoding-version'] = '0.1.0'
        X.attrs['shape'] = matrix.shape
        for key in ('data', 'indices', 'indptr'):
            X.create_dataset(key, data=getattr(matrix, key))
        for key, names in [('obs', ids), ('var', features)]:
            group = handle.create_group(key)
            group.attrs['_index'] = 'names'
            group.create_dataset('names', data=np.asarray(names, dtype=object),
                                 dtype=h5py.string_dtype('utf-8'))


class H5InputChecks(unittest.TestCase):
    def test_csr_and_csc_preserve_raw_counts_and_identifiers(self):
        counts = np.asarray([[1, 0, 3], [0, 7, 0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            for encoding in ('csr_matrix', 'csc_matrix'):
                path = Path(tmp) / f'{encoding}.h5ad'
                write_fixture(path, counts, ['β', 'a'], ['g0', 'g1', 'g2'], encoding=encoding)
                matrix, ids, features = prep.load_h5ad_counts(path)
                np.testing.assert_array_equal(matrix.toarray(), counts)
                self.assertEqual(ids.tolist(), ['β', 'a'])
                self.assertEqual(features.tolist(), ['g0', 'g1', 'g2'])
                self.assertTrue(sparse.isspmatrix_csr(matrix))

    def test_normalized_counts_bad_structure_and_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.h5ad'
            for value in (.1, -1.0, np.nan):
                write_fixture(path, [[value, 1], [2, 3]], ['a', 'b'], ['g0', 'g1'])
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'raw counts'):
                    prep.load_h5ad_counts(path)
            write_fixture(path, [[1, 0], [0, 1]], ['a', 'a'], ['g0', 'g1'])
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                prep.load_h5ad_counts(path)
            write_fixture(path, [[1, 0], [0, 1]], ['a', 'b'], ['g0', 'g1'])
            with h5py.File(path, 'r+') as handle:
                handle['X/indices'][0] = 2
            with self.assertRaisesRegex(ValueError, 'structure'):
                prep.load_h5ad_counts(path)
            write_fixture(path, [[1, 0], [0, 1]], ['a', 'b'], ['g0', 'g1'])
            with h5py.File(path, 'r+') as handle:
                handle['X'].attrs['encoding-version'] = 'unsupported'
            with self.assertRaisesRegex(ValueError, 'Unsupported'):
                prep.load_h5ad_counts(path)

    def test_alignment_and_joint_qc_uses_ids_not_row_positions(self):
        rna = [[1, 2, 0], [1, 1, 1], [0, 0, 1], [1, 1, 1]]
        atac = [[0, 0, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]
        R, T, ids, ir, ia, qc = prep.pair_and_qc(
            rna, atac, ['c', 'a', 'd', 'b'], ['b', 'd', 'c', 'a'], min_genes=2, min_peaks=2)
        self.assertEqual(ids.tolist(), ['a', 'c'])
        np.testing.assert_array_equal(ir, [1, 0])
        np.testing.assert_array_equal(ia, [3, 2])
        np.testing.assert_array_equal(R.toarray(), np.asarray(rna)[ir])
        np.testing.assert_array_equal(T.toarray(), np.asarray(atac)[ia])
        self.assertEqual(qc['paired_failing_either'], 2)

    def test_frozen_raw_hash_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = []
            for mode in ('RNA', 'ATAC'):
                path = Path(tmp) / f'Ma-2020-{mode}.h5ad'
                path.write_bytes(b'raw-count fixture')
                records.append({'filename': path.name, 'sha256': prep.sha256_file(path),
                                'bytes': path.stat().st_size, 'url': 'https://example.invalid/'})
            self.assertEqual(len(prep.verify_sources(tmp, {'files': records})), 2)
            path.write_bytes(b'raw-count fixturE')
            with self.assertRaisesRegex(ValueError, 'Frozen raw input changed'):
                prep.verify_sources(tmp, {'files': records})


class MetricAndRepresentationChecks(unittest.TestCase):
    def test_tiled_metric_matches_dense_float64_normalization(self):
        X = np.random.default_rng(101).normal(size=(17, 7))
        dense = squareform(pdist(X))
        expected = (dense / dense.max()).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            for block in (1, 5, 50):
                path = Path(tmp) / f'cost-{block}.npy'
                scale = prep.write_normalized_distances(path, X, block_rows=block)
                matrix = np.load(path, mmap_mode='r')
                np.testing.assert_array_equal(matrix, expected)
                self.assertEqual(scale, dense.max())
                self.assertEqual(matrix.dtype, np.float32)
                self.assertIsInstance(matrix, np.memmap)
            with self.assertRaisesRegex(ValueError, 'Degenerate'):
                prep.write_normalized_distances(Path(tmp) / 'degenerate.npy', np.ones((3, 2)))
            with self.assertRaises(FileExistsError):
                prep.write_normalized_distances(path, X)

    def test_representation_reproducibility_and_atac_binarization(self):
        rng = np.random.default_rng(31)
        R, T = rng.poisson(3, (18, 11)), rng.poisson(.7, (18, 14))
        T[:, 0], T[:, -1] = 1, 0
        kwargs = {'rna_components': 3, 'atac_components': 4, 'top_genes': 7}
        first = prep.representations(R, T, **kwargs)
        second = prep.representations(R, T, **kwargs)
        for a, b in zip(first[:4], second[:4]):
            np.testing.assert_array_equal(a, b)
        X, Y, _, peaks, _ = first
        np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(Y, axis=1), 1, atol=1e-12)
        self.assertNotIn(13, peaks)
        np.testing.assert_array_equal(Y, prep.representations(R, np.where(T > 0, 7, 0), **kwargs)[1])

    def test_frozen_arrays_have_bound_hashes_and_separate_pair_labels(self):
        X = np.random.default_rng(86).normal(size=(13, 5))
        ids = np.asarray([f'cell-{i:03d}' for i in range(13)])
        with tempfile.TemporaryDirectory() as tmp:
            records = prep.freeze_instances(tmp, X, 3 * X, ids, np.arange(13) + 100,
                                            np.arange(13) + 200, {'small': 6, 'large': 11},
                                            {'recipe_sha256': 'fixture'}, block_rows=4)
            small_ids = None
            for name, record in records.items():
                directory = Path(tmp) / name
                manifest = json.loads((directory / 'manifest.json').read_text())
                self.assertEqual(record['metadata_sha256'], prep.sha256_file(directory / 'manifest.json'))
                self.assertEqual(set(manifest['array_files_sha256']), {'A.npy', 'B.npy', 'a.npy', 'b.npy'})
                for filename, digest in manifest['array_files_sha256'].items():
                    self.assertEqual(digest, prep.sha256_file(directory / filename))
                A, B = np.load(directory / 'A.npy'), np.load(directory / 'B.npy')
                with np.load(directory / 'correspondence.npz') as corr:
                    f = corr['forward']
                    np.testing.assert_array_equal(corr['source_cell_ids'], corr['target_cell_ids'][f])
                    np.testing.assert_array_equal(corr['inverse'][f], np.arange(len(f)))
                    np.testing.assert_allclose(A, B[np.ix_(f, f)], atol=1e-7)
                    if small_ids is None:
                        small_ids = set(corr['source_cell_ids'])
                    else:
                        self.assertTrue(small_ids.issubset(corr['source_cell_ids']))
                self.assertEqual(np.load(directory / 'a.npy').dtype, np.float64)
                self.assertEqual(np.load(directory / 'A.npy').dtype, np.float32)
            with self.assertRaises(FileExistsError):
                prep.freeze_instances(tmp, X, X, ids, np.arange(13), np.arange(13), {'small': 6}, {})


if __name__ == '__main__':
    unittest.main()
