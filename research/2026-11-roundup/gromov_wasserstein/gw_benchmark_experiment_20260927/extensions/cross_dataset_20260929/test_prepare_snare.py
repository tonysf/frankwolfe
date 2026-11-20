"""Small scientific/provenance checks; no full dataset processing."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import sparse

import prepare_snare as snare


class PairingChecks(unittest.TestCase):
    def test_joint_filter_aligns_shuffled_barcodes_and_tracks_original_rows(self):
        rna_ids = ['c', 'a', 'd', 'b', 'rna-only']
        atac_ids = ['b', 'd', 'c', 'a', 'atac-only']
        rna = [[4, 2, 0], [1, 1, 1], [0, 0, 2], [2, 2, 2], [1, 1, 1]]
        atac = [[0, 0, 4], [3, 3, 3], [9, 8, 7], [6, 5, 4], [1, 1, 1]]
        R, T, ids, ir, ia, qc = snare.pair_and_qc(rna, atac, rna_ids, atac_ids,
                                                 min_genes=2, min_peaks=2)
        self.assertEqual(ids.tolist(), ['a', 'c'])
        np.testing.assert_array_equal(R.toarray(), np.asarray(rna)[[1, 0]])
        np.testing.assert_array_equal(T.toarray(), np.asarray(atac)[[3, 2]])
        np.testing.assert_array_equal(ir, [1, 0])
        np.testing.assert_array_equal(ia, [3, 2])
        self.assertEqual(qc['paired_before_qc'], 4)
        self.assertEqual(qc['paired_failing_either'], 2)

    def test_ambiguous_or_invalid_inputs_fail(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            snare.pair_and_qc([[1], [1]], [[1], [1]], ['a', 'a'], ['a', 'b'])
        for value in (-1, np.nan, .5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                snare.checked_counts([[value]], name='bad')

    def test_nesting_independent_orders_and_correspondence(self):
        sizes = {'small': 9, 'large': 17}
        first = snare.paired_orders(23, sizes)
        second = snare.paired_orders(23, sizes)
        self.assertTrue(set(first['small']['selected']).issubset(first['large']['selected']))
        for name, order in first.items():
            np.testing.assert_array_equal(order['target'][order['forward']], order['source'])
            np.testing.assert_array_equal(order['source'][order['inverse']], order['target'])
            self.assertFalse(np.array_equal(order['source'], order['target']))
            for key in order:
                np.testing.assert_array_equal(order[key], second[name][key])
        chosen = set(first['small']['selected'])
        for side in ('source', 'target'):
            expected = [i for i in first['large'][side] if i in chosen]
            np.testing.assert_array_equal(first['small'][side], expected)


class RepresentationChecks(unittest.TestCase):
    def test_reproducibility_normalization_and_unused_peaks(self):
        rng = np.random.default_rng(92)
        rna = sparse.csr_matrix(rng.poisson(3, (18, 11)))
        atac = rng.poisson(.6, (18, 14))
        atac[:, 0] = 1
        atac[:, -1] = 0
        kwargs = {'rna_components': 3, 'atac_components': 4, 'top_genes': 7}
        first = snare.representations(rna, atac, **kwargs)
        second = snare.representations(rna, atac, **kwargs)
        X, Y, _, peaks, metadata = first
        for a, b in zip(first[:4], second[:4]):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(metadata, second[-1])
        np.testing.assert_allclose(X.mean(axis=0), 0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(Y, axis=1), 1, atol=1e-12)
        self.assertEqual(Y.shape, (18, 3))
        self.assertNotIn(13, peaks)
        # Counts above one carry no additional ATAC information in this recipe.
        changed = np.where(atac > 0, 7, 0)
        Y_binary = snare.representations(rna, changed, **kwargs)[1]
        np.testing.assert_array_equal(Y, Y_binary)

    def test_gene_variance_ties_have_fixed_feature_index_order(self):
        # All columns identical: exact equal log variances, including nonzero variance.
        rna = np.ones((12, 6), dtype=int)
        atac = np.random.default_rng(31).integers(0, 3, size=(12, 8))
        atac[:, 0] = 1
        # A constant RNA matrix produces a zero PCA representation, which is
        # valid here; a fully degenerate distance matrix is rejected at freezing.
        with np.errstate(invalid='ignore'):
            result = snare.representations(rna, atac, rna_components=2, atac_components=3, top_genes=4)
        np.testing.assert_array_equal(result[2], [0, 1, 2, 3])
        with self.assertRaisesRegex(ValueError, 'Degenerate'):
            snare.normalized_distances(result[0])


class FrozenArtifactChecks(unittest.TestCase):
    def test_saved_correspondence_recovers_same_metric_and_remains_separate(self):
        X = np.random.default_rng(86).normal(size=(13, 5))
        ids = np.asarray([f'cell-{i:03d}' for i in range(13)])
        binding = {'recipe_sha256': 'test-recipe', 'raw_files': {'rna': 'test-hash'}}
        with tempfile.TemporaryDirectory() as directory:
            manifest = snare.freeze_instances(directory, X, 3 * X, ids,
                                              np.arange(13) + 100, np.arange(13) + 200,
                                              {'small': 6, 'large': 11}, binding)
            for record in manifest.values():
                path = Path(directory) / record['filename']
                self.assertEqual(snare.sha256_file(path), record['sha256'])
                self.assertEqual(record['binding'], binding)
                with np.load(path) as instance, np.load(Path(directory) / record['correspondence_file']) as corr:
                    self.assertNotIn('forward', instance.files)
                    self.assertNotIn('source_cell_ids', instance.files)
                    target = corr['forward']
                    np.testing.assert_array_equal(corr['source_cell_ids'], corr['target_cell_ids'][target])
                    np.testing.assert_allclose(instance['A'], instance['B'][np.ix_(target, target)], atol=1e-7)
                    np.testing.assert_allclose(instance['a'].sum(), 1)
                    self.assertEqual(instance['A'].dtype, np.float32)
                    self.assertEqual(instance['a'].dtype, np.float64)
                metadata = json.loads((Path(directory) / record['metadata_file']).read_text())
                self.assertEqual(metadata['sha256'], record['sha256'])
                self.assertEqual(metadata['correspondence_sha256'], record['correspondence_sha256'])
            with self.assertRaises(FileExistsError):
                snare.freeze_instances(directory, X, X, ids, np.arange(13), np.arange(13),
                                       {'small': 6}, binding)

    def test_raw_hash_mismatch_aborts_before_parsing(self):
        suffixes = ['cDNA.barcodes.tsv.gz', 'cDNA.counts.mtx.gz', 'cDNA.genes.tsv.gz',
                    'chromatin.barcodes.tsv.gz', 'chromatin.counts.mtx.gz', 'chromatin.peaks.tsv.gz']
        with tempfile.TemporaryDirectory() as directory:
            files = []
            for suffix in suffixes:
                path = Path(directory) / (snare.PREFIX + suffix)
                path.write_bytes(b'small immutable fixture')
                files.append({'filename': path.name, 'sha256': snare.sha256_file(path),
                              'bytes': path.stat().st_size, 'url': 'https://example.invalid/'})
            sources = {'files': files}
            self.assertEqual(len(snare.verify_sources(directory, sources)), 6)
            (Path(directory) / files[2]['filename']).write_bytes(b'small immutable fixturE')
            with self.assertRaisesRegex(ValueError, 'Frozen raw input changed'):
                snare.verify_sources(directory, sources)


if __name__ == '__main__':
    unittest.main()
