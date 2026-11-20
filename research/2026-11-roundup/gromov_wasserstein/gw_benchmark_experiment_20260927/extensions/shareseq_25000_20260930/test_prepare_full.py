"""Small checks for complete-population reuse and the provenance chain."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from scipy.spatial.distance import cdist

import prepare_full as full
import prepare_shareseq as prep


def fixture(directory):
    directory.mkdir()
    rng = np.random.default_rng(47)
    X, Y = rng.normal(size=(11, 4)), rng.normal(size=(11, 3))
    X[-1] *= 100  # ensure the full-population normalization includes an extreme cell
    ids = np.asarray([f'cell-{i:03d}' for i in range(11)])
    np.savez(directory / 'representations.npz', rna=X, atac=Y, paired_cell_ids=ids,
             source_original_indices=np.arange(11)+2, target_original_indices=np.arange(11)+5,
             rna_selected_feature_ids=np.asarray(['g0','g1','g2','g3']))
    recipe = copy.deepcopy(prep.RECIPE)
    old = dict(status='completed', dataset='tiny paired fixture', recipe=recipe,
        binding=dict(preparation_script_sha256=prep.sha256_file(prep.__file__),
            recipe_sha256=hashlib.sha256(prep.canonical_json(recipe)).hexdigest(),
            source_manifest_sha256='a'*64, representations_file='representations.npz',
            representations_sha256=prep.sha256_file(directory/'representations.npz'), raw_files={}),
        qc=dict(retained_paired_cells=11,rna_raw_cells=20,atac_raw_cells=20,paired_before_qc=14,
                paired_failing_either=3),
        representations=dict(rna_components=4,atac_components=3,
                             rna_representation_sha256=prep.array_sha256(X),
                             atac_representation_sha256=prep.array_sha256(Y)),
        versions={'numpy':'fixture'},python='fixture')
    prep.write_json(directory/'preparation.json',old)
    return X,Y,ids,prep.sha256_file(directory/'preparation.json')


class FullPreparationChecks(unittest.TestCase):
    def test_all_retained_cells_both_maps_and_full_metric_normalization(self):
        with tempfile.TemporaryDirectory() as temp:
            previous,out=Path(temp)/'previous',Path(temp)/'full'
            X,Y,ids,digest=fixture(previous)
            args=SimpleNamespace(previous_preparation=previous,out_dir=out,threads=1,distance_block_rows=4)
            with mock.patch.object(prep,'representations',side_effect=AssertionError('No refit permitted')):
                result=full.prepare(args,expected_manifest_sha256=digest)
            self.assertFalse(result['representation_refit'])
            self.assertEqual(result['instances']['shareseq_full']['n'],11)
            self.assertEqual(result['recipe']['instance_sizes'],{'shareseq_full':11})
            self.assertEqual(result['binding']['previous_preparation_manifest_sha256'],digest)
            self.assertEqual((previous/'representations.npz').read_bytes(),(out/'representations.npz').read_bytes())
            self.assertEqual((previous/'preparation.json').read_bytes(),(out/'previous_preparation.json').read_bytes())
            directory=out/'shareseq_full'
            with np.load(directory/'correspondence.npz') as corr:
                source,target=corr['source_population_indices'],corr['target_population_indices']
                np.testing.assert_array_equal(np.sort(source),np.arange(11))
                np.testing.assert_array_equal(np.sort(target),np.arange(11))
                np.testing.assert_array_equal(corr['source_cell_ids'],corr['target_cell_ids'][corr['forward']])
                np.testing.assert_array_equal(corr['inverse'][corr['forward']],np.arange(11))
                self.assertEqual(set(corr['source_cell_ids']),set(ids))
                self.assertEqual(set(corr['target_cell_ids']),set(ids))
                expected_order=prep.paired_orders(11,{'all':11})['all']
                np.testing.assert_array_equal(source,expected_order['source'])
                np.testing.assert_array_equal(target,expected_order['target'])
                for filename,coords,order in [('A.npy',X,source),('B.npy',Y,target)]:
                    distances=cdist(coords[order],coords[order]);expected=(distances/distances.max()).astype(np.float32)
                    np.testing.assert_array_equal(np.load(directory/filename),expected)
                    self.assertEqual(np.load(directory/filename).max(),1)
            np.testing.assert_array_equal(np.load(directory/'a.npy'),np.full(11,1/11))
            manifest=json.loads((directory/'manifest.json').read_text())
            self.assertEqual(manifest['binding'],result['binding'])
            self.assertEqual(result['instances']['shareseq_full']['metadata_sha256'],prep.sha256_file(directory/'manifest.json'))
            with self.assertRaises(FileExistsError):
                full.prepare(args,expected_manifest_sha256=digest)

    def test_previous_manifest_and_representation_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            previous=Path(temp)/'previous';_,_,_,digest=fixture(previous)
            path=previous/'preparation.json';saved=path.read_bytes()
            path.write_bytes(saved+b' ')
            with self.assertRaisesRegex(ValueError,'manifest hash'):
                full.load_previous(previous,expected_manifest_sha256=digest)
            path.write_bytes(saved)
            rep=previous/'representations.npz';rep.write_bytes(rep.read_bytes()+b' ')
            with self.assertRaisesRegex(ValueError,'representation file hash'):
                full.load_previous(previous,expected_manifest_sha256=digest)

    def test_partial_population_and_changed_helper_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            previous=Path(temp)/'previous';fixture(previous)
            path=previous/'preparation.json';manifest=json.loads(path.read_text())
            manifest['qc']['retained_paired_cells']=12
            prep.write_json(path,manifest)
            with self.assertRaisesRegex(ValueError,'complete valid paired population'):
                full.load_previous(previous,expected_manifest_sha256=prep.sha256_file(path))
            manifest['qc']['retained_paired_cells']=11
            manifest['binding']['preparation_script_sha256']='b'*64
            prep.write_json(path,manifest)
            with self.assertRaisesRegex(ValueError,'helper source'):
                full.load_previous(previous,expected_manifest_sha256=prep.sha256_file(path))


if __name__=='__main__':
    unittest.main()
