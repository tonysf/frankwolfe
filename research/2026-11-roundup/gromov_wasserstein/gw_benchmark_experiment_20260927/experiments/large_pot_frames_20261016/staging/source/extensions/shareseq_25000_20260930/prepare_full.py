"""Freeze all previously QC-retained SHARE-seq cells without refitting.

Run the full distance generation through the CPU scheduler. The original
31,890-cell representations are copied byte-for-byte from the completed
25,000-subset preparation, and all retained cells enter both modalities.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil

import numpy as np

import prepare_shareseq as prep

EXPECTED_PREVIOUS_MANIFEST_SHA256 = '1ab52cc35fbbb0492a8aa88c666bab73c36168f305000cd0f32b481f02f64bb7'


def load_previous(directory, *, expected_manifest_sha256=EXPECTED_PREVIOUS_MANIFEST_SHA256):
    """Verify the frozen completed preparation before reading representations."""
    directory = Path(directory).resolve()
    path = directory / 'preparation.json'
    if prep.sha256_file(path) != expected_manifest_sha256:
        raise ValueError('Previous preparation manifest hash does not match the frozen source.')
    previous = json.loads(path.read_text())
    if previous.get('status') != 'completed':
        raise ValueError('Previous preparation is incomplete.')
    binding = previous['binding']
    if binding['preparation_script_sha256'] != prep.sha256_file(prep.__file__):
        raise ValueError('Metric/order helper source differs from the previous preparation.')
    if binding['recipe_sha256'] != hashlib.sha256(prep.canonical_json(previous['recipe'])).hexdigest():
        raise ValueError('Previous recipe hash mismatch.')
    for field in ('source_order_seed', 'target_order_seed', 'cell_selection_seed', 'metric', 'objective'):
        if previous['recipe'][field] != prep.RECIPE[field]:
            raise ValueError('Previous metric/order recipe is incompatible: ' + field)
    repfile = Path(binding['representations_file'])
    if repfile.name != str(repfile):
        raise ValueError('Representations must be a simple relative filename.')
    repfile = directory / repfile
    if prep.sha256_file(repfile) != binding['representations_sha256']:
        raise ValueError('Previous representation file hash mismatch.')
    with np.load(repfile, allow_pickle=False) as data:
        arrays = {key: data[key] for key in ('rna', 'atac', 'paired_cell_ids',
                  'source_original_indices', 'target_original_indices')}
    X, Y, ids = arrays['rna'], arrays['atac'], arrays['paired_cell_ids']
    n = previous['qc']['retained_paired_cells']
    if isinstance(n, bool) or not isinstance(n, int) or n < 2:
        raise ValueError('Invalid retained population size.')
    if (X.shape != (n, previous['representations']['rna_components'])
            or Y.shape != (n, previous['representations']['atac_components'])
            or ids.shape != (n,) or ids.dtype.kind not in 'US'
            or len(set(ids)) != n or any(not value for value in ids)
            or not np.isfinite(X).all() or not np.isfinite(Y).all()):
        raise ValueError('Representations do not contain the complete valid paired population.')
    for key, array in (('rna', X), ('atac', Y)):
        if prep.array_sha256(array) != previous['representations'][key + '_representation_sha256']:
            raise ValueError('Representation array hash mismatch: ' + key)
    for key, raw_count in (('source_original_indices', previous['qc']['rna_raw_cells']),
                           ('target_original_indices', previous['qc']['atac_raw_cells'])):
        values = arrays[key]
        if (values.shape != (n,) or values.dtype.kind not in 'iu' or len(set(values)) != n
                or (values < 0).any() or (values >= raw_count).any()):
            raise ValueError('Invalid original cell indices: ' + key)
    return previous, arrays, repfile


def prepare(args, *, expected_manifest_sha256=EXPECTED_PREVIOUS_MANIFEST_SHA256):
    previous_dir, out = Path(args.previous_preparation).resolve(), Path(args.out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Full preparation requires a new or empty output directory.')
    previous, arrays, repfile = load_previous(previous_dir,
                                             expected_manifest_sha256=expected_manifest_sha256)
    out.mkdir(parents=True, exist_ok=True)
    # Copies make the new preparation portable and preserve the exact previous
    # bytes, including feature provenance arrays not used for distance creation.
    shutil.copyfile(repfile, out / 'representations.npz')
    shutil.copyfile(previous_dir / 'preparation.json', out / 'previous_preparation.json')
    if prep.sha256_file(out / 'representations.npz') != previous['binding']['representations_sha256']:
        raise ValueError('Copied representations failed verification.')
    if prep.sha256_file(out / 'previous_preparation.json') != expected_manifest_sha256:
        raise ValueError('Copied previous manifest failed verification.')
    n = len(arrays['paired_cell_ids'])
    recipe = copy.deepcopy(previous['recipe'])
    recipe.update(instance_sizes={'shareseq_full': n},
                  cell_selection='all previously jointly QC-retained paired cells; no subsampling',
                  representation_reuse='exact previous representation bytes; no normalization, PCA or LSI refit')
    binding = dict(previous['binding'])
    binding.update(
        previous_preparation_file='previous_preparation.json',
        previous_preparation_manifest_sha256=expected_manifest_sha256,
        previous_preparation_script_sha256=previous['binding']['preparation_script_sha256'],
        previous_recipe_sha256=previous['binding']['recipe_sha256'],
        preparation_script_sha256=prep.sha256_file(__file__),
        metric_helper_script_sha256=prep.sha256_file(prep.__file__),
        recipe_sha256=hashlib.sha256(prep.canonical_json(recipe)).hexdigest(),
        representations_file='representations.npz')
    instances = prep.freeze_instances(out, arrays['rna'], arrays['atac'], arrays['paired_cell_ids'],
        arrays['source_original_indices'], arrays['target_original_indices'], {'shareseq_full': n},
        binding, block_rows=args.distance_block_rows)
    manifest = dict(schema_version=1, status='completed', dataset=previous['dataset'],
        recipe=recipe, binding=binding, qc=previous['qc'], representations=previous['representations'],
        instances=instances, versions={package:importlib.metadata.version(package)
                                      for package in ('numpy', 'scipy', 'h5py', 'threadpoolctl')},
        python=platform.python_version(), representation_fit_versions=previous['versions'],
        representation_fit_python=previous['python'], requested_threads=args.threads,
        distance_block_rows=args.distance_block_rows,
        population='every cell retained by the unchanged original joint QC',
        representation_refit=False, previous_preparation_path=str(previous_dir))
    prep.write_json(out / 'preparation.json', manifest)
    print(json.dumps(dict(status='completed', manifest=str(out / 'preparation.json'),
                          instances={'shareseq_full': n}, representation_refit=False), sort_keys=True))
    return manifest


def main():
    from threadpoolctl import threadpool_limits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-preparation', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=int(os.environ.get('SLURM_CPUS_PER_TASK', '1')))
    parser.add_argument('--distance-block-rows', type=int, default=512)
    args = parser.parse_args()
    if min(args.threads, args.distance_block_rows) < 1:
        parser.error('Thread count and distance block size must be positive.')
    with threadpool_limits(limits=args.threads):
        prepare(args)


if __name__ == '__main__':
    main()
