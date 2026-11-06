"""Export the existing small synthetic NPZ without regenerating or rescaling it."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(source, metadata, output):
    source, metadata, output = map(Path, (source, metadata, output))
    description = json.loads(metadata.read_text())
    assert description['n'] == description['m'] == 512
    assert description['noise'] == 0 and description['seed'] == 1000
    assert description['known_optimum'] == 0
    with np.load(source, allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in ('A', 'B', 'a', 'b')}
    for key in ('A', 'B'):
        value = arrays[key]
        assert value.shape == (512, 512) and value.dtype == np.float32
        assert np.isfinite(value).all() and (value >= 0).all()
        assert np.array_equal(value, value.T)
        assert hashlib.sha256(value.tobytes()).hexdigest() == description[key+'_sha256']
    for key in ('a', 'b'):
        assert arrays[key].shape == (512,)
        assert np.array_equal(arrays[key], np.full(512, 1/512))
    output.mkdir(parents=True, exist_ok=False)
    for key, value in arrays.items():
        np.save(output/(key+'.npy'), value, allow_pickle=False)
    manifest = dict(schema_version=1, instance='metric512_s1000', n=512, m=512,
                    source_npz_sha256=sha256(source), source_metadata_sha256=sha256(metadata),
                    source_metadata=description,
                    transformation='NPZ arrays exported unchanged; no regeneration, normalization, or dtype conversion.',
                    array_files_sha256={key+'.npy': sha256(output/(key+'.npy')) for key in arrays})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    return dict(instance_manifest_sha256=sha256(output/'manifest.json'),
                array_files_sha256=manifest['array_files_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.metadata, args.out), indent=2))
