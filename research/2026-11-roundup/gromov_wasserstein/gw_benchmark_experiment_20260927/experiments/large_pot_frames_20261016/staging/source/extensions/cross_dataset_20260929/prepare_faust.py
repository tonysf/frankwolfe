"""Prepare three frozen FAUST registration pairs from a user-supplied archive.

No network access, account creation, license acceptance, or archive extraction.
Actual 6,890-vertex processing requires a Slurm allocation; helper tests are tiny.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import stat
import struct
import sys
import tarfile
import zipfile

import numpy as np
import scipy
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path


LICENSE_URL = "https://faust-leaderboard.is.tuebingen.mpg.de/license"
N_VERTICES = 6890
PAIRS = tuple((f"faust_subject{s}_pose0_1", 10*s, 10*s+1) for s in range(3))
MAX_MEMBER_BYTES = 16 * 1024 * 1024
PLY_TYPES = {
    'char': 'b', 'int8': 'b', 'uchar': 'B', 'uint8': 'B',
    'short': 'h', 'int16': 'h', 'ushort': 'H', 'uint16': 'H',
    'int': 'i', 'int32': 'i', 'uint': 'I', 'uint32': 'I',
    'float': 'f', 'float32': 'f', 'double': 'd', 'float64': 'd',
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_name(name):
    """Reject ambiguous/traversing archive paths, even though nothing is extracted."""
    path = PurePosixPath(name)
    if (not name or '\\' in name or path.is_absolute() or '..' in path.parts
            or any(':' in component for component in path.parts)):
        raise ValueError(f'Unsafe archive member: {name!r}')
    return path


def read_registration_members(archive):
    wanted = {f'tr_reg_{i:03d}.ply' for _, a, b in PAIRS for i in (a, b)}
    result = {}

    def add(name, size, loader, regular=True, directory=False):
        path = safe_name(name)
        if directory:
            return
        if not regular:
            raise ValueError(f'Archive links/special members are unsupported: {name}')
        if path.name not in wanted:
            return
        if 'registrations' not in path.parts:
            raise ValueError(f'Expected training/registrations member: {name}')
        if path.name in result:
            raise ValueError(f'Duplicate selected registration: {path.name}')
        if not 0 < size <= MAX_MEMBER_BYTES:
            raise ValueError(f'Unexpected registration size: {name}')
        payload = loader()
        if len(payload) != size:
            raise ValueError(f'Truncated registration: {name}')
        result[path.name] = {'member': name, 'bytes': payload,
                            'sha256': hashlib.sha256(payload).hexdigest()}

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                mode = member.external_attr >> 16
                add(member.filename, member.file_size, lambda m=member: z.read(m),
                    regular=not stat.S_ISLNK(mode) and
                    (stat.S_IFMT(mode) in (0, stat.S_IFREG, stat.S_IFDIR)),
                    directory=member.is_dir())
    else:
        with tarfile.open(archive, 'r:*') as z:
            for member in z:
                def load(m=member):
                    stream = z.extractfile(m)
                    if stream is None:
                        raise ValueError(f'Cannot read {m.name}')
                    with stream:
                        return stream.read(MAX_MEMBER_BYTES + 1)
                add(member.name, member.size, load, member.isfile(), member.isdir())
    if set(result) != wanted:
        raise ValueError(f'Missing frozen registrations: {sorted(wanted - set(result))}')
    return result


def read_ply(payload):
    """Read scalar vertex properties and triangular face lists in ASCII/binary PLY."""
    stream = io.BytesIO(payload)
    if stream.readline().strip() != b'ply':
        raise ValueError('Expected a PLY mesh.')
    fmt = None
    elements = []
    for _ in range(256):
        line = stream.readline()
        if not line or stream.tell() > 65536:
            raise ValueError('Missing or oversized PLY header.')
        words = line.decode('ascii').strip().split()
        if not words or words[0] in ('comment', 'obj_info'):
            continue
        if words[0] == 'end_header':
            break
        if words[0] == 'format' and len(words) == 3 and words[2] == '1.0':
            fmt = words[1]
        elif words[0] == 'element' and len(words) == 3:
            count = int(words[2])
            if not 0 < count <= 300000:
                raise ValueError('Unexpected PLY element count.')
            elements.append((words[1], count, []))
        elif words[0] == 'property' and elements:
            if len(words) == 3 and words[1] in PLY_TYPES:
                elements[-1][2].append((words[2], words[1]))
            elif (len(words) == 5 and words[1] == 'list' and
                  words[2] in PLY_TYPES and words[3] in PLY_TYPES):
                elements[-1][2].append((words[4], words[2], words[3]))
            else:
                raise ValueError('Unsupported PLY property.')
        else:
            raise ValueError('Unsupported PLY header.')
    else:
        raise ValueError('Missing end_header.')
    if fmt not in ('ascii', 'binary_little_endian', 'binary_big_endian'):
        raise ValueError('Unsupported PLY encoding.')
    if [x[0] for x in elements] != ['vertex', 'face']:
        raise ValueError('Expected vertex then face elements only.')
    endian = '<' if fmt == 'binary_little_endian' else '>'

    def scalar(kind, tokens):
        code = PLY_TYPES[kind]
        if fmt == 'ascii':
            if not tokens:
                raise ValueError('Truncated ASCII PLY record.')
            return float(tokens.pop(0)) if code in 'fd' else int(tokens.pop(0))
        size = struct.calcsize(code)
        raw = stream.read(size)
        if len(raw) != size:
            raise ValueError('Truncated binary PLY record.')
        return struct.unpack(endian + code, raw)[0]

    arrays = {}
    for element, count, properties in elements:
        names = [p[0] for p in properties]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate PLY property.')
        if element == 'vertex' and (not {'x', 'y', 'z'} <= set(names) or
                                    any(len(p) != 2 for p in properties)):
            raise ValueError('Expected scalar x/y/z vertex properties.')
        face_names = set(names) & {'vertex_indices', 'vertex_index'}
        if element == 'face' and len(face_names) != 1:
            raise ValueError('Expected exactly one face index list.')
        rows = []
        for _ in range(count):
            tokens = stream.readline().decode('ascii').split() if fmt == 'ascii' else []
            record = {}
            for prop in properties:
                if len(prop) == 2:
                    record[prop[0]] = scalar(prop[1], tokens)
                else:
                    if PLY_TYPES[prop[1]] in 'fd' or PLY_TYPES[prop[2]] in 'fd':
                        raise ValueError('Face lists must have integer count/indices.')
                    size = scalar(prop[1], tokens)
                    if size < 0 or size > 16:
                        raise ValueError('Unexpected PLY list length.')
                    record[prop[0]] = [scalar(prop[2], tokens) for _ in range(size)]
            if tokens:
                raise ValueError('Extra ASCII PLY record values.')
            if element == 'vertex':
                rows.append([record[k] for k in ('x', 'y', 'z')])
            else:
                face = record[next(iter(face_names))]
                if not isinstance(face, list) or len(face) != 3:
                    raise ValueError('Only triangular faces are supported.')
                rows.append(face)
        arrays[element] = np.asarray(rows, dtype=np.float64 if element == 'vertex' else np.int64)
    vertices, faces = arrays['vertex'], arrays['face']
    if not np.isfinite(vertices).all() or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError('Nonfinite vertices or invalid face indices.')
    if np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0):
        raise ValueError('Degenerate face indices.')
    return vertices, faces


def edge_geodesics(vertices, faces):
    """Mesh-edge shortest paths, an explicit approximation of surface geodesics."""
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.unique(np.sort(edges, axis=1), axis=0)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError('Nonpositive or nonfinite mesh edge length.')
    graph = coo_matrix((lengths, (edges[:, 0], edges[:, 1])),
                       shape=(len(vertices), len(vertices))).tocsr()
    distances = shortest_path(graph, method='D', directed=False)
    if not np.isfinite(distances).all():
        raise ValueError('Disconnected mesh.')
    maximum = float(distances.max())
    if maximum <= 0 or not np.allclose(distances, distances.T, atol=1e-12):
        raise ValueError('Invalid mesh distances.')
    return distances / maximum, maximum


def correspondence(n, source_seed, target_seed):
    source = np.random.default_rng(source_seed).permutation(n)
    target = np.random.default_rng(target_seed).permutation(n)
    true_target = np.argsort(target)[source]
    return source, target, true_target


def prepare(archive, out, runtime):
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Full FAUST preparation requires a scheduled Slurm allocation.')
    archive, out, runtime = map(Path, (archive, out, runtime))
    archive_sha = digest(archive)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('FAUST output directory must be new or empty.')
    out.mkdir(parents=True, exist_ok=True)
    registrations = read_registration_members(archive)
    sys.path.insert(0, str(runtime.resolve()))
    from make_instances import write_instance
    manifest = dict(schema_version=1, dataset='FAUST registered training meshes',
                    archive_sha256=archive_sha, license_url=LICENSE_URL,
                    archive_source='user_provided_licensed_training_archive',
                    reader_sha256=digest(__file__), writer_sha256=digest(runtime / 'make_instances.py'),
                    numpy_version=np.__version__, scipy_version=scipy.__version__,
                    slurm_job_id=os.environ['SLURM_JOB_ID'], instances={})
    for pair_index, (instance_id, left, right) in enumerate(PAIRS):
        selected = [registrations[f'tr_reg_{i:03d}.ply'] for i in (left, right)]
        meshes = [read_ply(item['bytes']) for item in selected]
        if any(len(v) != N_VERTICES for v, _ in meshes):
            raise ValueError('Expected all 6,890 registered template vertices.')
        if not np.array_equal(meshes[0][1], meshes[1][1]):
            raise ValueError('Registration pair must share identical template face indices.')
        A, sa = edge_geodesics(*meshes[0])
        B, sb = edge_geodesics(*meshes[1])
        source_seed, target_seed = 2026092900 + 2*pair_index, 2026092901 + 2*pair_index
        source, target, true_target = correspondence(N_VERTICES, source_seed, target_seed)
        path = out / f'{instance_id}.npz'
        metadata = dict(family='faust_registered_mesh', instance_id=instance_id,
                        source_member=selected[0]['member'], target_member=selected[1]['member'],
                        source_member_sha256=selected[0]['sha256'], target_member_sha256=selected[1]['sha256'],
                        archive_sha256=archive_sha, metric='mesh_edge_shortest_paths_approximate_surface_geodesic',
                        scale='each_matrix_divided_by_its_own_maximum', A_scale=sa, B_scale=sb,
                        source_permutation_seed=source_seed, target_permutation_seed=target_seed,
                        subset_rule='all_6890_template_vertices_independent_fixed_permutations',
                        known_optimum=None, correspondence_used_by_optimizer=False)
        write_instance(path, A[np.ix_(source, source)], B[np.ix_(target, target)], metadata)
        corr = out / f'{instance_id}.correspondence.npz'
        np.savez(corr, forward=true_target, inverse=np.argsort(true_target),
                 source_population_indices=source, target_population_indices=target)
        manifest['instances'][instance_id] = dict(
            n=N_VERTICES, filename=path.name, sha256=digest(path),
            metadata_file=path.with_suffix('.json').name, metadata_sha256=digest(path.with_suffix('.json')),
            correspondence_file=corr.name, correspondence_sha256=digest(corr),
            correspondence='shared_registered_template_vertex_index_not_official_raw_scan_metric')
        del A, B
    (out / 'faust_preparation.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.archive, args.out, args.runtime)
