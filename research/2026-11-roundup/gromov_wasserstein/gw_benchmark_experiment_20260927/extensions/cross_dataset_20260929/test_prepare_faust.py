import io
import struct
import sys
import tarfile
from pathlib import Path
import zipfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_faust as faust


def ply(binary=False):
    vertices = [(0., 0., 0.), (1., 0., 0.), (1., 1., 0.), (0., 1., 0.)]
    faces = [(0, 1, 2), (0, 2, 3)]
    encoding = 'binary_little_endian' if binary else 'ascii'
    header = ('ply\nformat ' + encoding + ' 1.0\nelement vertex 4\n'
              'property float x\nproperty float y\nproperty float z\n'
              'element face 2\nproperty list uchar int vertex_indices\nend_header\n').encode()
    if binary:
        return header + b''.join(struct.pack('<fff', *v) for v in vertices) + b''.join(
            struct.pack('<Biii', 3, *f) for f in faces)
    return header + ''.join(' '.join(map(str, v))+'\n' for v in vertices).encode() + b'3 0 1 2\n3 0 2 3\n'


@pytest.mark.parametrize('binary', [False, True])
def test_mesh_distance_is_edge_path_not_chord(binary):
    vertices, faces = faust.read_ply(ply(binary))
    distances, scale = faust.edge_geodesics(vertices, faces)
    assert scale == pytest.approx(2.)
    # No 1-3 diagonal: edge distance is 2, while the chord has length sqrt(2).
    assert distances[1, 3] == pytest.approx(1.)
    assert distances[0, 2] == pytest.approx(np.sqrt(2)/2)
    np.testing.assert_allclose(distances, distances.T)


def test_correspondence_survives_independent_permutations():
    source, target, forward = faust.correspondence(30, 2026092900, 2026092901)
    assert not np.array_equal(source, target)
    np.testing.assert_array_equal(target[forward], source)
    np.testing.assert_array_equal(source[np.argsort(forward)], target)


def test_disconnected_and_zero_length_meshes_rejected():
    vertices, faces = faust.read_ply(ply())
    with pytest.raises(ValueError, match='Disconnected'):
        faust.edge_geodesics(np.vstack([vertices, [5., 5., 5.]]), faces)
    vertices[1] = vertices[0]
    with pytest.raises(ValueError, match='Nonpositive'):
        faust.edge_geodesics(vertices, faces)


def test_invalid_faces_rejected():
    with pytest.raises(ValueError, match='invalid face'):
        faust.read_ply(ply().replace(b'3 0 2 3', b'3 0 2 9'))
    with pytest.raises(ValueError, match='Degenerate face'):
        faust.read_ply(ply().replace(b'3 0 2 3', b'3 0 2 2'))


@pytest.mark.parametrize('name', ['../evil', '/tmp/evil', 'a/../evil', 'C:/evil', 'a\\evil'])
def test_archive_paths_rejected(name):
    with pytest.raises(ValueError, match='Unsafe'):
        faust.safe_name(name)


@pytest.mark.parametrize('kind', ['zip', 'tar'])
def test_archive_reads_only_frozen_registrations(tmp_path, kind):
    archive = tmp_path / ('archive.'+kind)
    names = [f'training/registrations/tr_reg_{i:03d}.ply' for i in (0, 1, 10, 11, 20, 21)]
    if kind == 'zip':
        with zipfile.ZipFile(archive, 'w') as z:
            for name in names:
                z.writestr(name, ply())
    else:
        with tarfile.open(archive, 'w') as z:
            for name in names:
                member = tarfile.TarInfo(name)
                member.size = len(ply())
                z.addfile(member, io.BytesIO(ply()))
    members = faust.read_registration_members(archive)
    assert len(members) == 6
    assert members['tr_reg_010.ply']['bytes'] == ply()
    assert list(tmp_path.iterdir()) == [archive]


def test_duplicate_archive_registration_rejected(tmp_path):
    archive = tmp_path / 'archive.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('training/registrations/tr_reg_000.ply', ply())
        z.writestr('other/registrations/tr_reg_000.ply', ply())
    with pytest.raises(ValueError, match='Duplicate'):
        faust.read_registration_members(archive)


def test_missing_archive_registration_rejected(tmp_path):
    archive = tmp_path / 'archive.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('training/registrations/tr_reg_000.ply', ply())
    with pytest.raises(ValueError, match='Missing frozen'):
        faust.read_registration_members(archive)


def test_full_preparation_requires_scheduler(monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    with pytest.raises(RuntimeError, match='Slurm'):
        faust.prepare('unused.zip', 'unused', 'unused')
