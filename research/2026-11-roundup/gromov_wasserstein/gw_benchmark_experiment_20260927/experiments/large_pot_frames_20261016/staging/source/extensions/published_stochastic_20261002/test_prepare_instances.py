import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from prepare_instances import prepare_one,digest


def fixture(tmp_path):
    A=np.array([[0,1,.5],[1,0,.75],[.5,.75,0]],dtype=np.float32)
    arrays={'A':A,'B':A[::-1,::-1].copy(),'a':np.full(3,1/3),'b':np.full(3,1/3)}
    source=tmp_path/'original.npz'
    np.savez(source,**arrays)
    metadata={'n':3,'m':3,**{k+'_sha256':hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items() if k.isupper()}}
    m=tmp_path/'original.json';m.write_text(json.dumps(metadata))
    spec={'n':3,'source_input':{'kind':'npz','path':str(source),'metadata':str(m),'metadata_sha256':digest(m),'sha256':digest(source)}}
    return arrays,spec


def test_repackaging_preserves_every_array_and_binds_manifest(tmp_path):
    arrays,spec=fixture(tmp_path)
    target=tmp_path/'data'
    result=prepare_one('test',spec,target)
    m=json.loads((target/'manifest.json').read_text())
    assert not m['repackaging']['values_changed']
    assert result['manifest_sha256']==digest(target/'manifest.json')
    for name,expected in arrays.items():
        actual=np.load(target/(name+'.npy'))
        assert actual.dtype==expected.dtype
        np.testing.assert_array_equal(actual,expected)
        assert m['array_files_sha256'][name+'.npy']==digest(target/(name+'.npy'))
    with pytest.raises(FileExistsError):
        prepare_one('test',spec,target)


def test_changed_source_metadata_is_rejected(tmp_path):
    _,spec=fixture(tmp_path)
    Path(spec['source_input']['metadata']).write_text('{}')
    with pytest.raises(ValueError,match='metadata hash'):
        prepare_one('test',spec,tmp_path/'data')


def test_changed_array_value_binding_is_rejected(tmp_path):
    _,spec=fixture(tmp_path)
    path=Path(spec['source_input']['metadata'])
    value=json.loads(path.read_text());value['A_sha256']='0'*64
    path.write_text(json.dumps(value));spec['source_input']['metadata_sha256']=digest(path)
    with pytest.raises(ValueError,match='value hash'):
        prepare_one('test',spec,tmp_path/'data')


def test_existing_canonical_instance_is_verified_and_linked(tmp_path):
    _,spec=fixture(tmp_path)
    canonical=tmp_path/'canonical';prepare_one('test',spec,canonical)
    spec['source_input']={'kind':'canonical','directory':str(canonical),
        'metadata':str(canonical/'manifest.json'),'metadata_sha256':digest(canonical/'manifest.json')}
    target=tmp_path/'linked';prepare_one('test',spec,target)
    assert target.is_symlink() and target.resolve()==canonical
    assert (target/'manifest.json').read_bytes()==(canonical/'manifest.json').read_bytes()
