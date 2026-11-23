"""Publication overlays preserve numerical provenance and wait for completion."""

import hashlib
import json
from pathlib import Path

import pytest

from paper.experiments.stochastic_ct import compact_publication_handoff as handoff


CONTROLLER = '''
import hashlib,json
from pathlib import Path
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_prepared(root):
    p=read(root/'protocol.json')
    for name,expected in p['frozen_files'].items():
        if sha(root/name)!=expected: raise ValueError('Frozen fixture changed')
def validate_result(cfg,cpu,protocol,root):
    folder=root/'runs'/cfg['run_id']
    m,r=read(folder/'metadata.json'),read(folder/'result.json')
    if m['config']!=cfg or m['protocol_sha256']!=sha(root/'protocol.json'):
        raise ValueError('Original run provenance changed')
    if r['metadata']!=m or r['trace_sha256']!=sha(folder/'trace.json'):
        raise ValueError('Original trace changed')
    return dict(status=r['status'])
'''


@pytest.fixture
def prepared(tmp_path):
    original, output = tmp_path/'original', tmp_path/'publication'
    original.mkdir()
    source = original/'frozen_sources/robustness_campaign.py'
    source.parent.mkdir()
    source.write_text(CONTROLLER)
    configs = [dict(run_id=phase, phase=phase, method='fixture')
               for phase in ('development', 'evaluation')]
    plan = dict(runs=configs, assignments={'development': 0, 'evaluation': 2})
    handoff.write(original/'plans/plan.json', plan)
    handoff.write(original/'evaluation_plan.json', plan)
    handoff.write(original/'selection.json', {'fixture': True})
    handoff.write(original/'cohort.json', {'fixture': True})
    handoff.write(original/'stages/evaluation.json', {'status': 'completed'})
    handoff.write(original/'publication_config.json', {'expected_head': 'old numerical commit'})
    protocol = dict(frozen_sources={'robustness_campaign.py': handoff.digest(source)},
        numerical_sources={'fixture.py': 'original executed bytes'},
        frozen_files={name: handoff.digest(original/name) for name in
            ('selection.json', 'evaluation_plan.json', 'plans/plan.json', 'publication_config.json')})
    handoff.write(original/'protocol.json', protocol)
    for cfg in configs:
        folder = original/'runs'/cfg['run_id']
        meta = dict(config=cfg, protocol_sha256=handoff.digest(original/'protocol.json'))
        handoff.write(folder/'metadata.json', meta)
        handoff.write(folder/'trace.json', [dict(iteration=0, raw_loss=1.)])
        handoff.write(folder/'result.json', dict(status='completed', metadata=meta,
            trace_sha256=handoff.digest(folder/'trace.json')))
        (folder/'checkpoint.pkl').write_bytes(b'not publication evidence')
    handoff.write(original/'completion.json', dict(status='completed', evaluation_runs=1,
        protocol_sha256=handoff.digest(original/'protocol.json'),
        selection_sha256=handoff.digest(original/'selection.json'),
        evaluation_plan_sha256=handoff.digest(original/'evaluation_plan.json'),
        stage_files={'stages/evaluation.json': handoff.digest(original/'stages/evaluation.json')}))
    publication_code = tmp_path/'publication_code'
    publication_code.mkdir()
    renderer = publication_code/'compact_paper.py'
    renderer.write_text("""import json
def publish_expanded(root):
    result={'status':'completed_published','pages':2}
    (root/'report_update.json').write_text(json.dumps(result))
    return result
""")
    predecessor = tmp_path/'compact_predecessor'
    predecessor.mkdir()
    (predecessor/'paper.tex').write_text('Compact reviewed predecessor')
    helper = Path(handoff.__file__)
    config = dict(original_root=str(original), output_root=str(output),
        expected_original_protocol_sha256=handoff.digest(original/'protocol.json'),
        source_files={p.name: dict(path=str(p), sha256=handoff.digest(p))
                      for p in (helper, renderer)},
        predecessor_bundle=str(predecessor), publication=dict(expected_head='compact document commit',
            predecessor_files={'paper.tex': handoff.digest(predecessor/'paper.tex')}))
    config_path = tmp_path/'handoff_config.json'
    handoff.write(config_path, config)
    return config_path, config, original, output


def test_incomplete_campaign_does_not_render_or_copy_results(prepared):
    config_path, _, original, output = prepared
    (original/'completion.json').unlink()
    assert handoff.execute(config_path)['status'] == 'waiting_for_numerical_completion'
    assert not (output/'runs').exists()
    assert not (output/'report_update.json').exists()


def test_completed_overlay_preserves_original_evidence_and_publishes_once(prepared):
    config_path, config, original, output = prepared
    original_bytes = {str(p.relative_to(original)): p.read_bytes()
                      for p in original.rglob('*') if p.is_file()}
    assert handoff.execute(config_path) == {'status': 'completed_published', 'pages': 2}
    for name, raw in original_bytes.items():
        assert (original/name).read_bytes() == raw
    for name in ('protocol.json', 'completion.json', 'publication_config.json'):
        assert (output/('original_'+name)).read_bytes() == original_bytes[name]
    assert handoff.read(output/'publication_config.json') == config['publication']
    overlay = handoff.read(output/'protocol.json')
    assert overlay['kind'] == 'publication_overlay' and not overlay['new_numerical_execution']
    assert overlay['original_protocol_sha256'] == handoff.digest(original/'protocol.json')
    assert overlay['numerical_sources'] == handoff.read(original/'protocol.json')['numerical_sources']
    for name, expected in overlay['frozen_files'].items():
        assert handoff.digest(output/name) == expected
    for name, expected in overlay['frozen_sources'].items():
        assert handoff.digest(output/'frozen_sources'/name) == expected
    for path in (output/'runs').glob('*/*.json'):
        assert path.read_bytes() == (original/path.relative_to(output)).read_bytes()
        assert not path.is_symlink()
    assert not list((output/'runs').glob('*/checkpoint.pkl'))
    assert handoff.execute(config_path) == {'status': 'completed_published', 'pages': 2}


@pytest.mark.parametrize('changed', ['source', 'protocol', 'completion', 'stage', 'trace', 'configuration'])
def test_changed_provenance_blocks_handoff(prepared, changed):
    config_path, config, original, output = prepared
    if changed == 'source':
        Path(config['source_files']['compact_paper.py']['path']).write_text('# edited')
    elif changed == 'protocol':
        handoff.write(original/'protocol.json', {})
    elif changed == 'completion':
        p=original/'completion.json';x=handoff.read(p);x['evaluation_runs']=2;handoff.write(p,x)
    elif changed == 'stage':
        handoff.write(original/'stages/evaluation.json', {'status':'changed'})
    elif changed == 'trace':
        handoff.write(original/'runs/evaluation/trace.json', [dict(iteration=0,raw_loss=99.)])
    else:
        p=original/'runs/evaluation/metadata.json';x=handoff.read(p);x['config']['method']='changed';handoff.write(p,x)
    with pytest.raises(ValueError):
        handoff.execute(config_path)
    assert not (output/'report_update.json').exists()


def test_failed_evaluation_cannot_publish(prepared):
    config_path, _, original, output = prepared
    p=original/'runs/evaluation/result.json';x=handoff.read(p);x['status']='numerical_failure';handoff.write(p,x)
    with pytest.raises(ValueError,match='evaluation failed'):
        handoff.execute(config_path)
    assert not (output/'report_update.json').exists()


def test_failed_development_candidate_remains_in_evidence(prepared):
    config_path, _, original, output = prepared
    p=original/'runs/development/result.json';x=handoff.read(p);x['status']='numerical_failure';handoff.write(p,x)
    handoff.execute(config_path)
    assert handoff.read(output/'original_validation.json')['failed_development_runs']==1
    assert handoff.read(output/'runs/development/result.json')['status']=='numerical_failure'


def test_live_campaign_lock_prevents_handoff(prepared):
    config_path, _, original, output = prepared
    with (original/'campaign.lock').open('a') as lock:
        handoff.fcntl.flock(lock,handoff.fcntl.LOCK_EX|handoff.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            handoff.execute(config_path)
    assert not (output/'report_update.json').exists()


def test_changed_staged_evidence_is_not_overwritten_on_retry(prepared):
    config_path, config, original, output = prepared
    validation=handoff.validate_original(original)
    handoff.prepare_overlay(config_path,config,validation)
    # A hardlink would also mutate the source, so replace the destination inode.
    p=output/'runs/evaluation/trace.json';p.unlink();p.write_text('changed')
    with pytest.raises(ValueError,match='Previously staged evidence changed'):
        handoff.prepare_overlay(config_path,config,validation)
    assert not (output/'report_update.json').exists()
