"""Preparation and failure boundaries of the persistent campaign entrypoint."""

from pathlib import Path

import pytest

from paper.experiments.stochastic_ct import robustness_bootstrap as bootstrap


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    campaign = bootstrap.campaign
    frozen = tmp_path/'frozen_sources'
    frozen.mkdir()
    source = frozen/'robustness_reuse.py'
    source.write_text('# frozen fixture\n')
    catalog = tmp_path/'historical_reuse_catalog.json'
    campaign.write(catalog, dict(entries=[]))
    campaign.write(tmp_path/'implementation_validation.json', dict(status='passed',
        source_files={'robustness_reuse.py': dict(path=str(source), sha256=campaign.digest(source))},
        historical_reuse_catalog_sha256=campaign.digest(catalog)))
    calls = []
    monkeypatch.setattr(bootstrap.os, 'sched_getaffinity', lambda pid: set(campaign.CPUS))
    monkeypatch.setattr(bootstrap.os, 'sched_setaffinity', lambda pid, cpus: calls.append(('affinity', set(cpus))))
    monkeypatch.setattr(bootstrap, 'prepare_inputs',
        lambda root, **kwargs: (calls.append(('inputs', kwargs)) or ({'cohort': 'fixture'}, {'inputs': 'fixture'})))
    monkeypatch.setattr(campaign, 'prepare', lambda root: calls.append(('prepare', Path(root))))
    monkeypatch.setattr(campaign, 'verify_prepared', lambda root: calls.append(('verify', Path(root))))
    monkeypatch.setattr(campaign, 'execute', lambda root, **kwargs: calls.append(('execute', kwargs)))
    return tmp_path, calls


def test_preparation_persists_inputs_and_restores_worker_affinity(prepared):
    root, calls = prepared
    bootstrap.execute(root)
    assert bootstrap.campaign.read(root/'cohort.json') == {'cohort': 'fixture'}
    assert bootstrap.campaign.read(root/'inputs.json') == {'inputs': 'fixture'}
    assert calls == [('affinity', {0}), ('inputs', {'allow_download': True}),
                     ('affinity', {0, 2, 4, 6}), ('prepare', root), ('execute', {'publish': True})]


def test_resume_verifies_protocol_without_repreparing_inputs(prepared):
    root, calls = prepared
    bootstrap.campaign.write(root/'protocol.json', {'fixture': True})
    bootstrap.execute(root, publish=False)
    assert calls == [('verify', root), ('execute', {'publish': False})]


def test_catalog_change_stops_before_data_preparation(prepared):
    root, calls = prepared
    bootstrap.campaign.write(root/'historical_reuse_catalog.json', {'entries': ['changed']})
    with pytest.raises(ValueError, match='Historical reuse catalog changed'):
        bootstrap.execute(root)
    assert calls == []


def test_unvalidated_reuse_catalog_stops_before_data_preparation(prepared):
    root, calls = prepared
    gate = bootstrap.campaign.read(root/'implementation_validation.json')
    del gate['historical_reuse_catalog_sha256']
    bootstrap.campaign.write(root/'implementation_validation.json', gate)
    with pytest.raises(ValueError, match='Historical reuse catalog was not validated'):
        bootstrap.execute(root)
    assert calls == []


def test_source_change_stops_before_data_preparation(prepared):
    root, calls = prepared
    (root/'frozen_sources/robustness_reuse.py').write_text('# edited after validation\n')
    with pytest.raises(ValueError, match='Frozen source changed'):
        bootstrap.execute(root)
    assert calls == []


def test_publication_failure_preserves_completed_numerical_evaluation(prepared, monkeypatch):
    root, _ = prepared
    campaign = bootstrap.campaign
    campaign.write(root/'protocol.json', {'fixture': True})
    def fail_publication(root, **kwargs):
        campaign.update(root, 'evaluated', 'publication_pending', error='Git checkout advanced')
        raise ValueError('Git checkout advanced')
    monkeypatch.setattr(campaign, 'execute', fail_publication)
    with pytest.raises(ValueError, match='Git checkout advanced'):
        bootstrap.execute(root)
    state = campaign.read(root/'progress.json')
    assert (state['status'], state['phase']) == ('evaluated', 'publication_pending')


def test_execution_failure_remains_a_failure(prepared, monkeypatch):
    root, _ = prepared
    campaign = bootstrap.campaign
    campaign.write(root/'protocol.json', {'fixture': True})
    def fail_execution(root, **kwargs):
        campaign.update(root, 'running', 'evaluation')
        raise RuntimeError('Worker failed')
    monkeypatch.setattr(campaign, 'execute', fail_execution)
    with pytest.raises(RuntimeError, match='Worker failed'):
        bootstrap.execute(root)
    state = campaign.read(root/'progress.json')
    assert (state['status'], state['phase']) == ('failed', 'evaluation')
