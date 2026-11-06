"""Full-population requests must never silently consume the earlier subset."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('full_shareseq_campaign', HERE/'campaign.py')
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


def test_full_population_binding_rejects_subsets_and_changed_manifests(tmp_path):
    config = dict(instance_id='shareseq_full', expected_n=31890)
    full = dict(id='shareseq_full', n=31890, m=31890)
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(full))
    preparation = dict(status='completed', qc=dict(retained_paired_cells=31890),
        instances=dict(shareseq_full=dict(metadata_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
    campaign.validate_preparation(config, preparation, full, path)
    for incorrect in (dict(id='shareseq25000',n=25000,m=25000),
                      dict(id='shareseq_full',n=25000,m=25000),
                      dict(id='shareseq_full',n=31890,m=25000)):
        with pytest.raises(ValueError):
            campaign.validate_preparation(config,preparation,incorrect,path)
    preparation['qc']['retained_paired_cells'] = 32231
    with pytest.raises(ValueError):
        campaign.validate_preparation(config,preparation,full,path)
    preparation['qc']['retained_paired_cells'] = 31890
    path.write_text(json.dumps(dict(full,unexpected_change=True)))
    with pytest.raises(ValueError):
        campaign.validate_preparation(config,preparation,full,path)
