"""Campaign design/provenance tests; no optimizer or held-out data is accessed."""
from collections import Counter
import hashlib
import json

import numpy as np
import pytest

from prepare_stage_c_campaign import (CAMPAIGN_PATH, STOCHASTIC, CanonicalInstances,
                                     candidate_catalog, make_finals, make_tuning,
                                     selected_catalog)


class NoData:
    def metadata(self, description):
        return {}


def _design():
    campaign = json.loads(CAMPAIGN_PATH.read_text())
    return campaign, candidate_catalog(campaign)


def _selection(catalog):
    return dict(schema_version=1, campaign_config_sha256='a' * 64,
                tuning_manifest_sha256='b' * 64,
                selections={method: dict(status='selected', candidate_id='c00', score=1.)
                            for method in catalog})


def _resolve(selection, catalog):
    return selected_catalog(selection, catalog, campaign_sha256='a' * 64,
                            tuning_manifest_sha256='b' * 64)


def test_exact_tuning_cells_and_replicate_counts():
    campaign, catalog = _design()
    assert all(len(candidates) == 12 for candidates in catalog.values())
    assert catalog['smoothing-fw'][-1]['parameters'] == {'gamma0': 1., 'rho0': 1., 'mu_scale': 10.}
    runs = make_tuning(campaign, catalog, NoData())
    assert len(runs) == len({r['run_id'] for r in runs}) == 1164
    assert sum(r['budget'] for r in runs) == 11640
    assert Counter(r['method'] for r in runs) == {
        'smoothing-fw': 252, 'smoothing-row-projection': 252,
        'stochastic-kl': 252, 'native-sampled-gw': 252, 'full-kl': 84, 'pg': 72}
    for method in catalog:
        for candidate in catalog[method]:
            cells = [r for r in runs if r['method'] == method and r['candidate_id'] == candidate['candidate_id']]
            assert {r['seed'] for r in cells} == ({0, 1, 2} if method in STOCHASTIC else {0})
            assert {r['problem_seed'] for r in cells if r['family'] == 'synthetic'} == {1000, 1001, 1002}
            assert {r['noise'] for r in cells if r['family'] == 'synthetic'} == {0., .1}
            assert all(r['initialization'] == 'product' for r in cells)


def test_final_cells_match_initializations_and_do_not_duplicate_deterministic_products():
    campaign, catalog = _design()
    choices = _resolve(_selection(catalog), catalog)
    panels = make_finals(campaign, choices, NoData())
    assert {phase: len(runs) for phase, runs in panels.items()} == {
        'final_product': 253, 'final_randomized': 330, 'small_pg': 30}
    for phase, runs in panels.items():
        assert len({r['run_id'] for r in runs}) == len(runs)
        assert sum(r['budget'] for r in runs) == campaign['raw_solver_budget_seconds'][phase]
    product = panels['final_product']
    for method in ('full-kl', 'fw', 'smoothing-fw-exact'):
        cells = [r for r in product if r['method'] == method]
        assert len(cells) == len({r['instance'] for r in cells}) == 11
    randomized = panels['final_randomized']
    assert all(r['method'] != 'native-sampled-gw' for r in randomized)
    assert all(r['initialization'] == 'randomized' and '--initialization' in r['argv'] for r in randomized)
    for description in {r['instance'] for r in randomized}:
        for method in campaign['final_randomized_methods']:
            cells = [r for r in randomized if r['instance'] == description and r['method'] == method]
            assert {(r['seed'], r['init_seed']) for r in cells} == {(s, 900000 + s) for s in range(100, 105)}
    exact = [r for r in product if r['method'] == 'smoothing-fw-exact'][0]
    assert exact['candidate_id'] == choices['smoothing-fw']['candidate_id']
    assert exact['parameters']['oracle'] == 'exact'


@pytest.mark.parametrize('field', ['campaign_config_sha256', 'tuning_manifest_sha256'])
def test_stale_selection_provenance_is_rejected(field):
    _, catalog = _design()
    selection = _selection(catalog)
    selection[field] = '0' * 64
    with pytest.raises(ValueError, match='hash differs'):
        _resolve(selection, catalog)


def test_unavailable_invalid_or_nonfinite_selection_has_no_fallback():
    _, catalog = _design()
    for replacement in (
        {'status': 'no_valid_setting', 'candidate_id': None, 'score': None},
        {'status': 'selected', 'candidate_id': 'c99', 'score': 1.},
        {'status': 'selected', 'candidate_id': 'c00', 'score': float('inf')},
    ):
        selection = _selection(catalog)
        selection['selections']['smoothing-fw'] = replacement
        with pytest.raises(ValueError):
            _resolve(selection, catalog)


def test_product_normalizer_matches_direct_four_index_loss_and_verifies_hashes(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    A = np.array([[0., .6, .9], [.6, 0., .7], [.9, .7, 0.]], dtype=np.float32)
    B = np.array([[0., .8, .55], [.8, 0., .65], [.55, .65, 0.]], dtype=np.float32)
    a = np.array([.2, .3, .5], dtype=np.float64)
    b = np.array([.3, .4, .3], dtype=np.float64)
    path = data / 'toy.npz'
    np.savez(path, A=A, B=B, a=a, b=b, true_target=np.array(['ignored'], dtype=object))
    description = dict(instance='toy.npz', n=3, m=3)
    manifest = {key + '_sha256': hashlib.sha256(value.tobytes()).hexdigest() for key, value in [('A', A), ('B', B)]}
    path.with_suffix('.json').write_text(json.dumps(manifest))
    actual = CanonicalInstances(tmp_path).metadata(description)
    P = np.outer(a, b)
    K = (A.astype(np.float64)[:, :, None, None] - B.astype(np.float64)[None, None, :, :]) ** 2
    direct = np.einsum('ikjl,ij,kl->', K, P, P)
    assert actual['product_loss_float64'] == pytest.approx(direct, abs=1e-14)
    assert actual['instance_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    manifest['A_sha256'] = '0' * 64
    path.with_suffix('.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='hash mismatch'):
        CanonicalInstances(tmp_path).metadata(description)
