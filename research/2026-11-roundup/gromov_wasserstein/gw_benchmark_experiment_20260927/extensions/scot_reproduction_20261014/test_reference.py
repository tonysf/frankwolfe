"""Tiny mathematical/API tests; never runs the 1047-cell optimizer."""
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('scot_reference_under_test', HERE/'reference.py')
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


def author_source():
    return Path(os.environ.get('SCOT_AUTHOR_SOURCE',
                               ROOT/'logs/gw_literature_context_20261008/scotv1.py'))


def tiny_problem():
    A = np.array([[0., .7], [.7, 0.]])
    B = np.array([[0., .3, 1.], [.3, 0., .8], [1., .8, 0.]])
    P = np.array([[.1, 0., .6], [.2, .3, .4]])  # deliberately off marginal and mass 1.6
    return A, B, P, np.array([.3, .7]), np.array([.2, .3, .5])


def test_full_loss_matches_four_index_sum_off_marginals():
    A, B, P, a, b = tiny_problem()
    expected = sum((A[i, k]-B[j, l])**2 * P[i, j]*P[k, l]
                   for i in range(2) for j in range(3) for k in range(2) for l in range(3))
    assert ref.full_loss(A, B, P) == pytest.approx(expected, abs=2e-15)
    assert abs(ref.fixed_marginal_loss(A, B, P, a, b)-expected) > .01


def test_pot_crosscheck_uses_actual_marginals():
    ot = pytest.importorskip('ot')
    A, B, P, a, b = tiny_problem()
    record = ref.evaluate(A, B, P, a, b, ot)
    assert record['independent_loss_agrees']
    assert record['residuals']['mass'] == pytest.approx(1.6)
    const, hA, hB = ot.gromov.init_matrix(A, B, a, b, 'square_loss')
    assert ref.fixed_marginal_loss(A, B, P, a, b) == pytest.approx(float(ot.gromov.gwloss(const, hA, hB, P)))


@pytest.mark.parametrize('zero', [False, True])
def test_rounding_rectangular_sparse_and_zero_mass(zero):
    A, B, P, a, b = tiny_problem()
    if zero:
        P.fill(0)
    before = P.copy()
    Q = ref.round_transport(P, a, b)
    assert np.array_equal(P, before)
    assert (Q >= 0).all()
    np.testing.assert_allclose(Q.sum(1), a, atol=2e-16, rtol=0)
    np.testing.assert_allclose(Q.sum(0), b, atol=2e-16, rtol=0)


@pytest.mark.parametrize('bad', [np.nan, -.1])
def test_invalid_plan_cannot_be_repaired_or_scored(bad):
    A, B, P, a, b = tiny_problem()
    P[0, 0] = bad
    with pytest.raises(ValueError):
        ref.round_transport(P, a, b)
    with pytest.raises(ValueError):
        ref.evaluate(A, B, P, a, b, None)


def test_authors_find_correspondences_omits_solver_overrides():
    """Execute the exact released method body with an observing POT stand-in."""
    source = author_source()
    assert ref.sha256(source) == ref.INPUT_HASHES['scotv1.py']
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SCOT')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'find_correspondences')
    calls = []
    def solve(*args, **kwargs):
        calls.append((args, kwargs))
        return np.ones((2, 2))/4, {'gw_dist': .018}
    namespace = {'np': np, 'ot': SimpleNamespace(gromov=SimpleNamespace(entropic_gromov_wasserstein=solve))}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    model = SimpleNamespace(Cx=np.eye(2), Cy=np.eye(2), p=np.ones(2)/2, q=np.ones(2)/2)
    assert namespace['find_correspondences'](model, e=.001, verbose=False) == .018
    assert model.flag
    assert calls[0][1] == {'loss_fun': 'square_loss', 'epsilon': .001, 'log': True, 'verbose': False}
    assert len(calls[0][0]) == 4


def test_author_preprocessing_matches_direct_sklearn_calls():
    pytest.importorskip('sklearn')
    from sklearn.preprocessing import normalize
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import dijkstra
    source = ref.checked_source(author_source())
    rng = np.random.default_rng(71)
    X, Y = rng.normal(size=(7, 4)), rng.normal(size=(6, 3))
    model, (A, B, a, b) = ref.prepare(source.SCOT, X, Y, k=3)
    for data, actual in [(X, A), (Y, B)]:
        graph = kneighbors_graph(normalize(data, norm='l2', axis=1), 3,
                                 mode='connectivity', metric='correlation', include_self=True)
        D = dijkstra(graph, directed=False, return_predecessors=False)
        diameter = D[np.isfinite(D)].max()
        D[D > diameter] = diameter
        np.testing.assert_array_equal(actual, D/D.max())
    np.testing.assert_array_equal(a, np.ones(7)/7)
    np.testing.assert_array_equal(b, np.ones(6)/6)


def metrics(loss):
    return dict(loss=loss, independent_loss_agrees=True,
                residuals={'both_l1': 2e-16, 'min_entry': 0.})


def test_reproduction_gate_is_explicit_and_preserves_raw_infeasibility():
    initial, raw, rounded = metrics(ref.EXPECTED_START), metrics(.017), metrics(.018)
    raw['residuals']['both_l1'] = .0013
    args = dict(author_flag=True, author_loss_agrees=True, author_loss=.01652)
    assert ref.gate(initial, raw, rounded, **args)['passed']
    rounded['loss'] = .026
    assert not ref.gate(initial, raw, rounded, **args)['passed']
    rounded['loss'], rounded['residuals']['both_l1'] = .018, 1e-5
    assert not ref.gate(initial, raw, rounded, **args)['passed']
    rounded['residuals']['both_l1'] = 2e-16
    raw['independent_loss_agrees'] = False
    assert not ref.gate(initial, raw, rounded, **args)['passed']


def reference_binding_fixture(root):
    """Hash-only verifier fixture; its bytes are not scientific coupling data."""
    directory = root/'prepared/scot1047'
    directory.mkdir(parents=True)
    hashes = {}
    for name in ref.ARRAY_NAMES:
        (directory/name).write_bytes(b'verifier-only fixture '+name.encode())
        hashes[name] = ref.sha256(directory/name)
    manifest = dict(n=1047, m=1047, objective=ref.OBJECTIVE,
                    array_files_sha256=hashes, reference_status='passed')
    ref.write_json(directory/'manifest.json', manifest)
    archive = root/'provenance'
    archive.mkdir()
    source = author_source()
    (archive/'scotv1.py').write_bytes(source.read_bytes())
    for name in ['P_raw.npy', 'P_rounded.npy']:
        (root/name).write_bytes(b'verifier-only fixture '+name.encode())
    initial, raw, rounded = metrics(ref.EXPECTED_START), metrics(.017), metrics(.018)
    result = dict(status='passed', input_file_sha256=ref.INPUT_HASHES, initial=initial, raw=raw,
                  rounded=rounded, author_output=dict(basic_numerical_flag=True, fixed_expression_absolute_difference=0., gw_dist=.01652),
                  gate=ref.gate(initial, raw, rounded, author_flag=True, author_loss_agrees=True, author_loss=.01652),
                  prepared_instance=dict(relative_directory='prepared/scot1047', manifest_sha256=ref.sha256(directory/'manifest.json'), array_files_sha256=hashes),
                  raw_plan_sha256=ref.sha256(root/'P_raw.npy'), rounded_plan_sha256=ref.sha256(root/'P_rounded.npy'),
                  source_file_sha256={'provenance/scotv1.py': ref.sha256(archive/'scotv1.py')})
    ref.write_json(root/'reference.json', result)
    return result


@pytest.mark.parametrize('change', ['array', 'plan', 'manifest', 'gate', 'status'])
def test_verifier_rejects_corrupt_bound_artifacts(tmp_path, change):
    record = reference_binding_fixture(tmp_path)
    assert ref.verify_reference(tmp_path)['status'] == 'passed'
    if change in ('gate', 'status'):
        if change == 'gate':
            record['gate']['rounded_loss_threshold'] = .1
        else:
            record['status'] = 'failed_reproduction_gate'
        ref.write_json(tmp_path/'reference.json', record)
    else:
        path = {'array': 'prepared/scot1047/A.npy', 'plan': 'P_raw.npy',
                'manifest': 'prepared/scot1047/manifest.json'}[change]
        with (tmp_path/path).open('ab') as stream:
            stream.write(b'changed')
    with pytest.raises(ValueError):
        ref.verify_reference(tmp_path)


def test_changed_author_source_rejected_before_import(tmp_path):
    path = tmp_path/'scotv1.py'
    path.write_text('raise RuntimeError("must not execute")')
    with pytest.raises(ValueError, match='source hash'):
        ref.checked_source(path)
