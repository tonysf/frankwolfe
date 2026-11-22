"""Input isolation, view nesting and initialization controls for the new campaign."""

import numpy as np
import pytest

from paper.experiments.stochastic_ct import robustness_inputs as inputs


def test_nested_views_preserve_old_splits_and_training_only_calibration():
    previous = set()
    for count in inputs.TRAINING_VIEWS:
        train, heldout = inputs.view_split(count)
        assert len(train) == len(set(train)) == count
        assert previous <= set(train)
        assert not set(train) & set(heldout)
        assert set(inputs.CALIBRATION_VIEWS) <= set(train)
        assert np.array_equal(heldout, np.arange(9, 3600, 10))
        if count in (360, 1080):
            old_train, old_heldout = inputs.base_problem.view_split(count)
            np.testing.assert_array_equal(train, old_train)
            np.testing.assert_array_equal(heldout, old_heldout)
        previous = set(train)
    with pytest.raises(ValueError):
        inputs.view_split(90)


def test_metadata_only_confirmation_rule_and_pilot_exclusions():
    old = [601, 1201, 3041, 3150, 3381, 3611, 3791, 4131, 4300, 4531,
           4800, 4931, 5581, 5611, 5731, 5761, 5831, 5861]
    selected = inputs.select_confirmation([f'slice{i:05d}' for i in old])
    assert selected == {'slice00900': 'Mix1', 'slice02402': 'Mix2',
                        'slice04012': 'Mix3', 'slice05595': 'Almond',
                        'slice05745': 'Walnut', 'slice05845': 'LavaStone'}
    for sid, group in selected.items():
        low, high, separation = inputs.GROUP_RANGES[group]
        number = int(sid[5:])
        assert low <= number <= high
        assert min(abs(number - i) for i in old) >= separation


class SimpleProblem:
    train_ids = inputs.CALIBRATION_VIEWS

    def __init__(self, target):
        self.y = np.zeros((3600, target.size))
        self.y[self.train_ids] = target.ravel()
        self.calls = []

    def forward(self, image, views):
        self.calls.append(np.array(views, copy=True))
        return np.tile(image.ravel(), (len(views), 1))


@pytest.mark.parametrize('family', ['perturbed', 'random'])
def test_alternative_starts_reproducible_distinct_and_share_fixed_domain(family):
    rng = np.random.default_rng(2)
    U, V = rng.random((7, 3)), rng.random((7, 3))
    originals = U.copy(), V.copy()
    caps = 2 * np.array([np.linalg.norm(U, 2), np.linalg.norm(V, 2)])
    p = SimpleProblem(U @ V.T)
    first = inputs.alternative_start(p, U, V, caps, family, 123)
    second = inputs.alternative_start(p, U, V, caps, family, 123)
    third = inputs.alternative_start(p, U, V, caps, family, 124)
    for k in (0, 1):
        np.testing.assert_array_equal(first[k], second[k])
        np.testing.assert_array_equal((U, V)[k], originals[k])
        assert np.all(first[k] >= 0)
        assert np.linalg.norm(first[k], 2) <= caps[k] / 2 * (1 + 1e-12)
        assert not np.allclose(first[k], (U, V)[k])
        assert not np.allclose(first[k], third[k])
    assert len(p.calls) == 3
    assert all(np.array_equal(v, inputs.CALIBRATION_VIEWS) for v in p.calls)
    assert first[2]['reference_curvature_scales_retained']


def test_amplitude_fit_respects_the_fixed_radius_instead_of_changing_it():
    U, V = np.ones((4, 2)), np.ones((4, 2))
    caps = np.array([2*np.linalg.norm(U, 2), 2*np.linalg.norm(V, 2)])
    saved_caps = caps.copy()
    p = SimpleProblem(1000 * U @ V.T)
    a, b, details = inputs.alternative_start(p, U, V, caps, 'random', 91)
    np.testing.assert_array_equal(caps, saved_caps)
    assert details['amplitude_limited_by_reference_spectral_bounds']
    assert np.isclose(np.linalg.norm(a, 2), caps[0] / 2)
    assert np.isclose(np.linalg.norm(b, 2), caps[1] / 2)


def test_alternative_start_rejects_nontraining_calibration():
    p = SimpleProblem(np.ones((3, 3)))
    p.train_ids = np.arange(0, 3600, 40)
    with pytest.raises(ValueError, match='nontraining'):
        inputs.alternative_start(p, np.ones((3, 2)), np.ones((3, 2)), [8, 8], 'random', 3)


def test_new_problem_reuses_original_implementation_without_global_mutation(monkeypatch):
    calls = []
    original_split = inputs.base_problem.view_split

    def fake_init(self, path, size, n):
        calls.append((path, size, n))
        self.train_ids, self.validation_ids = original_split(n)

    monkeypatch.setattr(inputs.base_problem.CTProblem, '__init__', fake_init)
    p = inputs.NewCTProblem('unused.npz', 512, 2160)
    assert calls == [('unused.npz', 512, 360)]
    assert len(p.train_ids) == 2160
    assert inputs.base_problem.view_split is original_split
    assert inputs.NewCTProblem.forward is inputs.base_problem.CTProblem.forward
    assert inputs.NewCTProblem.backward is inputs.base_problem.CTProblem.backward


class ReceiptProblem(SimpleProblem):
    """Tiny operator exercising the real NPZ and JSON preparation boundary."""
    instances = []

    def __init__(self, data, size, n_train):
        self.U = np.arange(1.,9.).reshape(4,2)/10
        self.V = self.U[::-1].copy()
        super().__init__(1000*self.U@self.V.T)
        self.calibration_work = dict(forward_calls=0,backward_calls=0,forward_views=0,backward_views=0)
        self.closed = False
        self.instances.append(self)

    def calibration(self, rank, seed):
        return self.U.copy(),self.V.copy(),np.array([99.,99.]),np.array([1.,2.])

    def forward(self, image, views):
        self.calibration_work['forward_calls'] += 1
        self.calibration_work['forward_views'] += len(views)
        return super().forward(image,views)

    def close(self):
        self.closed = True


@pytest.mark.parametrize('family',['perturbed','random'])
def test_prepare_calibration_writes_native_json_scalars_and_reuses_receipt(tmp_path,monkeypatch,family):
    ReceiptProblem.instances=[]
    monkeypatch.setattr(inputs,'_CalibrationProblem',ReceiptProblem)
    data=tmp_path/'data.npz'
    data.write_bytes(b'fixed fake measurements')
    standard=inputs._prepare_calibration(tmp_path,data,'slice00001',2,'standard',201)
    output=inputs._prepare_calibration(tmp_path,data,'slice00001',2,family,202,standard)
    receipt=inputs.read(output.with_suffix('.json'))
    assert type(receipt['amplitude_limited_by_reference_spectral_bounds']) is bool
    assert receipt['amplitude_limited_by_reference_spectral_bounds'] is True
    assert type(receipt['fitted_amplitude']) is float
    assert receipt['calibration_work']['forward_views']==120
    assert receipt['reference_calibration_sha256']==inputs.sha256(standard)
    assert receipt['preparation_source_sha256']==inputs.sha256(inputs.__file__)
    assert receipt['calibration_sha256']==inputs.sha256(output)
    with np.load(standard) as reference,np.load(output) as alternative:
        np.testing.assert_array_equal(alternative['caps'],reference['caps'])
        np.testing.assert_array_equal(alternative['curvature'],reference['curvature'])
    assert inputs._prepare_calibration(tmp_path,data,'slice00001',2,family,202,standard)==output
    assert len(ReceiptProblem.instances)==2 and all(p.closed for p in ReceiptProblem.instances)


def test_resume_npz_without_receipt_recomputes_and_retains_provenance_guard(tmp_path,monkeypatch):
    ReceiptProblem.instances=[]
    monkeypatch.setattr(inputs,'_CalibrationProblem',ReceiptProblem)
    data=tmp_path/'data.npz'
    data.write_bytes(b'fixed fake measurements')
    standard=inputs._prepare_calibration(tmp_path,data,'slice00001',2,'standard',201)
    write=inputs.write
    def fail_receipt(path,value):
        raise TypeError('simulated receipt serialization failure')
    monkeypatch.setattr(inputs,'write',fail_receipt)
    with pytest.raises(TypeError,match='receipt serialization'):
        inputs._prepare_calibration(tmp_path,data,'slice00001',2,'random',202,standard)
    output=tmp_path/'calibration/slice00001_rank2_random_seed202.npz'
    receipt=output.with_suffix('.json')
    assert output.exists() and not receipt.exists() and ReceiptProblem.instances[-1].closed
    with np.load(output) as saved:
        original={name:saved[name].copy() for name in saved.files}
    monkeypatch.setattr(inputs,'write',write)
    assert inputs._prepare_calibration(tmp_path,data,'slice00001',2,'random',202,standard)==output
    assert receipt.exists() and len(ReceiptProblem.instances)==3
    with np.load(output) as saved:
        for name,value in original.items():
            np.testing.assert_array_equal(saved[name],value)
    record=inputs.read(receipt)
    record['preparation_source_sha256']='0'*64
    inputs.write(receipt,record)
    with pytest.raises(ValueError,match='provenance changed'):
        inputs._prepare_calibration(tmp_path,data,'slice00001',2,'random',202,standard)
    assert len(ReceiptProblem.instances)==3
