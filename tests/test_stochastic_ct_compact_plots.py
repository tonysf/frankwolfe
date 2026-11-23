"""Compact presentation preserves native summaries and signed SD bands."""

from copy import deepcopy

from matplotlib.axes import Axes
import numpy as np

from paper.experiments.stochastic_ct import compact_plots as plotting


def test_statistics_reuses_exact_group_balanced_native_alignment():
    runs = {}
    for sid, work, loss, distance in [('a', [0, 2, 4], [1., .5, .25], [0., .1, .2]),
                                      ('b', [0, 1, 4], [1., .75, .5], [0., 0., .1])]:
        runs[sid, 1, 360, 'method'] = dict(trace=[dict(iteration=k,
            comparison_views=w, comparison_oracle_calls=w, raw_loss=f, d_distance=d)
            for k, (w, f, d) in enumerate(zip(work, loss, distance))])
    groups = {'a': 'A', 'b': 'B'}
    result = plotting.statistics(runs, ['a', 'b'], groups, 360, {'method': ('Method', '#0072B2')})
    assert [r['weight'] for r in result['methods']['method']['runs']] == [.5, .5]
    for axis, _ in plotting.AXES:
        for metric, _ in plotting.METRICS:
            x, mean, sd = plotting.reference.summarize(runs, ['a', 'b'], groups, 360, 'method', metric, axis)
            rendered = result['methods']['method']['axes'][axis]
            np.testing.assert_array_equal(rendered['grid'], x)
            np.testing.assert_array_equal(rendered['metrics'][metric]['mean'], mean)
            np.testing.assert_array_equal(rendered['metrics'][metric]['standard_deviation'], sd)
    work = result['methods']['method']['axes']['comparison_views']
    assert work['grid'] == [0, 1, 2, 4]
    assert work['metrics']['raw_loss']['mean'] == [1., .875, .625, .375]


def test_renderer_keeps_negative_sd_bounds_and_input_bytes(tmp_path, monkeypatch):
    metrics = {'raw_loss': dict(mean=[1., .5], standard_deviation=[0., .1]),
               'd_distance': dict(mean=[0., .1], standard_deviation=[0., .2])}
    stats = dict(training_views=360, methods={'method': dict(label='Method', color='#0072B2',
        axes={axis: dict(grid=[0, 2], metrics=deepcopy(metrics)) for axis, _ in plotting.AXES})})
    before = deepcopy(stats)
    bands = []
    original = Axes.fill_between
    def capture(self, x, lower, upper, **kwargs):
        bands.append((np.array(lower), np.array(upper), kwargs.get('step')))
        return original(self, x, lower, upper, **kwargs)
    monkeypatch.setattr(Axes, 'fill_between', capture)
    receipt = plotting.render_statistics(stats, tmp_path/'compact')
    assert stats == before
    assert receipt['mean_and_sd_unchanged']
    assert receipt['row_scales']['d_distance']['scale'] == 'symlog'
    assert receipt['row_scales']['d_distance']['lower_band_min'] == -.1
    assert receipt['row_scales']['d_distance']['negative_band_preserved']
    for index, (lower, upper, step) in enumerate(bands):
        source = metrics['raw_loss' if index < 3 else 'd_distance']
        mean, sd = np.array(source['mean']), np.array(source['standard_deviation'])
        np.testing.assert_array_equal(lower, mean-sd)
        np.testing.assert_array_equal(upper, mean+sd)
        assert step == (None if index % 3 == 0 else 'post')
    assert len(bands) == 6
    assert all((tmp_path/name).is_file() for name in ('compact.pdf', 'compact.png', 'compact.json'))
