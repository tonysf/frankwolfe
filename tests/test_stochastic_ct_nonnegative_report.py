"""New-domain evaluation and native-work aggregation for the CT comparison."""

from pathlib import Path

import numpy as np
import pytest

from paper.experiments.stochastic_ct import nonnegative_report as report
from paper.experiments.stochastic_ct import nonnegative_plots as plotting


def test_new_domain_uses_actual_factors_and_preserves_total_work():
    row = dict(iteration=1, forward_views=720, backward_views=1080,
               gradient_calls=1, oracle_calls=3, raw_loss=.2, feasible_loss=.2, d_distance=0.)
    original = dict(row)
    converted = report.comparison_row(row, "mu")
    assert converted["comparison_views"] == 1080
    assert converted["comparison_oracle_calls"] == 3
    assert row == original
    with pytest.raises(ValueError, match="repaired or became infeasible"):
        report.comparison_row(dict(row, feasible_loss=.21), "anls")
    old = report.comparison_row(dict(row, curvature_forward_views=360, curvature_backward_views=720), "palm")
    assert old["comparison_views"] == 360
    assert old["comparison_oracle_calls"] == 1
    assert old["forward_views"] == 720 and old["backward_views"] == 1080


def test_terminal_work_is_kept_without_fabricated_iterations():
    runs, slices, groups = report.fixture_runs()
    x, mean, sd = plotting.summarize(runs, slices, groups, 360, "anls", "raw_loss", "iteration")
    assert x.tolist() == [0, 1, 2]
    cost, cost_mean, cost_sd = plotting.summarize(runs, slices, groups, 360, "anls", "raw_loss", "comparison_views")
    assert cost.tolist() == [0, 360, 720, 1080]
    assert cost_mean[-1] == cost_mean[-2]
    assert np.allclose(mean, cost_mean[:3])
    assert np.allclose(sd, cost_sd[:3])
    chosen, weights = plotting.select_and_weight(runs, slices, groups, 360, "anls")
    assert len(chosen) == 14 and np.isclose(weights.sum(), 1.)
    for group in set(groups.values()):
        assert np.isclose(sum(w for (key, _), w in zip(chosen, weights) if groups[key[0]] == group), 1 / 6)


def test_eight_arm_summary_and_paper_keep_two_figures():
    runs, slices, groups = report.fixture_runs()
    summary = report.summarize(runs, slices, groups)
    assert summary["physical_runs"] == 504 and summary["added_runs"] == 56
    for size in ("360", "1080"):
        assert set(summary["cohorts"][size]["methods"]) == set(report.VARIANTS)
        assert summary["cohorts"][size]["methods"]["mu"]["relative_d_distance"]["median_across_groups"] == 0
    repository = Path(__file__).resolve().parents[1]
    predecessor = repository / "results/stochastic_matrix_factorization/ct/spectral_balls/expanded/reports/paper_section/revised/ct_experiments.tex"
    source = predecessor.read_text()
    rendered = report.source_text(source, summary, 10)
    assert rendered.count(r"\begin{figure}") == 2
    assert rendered.count(r"\begin{table}") == 1
    assert "Domain" in rendered and r"$C\cap D$" in rendered and "$D$" in rendered
    assert "All three baselines project each factor" in rendered
    assert "Neither additional baseline has a spectral cap" in rendered
    # Existing SFRAMES/ProxSARAH numerical comparisons remain unchanged.
    assert r"\(13.4\%\) lower" in rendered and r"\(21.2\%\) lower" in rendered
    del runs[next(iter(runs))]
    with pytest.raises(ValueError, match="504 physical runs"):
        report.summarize(runs, slices, groups)
