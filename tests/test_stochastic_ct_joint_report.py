"""Native joint-work accounting and preservation of the existing CT comparison."""

import numpy as np
import pytest

from paper.experiments.stochastic_ct import joint_lbfgsb_report as report


def test_joint_gradient_trials_are_charged_without_factor_repair():
    row = dict(iteration=1, forward_views=1080, backward_views=1080,
               gradient_calls=3, oracle_calls=3, curvature_forward_views=360,
               curvature_backward_views=360, raw_loss=.2, feasible_loss=.2, d_distance=0.)
    initial = dict(row, iteration=0, forward_views=0, backward_views=0,
                   gradient_calls=0, oracle_calls=0, curvature_forward_views=0, curvature_backward_views=0)
    run = report.prepare_run([initial, row], {'initial_factor_frobenius_norm': 2.}, report.METHOD, 'fixture')
    assert run['method'] == report.METHOD
    assert run['trace'][-1]['comparison_views'] == 1080
    assert run['trace'][-1]['comparison_oracle_calls'] == 3
    with pytest.raises(ValueError, match='repaired or became infeasible'):
        report.prepare_run([initial, dict(row, feasible_loss=.21)],
                           {'initial_factor_frobenius_norm': 2.}, report.METHOD, 'fixture')


def test_joint_cohort_preserves_prior_statistics_and_terminal_work():
    runs, slices, groups = report.fixture_runs()
    old = {key: run for key, run in runs.items() if key[3] != report.METHOD}
    previous = report.base.summarize(old, slices, groups)
    current = report.summarize(runs, slices, groups)
    assert current['physical_runs'] == 532 and current['added_runs'] == 28
    for size in ('360', '1080'):
        for method in report.base.VARIANTS:
            assert current['cohorts'][size]['methods'][method] == previous['cohorts'][size]['methods'][method]
    x, mean, sd = report.plotting.summarize(runs, slices, groups, 360, report.METHOD, 'raw_loss', 'iteration')
    work, wmean, wsd = report.plotting.summarize(runs, slices, groups, 360, report.METHOD, 'raw_loss', 'comparison_views')
    assert x.tolist() == [0, 1, 2] and work.tolist() == [0, 360, 720, 1080]
    assert wmean[-1] == wmean[-2]
    assert np.allclose(mean, wmean[:3]) and np.allclose(sd, wsd[:3])
    del runs[next(k for k in runs if k[3] == report.METHOD)]
    with pytest.raises(ValueError, match='532 physical runs'):
        report.summarize(runs, slices, groups)


def test_document_keeps_estimator_explanation_and_two_figures():
    # Only the predecessor passages needed by the editorial transformation;
    # independent of future publication replacing the repository's live PDF.
    source = r'''least squares (NNLS) and multiplicative updates without spectral bounds.
The additional NNLS and multiplicative baselines minimize the same loss
\paragraph{Stochastic gradient estimator.}
The estimator explanation and its averaging recurrence must remain verbatim.
\[ \overline G_k=(1-\alpha_k)\overline G_{k-1}+\alpha_k\widehat G_k. \]
\paragraph{Evaluation.}
The original methods count gradient views; NNLS and
multiplicative updates count every solver forward/adjoint application,
and rescales it radially into its spectral ball if needed. NNLS and
multiplicative updates are evaluated at their actual nonnegative factors,
At \(N=360\), raw losses are placeholder fixture values; differences also reflect the
optimization domain.
\begin{table}[!htbp]old table\end{table}
without a tuned step size. Neither additional baseline has a spectral cap.
\begin{figure}[!htbp]
an accepted inner factor step for NNLS and a factor update for multiplicative
updates; iteration costs differ between methods.
\end{figure}
\begin{figure}[!htbp]
The three projection baselines and both additional nonnegative methods
\end{figure}
'''
    runs, slices, groups = report.fixture_runs()
    result = report.source_text(source, report.summarize(runs, slices, groups), 20)
    assert 'The estimator explanation and its averaging recurrence must remain verbatim.' in result
    assert r'\overline G_k=(1-\alpha_k)\overline G_{k-1}+\alpha_k\widehat G_k.' in result
    assert result.count(r'\begin{figure}') == 2 and result.count(r'\begin{table}') == 1
    assert result.count('Joint L-BFGS-B & $D$') == 2
    assert 'one curvature history with memory 20' in result
    assert 'None of these three\nadditional baselines has a spectral cap.' in result
