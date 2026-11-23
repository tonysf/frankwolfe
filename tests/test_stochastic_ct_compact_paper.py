"""The short section reports measured values and enforces its page limit."""

from copy import deepcopy
from pathlib import Path
import runpy

import pytest

from paper.experiments.stochastic_ct import compact_paper as paper


HERE = Path(__file__).resolve().parent
REPORT_FIXTURE = runpy.run_path(str(HERE/'test_stochastic_ct_robustness_report.py'))
ABLATION_FIXTURE = runpy.run_path(str(HERE/'test_stochastic_ct_robustness_ablation_text.py'))


def expanded_fixture(root):
    """Explicitly synthetic endpoint evidence; never an experimental outcome."""
    cohort, configs = REPORT_FIXTURE['make_main_fixture'](root)
    summary = paper.evidence.summarize(root, 'fixture')
    ablations = ABLATION_FIXTURE['ablation_fixture']()['studies']
    for cells in ablations.values():
        for cell in cells:
            cell['metrics']['relative_d_distance'] = deepcopy(
                cell['metrics']['fixed_reference_relative_d_distance'])
    summary['studies'].update(ablations)
    summary['stage'] = 'fixture'
    return cohort, configs, summary, REPORT_FIXTURE['selection_fixture']()


@pytest.fixture(scope='module')
def expanded(tmp_path_factory):
    return expanded_fixture(tmp_path_factory.mktemp('compact_paper_fixture'))


def test_expanded_section_keeps_setup_values_and_two_figure_roles(expanded):
    _, _, summary, selection = expanded
    selection = deepcopy(selection)
    selection['global_selection']['sframes_b64']['loss']['parameters'].update(beta_scale=16, gamma0=.2)
    body = paper.expanded_body(summary, selection)
    assert r'g=\iota_D' in body and r'T=\operatorname{Id}' in body
    assert r'\|U\|_{\rm op}\le R_U' in body and r'\widehat G_k=(Z_kV_k,Z_k^\top U_k)' in body
    assert r'(c_\beta,\gamma_0)=(16,0.2)' in body
    assert r'multiplies evaluated loss by \(1.50\)' in body
    assert r'from \(0.20\%\) to \(0.60\%\)' in body
    assert body.count(r'\begin{figure}') == 2 and body.count(r'\begin{table}') == 1
    assert 'Method comparison at $N=1080$' in body
    assert 'Stochastic controls on six slices at $N=1080$' in body
    assert '360 views held out' in body and r'N\in\{120,360,1080,2160\}' in body
    assert 'seed' not in body and '224' not in body and '300-run' not in body


def test_result_prose_changes_when_the_measured_comparison_reverses(expanded):
    _, _, summary, selection = expanded
    summary = deepcopy(summary)
    sf = next(c for c in summary['studies']['main']
              if c['n_train'] == 1080 and c['variant'] == 'sframes_joint')
    sf['metrics']['feasible_loss']['median_across_groups'] = 100.
    summary['studies']['averaging_control'][0]['metrics']['feasible_loss']['median_across_groups'] = .25
    body = paper.expanded_body(summary, selection)
    assert 'higher than deterministic FraMES' in body and 'higher than ProxSARAH' in body
    assert r'multiplies evaluated loss by \(0.25\)' in body


def test_expanded_driver_omits_only_the_uncited_palm_reference():
    old = paper.driver()
    new = paper.driver(include_palm=False)
    assert r'\bibitem{palm}' in old and r'\bibitem{palm}' not in new
    assert len(old.splitlines()) == len(new.splitlines())+1
    for name in ('detect', 'spring', 'proxsarah'):
        assert r'\bibitem{'+name+'}' in new


@pytest.mark.parametrize('pages,log,error', [(3, '', 'exceeds two pages'),
    (2, r'Overfull \hbox', 'overflow'), (2, 'undefined references', 'unresolved')])
def test_compile_gate_rejects_long_or_broken_output(tmp_path, monkeypatch, pages, log, error):
    (tmp_path/'sframes_ct_section.log').write_text(log)
    monkeypatch.setattr(paper.subprocess, 'run', lambda *args, **kw: None)
    monkeypatch.setattr(paper.subprocess, 'check_output', lambda *args, **kw: f'Pages: {pages}\n')
    with pytest.raises(ValueError, match=error):
        paper.compile_report(tmp_path)


@pytest.mark.parametrize('completion', [{'status': 'running', 'publication_overlay': True},
                                     {'status': 'completed'}])
def test_publication_requires_completed_overlay_evidence(tmp_path, completion):
    paper.write(tmp_path/'completion.json', completion)
    with pytest.raises(ValueError, match='completed numerical-evidence overlay'):
        paper.publish_expanded(tmp_path)
