"""The compact paper states measured, matched ablation outcomes."""

from copy import deepcopy

import pytest

from paper.experiments.stochastic_ct import robustness_report as report


def ablation_fixture():
    def cell(variant, loss, violation, *, n=1080, radius=2, batch=64, runs=30, fresh=False):
        slices = [f'fresh{i}' if fresh else f'panel{i}' for i in range(6)]
        def metric(value):
            return dict(median_across_groups=value,by_slice={sid:value for sid in slices},
                        by_group={f'group{i}':value for i in range(6)})
        return dict(variant=variant,n_train=n,rank=64,radius_scale=radius,initialization='standard',
            parameters=dict(batch=batch,beta_scale=4.,gamma0=.1,rho0=1.),physical_runs=runs,
            metrics=dict(feasible_loss=metric(loss),fixed_reference_relative_d_distance=metric(violation)))
    studies = dict(
        batch=[cell('sframes_b16',.8,.001,batch=16),cell('sframes_b64',1.,.002),
               cell('sframes_b256',1.2,.003,batch=256),cell('sframes_full',2.,.004,batch='full',runs=6)],
        averaging_control=[cell('sframes_noavg_paired',1.5,.006)],
        fresh_confirmation=[cell('sframes_joint',.5,.003,n=360,fresh=True),
            cell('prox_sarah',1.,0.,n=360,fresh=True),cell('sframes_joint',3.,.005,fresh=True),
            cell('prox_sarah',2.,0.,fresh=True)],
        radius=[cell('sframes_joint',1.4,.001,radius=1,runs=18),
                cell('sframes_joint',1.,.002,radius=2,runs=18),
                cell('sframes_joint',.7,.008,radius=4,runs=18)])
    return dict(stage='completed',studies=studies)


def test_ablation_text_reports_values_without_predetermining_winners():
    text=report.ablation_results(ablation_fixture())
    assert r'ratios \(0.8,1,1.2,2\)' in text
    assert r'loss ratio \(1.5\)' in text
    assert r'violation \(0.2\%\to 0.6\%\)' in text
    assert r'ratios are \(0.5,1.5\)' in text
    assert r'loss ratios \(1.4,1,0.7\)' in text
    assert r'violations \(0.1,0.2,0.8\%\)' in text
    assert 'separately tuned' in text and 'within the same acquisition groups' in text
    assert 80 <= len(text.split()) <= 120
    assert 'better' not in text and 'improves' not in text


def test_reversing_averaging_result_changes_reported_ratio():
    summary=ablation_fixture()
    summary['studies']['averaging_control'][0]['metrics']['feasible_loss']['median_across_groups']=.25
    text=report.ablation_results(summary)
    assert r'loss ratio \(0.25\)' in text
    assert r'loss ratio \(1.5\)' not in text


@pytest.mark.parametrize('mutation,match',[
    (lambda s:s['studies']['averaging_control'][0]['parameters'].update(gamma0=.2),'parameters'),
    (lambda s:s['studies']['averaging_control'][0].update(physical_runs=18),'repetition'),
    (lambda s:s['studies']['fresh_confirmation'][0]['metrics']['feasible_loss']['by_slice'].pop('fresh0'),'cohorts'),
    (lambda s:s['studies']['radius'][0]['parameters'].update(beta_scale=16.),'parameters'),
    (lambda s:s['studies']['batch'].append(deepcopy(s['studies']['batch'][0])),'ambiguous'),
    (lambda s:s['studies'].pop('radius'),'Incomplete'),
])
def test_unmatched_or_incomplete_evidence_cannot_be_summarized(mutation,match):
    summary=ablation_fixture()
    mutation(summary)
    with pytest.raises(ValueError,match=match):
        report.ablation_results(summary)


def test_only_explicit_fixture_may_omit_all_ablation_results():
    assert report.ablation_results(dict(stage='fixture',studies={'main':[]}))==''
    with pytest.raises(ValueError,match='Incomplete'):
        report.ablation_results(dict(stage='completed',studies={'main':[]}))
