"""Protocol tests without ASTRA, CT data, or production campaign writes."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / 'paper/experiments/stochastic_ct/nonnegative_campaign.py'
SPEC = importlib.util.spec_from_file_location('nonnegative_campaign_tests', SOURCE)
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def cohort_inputs():
    slices = [f'slice{i:05d}' for i in range(14)]
    development = ['slice99998', 'slice99999']
    cohort = dict(evaluation_slices=slices, development_slices=development,
                  evaluation_slice_groups={sid: str(k % 6) for k, sid in enumerate(slices)})
    inputs = {sid: dict(data_file=f'/data/{sid}.npz', data_sha256='data',
              calibration_file=f'/calibration/{sid}.npz', calibration_sha256='calibration',
              calibration_seed=201 if sid in slices else 110) for sid in slices + development}
    return cohort, inputs


def row(iteration, calls, method='mu', kind='accepted_update', terminal=False):
    rhs = int(calls > 0 and method == 'mu')
    values = dict(iteration=iteration, observation_type=kind,
                  forward_calls=calls, backward_calls=calls+rhs,
                  forward_views=360*calls, backward_views=360*(calls+rhs),
                  gradient_calls=calls, normal_operator_calls=calls if method == 'mu' else 0,
                  rhs_backward_calls=rhs, rhs_backward_views=360*rhs,
                  oracle_calls=calls+rhs, completed_blocks=iteration if method == 'mu' else 0,
                  outer_sweeps=iteration//2 if method == 'mu' else 0,
                  raw_loss=1/(iteration+1), feasible_loss=1/(iteration+1),
                  validation_loss=2/(iteration+1), d_distance=0.0,
                  factor_u_sha256=f'U{iteration}', factor_v_sha256=f'V{iteration}')
    event = dict(values)
    values['last_step'] = event
    return values


def mu_trace():
    return [row(0, 0, kind='initial'), row(1, 1), row(2, 2), row(2, 2, kind='budget_stop')]


class PlanTests(unittest.TestCase):
    def test_evaluation_is_paired_without_duplicate_seed_runs(self):
        cohort, inputs = cohort_inputs()
        params = {'anls': dict(campaign.ANLS_PARAMETERS, inner_maxiter=10), 'mu': {}}
        plan = campaign.make_plan(cohort, inputs, params)
        self.assertEqual(plan, campaign.make_plan(cohort, inputs, params))
        self.assertEqual(len(plan['runs']), 56)
        self.assertEqual(sorted(plan['worker_run_counts'].values()), [14, 14, 14, 14])
        for cfg in plan['runs']:
            self.assertNotIn('seed', cfg)
            self.assertEqual(cfg['calibration_seed'], 201)
            self.assertEqual(cfg['view_budget'], 69120)
            self.assertNotIn('caps', cfg)

    def test_development_is_complete_and_disjoint(self):
        cohort, inputs = cohort_inputs()
        plan = campaign.make_development_plan(cohort, inputs)
        self.assertEqual(len(plan['runs']), 12)
        self.assertEqual({c['parameters']['inner_maxiter'] for c in plan['runs']}, {5, 10, 20})
        self.assertEqual({c['calibration_seed'] for c in plan['runs']}, {110})
        self.assertTrue({c['slice_id'] for c in plan['runs']}.isdisjoint(cohort['evaluation_slices']))
        self.assertEqual({c['view_budget'] for c in plan['runs']}, {34560})

    def test_selection_uses_training_only_and_breaks_ties_conservatively(self):
        cohort, inputs = cohort_inputs()
        plan = campaign.make_development_plan(cohort, inputs)
        cases = [dict(status='completed', run_id=c['run_id'], n_train=c['n_train'],
                      relative_loss=.2 if c['parameters']['inner_maxiter'] in (5, 10) else .3,
                      validation_loss=0 if c['parameters']['inner_maxiter'] == 20 else 10)
                 for c in plan['runs']]
        selection = campaign.select_parameters(cases, plan)
        self.assertEqual(selection['selected_inner_maxiter'], 5)
        with self.assertRaisesRegex(ValueError, 'all development'):
            campaign.select_parameters(cases[:-1], plan)
        cases[0]['status'] = 'failed'
        with self.assertRaisesRegex(ValueError, 'failed'):
            campaign.select_parameters(cases, plan)

    def test_frozen_roundtrip_handles_json_integer_keys_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'selection.json'
            campaign.frozen(path, {360: 1.0})
            campaign.frozen(path, {360: 1.0})
            with self.assertRaisesRegex(ValueError, 'Frozen input'):
                campaign.frozen(path, {360: 2.0})


class TraceTests(unittest.TestCase):
    def test_mu_rhs_cost_is_asymmetric_and_final_repeated_state_is_allowed(self):
        cfg = dict(method='mu', n_train=360, view_budget=1080)
        self.assertTrue(campaign.validate_trace(mu_trace(), cfg))
        self.assertEqual(mu_trace()[-1]['forward_views'], 720)
        self.assertEqual(mu_trace()[-1]['backward_views'], 1080)

    def test_missing_rhs_or_inner_work_is_rejected(self):
        cfg = dict(method='mu', n_train=360, view_budget=1080)
        trace = mu_trace()
        trace[-1]['oracle_calls'] -= 1
        with self.assertRaisesRegex(ValueError, 'solver counter'):
            campaign.validate_trace(trace, cfg)
        trace = mu_trace()
        trace[1]['rhs_backward_calls'] = 0
        with self.assertRaises(ValueError):
            campaign.validate_trace(trace, cfg)

    def test_terminal_unaccepted_work_retains_factor_state(self):
        cfg = dict(method='anls', n_train=360, view_budget=1080)
        trace = [row(0, 0, 'anls', 'initial'), row(1, 2, 'anls'), row(1, 3, 'anls', 'budget_stop')]
        self.assertTrue(campaign.validate_trace(trace, cfg))
        trace[-1]['factor_u_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'changed factors'):
            campaign.validate_trace(trace, cfg)

    def test_missing_state_and_overbudget_calls_are_rejected(self):
        cfg = dict(method='mu', n_train=360, view_budget=1080)
        with self.assertRaisesRegex(ValueError, 'Missing native'):
            campaign.validate_trace([mu_trace()[0], *mu_trace()[2:]], cfg)
        with self.assertRaisesRegex(ValueError, 'exceeded'):
            campaign.validate_trace(mu_trace(), dict(cfg, view_budget=720))


class ResultTests(unittest.TestCase):
    def fixture(self, root):
        cfg = dict(run_id='fixture_mu', method='mu', slice_id='slice00001', n_train=360,
                   calibration_seed=201, view_budget=1080, data_sha256='data',
                   calibration_sha256='calibration', parameters={})
        entry = dict(initial_factor_sha256={'U': 'U0', 'V': 'V0'},
                     historical_reference_caps=[1, 2], initial_factor_frobenius_norm=1.0)
        protocol = dict(numerical_sources={'ct_problem.py': 'source'}, inputs={cfg['slice_id']: entry})
        campaign.write(root / 'protocol.json', protocol)
        meta = dict(config=cfg, cpu=0, source_sha256=protocol['numerical_sources'], geometry='nonnegative',
                    numpy='2.5.4', scipy='1.18.1',
                    spectral_caps_are_constraints=False, deterministic=True,
                    data_sha256='data', calibration_sha256='calibration',
                    initial_u_sha256='U0', initial_v_sha256='V0', reference_spectral_caps=[1, 2],
                    protocol_sha256=campaign.digest(root / 'protocol.json'), optimizer={},
                    train_view_ids=list(range(0, 3600, 10)), validation_view_ids=list(range(9, 3600, 10)))
        trace = mu_trace()
        result = dict(status='completed', termination='budget_stop', metadata=meta,
                      initial=trace[0], final=trace[-1], iteration=2, observations=4)
        folder = root / 'runs' / cfg['run_id']
        for name, content in [('metadata.json', meta), ('result.json', result), ('trace.json', trace)]:
            campaign.write(folder / name, content)
        return cfg, dict(groups={cfg['slice_id']: 'fixture'}), protocol, meta, result

    def test_valid_result_retains_native_endpoint_and_initialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg, plan, protocol, _, _ = self.fixture(root)
            checked = campaign.validate_result(cfg, 0, plan, protocol, root)
            self.assertEqual(checked['endpoint'], mu_trace()[-1])
            self.assertEqual(checked['forward_view_slack'], 360)
            self.assertEqual(checked['backward_view_slack'], 0)
            self.assertEqual(checked['initial_factor_sha256'], {'U': 'U0', 'V': 'V0'})

    def test_spectral_domain_and_early_uncertified_stops_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg, plan, protocol, meta, result = self.fixture(root)
            folder = root / 'runs' / cfg['run_id']
            meta['spectral_caps_are_constraints'] = True
            campaign.write(folder / 'metadata.json', meta)
            with self.assertRaisesRegex(ValueError, 'constraint'):
                campaign.validate_result(cfg, 0, plan, protocol, root)
            cfg, plan, protocol, meta, result = self.fixture(root)
            result['termination'] = 'stalled'
            campaign.write(folder / 'result.json', result)
            with self.assertRaisesRegex(ValueError, 'early termination'):
                campaign.validate_result(cfg, 0, plan, protocol, root)


if __name__ == '__main__':
    unittest.main()
