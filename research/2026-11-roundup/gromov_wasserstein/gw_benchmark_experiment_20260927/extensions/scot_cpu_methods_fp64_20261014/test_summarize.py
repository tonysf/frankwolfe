"""Small report-only checks using the preserved 200-update metadata.

No arrays, optimizers, NumPy or Torch are loaded. Run with:
python -m unittest discover -s extensions/scot_cpu_methods_fp64_20261014 -p test_summarize.py
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('scot_methods_report',Path(__file__).with_name('summarize.py'))
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


class ReportChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        folder = ROOT/'experiments/scot_cpu_methods_fp64_20261014'
        cls.pilots = {name:json.loads((folder/name/'metadata.json').read_text())
                      for name in ('pilot_smd','pilot_msbpg')}

    def test_real_pilot_certificates_and_paper_parameters(self):
        for name, metadata in self.pilots.items():
            with self.subTest(pilot=name):
                self.assertEqual(REPORT.inner_certificates(metadata)['accepted_solves'],200)
                self.assertTrue(REPORT.paper_parameters(metadata))
                self.assertEqual(REPORT.output_selection(metadata)['index'],153)

    def test_guarded_gap_cannot_be_hidden_by_small_raw_gap(self):
        metadata = copy.deepcopy(self.pilots['pilot_smd'])
        metadata['inner_trace'][0]['gap_roundoff_tolerance'] = 1e-5
        with self.assertRaisesRegex(ValueError,'guarded numerical criteria'):
            REPORT.inner_certificates(metadata)

    def test_reported_maximum_cannot_hide_bad_step(self):
        metadata = copy.deepcopy(self.pilots['pilot_msbpg'])
        metadata['inner_trace'][75]['marginal_l1'] = 2e-7
        with self.assertRaisesRegex(ValueError,'guarded numerical criteria'):
            REPORT.inner_certificates(metadata)

    def test_momentum_uses_current_beta_not_next_beta(self):
        metadata = copy.deepcopy(self.pilots['pilot_msbpg'])
        metadata['inner_trace'][2]['beta'] = metadata['inner_trace'][2]['beta_next']
        with self.assertRaisesRegex(ValueError,'momentum weight'):
            REPORT.paper_parameters(metadata)

    def test_smd_step_change_is_not_accepted(self):
        metadata = copy.deepcopy(self.pilots['pilot_smd'])
        metadata['inner_trace'][20]['eta'] *= 2
        with self.assertRaisesRegex(ValueError,'SMD step'):
            REPORT.paper_parameters(metadata)

    def test_terminal_iterate_is_not_random_paper_output(self):
        metadata = copy.deepcopy(self.pilots['pilot_smd'])
        metadata['output_selection']['index'] = 200
        with self.assertRaisesRegex(ValueError,'conflated'):
            REPORT.output_selection(metadata)

    def test_real_pilot_recorded_curve_feasibility(self):
        for metadata in self.pilots.values():
            audit = REPORT.recorded_curve_feasibility(metadata['trace'])
            self.assertEqual(audit['observed_count'],len(metadata['trace']))
            self.assertLessEqual(audit['maximum_recorded_marginal_l1'],REPORT.STRICT_FEASIBILITY)

    def test_bad_intermediate_rounded_residual_is_rejected(self):
        for field, value in [('both_l1',1e-10),('min_entry',-1e-30),('col_l2',float('nan'))]:
            with self.subTest(field=field):
                trace = copy.deepcopy(self.pilots['pilot_smd']['trace'])
                trace[2]['rounded_residuals'][field] = value
                with self.assertRaisesRegex(ValueError,'current-rounded checkpoint'):
                    REPORT.recorded_curve_feasibility(trace)

    def test_independent_winner_reversal_requires_review(self):
        records, selected = [], {}
        for method in REPORT.METHODS[1:]:
            for k in range(4):
                candidate = dict(id=f'{method}_{k}')
                records.append(dict(method_key=method,phase='tune',candidate=candidate,
                    independent=dict(last_rounded=dict(loss=float(k+1),pot_actual_marginals_loss=float(k+1))),
                    recorded_float64=dict(last_rounded=dict(loss=float(k+1)))))
            selected[method] = dict(id=f'{method}_0')
        self.assertEqual(len(REPORT.independent_selection(records,selected)),2)
        records[1]['independent']['last_rounded']['loss'] = .5
        with self.assertRaisesRegex(ValueError,'without automatic reselection'):
            REPORT.independent_selection(records,selected)

    def test_missing_seed_never_gets_three_seed_band(self):
        records = []
        for method in REPORT.METHODS:
            for seed in REPORT.SEEDS:
                records.append(dict(method_key=method,phase='confirm',seed=seed,
                    observations=[dict(iteration=k,rounded_loss=.07-k*1e-6) for k in REPORT.CHECKPOINTS]))
        self.assertEqual(len(REPORT.curves(records)),3*len(REPORT.CHECKPOINTS))
        records.pop()
        with self.assertRaisesRegex(ValueError,'Missing fresh seed'):
            REPORT.curves(records)


if __name__ == '__main__':
    unittest.main()
