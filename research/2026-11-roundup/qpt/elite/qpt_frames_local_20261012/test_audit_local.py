"""No optimization: independent auditor arithmetic and failure-path tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

import audit_local as audit


class IndependentAuditTests(unittest.TestCase):
    def test_design_is_physical(self):
        probes, effects = audit.dense_design()
        np.testing.assert_allclose(np.sum(abs(probes)**2,axis=1),1)
        np.testing.assert_allclose(effects.sum(axis=0),np.eye(4),atol=1e-15)
        self.assertGreaterEqual(np.min(np.linalg.eigvalsh(effects)),-1e-15)

    def test_identity_probabilities(self):
        truth = np.eye(4,dtype=complex).reshape(16,1)
        values = audit.born_values(truth).reshape(16,8)
        np.testing.assert_allclose(values.sum(axis=1),1)
        self.assertGreaterEqual(values.min(),-1e-15)
        self.assertAlmostEqual(values[0,0],.1)
        self.assertAlmostEqual(values[1,0],0)

    def test_truth_metrics(self):
        truth = audit.independent_truth(0)
        metric = audit.independent_metrics(truth,truth,audit.born_values(truth),5e-5)
        self.assertLess(metric['normalized_tp_violation'],1e-14)
        self.assertEqual(metric['rmse'],0)
        self.assertAlmostEqual(metric['process_fidelity'],1)
        self.assertTrue(audit.gate_and_deficit(metric)[0])
        self.assertFalse(audit.gate_and_deficit(metric,'failed')[0])

    def test_scaled_truth_tp_not_fidelity(self):
        truth = audit.independent_truth(1)
        metric = audit.independent_metrics(.9*truth,truth,audit.born_values(truth),5e-5)
        self.assertAlmostEqual(metric['process_fidelity'],1)
        self.assertAlmostEqual(metric['normalized_tp_violation'],.19)
        self.assertFalse(audit.gate_and_deficit(metric)[0])

    def test_initializer_normalization(self):
        initial = audit.independent_initializer(200000)
        self.assertEqual(initial.shape,(16,1))
        self.assertAlmostEqual(np.linalg.norm(initial),2)

    def test_persistent_not_first_gate(self):
        self.assertEqual(audit.gate_crossings(np.array([0,10,20,30]),[False,True,False,True]),(10,30))
        self.assertEqual(audit.gate_crossings(np.array([0,10]),[True,False]),(0,None))
        self.assertEqual(audit.gate_crossings(np.array([0,10]),[False,False]),(None,None))

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            audit.close(np.nan,np.nan,'must reject')

    def test_plan_split_and_duplicate(self):
        plan = json.loads((audit.ROOT/'plan_pilot.json').read_text())
        audit.check_plan(plan)
        bad = copy.deepcopy(plan)
        bad['trials'][0]['channel'] = 1000
        with self.assertRaisesRegex(ValueError,'forbidden'):
            audit.check_plan(bad)
        bad['trials'][0]['channel'] = 2
        with self.assertRaisesRegex(ValueError,'leaks'):
            audit.check_plan(bad)
        bad = copy.deepcopy(plan)
        bad['trials'].append(copy.deepcopy(bad['trials'][0]))
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            audit.check_plan(bad)

    def test_missing_archive_is_not_complete(self):
        plan_path = audit.ROOT/'plan_pilot.json'
        plan = json.loads(plan_path.read_text())
        with tempfile.TemporaryDirectory(prefix='frames-audit-test-') as temp:
            result = audit.check_trial(Path(temp),plan_path,plan,plan['trials'][0])
        self.assertFalse(result['valid'])
        self.assertFalse(result['archive_complete'])
        self.assertEqual(len(result['missing']),4)

    def _long_plan(self):
        path = audit.ROOT/'plan_long.json'
        if not path.exists():
            self.skipTest('Recorded long-stage selection is not available yet')
        return path,json.loads(path.read_text())

    def test_actual_pinned_selection_schema(self):
        path,plan = self._long_plan()
        chain = audit.check_selection_chain(path,plan)
        self.assertEqual(len(chain),1)
        self.assertEqual(chain[0]['parent_phase'],'screen')
        self.assertEqual(chain[0]['selected_config_ids'],plan['selected_config_ids'])

    def test_selection_checksum_mutation_rejected(self):
        path,plan = self._long_plan()
        plan['selection']['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):
            audit.check_selection_chain(path,plan)

    def test_selection_seed_parameter_budget_mutations_rejected(self):
        path,original = self._long_plan()
        for field,value in (('channel',2),('alpha0',.1),('steps',800000),('sampling_replicate',1)):
            with self.subTest(field=field):
                plan = copy.deepcopy(original)
                plan['trials'][0][field] = value
                with self.assertRaisesRegex(ValueError,'Promoted trial'):
                    audit.check_selection_chain(path,plan)

    def test_selection_winner_mutation_rejected(self):
        path,plan = self._long_plan()
        plan['selected_config_ids'] = ['penalty-a0-p0']
        with self.assertRaisesRegex(ValueError,'declared ranking rule'):
            audit.check_selection_chain(path,plan)

    def test_parent_file_tampering_rejected(self):
        path,plan = self._long_plan()
        real_digest = audit.digest
        def altered_digest(target):
            target = Path(target)
            if target.name == 'summary.json' and target.parent.parent.name == 'screen':
                return '0'*64
            return real_digest(target)
        with mock.patch.object(audit,'digest',side_effect=altered_digest):
            with self.assertRaisesRegex(ValueError,'Parent trial file changed'):
                audit.check_selection_chain(path,plan)

    def test_confirmation_requires_real_pin_schema(self):
        _,plan = self._long_plan()
        plan['phase'] = 'confirmation'
        plan['trials'] = [copy.deepcopy(plan['trials'][0])]
        plan['trials'][0]['channel'] = 2
        plan['selection']['parent_phase'] = 'long'
        audit.check_plan(plan)
        del plan['selection']
        plan['selection_sha256'] = 'legacy-loose-field-not-sufficient'
        with self.assertRaisesRegex(ValueError,'pinned long-stage'):
            audit.check_plan(plan)

    def test_actual_full_confirmation_chain_when_available(self):
        path = audit.ROOT/'plan_confirmation.json'
        if not path.exists():
            self.skipTest('Frozen confirmation plan is not available yet')
        plan = json.loads(path.read_text())
        audit.check_plan(plan)
        chain = audit.check_selection_chain(path,plan)
        self.assertEqual([link['parent_phase'] for link in chain],['screen','long'])
        self.assertEqual(chain[-1]['selected_config_ids'],plan['selected_config_ids'])


if __name__ == '__main__':
    unittest.main()
