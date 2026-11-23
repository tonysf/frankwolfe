"""Small independent physical checks: no optimizer and no remote access."""
import unittest
from copy import deepcopy
import math
import numpy as np
import audit_followup as audit


class PhysicalTests(unittest.TestCase):
    def trial(self,n=2,c=2,i=1,s=10):
        return {'id':f'n{n}-c{c}-i{i}-s{s}','config_id':'rho-standard','n_qubits':n,'channel':c,
                'initialization_replicate':i,'initialization_seed':200000+10*c+i,
                'sampling_replicate':s,'sampling_seed':300000+100*c+s,
                'steps':400000,'metrics_every':4000,'tau':math.sqrt(2**n),**audit.FROZEN}

    def plan(self):
        return {'scope':'local_frames_frozen_followup','source_commit':audit.COMMIT,'gate':audit.GATE,
                'phase':'robustness','trials':[self.trial(c=c,s=s) for c in range(2,8) for s in (10,11)],
                'parent_pins':{str(audit.OLD/name):sha for name,sha in audit.PARENT_HASHES.items()},
                'resource_rule':audit.RESOURCE_RULE,'reserved_final_channels':list(range(1000,1020))}

    def test_frozen_plan_and_mutations(self):
        good = self.plan()
        audit.check_plan(good)
        mutations = [('alpha0',.01),('sampling_seed',123),('initialization_seed',123),
                     ('channel',1000),('steps',800000),('tau',3.),('n_qubits',3)]
        for field,value in mutations:
            with self.subTest(field=field):
                bad = deepcopy(good)
                bad['trials'][0][field] = value
                with self.assertRaises(ValueError):
                    audit.check_plan(bad)

    def test_missing_and_duplicate_trial_rejected(self):
        bad = self.plan()
        bad['trials'].pop()
        with self.assertRaisesRegex(ValueError,'inventory'):
            audit.check_plan(bad)
        bad = self.plan()
        bad['trials'].append(deepcopy(bad['trials'][0]))
        with self.assertRaisesRegex(ValueError,'duplicate'):
            audit.check_plan(bad)

    def test_parent_pin_mutation_rejected(self):
        bad = self.plan()
        bad['parent_pins'][str(audit.OLD/'selection_long.json')] = '0'*64
        with self.assertRaisesRegex(ValueError,'parent pin'):
            audit.check_plan(bad)

    def test_resource_thresholds_and_no_gate_argument(self):
        self.assertEqual(audit.resource_horizon(2.,10.,100.)[1],400000)
        self.assertEqual(audit.resource_horizon(4.,10.,100.)[1],100000)
        self.assertEqual(audit.resource_horizon(8.,10.,100.)[1],0)
        self.assertEqual(audit.resource_horizon(2.,10.,2048.)[1],0)
        with self.assertRaises(ValueError):
            audit.resource_horizon(np.nan,10.,100.)
        with self.assertRaises(ValueError):
            audit.resource_horizon(2.,10.,100.,1000)

    def test_scaled_truth_has_tp_error_at_each_dimension(self):
        for d in (4,8,16):
            truth = audit.independent_truth(0,d)
            metric = audit.independent_metrics(.9*truth,truth,audit.born_values(truth,d),5e-5,d)
            self.assertAlmostEqual(metric['normalized_tp_violation'],.19)
            self.assertAlmostEqual(metric['relative_rmse'],.19)
            self.assertAlmostEqual(metric['process_fidelity'],1)
            self.assertFalse(audit.gate_and_deficit(metric)[0])

    def test_dimension_specific_povm(self):
        self.assertEqual(audit.povm_constants(4),(.1,.1))
        self.assertAlmostEqual(audit.povm_constants(16)[0],1/37)
        self.assertNotEqual(audit.povm_constants(8)[0],.1)
        for d in (4,8,16):
            probes,effects = audit.physical_design(d)
            self.assertEqual(probes.shape,(d*d,d))
            self.assertEqual(effects.shape,(2*d,d,d))
            np.testing.assert_allclose(effects.sum(axis=0),np.eye(d),atol=1e-14)
            self.assertGreaterEqual(np.linalg.eigvalsh(effects).min(),-1e-14)
            np.testing.assert_allclose(np.sum(abs(probes)**2,axis=1),1)

    def test_identity_and_haar_at_each_dimension(self):
        for d in (4,8,16):
            for truth in (np.eye(d,dtype=complex).reshape(d*d,1),audit.independent_truth(0,d)):
                values = audit.born_values(truth,d)
                np.testing.assert_allclose(values.reshape(d*d,2*d).sum(axis=1),1,atol=1e-14)
                self.assertGreaterEqual(values.min(),-1e-14)
                metric = audit.independent_metrics(truth,truth,values,5e-5,d)
                self.assertLess(metric['normalized_tp_violation'],1e-14)
                self.assertEqual(metric['rmse'],0)
                self.assertEqual(metric['relative_rmse'],0)
                self.assertTrue(audit.gate_and_deficit(metric)[0])

    def test_initializer_replicates_separate(self):
        for d in (4,8,16):
            first = audit.independent_initializer(200021,d)
            second = audit.independent_initializer(200022,d)
            self.assertEqual(first.shape,(d*d,1))
            self.assertAlmostEqual(np.linalg.norm(first),np.sqrt(d))
            self.assertFalse(np.array_equal(first,second))

    def test_pair_groups_separate_data_initialization_sampling(self):
        c = {'n_qubits':2,'channel':2,'initialization_replicate':1,'sampling_replicate':10}
        keys = audit.pair_keys(c)
        new_init = audit.pair_keys(dict(c,initialization_replicate=2))
        new_stream = audit.pair_keys(dict(c,sampling_replicate=11))
        self.assertEqual(keys[0],new_init[0])
        self.assertNotEqual(keys[1],new_init[1])
        self.assertEqual(keys[2],new_init[2])
        self.assertEqual(keys[1],new_stream[1])
        self.assertNotEqual(keys[2],new_stream[2])

    def test_failed_run_cannot_pass(self):
        metric = {'process_fidelity':1.,'rmse':0.,'normalized_tp_violation':0.}
        self.assertTrue(audit.gate_and_deficit(metric)[0])
        self.assertFalse(audit.gate_and_deficit(metric,'failed')[0])

    def test_persistent_crossing_not_first(self):
        self.assertEqual(audit.gate_crossings(np.array([0,10,20,30]),[False,True,False,True]),(10,30))

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            audit.close(np.nan,np.nan,'nan')


if __name__ == '__main__':
    unittest.main()
