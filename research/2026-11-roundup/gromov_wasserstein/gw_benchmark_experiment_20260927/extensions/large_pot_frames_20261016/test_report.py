import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import report

class ReportTests(unittest.TestCase):
    def fixture(self):
        trace=[dict(iteration=k,raw_loss=f,rounded_loss=f,raw_entropy=-2.,rounded_entropy=-2.,raw_regularized_loss=f-.002,rounded_regularized_loss=f-.002,rounded_residuals={'both_l1':1e-15}) for k,f in [(0,.1),(2,.05)]]
        return dict(trace=trace,target_updates=2,completed_updates=2,final_loss=.05,method='frames',config={'epsilon':.001,'pairs':1024},counters={'stochastic_gradient_calls':2,'pair_samples':2048})
    def test_valid_trace(self):
        self.assertEqual(len(report.audit_trace(self.fixture())),2)
    def test_rejects_missing_endpoint(self):
        m=self.fixture();m['trace'].pop()
        with self.assertRaises(ValueError):report.audit_trace(m)
    def test_rejects_bad_feasibility(self):
        m=self.fixture();m['trace'][-1]['rounded_residuals']['both_l1']=1e-5
        with self.assertRaises(ValueError):report.audit_trace(m)
    def test_rejects_oracle_count(self):
        m=self.fixture();m['counters']['pair_samples']=100
        with self.assertRaises(ValueError):report.audit_trace(m)
    def test_rejects_inconsistent_entropy(self):
        m=self.fixture();m['trace'][-1]['rounded_regularized_loss']=.1
        with self.assertRaises(ValueError):report.audit_trace(m)
    def test_no_extrapolation_or_invented_seeds(self):
        a=[{'iteration':0,'rounded_loss':1.},{'iteration':2,'rounded_loss':.2}]
        b=[{'iteration':0,'rounded_loss':1.},{'iteration':1,'rounded_loss':.5},{'iteration':2,'rounded_loss':.4}]
        r=report.aggregate([a,b]);self.assertEqual([x['iteration'] for x in r],[0,2]);self.assertAlmostEqual(r[-1]['median'],.3);self.assertEqual(r[-1]['n'],2)

if __name__=='__main__':unittest.main()
