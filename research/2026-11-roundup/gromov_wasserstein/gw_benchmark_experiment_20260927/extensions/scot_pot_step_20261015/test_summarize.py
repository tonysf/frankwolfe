"""Small synthetic file fixtures test the reporter's rejection boundaries."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
import ot

SOURCE = Path(__file__).with_name('summarize.py')
SPEC = importlib.util.spec_from_file_location('pot_step_report_tests', SOURCE)
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


class ReporterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.out = Path(self.temporary.name)
        instance = self.out/'instance'
        instance.mkdir()
        self.arrays = [np.array([[0., 1.], [1., 0.]]),
                       np.array([[0., 1.], [1., 0.]]),
                       np.array([.5, .5]), np.array([.5, .5])]
        for filename, array in zip(reporter.ARRAY_NAMES, self.arrays):
            np.save(instance/filename, array)
        hashes = {name: reporter.sha256(instance/name) for name in reporter.ARRAY_NAMES}
        source_name = str(SOURCE.relative_to(reporter.ROOT))
        self.config = dict(iterations=3, epsilon=.001, pairs_per_call=1024,
                           inner_tol=1e-7, inner_gap_tol=1e-7,
                           checkpoint_iterations=[0, 1, 3], instance=str(instance),
                           array_files_sha256=hashes,
                           source_file_sha256={source_name: reporter.sha256(SOURCE)},
                           runs=[dict(id=name, oracle=oracle, seed=seed, output_seed=seed+20000,
                                      iterations=3, cpu=cpu)
                                 for name, oracle, seed, cpu in
                                 [('full_gradient', 'full', 811, 0), ('pair_seed811', 'pair', 811, 2),
                                  ('pair_seed812', 'pair', 812, 4), ('pair_seed813', 'pair', 813, 6)]])
        reporter.write(self.out/'config.json', self.config)
        self.plan = np.full((2, 2), .25)
        for spec in self.config['runs']:
            directory = self.out/'runs'/spec['id']
            directory.mkdir(parents=True)
            np.save(directory/'plan.npy', self.plan)
            value = reporter.evaluate_plan(directory/'plan.npy', self.arrays, .001, np, ot, rounded=True)
            observation = {}
            for prefix in ('raw', 'rounded'):
                observation.update({prefix+'_'+field: value[field]
                                    for field in ('loss', 'entropy', 'regularized_loss', 'residuals')})
            trace = [dict(iteration=k, **observation, plan_file='plan.npy', plan_sha256=value['file_sha256'])
                     for k in (0, 1, 3)]
            inner = [dict(iteration=k, eta=1000.,
                          inner_tolerance=1e-7, normalized_gap_tolerance=1e-7, inner_iterations=1,
                          converged=True, proximal_optimality_certified=True, certificate_valid=True,
                          primal_objective=.1, dual_objective=.1, primal_dual_gap=0.,
                          stable_primal_dual_gap=0., gap_roundoff_tolerance=1e-14,
                          normalized_primal_dual_gap=0., normalized_gap_upper_bound=1e-14,
                          marginal_l1=0., primal_witness_marginal_l1=0.) for k in range(3)]
            full = spec['oracle'] == 'full'
            metadata = dict(status='completed', config=dict(oracle=spec['oracle'], seed=spec['seed'],
                            output_seed=spec['output_seed'], iterations=3, epsilon=.001, pairs=1024,
                            inner_tol=1e-7, inner_gap_tol=1e-7, inner_max=10000,
                            inner_check_every=10, instance=str(instance)),
                            source_file_sha256=self.config['source_file_sha256'], array_files_sha256=hashes,
                            counters=dict(completed_updates=3, full_gradient_calls=3 if full else 0,
                                          stochastic_gradient_calls=0 if full else 3,
                                          pair_samples=0 if full else 3072, inner_stopping_failures=0),
                            inner_trace=inner, trace=trace,
                            schedule=dict(kind='pot-fixed', cmax=1., epsilon=.001,
                                          eta0=1000., eta_last=1000.,
                                          inner_tol=1e-7, inner_gap_tol=1e-7, inner_max=10000,
                                          inner_check_every=10),
                            dimensions=dict(n=2, m=2),
                            environment=dict(dtype='float64', device='cpu', distributed=False,
                                             torch_threads=1, torch_interop_threads=1,
                                             deterministic_algorithms=True,
                                             thread_environment=reporter.THREAD_ENV,
                                             cpu_affinity=[spec['cpu']]),
                            selected_output=dict(index=1, output_seed=spec['output_seed'],
                                                 probability=1/3, diagnostic_only=True,
                                                 **observation),
                            saved_plans={name: dict(file='plan.npy', file_sha256=value['file_sha256'])
                                         for name in reporter.PLAN_KEYS})
            reporter.write(directory/'metadata.json', metadata)

    def tearDown(self):
        self.temporary.cleanup()

    def change_metadata(self, mutate, name='pair_seed811'):
        path = self.out/'runs'/name/'metadata.json'
        metadata = reporter.read(path)
        mutate(metadata)
        reporter.write(path, metadata)

    def build(self, **kwargs):
        return reporter.build_report(self.out, make_plot=False, **kwargs)

    def test_complete_independent_report_and_csv(self):
        result = self.build()
        self.assertEqual(result['status'], 'complete_validated')
        self.assertEqual(result['completed_run_count'], 4)
        self.assertEqual(result['endpoint_summary']['pair']['final_loss']['n'], 3)
        self.assertAlmostEqual(result['endpoint_summary']['full']['final_loss']['median'], .5)
        self.assertTrue((self.out/'report'/'curve_data.csv').is_file())
        self.assertIn('endpoints.csv', result['artifacts_sha256'])

    def test_missing_completion_fails_and_partial_is_explicit(self):
        self.change_metadata(lambda meta: meta.update(status='running'))
        with self.assertRaisesRegex(ValueError, 'Missing completed run'):
            self.build()
        partial = self.build(partial=True)
        self.assertEqual(partial['status'], 'partial_validated')
        self.assertEqual(partial['completed_run_count'], 3)
        self.assertEqual(partial['pending'], [dict(id='pair_seed811', status='running')])
        self.assertTrue((self.out/'report_partial'/'report.json').is_file())
        self.assertFalse((self.out/'report'/'report.json').exists())

    def test_changed_plan_rejected_even_if_numbers_were_logged(self):
        path = self.out/'runs'/'pair_seed811'/'plan.npy'
        np.save(path, self.plan+1e-4)
        with self.assertRaisesRegex(ValueError, 'Changed saved plan'):
            self.build()

    def test_wrong_entropy_is_independently_detected(self):
        self.change_metadata(lambda meta: meta['trace'][1].update(rounded_entropy=-10.))
        with self.assertRaisesRegex(ValueError, 'rounded entropy'):
            self.build()

    def test_changed_dataset_rejected(self):
        np.save(Path(self.config['instance'])/'A.npy', self.arrays[0]+.01)
        with self.assertRaisesRegex(ValueError, 'Changed input array'):
            self.build()

    def test_wrong_source_hash_rejected(self):
        self.config['source_file_sha256'][str(SOURCE.relative_to(reporter.ROOT))] = '0'*64
        reporter.write(self.out/'config.json', self.config)
        with self.assertRaisesRegex(ValueError, 'Changed frozen source'):
            self.build()

    def test_failed_guarded_certificate_rejected(self):
        self.change_metadata(lambda meta: meta['inner_trace'][1].update(gap_roundoff_tolerance=1e-4))
        with self.assertRaisesRegex(ValueError, 'guarded criteria'):
            self.build()

    def test_changed_schedule_rejected(self):
        self.change_metadata(lambda meta: meta['inner_trace'][2].update(eta=1/(4.002*math.sqrt(3))))
        with self.assertRaisesRegex(ValueError, 'Fixed POT step size'):
            self.build()

    def test_wrong_schedule_kind_rejected(self):
        self.change_metadata(lambda meta: meta['schedule'].update(kind='theoretical-decay'))
        with self.assertRaisesRegex(ValueError, 'schedule kind'):
            self.build()

    def test_nonuniform_diagnostic_output_rejected(self):
        self.change_metadata(lambda meta: meta['selected_output'].update(probability=.5))
        with self.assertRaisesRegex(ValueError, 'output probability'):
            self.build()

    def test_zero_plan_entries_have_finite_entropy(self):
        path = self.out/'zero_entries.npy'
        np.save(path, np.diag([.5, .5]).astype(np.float64))
        value = reporter.evaluate_plan(path, self.arrays, .001, np, ot, rounded=True)
        self.assertEqual(value['loss'], 0.)
        self.assertAlmostEqual(value['entropy'], math.log(.5)-1)
        self.assertEqual(value['residuals']['min_entry'], 0.)

    def test_failed_run_remains_partial(self):
        self.change_metadata(lambda meta: meta.update(status='failed'))
        with self.assertRaisesRegex(ValueError, 'Missing completed run'):
            self.build()
        partial = self.build(partial=True)
        self.assertEqual(partial['status'], 'partial_validated')
        self.assertEqual(partial['pending'], [dict(id='pair_seed811', status='failed')])
        self.assertEqual(partial['completed_run_count'], 3)

    def test_missing_inner_solve_rejected(self):
        self.change_metadata(lambda meta: meta['inner_trace'].pop())
        with self.assertRaisesRegex(ValueError, 'Missing or failed proximal solves'):
            self.build()

    def test_inconsistent_oracle_counters_rejected(self):
        self.change_metadata(lambda meta: meta['counters'].update(pair_samples=1))
        with self.assertRaisesRegex(ValueError, 'oracle work counters'):
            self.build()

    def test_selected_last_iterate_rejected(self):
        self.change_metadata(lambda meta: meta['selected_output'].update(index=3))
        with self.assertRaisesRegex(ValueError, 'pre-update output'):
            self.build()

    def test_unsafe_plan_path_rejected(self):
        self.change_metadata(lambda meta: meta['trace'][1].update(plan_file='../plan.npy'))
        with self.assertRaisesRegex(ValueError, 'safe relative artifact'):
            self.build()

    def test_1047_product_marginals_use_accurate_fast_axis_reductions(self):
        marginal = np.full(1047, 1/1047, dtype=np.float64)
        product = np.outer(marginal, marginal)
        old_error = float(np.abs(product.sum(1)-marginal).sum()
                          + np.abs(product.sum(0)-marginal).sum())
        rows, cols = reporter.fp64_marginals(product, np)
        stable_error = float(np.abs(rows-marginal).sum()+np.abs(cols-marginal).sum())
        reference_rows = np.array([math.fsum(row) for row in product])
        reference_cols = np.array([math.fsum(col) for col in product.T])
        # On this frozen NumPy/FP64 setup the old strided reduction reports
        # 2.3838e-14 despite a product plan.  The strict 100-eps gate is retained.
        self.assertGreater(old_error, reporter.STRICT_FEASIBILITY)
        self.assertLess(stable_error, 4*np.finfo(np.float64).eps)
        np.testing.assert_allclose(rows, reference_rows, rtol=0, atol=3*np.spacing(marginal[0]))
        np.testing.assert_allclose(cols, reference_cols, rtol=0, atol=3*np.spacing(marginal[0]))
        # A Fortran-ordered saved layout must have the same accurate result.
        transposed_rows, transposed_cols = reporter.fp64_marginals(product.T, np)
        np.testing.assert_array_equal(rows, transposed_rows)
        np.testing.assert_array_equal(cols, transposed_cols)

    def test_actual_plan_infeasibility_still_fails_strict_gate(self):
        bad_plan = self.plan.copy()
        bad_plan[0, 0] += 1e-10
        path = self.out/'actually_infeasible.npy'
        np.save(path, bad_plan)
        with self.assertRaisesRegex(ValueError, 'violates marginal feasibility'):
            reporter.evaluate_plan(path, self.arrays, .001, np, ot, rounded=True)


if __name__ == '__main__':
    unittest.main()
