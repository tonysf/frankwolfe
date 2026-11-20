"""Small controller contracts; these tests never execute a numerical solver."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fp64_controller_contract', HERE / 'compare_frames.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='scot_fp64_contract_')
        self.out = Path(self.temporary.name)
        self.binding = dict(reference_sha256='a' * 64, baseline_sha256='b' * 64,
                            instance_manifest_sha256='c' * 64,
                            array_files_sha256={name: 'd' * 64 for name in c.reference.ARRAY_NAMES})
        self.reference = dict(rounded=dict(loss=.02), prepared_instance=dict(
                              manifest_sha256=self.binding['instance_manifest_sha256'],
                              array_files_sha256=self.binding['array_files_sha256']))
        self.source = dict(source_commit=c.SOURCE_COMMIT, controller_sha256='e' * 64,
                           source_file_sha256={'run_pair.py': 'f' * 64})

    def tearDown(self):
        self.temporary.cleanup()

    def test_same_candidates_and_schedule_budget_except_precision(self):
        old = json.loads((c.ROOT / 'experiments/scot_cpu_20261014/local_config.json').read_text())
        self.assertEqual(c.CANDIDATES, old['candidates'])
        self.assertEqual(c.HORIZON, old['fixed_horizon'])
        self.assertEqual(c.CHECKPOINTS, old['checkpoint_iterations'])
        self.assertEqual(c.SEEDS, old['confirmation_seeds'])
        for candidate in c.CANDIDATES:
            command = c.command(self.out, 'tune', candidate, 810)
            args = dict(zip(command[2::2], command[3::2]))
            self.assertEqual(args['--dtype'], 'float64')
            self.assertEqual(args['--iterations'], '2000')
            self.assertEqual(args['--pairs-per-call'], '1024')
            self.assertEqual(args['--seed'], '810')
            self.assertEqual(args['--output-seed'], '20810')
            self.assertEqual(args['--initialization'], 'product')
            self.assertEqual(args['--gamma0'], str(candidate['gamma0']))
            self.assertEqual(args['--rho0'], str(candidate['rho0']))
            self.assertEqual(args['--mu-scale'], str(candidate['mu_scale']))
        pilot = c.command(self.out, 'pilot', c.CANDIDATES[0], 809)
        args = dict(zip(pilot[2::2], pilot[3::2]))
        self.assertEqual(args['--iterations'], '200')
        self.assertEqual(args['--checkpoints'], '')
        self.assertEqual(Path(args['--out']).name, 'pilot_fw04_fp64')

    def test_configuration_is_created_once_and_bound_to_precision(self):
        with mock.patch.object(c, 'inputs', return_value=(self.reference, self.binding)), \
                mock.patch.object(c, 'identity', return_value=self.source):
            value = c.frozen_config(self.out, create=True)
            self.assertEqual(value['optimizer_dtype'], 'float64')
            self.assertEqual(value['baseline_sha256'], self.binding['baseline_sha256'])
            self.assertEqual(c.frozen_config(self.out), value)
            with self.assertRaises(FileExistsError):
                c.frozen_config(self.out, create=True)
            value['optimizer_dtype'] = 'float32'
            c.write_json(self.out / 'local_config.json', value)
            with self.assertRaisesRegex(ValueError, 'configuration'):
                c.frozen_config(self.out)

    def test_selection_uses_terminal_fp64_loss_and_original_tie_order(self):
        rows = [dict(candidate=x, last_rounded_loss_float64=.03) for x in c.CANDIDATES]
        self.assertEqual(c.select(rows)['id'], 'fw04')
        rows[-1]['last_rounded_loss_float64'] = .02
        self.assertEqual(c.select(rows)['id'], 'fw08')

    def test_confirmation_rejects_the_tuning_seed_before_execution(self):
        with mock.patch.object(c, 'execute') as execute:
            with self.assertRaisesRegex(ValueError, 'Confirmation seed'):
                c.phase_confirm(self.out, 810)
            execute.assert_not_called()

    def test_missing_confirmation_cannot_finalize(self):
        chosen = dict(selected=c.CANDIDATES[0], tuning=[])
        with mock.patch.object(c, 'frozen_config', return_value={}), \
                mock.patch.object(c, 'inputs', return_value=(self.reference, self.binding)), \
                mock.patch.object(c, 'selection', return_value=chosen), \
                mock.patch.object(c, 'audited_run', side_effect=FileNotFoundError('seed813')):
            with self.assertRaises(FileNotFoundError):
                c.phase_finalize(self.out)
        self.assertFalse((self.out / 'completed.json').exists())

    def test_timeout_leaves_a_failed_receipt_and_refuses_reexecution(self):
        with mock.patch.object(c, 'identity', return_value=self.source), \
                mock.patch.object(c.subprocess, 'run', side_effect=subprocess.TimeoutExpired('fake solver', 3600)):
            with self.assertRaises(subprocess.TimeoutExpired):
                c.execute(self.out, 'pilot', c.CANDIDATES[0], 809, self.binding)
            receipt = c.read(self.out / 'pilot_fw04_fp64_receipt.json')
            self.assertEqual(receipt['status'], 'failed')
            self.assertIn('TimeoutExpired', receipt['error'])
            with self.assertRaisesRegex(ValueError, 'Refusing to reuse'):
                c.execute(self.out, 'pilot', c.CANDIDATES[0], 809, self.binding)

    def test_stage_receipt_and_lock_do_not_overwrite_a_completed_attempt(self):
        argv = ['compare_frames.py', '--phase', 'pilot', '--out', str(self.out)]
        with mock.patch.object(sys, 'argv', argv), mock.patch.object(c, 'phase_pilot', return_value={'test': True}):
            c.main()
            before = (self.out / 'stage_pilot.json').read_bytes()
            self.assertFalse((self.out / '.controller.lock').exists())
            with self.assertRaises(FileExistsError):
                c.main()
            self.assertEqual((self.out / 'stage_pilot.json').read_bytes(), before)
            self.assertFalse((self.out / '.controller.lock').exists())


if __name__ == '__main__':
    unittest.main()
