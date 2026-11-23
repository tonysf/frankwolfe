"""Recovery orchestration tests only: no numerical workers or host services."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('scot_recovery_test', HERE / 'resume_campaign.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name)
        self.expected = dict(pid=12, start_ticks=99, cmdline=['python', 'controller.py'])
        self.original = dict(self.expected, state='S')

    def tearDown(self):
        self.tmp.cleanup()

    def test_process_stat_parser_handles_spaces_and_parentheses(self):
        directory = self.out / '12'
        directory.mkdir()
        fields = ['S'] + ['0'] * 18 + ['99']
        (directory / 'stat').write_text('12 (a name (with spaces)) ' + ' '.join(fields))
        (directory / 'cmdline').write_bytes(b'python\0controller.py\0')
        self.assertEqual(r.process_identity(12, self.out), self.original)
        self.assertIsNone(r.process_identity(99, self.out))

    def test_host_pid_namespace_must_match(self):
        (self.out / '1').mkdir()
        (self.out / '1/comm').write_text('systemd\n')
        expected = dict(pid1_comm='systemd', pid_namespace='pid:[42]')
        with mock.patch.object(r.os, 'readlink', return_value='pid:[42]'):
            self.assertEqual(r.host_identity(expected, self.out), expected)
        with mock.patch.object(r.os, 'readlink', return_value='pid:[43]'):
            with self.assertRaisesRegex(ValueError, 'namespace'):
                r.host_identity(expected, self.out)

    def test_initial_pid_identity_is_required(self):
        for observed in (None, dict(self.original, start_ticks=100), dict(self.original, cmdline=['other'])):
            with self.subTest(observed=observed), self.assertRaisesRegex(ValueError, 'identity'):
                r.wait_for_original(self.expected, identity=lambda *a: observed)

    def test_wait_tracks_child_after_original_exit_and_ignores_reused_pid(self):
        ticks = [0]
        child = dict(pid=13, start_ticks=101, cmdline=['python', 'run_pair.py'], state='R')
        def identity(pid, proc):
            if pid == 12:
                return self.original if ticks[0] == 0 else dict(self.original, start_ticks=100)
            return child if ticks[0] < 120 else None
        def sleep(seconds):
            ticks[0] += seconds
        value = r.wait_for_original(self.expected, identity=identity,
                    descendants=lambda pid, proc: [child] if pid == 12 else [],
                    clock=lambda: ticks[0], sleep=sleep)
        self.assertEqual(value['elapsed_seconds'], 120)
        self.assertEqual(value['observations'], 3)
        self.assertEqual([row['pid'] for row in value['observed_processes']], [12, 13])
        self.assertFalse(value['signals_sent'])

    def test_wait_is_bounded_and_rejects_changed_command(self):
        ticks = [0]
        with self.assertRaisesRegex(ValueError, 'bounded wait'):
            r.wait_for_original(self.expected, timeout=60, identity=lambda *a: self.original,
                                descendants=lambda *a: [], clock=lambda: ticks[0],
                                sleep=lambda seconds: ticks.__setitem__(0, ticks[0] + seconds))
        with self.assertRaisesRegex(ValueError, 'changed command'):
            r.wait_for_original(self.expected,
                identity=mock.Mock(side_effect=[self.original, dict(self.original, cmdline=['changed'])]))

    def test_global_lock_refuses_second_writer(self):
        with r.exclusive_lock(self.out / 'lock'):
            with self.assertRaisesRegex(RuntimeError, 'exclusive lock'):
                with r.exclusive_lock(self.out / 'lock'):
                    self.fail('Second writer must not enter.')

    def test_existing_confirmation_prevents_any_mutation(self):
        (self.out / 'runs').mkdir()
        (self.out / 'runs/confirm_test_seed811').mkdir()
        with mock.patch.object(r.c, 'configuration') as config:
            with self.assertRaisesRegex(ValueError, 'already exists'):
                r.resume_after_wait(self.out, {})
            config.assert_not_called()

    def test_audit_failure_prevents_replacement_and_new_stages(self):
        with mock.patch.object(r.c, 'configuration', return_value=({}, {})), \
             mock.patch.object(r.c, 'tune_cells', return_value=[('m', {}, 810)]), \
             mock.patch.object(r.c, 'audited', side_effect=ValueError('invalid tuning')), \
             mock.patch.object(r, 'preserve_failed_stages') as archive, \
             mock.patch.object(r.c, 'durable_stage') as stage:
            with self.assertRaisesRegex(ValueError, 'invalid tuning'):
                r.resume_after_wait(self.out, {})
            archive.assert_not_called()
            stage.assert_not_called()

    def test_recovery_preserves_failure_and_orders_unchanged_operations(self):
        for label in ('campaign', 'tune_both_0_end'):
            r.c.create_json(self.out / f'stage_{label}.json', dict(status='failed', stage=label, error='original error'))
        before = {name: (self.out / name).read_bytes() for name in ('stage_campaign.json', 'stage_tune_both_0_end.json')}
        cells = [('method', {'id': str(k)}, 810) for k in range(8)]
        rows = [dict(candidate=cell[1]) for cell in cells]
        operations, stages = [], []
        original_stage = r.c.durable_stage
        def stage(out, label, operation):
            stages.append(label)
            return original_stage(out, label, operation)
        def choose(*args):
            operations.append('select')
            return {'selection': True}
        def wave(args, *rest):
            self.assertEqual(vars(args), dict(phase='confirm', method=None, start=0, stop=None))
            operations.append('confirm')
            return {'runs': []}
        with mock.patch.object(r.c, 'configuration', return_value=({}, {})), \
             mock.patch.object(r.c, 'tune_cells', return_value=cells), \
             mock.patch.object(r.c, 'audited', side_effect=copy.deepcopy(rows)) as audit, \
             mock.patch.object(r.c, 'choose', side_effect=choose), \
             mock.patch.object(r.c, 'wave', side_effect=wave), \
             mock.patch.object(r.c, 'finalize', side_effect=lambda *a: operations.append('finalize')), \
             mock.patch.object(r, 'render_report', side_effect=lambda *a: operations.append('report')), \
             mock.patch.object(r.c, 'durable_stage', side_effect=stage):
            receipt = {}
            r.resume_after_wait(self.out, receipt)
        self.assertEqual(audit.call_count, 8)
        self.assertEqual(stages, ['tune_both_0_end', 'select', 'confirm_both_0_end', 'finalize', 'report'])
        self.assertEqual(operations, ['select', 'confirm', 'finalize', 'report'])
        self.assertEqual((self.out / 'stage_campaign.json').read_bytes(), before['stage_campaign.json'])
        for name, data in before.items():
            self.assertEqual((self.out / 'recovery_evidence' / name).read_bytes(), data)
        tune = r.c.read(self.out / 'stage_tune_both_0_end.json')
        self.assertEqual(tune['result'], dict(runs=rows, selection_sha256=None))
        self.assertEqual(tune['controller_sha256'], r.c.sha256(HERE / 'run_comparison.py'))
        self.assertEqual(receipt['audited_tuning_count'], 8)


if __name__ == '__main__':
    unittest.main()
