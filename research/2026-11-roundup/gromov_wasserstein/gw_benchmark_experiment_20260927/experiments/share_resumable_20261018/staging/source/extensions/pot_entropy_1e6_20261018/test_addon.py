import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


c = module('entropy_addon_controller_test', HERE/'campaign.py')
v = module('entropy_addon_validation_test', HERE/'validation.py')


class AddonTests(unittest.TestCase):
    def setUp(self):
        self.config = c.load_config(HERE/'campaign.json')
        v.torch.set_num_threads(1)

    def test_only_eight_new_trials_and_reused_runner(self):
        tasks = c.base.tasks(self.config)
        self.assertEqual(len(tasks)*len(c.DATASETS), 8)
        self.assertEqual({t['epsilon'] for t in tasks}, {1e-6})
        self.assertEqual(len({t['task_id'] for t in tasks}), 4)
        command = c.runner_command('/source', '/out', self.config, 'snare_full', tasks[-1], 7000, '/checkpoints')
        self.assertEqual(command[1], '/source/extensions/pot_entropy_init_20261018/runner.py')
        args = v.runner.parser().parse_args(command[2:])
        self.assertEqual(v.runner.immutable_config(args), c.base.task_config(self.config, tasks[-1]))
        self.assertEqual(args.epsilon, 1e-6)

    def test_independent_reference_and_log_fallback_at_1e6(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory)/'input'
            v.write_fixture(fixture)
            checks = v.tiny_checks(fixture, 'cpu')
            self.assertEqual(len(checks), 3)
            self.assertEqual(checks[-1]['name'], 'low_entropy_underflow_log_fallback')

    def test_deadline_and_resume_at_1e6(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v.write_fixture(root/'input')
            def args(name):
                return v.runner.parser().parse_args(['--instance', str(root/'input'), '--out', str(root/name),
                    '--device', 'cpu', '--epsilon', '1e-6', '--initialization', 'permutation-mixture',
                    '--seed', '901', '--iterations', '4', '--max-seconds', '100', '--retain-checkpoint'])
            full = v.runner.run(args('full'))
            real_clock, expiry = time.monotonic, [False]
            original_iteration = v.runner.DeadlineCertifiedPOTProx._scaling_iteration
            def interrupt(prox, *a, **kw):
                expiry[0] = True
                return original_iteration(prox, *a, **kw)
            def clock():
                return real_clock()+(1000 if expiry[0] else 0)
            with patch.object(v.runner.time, 'monotonic', side_effect=clock), patch.object(
                    v.runner.DeadlineCertifiedPOTProx, '_scaling_iteration', new=interrupt):
                partial = v.runner.run(args('resumed'))
            self.assertEqual(partial['status'], 'checkpointed')
            self.assertEqual(partial['completed_updates'], 0)
            self.assertEqual(partial['counters']['interrupted_inner_attempts'], 1)
            resumed = v.runner.run(args('resumed'))
            for key in ('trace', 'inner_trace', 'outer_trace', 'stopping_reason'):
                self.assertEqual(full[key], resumed[key], key)
            states = [v.CheckpointStore(root/name/'checkpoints').load(identity=meta['identity'])[0]
                      for name, meta in [('full', full), ('resumed', resumed)]]
            self.assertTrue(v.torch.equal(states[0]['plan'], states[1]['plan']))

    def test_initial_submission_only_cpu_gate_and_parent_bridges(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            def submit(*args, **kwargs):
                calls.append((args, kwargs))
                return str(100+len(calls))
            with patch.object(c.base, 'check_source'), patch.object(c, 'check_storage'), patch.object(c, 'parent_identity'), patch.object(c.base, 'identity', return_value={}), patch.object(c.base, 'submit_job', side_effect=submit):
                c.submit_initial(HERE.parents[1], directory, HERE/'campaign.json')
                with self.assertRaises(FileExistsError):
                    c.submit_initial(HERE.parents[1], directory, HERE/'campaign.json')
            self.assertEqual(len(calls), 3)
            self.assertEqual(calls[1][0][2], 'await-parent')
            self.assertEqual(calls[1][1]['dependency'], 'afterok:101,afterany:2082920')
            self.assertEqual(calls[2][1]['dependency'], 'afterok:101,afterany:2082923')

    def bridge_fixture(self, root, *, completed=False):
        parent = root/'parent'
        dataset = parent/'results/bmmc6224'
        dataset.mkdir(parents=True)
        if completed:
            c.write(dataset/'completed.json', dict(status='completed', runs=16, successful_runs=15, unsuccessful_runs=1))
        else:
            c.write(dataset/'dispatch_state.json', dict(status='continued', next_dispatch_job_id='9999'))
        return parent

    def test_bridge_follows_continuation_dispatcher_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = self.bridge_fixture(root)
            with patch.dict(os.environ, {'SLURM_JOB_ID': '8888'}), patch.object(c.base, 'require_gate'), patch.object(c.base, 'identity', return_value={}), patch.object(c, 'parent_identity', return_value=(parent, {})), patch.object(c.base, 'slurm_terminal_record', return_value=dict(state='COMPLETED', exit_code='0:0')), patch.object(c.base, 'submit_job', return_value='10000') as submit:
                first = c.await_parent(HERE.parents[1], root, HERE/'campaign.json', 'bmmc6224', '2082920')
                again = c.await_parent(HERE.parents[1], root, HERE/'campaign.json', 'bmmc6224', '2082920')
            self.assertEqual(first, again)
            self.assertEqual(first['status'], 'waiting')
            self.assertEqual(submit.call_count, 1)
            self.assertEqual(submit.call_args.kwargs['dependency'], 'afterok:8888,afterany:9999')
            self.assertEqual(submit.call_args.kwargs['parent_job'], '9999')

    def test_bridge_releases_gpu_work_only_after_all_parent_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = self.bridge_fixture(root, completed=True)
            with patch.dict(os.environ, {'SLURM_JOB_ID': '8888'}), patch.object(c.base, 'require_gate'), patch.object(c.base, 'identity', return_value={}), patch.object(c, 'parent_identity', return_value=(parent, {})), patch.object(c.base, 'slurm_terminal_record', return_value=dict(state='COMPLETED', exit_code='0:0')), patch.object(c.base, 'submit_job', side_effect=['101', '102', '103']) as submit:
                result = c.await_parent(HERE.parents[1], root, HERE/'campaign.json', 'bmmc6224', '2082920')
            self.assertEqual(result['status'], 'released')
            self.assertEqual(submit.call_count, 3)
            self.assertEqual(submit.call_args_list[0].kwargs['dependency'], 'afterok:8888')
            self.assertEqual(submit.call_args_list[1].kwargs['dependency'], 'afterok:101')
            self.assertEqual(submit.call_args_list[2].kwargs['dependency'], 'afterany:102')

    def test_bridge_freezes_on_parent_infrastructure_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = self.bridge_fixture(root)
            with patch.dict(os.environ, {'SLURM_JOB_ID': '8888'}), patch.object(c.base, 'require_gate'), patch.object(c.base, 'identity', return_value={}), patch.object(c, 'parent_identity', return_value=(parent, {})), patch.object(c.base, 'slurm_terminal_record', return_value=dict(state='FAILED', exit_code='1:0')), patch.object(c.base, 'submit_job') as submit:
                with self.assertRaisesRegex(ValueError, 'did not succeed'):
                    c.await_parent(HERE.parents[1], root, HERE/'campaign.json', 'bmmc6224', '2082920')
            submit.assert_not_called()
            self.assertEqual(c.read(root/'bridges/bmmc6224/failed.json')['status'], 'failed')

    def test_changed_parent_identity_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = json.loads(json.dumps(self.config))
            config['additive_grid_parent']['root'] = directory
            c.write(Path(directory)/'campaign_identity.json', dict(config_sha256='wrong', source_manifest_sha256='wrong'))
            with self.assertRaisesRegex(ValueError, 'identity'):
                c.parent_identity(config)

    def test_storage_reservation_covers_retained_share(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = dict(quota_verified=True, quota_available_bytes=68*1024**3, checkpoint_root=directory, verified_utc=c.now())
            c.write(root/'storage_receipt.json', receipt)
            self.assertEqual(c.check_storage(root, self.config, fresh=True), root)
            receipt['quota_available_bytes'] -= 1
            c.write(root/'storage_receipt.json', receipt)
            with self.assertRaisesRegex(ValueError, 'headroom'):
                c.check_storage(root, self.config)


if __name__ == '__main__':
    unittest.main()
