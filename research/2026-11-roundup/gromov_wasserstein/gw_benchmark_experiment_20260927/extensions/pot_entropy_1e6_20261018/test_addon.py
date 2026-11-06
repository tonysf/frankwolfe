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
            receipt = dict(quota_verified=True, quota_available_bytes=65*1024**3, checkpoint_root=directory, verified_utc=c.now())
            c.write(root/'storage_receipt.json', receipt)
            self.assertEqual(c.check_storage(root, self.config, fresh=True), root)
            receipt['quota_available_bytes'] -= 1
            c.write(root/'storage_receipt.json', receipt)
            with self.assertRaisesRegex(ValueError, 'headroom'):
                c.check_storage(root, self.config)

    def terminal_fixture(self, root, *, unsuccessful=False):
        source = root/'source'
        source.mkdir()
        c.write(source/c.EXT/'campaign.json', self.config)
        (source/'SHA256SUMS').write_text('test source manifest')
        (source/'numerical.py').write_text('test numerical source')
        c.write(root/'campaign_identity.json', c.base.identity(source, source/c.EXT/'campaign.json'))
        checkpoint_root = root/'checkpoints'
        checkpoint_root.mkdir()
        c.write(root/'storage_receipt.json', dict(checkpoint_root=str(checkpoint_root)))
        task = c.base.tasks(self.config)[0]
        instance = self.config['instances']['bmmc6224']
        identity = dict(config=c.base.task_config(self.config, task), n=instance['n'], m=instance['n'],
            world_size=1, array_files_sha256=instance['array_files_sha256'],
            manifest_sha256=instance['instance_manifest_sha256'],
            source_file_sha256={'numerical.py': c.base.sha256(source/'numerical.py')})
        out = root/'results/bmmc6224'/task['task_id']
        iteration = 3 if unsuccessful else 2000
        if unsuccessful:
            c.write(out/'failure.json', dict(status='failed', failure_kind='inner_certificate_not_met',
                    completed_updates=iteration, identity=identity, certificate={'inner_iterations':100000}))
            c.write(out/'unsuccessful.json', dict(status='unsuccessful', failure_sha256=c.base.sha256(out/'failure.json')))
        else:
            c.write(out/'metadata.json', dict(status='completed', identity=identity, config=identity['config'],
                target_updates=2000, completed_updates=iteration, final_loss=.03,
                schedule=dict(inner_tolerance=1e-7, gap_tolerance=1e-7, eta=1/task['epsilon']),
                trace=[dict(iteration=iteration)]))
        folder = checkpoint_root/'bmmc6224'/task['task_id']
        generation = folder/'generation-000002000-test'
        generation.mkdir(parents=True)
        shard = generation/'rank-0000.pt'
        shard.write_bytes(b'large checkpoint state')
        manifest = dict(identity=identity, world_size=1, iteration=iteration, generation=generation.name,
            shards=[dict(file=shard.name, bytes=shard.stat().st_size, sha256=c.base.sha256(shard))])
        c.write(generation/'manifest.json', manifest)
        c.write(folder/'COMMITTED.json', manifest)
        return source, task, instance, out, folder, manifest

    def test_validated_terminal_cleanup_is_idempotent_and_preserves_evidence(self):
        for unsuccessful in (False, True):
            with self.subTest(unsuccessful=unsuccessful), tempfile.TemporaryDirectory() as directory:
                source, task, instance, out, folder, manifest = self.terminal_fixture(Path(directory), unsuccessful=unsuccessful)
                self.assertTrue(c.task_finished(out, task, instance, self.config, source))
                self.assertFalse((folder/'COMMITTED.json').exists())
                self.assertEqual(list(folder.glob('generation-*')), [])
                receipt = c.read(out/'terminal_checkpoint_cleanup.json')
                self.assertEqual(receipt['status'], 'completed')
                self.assertEqual(receipt['committed_checkpoint'], manifest)
                self.assertEqual(receipt['generations'][manifest['generation']]['manifest'], manifest)
                self.assertTrue(c.task_finished(out, task, instance, self.config, source))
                self.assertEqual(c.read(out/'terminal_checkpoint_cleanup.json'), receipt)

    def test_cleanup_resumes_after_marker_removal_and_rejects_invalid_terminal(self):
        with tempfile.TemporaryDirectory() as directory:
            source, task, instance, out, folder, manifest = self.terminal_fixture(Path(directory))
            with patch.object(c.shutil, 'rmtree', side_effect=OSError('injected kill during prune')):
                with self.assertRaisesRegex(OSError, 'injected kill'):
                    c.task_finished(out, task, instance, self.config, source)
            self.assertFalse((folder/'COMMITTED.json').exists())
            self.assertEqual(c.read(out/'terminal_checkpoint_cleanup.json')['status'], 'pruning')
            self.assertTrue(c.task_finished(out, task, instance, self.config, source))
            self.assertEqual(c.read(out/'terminal_checkpoint_cleanup.json')['status'], 'completed')
        with tempfile.TemporaryDirectory() as directory:
            source, task, instance, out, folder, manifest = self.terminal_fixture(Path(directory))
            meta = c.read(out/'metadata.json')
            meta['final_loss'] = float('nan')
            # Direct JSON permits the deliberately invalid receipt for rejection.
            (out/'metadata.json').write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'NaN|nonfinite'):
                c.task_finished(out, task, instance, self.config, source)
            self.assertTrue((folder/'COMMITTED.json').exists())


if __name__ == '__main__':
    unittest.main()
