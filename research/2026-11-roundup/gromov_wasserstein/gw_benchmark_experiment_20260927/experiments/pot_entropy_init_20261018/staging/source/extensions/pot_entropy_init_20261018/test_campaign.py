import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('entropy_campaign_tested', HERE/'campaign.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.config = c.load_config(HERE/'campaign.json')

    def identity(self, task, source, dataset='bmmc6224'):
        instance = self.config['instances'][dataset]
        (source/'numerical.py').write_text('immutable test source')
        return dict(config=c.task_config(self.config, task), n=instance['n'], m=instance['n'], world_size=1,
                    array_files_sha256=instance['array_files_sha256'], manifest_sha256=instance['instance_manifest_sha256'],
                    source_file_sha256={'numerical.py': c.sha256(source/'numerical.py')})

    def result(self, task, source, updates=2000):
        identity = self.identity(task, source)
        return dict(status='completed', identity=identity, config=identity['config'], target_updates=2000,
                    completed_updates=updates, final_loss=.03, stable_updates=10, last_outer_update_l1=1e-8,
                    stopping_reason='outer_update_l1_tolerance' if updates<2000 else 'iteration_budget',
                    schedule=dict(inner_tolerance=1e-7, gap_tolerance=1e-7, eta=1/task['epsilon']),
                    trace=[dict(iteration=updates)])

    def test_design_has_complete_cross_without_duplicate_tasks(self):
        tasks = c.tasks(self.config)
        self.assertEqual(len(tasks), 16)
        self.assertEqual(len({t['task_id'] for t in tasks}), 16)
        self.assertEqual({t['method'] for t in tasks}, {'pot-full'})
        self.assertEqual({t['iterations'] for t in tasks}, {2000})
        self.assertEqual({(t['epsilon'], t['initialization'], t['seed']) for t in tasks},
                         {(e, i['kind'], i['seed']) for e in self.config['epsilons'] for i in self.config['initializations']})

    def test_early_stop_requires_evidence_and_same_source_data_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            task = c.tasks(self.config)[0]
            good = self.result(task, source, updates=75)
            c.validate_result(good, task, self.config['instances']['bmmc6224'], self.config, source)
            for changes in [dict(completed_updates=49), dict(stable_updates=9), dict(last_outer_update_l1=2e-6),
                            dict(stopping_reason='iteration_budget'), dict(final_loss=float('nan'))]:
                bad = dict(good, **changes)
                with self.assertRaises(ValueError):
                    c.validate_result(bad, task, self.config['instances']['bmmc6224'], self.config, source)
            bad = copy.deepcopy(good)
            bad['identity']['config']['epsilon'] = .01
            with self.assertRaisesRegex(ValueError, 'configuration'):
                c.validate_result(bad, task, self.config['instances']['bmmc6224'], self.config, source)
            (source/'numerical.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'source'):
                c.validate_result(good, task, self.config['instances']['bmmc6224'], self.config, source)

    def test_inner_failure_needs_its_identity_and_certificate(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, task = Path(tmp), c.tasks(self.config)[3]
            failure = dict(status='failed', failure_kind='inner_certificate_not_met', completed_updates=3,
                           identity=self.identity(task, source), certificate={'inner_iterations':100000})
            c.validate_numerical_failure(failure, task, self.config['instances']['bmmc6224'], self.config, source)
            for change in [dict(failure_kind='runtime_error'), dict(certificate=None), dict(completed_updates=2000)]:
                with self.assertRaises(ValueError):
                    c.validate_numerical_failure(dict(failure, **change), task, self.config['instances']['bmmc6224'], self.config, source)

    def test_command_uses_only_full_gradient_fp64_and_is_compatible_with_runner_parser(self):
        # Importing the runner is appropriate in a CPU test allocation/local test.
        runner_spec = importlib.util.spec_from_file_location('entropy_command_test_runner', HERE/'runner.py')
        runner = importlib.util.module_from_spec(runner_spec)
        runner_spec.loader.exec_module(runner)
        task = c.tasks(self.config)[-1]
        command = c.runner_command('/source', '/out', self.config, 'snare_full', task, 7000, '/checkpoints')
        args = runner.parser().parse_args(command[2:])
        self.assertEqual(runner.immutable_config(args), c.task_config(self.config, task))
        self.assertEqual(args.checkpoint_dir, '/checkpoints/snare_full/'+task['task_id'])
        self.assertFalse(args.retain_checkpoint)
        self.assertEqual(args.epsilon, .00003)

    def test_storage_accounts_for_existing_share_reservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = dict(quota_verified=True, quota_available_bytes=70*1024**3, checkpoint_root=tmp, verified_utc=c.now())
            c.write(root/'storage_receipt.json', receipt)
            self.assertEqual(c.check_storage(root, self.config, fresh=True), root)
            receipt['quota_available_bytes'] -= 1
            c.write(root/'storage_receipt.json', receipt)
            with self.assertRaisesRegex(ValueError, 'headroom'):
                c.check_storage(root, self.config)

    def test_initial_jobs_depend_on_new_gates_and_duplicate_submit_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = []
            def fake_submit(*args, **kwargs):
                jobs.append((args, kwargs))
                return str(100+len(jobs))
            with patch.object(c, 'check_source'), patch.object(c, 'check_storage'), patch.object(c, 'identity', return_value={}), patch.object(c, 'submit_job', side_effect=fake_submit):
                c.submit_initial(HERE.parents[1], tmp, HERE/'campaign.json')
                with self.assertRaises(FileExistsError):
                    c.submit_initial(HERE.parents[1], tmp, HERE/'campaign.json')
            self.assertEqual(len(jobs), 7)
            self.assertEqual(jobs[1][1]['dependency'], 'afterok:101')
            self.assertEqual(jobs[2][1]['dependency'], 'afterok:102')
            self.assertEqual(jobs[3][1]['dependency'], 'afterany:103')
            self.assertEqual(jobs[4][1]['dependency'], 'afterok:101')
            self.assertEqual(jobs[5][1]['dependency'], 'afterok:105')
            self.assertEqual(jobs[6][1]['dependency'], 'afterany:106')

    def test_numerical_failure_is_recorded_then_next_task_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'source'
            source.mkdir()
            config = source/c.EXT/'campaign.json'
            c.write(config, self.config)
            task_list = c.tasks(self.config)[:2]
            task = task_list[0]
            failure = dict(status='failed', failure_kind='inner_certificate_not_met', completed_updates=3,
                           identity=self.identity(task, source), certificate={'inner_iterations':100000})
            calls = []
            def fake_run(command, **kwargs):
                calls.append(command)
                out = Path(command[command.index('--out')+1])
                if len(calls) == 1:
                    c.write(out/'failure.json', failure)
                    return subprocess.CompletedProcess(command, 1)
                c.write(out/'progress.json', dict(status='checkpointed'))
                return subprocess.CompletedProcess(command, 0)
            with patch.object(c, 'tasks', return_value=task_list), patch.object(c, 'identity', return_value={}), patch.object(c, 'require_gate'), patch.object(c, 'check_storage', return_value=root/'checkpoints'), patch.dict(os.environ, {'SLURM_JOB_ID':'1234'}), patch.object(c.subprocess, 'run', side_effect=fake_run):
                c.production(source, root, config, 'bmmc6224')
            self.assertEqual(len(calls), 2)
            self.assertEqual(c.read(root/'results/bmmc6224'/task['task_id']/'unsuccessful.json')['status'], 'unsuccessful')
            self.assertEqual(c.read(root/'results/bmmc6224/chunk_1234.json')['status'], 'checkpointed')

    def test_infrastructure_failure_never_skips_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(c, 'identity', return_value={}), patch.object(c, 'require_gate'), patch.object(c, 'check_storage', return_value=root), patch.dict(os.environ, {'SLURM_JOB_ID':'1234'}), patch.object(c.subprocess, 'run', return_value=subprocess.CompletedProcess([],1)) as run:
                with self.assertRaisesRegex(ValueError, 'classified'):
                    c.production(HERE.parents[1], root, HERE/'campaign.json', 'bmmc6224')
                self.assertEqual(run.call_count, 1)
            self.assertEqual(c.read(root/'results/bmmc6224/chunk_1234.json')['status'], 'failed')

    def test_dispatch_resumes_verified_timeout_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cursor = dict(task_index=2, task_id='test', iteration=100, has_checkpoint=True)
            with patch.dict(os.environ, {'SLURM_JOB_ID':'8888'}), patch.object(c, 'slurm_terminal_record', return_value=dict(job_id='1234',state='TIMEOUT',exit_code='0:15')), patch.object(c, 'check_storage', return_value=root), patch.object(c, 'checkpoint_cursor', return_value=cursor), patch.object(c, 'summarize'), patch.object(c, 'submit_job', side_effect=['9999','10000']) as submit:
                first = c.dispatch(HERE.parents[1], root, HERE/'campaign.json', 'bmmc6224', '1234')
                again = c.dispatch(HERE.parents[1], root, HERE/'campaign.json', 'bmmc6224', '1234')
                self.assertEqual(first, again)
                self.assertEqual(submit.call_count, 2)
                self.assertEqual(submit.call_args_list[0].kwargs['dependency'], 'afterok:8888')
                self.assertEqual(submit.call_args_list[1].kwargs['dependency'], 'afterany:9999')

    def test_dispatch_freezes_failed_scheduler_and_repeated_no_progress(self):
        for state in ('FAILED','OUT_OF_MEMORY','CANCELLED'):
            with self.subTest(state=state), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with patch.dict(os.environ, {'SLURM_JOB_ID':'8888'}), patch.object(c, 'slurm_terminal_record', return_value=dict(job_id='1234',state=state,exit_code='1:0')), patch.object(c, 'submit_job') as submit:
                    with self.assertRaisesRegex(ValueError, 'scheduler'):
                        c.dispatch(HERE.parents[1], root, HERE/'campaign.json', 'bmmc6224', '1234')
                    submit.assert_not_called()
                self.assertEqual(c.read(root/'results/bmmc6224/failed.json')['status'], 'failed')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cursor = dict(task_index=0, iteration=0, has_checkpoint=True)
            c.write(root/'results/bmmc6224/dispatch_state.json', dict(next_job_id='1234',cursor=cursor,no_progress_count=2))
            with patch.dict(os.environ, {'SLURM_JOB_ID':'8888'}), patch.object(c, 'slurm_terminal_record', return_value=dict(job_id='1234',state='NODE_FAIL',exit_code='0:0')), patch.object(c, 'check_storage', return_value=root), patch.object(c, 'checkpoint_cursor', return_value=cursor), patch.object(c, 'submit_job') as submit:
                with self.assertRaisesRegex(ValueError, 'no committed progress'):
                    c.dispatch(HERE.parents[1], root, HERE/'campaign.json', 'bmmc6224', '1234')
                submit.assert_not_called()

    def test_checkpoint_verification_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = c.tasks(self.config)[0]
            identity = self.identity(task, root)
            directory = root/'checkpoints/bmmc6224'/task['task_id']
            generation = directory/'generation-000000100-test'
            generation.mkdir(parents=True)
            shard = generation/'rank-0000.pt'
            shard.write_bytes(b'checkpoint')
            record = dict(world_size=1,iteration=100,identity=identity,generation=generation.name,
                          shards=[dict(file=shard.name,bytes=shard.stat().st_size,sha256=c.sha256(shard))])
            c.write(directory/'COMMITTED.json', record)
            c.write(generation/'manifest.json', record)
            cursor = c.checkpoint_cursor(root, root, self.config, 'bmmc6224', root/'checkpoints')
            self.assertEqual(cursor['iteration'],100)
            shard.write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                c.checkpoint_cursor(root, root, self.config, 'bmmc6224', root/'checkpoints')

    def test_sbatch_request_cannot_duplicate_or_repeat_ambiguous_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = subprocess.CompletedProcess([], 0, stdout='1234\n', stderr='')
            with patch.object(c.subprocess, 'run', return_value=fake) as call:
                a = c.submit_job(HERE.parents[1], root, 'production','bmmc6224',request_key='same')
                b = c.submit_job(HERE.parents[1], root, 'production','bmmc6224',request_key='same')
                self.assertEqual(a,b)
                self.assertEqual(call.call_count,1)
            request = next((root/'submission_requests').glob('*.json'))
            record = c.read(request)
            record['status'] = 'submitting'
            c.write(request,record)
            with patch.object(c.subprocess, 'run') as call:
                with self.assertRaisesRegex(ValueError,'Ambiguous'):
                    c.submit_job(HERE.parents[1], root, 'production','bmmc6224',request_key='same')
                call.assert_not_called()


if __name__ == '__main__':
    unittest.main()
