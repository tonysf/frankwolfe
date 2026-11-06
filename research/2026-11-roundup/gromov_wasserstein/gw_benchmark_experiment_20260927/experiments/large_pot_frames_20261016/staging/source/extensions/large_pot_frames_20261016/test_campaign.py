import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('large_campaign_tested', HERE / 'campaign.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.config = c.load_config(HERE / 'campaign.json')

    def result(self, task, dataset='bmmc6224', loss=.03):
        cfg = dict(method=task['method'], seed=task['seed'], iterations=task['iterations'],
                   dtype='float64', pairs=1024, epsilon=.001, inner_max=100000, diagnostic_every=500,
                   **task.get('prefactors', {}))
        instance = self.config['instances'][dataset]
        return dict(status='completed', completed_updates=task['iterations'], target_updates=task['iterations'],
                    final_loss=loss, config=cfg, array_files_sha256=instance['array_files_sha256'],
                    manifest_sha256=instance['instance_manifest_sha256'], dimensions=dict(n=instance['n'],m=instance['n']),
                    world_size=instance['world_size'], schedule=dict(inner_tolerance=1e-7,gap_tolerance=1e-7,eta=1000.,
                    gamma_exponent=.75,rho_exponent=.5,mu_exponent=.25))

    def test_grid_includes_old_controls_and_separates_development_confirmation(self):
        grid = c.candidates(self.config)
        self.assertEqual(len(grid), 24)
        for instance in self.config['instances'].values():
            self.assertIn(instance['old_prefactors'], [x['prefactors'] for x in grid])
        tasks = c.tasks(self.config, grid[3])
        self.assertEqual(len(tasks), 55)
        self.assertEqual(len({t['task_id'] for t in tasks}), 55)
        self.assertEqual(sum(t['method'] == 'pot-full' for t in tasks), 1)
        self.assertEqual(sum(t['method'] == 'pot-pair' for t in tasks), 3)
        self.assertEqual({t['seed'] for t in tasks if t['task_id'].startswith('development/')}, {810, 814})
        self.assertEqual({t['seed'] for t in tasks if t['task_id'].startswith('confirmation/')}, {811, 812, 813})
        self.assertTrue(all(t['iterations'] == 20000 for t in tasks if t['method'] == 'frames'))

    def test_selection_uses_both_complete_seed_losses_and_deterministic_tie(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'bmmc6224'
            for task in c.tasks(self.config):
                if task['method'] == 'frames':
                    # c00 and c01 share the winning median; c00 wins fixed ID tie break.
                    loss = .02 if task['candidate_id'] in ('c00', 'c01') else .04
                    c.write(root / task['task_id'] / 'metadata.json', self.result(task, loss=loss))
            selected = c.selection(self.config, root)
            self.assertEqual(selected['candidate_id'], 'c00')
            self.assertEqual(selected['development_seed_count'], 2)
            self.assertEqual(len(selected['ranking']), 24)
            task = next(t for t in c.tasks(self.config) if t.get('candidate_id') == 'c00')
            meta = self.result(task)
            meta['completed_updates'] -= 1
            c.write(root / task['task_id'] / 'metadata.json', meta)
            with self.assertRaisesRegex(ValueError, 'horizon'):
                c.selection(self.config, root)

    def test_reject_result_with_wrong_oracle_dtype_data_seed_or_prefactor(self):
        task = c.tasks(self.config)[4]
        for key, value in [('method', 'pot-full'), ('dtype', 'float32'), ('seed', 100), ('pairs', 1), ('mu_scale', 999)]:
            meta = self.result(task)
            meta['config'][key] = value
            with self.assertRaises(ValueError):
                c.validate_result(meta, task, self.config['instances']['bmmc6224'], self.config)
        meta = self.result(task)
        meta['array_files_sha256'] = {}
        with self.assertRaisesRegex(ValueError, 'dataset'):
            c.validate_result(meta, task, self.config['instances']['bmmc6224'], self.config)

    def test_runner_command_matches_topology_and_uses_external_checkpoint_root(self):
        task = c.tasks(self.config)[4]
        single = c.runner_command('/source', '/results', self.config, 'bmmc6224', task, 7000, '/checkpoints')
        four = c.runner_command('/source', '/results', self.config, 'shareseq_full', task, 7000, '/checkpoints')
        self.assertNotIn('torch.distributed.run', single)
        self.assertIn('--nproc-per-node=4', four)
        self.assertEqual(four[four.index('--max-seconds') + 1], '7000')
        self.assertEqual(four[four.index('--checkpoint-dir') + 1], '/checkpoints/shareseq_full/' + task['task_id'])
        self.assertEqual(four[four.index('--dtype') + 1], 'float64')
        self.assertNotIn('--retain-checkpoint', four)

    def test_quota_receipt_must_represent_quota_not_filesystem_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = dict(quota_verified=True, quota_available_bytes=60 * 1024 ** 3,
                           checkpoint_root=tmp, verified_utc=c.now())
            c.write(root / 'storage_receipt.json', receipt)
            self.assertEqual(c.check_storage(root, self.config, fresh=True), root)
            receipt['quota_available_bytes'] -= 1
            c.write(root / 'storage_receipt.json', receipt)
            with self.assertRaisesRegex(ValueError, 'headroom'):
                c.check_storage(root, self.config)
            receipt['quota_available_bytes'] += 1
            receipt['quota_verified'] = False
            c.write(root / 'storage_receipt.json', receipt)
            with self.assertRaisesRegex(ValueError, 'quota'):
                c.check_storage(root, self.config)

    def test_submission_dependencies_hold_production_behind_actual_size_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            jobs = []
            def fake_submit(*args, **kwargs):
                jobs.append((args, kwargs))
                return str(100 + len(jobs))
            with patch.object(c, 'check_source'), patch.object(c, 'check_storage'), patch.object(c, 'identity', return_value={}), patch.object(c, 'submit_job', side_effect=fake_submit):
                c.submit_initial(HERE.parents[1], tmp, HERE / 'campaign.json')
            self.assertEqual(len(jobs), 13)
            self.assertEqual(jobs[1][1]['dependency'], 'afterok:101')
            self.assertEqual(jobs[2][1]['dependency'], 'afterok:101')
            self.assertEqual(jobs[4][0][-1], 'afterok:104')
            self.assertEqual(jobs[7][0][-1], 'afterok:107')
            self.assertEqual(jobs[10][0][-1], 'afterok:110')
            self.assertEqual(jobs[5][1]['dependency'], 'afterany:105')
            self.assertTrue(jobs[12][1]['dependency'].startswith('afterany:'))
            with patch.object(c, 'check_source'), patch.object(c, 'check_storage'):
                with self.assertRaises(FileExistsError):
                    c.submit_initial(HERE.parents[1], tmp, HERE / 'campaign.json')

    def test_numerical_failure_is_terminal_and_never_submits_continuation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(c, 'identity', return_value={}), patch.object(c, 'require_gate'), patch.object(c, 'check_storage', return_value=root), patch.dict(os.environ, {'SLURM_JOB_ID': '1234'}), patch.object(c.subprocess, 'run', side_effect=RuntimeError('inner certificate failed')), patch.object(c, 'submit_job') as submit:
                with self.assertRaisesRegex(RuntimeError, 'inner certificate'):
                    c.production(HERE.parents[1], root, HERE / 'campaign.json', 'bmmc6224')
                submit.assert_not_called()
            failed = c.read(root / 'results/bmmc6224/chunk_1234.json')
            self.assertEqual(failed['status'], 'failed')
            self.assertIn('inner certificate', failed['error'])

    def test_gpu_chunk_does_not_submit_its_own_continuation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = c.tasks(self.config)[0]
            c.write(root / 'results/bmmc6224' / task['task_id'] / 'progress.json', {'status': 'checkpointed'})
            with patch.object(c, 'identity', return_value={}), patch.object(c, 'require_gate'), patch.object(c, 'check_storage', return_value=root), patch.dict(os.environ, {'SLURM_JOB_ID': '1234'}), patch.object(c.subprocess, 'run'), patch.object(c, 'submit_job', return_value='9999') as submit:
                c.production(HERE.parents[1], root, HERE / 'campaign.json', 'bmmc6224')
                submit.assert_not_called()
            receipt = c.read(root / 'results/bmmc6224/chunk_1234.json')
            self.assertEqual(receipt['status'], 'checkpointed')

    def test_dispatcher_resumes_verified_timeout_and_duplicate_event_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scheduler = dict(job_id='1234', state='TIMEOUT', exit_code='0:15')
            cursor = dict(task_index=0, task_id='baseline/pot-full_seed810', iteration=700, has_checkpoint=True)
            with patch.dict(os.environ, {'SLURM_JOB_ID': '8888'}), patch.object(c, 'slurm_terminal_record', return_value=scheduler), patch.object(c, 'check_storage', return_value=root), patch.object(c, 'checkpoint_cursor', return_value=cursor), patch.object(c, 'submit_job', side_effect=['9999', '10000']) as submit:
                decision = c.dispatch(HERE.parents[1], root, HERE / 'campaign.json', 'bmmc6224', '1234')
                duplicate = c.dispatch(HERE.parents[1], root, HERE / 'campaign.json', 'bmmc6224', '1234')
                self.assertEqual(decision, duplicate)
                self.assertEqual(submit.call_count, 2)
                self.assertEqual(submit.call_args_list[0].kwargs['dependency'], 'afterok:8888')
                self.assertEqual(submit.call_args_list[1].kwargs['dependency'], 'afterany:9999')
            self.assertEqual(decision['next_job_id'], '9999')
            self.assertEqual(decision['no_progress_count'], 0)

    def test_dispatcher_freezes_oom_failed_and_explicit_numerical_failure(self):
        for state, validation_error in [('OUT_OF_MEMORY', None), ('FAILED', None), ('TIMEOUT', ValueError('Explicit runner numerical failure'))]:
            with self.subTest(state=state), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                scheduler = dict(job_id='1234', state=state, exit_code='1:0')
                with patch.dict(os.environ, {'SLURM_JOB_ID': '8888'}), patch.object(c, 'slurm_terminal_record', return_value=scheduler), patch.object(c, 'check_storage', return_value=root), patch.object(c, 'checkpoint_cursor', side_effect=validation_error), patch.object(c, 'submit_job', return_value='10000') as submit:
                    with self.assertRaises(ValueError):
                        c.dispatch(HERE.parents[1], root, HERE / 'campaign.json', 'bmmc6224', '1234')
                    self.assertEqual(submit.call_count, 1)
                    self.assertEqual(submit.call_args.args[2], 'report')
                self.assertEqual(c.read(root / 'results/bmmc6224/failed.json')['status'], 'failed')

    def test_dispatcher_stops_repeated_no_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cursor = dict(task_index=0, task_id='baseline/pot-full_seed810', iteration=0, has_checkpoint=True)
            c.write(root / 'results/bmmc6224/dispatch_state.json', dict(next_job_id='1234', cursor=cursor, no_progress_count=2))
            with patch.dict(os.environ, {'SLURM_JOB_ID': '8888'}), patch.object(c, 'slurm_terminal_record', return_value=dict(job_id='1234', state='NODE_FAIL', exit_code='0:0')), patch.object(c, 'check_storage', return_value=root), patch.object(c, 'checkpoint_cursor', return_value=cursor), patch.object(c, 'submit_job', return_value='10000') as submit:
                with self.assertRaisesRegex(ValueError, 'no committed progress'):
                    c.dispatch(HERE.parents[1], root, HERE / 'campaign.json', 'bmmc6224', '1234')
                self.assertEqual(submit.call_count, 1)
                self.assertEqual(submit.call_args.args[2], 'report')

    def test_checkpoint_cursor_verifies_configuration_generation_and_shard_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            source.mkdir()
            (source / 'runner.py').write_text('frozen numerical implementation')
            task = c.tasks(self.config)[0]
            instance = self.config['instances']['bmmc6224']
            directory = root / 'checkpoints/bmmc6224' / task['task_id']
            generation = directory / 'generation-000000100-test'
            generation.mkdir(parents=True)
            shard = generation / 'rank-0000.pt'
            shard.write_bytes(b'pretend checkpoint tensors')
            identity = dict(config=dict(method=task['method'],seed=task['seed'],iterations=task['iterations'],epsilon=.001,
                pairs=1024,gamma0=1.,rho0=1.,mu_scale=10.,dtype='float64',device='cuda',inner_max=100000,
                diagnostic_every=500,exact_block_rows=128),source_file_sha256={'runner.py':c.sha256(source/'runner.py')},
                world_size=1,n=instance['n'],m=instance['n'],array_files_sha256=instance['array_files_sha256'],
                manifest_sha256=instance['instance_manifest_sha256'])
            record = dict(world_size=1,iteration=100,identity=identity,generation=generation.name,
                shards=[dict(file=shard.name,bytes=shard.stat().st_size,sha256=c.sha256(shard))])
            c.write(directory / 'COMMITTED.json', record)
            c.write(generation / 'manifest.json', record)
            cursor = c.checkpoint_cursor(source, root, self.config, 'bmmc6224', root / 'checkpoints')
            self.assertEqual(cursor['iteration'], 100)
            shard.write_bytes(b'corrupted tensors')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                c.checkpoint_cursor(source, root, self.config, 'bmmc6224', root / 'checkpoints')

    def test_sbatch_request_receipt_prevents_duplicate_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = type('Completed', (), dict(returncode=0, stdout='1234\n', stderr=''))()
            with patch.object(c.subprocess, 'run', return_value=fake) as call:
                first = c.submit_job(HERE.parents[1], root, 'production', 'bmmc6224', request_key='continuation-key')
                second = c.submit_job(HERE.parents[1], root, 'production', 'bmmc6224', request_key='continuation-key')
                self.assertEqual(first, second)
                self.assertEqual(call.call_count, 1)


if __name__ == '__main__':
    unittest.main()
