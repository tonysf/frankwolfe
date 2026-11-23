"""Failure-preserving gates bind probe work, dense directions, and source receipts."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c = load('_probe_campaign_test', ROOT/'campaign.py')
previous = load('_probe_previous_fixture', ROOT.parent/'batch_size_20261008/test_campaign.py')


def metadata(expected, provenance):
    value = previous.metadata(expected, provenance)
    T = expected['iterations']
    value['dimensions'] = dict(source=512, target=512)
    for plan in value['ranks'][0]['saved_plans'].values():
        plan.update(shape=[512,512], array_bytes=512*512*4)
    residuals = dict(row_l1=.001, col_l1=.1, col_l2=.01, both_l1=.101, mass_error=0., min_entry=0.)
    rounded = dict(row_l1=1e-8, col_l1=1e-8, col_l2=1e-9, both_l1=2e-8, mass_error=0., min_entry=0.)
    checkpoints = list(map(int, expected['checkpoint_iterations'].split(',')))
    value['trace'] = []
    for k in [0, *checkpoints]:
        value['trace'].append(dict(kind='initial' if k == 0 else 'last' if k == T else 'current_diagnostic',
            iteration=k, raw_loss=.15, rounded_loss=.2, raw_residuals=residuals, rounded_residuals=rounded,
            repair_l1=.1, actual_solver_seconds=k*.001+.1, repair_seconds_current=.001,
            time_to_this_feasible_output=k*.001+.101, actual_end_to_end_seconds=k*.001+2.))
    for name, plan in value['final_float64_validation']['plans'].items():
        plan['marginal_residuals'] = rounded if name.endswith('_rounded') else residuals
    value['direction_trace'] = []
    choices = dict(selected_column_count=2, largest_row_fraction=.6, largest_source_mass=.6)
    for k in [0, *checkpoints[:-1]]:
        value['direction_trace'].append(dict(iteration=k, plan_index=k, momentum_index=k,
            position='after momentum update, before FW step', optimizer_dtype='float32',
            full_gradient_dtype='float64', norm_dtype='float64', gamma=1/(k+1)**.75,
            rho_scheduled=.3/math.sqrt(k+1), rho_effective=1. if k == 0 else .3/math.sqrt(k+1),
            mu=10/512/(k+1)**.25, momentum_initialization_reuses_first_gradient=k == 0,
            residuals=residuals, penalty_value=.2, raw_structure_l1=.1,
            row_centered_momentum_norm=2., row_centered_full_gradient_norm=1., row_centered_penalty_norm=3.,
            row_centered_momentum_error_norm=2., penalty_to_momentum_ratio=1.5,
            row_centered_momentum_relative_error=2., lmo_agreement_with_without_penalty=.5,
            lmo_agreement_momentum_full_gradient_without_penalty=.6,
            lmo_agreement_momentum_full_gradient_with_penalty=.7,
            lmo_with_penalty=choices, lmo_without_penalty=choices,
            actual_solver_seconds=k*.001+.1001, actual_end_to_end_seconds=k*.001+2.1))
    value['timing'].update(solver_seconds=T*.001+.1, total_elapsed_seconds=T*.001+10.,
                          direction_diagnostic_seconds=.1, progress_logging_seconds=.01)
    value['progress_reporting'] = dict(completed_updates_recorded=[0,T])
    return value


class Fixture:
    def __init__(self, directory):
        self.root = Path(directory)
        self.source, self.prepared, self.out = (self.root/name for name in ('source','prepared','results'))
        self.path = self.root/'campaign.json'
        self.config = c.read_json(ROOT/'campaign.json')
        self.config.pop('baseline', None)
        for name in set((*c.SOURCE_FILES, *c.CONTROLLER_FILES)):
            path = self.source/'extensions'/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
        folder = self.prepared/c.DATASET
        folder.mkdir(parents=True)
        c.write_json(folder/'manifest.json', dict(array_files_sha256=self.config['instances'][c.DATASET]['array_files_sha256']))
        self.config['instances'][c.DATASET]['instance_manifest_sha256'] = c.sha256(folder/'manifest.json')
        c.write_json(self.path, self.config)
        provenance = c.identity(self.path, self.source, self.prepared, c.DATASET)
        c.write_json(self.source/'extensions/probe8192_20261009/local_validation.json',
                     dict(status='passed', source_file_sha256=provenance['source_file_sha256'],
                          tests=dict(passed=8, failures=0, errors=0)))

    def args(self, action='preflight', index=0):
        return argparse.Namespace(config=str(self.path), source=str(self.source), prepared=str(self.prepared),
            out=str(self.root/'preflight') if action == 'preflight' else str(self.out),
            instance=c.DATASET, task_index=index, gate=str(self.root/'preflight/gate.json'))

    def execute(self, command, log, timeout):
        if 'verify_gpu.py' in command[1]:
            out = Path(command[command.index('--out')+1])
            out.mkdir(parents=True)
            provenance = c.identity(self.path, self.source, self.prepared, c.DATASET)
            runs = {}
            for name in ('q2048_legacy','q2048_enabled','q8192_disabled','q8192_enabled'):
                path = out/(name+'.json')
                path.write_text('{}')
                runs[name] = dict(metadata_file=str(path), metadata_sha256=c.sha256(path))
            reference = {name: provenance['source_file_sha256'][name] for name in c.base.published.SOURCE_FILES}
            reference['batch_size_20261008/run_batch.py'] = provenance['controller_source_sha256']['batch_size_20261008/run_batch.py']
            c.write_json(out/'parity.json', dict(status='passed', checks={key:dict(status='passed')
                for key in ('q8192_enabled_disabled','q2048_legacy')}, runs=runs,
                source_file_sha256=provenance['source_file_sha256'], reference_source_file_sha256=reference,
                array_files_sha256=provenance['array_files_sha256'], instance_manifest_sha256=provenance['instance_manifest_sha256'],
                helper_source_sha256={name:provenance['controller_source_sha256']['probe8192_20261009/'+name]
                                      for name in ('verify_gpu.py','trajectory_parity.py')}))
        else:
            request = c.read_json(log.parent/'request.json')
            expected = request['expected_runner_config']
            out = Path(expected['out'])
            out.mkdir(parents=True)
            c.write_json(out/'metadata.json', metadata(expected, request))
        log.write_text('fixture execution')
        return 0


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='probe8192_test_')
        self.addCleanup(self.tmp.cleanup)
        self.f = Fixture(self.tmp.name)

    def test_gate_then_three_frozen_runs_and_tamper_detection(self):
        f = self.f
        self.assertEqual(c.preflight(f.args(), executor=f.execute)['status'], 'passed')
        config = c.load_config(f.path)
        for index, seed in enumerate((710,711,712)):
            c.run_task(f.args('run', index), executor=f.execute)
            result = c.validate_run(config, f.source, f.prepared, f.out, c.DATASET,8192,seed)
            self.assertEqual(result['metadata']['counters']['sketch_probes'], 8192*200000)
            self.assertEqual(len(result['metadata']['direction_trace']), 209)
        with self.assertRaises(FileExistsError):
            c.run_task(f.args('run'), executor=f.execute)
        source = f.source/'extensions/probe8192_20261009/direction_diagnostics.py'
        source.write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Preflight source'):
            c.validate_gate(config, f.source, f.prepared, f.args().gate)

    def test_missing_local_tests_or_failed_gpu_parity_blocks_launch(self):
        f = self.f
        path = f.source/'extensions/probe8192_20261009/local_validation.json'
        value = c.read_json(path)
        value['tests']['failures'] = 1
        c.write_json(path,value)
        with self.assertRaisesRegex(ValueError, 'local test evidence'):
            c.preflight(f.args(), executor=f.execute)
        self.assertEqual(c.read_json(Path(f.args().out)/'gate.json')['status'], 'failed')

    def test_dense_indices_ratios_and_walltimes_are_checked(self):
        f = self.f
        cfg = c.load_config(f.path)
        expected = c.runner_config(cfg,f.prepared,f.out,c.DATASET,8192,710,preflight=True)
        template = metadata(expected,c.identity(f.path,f.source,f.prepared,c.DATASET))
        for edit, text in ((lambda v:v['direction_trace'].pop(1), 'preupdate'),
            (lambda v:v['direction_trace'][1].update(penalty_to_momentum_ratio=8), 'norm ratio'),
            (lambda v:v['direction_trace'][1].update(momentum_index=99), 'index mismatch'),
            (lambda v:v['trace'][0].update(actual_end_to_end_seconds=-1), 'nonnegative')):
            value=copy.deepcopy(template);edit(value)
            with self.subTest(text=text), self.assertRaisesRegex(ValueError,text):
                c.validate_diagnostics(value,expected)

    def test_resource_gate_accounts_for_dense_diagnostics_and_never_loss(self):
        f = self.f
        cfg=c.load_config(f.path)
        expected=c.runner_config(cfg,f.prepared,f.out,c.DATASET,8192,710,preflight=True)
        value=metadata(expected,c.identity(f.path,f.source,f.prepared,c.DATASET))
        value['final_float64_validation']['plans']['last_rounded']['loss']=1e30
        self.assertTrue(c.resource_checks(cfg,value)['passed'])
        value['timing']['direction_diagnostic_seconds']=1000
        value['timing']['total_elapsed_seconds']=1100
        with self.assertRaisesRegex(ValueError,'runtime estimate'):c.resource_checks(cfg,value)


if __name__ == '__main__':
    unittest.main()
