"""Launch/selection contracts; the shared runner has separate numerical tests."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('scot_comparison_contract', ROOT / 'comparison.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class ComparisonContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='scot_comparison_')
        self.base = Path(self.tmp.name)
        self.config_path = self.base / 'config.json'
        self.config_path.write_bytes((ROOT / 'config.json').read_bytes())
        self.config = c.load_config(self.config_path)
        self.source = ROOT.parents[1]
        self.prepared = dict(manifest_sha256='a' * 64, array_files_sha256={name: 'b' * 64 for name in c.pair.ARRAYS})
        self.binding = dict(config_sha256=c.sha256(self.config_path), reference_sha256='c' * 64)

    def tearDown(self):
        self.tmp.cleanup()

    def write_config(self, value):
        value = copy.deepcopy(value)
        value.pop('_config_path', None)
        c.write(self.config_path, value)

    def test_candidate_and_fresh_seed_mapping_preserves_runner_defaults(self):
        seen = []
        for index in range(27):
            method, candidate = c.candidate_at(self.config, index)
            self.assertEqual(method, c.METHODS[index // 9])
            seen.append(candidate['id'])
            lc = c.legacy_config(self.config, self.prepared, method, candidate, 'tune')
            values = c.pair.arguments(lc, self.base / 'prepared', self.base / 'output', 'scot1047', method, 810, 'production')
            expected = c.pair.expected_config(self.source, values)
            self.assertEqual(expected['iterations'], 20000)
            self.assertEqual((expected['seed'], expected['init_seed'], expected['output_seed']), (810, 10810, 20810))
            self.assertEqual((expected['oracle'], expected['pairs_per_call']), ('pair-sampling', 1024))
            self.assertEqual((expected['inner_tol'], expected['inner_gap_tol']), (1e-7, 1e-7))
            self.assertEqual(expected['initialization'], 'product')
            self.assertEqual(expected['save_plans'], 'none')
            for name, value in candidate.items():
                if name != 'id':
                    self.assertEqual(expected[name], value)
        self.assertEqual(len(set(seen)), 27)
        for index in (-1, 27):
            with self.assertRaises(ValueError):
                c.candidate_at(self.config, index)
        self.assertFalse({self.config['tuning_seed'], self.config['preflight']['seed']} & set(self.config['final_seeds']))
        self.assertEqual([(method, seed) for method in c.METHODS for seed in self.config['final_seeds']][8], ('published-msbpg', 813))

    def test_cannot_tune_exponents_change_tolerance_or_reuse_development_seeds(self):
        changes = [
            lambda x: x['methods']['smoothing-fw'][0].update(gamma_exponent=.5),
            lambda x: x['runner'].update(inner_gap_tol=1e-6),
            lambda x: x.update(final_seeds=[810, 811, 812]),
            lambda x: x.update(tuning_iterations=5000),
            lambda x: x['methods']['published-msbpg'][0].update(coupling_fraction=1.1),
            lambda x: x['methods']['published-smd'][0].update(prefactor_fraction=0),
        ]
        for change in changes:
            value = copy.deepcopy(self.config)
            change(value)
            self.write_config(value)
            with self.assertRaises(ValueError):
                c.load_config(self.config_path)

    def reference_fixture(self):
        reference = self.base / 'reference'
        prepared = reference / 'prepared/scot1047'
        prepared.mkdir(parents=True)
        for name in c.pair.ARRAYS:
            (prepared / name).write_bytes(('stand-in bytes for ' + name).encode())
        hashes = {name: c.sha256(prepared / name) for name in c.pair.ARRAYS}
        c.write(prepared / 'manifest.json', dict(n=1047, m=1047, array_files_sha256=hashes))
        record = dict(status='passed', prepared_instance=dict(relative_directory='prepared/scot1047',
                      manifest_sha256=c.sha256(prepared / 'manifest.json'), array_files_sha256=hashes))
        c.write(reference / 'reference.json', record)
        args = SimpleNamespace(config=str(self.config_path), source=str(self.source), reference=str(reference),
                               out=str(self.base / 'out'), candidate_index=0, cpu_checks='unused', gpu_checks='unused')
        return args, record, prepared

    def test_failed_reference_and_changed_arrays_block_execution(self):
        args, reference, directory = self.reference_fixture()
        verifier = SimpleNamespace(verify_reference=lambda path: reference)
        with mock.patch.object(c, 'module', return_value=verifier), mock.patch.object(c.pair, 'run_cell') as run:
            context = c.context(args)
            self.assertEqual(context[2], directory.parent)
            reference['status'] = 'failed_reproduction_gate'
            with self.assertRaisesRegex(ValueError, 'did not reproduce'):
                c.preflight(args)
            run.assert_not_called()
            reference['status'] = 'passed'
            (directory / 'A.npy').write_bytes(b'changed geometry')
            with self.assertRaisesRegex(ValueError, 'array files changed'):
                c.context(args)

    def test_missing_or_failed_tuning_never_creates_selection(self):
        args = SimpleNamespace(out=str(self.base / 'out'))
        Path(args.out).mkdir()
        context = (self.config, self.source, self.base / 'prepared', self.prepared, self.binding)
        for failure in (FileNotFoundError('missing tuning cell'), ValueError('failed tuning cell')):
            with mock.patch.object(c, 'context', return_value=context), \
                    mock.patch.object(c, 'verify_gate', return_value={}), \
                    mock.patch.object(c, 'check_cell', side_effect=failure):
                with self.assertRaises(type(failure)):
                    c.select(args)
                self.assertFalse((Path(args.out) / 'selection.json').exists())

    def selection_fixture(self):
        out = self.base / 'comparison'
        records = []
        for index, (method, candidate) in enumerate(c.candidates(self.config)):
            target = c.cell_directory(out, 'tune', method, candidate, 810)
            (target / 'output').mkdir(parents=True)
            metadata = target / 'output/metadata.json'
            c.write(metadata, dict(test_fixture=True, candidate=candidate['id']))
            c.write(target / 'request.json', dict(test_fixture=True, method=method))
            loss = .02 + (index % 9) * .001
            c.write(target / 'status.json', dict(status='completed', metadata_sha256=c.sha256(metadata),
                                                 request_sha256=c.sha256(target / 'request.json'), last_rounded_loss=loss))
            records.append(dict(method=method, candidate=candidate, seed=810, iterations=20000,
                                last_rounded_loss=loss, metadata_file=str(metadata), metadata_sha256=c.sha256(metadata),
                                status_sha256=c.sha256(target / 'status.json'), request_sha256=c.sha256(target / 'request.json')))
        selection = dict(status='completed', binding=self.binding, metric=self.config['selection_metric'],
                         tuning_seed=810, tuning_iterations=20000, candidates=records, selected=c.select_candidates(records))
        c.write(out / 'selection.json', selection)
        return out, selection

    def test_selection_requires_every_candidate_and_immutable_evidence(self):
        out, selection = self.selection_fixture()
        args = (self.config, self.source, self.base / 'prepared', self.prepared, out, self.binding)
        value, receipt = c.verify_selection(*args)
        self.assertEqual(value['selected']['smoothing-fw']['id'], 'fw00')
        self.assertEqual(receipt['sha256'], c.sha256(out / 'selection.json'))
        bad = copy.deepcopy(selection)
        bad['candidates'].pop()
        c.write(out / 'selection.json', bad)
        with self.assertRaisesRegex(ValueError, 'omitted or duplicated'):
            c.verify_selection(*args)
        c.write(out / 'selection.json', selection)
        Path(selection['candidates'][0]['metadata_file']).write_text('changed inner certificates')
        with self.assertRaisesRegex(ValueError, 'evidence changed'):
            c.verify_selection(*args)

    def test_common_feasible_loss_selection_and_stable_tie_break(self):
        records = [dict(method=method, candidate=candidate, last_rounded_loss=.01)
                   for method, candidate in reversed(c.candidates(self.config))]
        selected = c.select_candidates(records)
        self.assertEqual([selected[method]['id'] for method in c.METHODS], ['fw00', 'smd00', 'msbpg00'])
        next(row for row in records if row['candidate']['id'] == 'fw08')['last_rounded_loss'] = .001
        self.assertEqual(c.select_candidates(records)['smoothing-fw']['id'], 'fw08')

    def test_duplicate_terminal_diagnostics_do_not_duplicate_observations(self):
        trace = [dict(kind='initial', iteration=0, rounded_loss=.07, time_to_this_feasible_output=.01),
                 dict(kind='selected', iteration=1, rounded_loss=.06, time_to_this_feasible_output=.02),
                 dict(kind='current_diagnostic', iteration=20000, rounded_loss=.03, time_to_this_feasible_output=60.),
                 dict(kind='last', iteration=20000, rounded_loss=.03, time_to_this_feasible_output=60.01)]
        result = c.current_trajectory(dict(trace=trace))
        self.assertEqual([row['iteration'] for row in result], [0, 20000])
        self.assertEqual(result[-1]['time_to_this_feasible_output'], 60.01)
        trace[-1]['rounded_loss'] = .04
        with self.assertRaisesRegex(ValueError, 'disagree'):
            c.current_trajectory(dict(trace=trace))

    def test_after_any_report_preserves_failed_prerequisite_evidence(self):
        out = self.base / 'blocked'
        gate_path = out / 'preflight/fw00/gate.json'
        gate_path.parent.mkdir(parents=True)
        c.write(gate_path, dict(status='failed', error='deliberate runtime forecast failure'))
        args = SimpleNamespace(out=str(out), config=str(self.config_path), reference=str(self.base / 'missing-reference.json'))
        with mock.patch.object(c, 'completed_report', side_effect=FileNotFoundError('selection.json')):
            c.report(args)
        record = c.read(out / 'report.json')
        self.assertEqual(record['status'], 'blocked_dependency')
        self.assertEqual(record['stage_receipts'][0]['status'], 'failed')
        self.assertIn('runtime forecast', record['stage_receipts'][0]['error'])
        self.assertEqual(record['runs'], [])
        self.assertFalse((out / 'selection.json').exists())


if __name__ == '__main__':
    unittest.main()
