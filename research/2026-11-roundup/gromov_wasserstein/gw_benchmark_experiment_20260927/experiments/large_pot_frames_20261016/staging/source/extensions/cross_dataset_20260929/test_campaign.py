"""CPU-only tests of the frozen cell plan and development-set selection policy."""
import copy
from collections import Counter
import importlib.util
import itertools
import json
from pathlib import Path
import random
import tempfile
import sys
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("cross_dataset_campaign", HERE / "campaign.py")
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def fixture_config():
    # Equal dimensions deliberately ensure that selection is keyed by dataset ID.
    fw = dict(('c'+str(i).zfill(2), dict(gamma0=g, mu_scale=m))
              for i, (g, m) in enumerate(itertools.product((.1, .3, 1.), (.1, 1., 10., 100.))))
    kl = dict(('c'+str(i).zfill(2), dict(eta=e, eta_power=0., inner_tol=t))
              for i, (e, t) in enumerate(itertools.product((.1, 1., 3., 10., 30., 100.), (1e-4, 1e-6))))
    return dict(instances=[dict(id='rna_atac', n=2, filename='rna_atac.npz', sha256='a'*64,
                               correspondence_file='rna_atac_matches.npz', correspondence_sha256='b'*64),
                           dict(id='shapes', n=2, filename='shapes.npz', sha256='c'*64)],
                methods=['smoothing-fw', 'stochastic-kl-momentum'], initializations=['product', 'randomized'],
                development_seeds=[10,11,12], final_seeds=list(range(300,305)),
                budgets=dict(tuning=60, final=300), candidates={'smoothing-fw': fw, 'stochastic-kl-momentum': kl},
                data_root='/frozen/data', runtime_commit='d'*40,
                pilot_runner_sha256='9'*64, runtime_file_sha256={'run_benchmark.py': 'e'*64}, selection_rule='median all six cells',
                shared=dict(oracle='sketch', batch=128, rho0=.3),loss_target_fractions=[.99,.97,.95,.90])


def arguments(run):
    """Decode these value-bearing runner arguments without importing the runner."""
    argv = run["argv"][1:]
    return dict(zip(argv[::2], argv[1::2]))


class CampaignSelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = fixture_config()
        self.records = [
            dict(instance_id=instance["id"], n=instance["n"], method=method, candidate_id=cid, initialization=start,
                 seed=seed, eligible=True, loss=10.0 + int(cid[1:]))
            for instance in self.config["instances"]
            for method in self.config["methods"]
            for cid in self.config["candidates"][method]
            for start in self.config["initializations"]
            for seed in self.config["development_seeds"]
        ]
        self.first_block = (self.config["instances"][0]["id"], self.config["methods"][0])

    def group(self, cid, block=None):
        n, method = block or self.first_block
        return [r for r in self.records
                if (r["instance_id"], r["method"], r["candidate_id"]) == (n, method, cid)]

    def selection(self):
        return campaign.choose_candidates(self.records, self.config)

    def chosen(self, result, block=None):
        n, method = block or self.first_block
        return result["selected"][str(n)][method]["candidate_id"]

    def tuning_tasks(self):
        return [campaign.task_specs(self.config, "tuning", task)
                for task in range(campaign.task_count(self.config, "tuning"))]

    def final_tasks(self):
        selected = self.selection()
        return [campaign.task_specs(self.config, "final", task, selected)
                for task in range(campaign.task_count(self.config, "final"))]

    def test_tuning_has_every_prescribed_cell_once(self):
        tasks = self.tuning_tasks()
        runs = list(itertools.chain.from_iterable(tasks))
        self.assertEqual(len(tasks), 24)
        self.assertTrue(all(len(task) == 12 for task in tasks))
        self.assertEqual(len(runs), 288)
        self.assertEqual(len(runs), 288)
        self.assertEqual(len({r["run_id"] for r in runs}), 288)
        actual = {(r["instance_id"], r["method"], r["candidate_id"], r["initialization"], r["seed"])
                  for r in runs}
        expected = {(r["instance_id"], r["method"], r["candidate_id"], r["initialization"], r["seed"])
                    for r in self.records}
        self.assertEqual(actual, expected)

    def test_final_has_every_prescribed_cell_once_and_no_development_seeds(self):
        tasks = self.final_tasks()
        runs = list(itertools.chain.from_iterable(tasks))
        self.assertEqual(len(tasks), 20)
        self.assertTrue(all(len(task) == 2 for task in tasks))
        self.assertEqual(len(runs), 40)
        self.assertEqual(len(runs), 40)
        self.assertEqual(len({r["run_id"] for r in runs}), 40)
        expected = set(itertools.product([i["id"] for i in self.config["instances"]], self.config["methods"],
                                        self.config["initializations"], self.config["final_seeds"]))
        actual = {(r["instance_id"], r["method"], r["initialization"], r["seed"]) for r in runs}
        self.assertEqual(actual, expected)
        self.assertTrue(set(self.config["development_seeds"]).isdisjoint(
            self.config["final_seeds"]))

    def test_selection_is_independent_per_instance_even_when_dimensions_match(self):
        blocks = list(itertools.product([i["id"] for i in self.config["instances"]], self.config["methods"]))
        winners = dict(zip(blocks, ["c03", "c07", "c02", "c11"]))
        for block, cid in winners.items():
            for record in self.group(cid, block):
                record["loss"] = 0.25
        random.Random(5).shuffle(self.records)
        result = self.selection()
        for block, cid in winners.items():
            self.assertEqual(self.chosen(result, block), cid)
        for task in range(campaign.task_count(self.config, "final")):
            for run in campaign.task_specs(self.config, "final", task, result):
                cid = winners[(run["instance_id"], run["method"])]
                self.assertEqual(run["candidate_id"], cid)
                self.assertEqual(run["parameters"], self.config["candidates"][run["method"]][cid])
                for key, value in run["parameters"].items():
                    self.assertEqual(arguments(run)["--" + key.replace("_", "-")], str(value))

    def test_paired_runs_share_inputs_oracle_momentum_and_solver_budget(self):
        for phase, tasks in (("tuning", self.tuning_tasks()), ("final", self.final_tasks())):
            with self.subTest(phase=phase):
                for task in tasks:
                    for left, right in zip(task[::2], task[1::2]):
                        self.assertEqual({left["method"], right["method"]}, set(self.config["methods"]))
                        for key in ("n", "initialization", "seed", "init_seed", "budget", "instance"):
                            self.assertEqual(left[key], right[key], key)
                        left_args, right_args = arguments(left), arguments(right)
                        for key in ("--initialization", "--seed", "--init-seed", "--budget",
                                    "--checkpoints", "--oracle", "--batch", "--rho0", "--dtype",
                                    "--device", "--threads", "--inner-max"):
                            self.assertEqual(left_args[key], right_args[key], key)
                        self.assertEqual(left_args["--oracle"], self.config["shared"]["oracle"])
                        self.assertEqual(int(left_args["--batch"]), self.config["shared"]["batch"])
                        self.assertEqual(float(left_args["--rho0"]), self.config["shared"]["rho0"])
                        self.assertEqual(left["budget"], self.config["budgets"][phase])
                        expected_init_seed = 100 if left["initialization"] == "product" else 900000 + left["seed"]
                        self.assertEqual(left["init_seed"], expected_init_seed)

    def test_method_order_is_balanced_in_each_phase(self):
        for phase, tasks in (("tuning", self.tuning_tasks()), ("final", self.final_tasks())):
            with self.subTest(phase=phase):
                first_methods = Counter(run["method"] for task in tasks for run in task[::2])
                expected_each = 72 if phase == "tuning" else 10
                self.assertEqual(first_methods, Counter({m: expected_each for m in self.config["methods"]}))

    def test_one_invalid_cell_disqualifies_whole_candidate_without_dropping_it(self):
        for record in self.group("c00"):
            record["loss"] = 0.0
        self.group("c00")[0]["eligible"] = False
        result = self.selection()
        self.assertEqual(self.chosen(result), "c01")
        score = next(s for s in result["scores"]
                     if (s["instance_id"], s["method"], s["candidate_id"]) == (*self.first_block, "c00"))
        self.assertFalse(score["eligible"])
        self.assertIsNone(score["score"])
        self.assertEqual(score["valid_count"], 5)
        self.assertEqual(len(result["scores"]), 48)

    def test_missing_cell_blocks_selection(self):
        self.records.pop()
        with self.assertRaisesRegex(ValueError, "every prescribed development cell exactly once"):
            self.selection()

    def test_duplicate_cell_blocks_selection_even_when_count_is_unchanged(self):
        self.records[-1] = dict(self.records[0])
        with self.assertRaisesRegex(ValueError, "every prescribed development cell exactly once"):
            self.selection()

    def test_unexpected_cell_blocks_selection(self):
        self.records[0]["seed"] = 123456
        with self.assertRaisesRegex(ValueError, "every prescribed development cell exactly once"):
            self.selection()

    def test_all_ineligible_in_one_block_blocks_selection(self):
        for record in self.records:
            if (record["instance_id"], record["method"]) == self.first_block:
                record["eligible"] = False
        with self.assertRaisesRegex(ValueError, "No eligible candidate for"):
            self.selection()

    def test_missing_or_nonfinite_loss_disqualifies_whole_candidate(self):
        original = copy.deepcopy(self.records)
        for invalid_loss in (None, float("nan"), float("inf"), -float("inf")):
            with self.subTest(loss=invalid_loss):
                self.records = copy.deepcopy(original)
                for record in self.group("c00"):
                    record["loss"] = 0.0
                self.group("c00")[0]["loss"] = invalid_loss
                result = self.selection()
                self.assertEqual(self.chosen(result), "c01")
                self.assertFalse(next(s["eligible"] for s in result["scores"]
                    if (s["instance_id"], s["method"], s["candidate_id"]) == (*self.first_block, "c00")))

    def test_exact_score_tie_uses_candidate_id_not_record_or_config_order(self):
        method = self.first_block[1]
        self.config["candidates"][method] = dict(reversed(list(self.config["candidates"][method].items())))
        self.records.reverse()
        for cid in ("c00", "c01"):
            for record in self.group(cid):
                record["loss"] = 0.0
        self.assertEqual(self.chosen(self.selection()), "c00")

    def test_score_is_median_over_all_six_start_seed_cells(self):
        for record, loss in zip(self.group("c00"), [0.0, 0.0, 0.0, 1.0, 1.0, 100.0]):
            record["loss"] = loss
        for record in self.group("c01"):
            record["loss"] = 1.0
        result = self.selection()
        self.assertEqual(self.chosen(result), "c00")
        self.assertEqual(result["selected"][str(self.first_block[0])][self.first_block[1]]["score"], 0.5)


@unittest.skipIf(np is None, "Saved-plan tests require NumPy; selection tests remain stdlib-only.")
class SavedOutputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="bmmc_saved_plan_test_")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "final.npz"
        # Each row has mass 1/2. The raw plan sends all mass to column 0;
        # the rounded plan has both prescribed uniform marginals.
        self.raw = np.array([[0.5, 0.0], [0.5, 0.0]], dtype=np.float32)
        self.rounded = np.array([[0.25, 0.25], [0.25, 0.25]], dtype=np.float32)
        self.metadata = {"final_float64_validation": {
            "raw": {"row_l1": 0.0, "col_l1": 1.0},
            "rounded": {"row_l1": 0.0, "col_l1": 0.0},
        }}

    def save(self, raw=None, rounded=None):
        np.savez(self.path, P_raw=self.raw if raw is None else raw,
                 P_rounded=self.rounded if rounded is None else rounded)

    def validate(self):
        return campaign.validate_saved_output(self.path, 2, self.metadata)

    def test_valid_output_allows_unrepaired_raw_column_marginal(self):
        self.save()
        audit = self.validate()
        self.assertEqual(audit["sha256"], campaign.digest(self.path))
        self.assertEqual(audit["arrays"]["P_raw"]["col_l1"], 1.0)
        self.assertEqual(audit["arrays"]["P_rounded"]["row_l1"], 0.0)
        self.assertEqual(audit["arrays"]["P_rounded"]["col_l1"], 0.0)

    def test_valid_output_accepts_a_sparse_feasible_coupling(self):
        self.save(rounded=np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.float32))
        self.validate()

    def test_missing_rounded_plan_is_rejected(self):
        np.savez(self.path, P_raw=self.raw)
        with self.assertRaises(ValueError):
            self.validate()

    def test_wrong_shape_is_rejected_for_either_saved_plan(self):
        for name in ("raw", "rounded"):
            with self.subTest(plan=name):
                self.save(**{name: np.full((2, 3), 0.25, dtype=np.float32)})
                with self.assertRaises(ValueError):
                    self.validate()

    def test_wrong_dtype_is_rejected_for_either_saved_plan(self):
        for name in ("raw", "rounded"):
            with self.subTest(plan=name):
                self.save(**{name: getattr(self, name).astype(np.float64)})
                with self.assertRaises(ValueError):
                    self.validate()

    def test_nonfinite_and_negative_entries_are_rejected_in_either_plan(self):
        for name in ("raw", "rounded"):
            for value in (float("nan"), float("inf"), -float("inf"), -0.25):
                with self.subTest(plan=name, invalid_value=value):
                    bad = getattr(self, name).copy()
                    bad[0, 0] = value
                    self.save(**{name: bad})
                    with self.assertRaises(ValueError):
                        self.validate()

    def test_infeasible_rounded_plan_is_rejected_even_with_truthful_metadata(self):
        cases = [
            # Correct rows, wrong columns; mass and nonnegativity alone do not suffice.
            ([[0.5, 0.0], [0.5, 0.0]], 0.0, 1.0),
            # Correct columns, wrong rows.
            ([[0.5, 0.5], [0.0, 0.0]], 1.0, 0.0),
        ]
        for plan, row_l1, col_l1 in cases:
            with self.subTest(row_l1=row_l1, col_l1=col_l1):
                self.metadata["final_float64_validation"]["rounded"] = {
                    "row_l1": row_l1, "col_l1": col_l1}
                self.save(rounded=np.array(plan, dtype=np.float32))
                with self.assertRaisesRegex(ValueError, "infeasible"):
                    self.validate()

    def test_residual_metadata_must_match_both_marginals_of_each_saved_plan(self):
        original = copy.deepcopy(self.metadata)
        for plan in ("raw", "rounded"):
            for marginal in ("row_l1", "col_l1"):
                with self.subTest(plan=plan, marginal=marginal):
                    self.metadata = copy.deepcopy(original)
                    self.metadata["final_float64_validation"][plan][marginal] += 0.125
                    self.save()
                    with self.assertRaisesRegex(ValueError, "disagree"):
                        self.validate()


def metadata_for(run, config):
    args = {key.removeprefix('--').replace('-', '_'): value for key, value in arguments(run).items()}
    for key in ('seed', 'init_seed', 'batch', 'threads', 'inner_max', 'gap_max_n'):
        args[key] = int(args[key])
    for key in ('budget', 'rho0', 'gamma0', 'mu_scale', 'eta', 'eta_power', 'inner_tol'):
        if key in args:
            args[key] = float(args[key])
    return dict(source_commit=config['runtime_commit'], source_dirty=False,
        source_file_sha256=config['runtime_file_sha256'], instance_sha256=run['instance_sha256'],
        correspondence_sha256=run.get('correspondence_sha256'), config=args,
        theory_config=dict(alpha=.75,beta=.5,delta=.25), status='completed',
        final_float64_validation=dict(raw_loss=1.,rounded_loss=1.,
            raw=dict(row_l1=0.,col_l1=0.,both_l1=0.), rounded=dict(row_l1=0.,col_l1=0.,both_l1=0.)),
        counters=dict(completed_updates=10),timing=dict(solver_seconds=run['budget']),
        memory=dict(cuda_peak_reserved_bytes=2**30,cpu_process_peak_rss_kib=2**20),
        array_sha256={'C1':'1'*64,'C2':'2'*64},initialization_plan_sha256='f'*64,
        environment={**{key:'same' for key in ('hostname','platform','packages','cuda_runtime','nvidia_smi')},
            'gpu':dict(name='NVIDIA A100',total_memory_bytes=40*1024**3)})


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.config = fixture_config()
        self.run = campaign.task_specs(self.config, 'tuning', 0)[0]
        self.metadata = metadata_for(self.run, self.config)

    def test_valid_metadata_and_correspondence_binding(self):
        self.assertTrue(campaign.validate_metadata(self.metadata,self.run,self.config))
        self.assertEqual(arguments(self.run)['--correspondence-file'], '/frozen/data/rna_atac_matches.npz')
        unpaired = campaign.task_specs(self.config,'tuning',12)[0]
        self.assertNotIn('--correspondence-file', unpaired['argv'])

    def test_source_data_labels_schedule_or_checkpoint_mutation_is_rejected(self):
        mutations = [lambda m:m.update(source_commit='changed'),lambda m:m.update(source_dirty=True),
            lambda m:m.update(source_file_sha256={}),lambda m:m.update(instance_sha256='changed'),
            lambda m:m.update(correspondence_sha256='changed'),
            lambda m:m['config'].update(correspondence_file='/other/matches.npz'),
            lambda m:m['theory_config'].update(alpha=.5),lambda m:m['config'].update(checkpoints='1')]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                metadata = copy.deepcopy(self.metadata); mutation(metadata)
                with self.assertRaises(ValueError):
                    campaign.validate_metadata(metadata,self.run,self.config)

    def test_instance_id_path_and_data_hash_must_be_frozen(self):
        for field, value in [('id','../escape'),('filename','../escape.npz'),('sha256',''),
                             ('correspondence_file','/absolute.npz')]:
            with self.subTest(field=field):
                config = copy.deepcopy(self.config);config['instances'][0][field]=value
                with self.assertRaises(ValueError):
                    campaign.task_specs(config,'tuning',0)

    def test_pilot_seed_is_disjoint_and_parameters_do_not_use_selection(self):
        runs = campaign.pilot_specs(self.config)
        self.assertEqual(len(runs),4)
        for run in runs:
            self.assertEqual((run['seed'],run['budget'],run['initialization']),(250,30,'product'))
            self.assertNotIn(run['seed'],self.config['development_seeds']+self.config['final_seeds'])
        self.assertEqual(runs[1]['parameters']['eta'],.1)

    def test_full_final_checkpoints_and_dynamic_counts(self):
        self.config['instances'].append(dict(id='third',n=3,filename='third.npz',sha256='f'*64))
        self.assertEqual(campaign.task_count(self.config,'tuning'),36)
        self.assertEqual(campaign.task_count(self.config,'final'),30)
        selection={'selected':{i['id']:{m:{'candidate_id':'c00'} for m in self.config['methods']}
                               for i in self.config['instances']}}
        runs=campaign.task_specs(self.config,'final',29,selection)
        self.assertEqual(runs[0]['instance_id'],'third')
        self.assertEqual(arguments(runs[0])['--checkpoints'],'10,30,100,300')
        with self.assertRaises(ValueError):
            campaign.task_specs(self.config,'final',30,selection)


@unittest.skipIf(np is None, 'Pilot audits require NumPy.')
class PilotGateTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory(prefix='dataset_pilot_gate_')
        self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name)
        self.config=fixture_config()
        self.config.update(config_path=str(self.root/'campaign.json'),data_root=str(self.root/'data'),
                           pilot_output=str(self.root/'pilot'))
        Path(self.config['data_root']).mkdir()
        Path(self.config['pilot_output']).mkdir()
        for instance in self.config['instances']:
            for key,hash_key in [('filename','sha256'),('correspondence_file','correspondence_sha256')]:
                if key in instance:
                    path=Path(self.config['data_root'])/instance[key]
                    path.write_bytes(instance[key].encode())
                    instance[hash_key]=campaign.digest(path)
        campaign.write(Path(self.config['config_path']),self.config)
        records=[]
        for i,run in enumerate(campaign.pilot_specs(self.config)):
            metadata=metadata_for(run,self.config)
            folder=Path(self.config['pilot_output'])/str(i);folder.mkdir()
            campaign.write(folder/'metadata.json',metadata)
            P=np.full((2,2),.25,dtype=np.float32)
            np.savez(folder/'final.npz',P_raw=P,P_rounded=P)
            (folder/'trace.csv').write_text('kind,rounded_loss\ninitial,1.0\n')
            records.append(dict(run_spec=run,metadata_path=str(i)+'/metadata.json',
                trace_path=str(i)+'/trace.csv',trace_sha256=campaign.digest(folder/'trace.csv'),initial_rounded_loss=1.,
                metadata_sha256=campaign.digest(folder/'metadata.json'),final_path=str(i)+'/final.npz',
                final_sha256=campaign.digest(folder/'final.npz'),returncode=0,
                resource_gate_passed=True,memory=metadata['memory']))
        self.state=dict(status='completed',runtime_commit=self.config['runtime_commit'],
            config_sha256=campaign.config_digest(self.config),controller_sha256=campaign.digest(campaign.__file__),
            records=records,job_id='123',resource_gate_passed=True,pilot_runner_sha256=self.config['pilot_runner_sha256'],common_loss_targets={i['id']:dict(baseline_loss=1.,
                fractions=[.99,.97,.95,.90],losses=[.99,.97,.95,.90]) for i in self.config['instances']})
        self.save_state()

    def save_state(self):
        campaign.write(Path(self.config['pilot_output'])/'pilot.json',self.state)

    def test_complete_pilot_and_eligibility_bindings_pass(self):
        hashes=campaign.pilot_guard(self.config)
        self.assertEqual(len(hashes),3)
        campaign.write(self.root/'eligibility.json',dict(status='passed',input_sha256=hashes,
            config_sha256=campaign.config_digest(self.config),controller_sha256=campaign.digest(campaign.__file__),
            pilot_sha256=campaign.digest(Path(self.config['pilot_output'])/'pilot.json')))
        campaign.eligibility_guard(self.config,self.root)
        self.state['job_id']='456';self.save_state()
        with self.assertRaisesRegex(ValueError,'eligibility binding'):
            campaign.eligibility_guard(self.config,self.root)

    def test_missing_pilot_instance_cannot_be_dropped(self):
        self.state['records'].pop();self.save_state()
        with self.assertRaisesRegex(ValueError,'complete prescribed'):
            campaign.pilot_guard(self.config)

    def test_pilot_launcher_binding_and_resource_flags_are_required(self):
        original=copy.deepcopy(self.state)
        for mutation in (lambda s:s.update(pilot_runner_sha256='changed'),
                         lambda s:s.update(resource_gate_passed=False),
                         lambda s:s['records'][0].update(resource_gate_passed=False)):
            with self.subTest(mutation=mutation):
                self.state=copy.deepcopy(original);mutation(self.state);self.save_state()
                with self.assertRaises(ValueError):
                    campaign.pilot_guard(self.config)

    def test_recorded_wrong_gpu_or_excess_memory_blocks_despite_true_flags(self):
        record=self.state['records'][0]
        path=Path(self.config['pilot_output'])/record['metadata_path']
        original=json.loads(path.read_text())
        for mutation in (lambda m:m['environment']['gpu'].update(name='other GPU'),
                         lambda m:m['memory'].update(cuda_peak_reserved_bytes=35*1024**3),
                         lambda m:m['memory'].update(cpu_process_peak_rss_kib=30*1024**2)):
            with self.subTest(mutation=mutation):
                metadata=copy.deepcopy(original);mutation(metadata);campaign.write(path,metadata)
                record['metadata_sha256']=campaign.digest(path);record['memory']=metadata['memory'];self.save_state()
                with self.assertRaisesRegex(ValueError,'resource gate'):
                    campaign.pilot_guard(self.config)

    def test_modified_loss_target_is_rejected(self):
        self.state['common_loss_targets']['rna_atac']['losses'][0]=.01;self.save_state()
        with self.assertRaisesRegex(ValueError,'Common loss targets differ'):
            campaign.pilot_guard(self.config)

    def test_modified_pilot_trace_is_rejected(self):
        (Path(self.config['pilot_output'])/'0/trace.csv').write_text('kind,rounded_loss\ninitial,.5\n')
        with self.assertRaisesRegex(ValueError,'trace binding'):
            campaign.pilot_guard(self.config)

    def test_changed_correspondence_file_blocks_gate(self):
        (Path(self.config['data_root'])/self.config['instances'][0]['correspondence_file']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'Frozen input changed'):
            campaign.pilot_guard(self.config)

    def test_changed_pilot_metadata_blocks_gate(self):
        (Path(self.config['pilot_output'])/'0/metadata.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'metadata binding'):
            campaign.pilot_guard(self.config)

    def test_invalid_saved_plan_blocks_gate_even_when_hash_is_updated(self):
        path=Path(self.config['pilot_output'])/'0/final.npz'
        P=np.zeros((2,2),dtype=np.float32)
        np.savez(path,P_raw=P,P_rounded=P)
        self.state['records'][0]['final_sha256']=campaign.digest(path);self.save_state()
        with self.assertRaisesRegex(ValueError,'disagree'):
            campaign.pilot_guard(self.config)

    def test_pair_environment_or_initial_plan_mismatch_blocks_gate(self):
        record=self.state['records'][1]
        path=Path(self.config['pilot_output'])/record['metadata_path']
        original=json.loads(path.read_text())
        for key in ('environment','initialization_plan_sha256'):
            with self.subTest(key=key):
                metadata=copy.deepcopy(original)
                if key=='environment':metadata[key]['hostname']='other-host'
                else:metadata[key]='other-plan'
                campaign.write(path,metadata);record['metadata_sha256']=campaign.digest(path);self.save_state()
                with self.assertRaisesRegex(ValueError,'Paired'):
                    campaign.pilot_guard(self.config)


class ReportObservationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('cross_dataset_report',HERE/'report.py')
        cls.report=importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules,{'campaign':campaign}):
            spec.loader.exec_module(cls.report)

    @staticmethod
    def trace(losses, times=None):
        times=times or [0.2,10.4,30.6,100.8][:len(losses)]
        return [dict(kind='initial' if i==0 else 'checkpoint',iteration=i*10,rounded_loss=loss,
                     actual_solver_seconds=max(0,t-.2),time_to_this_feasible_output=t)
                for i,(loss,t) in enumerate(zip(losses,times))]

    def test_first_recorded_nonmonotone_attainment_and_previous_observation(self):
        result=self.report.observed_attainment(self.trace([1.,.9,.95,.7]),.92)
        self.assertEqual(result['status'],'attained')
        self.assertEqual(result['first_observed_seconds'],10.4)
        self.assertEqual(result['previous_observation']['rounded_loss'],1.)
        self.assertFalse(result['censored'])

    def test_initial_attainment_keeps_actual_recorded_repair_time(self):
        result=self.report.observed_attainment(self.trace([.8,.7]),.9)
        self.assertTrue(result['initial'])
        self.assertEqual(result['first_observed_seconds'],.2)
        self.assertIsNone(result['previous_observation'])

    def test_missing_checkpoint_never_invents_a_crossing(self):
        trace=self.trace([1.,.95,.93])
        trace.insert(1,dict(kind='missed',iteration=5,rounded_loss=.1,time_to_this_feasible_output=3.))
        result=self.report.observed_attainment(trace,.9)
        self.assertEqual(result['status'],'not_observed')
        self.assertTrue(result['censored'])
        self.assertEqual(result['last_observation']['solver_plus_repair_seconds'],30.6)

    def test_missing_trace_is_unavailable_not_censored(self):
        result=self.report.observed_attainment([],.9)
        self.assertEqual(result['status'],'missing_trace')
        self.assertFalse(result['censored'])

    def test_missing_runs_remain_in_attainment_denominator(self):
        config=fixture_config()
        targets=dict(valid=True,targets={i['id']:dict(fractions=[.9],losses=[.9]) for i in config['instances']})
        record=dict(instance_id='rna_atac',initialization='product',method='smoothing-fw',seed=300,
                    eligible=True,trace=self.trace([1.,.8]))
        rows,groups=self.report.attainment_summary(config,[record],targets)
        group=next(g for g in groups if g['instance_id']=='rna_atac' and g['initialization']=='product' and g['method']=='smoothing-fw')
        self.assertEqual(group['attained_count'],1)
        self.assertEqual(group['counts']['unavailable'],4)
        self.assertEqual(group['first_observed_seconds_among_attained']['expected_count'],5)
        self.assertEqual(group['censored_count'],0)

    def test_correspondence_metrics_use_current_trace_not_selection_loss(self):
        trace=self.trace([1.,.8])
        trace[-1].update(correspondence_top1_accuracy=.7,correspondence_rank_fraction=.1,correspondence_true_pair_mass=.3)
        metrics=self.report.endpoint_metrics(dict(trace=trace,final_float64_validation=dict(rounded_loss=.79)))
        self.assertEqual(metrics['rounded_loss_float64'],.79)
        self.assertEqual(metrics['correspondence_top1_accuracy'],.7)

    def test_empty_campaign_produces_only_missing_counts_and_null_metrics(self):
        config=fixture_config()
        with tempfile.TemporaryDirectory(prefix='dataset_report_empty_') as directory:
            root=Path(directory)
            config.update(config_path=str(root/'campaign.json'),pilot_output=str(root/'pilot'))
            campaign.write(Path(config['config_path']),config)
            tasks,records,selection,issues=self.report.collect(config,root)
            self.assertEqual(len(tasks),20)
            self.assertTrue(all(t['status']=='missing' for t in tasks))
            self.assertFalse(selection['valid']);self.assertEqual(records,[])
            groups,curves=self.report.aggregate(config,records)
            self.assertTrue(all(g['valid_count']==0 for g in groups))
            self.assertTrue(all(m['median'] is None for g in groups for m in g['metrics'].values()))
            self.assertEqual(self.report.plot(config,curves,root),[])
            self.assertFalse(self.report.common_targets(config)['valid'])


if __name__ == "__main__":
    unittest.main()
