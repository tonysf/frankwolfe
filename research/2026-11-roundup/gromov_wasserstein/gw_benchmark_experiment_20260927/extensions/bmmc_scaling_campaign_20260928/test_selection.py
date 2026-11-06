"""CPU-only tests of the frozen cell plan and development-set selection policy."""
import copy
from collections import Counter
import importlib.util
import itertools
import json
from pathlib import Path
import random
import tempfile
import unittest

try:
    import numpy as np
except ImportError:
    np = None

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("bmmc_scaling_campaign", HERE / "campaign.py")
campaign = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(campaign)


def arguments(run):
    """Decode these value-bearing runner arguments without importing the runner."""
    argv = run["argv"][1:]
    return dict(zip(argv[::2], argv[1::2]))


class CampaignSelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((HERE / "campaign.json").read_text())
        self.records = [
            dict(n=n, method=method, candidate_id=cid, initialization=start,
                 seed=seed, eligible=True, loss=10.0 + int(cid[1:]))
            for n in self.config["sizes"]
            for method in self.config["methods"]
            for cid in self.config["candidates"][method]
            for start in self.config["initializations"]
            for seed in self.config["development_seeds"]
        ]
        self.first_block = (self.config["sizes"][0], self.config["methods"][0])

    def group(self, cid, block=None):
        n, method = block or self.first_block
        return [r for r in self.records
                if (r["n"], r["method"], r["candidate_id"]) == (n, method, cid)]

    def selection(self):
        return campaign.choose_candidates(self.records, self.config)

    def chosen(self, result, block=None):
        n, method = block or self.first_block
        return result["selected"][str(n)][method]["candidate_id"]

    def tuning_tasks(self):
        return [campaign.task_specs(self.config, "tuning", task)
                for task in range(self.config["task_counts"]["tuning"])]

    def final_tasks(self):
        selected = self.selection()
        return [campaign.task_specs(self.config, "final", task, selected)
                for task in range(self.config["task_counts"]["final"])]

    def test_tuning_has_every_prescribed_cell_once(self):
        tasks = self.tuning_tasks()
        runs = list(itertools.chain.from_iterable(tasks))
        self.assertEqual(len(tasks), 24)
        self.assertTrue(all(len(task) == 12 for task in tasks))
        self.assertEqual(len(runs), 288)
        self.assertEqual(len(runs), self.config["run_counts"]["tuning"])
        self.assertEqual(len({r["run_id"] for r in runs}), 288)
        actual = {(r["n"], r["method"], r["candidate_id"], r["initialization"], r["seed"])
                  for r in runs}
        expected = {(r["n"], r["method"], r["candidate_id"], r["initialization"], r["seed"])
                    for r in self.records}
        self.assertEqual(actual, expected)

    def test_final_has_every_prescribed_cell_once_and_no_development_seeds(self):
        tasks = self.final_tasks()
        runs = list(itertools.chain.from_iterable(tasks))
        self.assertEqual(len(tasks), 20)
        self.assertTrue(all(len(task) == 2 for task in tasks))
        self.assertEqual(len(runs), 40)
        self.assertEqual(len(runs), self.config["run_counts"]["final"])
        self.assertEqual(len({r["run_id"] for r in runs}), 40)
        expected = set(itertools.product(self.config["sizes"], self.config["methods"],
                                        self.config["initializations"], self.config["final_seeds"]))
        actual = {(r["n"], r["method"], r["initialization"], r["seed"]) for r in runs}
        self.assertEqual(actual, expected)
        self.assertTrue(set(self.config["development_seeds"]).isdisjoint(
            self.config["final_seeds"]))

    def test_selection_is_independent_per_size_and_method(self):
        blocks = list(itertools.product(self.config["sizes"], self.config["methods"]))
        winners = dict(zip(blocks, ["c03", "c07", "c02", "c11"]))
        for block, cid in winners.items():
            for record in self.group(cid, block):
                record["loss"] = 0.25
        random.Random(5).shuffle(self.records)
        result = self.selection()
        for block, cid in winners.items():
            self.assertEqual(self.chosen(result, block), cid)
        for task in range(self.config["task_counts"]["final"]):
            for run in campaign.task_specs(self.config, "final", task, result):
                cid = winners[(run["n"], run["method"])]
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
                     if (s["n"], s["method"], s["candidate_id"]) == (*self.first_block, "c00"))
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
            if (record["n"], record["method"]) == self.first_block:
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
                    if (s["n"], s["method"], s["candidate_id"]) == (*self.first_block, "c00")))

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


if __name__ == "__main__":
    unittest.main()
