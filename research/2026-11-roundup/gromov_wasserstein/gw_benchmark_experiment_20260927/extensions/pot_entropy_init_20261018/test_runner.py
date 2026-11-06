import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import ot
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("entropy_init_runner_tests", HERE/"runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fixture(n=5, *, zero=False):
    rng = np.random.default_rng(171)
    x,y = rng.normal(size=(n,2)), rng.normal(size=(n,2))
    A = torch.tensor(.01*np.linalg.norm(x[:,None]-x[None,:],axis=-1),dtype=torch.float64)
    B = torch.tensor(.01*np.linalg.norm(y[:,None]-y[None,:],axis=-1),dtype=torch.float64)
    if zero:
        A.zero_()
        B.zero_()
    a = torch.full((n,),1/n,dtype=torch.float64)
    return runner.RowShardedGW(A,B,a,a.clone(),exact_block_rows=2)


def write_instance(path, *, zero=False):
    path.mkdir()
    p = fixture(zero=zero)
    arrays = dict(A=p.A.numpy().astype(np.float32), B=p.B.numpy().astype(np.float32),
                  a=p.a.numpy(), b=p.b.numpy())
    for name,value in arrays.items():
        np.save(path/(name+".npy"),value)
    (path/"manifest.json").write_text(json.dumps(dict(n=p.n,m=p.m,
        array_files_sha256={name+".npy":runner.sha256(path/(name+".npy")) for name in arrays})))


def arguments(root, suffix, *, initialization="product", iterations=7, chunk=100, extra=()):
    return runner.parser().parse_args(["--instance",str(root/"input"),"--out",str(root/suffix),
        "--device","cpu","--initialization",initialization,"--seed","812","--epsilon",".0003",
        "--iterations",str(iterations),"--chunk-updates",str(chunk),"--checkpoint-every","3",
        "--retain-checkpoint",*extra])


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        os.environ["OMP_NUM_THREADS"] = "1"

    def test_initializations_feasible_positive_seeded_and_distinct(self):
        p = fixture(n=13)
        product,_ = runner.initialize(p,"product",seed=11)
        first,info = runner.initialize(p,"permutation-mixture",seed=11)
        same,same_info = runner.initialize(p,"permutation-mixture",seed=11)
        other,other_info = runner.initialize(p,"permutation-mixture",seed=12)
        torch.testing.assert_close(product,p.a[:,None]*p.b[None,:],rtol=0,atol=0)
        torch.testing.assert_close(first,same,rtol=0,atol=0)
        self.assertEqual(info,same_info)
        self.assertNotEqual(info["permutation_sha256"],other_info["permutation_sha256"])
        self.assertFalse(torch.equal(first,product))
        self.assertFalse(torch.equal(first,other))
        self.assertTrue(bool((first>0).all()))
        self.assertLessEqual(p.marginal_residuals(first)["both_l1"],1e-14)
        self.assertAlmostEqual((first-product).abs().sum().item(), 1.8*(1-1/p.n))

    def test_nonuniform_or_rectangular_permutation_rejected(self):
        p = fixture()
        a = torch.tensor([.1,.15,.2,.25,.3],dtype=torch.float64)
        nonuniform = runner.RowShardedGW(p.A,p.B,a,a.clone(),exact_block_rows=2)
        with self.assertRaisesRegex(ValueError,"uniform"):
            runner.initialize(nonuniform,"permutation-mixture",seed=1)
        rectangular = runner.RowShardedGW(p.A,p.B[:3,:3],p.a,torch.full((3,),1/3,dtype=torch.float64),exact_block_rows=2)
        with self.assertRaisesRegex(ValueError,"uniform"):
            runner.initialize(rectangular,"permutation-mixture",seed=1)

    def test_updates_match_pot_for_both_initializations_at_lower_entropy(self):
        p = fixture()
        for initialization in ("product","permutation-mixture"):
            for epsilon in (.001,.0003,.0001):
                with self.subTest(initialization=initialization,epsilon=epsilon):
                    initial,_ = runner.initialize(p,initialization,seed=812)
                    actual = initial.clone()
                    prox = runner.DistributedCertifiedPOTProx(p,certificate_block_rows=2)
                    for _ in range(4):
                        actual,info = runner.pot_step(p,runner.product_log_reference(p),p.gradient(actual),
                                                     prox,epsilon=epsilon)
                        self.assertLessEqual(info["marginal_l1"],1e-7)
                        self.assertLessEqual(info["normalized_gap_upper_bound"],1e-7)
                    expected = ot.gromov.entropic_gromov_wasserstein(p.A.numpy(),p.B.numpy(),p.a.numpy(),p.b.numpy(),
                        loss_fun="square_loss",epsilon=epsilon,G0=initial.numpy(),solver="PGD",max_iter=4,tol=0.,
                        numItermax=100000,stopThr=1e-14)
                    np.testing.assert_allclose(actual.numpy(),expected,atol=4e-8,rtol=0)

    def test_update_norm_and_consecutive_stopping_semantics(self):
        p = fixture()
        old,_ = runner.initialize(p,"product",seed=1)
        new,_ = runner.initialize(p,"permutation-mixture",seed=1)
        self.assertAlmostEqual(runner.outer_update_l1(p,old,new),(old-new).abs().sum().item())
        count=0
        for iteration in range(1,50):
            count,stop = runner.stopping_update(iteration,1e-7,count,tolerance=1e-6,minimum=50,consecutive=10)
            self.assertFalse(stop)
        count,stop = runner.stopping_update(50,1e-7,count,tolerance=1e-6,minimum=50,consecutive=10)
        self.assertTrue(stop)
        count,stop = runner.stopping_update(51,1e-5,count,tolerance=1e-6,minimum=50,consecutive=10)
        self.assertEqual(count,0)
        self.assertFalse(stop)
        for iteration in range(52,61):
            count,stop = runner.stopping_update(iteration,1e-6,count,tolerance=1e-6,minimum=50,consecutive=10)
            self.assertFalse(stop)
        count,stop = runner.stopping_update(61,1e-6,count,tolerance=1e-6,minimum=50,consecutive=10)
        self.assertTrue(stop)

    def test_resume_equivalence_for_both_initializations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_instance(root/"input")
            for initialization in ("product","permutation-mixture"):
                full = runner.run(arguments(root,initialization+"-full",initialization=initialization))
                partial = runner.run(arguments(root,initialization+"-resume",initialization=initialization,chunk=3))
                self.assertEqual(partial["status"],"checkpointed")
                resumed = runner.run(arguments(root,initialization+"-resume",initialization=initialization))
                self.assertEqual(resumed["status"],"completed")
                self.assertEqual(resumed["stopping_reason"],"iteration_budget")
                for key in ("trace","inner_trace","outer_trace","counters","initialization","stable_updates"):
                    self.assertEqual(full[key],resumed[key],key)
                state1,_=runner.CheckpointStore(root/(initialization+"-full")/"checkpoints").load(identity=full["identity"])
                state2,_=runner.CheckpointStore(root/(initialization+"-resume")/"checkpoints").load(identity=resumed["identity"])
                self.assertTrue(torch.equal(state1["plan"],state2["plan"]))
                self.assertTrue(torch.equal(state1["torch_rng_state"],state2["torch_rng_state"]))

    def test_early_stop_survives_checkpoint_and_final_diagnostic_is_exact_endpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_instance(root/"input",zero=True)
            kwargs = dict(initialization="permutation-mixture",iterations=100,
                          extra=("--outer-min-iterations","12","--outer-stable-updates","10"))
            full = runner.run(arguments(root,"full",**kwargs))
            partial = runner.run(arguments(root,"resumed",chunk=7,**kwargs))
            self.assertGreater(partial["stable_updates"],0)
            resumed = runner.run(arguments(root,"resumed",**kwargs))
            self.assertEqual(resumed["status"],"completed")
            self.assertEqual(resumed["completed_updates"],12)
            self.assertEqual(resumed["stopping_reason"],"outer_update_l1_tolerance")
            self.assertEqual(resumed["trace"][-1]["iteration"],12)
            self.assertEqual(full["trace"],resumed["trace"])
            self.assertEqual(full["outer_trace"],resumed["outer_trace"])
            self.assertGreaterEqual(resumed["stable_updates"],10)
            self.assertFalse(resumed["stopping_criterion"]["global_optimality_certificate"])

    def test_deadline_during_inner_solve_checkpoints_unchanged_plan_and_resumes_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_instance(root/"input")
            full = runner.run(arguments(root,"full",initialization="permutation-mixture"))
            real_clock = time.monotonic
            expiry = [False]
            calls = [0]
            original_iteration = runner.DeadlineCertifiedPOTProx._scaling_iteration
            def interrupt(prox,*args,**kwargs):
                calls[0] += 1
                if calls[0] == 3:
                    expiry[0] = True
                return original_iteration(prox,*args,**kwargs)
            def clock():
                return real_clock() + (1000 if expiry[0] else 0)
            with patch.object(runner.time,"monotonic",side_effect=clock), patch.object(
                    runner.DeadlineCertifiedPOTProx,"_scaling_iteration",new=interrupt):
                partial = runner.run(arguments(root,"resumed",initialization="permutation-mixture",
                                               extra=("--max-seconds","100")))
            self.assertEqual(partial["status"],"checkpointed")
            self.assertEqual(partial["completed_updates"],0)
            self.assertEqual(partial["counters"]["full_gradient_calls"],0)
            self.assertEqual(partial["counters"]["interrupted_inner_attempts"],1)
            self.assertFalse((root/"resumed"/"failure.json").exists())
            state,_ = runner.CheckpointStore(root/"resumed"/"checkpoints").load(identity=partial["identity"])
            problem,_ = runner.load_problem(root/"input",device="cpu")
            initial,_ = runner.initialize(problem,"permutation-mixture",seed=812)
            self.assertTrue(torch.equal(state["plan"],initial))
            resumed = runner.run(arguments(root,"resumed",initialization="permutation-mixture"))
            for key in ("trace","inner_trace","outer_trace","stable_updates","stopping_reason"):
                self.assertEqual(full[key],resumed[key],key)
            state1,_=runner.CheckpointStore(root/"full"/"checkpoints").load(identity=full["identity"])
            state2,_=runner.CheckpointStore(root/"resumed"/"checkpoints").load(identity=resumed["identity"])
            self.assertTrue(torch.equal(state1["plan"],state2["plan"]))

    def test_numerical_failure_carries_identity_for_campaign_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_instance(root/"input")
            args = arguments(root,"out")
            error = runner.shared.ProximalSolveError("inner budget exhausted",certificate={"converged":False})
            with patch.object(runner,"pot_step",side_effect=error):
                with self.assertRaises(runner.shared.ProximalSolveError):
                    runner.run(args)
            failure=json.loads((root/"out"/"failure.json").read_text())
            self.assertEqual(failure["failure_kind"],"inner_certificate_not_met")
            self.assertEqual(failure["config"],runner.immutable_config(args))
            self.assertEqual(failure["source_file_sha256"],runner.source_hashes())
            self.assertEqual(failure["identity"]["config"],runner.immutable_config(args))
            self.assertTrue(failure["checkpoint_removed"])
            self.assertFalse((root/"out"/"checkpoints"/"COMMITTED.json").exists())
            self.assertEqual(failure["checkpoint"]["iteration"],0)
            self.assertEqual(failure["trace"][0]["iteration"],0)
            self.assertIn("inner_trace",failure)
            self.assertIn("outer_trace",failure)
            self.assertIn("initialization",failure)

    def test_runtime_failure_keeps_resumable_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_instance(root/"input")
            with patch.object(runner,"pot_step",side_effect=RuntimeError("injected infrastructure failure")):
                with self.assertRaisesRegex(RuntimeError,"infrastructure"):
                    runner.run(arguments(root,"out"))
            failure=json.loads((root/"out"/"failure.json").read_text())
            self.assertEqual(failure["failure_kind"],"runtime_error")
            self.assertFalse(failure["checkpoint_removed"])
            self.assertTrue((root/"out"/"checkpoints"/"COMMITTED.json").exists())

    def test_changed_initialization_refuses_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_instance(root/"input")
            runner.run(arguments(root,"out",chunk=2))
            with self.assertRaisesRegex(ValueError,"configuration"):
                runner.run(arguments(root,"out",initialization="permutation-mixture"))


if __name__=="__main__":
    unittest.main()
