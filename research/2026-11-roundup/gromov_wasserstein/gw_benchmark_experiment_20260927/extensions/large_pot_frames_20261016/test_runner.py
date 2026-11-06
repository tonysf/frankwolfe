import importlib.util
import fcntl
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import ot
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("large_pot_frames_runner", HERE/"runner.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
from checkpoint import CheckpointStore, sha256
from prox import DistributedCertifiedPOTProx
from oracle import PairSample


def fixture():
    A = torch.tensor([[0.,.035,.1],[.035,0.,.07],[.1,.07,0.]], dtype=torch.float64)
    B = torch.tensor([[0.,.06,.085],[.06,0.,.02],[.085,.02,0.]], dtype=torch.float64)
    P = torch.tensor([[.12,.03,.05],[.03,.17,.1],[.15,.2,.15]], dtype=torch.float64)
    return runner.RowShardedGW(A,B,P.sum(1),P.sum(0),exact_block_rows=2), P


def write_instance(path):
    p, _ = fixture()
    path.mkdir()
    arrays = dict(A=p.A.numpy().astype(np.float32),B=p.B.numpy().astype(np.float32),
                  a=p.a.numpy(),b=p.b.numpy())
    for name, value in arrays.items():
        np.save(path/(name+".npy"),value)
    manifest = dict(n=p.n,m=p.m,array_files_sha256={name+".npy":sha256(path/(name+".npy")) for name in arrays})
    (path/"manifest.json").write_text(json.dumps(manifest))


def distributed_resume_worker(rank, world, directory, rendezvous):
    """Standalone worker for the opt-in scheduled/host loopback gate."""
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    dist.init_process_group("gloo",rank=rank,world_size=world,init_method="file://"+rendezvous)
    root=Path(directory)
    try:
        for method in ("pot-full","pot-pair","frames"):
            def arguments(suffix,chunk):
                return runner.parser().parse_args(["--instance",str(root/"input"),
                    "--out",str(root/(method+suffix)),"--method",method,"--device","cpu",
                    "--iterations","4","--chunk-updates",str(chunk),"--pairs","16",
                    "--checkpoint-every","2","--diagnostic-every","2","--retain-checkpoint"])
            full=runner.run(arguments("-full",4))
            partial=runner.run(arguments("-resume",2))
            resumed=runner.run(arguments("-resume",4))
            assert partial["status"]=="checkpointed" and resumed["status"]=="completed"
            assert full["trace"]==resumed["trace"]
            assert full["counters"]==resumed["counters"]
            s1,_=CheckpointStore(root/(method+"-full")/"checkpoints",rank=rank,world_size=world).load(identity=full["identity"])
            s2,_=CheckpointStore(root/(method+"-resume")/"checkpoints",rank=rank,world_size=world).load(identity=resumed["identity"])
            for name in ("plan","generator_state"):
                assert torch.equal(s1[name],s2[name]), (method,name,rank)
            if method=="frames":
                assert torch.equal(s1["momentum"],s2["momentum"])
        dist.barrier()
    finally:
        dist.destroy_process_group()


class DistributedResumeTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("LARGE_GW_DISTRIBUTED_TEST")=="1", "opt-in loopback distributed gate")
    def test_four_rank_resume_all_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            instance=root/"input"
            instance.mkdir()
            rng=np.random.default_rng(771)
            x,y=rng.normal(size=(7,3)),rng.normal(size=(5,3))
            arrays=dict(A=(.03*np.linalg.norm(x[:,None]-x[None,:],axis=-1)).astype(np.float32),
                        B=(.03*np.linalg.norm(y[:,None]-y[None,:],axis=-1)).astype(np.float32),
                        a=np.full(7,1/7,dtype=np.float64),b=np.full(5,1/5,dtype=np.float64))
            for name,value in arrays.items():
                np.save(instance/(name+".npy"),value)
            (instance/"manifest.json").write_text(json.dumps(dict(n=7,m=5,
                array_files_sha256={name+".npy":sha256(instance/(name+".npy")) for name in arrays})))
            mp.spawn(distributed_resume_worker,args=(4,directory,str(root/"rendezvous")),nprocs=4,join=True)


class RunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        os.environ["OMP_NUM_THREADS"] = "1"

    def test_pot_full_matches_published_pgd_update(self):
        p, initial = fixture()
        prox = DistributedCertifiedPOTProx(p, certificate_block_rows=2)
        actual = initial.clone()
        for _ in range(4):
            actual, info = runner.pot_step(p,runner.product_log_reference(p),p.gradient(actual),prox)
            self.assertLessEqual(info["marginal_l1"],1e-7)
            self.assertLessEqual(info["normalized_gap_upper_bound"],1e-7)
        expected = ot.gromov.entropic_gromov_wasserstein(p.A.numpy(),p.B.numpy(),p.a.numpy(),p.b.numpy(),
            loss_fun="square_loss",epsilon=.001,G0=initial.numpy(),solver="PGD",max_iter=4,tol=0.,
            numItermax=10000,stopThr=1e-14)
        np.testing.assert_allclose(actual.numpy(),expected,atol=3e-8,rtol=0)

    def test_pair_oracle_expectation_matches_full_off_feasibility(self):
        p, P = fixture()
        P = P*1.7
        P[0,1] = 0
        expected = torch.zeros_like(P)
        for i in range(p.n):
            for j in range(p.m):
                sample = PairSample(torch.tensor([i]),torch.tensor([j]),P.sum())
                expected.add_(runner.pair_gradient(p,sample),alpha=(P[i,j]/P.sum()).item())
        torch.testing.assert_close(expected,p.gradient(P),atol=1e-15,rtol=1e-14)

    def test_bad_certificate_and_infeasible_plan_rejected(self):
        p,P = fixture()
        good = DistributedCertifiedPOTProx(p, certificate_block_rows=2)
        result = good(runner.product_log_reference(p),p.gradient(P),1000.)
        for change, scale in (({"normalized_primal_dual_gap":1.},1.),({},.9)):
            class Bad:
                marginal_residuals = good.marginal_residuals
                def __call__(self,*args,**kwargs):
                    return result[0]*scale,result[1],dict(result[2],**change)
            with self.assertRaises(runner.ProximalSolveError):
                runner.pot_step(p,runner.product_log_reference(p),p.gradient(P),Bad())

    def test_fp32_distances_promoted_without_data_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"input"
            write_instance(path)
            p,record = runner.load_problem(path,device="cpu")
            np.testing.assert_array_equal(p.A.numpy(),np.load(path/"A.npy").astype(np.float64))
            np.testing.assert_array_equal(p.a.numpy(),np.load(path/"a.npy"))
            self.assertEqual(record["original_array_dtypes"]["A"],"float32")

    def test_schedule_matches_fixed_theoretical_exponents(self):
        gamma,rho,mu = runner.frames_schedule(15,100,.8,.4,10.)
        self.assertAlmostEqual(gamma,.8/8)
        self.assertAlmostEqual(rho,.4/4)
        self.assertAlmostEqual(mu,10./100/2)

    def test_loss_diagnostics_against_independent_tensor_sum(self):
        p,P = fixture()
        report = runner.diagnose(p,P,.001,2,momentum=p.gradient(P),prefactors=(1.,1.,10.))
        expected = sum((p.A[i,k]-p.B[j,l]).item()**2*P[i,j].item()*P[k,l].item()
                       for i in range(p.n) for j in range(p.m) for k in range(p.n) for l in range(p.m))
        self.assertAlmostEqual(report["raw_loss"],expected,places=15)
        self.assertLessEqual(report["rounded_residuals"]["both_l1"],1e-13)
        self.assertIn("lmo_changed_source_mass",report)

    def test_all_methods_resume_exactly_and_keep_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            instance=root/"input"
            write_instance(instance)
            for method in ("pot-full","pot-pair","frames"):
                with self.subTest(method=method):
                    def arguments(out,chunk):
                        return runner.parser().parse_args(["--instance",str(instance),"--out",str(out),
                            "--method",method,"--device","cpu","--iterations","4","--chunk-updates",str(chunk),
                            "--pairs","16","--checkpoint-every","2","--diagnostic-every","2","--retain-checkpoint"])
                    full=runner.run(arguments(root/(method+"-full"),4))
                    partial=runner.run(arguments(root/(method+"-resume"),2))
                    self.assertEqual(partial["status"],"checkpointed")
                    resumed=runner.run(arguments(root/(method+"-resume"),4))
                    self.assertEqual(resumed["status"],"completed")
                    self.assertEqual(full["trace"],resumed["trace"])
                    self.assertEqual(full["counters"],resumed["counters"])
                    s1,_=CheckpointStore(root/(method+"-full")/"checkpoints").load(identity=full["identity"])
                    s2,_=CheckpointStore(root/(method+"-resume")/"checkpoints").load(identity=resumed["identity"])
                    for name in ("plan","generator_state"):
                        self.assertTrue(torch.equal(s1[name],s2[name]))
                    if method=="frames":
                        self.assertTrue(torch.equal(s1["momentum"],s2["momentum"]))

    def test_completed_state_cleanup_and_duplicate_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_instance(root/"input")
            args=runner.parser().parse_args(["--instance",str(root/"input"),"--out",str(root/"out"),
                "--method","frames","--device","cpu","--iterations","1"])
            record=runner.run(args)
            self.assertEqual(record["status"],"completed")
            self.assertFalse((root/"out"/"checkpoints"/"COMMITTED.json").exists())
            self.assertTrue((root/"out"/"metadata.json").exists())
            with self.assertRaisesRegex(ValueError,"duplicate"):
                runner.run(args)

    def test_process_lock_refuses_duplicate_before_loading_data(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)/"out"
            out.mkdir()
            args=runner.parser().parse_args(["--instance","not-read","--out",str(out),
                "--method","frames","--device","cpu","--iterations","1"])
            with (out/"run.lock").open("a+") as handle:
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError,"temporarily unavailable"):
                    runner.run(args)

    def test_time_based_checkpoint_threshold_is_obeyed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_instance(root/"input")
            args=runner.parser().parse_args(["--instance",str(root/"input"),"--out",str(root/"out"),
                "--method","frames","--device","cpu","--iterations","3", "--pairs","8",
                "--checkpoint-every","5000","--checkpoint-seconds","1e-12"])
            original=CheckpointStore.save
            iterations=[]
            def record_save(store,state,**kwargs):
                iterations.append(kwargs["iteration"])
                return original(store,state,**kwargs)
            with patch.object(CheckpointStore,"save",new=record_save):
                runner.run(args)
            self.assertEqual(iterations,[0,1,2,3])


if __name__ == "__main__":
    unittest.main()
