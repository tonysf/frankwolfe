# Final A100 preflight

Job **1961539** completed with exit **0** in **3m24s**, on source `af7b7c549351b81de2aa886b987bc3e56ffeff9c`. Audit status: **passed**.

- **125/125 pytest tests passed**, with no failures, errors or skips.
- **18 GPU projection cases passed** across n=16,128,1000, including repeated bitwise equality.
- **20 actual runner smokes passed**: 14 toy method/start combinations, native Sampled GW, four momentum-KL campaign-size/start combinations, and one allocated dispatcher trial.
- **182 artifact files** (21,911,630 bytes), plus the job log, matched remote SHA256 checksums. Remote manifests were identical before and after transfer.

| Momentum KL instance | Start | Completed updates | Actual rounded marginal L1 |
|---|---|---:|---:|
| 1000 | product | 24 | 1.18e-07 |
| 1000 | randomized | 11 | 1.21e-07 |
| 512 | product | 32 | 6.42e-08 |
| 512 | randomized | 8 | 6.11e-08 |

All runner metadata records clean source, A100 float32, deterministic algorithms, highest matmul precision, AMP/TF32 disabled, and one intra-op CPU thread. Their recorded inter-op setting is 32; the separate environment-capture and projection-validator process uses 1. Slurm/CUDA visibility and both frozen dispatcher hashes were preserved.

Independent CPU float64 checks of all 20 saved raw/rounded outputs passed. Maximum actual rounded marginal-L1 residual: **1.21e-07**; maximum CPU-versus-GPU float64 rounded-loss difference: **2.22e-16**. Outputs were promoted intact, without another repair.

Native Sampled GW completed with warnings (1 recorded); its inner iteration/failure counts remain unknown. These checks do not establish optimizer convergence or a speed advantage.

Artifacts: [checks and detailed evidence](final_a100_preflight.json), [job log](../ruche_results/preflight_1961539/campaign-check-1961539.out), [remote checksum manifest](../ruche_results/preflight_1961539/REMOTE_ARTIFACT_SHA256SUMS), [scheduler accounting](../ruche_results/preflight_1961539/job_accounting.txt).
