# Original-noise n10 pilot comparison

All experiments use the same sigma=.05 dataset, truth, U0, initialization/sampling seeds0, rank1, tau40, complex128 and validated product-state source. Fixed512-row metrics; full-factor fidelity and exact TP. No optimizer/default/source changes were made for these pilots.

| Run | Steps | Batch | Sampled rows | Fidelity % | TP violation |
|---|---:|---:|---:|---:|---:|
| baseline | 5000 | 65536 | 327,680,000 | 5.0165803e-05 | 25.2798 |
| baseline | 10000 | 65536 | 655,360,000 | 0.00019353969 | 24.0095 |
| baseline | 20000 | 65536 | 1,310,720,000 | 0.00025891748 | 20.2976 |
| baseline | 100000 | 65536 | 6,553,600,000 | 0.0040148002 | 10.407 |
| weaker_tp | 10000 | 65536 | 655,360,000 | 4.9590855e-05 | 27.9749 |
| averaging | 10000 | 65536 | 655,360,000 | 0.00022549292 | 26.74 |
| large_batch | 5000 | 262144 | 1,310,720,000 | 0.00038156164 | 27.2364 |

Baseline uses beta0=1e9; all new pilots beta0=1e10. Weaker-TP-only and larger-batch pilots retain rho=2/(k+4)^.6 and gamma=10/(k+10). The averaging pilot uses rho=.2/(k+4)^.6 and gamma=1/(k+100), changing both to address variance and momentum lag. These are finite one-seed pilots; low early fidelity is not a proof that longer runs cannot recover.

At equal1,310,720,000 draws, large-batch/5000steps fidelity is .00038156%, compared with baseline/20000steps .00025892%. Neither is useful recovery. At10000steps, weaker TP has worse fidelity and TP than baseline; stronger averaging plus slower steps has similar near-zero fidelity.

Optimizer times: weaker-TP950.046s, averaging949.892s, larger-batch1792.193s. Runner wall times1048.906s,1048.771s,1989.929s respectively. Array0 packing the two small-batch runs took35m23s; array1 including its separate100step smoke took34m21s. Both COMPLETED0; stderr empty.

Next bounded comparison: array1888731, beta0=1e7 versus1e8, sameB65536/10k and originalrho/gamma, each in an A100 allocation with30min cap. Earlier beta~1e7 n10 tests usedB512 and a different step schedule, so they are not matched controls. This closes the penalty comparison around1e9 rather than extrapolating the initial TP-gradient norm into a causal claim. New directory: fidelity99_stronger_tp_20260913_01. Preserve every earlier archive and restart.
