# n=8 fixed-penalty MiFGD calibration

This adds the published MiFGD two-sequence/Nesterov recursion as a distinct
baseline. It is adapted to the existing stochastic product-state QPT oracle.

- Update: `Z_k = U_k + mu*(U_k-U_{k-1})`, followed by one factor-gradient
  step evaluated at `Z_k`.
- Fixed objective: sampled measurement loss plus
  `tp_weight * 0.5*||T(UU^H)-I||_F^2`.
- No TP smoothing and no norm-ball projection.
- Published practical momentum `mu=0.75`; `mu=0` will be retained as the FGD
  control in the held-out comparison.
- Calibration: seed 0, 20,000 steps, fixed `tp_weight=1`, constant eta grid
  `{1e-6, 1e-5, 1e-4, 1e-3}` (including the paper's practical eta).
- A candidate is admissible if finite with terminal TP violation at most 1.0.
  Among admissible candidates, select maximum terminal fidelity.
- Only after an admissible calibration will eta be frozen for 200,000-step
  held-out runs on seeds 1--3.

The algorithmic recursion follows Kim et al., arXiv:2104.07006. The fixed TP
penalty follows Quiroga and Kyrillidis, arXiv:2312.01311.
