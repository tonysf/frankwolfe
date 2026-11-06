# n=8 FRAMES versus projected momentum-SGD

This is a controlled QPT baseline campaign using the immutable `source_fast`
bundle selected on 2026-09-23 and the existing verified `n8.npz` dataset.

- Common configuration: rank 1, tau 20, noise 0.05, batch 65536,
  initialization seed 0, product-state measurements, float64, 200,000 updates,
  rho `0.6/(k+4)^0.5`, and smoothing `1e7/(k+1)^0.25`.
- FRAMES uses `0.5/(k+10)^0.75` for its convex-combination step.
- Momentum-SGD uses the same gradient EMA and TP term, replacing only the LMO
  update by a projected gradient update. Its learning rate is
  `eta_scale/(k+10)^0.75`.
- Predeclared tuning grid on sampling seed 0: eta scales 0.1, 0.3, 1.0, 3.0;
  20,000 updates each. Select maximum terminal fidelity, with deterministic
  ties broken by lower TP violation, lower loss, then lower eta.
- Freeze the selected eta and compare both methods on held-out sampling seeds
  1, 2, and 3, with exactly paired minibatches and identical initialization.
- Run at most one campaign GPU at a time. Validate frozen-source/data hashes in
  every allocation and independently verify every saved factor.

Seed 0 is used only for baseline selection; it is not included in the final
three-seed comparison.
