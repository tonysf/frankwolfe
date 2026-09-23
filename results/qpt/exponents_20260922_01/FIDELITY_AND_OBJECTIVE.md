# Fidelity versus the optimized objective

The factor is U and the represented process matrix is χ = UU†. In the rank-one runs U has one column, denoted u. Let c be the complete synthetic truth factor, so χ⋆ = cc†.

The smooth objective term is the measurement least-squares loss:

\[
f(U)=\frac{1}{2M}\sum_{s=1}^{M}
\left(\mathcal A_s(UU^\dagger)-y_s\right)^2.
\]

Here y_s includes the fixed observation noise. Stochastic batches estimate its gradient. The plotted measurement loss is an estimate using a fixed, independent set of 512 measurement rows, not a full pass through the table.

The composite nonsmooth term enforces trace preservation. With Z = T(UU†), g(Z) is the indicator of the singleton {I}: zero when Z = I and infinity otherwise. The implemented smoothed objective at update k is

\[
f(U)+g_{\beta_k}(T(UU^\dagger))
=f(U)+\frac{\|T(UU^\dagger)-I\|_F^2}{2\beta_k},
\qquad \|U\|_{\mathrm{op}}\le\tau.
\]

The reported rank-one fidelity proxy is a separate recovery diagnostic:

\[
F(u,c)=\frac{|c^\dagger u|^2}{\|c\|^2\|u\|^2}.
\]

It is the squared overlap between normalized complete factors, equivalently the pure-state fidelity of their trace-normalized rank-one process matrices. F = 1 for factors proportional to the truth; F = 0 for orthogonal factors. Overall phase and nonzero scale do not change it. Therefore high fidelity alone does not certify normalization or trace preservation. The saved TP residual must be assessed separately, particularly because the iterates are only approximately TP.

Fidelity is not f, not the smoothed objective, and not computed from sampled measurement rows. It uses the known synthetic truth, which a real experimental dataset would generally not provide. The implementation uses ||c||² = d for these Haar-unitary truths; the independent NumPy verifier explicitly normalizes by both complete factor norms. The smoothed gap is also a separate diagnostic.

# New schedules

Only the exponents change: step 1 → 3/4, smoothing stays 1/4, momentum 0.6 → 1/2. Scales and offsets stay at their recorded case-specific values. For n = 8 the requested schedules are

\[
\gamma_k=\min\{1,10/(k+10)^{3/4}\},\quad
\beta_k=10^7/(k+1)^{1/4},\quad
\rho_k=\min\{1,2/(k+4)^{1/2}\}.
\]

Momentum obeys d_k = (1-ρ_k)d_{k-1} + ρ_k ĝ_k, with the first gradient assigned directly. Thus ρ is the new-gradient weight, not the retained weight. Lower step/momentum exponents produce larger late γ/ρ at fixed scales and offsets; the step cap is active for the first 12 n = 8/10 updates. There is no smoothing cap.

All 12 cases start afresh from the original initialization; old-schedule restarts are not used. Fixed reported iteration budgets allow direct trajectory comparisons even if a new run crosses 99% earlier. This is a matched exponent comparison, not a retuning of scales or a proof that these settings are optimal. Exact cases, datasets, hashes and job IDs are in plan.json and manifest.json. The final PDF will preserve the prior results and compare both schedules.
