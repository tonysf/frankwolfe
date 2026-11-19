"""The constant-step ProxSARAH-v3 experiment in Pham et al. (JMLR 2020).

Source: https://www.jmlr.org/papers/volume21/19-248/19-248.pdf
Section 5, p. 21 names v3 and specifies gamma=.99, m=floor(sqrt(n)),
and b=floor(sqrt(n)/C); Section 3.4, p. 18 specifies C and eta.
These are the numerical Section 5 choices, not the finite-n rounding
formula (36), and not the different parameters in the authors' drivers.

We apply that recipe to f / loss_scale, with assumed normalized L=1.
Since the nonsmooth term is an indicator, its proximal map is unchanged;
using eta / loss_scale with the original gradient is exactly equivalent.
loss_scale is fixed for a run. A development-tuned value is a practical
smoothness estimate, not a certified global Lipschitz constant.

The existing joint SARAH/proximal/averaging implementation supplies
Algorithm 1 and estimator (29). An epoch contains a full snapshot update
followed by m recursive updates. There are no minibatch snapshots.
"""

import math
import operator

from methods import ProxSARAH


def parameters(n_train, loss_scale):
    """Return all v3 choices; only the objective's fixed scale is adjustable."""
    if isinstance(n_train, bool):
        raise ValueError("n_train must be a positive integer")
    try:
        n_train = operator.index(n_train)
    except TypeError as exc:
        raise ValueError("n_train must be a positive integer") from exc
    if n_train < 1:
        raise ValueError("n_train must be a positive integer")
    loss_scale = float(loss_scale)
    if not math.isfinite(loss_scale) or loss_scale <= 0:
        raise ValueError("loss_scale must be finite and positive")

    normalized_lipschitz = 1.0
    gamma = 0.99
    C = 2.0 / (3.0 * normalized_lipschitz**2 * gamma**2)
    eta_normalized = 2.0 / (4.0 + gamma * normalized_lipschitz)
    eta = eta_normalized / loss_scale
    if not math.isfinite(eta) or eta <= 0:
        raise ValueError("loss_scale gives a nonfinite or zero effective step")
    inner_steps = math.isqrt(n_train)
    batch = math.floor(math.sqrt(n_train) / C)
    if not 1 <= batch <= n_train:
        raise ValueError("the paper's v3 batch formula is outside [1, n_train]")
    return {
        "paper_variant": "ProxSARAH-v3",
        "normalized_lipschitz": normalized_lipschitz,
        "loss_scale": loss_scale,
        "gamma": gamma,
        "relaxation": gamma,
        "C": C,
        "eta_normalized": eta_normalized,
        "eta": eta,
        "batch": batch,
        "inner_steps": inner_steps,
        "snapshot_batch": n_train,
        "updates_per_epoch": inner_steps + 1,
    }


class ProxSARAHV3(ProxSARAH):
    """Named paper variant with full snapshots and coupled fixed parameters."""

    def __init__(self, problem, U, V, caps, *, seed, config):
        if set(config) != {"loss_scale"}:
            raise ValueError("ProxSARAH-v3 accepts only the fixed loss_scale")
        derived = parameters(len(problem.train_ids), config["loss_scale"])
        super().__init__(problem, U, V, caps, seed=seed, config=derived)

    def _update(self):
        details = super()._update()
        return {
            **details,
            "epoch_position": self.k % (self.inner_steps + 1),
            "eta": self.eta,
            "gamma": self.relaxation,
            "curvature_operator_calls": 0,
            "curvature_forward_views": 0,
            "curvature_backward_views": 0,
            "oracle_calls": self.gradient_calls,
        }


def make_solver(problem, U, V, caps, *, seed, config):
    return ProxSARAHV3(problem, U, V, caps, seed=seed, config=config)
