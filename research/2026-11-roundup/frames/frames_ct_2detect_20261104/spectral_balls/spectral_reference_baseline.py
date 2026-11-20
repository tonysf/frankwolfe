"""SPRING-SARAH, with audited constants and numerically certified block proxes.

Reference: Driggs et al., SIAM J. Imaging Sciences 14(4), 2021, Algorithm 1,
Theorem 3.1, Proposition 2.2, Corollary 3.3 and Appendix C.

The mathematical theorem assumes exact proximal maps. This implementation uses
Dykstra projections with a primal/dual gap certificate, and explicitly records
the floating-point floor. It does NOT claim a fixed numerical floor is covered
by the exact-prox theorem, nor that a summable target by itself proves an inexact
SPRING convergence theorem. The practical mode is a separate heuristic control.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math
from typing import Callable

import numpy as np


class ProjectionFailure(RuntimeError):
    def __init__(self, certificate):
        self.certificate = certificate
        super().__init__("Dykstra projection did not attain its numerical target: "
                         + repr(certificate))


@dataclass(frozen=True)
class ProjectionCertificate:
    iterations: int
    status: str
    requested_distance: float
    attained_distance_bound: float
    duality_gap: float
    stable_duality_gap: float
    roundoff_allowance: float
    primal_objective: float
    dual_objective: float
    primal_feasibility: float
    stationarity_residual: float
    exact_prox_theorem: bool = False
    method: str = "dykstra"


def _dot(a, b):
    # Extended accumulation substantially reduces cancellation in the gap.
    return np.sum(a.astype(np.longdouble) * b.astype(np.longdouble),
                  dtype=np.longdouble)


def project_l1(z, radius):
    """Euclidean projection onto an ENTRYWISE l1 ball; signs are preserved."""
    if radius <= 0:
        if radius < 0:
            raise ValueError("l1 radius must be nonnegative")
        return np.zeros_like(z)
    a = np.abs(z).ravel()
    if float(np.sum(a, dtype=np.longdouble)) <= radius:
        return z.copy()
    ordered = np.sort(a)[::-1]
    cssv = np.cumsum(ordered, dtype=np.float64)
    active = np.flatnonzero(ordered > (cssv - radius) / np.arange(1, a.size + 1))
    rho = int(active[-1])
    threshold = (cssv[rho] - radius) / (rho + 1)
    out = np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)
    # An inward radial correction compensates for summation rounding.
    norm = float(np.sum(np.abs(out), dtype=np.longdouble))
    if norm > radius:
        out *= radius / norm * (1 - 8 * np.finfo(float).eps)
    return out


def _project_spectral_parts(z, radius):
    left, s, right = np.linalg.svd(z, full_matrices=False)
    if s.size == 0 or s[0] <= radius:
        return z.copy(), np.zeros_like(z), np.longdouble(0.0), float(s[0]) if s.size else 0.0
    clipped = np.minimum(s, radius)
    projected = (left * clipped) @ right
    correction = z - projected
    # This support expression follows from the same SVD used for the projection.
    support = np.longdouble(radius) * np.sum(np.maximum(s - radius, 0.0),
                                              dtype=np.longdouble)
    return projected, correction, support, float(s[0]) if s.size else 0.0


def project_intersection(z, radius, *, kind="nonnegative", l1_radius=None,
                         iteration=0, distance_tolerance=1e-8,
                         tolerance_power=1.1, max_iterations=2000,
                         check_every=5, raise_on_failure=True):
    """Project onto a spectral ball intersected with an orthant or l1 ball.

    For min_c .5||c-z||^2 + I_A(c)+I_D(c), dual variables p,q give
      d(p,q)=<z,p+q>-.5||p+q||^2-sigma_A(p)-sigma_D(q).
    A feasible c therefore satisfies ||c-P_K(z)|| <= sqrt(2*(primal-dual)).
    sigma_A(p)=radius*||p||_*, sigma_D(q)=0 for q<=0 (orthant), and
    sigma_D(q)=l1_radius*||q||_infty (l1 ball).

    The requested distance is summable in outer iterations, until a reported
    floating-point floor dominates. The allowance is a conservative numerical
    rounding margin, NOT an interval-arithmetic verification of LAPACK's SVD.
    No iterate-change-only stopping test is used.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or not np.all(np.isfinite(z)):
        raise ValueError("projection expects a finite matrix")
    if radius <= 0 or distance_tolerance <= 0 or tolerance_power <= 1:
        raise ValueError("positive radius/tolerance and power > 1 required")
    if kind not in ("nonnegative", "l1"):
        raise ValueError("kind must be nonnegative or l1")
    if kind == "l1" and (l1_radius is None or l1_radius <= 0):
        raise ValueError("a positive entrywise l1 radius is required")
    if max_iterations < 1 or check_every < 1:
        raise ValueError("positive iteration/check counts required")
    requested = distance_tolerance / (iteration + 1) ** tolerance_power
    eps = np.finfo(float).eps
    d_project = (lambda a: np.maximum(a, 0.0)) if kind == "nonnegative" else (
        lambda a: project_l1(a, l1_radius))

    def side_violation(a):
        return (max(0.0, float(-a.min())) if kind == "nonnegative" else
                max(0.0, float(np.sum(np.abs(a), dtype=np.longdouble)) - l1_radius))

    def certify(x, p, q, support_a, inner, method="dykstra", opnorm=None):
        if opnorm is None:
            opnorm = float(np.linalg.norm(x, 2))
        scale = min(1.0, radius / max(opnorm, np.finfo(float).tiny))
        if scale < 1.0 or opnorm >= radius * (1 - 8 * eps):
            scale *= 1 - 16 * eps
        candidate = x * scale
        # Scaling toward zero preserves nonnegativity and the entrywise l1 ball.
        support_d = (np.longdouble(0.0) if kind == "nonnegative" else
                     np.longdouble(l1_radius) * np.max(np.abs(q)))
        residual = z - p - q
        primal = np.longdouble(0.5) * _dot(candidate - z, candidate - z)
        dual = _dot(z, p + q) - np.longdouble(0.5) * _dot(p + q, p + q) - support_a - support_d
        gap_direct = primal - dual
        # Algebraically identical, with much less subtractive cancellation.
        gap_stable = (np.longdouble(0.5) * _dot(candidate - residual, candidate - residual)
                      + support_a - _dot(candidate, p)
                      + support_d - _dot(candidate, q))
        scale_gap = max(float(abs(primal)), float(abs(dual)),
                        float(abs(support_a)), float(abs(support_d)),
                        float(abs(_dot(candidate, p))), float(abs(_dot(candidate, q))),
                        np.finfo(float).tiny)
        # p and q are formed by subtracting projected matrices, e.g. q=z-x.
        # Their absolute roundoff is governed by z/x, not by a tiny correction.
        # The dual expression also contains <z,p+q>. Omitting the input scale
        # incorrectly asks Dykstra to iterate below representable precision when
        # a very small step crosses an active l1 boundary.
        input_scale = max(float(_dot(z, z)), float(_dot(candidate, candidate)),
                          float(_dot(x, x)))
        if np.any(p) or np.any(q) or np.any(candidate != z):
            scale_gap = max(scale_gap, input_scale)
        allowance = 64 * eps * max(1, min(z.shape)) * scale_gap
        raw_gap = max(0.0, float(gap_direct), float(gap_stable))
        distance_bound = math.sqrt(2 * (raw_gap + allowance))
        feasibility = max(side_violation(candidate),
                          max(0.0, opnorm * scale - radius))
        target_met = distance_bound <= requested
        # When the desired squared error is below floating-point resolution,
        # reach the calculated gap floor and report that limitation explicitly.
        numerical_met = raw_gap <= 0.5 * requested ** 2 + allowance
        status = "requested_accuracy" if target_met else ("numerical_floor" if numerical_met else "not_converged")
        certificate = ProjectionCertificate(
            inner, status, requested, distance_bound, raw_gap,
            float(gap_stable), allowance, float(primal), float(dual),
            feasibility, float(np.linalg.norm(z - x - p - q)), False, method)
        accepted = numerical_met and feasibility <= 64 * eps * max(1.0, radius, l1_radius or 0.0)
        return candidate, certificate, accepted

    # If P_D(z) belongs to A, it minimizes distance over A intersection D too:
    # the intersection is a subset of D and contains the unconstrained minimizer.
    # This is an exact one-set reduction, not alternating projections without
    # Dykstra corrections. It avoids an unnecessary numerical iteration when
    # only a tiny l1 threshold is active.
    side = d_project(z)
    side_opnorm = float(np.linalg.norm(side, 2))
    if side_opnorm <= radius and side_violation(side) == 0:
        q_side = z - side
        if kind == "nonnegative":
            q_side = np.minimum(q_side, 0.0)
        candidate, certificate, accepted = certify(
            side, np.zeros_like(z), q_side, np.longdouble(0), 1,
            "single_side_projection", side_opnorm)
        if accepted:
            return candidate, certificate

    # The symmetric reduction applies if the spectral projection already meets
    # the side constraint. Only genuinely coupled cases need Dykstra iterations.
    spectral, spectral_p, spectral_support, _ = _project_spectral_parts(z, radius)
    if side_violation(spectral) == 0:
        candidate, certificate, accepted = certify(
            spectral, spectral_p, np.zeros_like(z), spectral_support, 1,
            "single_spectral_projection")
        if accepted:
            return candidate, certificate

    x = z.copy()
    p = np.zeros_like(z)
    q = np.zeros_like(z)
    certificate = None
    candidate = None
    for inner in range(1, max_iterations + 1):
        y, p, support_a, _ = _project_spectral_parts(x + p, radius)
        arg_d = y + q
        x = d_project(arg_d)
        q = arg_d - x
        if kind == "nonnegative":
            q = np.minimum(q, 0.0)
        if inner != 1 and inner % check_every and inner != max_iterations:
            continue
        candidate, certificate, accepted = certify(x, p, q, support_a, inner)
        if accepted:
            return candidate, certificate
    if raise_on_failure:
        raise ProjectionFailure(asdict(certificate))
    return candidate, certificate


def theorem_parameters(n_components, component_lipschitz,
                       block_lipschitz_u, block_lipschitz_v):
    """Literal SPRING-SARAH Corollary 3.3, not its practical Sec.5.1 steps."""
    if n_components < 1:
        raise ValueError("n_components must be positive")
    L = max(component_lipschitz, block_lipschitz_u, block_lipschitz_v)
    if not np.isfinite(L) or L <= 0:
        raise ValueError("a finite positive Lipschitz bound is required")
    step = 1 / (2 * L * math.sqrt(30 * n_components))
    assert step < 1 / (4 * max(block_lipschitz_u, np.finfo(float).tiny))
    assert step < 1 / (4 * max(block_lipschitz_v, np.finfo(float).tiny))
    return dict(step_u=step, step_v=step, refresh_probability=1 / n_components,
                L=L, V1=2 * L ** 2, V_upsilon=2 * L ** 2,
                V2=2 * L, rho=1 / n_components,
                theorem="SPRING Corollary 3.3 / Theorem 3.1")


class SpringSARAH:
    """Alternating, loopless SARAH with independent fresh block sampling.

    full_gradient(U,V) and batch_gradient(U,V,indices) return BOTH gradients.
    sample_indices(rng,batch_size), if omitted, draws a uniform subset WITHOUT
    replacement as specified in the paper. n_components counts image columns,
    not pixels, for the primary factorization experiment.
    """
    def __init__(self, U, V, *, full_gradient: Callable, batch_gradient: Callable,
                 n_components, radius_u, radius_v, l1_radius,
                 component_lipschitz, block_lipschitz_u, block_lipschitz_v,
                 batch_size=1, mode="theorem", practical_step_factor=0.5,
                 refresh_period=None, sample_indices=None, seed=0,
                 projection_tolerance=1e-8, projection_power=1.1,
                 projection_max_iterations=2000):
        self.U = np.asarray(U, dtype=float).copy()
        self.V = np.asarray(V, dtype=float).copy()
        self.full_gradient = full_gradient
        self.batch_gradient = batch_gradient
        self.n_components = int(n_components)
        self.batch_size = int(batch_size)
        if not 1 <= self.batch_size <= self.n_components:
            raise ValueError("batch_size must be between one and n_components")
        self.sample_indices = sample_indices
        self.rng = np.random.default_rng(seed)
        self.radius_u, self.radius_v, self.l1_radius = radius_u, radius_v, l1_radius
        self.projection_tolerance = projection_tolerance
        self.projection_power = projection_power
        self.projection_max_iterations = projection_max_iterations
        self.mode = mode
        self.parameters = theorem_parameters(n_components, component_lipschitz,
                                             block_lipschitz_u, block_lipschitz_v)
        if mode == "theorem":
            if refresh_period is not None and refresh_period != n_components:
                raise ValueError("literal Corollary 3.3 requires refresh_period=n_components")
        elif mode == "practical":
            if not 0 < practical_step_factor <= 1:
                raise ValueError("practical step factor must be in (0,1]")
            self.parameters.update(step_u=practical_step_factor / block_lipschitz_u,
                                   step_v=practical_step_factor / block_lipschitz_v,
                                   theorem=None,
                                   label="SPRING-SARAH practical constant component-bound control; not theorem-prescribed")
            if refresh_period is not None:
                if refresh_period < 1:
                    raise ValueError("refresh_period must be at least one")
                self.parameters["refresh_probability"] = 1 / refresh_period
        else:
            raise ValueError("mode must be theorem or practical")
        self.iteration = 0
        self.oracle_calls = 0
        self.full_gradient_calls = 0
        self.sampled_gradient_calls = 0
        self._estimate_u = None
        self._estimate_v = None
        self._previous_u_point = None
        self._previous_v_point = None
        # A feasible start is part of this constrained comparison, not something
        # silently repaired differently for different methods.
        for arr, radius, kind, budget in [(self.U, radius_u, "nonnegative", None),
                                           (self.V, radius_v, "l1", l1_radius)]:
            if (np.linalg.norm(arr, 2) > radius * (1 + 1e-12)
                or (kind == "nonnegative" and arr.min() < -1e-12)
                or (kind == "l1" and np.abs(arr).sum() > budget * (1 + 1e-12))):
                raise ValueError("SPRING requires the supplied common initial point to be feasible")

    def _indices(self):
        if self.sample_indices is not None:
            return self.sample_indices(self.rng, self.batch_size)
        return self.rng.choice(self.n_components, self.batch_size, replace=False)

    def _estimate(self, U, V, previous, estimate, block):
        refresh = previous is None or self.rng.random() < self.parameters["refresh_probability"]
        if refresh:
            result = self.full_gradient(U, V)[block]
            counts = (self.n_components, 1, 0)
        else:
            indices = self._indices()
            result = (estimate + self.batch_gradient(U, V, indices)[block]
                      - self.batch_gradient(*previous, indices)[block])
            counts = (2 * self.batch_size, 0, 2)
        return result, refresh, counts

    def step(self):
        old_u, old_v = self.U.copy(), self.V.copy()
        gu, refresh_u, count_u = self._estimate(old_u, old_v, self._previous_u_point,
                                               self._estimate_u, 0)
        new_u, cert_u = project_intersection(
            old_u - self.parameters["step_u"] * gu, self.radius_u,
            kind="nonnegative", iteration=self.iteration,
            distance_tolerance=self.projection_tolerance,
            tolerance_power=self.projection_power,
            max_iterations=self.projection_max_iterations)
        # The second estimate tracks mixed points (U_{k+1},V_k), not (U_k,V_k).
        gv, refresh_v, count_v = self._estimate(new_u, old_v, self._previous_v_point,
                                               self._estimate_v, 1)
        new_v, cert_v = project_intersection(
            old_v - self.parameters["step_v"] * gv, self.radius_v,
            kind="l1", l1_radius=self.l1_radius, iteration=self.iteration,
            distance_tolerance=self.projection_tolerance,
            tolerance_power=self.projection_power,
            max_iterations=self.projection_max_iterations)
        self._previous_u_point = (old_u, old_v)
        self._previous_v_point = (new_u.copy(), old_v.copy())
        self._estimate_u, self._estimate_v = gu.copy(), gv.copy()
        self.U, self.V = new_u, new_v
        self.iteration += 1
        step_calls = count_u[0] + count_v[0]
        self.oracle_calls += step_calls
        self.full_gradient_calls += count_u[1] + count_v[1]
        self.sampled_gradient_calls += count_u[2] + count_v[2]
        return dict(iteration=self.iteration, oracle_calls=self.oracle_calls,
                    step_oracle_calls=step_calls, refresh_u=refresh_u, refresh_v=refresh_v,
                    projection_u=asdict(cert_u), projection_v=asdict(cert_v),
                    step_u=self.parameters["step_u"], step_v=self.parameters["step_v"],
                    mode=self.mode, exact_prox_theorem=False)

    def state_dict(self):
        fields = ("U", "V", "iteration", "oracle_calls", "full_gradient_calls",
                  "sampled_gradient_calls", "_estimate_u", "_estimate_v",
                  "_previous_u_point", "_previous_v_point", "parameters")
        state = {name: copy.deepcopy(getattr(self, name)) for name in fields}
        state["rng_state"] = copy.deepcopy(self.rng.bit_generator.state)
        state["n_components"] = self.n_components
        state["batch_size"] = self.batch_size
        state["mode"] = self.mode
        state["bounds"] = (self.radius_u, self.radius_v, self.l1_radius)
        state["projection_config"] = (self.projection_tolerance, self.projection_power,
                                      self.projection_max_iterations)
        return state

    def load_state_dict(self, state):
        if (state["n_components"] != self.n_components or state["batch_size"] != self.batch_size
            or state["mode"] != self.mode
            or tuple(state["bounds"]) != (self.radius_u, self.radius_v, self.l1_radius)
            or tuple(state["projection_config"]) != (self.projection_tolerance,
                                                        self.projection_power,
                                                        self.projection_max_iterations)
            or state["parameters"] != self.parameters):
            raise ValueError("checkpoint and constructor configuration do not match")
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        for name in ("U", "V", "iteration", "oracle_calls", "full_gradient_calls",
                     "sampled_gradient_calls", "_estimate_u", "_estimate_v",
                     "_previous_u_point", "_previous_v_point"):
            setattr(self, name, copy.deepcopy(state[name]))
