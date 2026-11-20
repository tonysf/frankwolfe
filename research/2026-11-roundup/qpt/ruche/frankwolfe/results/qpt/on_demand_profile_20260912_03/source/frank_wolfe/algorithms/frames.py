import numpy as np
from tqdm import tqdm
from frank_wolfe.algorithms.base import FrankWolfe


class FramesFrankWolfe(FrankWolfe):
    """
    FRAMES: Frank-Wolfe with Moreau envelope smoothing.

    Solves: min_{x in C} f(x) + g(Tx)

    where f is smooth, g is either an indicator (Assumption 1.4(I)) or
    Lipschitz weakly convex (Assumption 1.4(II)), and T is linear.  The
    implementation also exposes ``linear_operator_adjoint_at`` for explicitly
    experimental nonlinear-composite uses; the standard FRAMES assumptions
    still require a linear T.

    The algorithm smooths g via its Moreau envelope with parameter beta_k -> 0
    and applies one FW step per iteration with open-loop step size gamma_k -> 0.

    Schedules (Theorem 3.5):
        gamma_k = 1 / (k+1)^{1/2}
        beta_k  = beta_0 / (k+1)^{1/4}
    """

    def __init__(self, objective_fn, lmo_fn, prox_fn, objective_type):
        """
        Parameters
        ----------
        objective_fn : ObjectiveFunction
            Must implement ``evaluate``, ``gradient``, ``linear_operator``,
            and either ``linear_operator_adjoint`` or the point-dependent
            ``linear_operator_adjoint_at`` hook.
        lmo_fn : callable
            Linear minimization oracle for the constraint set C.
        prox_fn : callable
            prox_{beta * g}(y) for the nonsmooth term g.
            Signature: prox_fn(y, beta) -> array.
        objective_type : str
            "indicator" for g = iota_D (Assumption 1.4(I)),
            "lipschitz" for g Lipschitz weakly convex (Assumption 1.4(II)).
        """
        super().__init__(objective_fn, lmo_fn)
        self.prox = prox_fn
        self.objective_type = objective_type
        self.ns_gaps = None

    @staticmethod
    def _resolve_schedule(schedule, default_schedule, name):
        if schedule is None:
            return default_schedule
        if callable(schedule):
            return schedule
        if np.isscalar(schedule):
            return lambda _: schedule
        raise TypeError(f"{name} must be a callable, scalar, or None.")

    @staticmethod
    def _schedule_value(schedule, iteration, name):
        value = schedule(iteration)
        if (
            not np.isscalar(value)
            or isinstance(value, (str, bytes))
            or np.iscomplexobj(value)
        ):
            raise TypeError(
                f"{name} must return a real scalar; got {value!r} at "
                f"iteration {iteration}."
            )
        try:
            return float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError(
                f"{name} must return a real scalar; got {value!r} at "
                f"iteration {iteration}."
            ) from error

    def _apply_operator_adjoint(self, point, value):
        """Apply a point-dependent adjoint when one is available."""

        adjoint_at = getattr(
            self.objective, "linear_operator_adjoint_at", None
        )
        if callable(adjoint_at):
            return adjoint_at(point, value)
        return self.objective.linear_operator_adjoint(value)

    def run(self, x0, beta0=1.0, n_steps=int(1e2), show_progress=True):
        self.x = self.lmo(self.objective.gradient(x0))
        self.func_vals = np.zeros(n_steps)
        self.gaps = np.zeros(n_steps)
        self.ns_gaps = np.zeros(n_steps)
        self.num_oracles = np.zeros(n_steps)

        for i in tqdm(range(n_steps), desc="FRAMES Progress", disable=not show_progress):
            # Algorithm 1 schedules
            beta = beta0 / (i + 1) ** 0.25
            step_size = 1.0 / (i + 1) ** 0.5

            # Smoothed gradient: nabla f(x) + T^* (Tx - prox_{beta g}(Tx)) / beta
            grad = self.objective.gradient(self.x)
            Tx = self.objective.linear_operator(self.x)
            moreau_grad = self._apply_operator_adjoint(
                self.x, Tx - self.prox(Tx, beta)
            ) / beta
            combined_grad = grad + moreau_grad

            # LMO step
            direction = self.lmo(combined_grad)
            self.num_oracles[i] += 1

            # Smoothed gap: <nabla Phi_k(x_k), x_k - s_k>
            gap = np.sum(combined_grad * (self.x - direction))
            self.gaps[i] = gap

            # Objective value f(x_k) (without g, since g may be infinite)
            self.func_vals[i] = self.objective.evaluate(self.x)

            # Nonsmooth gap / feasibility measure
            if self.objective_type == "indicator":
                # 0.5 * dist_D^2(x_k) when T = Id,
                # or 0.5 * ||Tx - P_D(Tx)||^2 more generally
                ns_gap = 0.5 * np.linalg.norm(
                    (Tx - self.prox(Tx, beta)).flatten()
                ) ** 2
            elif self.objective_type == "lipschitz":
                ns_grad = self._apply_operator_adjoint(
                    self.x, self.objective.minimal_norm_selection(Tx)
                )
                combined_ns_grad = grad + ns_grad
                ns_direction = self.lmo(combined_ns_grad)
                ns_gap = np.sum(combined_ns_grad * (self.x - ns_direction))
            else:
                raise ValueError(f"Unknown objective type: {self.objective_type}")

            self.ns_gaps[i] = ns_gap

            # Update: x_{k+1} = (1 - gamma_k) x_k + gamma_k s_k
            self.x = (1 - step_size) * self.x + step_size * direction

        self.num_oracles = np.cumsum(self.num_oracles)


Frames = FramesFrankWolfe


class AdaptiveFramesFrankWolfe(FramesFrankWolfe):
    """FRAMES with a gap-adaptive Moreau smoothing parameter.

    The Frank-Wolfe step size defaults to the usual open-loop schedule
    ``gamma_k = 1 / sqrt(k + 1)`` and can be replaced when ``run`` is called.
    Starting from ``beta_0``, the smoothing parameter is held fixed until the
    Frank-Wolfe gap of the current smoothed objective is strictly below it.
    The update for the following iteration is

        beta_{k+1} = beta_k / 2,  if gap_k < beta_k,
                     beta_k,      otherwise.

    The strict comparison means that equality does not trigger a reduction.
    """

    def __init__(self, objective_fn, lmo_fn, prox_fn, objective_type):
        super().__init__(objective_fn, lmo_fn, prox_fn, objective_type)
        self.smoothing_parameters = None
        self.step_sizes = None
        self.step_size_schedule = None
        self.next_smoothing_parameter = None

    def run(
        self,
        x0,
        beta0=1.0,
        n_steps=int(1e2),
        show_progress=True,
        *,
        step_size_schedule=None,
    ):
        """Run adaptive FRAMES from ``x0``.

        ``smoothing_parameters[k]`` stores the value used at iteration ``k``.
        If the final observed gap triggers a reduction,
        ``next_smoothing_parameter`` stores the value that would be used by
        the next iteration.

        ``step_size_schedule`` may be a callable of the zero-based iteration
        or a scalar constant.  Values must lie in ``[0, 1]``.  When omitted,
        the usual ``1 / sqrt(k + 1)`` open-loop schedule is used.  Realized
        values are stored in ``step_sizes``.
        """
        if not isinstance(n_steps, (int, np.integer)) or n_steps < 0:
            raise ValueError("n_steps must be a nonnegative integer.")
        if (
            not np.isscalar(beta0)
            or isinstance(beta0, (str, bytes))
            or np.iscomplexobj(beta0)
        ):
            raise TypeError("beta0 must be a positive finite real number.")
        try:
            beta = float(beta0)
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError(
                "beta0 must be a positive finite real number."
            ) from error
        if not np.isfinite(beta) or beta <= 0.0:
            raise ValueError("beta0 must be a positive finite real number.")

        step_size_fn = self._resolve_schedule(
            step_size_schedule,
            lambda iteration: 1.0 / (iteration + 1) ** 0.5,
            "step_size_schedule",
        )
        self.step_size_schedule = step_size_fn

        self.x = np.asarray(self.lmo(self.objective.gradient(x0)))
        if not np.all(np.isfinite(self.x)):
            raise ValueError("The initial LMO point must contain finite values.")
        self.func_vals = np.zeros(n_steps)
        self.gaps = np.zeros(n_steps)
        self.ns_gaps = np.zeros(n_steps)
        self.num_oracles = np.zeros(n_steps)
        self.smoothing_parameters = np.zeros(n_steps)
        self.step_sizes = np.zeros(n_steps)

        iterations = tqdm(
            range(n_steps),
            desc="Adaptive FRAMES Progress",
            disable=not show_progress,
        )
        for i in iterations:
            step_size = self._schedule_value(
                step_size_fn, i, "step_size_schedule"
            )
            if not np.isfinite(step_size) or not 0.0 <= step_size <= 1.0:
                raise ValueError(
                    "step_size_schedule must return a finite value in [0, 1]; "
                    f"got {step_size!r} at iteration {i}."
                )
            self.smoothing_parameters[i] = beta
            self.step_sizes[i] = step_size

            grad = np.asarray(self.objective.gradient(self.x))
            Tx = np.asarray(self.objective.linear_operator(self.x))
            prox_Tx = np.asarray(self.prox(Tx, beta))
            if prox_Tx.shape != Tx.shape:
                raise ValueError(
                    "The proximal result must have the same shape as Tx; "
                    f"got {prox_Tx.shape} and {Tx.shape}."
                )
            moreau_grad = np.asarray(
                self._apply_operator_adjoint(
                    self.x, Tx - prox_Tx
                )
            ) / beta
            combined_grad = grad + moreau_grad
            if combined_grad.shape != self.x.shape:
                raise ValueError(
                    "The smoothed gradient must have the same shape as x; "
                    f"got {combined_grad.shape} and {self.x.shape}."
                )
            if not np.all(np.isfinite(combined_grad)):
                raise ValueError(
                    "The smoothed gradient must contain finite values."
                )

            direction = np.asarray(self.lmo(combined_grad))
            self.num_oracles[i] += 1
            if direction.shape != self.x.shape:
                raise ValueError(
                    "The LMO direction must have the same shape as x; "
                    f"got {direction.shape} and {self.x.shape}."
                )
            if not np.all(np.isfinite(direction)):
                raise ValueError("The LMO direction must contain finite values.")

            gap = np.vdot(combined_grad, self.x - direction).real
            if not np.isfinite(gap):
                raise ValueError(
                    "The smoothed Frank-Wolfe gap must be finite."
                )
            self.gaps[i] = gap
            self.func_vals[i] = self.objective.evaluate(self.x)

            if self.objective_type == "indicator":
                self.ns_gaps[i] = 0.5 * np.linalg.norm(
                    (Tx - prox_Tx).flatten()
                ) ** 2
            elif self.objective_type == "lipschitz":
                ns_grad = self._apply_operator_adjoint(
                    self.x, self.objective.minimal_norm_selection(Tx)
                )
                combined_ns_grad = grad + ns_grad
                ns_direction = self.lmo(combined_ns_grad)
                self.ns_gaps[i] = np.vdot(
                    combined_ns_grad, self.x - ns_direction
                ).real
            else:
                raise ValueError(
                    f"Unknown objective type: {self.objective_type}"
                )

            self.x = (1 - step_size) * self.x + step_size * direction
            if gap < beta:
                reduced_beta = beta * 0.5
                if reduced_beta == 0.0:
                    raise FloatingPointError(
                        "The smoothing parameter cannot be halved without "
                        "floating-point underflow."
                    )
                beta = reduced_beta

        self.next_smoothing_parameter = beta
        self.num_oracles = np.cumsum(self.num_oracles)


AdaptiveFrames = AdaptiveFramesFrankWolfe


class StochasticFramesFrankWolfe(FramesFrankWolfe):
    """Stochastic FRAMES with a momentum estimate of the smooth gradient.

    Solves ``min_{x in C} E[f(x, xi)] + g(Tx)`` using one stochastic
    gradient sample per iteration.  Only the gradient of the smooth term is
    estimated:

        d_0 = grad f(x_0, xi_0),
        d_k = (1 - rho_k) d_{k-1} + rho_k grad f(x_k, xi_k),  k >= 1.

    The Moreau-envelope gradient is evaluated exactly at ``x_k`` and added to
    ``d_k`` before the linear minimization oracle is called.  The FRAMES step
    size and smoothing schedules are retained by default and can be replaced
    when ``run`` is called.
    """

    def __init__(
        self,
        objective_fn,
        lmo_fn,
        prox_fn,
        objective_type,
        stochastic_gradient_fn=None,
    ):
        """
        Parameters
        ----------
        objective_fn : ObjectiveFunction
            Must implement ``evaluate``, ``linear_operator``, and either
            ``linear_operator_adjoint`` or ``linear_operator_adjoint_at``.  It
            must also implement
            ``stochastic_gradient(x)`` unless ``stochastic_gradient_fn`` is
            supplied.
        lmo_fn : callable
            Linear minimization oracle for the constraint set C.
        prox_fn : callable
            ``prox_{beta * g}(y)`` for the nonsmooth term g.
        objective_type : str
            Either ``"indicator"`` or ``"lipschitz"``.
        stochastic_gradient_fn : callable, optional
            Callable with signature ``stochastic_gradient_fn(x) -> array``.
            The callable can own an RNG or sampler.  If omitted,
            ``objective_fn.stochastic_gradient`` is used.
        """
        super().__init__(objective_fn, lmo_fn, prox_fn, objective_type)

        if objective_type not in {"indicator", "lipschitz"}:
            raise ValueError(f"Unknown objective type: {objective_type}")

        if stochastic_gradient_fn is None:
            stochastic_gradient_fn = getattr(
                objective_fn, "stochastic_gradient", None
            )
        if not callable(stochastic_gradient_fn):
            raise TypeError(
                "A callable stochastic gradient oracle is required. Provide "
                "stochastic_gradient_fn or implement "
                "objective.stochastic_gradient(x)."
            )

        self.stochastic_gradient = stochastic_gradient_fn
        self.gradient_estimate = None
        self.gradient_estimator = None
        self.estimated_gaps = None
        self.momentum_weights = None
        self.smoothing_parameters = None
        self.step_sizes = None
        self.num_gradient_oracles = None
        self.num_stochastic_oracles = None
        self.rho_schedule = None
        self.smoothing_schedule = None
        self.step_size_schedule = None

    @staticmethod
    def _default_rho(iteration):
        """Standard momentum stochastic Frank-Wolfe schedule."""
        return min(1.0, 4.0 / (iteration + 8) ** (2.0 / 3.0))

    def run(
        self,
        x0,
        beta0=1.0,
        n_steps=int(1e2),
        show_progress=True,
        rho_schedule=None,
        *,
        smoothing_schedule=None,
        step_size_schedule=None,
        evaluate_objective=True,
        iterate_callback=None,
        iterate_callback_frequency=1,
    ):
        """Run stochastic FRAMES from a feasible initial point.

        Parameters
        ----------
        x0 : array_like
            Initial point in C.  Unlike the legacy deterministic runner, this
            point is used directly so that exactly one stochastic gradient is
            requested per iteration.
        beta0 : float, default=1.0
            Initial Moreau smoothing parameter for the default schedule.  It
            is not applied when ``smoothing_schedule`` is supplied.
        n_steps : int, default=100
            Number of stochastic FRAMES iterations.
        show_progress : bool, default=True
            Whether to display a tqdm progress bar.
        rho_schedule : callable or float, optional
            Weight of the new stochastic gradient in the momentum estimate.
            A callable receives the zero-based iteration.  A scalar applies a
            constant weight.  Values must lie in ``(0, 1]``.  The schedule is
            evaluated and recorded at iteration zero, but the first sample
            initializes ``d_0`` exactly, independently of ``rho_0``.  The
            default is ``rho_k = 4 / (k + 8)^(2/3)``.
        smoothing_schedule : callable or float, optional
            Moreau smoothing parameter ``beta_k``.  A callable receives the
            zero-based iteration and returns the actual parameter; a scalar
            makes it constant.  Values must be positive.  The default is
            ``beta0 / (k + 1)^(1/4)``.
        step_size_schedule : callable or float, optional
            Frank-Wolfe step size ``gamma_k``.  A callable receives the
            zero-based iteration and returns the actual step size; a scalar
            makes it constant.  Values must lie in ``[0, 1]``.  The default is
            ``1 / (k + 1)^(1/2)``.
        evaluate_objective : bool, default=True
            Whether to call ``objective.evaluate`` at every iteration.  Set
            this to ``False`` when a full objective pass is much more
            expensive than one stochastic-gradient sample.  In that case,
            ``func_vals`` is filled with ``nan``.
        iterate_callback : callable, optional
            Called as ``iterate_callback(completed_steps, x)`` at the initial
            and final points and according to ``iterate_callback_frequency``.
            A copy of the iterate is supplied so the callback cannot mutate
            the optimizer state.
        iterate_callback_frequency : int, default=1
            Call the iterate callback after every this many completed steps.
            Zero records only the initial and final points.  The final point
            is always emitted even when it is not a multiple of the frequency.

        Notes
        -----
        ``num_oracles`` counts the LMO calls used to update the iterate.  In
        ``"lipschitz"`` mode, the additional LMO used only to report the
        nonsmooth-gap diagnostic is not included, matching deterministic
        FRAMES.
        """
        if not isinstance(n_steps, (int, np.integer)) or n_steps < 0:
            raise ValueError("n_steps must be a nonnegative integer.")
        if not np.isfinite(beta0) or beta0 <= 0:
            raise ValueError("beta0 must be a positive finite number.")
        if not isinstance(evaluate_objective, (bool, np.bool_)):
            raise TypeError("evaluate_objective must be a boolean.")
        if iterate_callback is not None and not callable(iterate_callback):
            raise TypeError("iterate_callback must be callable or None.")
        if (
            not isinstance(iterate_callback_frequency, (int, np.integer))
            or iterate_callback_frequency < 0
        ):
            raise ValueError(
                "iterate_callback_frequency must be a nonnegative integer."
            )

        rho_fn = self._resolve_schedule(
            rho_schedule,
            self._default_rho,
            "rho_schedule",
        )
        smoothing_fn = self._resolve_schedule(
            smoothing_schedule,
            lambda iteration: beta0 / (iteration + 1) ** 0.25,
            "smoothing_schedule",
        )
        step_size_fn = self._resolve_schedule(
            step_size_schedule,
            lambda iteration: 1.0 / (iteration + 1) ** 0.5,
            "step_size_schedule",
        )
        self.rho_schedule = rho_fn
        self.smoothing_schedule = smoothing_fn
        self.step_size_schedule = step_size_fn

        self.x = np.array(x0, copy=True)
        self.func_vals = np.full(n_steps, np.nan)
        self.gaps = np.zeros(n_steps)
        self.estimated_gaps = self.gaps
        self.ns_gaps = np.zeros(n_steps)
        self.num_oracles = np.zeros(n_steps)
        self.num_stochastic_oracles = np.zeros(n_steps)
        self.num_gradient_oracles = self.num_stochastic_oracles
        self.momentum_weights = np.zeros(n_steps)
        self.smoothing_parameters = np.zeros(n_steps)
        self.step_sizes = np.zeros(n_steps)

        estimate_dtype = np.result_type(self.x.dtype, np.float64)
        gradient_estimate = np.zeros_like(self.x, dtype=estimate_dtype)
        self.gradient_estimate = gradient_estimate.copy()
        self.gradient_estimator = self.gradient_estimate

        if iterate_callback is not None:
            iterate_callback(0, self.x.copy())

        iterations = tqdm(
            range(n_steps),
            desc="Stochastic FRAMES Progress",
            disable=not show_progress,
        )
        for i in iterations:
            rho = self._schedule_value(rho_fn, i, "rho_schedule")
            beta = self._schedule_value(
                smoothing_fn, i, "smoothing_schedule"
            )
            step_size = self._schedule_value(
                step_size_fn, i, "step_size_schedule"
            )
            if not np.isfinite(rho) or not 0.0 < rho <= 1.0:
                raise ValueError(
                    "rho_schedule must return a finite value in (0, 1]; "
                    f"got {rho!r} at iteration {i}."
                )
            if not np.isfinite(beta) or beta <= 0.0:
                raise ValueError(
                    "smoothing_schedule must return a positive finite value; "
                    f"got {beta!r} at iteration {i}."
                )
            if not np.isfinite(step_size) or not 0.0 <= step_size <= 1.0:
                raise ValueError(
                    "step_size_schedule must return a finite value in [0, 1]; "
                    f"got {step_size!r} at iteration {i}."
                )
            self.momentum_weights[i] = rho
            self.smoothing_parameters[i] = beta
            self.step_sizes[i] = step_size

            stochastic_grad = np.asarray(self.stochastic_gradient(self.x))
            self.num_stochastic_oracles[i] += 1
            if stochastic_grad.shape != self.x.shape:
                raise ValueError(
                    "The stochastic gradient must have the same shape as x; "
                    f"got {stochastic_grad.shape} and {self.x.shape}."
                )
            if not np.all(np.isfinite(stochastic_grad)):
                raise ValueError(
                    "The stochastic gradient must contain finite values."
                )
            if i == 0:
                gradient_estimate = stochastic_grad.copy()
            else:
                gradient_estimate = (
                    (1.0 - rho) * gradient_estimate + rho * stochastic_grad
                )
            self.gradient_estimate = gradient_estimate.copy()
            self.gradient_estimator = self.gradient_estimate

            Tx = np.asarray(self.objective.linear_operator(self.x))
            prox_Tx = np.asarray(self.prox(Tx, beta))
            if prox_Tx.shape != Tx.shape:
                raise ValueError(
                    "The proximal result must have the same shape as Tx; "
                    f"got {prox_Tx.shape} and {Tx.shape}."
                )
            if not np.all(np.isfinite(prox_Tx)):
                raise ValueError("The proximal result must contain finite values.")
            moreau_residual = Tx - prox_Tx
            moreau_grad = np.asarray(
                self._apply_operator_adjoint(
                    self.x, moreau_residual
                )
            ) / beta
            if moreau_grad.shape != self.x.shape:
                raise ValueError(
                    "The adjoint Moreau gradient must have the same shape as x; "
                    f"got {moreau_grad.shape} and {self.x.shape}."
                )
            if not np.all(np.isfinite(moreau_grad)):
                raise ValueError(
                    "The adjoint Moreau gradient must contain finite values."
                )
            combined_grad = gradient_estimate + moreau_grad

            direction = np.asarray(self.lmo(combined_grad))
            self.num_oracles[i] += 1
            if direction.shape != self.x.shape:
                raise ValueError(
                    "The LMO direction must have the same shape as x; "
                    f"got {direction.shape} and {self.x.shape}."
                )
            if not np.all(np.isfinite(direction)):
                raise ValueError("The LMO direction must contain finite values.")

            self.gaps[i] = np.vdot(
                combined_grad, self.x - direction
            ).real
            if evaluate_objective:
                self.func_vals[i] = self.objective.evaluate(self.x)

            if self.objective_type == "indicator":
                self.ns_gaps[i] = 0.5 * np.linalg.norm(
                    moreau_residual.flatten()
                ) ** 2
            elif self.objective_type == "lipschitz":
                ns_grad = self._apply_operator_adjoint(
                    self.x, self.objective.minimal_norm_selection(Tx)
                )
                combined_ns_grad = gradient_estimate + ns_grad
                ns_direction = np.asarray(self.lmo(combined_ns_grad))
                if ns_direction.shape != self.x.shape:
                    raise ValueError(
                        "The diagnostic LMO direction must have the same "
                        f"shape as x; got {ns_direction.shape} and "
                        f"{self.x.shape}."
                    )
                if not np.all(np.isfinite(ns_direction)):
                    raise ValueError(
                        "The diagnostic LMO direction must contain finite "
                        "values."
                    )
                self.ns_gaps[i] = np.vdot(
                    combined_ns_grad, self.x - ns_direction
                ).real
            else:
                raise ValueError(
                    f"Unknown objective type: {self.objective_type}"
                )

            self.x = (1 - step_size) * self.x + step_size * direction
            completed_steps = i + 1
            callback_due = (
                completed_steps == n_steps
                or (
                    iterate_callback_frequency > 0
                    and completed_steps % iterate_callback_frequency == 0
                )
            )
            if iterate_callback is not None and callback_due:
                iterate_callback(i + 1, self.x.copy())

        self.num_oracles = np.cumsum(self.num_oracles)
        self.num_stochastic_oracles = np.cumsum(self.num_stochastic_oracles)
        self.num_gradient_oracles = self.num_stochastic_oracles


StochasticFrames = StochasticFramesFrankWolfe
