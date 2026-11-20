"""Matrix-free nonnegative factorization without spectral constraints.

``anls`` alternates numerical NNLS solves using SciPy 1.18.1's L-BFGS-B
reverse-communication implementation. Its complete work arrays are checkpointed;
the implementation is deliberately version-pinned rather than pretending that
private SciPy restart state is portable. Each accepted inner iterate is exposed.

``mu`` alternates multiplicative block updates. For a fixed other factor, write
the quadratic gradient as Hx-b, where H is entrywise nonnegative because the CT
projector and the other factor are nonnegative. The update is
``x * positive(b) / (Hx + negative(b))``. Splitting the signed right-hand side
is necessary: the measured sinogram is not entrywise nonnegative. No factor or
denominator floor, spectral projection, or radial repair is applied.

The problem supplies forward(image, ids), backward(sinogram, ids), y, and
train_ids. Every solver FP/BP is counted, including unsuccessful line-search
trials and the cached MU right-hand-side adjoint. Diagnostics are external.
"""

from copy import deepcopy

import numpy as np


SCIPY_VERSION = "1.18.1"
ANLS_DEFAULTS = dict(memory=10, relative_kkt_tolerance=1e-3, ftol=1e-9,
                     maxls=20, inner_maxiter=10)
COUNTERS = ("forward_views", "backward_views", "forward_calls", "backward_calls",
            "gradient_calls", "normal_operator_calls", "rhs_backward_calls",
            "rhs_backward_views", "oracle_calls")


def parameters(method, config=None):
    """Return the complete frozen parameter dictionary; reject unknown choices."""
    if method not in ("anls", "mu"):
        raise ValueError("Expected method 'anls' or 'mu'")
    supplied = {} if config is None else dict(config)
    if method == "mu":
        if supplied:
            raise ValueError("The multiplicative update has no tuning parameters")
        return {}
    if set(supplied) - set(ANLS_DEFAULTS):
        raise ValueError("Unknown ANLS parameter")
    result = ANLS_DEFAULTS | supplied
    for key in ("memory", "maxls", "inner_maxiter"):
        value = result[key]
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(key + " must be a positive integer")
        result[key] = int(value)
    for key in ("relative_kkt_tolerance", "ftol"):
        value = float(result[key])
        if not np.isfinite(value) or value < 0:
            raise ValueError(key + " must be finite and nonnegative")
        result[key] = value
    if result["relative_kkt_tolerance"] >= 1:
        raise ValueError("relative_kkt_tolerance must be less than one")
    return result


def projected_gradient_inf(x, gradient):
    """SciPy L-BFGS-B's lower-bound projected-gradient infinity norm.

    For lower bound zero this is ||min(x, gradient)||_inf, equivalently the
    unit-step gradient mapping ||x - max(x-gradient, 0)||_inf. It vanishes
    exactly at NNLS KKT points; positive gradients near zero are clipped by x.
    """
    return float(np.max(np.abs(np.minimum(x, gradient))))


def multiplicative_update(factor, normal_gradient, rhs_gradient):
    """Signed-RHS MU with explicit zero rules and no hidden positive floor.

    A zero numerator with positive denominator sets the entry to zero. A 0/0
    entry is unchanged; no denominator or factor floor is inserted.
    Existing zero factors remain zero: this is the usual MU zero-locking rule.
    Positive numerator/zero denominator is inconsistent with the assumed
    nonnegative least-squares operator and raises rather than adding epsilon.
    """
    x, q, b = (np.asarray(value, dtype=np.float64)
               for value in (factor, normal_gradient, rhs_gradient))
    if x.shape != q.shape or x.shape != b.shape:
        raise ValueError("MU factors and quadratic terms must have matching shapes")
    if any(not np.all(np.isfinite(value)) for value in (x, q, b)):
        raise FloatingPointError("Nonfinite MU quadratic term")
    if np.any(x < 0) or np.any(q < 0):
        raise ValueError("MU requires nonnegative factors and a nonnegative normal operator")
    numerator, denominator = np.maximum(b, 0), q + np.maximum(-b, 0)
    if np.any((denominator == 0) & (numerator > 0)):
        raise FloatingPointError("Positive MU numerator with zero denominator")
    ratio = np.ones_like(x)
    np.divide(numerator, denominator, out=ratio, where=denominator > 0)
    result = x * ratio
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Nonfinite multiplicative update")
    return result


class NonnegativeSolver:
    """Shared accounting, accepted-factor state, and atomic operator calls."""

    method = None

    def __init__(self, problem, U, V, *, config=None):
        self.problem = problem
        self.config = parameters(self.method, config)
        self.train_ids = np.array(problem.train_ids, dtype=np.intp, copy=True)
        if self.train_ids.ndim != 1 or not len(self.train_ids):
            raise ValueError("Expected nonempty training view IDs")
        self.observed = np.asarray(problem.y)[self.train_ids]
        if self.observed.ndim != 2 or not np.all(np.isfinite(self.observed)):
            raise ValueError("Expected a finite views-by-detectors sinogram")
        self.U, self.V = (np.array(value, dtype=np.float64, copy=True) for value in (U, V))
        self._validate_factors()
        self.counters = dict.fromkeys(COUNTERS, 0)
        self.iteration = self.completed_blocks = self.outer_sweeps = 0
        self.block = 0
        self.status = "running"
        self.phase = "start"
        self.work = {}
        reasons = ("kkt", "relative_function_decrease", "inner_maxiter", "line_search_failed")
        self.block_termination_counts = dict.fromkeys(
            reasons if self.method == "anls" else ("multiplicative_update",), 0)

    def _validate_factors(self):
        if (self.U.ndim != 2 or self.V.ndim != 2 or not self.U.size or not self.V.size
                or self.U.shape[1] != self.V.shape[1]):
            raise ValueError("Expected nonempty factors with matching ranks")
        if any(not np.all(np.isfinite(x)) or np.any(x < 0) for x in (self.U, self.V)):
            raise ValueError("Accepted factors must be finite and nonnegative")

    def _event(self, observation_type, **details):
        return dict(observation_type=observation_type,
                    accepted=observation_type == "accepted_update",
                    iteration=self.iteration, accepted_updates=self.iteration,
                    completed_blocks=self.completed_blocks, outer_sweeps=self.outer_sweeps,
                    block_termination_counts=deepcopy(self.block_termination_counts),
                    block=self.block, status=self.status, **self.counters, **details)

    def _affordable(self, cap, *, forward=0, backward=0):
        if cap is None:
            return True
        if isinstance(cap, bool) or int(cap) != cap or cap < 0:
            raise ValueError("view_budget must be a nonnegative integer")
        n = len(self.train_ids)
        return (self.counters["forward_views"] + forward*n <= cap
                and self.counters["backward_views"] + backward*n <= cap)

    def _forward(self, image):
        self.counters["forward_calls"] += 1
        self.counters["forward_views"] += len(self.train_ids)
        value = np.asarray(self.problem.forward(image, self.train_ids))
        if value.shape != self.observed.shape or not np.all(np.isfinite(value)):
            raise FloatingPointError("Invalid forward result")
        return value

    def _backward(self, sinogram):
        self.counters["backward_calls"] += 1
        self.counters["backward_views"] += len(self.train_ids)
        value = np.asarray(self.problem.backward(sinogram, self.train_ids), dtype=np.float64)
        if value.shape != (self.U.shape[0], self.V.shape[0]) or not np.all(np.isfinite(value)):
            raise FloatingPointError("Invalid adjoint result")
        return value / self.observed.size

    def _pair_finished(self):
        self.counters["gradient_calls"] += 1
        self.counters["oracle_calls"] += 1

    def _factor_gradient(self, image_gradient):
        return image_gradient @ self.V if self.block == 0 else image_gradient.T @ self.U

    def _replace_factor(self, factor):
        if self.block == 0:
            self.U = np.array(factor, dtype=np.float64, copy=True)
        else:
            self.V = np.array(factor, dtype=np.float64, copy=True)
        self._validate_factors()
        self.iteration += 1

    def _finish_block(self):
        self.completed_blocks += 1
        self.outer_sweeps = self.completed_blocks // 2
        self.block = 1 - self.block

    def step(self, view_budget=None):
        """Advance until one accepted update, a budget stop, or solver termination."""
        while True:
            event = self.advance(view_budget=view_budget)
            if event["observation_type"] != "operator_work":
                return event

    def state_dict(self):
        return deepcopy(dict(schema_version=1, method=self.method, config=self.config,
            train_ids=self.train_ids, observed_shape=self.observed.shape,
            U=self.U, V=self.V, counters=self.counters, iteration=self.iteration,
            completed_blocks=self.completed_blocks, outer_sweeps=self.outer_sweeps,
            block_termination_counts=self.block_termination_counts,
            block=self.block, status=self.status, phase=self.phase, work=self.work,
            scipy_version=SCIPY_VERSION if self.method == "anls" else None,
            lbfgsb_integer_dtype=np.dtype(self.integer_dtype).str if self.method == "anls" else None))

    def load_state_dict(self, state):
        state = deepcopy(state)
        if (state.get("schema_version") != 1 or state.get("method") != self.method
                or state.get("config") != self.config
                or not np.array_equal(state.get("train_ids"), self.train_ids)
                or tuple(state.get("observed_shape", ())) != self.observed.shape
                or np.shape(state.get("U")) != self.U.shape
                or np.shape(state.get("V")) != self.V.shape):
            raise ValueError("Checkpoint configuration or problem dimensions differ")
        if self.method == "anls" and state.get("scipy_version") != SCIPY_VERSION:
            raise ValueError("L-BFGS-B checkpoint requires the pinned SciPy version")
        if self.method == "anls" and state.get("lbfgsb_integer_dtype") != np.dtype(self.integer_dtype).str:
            raise ValueError("L-BFGS-B checkpoint integer ABI differs")
        c = state["counters"]
        if (set(c) != set(COUNTERS) or any(int(v) != v or v < 0 for v in c.values())
                or c["forward_views"] != len(self.train_ids)*c["forward_calls"]
                or c["backward_views"] != len(self.train_ids)*c["backward_calls"]
                or c["rhs_backward_views"] != len(self.train_ids)*c["rhs_backward_calls"]
                or c["backward_calls"] != c["gradient_calls"] + c["rhs_backward_calls"]
                or c["forward_calls"] - c["gradient_calls"] != (state["phase"] == "backward")
                or c["oracle_calls"] != c["gradient_calls"] + c["rhs_backward_calls"]):
            raise ValueError("Invalid checkpoint work counters")
        if (state["block"] not in (0, 1)
                or any(isinstance(state[key], bool) or int(state[key]) != state[key] or state[key] < 0
                       for key in ("iteration", "completed_blocks", "outer_sweeps"))
                or state["outer_sweeps"] != state["completed_blocks"] // 2
                or state["block"] != state["completed_blocks"] % 2
                or state["status"] not in ("running", "converged", "stalled")):
            raise ValueError("Invalid checkpoint iteration state")
        phases = ("start", "setulb", "forward", "backward") if self.method == "anls" else (
            "rhs", "forward", "backward")
        if state["phase"] not in phases:
            raise ValueError("Invalid checkpoint phase")
        stops = state.get("block_termination_counts", {})
        if (set(stops) != set(self.block_termination_counts)
                or any(int(v) != v or v < 0 for v in stops.values())
                or sum(stops.values()) != state["completed_blocks"]):
            raise ValueError("Invalid checkpoint block termination counts")
        if self.method == "anls":
            if c["normal_operator_calls"] or c["rhs_backward_calls"]:
                raise ValueError("Invalid ANLS work counters")
            if state["phase"] != "start":
                self._validate_lbfgsb_state(state)
        elif (c["normal_operator_calls"] != c["gradient_calls"]
              or c["rhs_backward_calls"] not in (0, 1)):
            raise ValueError("Invalid MU work counters")
        for key in ("U", "V", "counters", "iteration", "completed_blocks",
                    "outer_sweeps", "block_termination_counts", "block", "status", "phase", "work"):
            setattr(self, key, state[key])
        self._validate_factors()


class MultiplicativeUpdates(NonnegativeSolver):
    method = "mu"

    def __init__(self, problem, U, V, *, config=None):
        super().__init__(problem, U, V, config=config)
        if np.any(self.U <= 0) or np.any(self.V <= 0):
            raise ValueError("MU requires the shared strictly positive initialization")
        self.phase = "rhs"

    def advance(self, view_budget=None):
        if self.phase == "rhs":
            if not self._affordable(view_budget, backward=1):
                return self._event("budget_stop")
            self.work["rhs_image"] = self._backward(self.observed)
            self.counters["rhs_backward_calls"] += 1
            self.counters["rhs_backward_views"] += len(self.train_ids)
            self.counters["oracle_calls"] += 1
            self.phase = "forward"
            return self._event("operator_work", operation="rhs_backward")
        if self.phase == "forward":
            # Reserve both directions before starting a pair; no unusable last FP.
            if not self._affordable(view_budget, forward=1, backward=1):
                return self._event("budget_stop")
            self.work["predicted"] = self._forward(self.U @ self.V.T)
            self.phase = "backward"
            return self._event("operator_work", operation="forward")
        if self.phase == "backward":
            if not self._affordable(view_budget, backward=1):
                return self._event("budget_stop")
            normal = self._backward(self.work.pop("predicted"))
            self._pair_finished()
            self.counters["normal_operator_calls"] += 1
            q, b = self._factor_gradient(normal), self._factor_gradient(self.work["rhs_image"])
            current = self.U if self.block == 0 else self.V
            updated_block = self.block
            self._replace_factor(multiplicative_update(current, q, b))
            self.block_termination_counts["multiplicative_update"] += 1
            self._finish_block()
            self.phase = "forward"
            return self._event("accepted_update", updated_block=updated_block,
                               block_inner_iteration=1, block_finished=True,
                               block_stop_reason="multiplicative_update")
        raise ValueError("Unknown MU checkpoint phase")


class AlternatingNNLS(NonnegativeSolver):
    """Inexact alternating NNLS with version-pinned L-BFGS-B inner solves."""

    method = "anls"

    def __init__(self, problem, U, V, *, config=None):
        import scipy
        from scipy.optimize import _lbfgsb, _lbfgsb_py

        if scipy.__version__ != SCIPY_VERSION:
            raise RuntimeError("Exact ANLS restart requires SciPy " + SCIPY_VERSION)
        self._setulb = _lbfgsb.setulb
        self.integer_dtype = np.int64 if _lbfgsb_py.HAS_ILP64 else np.int32
        super().__init__(problem, U, V, config=config)
        self.work = dict(no_update_blocks=0, no_update_reasons=[])

    def _validate_lbfgsb_state(self, state):
        """Check dimensions/dtypes before passing checkpoint buffers to C."""
        lb = state["work"]["lb"]
        n = np.size(state["U"] if state["block"] == 0 else state["V"])
        memory = self.config["memory"]
        floats = {"x": (n,), "l": (n,), "u": (n,), "f": (), "g": (n,),
                  "wa": (2*memory*n+5*n+11*memory*memory+8*memory,), "dsave": (29,)}
        integers = {"nbd": (n,), "iwa": (3*n,), "task": (2,), "ln_task": (2,),
                    "lsave": (4,), "isave": (44,)}
        for fields, dtype in ((floats, np.float64), (integers, self.integer_dtype)):
            for key, shape in fields.items():
                value = lb.get(key)
                if (not isinstance(value, np.ndarray) or value.shape != shape
                        or value.dtype != dtype or not value.flags.c_contiguous
                        or not np.all(np.isfinite(value))):
                    raise ValueError("Invalid L-BFGS-B restart array: " + key)
        if (np.any(lb["l"] != 0) or np.any(lb["u"] != 0) or np.any(lb["nbd"] != 1)
                or not np.isfinite(lb["pgtol"]) or lb["pgtol"] < 0):
            raise ValueError("Invalid L-BFGS-B bounds or tolerance")

    def _start_block(self):
        x = np.array(self.U if self.block == 0 else self.V, copy=True).ravel()
        n, memory = x.size, self.config["memory"]
        self.work["lb"] = dict(x=x, l=np.zeros(n), u=np.zeros(n),
            nbd=np.ones(n, dtype=self.integer_dtype), f=np.array(0., dtype=np.float64),
            g=np.zeros(n), wa=np.zeros(2*memory*n + 5*n + 11*memory*memory + 8*memory),
            iwa=np.zeros(3*n, dtype=self.integer_dtype), task=np.zeros(2, dtype=self.integer_dtype),
            ln_task=np.zeros(2, dtype=self.integer_dtype), lsave=np.zeros(4, dtype=self.integer_dtype),
            isave=np.zeros(44, dtype=self.integer_dtype), dsave=np.zeros(29),
            pgtol=0., initial_kkt=None, inner_iterations=0)
        self.phase = "setulb"

    def _end_block(self, reason):
        lb = self.work["lb"]
        no_update = lb["inner_iterations"] == 0
        self.work["no_update_blocks"] = self.work["no_update_blocks"] + 1 if no_update else 0
        self.work["no_update_reasons"] = self.work["no_update_reasons"] + [reason] if no_update else []
        self.block_termination_counts[reason] += 1
        self.work["last_block_stop_reason"] = reason
        self._finish_block()
        self.phase = "start"
        if self.work["no_update_blocks"] >= 2:
            self.status = ("converged" if self.work["no_update_reasons"][-2:] == ["kkt", "kkt"]
                           else "stalled")

    def advance(self, view_budget=None):
        if self.status != "running":
            return self._event(self.status)
        while True:
            if self.phase == "start":
                self._start_block()
            lb = self.work["lb"]
            if self.phase == "setulb":
                self._setulb(self.config["memory"], lb["x"], lb["l"], lb["u"],
                    lb["nbd"], lb["f"], lb["g"], self.config["ftol"]/np.finfo(float).eps,
                    lb["pgtol"], lb["wa"], lb["iwa"], lb["task"], lb["lsave"],
                    lb["isave"], lb["dsave"], self.config["maxls"], lb["ln_task"])
                task = int(lb["task"][0])
                if task == 3:
                    self.phase = "forward"
                elif task == 1:
                    shape = self.U.shape if self.block == 0 else self.V.shape
                    updated_block = self.block
                    self._replace_factor(lb["x"].reshape(shape))
                    lb["inner_iterations"] += 1
                    self.work["no_update_blocks"] = 0
                    self.work["no_update_reasons"] = []
                    kkt = projected_gradient_inf(lb["x"], lb["g"])
                    finished = lb["inner_iterations"] >= self.config["inner_maxiter"]
                    details = dict(updated_block=updated_block,
                        block_inner_iteration=lb["inner_iterations"], block_finished=finished,
                        block_stop_reason="inner_maxiter" if finished else None,
                        kkt_inf_norm=kkt, kkt_target=lb["pgtol"],
                        solver_loss=float(lb["f"]))
                    if finished:
                        self._end_block("inner_maxiter")
                    return self._event("accepted_update", **details)
                elif task in (4, 8):
                    # SciPy may finish on relative function decrease or fail its
                    # line search. Neither is silently called KKT convergence.
                    kkt = projected_gradient_inf(lb["x"], lb["g"])
                    if task == 8:
                        reason = "line_search_failed"
                    elif int(lb["task"][1]) == 401:
                        reason = "kkt"
                    elif int(lb["task"][1]) == 402:
                        reason = "relative_function_decrease"
                    else:
                        raise FloatingPointError("Unknown L-BFGS-B convergence code "
                                                 + repr(lb["task"].tolist()))
                    self._end_block(reason)
                    if self.status != "running":
                        return self._event(self.status, block_stop_reason=reason,
                                           kkt_inf_norm=kkt, kkt_target=lb["pgtol"])
                    continue
                else:
                    raise FloatingPointError("Unexpected L-BFGS-B task " + repr(lb["task"].tolist()))
            if self.phase == "forward":
                if not self._affordable(view_budget, forward=1, backward=1):
                    return self._event("budget_stop")
                shape = self.U.shape if self.block == 0 else self.V.shape
                trial = lb["x"].reshape(shape)
                image = trial @ self.V.T if self.block == 0 else self.U @ trial.T
                # Preserve the projector's native residual arithmetic, matching
                # CTProblem.grad/image_loss, then accumulate squared loss in f64.
                self.work["residual"] = self._forward(image) - self.observed
                self.phase = "backward"
                return self._event("operator_work", operation="forward")
            if self.phase == "backward":
                if not self._affordable(view_budget, backward=1):
                    return self._event("budget_stop")
                residual = self.work.pop("residual")
                image_gradient = self._backward(residual)
                self._pair_finished()
                lb["f"][...] = 0.5 * np.mean(np.square(residual, dtype=np.float64))
                lb["g"][...] = self._factor_gradient(image_gradient).ravel()
                if not np.isfinite(lb["f"]) or not np.all(np.isfinite(lb["g"])):
                    raise FloatingPointError("Nonfinite NNLS objective or gradient")
                if lb["initial_kkt"] is None:
                    lb["initial_kkt"] = projected_gradient_inf(lb["x"], lb["g"])
                    lb["pgtol"] = self.config["relative_kkt_tolerance"] * lb["initial_kkt"]
                self.phase = "setulb"
                return self._event("operator_work", operation="backward")


def make_solver(method, problem, U, V, config=None):
    """Construct a solver; initialization is supplied by the shared protocol."""
    classes = {"anls": AlternatingNNLS, "mu": MultiplicativeUpdates}
    if method not in classes:
        raise ValueError("Expected method 'anls' or 'mu'")
    return classes[method](problem, U, V, config=config)
