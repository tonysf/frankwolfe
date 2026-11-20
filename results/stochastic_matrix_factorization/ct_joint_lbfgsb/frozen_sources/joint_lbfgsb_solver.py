"""Joint, matrix-free L-BFGS-B for nonnegative CT factors.

The coordinate vector concatenates U and V. A single L-BFGS-B state and its
curvature memory persist throughout the run; neither block alternation nor
spectral constraints enter this solver. Every requested function/gradient pair,
including rejected line-search trials, consumes one full forward/adjoint pair.
"""

from copy import deepcopy

import numpy as np

try:
    from .nonnegative_solvers import COUNTERS, SCIPY_VERSION, NonnegativeSolver, projected_gradient_inf
except ImportError:
    from nonnegative_solvers import COUNTERS, SCIPY_VERSION, NonnegativeSolver, projected_gradient_inf


DEFAULTS = dict(memory=10, relative_kkt_tolerance=1e-3, ftol=1e-9, maxls=20)


def parameters(config=None):
    supplied = {} if config is None else dict(config)
    if set(supplied) - set(DEFAULTS):
        raise ValueError("Unknown joint L-BFGS-B parameter")
    result = DEFAULTS | supplied
    for key in ("memory", "maxls"):
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


class JointLBFGSB(NonnegativeSolver):
    """Version-pinned reverse communication with separate accepted/trial states."""

    method = "joint_lbfgsb"

    def __init__(self, problem, U, V, *, config=None):
        import scipy
        from scipy.optimize import _lbfgsb, _lbfgsb_py

        if scipy.__version__ != SCIPY_VERSION:
            raise RuntimeError("Exact joint L-BFGS-B restart requires SciPy " + SCIPY_VERSION)
        self._setulb = _lbfgsb.setulb
        self.integer_dtype = np.int64 if _lbfgsb_py.HAS_ILP64 else np.int32
        self.problem = problem
        self.config = parameters(config)
        self.train_ids = np.array(problem.train_ids, dtype=np.intp, copy=True)
        if self.train_ids.ndim != 1 or not len(self.train_ids):
            raise ValueError("Expected nonempty training view IDs")
        self.observed = np.asarray(problem.y)[self.train_ids]
        if self.observed.ndim != 2 or not np.all(np.isfinite(self.observed)):
            raise ValueError("Expected a finite views-by-detectors sinogram")
        self.U, self.V = (np.array(value, dtype=np.float64, copy=True) for value in (U, V))
        self._validate_factors()
        self.counters = dict.fromkeys(COUNTERS, 0)
        self.iteration = self.completed_blocks = self.outer_sweeps = self.block = 0
        self.block_termination_counts = {}
        self.status, self.phase, self.work = "running", "start", {}

    def _unpack(self, x):
        return x[:self.U.size].reshape(self.U.shape), x[self.U.size:].reshape(self.V.shape)

    def _start(self):
        x = np.concatenate((self.U.ravel(), self.V.ravel()))
        n, memory = x.size, self.config["memory"]
        self.work["lb"] = dict(x=x, l=np.zeros(n), u=np.zeros(n),
            nbd=np.ones(n, dtype=self.integer_dtype), f=np.array(0., dtype=np.float64),
            g=np.zeros(n), wa=np.zeros(2*memory*n + 5*n + 11*memory*memory + 8*memory),
            iwa=np.zeros(3*n, dtype=self.integer_dtype), task=np.zeros(2, dtype=self.integer_dtype),
            ln_task=np.zeros(2, dtype=self.integer_dtype), lsave=np.zeros(4, dtype=self.integer_dtype),
            isave=np.zeros(44, dtype=self.integer_dtype), dsave=np.zeros(29),
            pgtol=0., initial_kkt=None)
        self.phase = "setulb"

    def _details(self):
        lb = self.work["lb"]
        return dict(kkt_inf_norm=projected_gradient_inf(lb["x"], lb["g"]),
                    kkt_target=lb["pgtol"], initial_kkt=lb["initial_kkt"],
                    solver_loss=float(lb["f"]), native_task=lb["task"].tolist())

    def advance(self, view_budget=None):
        if self.status != "running":
            return self._event(self.status, **self.work["terminal_details"])
        if self.phase == "start":
            self._start()
        lb = self.work["lb"]
        if self.phase == "setulb":
            self._setulb(self.config["memory"], lb["x"], lb["l"], lb["u"],
                lb["nbd"], lb["f"], lb["g"], self.config["ftol"] / np.finfo(float).eps,
                lb["pgtol"], lb["wa"], lb["iwa"], lb["task"], lb["lsave"],
                lb["isave"], lb["dsave"], self.config["maxls"], lb["ln_task"])
            task = int(lb["task"][0])
            if task == 3:
                self.phase = "forward"
            elif task == 1:
                self.U, self.V = (np.array(x, copy=True) for x in self._unpack(lb["x"]))
                self._validate_factors()
                self.iteration += 1
                return self._event("accepted_update", updated_blocks="both", **self._details())
            elif task in (4, 8):
                code = int(lb["task"][1])
                if task == 8:
                    reason = "line_search_failed"
                    self.status = "stalled"
                elif code in (401, 402):
                    reason = "kkt" if code == 401 else "relative_function_decrease"
                    self.status = "converged"
                else:
                    raise FloatingPointError("Unknown joint L-BFGS-B convergence code " + repr(lb["task"].tolist()))
                self.work["terminal_details"] = dict(stop_reason=reason, block_stop_reason=reason,
                                                      **self._details())
                return self._event(self.status, **self.work["terminal_details"])
            else:
                raise FloatingPointError("Unexpected L-BFGS-B task " + repr(lb["task"].tolist()))
        if self.phase == "forward":
            if not self._affordable(view_budget, forward=1, backward=1):
                return self._event("budget_stop")
            trial_u, trial_v = self._unpack(lb["x"])
            # Match the CT projector's native residual arithmetic, accumulating
            # the squared objective in float64 as in the preceding baselines.
            self.work["residual"] = self._forward(trial_u @ trial_v.T) - self.observed
            self.phase = "backward"
            return self._event("operator_work", operation="forward")
        if self.phase == "backward":
            if not self._affordable(view_budget, backward=1):
                return self._event("budget_stop")
            residual = self.work.pop("residual")
            image_gradient = self._backward(residual)
            self._pair_finished()
            trial_u, trial_v = self._unpack(lb["x"])
            lb["f"][...] = 0.5 * np.mean(np.square(residual, dtype=np.float64))
            lb["g"][...] = np.concatenate(((image_gradient @ trial_v).ravel(),
                                           (image_gradient.T @ trial_u).ravel()))
            if not np.isfinite(lb["f"]) or not np.all(np.isfinite(lb["g"])):
                raise FloatingPointError("Nonfinite joint objective or gradient")
            if lb["initial_kkt"] is None:
                lb["initial_kkt"] = projected_gradient_inf(lb["x"], lb["g"])
                lb["pgtol"] = self.config["relative_kkt_tolerance"] * lb["initial_kkt"]
            self.phase = "setulb"
            return self._event("operator_work", operation="backward")
        raise ValueError("Unknown joint L-BFGS-B phase")

    def state_dict(self):
        return deepcopy(dict(schema_version=1, method=self.method, config=self.config,
            train_ids=self.train_ids, observed_shape=self.observed.shape, U=self.U, V=self.V,
            counters=self.counters, iteration=self.iteration, completed_blocks=0, outer_sweeps=0,
            block=0, block_termination_counts={}, status=self.status, phase=self.phase, work=self.work,
            scipy_version=SCIPY_VERSION, lbfgsb_integer_dtype=np.dtype(self.integer_dtype).str))

    def _validate_lbfgsb_state(self, state):
        lb = state["work"]["lb"]
        n, memory = self.U.size + self.V.size, self.config["memory"]
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
        if state["phase"] == "backward":
            residual = state["work"].get("residual")
            if (not isinstance(residual, np.ndarray) or residual.shape != self.observed.shape
                    or not np.all(np.isfinite(residual))):
                raise ValueError("Invalid pending residual")

    def load_state_dict(self, state):
        state = deepcopy(state)
        if (state.get("schema_version") != 1 or state.get("method") != self.method
                or state.get("config") != self.config
                or not np.array_equal(state.get("train_ids"), self.train_ids)
                or tuple(state.get("observed_shape", ())) != self.observed.shape
                or np.shape(state.get("U")) != self.U.shape or np.shape(state.get("V")) != self.V.shape):
            raise ValueError("Checkpoint configuration or problem dimensions differ")
        if state.get("scipy_version") != SCIPY_VERSION:
            raise ValueError("L-BFGS-B checkpoint requires the pinned SciPy version")
        if state.get("lbfgsb_integer_dtype") != np.dtype(self.integer_dtype).str:
            raise ValueError("L-BFGS-B checkpoint integer ABI differs")
        c = state["counters"]
        if (set(c) != set(COUNTERS) or any(isinstance(v, bool) or int(v) != v or v < 0 for v in c.values())
                or c["forward_views"] != len(self.train_ids)*c["forward_calls"]
                or c["backward_views"] != len(self.train_ids)*c["backward_calls"]
                or c["normal_operator_calls"] or c["rhs_backward_calls"] or c["rhs_backward_views"]
                or c["backward_calls"] != c["gradient_calls"]
                or c["forward_calls"] - c["gradient_calls"] != (state["phase"] == "backward")
                or c["oracle_calls"] != c["gradient_calls"]):
            raise ValueError("Invalid checkpoint work counters")
        if (isinstance(state["iteration"], bool) or int(state["iteration"]) != state["iteration"]
                or state["iteration"] < 0 or state["iteration"] > c["gradient_calls"]
                or any(state[key] != 0 for key in ("block", "completed_blocks", "outer_sweeps"))
                or state["block_termination_counts"] != {}
                or state["status"] not in ("running", "converged", "stalled")
                or state["phase"] not in ("start", "setulb", "forward", "backward")):
            raise ValueError("Invalid checkpoint iteration state")
        if state["phase"] != "start":
            self._validate_lbfgsb_state(state)
        if state["status"] != "running" and "terminal_details" not in state["work"]:
            raise ValueError("Missing terminal details")
        for key in ("U", "V", "counters", "iteration", "status", "phase", "work"):
            setattr(self, key, state[key])
        self._validate_factors()


def make_solver(method, problem, U, V, config=None):
    if method != "joint_lbfgsb":
        raise ValueError("Expected method 'joint_lbfgsb'")
    return JointLBFGSB(problem, U, V, config=config)
