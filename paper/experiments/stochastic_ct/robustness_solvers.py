"""Budget-guarded solvers for the CT mechanism and robustness study.

The historical SFRAMES, projections, and practical curvature code are vendored
with import-only changes.  Deterministic FraMES uses exact current gradients
and exponents (2/3, 1/3), as Proposition 2.12 of the manuscript prescribes.
The no-averaging ablation retains the stochastic exponents (3/4, 1/4).

``view_budget`` caps gradient work separately in each direction.  Curvature
applications remain in total counters and are separately identified.  A whole
native update is checked before any operator or random draw is consumed.
"""

from copy import deepcopy

import numpy as np

try:
    from .robustness_reference import methods as reference
    from .robustness_reference.local_curvature import SpringSARAHLocalCurvature
    from .robustness_reference.palm_paper import PALMPaper
    from .joint_lbfgsb_solver import JointLBFGSB
except ImportError:
    from robustness_reference import methods as reference
    from robustness_reference.local_curvature import SpringSARAHLocalCurvature
    from robustness_reference.palm_paper import PALMPaper
    from joint_lbfgsb_solver import JointLBFGSB


class NumericalFailure(FloatingPointError):
    """Nonfinite arithmetic from a numerical candidate, never a projection certificate."""


METHODS = ('sframes', 'sframes_noavg', 'frames', 'prox_sarah', 'spring_sarah',
           'palm', 'joint_lbfgsb')
SMOOTHING = ('sframes', 'sframes_noavg', 'frames')
COUNTERS = ('forward_views', 'backward_views', 'forward_calls', 'backward_calls',
    'gradient_calls', 'normal_operator_calls', 'rhs_backward_calls', 'rhs_backward_views',
    'oracle_calls', 'gradient_forward_views', 'gradient_backward_views',
    'curvature_forward_views', 'curvature_backward_views', 'curvature_operator_calls',
    'projection_calls', 'projection_inner_iterations', 'projection_dykstra_calls',
    'projection_floor_count', 'lmo_calls')


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(name + ' must be finite and positive')
    return value


def parameters(method, n_train, curvatures, config=None):
    """Resolve declared parameters without changing any published update rule."""
    if method not in METHODS:
        raise ValueError('Unknown robustness solver: ' + str(method))
    supplied = dict(config or {})
    if method == 'joint_lbfgsb':
        try:
            from .joint_lbfgsb_solver import parameters as joint_parameters
        except ImportError:
            from joint_lbfgsb_solver import parameters as joint_parameters
        return joint_parameters(supplied)
    curvature = np.asarray(curvatures, dtype=float)
    if curvature.shape != (2,) or not np.isfinite(curvature).all() or np.any(curvature <= 0):
        raise ValueError('Expected two positive reference curvature scales')
    if method in SMOOTHING:
        allowed = {'batch', 'gamma0', 'beta_scale', 'rho0'}
        if supplied.keys() - allowed:
            raise ValueError('Unknown smoothing parameter')
        result = dict(batch='full' if method == 'frames' else 64, gamma0=0.1,
                      beta_scale=4., rho0=1.) | supplied
        batch = n_train if result['batch'] == 'full' else result['batch']
        if isinstance(batch, bool) or int(batch) != batch or not 1 <= batch <= n_train:
            raise ValueError('Batch must be full or an integer in [1,N]')
        result['batch'] = int(batch)
        for name in ('gamma0', 'rho0', 'beta_scale'):
            result[name] = _positive(result[name], name)
        if result['gamma0'] > 1 or result['rho0'] > 1:
            raise ValueError('Averaging and FW step prefactors must be at most one')
        if method == 'frames' and result['batch'] != n_train:
            raise ValueError('Deterministic FraMES requires the full gradient')
        if method != 'sframes' and result['rho0'] != 1:
            raise ValueError('The no-averaging variants do not tune rho0')
        result['mu0'] = result['beta_scale'] / float(curvature.max())
        return result
    if method == 'prox_sarah':
        if supplied.keys() - {'curvature_scale'}:
            raise ValueError('Only the ProxSARAH curvature scale is tuned')
        scale = _positive(supplied.get('curvature_scale', 0.25), 'curvature_scale')
        relaxation = 0.99
        constant = 2. / (3. * relaxation**2)
        return dict(curvature_scale=scale, batch=max(1, int(np.sqrt(n_train) / constant)),
            snapshot_batch=n_train, inner_steps=max(1, int(np.sqrt(n_train))),
            eta=(2. / (4. + relaxation)) / (scale * float(curvature.sum())),
            relaxation=relaxation)
    if method == 'spring_sarah':
        if supplied.keys() - {'batch', 'refresh_period'}:
            raise ValueError('SPRING retains the published practical step rule')
        result = dict(batch=min(64, n_train), refresh_period=16.) | supplied
        batch = result['batch']
        if isinstance(batch, bool) or int(batch) != batch or not 1 <= batch <= n_train:
            raise ValueError('Invalid SPRING batch')
        result['batch'] = int(batch)
        result['refresh_period'] = _positive(result['refresh_period'], 'refresh_period')
        if result['refresh_period'] < 1:
            raise ValueError('SPRING refresh period must be at least one')
        return result | dict(power_iterations=5, step_coefficient=0.5)
    if supplied:
        raise ValueError('PALM has no tunable step multiplier in this study')
    return dict(power_iterations=5, step_coefficient=1., curvature_batch_size=n_train)


class NoAverageSFrames(reference.SFrames):
    gamma_exponent, beta_exponent = 0.75, 0.25

    def _update(self):
        # Keep the inherited sampler and LMO exactly matched to SFRAMES.  The
        # current smooth gradient replaces its average before the exact penalty.
        self.mU, self.mV = self.grad(self.U, self.V, self.sample(self.batch))
        t = self.k + 1
        gamma = self.gamma0 / t**self.gamma_exponent
        beta = self.mu0 / t**self.beta_exponent
        atom_u, cert_u = reference.spectral_lmo(self.mU + np.minimum(self.U, 0) / beta,
                                               self.ru, return_certificate=True)
        atom_v, cert_v = reference.spectral_lmo(self.mV + np.minimum(self.V, 0) / beta,
                                               self.rv, return_certificate=True)
        self.U = self.U + gamma * (atom_u - self.U)
        self.V = self.V + gamma * (atom_v - self.V)
        return dict(gamma=gamma, rho=1., mu=beta,
                    lmo_support_gap_bound=cert_u['support_gap_bound'] + cert_v['support_gap_bound'])


class DeterministicFrames(NoAverageSFrames):
    gamma_exponent, beta_exponent = 2. / 3., 1. / 3.


class Solver:
    """Uniform accepted-iterate interface around the validated native solvers."""

    def __init__(self, method, problem, U, V, caps, *, seed, curvatures, config=None):
        self.method, self.problem = method, problem
        self.caps = tuple(float(x) for x in caps)
        self.seed = int(seed)
        self.config = parameters(method, len(problem.train_ids), curvatures, config)
        self.counters = dict.fromkeys(COUNTERS, 0)
        self.last_event = None
        if method == 'joint_lbfgsb':
            self.native = JointLBFGSB(problem, U, V, config=self.config)
        else:
            classes = dict(sframes=reference.SFrames, sframes_noavg=NoAverageSFrames,
                frames=DeterministicFrames, prox_sarah=reference.ProxSARAH,
                spring_sarah=SpringSARAHLocalCurvature, palm=PALMPaper)
            native_config = dict(self.config)
            native_config.pop('beta_scale', None)
            native_config.pop('curvature_scale', None)
            self.native = classes[method](problem, U, V, self.caps,
                seed=100_000 + self.seed, config=native_config)

    @property
    def U(self):
        return self.native.U

    @property
    def V(self):
        return self.native.V

    @property
    def iteration(self):
        return self.native.iteration if self.method == 'joint_lbfgsb' else self.native.k

    @property
    def completed_blocks(self):
        return self.native.completed_blocks if self.method == 'joint_lbfgsb' else 0

    @property
    def outer_sweeps(self):
        return self.native.outer_sweeps if self.method == 'joint_lbfgsb' else 0

    @property
    def spectral_constrained(self):
        return self.method != 'joint_lbfgsb'

    def beta(self):
        if self.method not in SMOOTHING:
            return None
        exponent = 1. / 3. if self.method == 'frames' else 0.25
        # At accepted state x_k, the next update uses beta_k (index starts at 0).
        return self.config['mu0'] / (self.iteration + 1)**exponent

    def _event(self, kind, **details):
        return dict(observation_type=kind, accepted=kind == 'accepted_update',
                    iteration=self.iteration, **self.counters, **details)

    def next_gradient_views(self):
        """Peek at the next native update without advancing either RNG stream."""
        n, solver = len(self.problem.train_ids), self.native
        if self.method in SMOOTHING:
            return self.config['batch']
        if self.method == 'prox_sarah':
            return n if solver.k % (solver.inner_steps + 1) == 0 else 2 * solver.batch
        if self.method == 'palm':
            return 2 * n
        if self.method == 'spring_sarah':
            if solver.previous_u is None:
                return 2 * n
            rng = np.random.default_rng()
            rng.bit_generator.state = deepcopy(solver.rng.bit_generator.state)
            cost = 0
            for _ in range(2):
                refresh = rng.random() < solver.refresh_probability
                cost += n if refresh else 2 * solver.batch
                if not refresh and solver.batch != n:
                    rng.choice(solver.train_ids, solver.batch, replace=False)
            return cost
        return n

    def _sync_joint_counters(self):
        for key, value in self.native.counters.items():
            self.counters[key] = int(value)
        self.counters['gradient_forward_views'] = self.counters['forward_views']
        self.counters['gradient_backward_views'] = self.counters['backward_views']

    def advance(self, view_budget=None):
        try:
            return self._advance(view_budget)
        except FloatingPointError as error:
            if any(word in str(error).lower() for word in ('nonfinite', 'overflow', 'invalid value')):
                raise NumericalFailure(str(error)) from error
            raise
        except ValueError as error:
            if 'infs or nans' in str(error).lower():
                raise NumericalFailure(str(error)) from error
            raise

    def _advance(self, view_budget=None):
        if view_budget is not None and (isinstance(view_budget, bool)
                or int(view_budget) != view_budget or view_budget < 0):
            raise ValueError('view_budget must be a nonnegative integer')
        if self.method == 'joint_lbfgsb':
            event = self.native.advance(view_budget=view_budget)
            self._sync_joint_counters()
            self.last_event = dict(event) | self.counters
            return self.last_event
        cost = self.next_gradient_views()
        if view_budget is not None and any(self.counters[k] + cost > view_budget
                for k in ('gradient_forward_views', 'gradient_backward_views')):
            self.last_event = self._event('budget_stop')
            return self.last_event
        details = self.native.step()
        curvature_calls = int(details.get('curvature_operator_calls', 0))
        gradients = int(details['gradient_calls'])
        for direction in ('forward', 'backward'):
            total = int(details[direction + '_views'])
            curvature = int(details.get('curvature_' + direction + '_views', 0))
            if total - curvature != cost:
                raise ValueError('Native update work differs from preflight cost')
            self.counters[direction + '_views'] += total
            self.counters['curvature_' + direction + '_views'] += curvature
            self.counters['gradient_' + direction + '_views'] += total - curvature
            self.counters[direction + '_calls'] += gradients + curvature_calls
        self.counters['gradient_calls'] += gradients
        self.counters['curvature_operator_calls'] += curvature_calls
        self.counters['oracle_calls'] += gradients + curvature_calls
        for key in ('projection_calls', 'projection_inner_iterations',
                    'projection_dykstra_calls', 'projection_floor_count'):
            self.counters[key] += int(details.get(key, 0))
        if self.method in SMOOTHING:
            self.counters['lmo_calls'] += 2
        self.last_event = self._event('accepted_update', native_step=details)
        return self.last_event

    def step(self, view_budget=None):
        while True:
            event = self.advance(view_budget)
            if event['observation_type'] != 'operator_work':
                return event

    def state_dict(self):
        return deepcopy(dict(schema_version=1, method=self.method, caps=self.caps,
            config=self.config, seed=self.seed, counters=self.counters,
            native=self.native.state_dict(), last_event=self.last_event))

    def load_state_dict(self, state):
        if (state.get('schema_version') != 1 or state.get('method') != self.method
                or tuple(state.get('caps', ())) != self.caps or state.get('config') != self.config
                or state.get('seed') != self.seed):
            raise ValueError('Robustness solver checkpoint does not match configuration')
        counts = state.get('counters', {})
        if (set(counts) != set(COUNTERS) or any(isinstance(v, bool) or int(v) != v or v < 0
                                              for v in counts.values())):
            raise ValueError('Invalid work counters')
        for direction in ('forward', 'backward'):
            if counts[direction + '_views'] != (counts['gradient_' + direction + '_views']
                                               + counts['curvature_' + direction + '_views']):
                raise ValueError('Gradient and curvature work do not sum to total work')
        self.native.load_state_dict(state['native'])
        self.counters = deepcopy(counts)
        self.last_event = deepcopy(state.get('last_event'))
        if self.method == 'joint_lbfgsb':
            before = dict(self.counters)
            self._sync_joint_counters()
            if self.counters != before:
                raise ValueError('Joint L-BFGS-B counters differ from native state')


def make_solver(method, problem, U, V, caps, *, seed, curvatures, config=None):
    return Solver(method, problem, U, V, caps, seed=seed, curvatures=curvatures, config=config)
