import numpy as np
from tqdm import tqdm
from frank_wolfe.algorithms.base import FrankWolfe


class NoNoFrankWolfe(FrankWolfe):
    """
    NonSmooth Frank-Wolfe (NSFW) algorithm from Algorithm 1.

    Solves: min_{x in C} f(x) + g(Tx)

    where f is smooth, g is either an indicator (Assumption 1.4(I)) or
    Lipschitz weakly convex (Assumption 1.4(II)), and T is linear.

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
            Must implement evaluate, gradient, linear_operator, linear_operator_adjoint.
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

    def run(self, x0, beta0=1.0, n_steps=int(1e2)):
        self.x = self.lmo(self.objective.gradient(x0))
        self.func_vals = np.zeros(n_steps)
        self.gaps = np.zeros(n_steps)
        self.ns_gaps = np.zeros(n_steps)
        self.num_oracles = np.zeros(n_steps)

        for i in tqdm(range(n_steps), desc="NSFW Progress"):
            # Algorithm 1 schedules
            beta = beta0 / (i + 1) ** 0.25
            step_size = 1.0 / (i + 1) ** 0.5

            # Smoothed gradient: nabla f(x) + T^* (Tx - prox_{beta g}(Tx)) / beta
            grad = self.objective.gradient(self.x)
            Tx = self.objective.linear_operator(self.x)
            moreau_grad = self.objective.linear_operator_adjoint(
                Tx - self.prox(Tx, beta)
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
                ns_grad = self.objective.linear_operator_adjoint(
                    self.objective.minimal_norm_selection(Tx)
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
