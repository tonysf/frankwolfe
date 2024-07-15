import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from frank_wolfe.algorithms.base import FrankWolfe

class NoNoFrankWolfe(FrankWolfe):
    def __init__(self, objective_fn, lmo_fn, prox_fn, objective_type):
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

        for i in tqdm(range(n_steps), desc="NoNoFrank-Wolf Progress"):
            beta = beta0 / np.log(i + 2)
            step_size = 2.0 / np.sqrt((i + 2))

            grad = self.objective.gradient(self.x)
            Tx = self.objective.linear_operator(self.x)
            moreau_grad = self.objective.linear_operator_adjoint(Tx - self.prox(Tx, beta))/beta
            combined_grad = grad + moreau_grad
            
            # Note that svds uses random sampling so fix the seed to compare to other solvers etc
            np.random.seed(42)
            direction = self.lmo(combined_grad)
            self.num_oracles[i] += 1

            gap = np.sum(combined_grad * (self.x - direction))
            self.gaps[i] = gap

            func_val = self.objective.evaluate(self.x)
            self.func_vals[i] = func_val

            if self.objective_type == "indicator":
                # Implement the ns_gap calculation you want here
                # ns_gap = 0.5 * np.linalg.norm(beta * moreau_grad.flatten()) ** 2
                ns_gap = 0.5 * np.linalg.norm((self.x - self.prox(self.x, beta)).flatten()) ** 2
            elif self.objective_type == "lipschitz":
                ns_grad = self.objective.linear_operator_adjoint(self.objective.minimal_norm_selection(Tx))
                combined_ns_grad = grad + ns_grad
                ns_direction = self.lmo(combined_ns_grad)
                ns_gap = np.sum(combined_ns_grad * (self.x - ns_direction))
            else:
                raise ValueError(f"Unknown objective type: {self.objective_type}")

            self.ns_gaps[i] = ns_gap

            self.x = (1 - step_size) * self.x + step_size * direction
        self.num_oracles = np.cumsum(self.num_oracles)

    def plot_convergence(self):
        gaps = self.gaps
        n_steps = len(gaps)
        min_gaps = np.zeros(n_steps)
        avg_gaps = np.zeros(n_steps)
        min_gaps[0] = gaps[0]
        avg_gaps[0] = gaps[0]

        for i in range(1, n_steps):
            # Update min_gaps and avg_gaps using online update
            min_gaps[i] = min(gaps[i], min_gaps[i-1])
            avg_gaps[i] = avg_gaps[i-1] + (gaps[i] - avg_gaps[i-1]) / (i + 1)

        fig, axs = plt.subplots(1, 3, figsize=(20, 6))

        if self.objective_type == "indicator":
            middle_title = r"Squared distance to feasible region $\mathcal{D}$"
            middle_label = r"$\frac{1}{2}\mathrm{dist}_{\mathcal{D}}$"
            middle_data = self.ns_gaps
        elif self.objective_type == "lipschitz":
            middle_title = "Difference in gaps"
            middle_label = "gaps - ns_gaps"
            middle_data = self.gaps - self.ns_gaps
        else:
            ValueError(f"Unknown objective type: {self.objective_type}")

        # Plot functional values
        axs[0].plot(range(n_steps), self.func_vals, label=r'Functional values')
        axs[0].plot(range(0, n_steps), np.sqrt(n_steps + 1) * (1.1 * self.func_vals[n_steps//2]) / np.sqrt(np.array(range(1, n_steps + 1))), '--', label=r'$O\left(k^{-1/2}\right)$')
        axs[0].set_title(r'Functional values')
        axs[0].set_xlabel('Iterations')
        axs[0].set_ylabel('Functional Values')
        axs[0].legend()

        # Plot feasibility
        axs[1].plot(range(n_steps), middle_data, label=middle_label)
        axs[1].set_title(middle_title)
        axs[1].set_xlabel('Iterations')
        axs[1].set_ylabel(middle_label)
        axs[1].legend()

        # Plot gaps
        axs[2].semilogy(range(n_steps), self.gaps, label='Smoothed Gaps', linewidth=3)
        axs[2].semilogy(range(n_steps), min_gaps, label='Min (Smoothed Gaps)')
        axs[2].semilogy(range(n_steps), avg_gaps, label='Avg (Smoothed Gaps)')
        if self.objective_type == "lipschitz":
            axs[2].semilogy(range(n_steps), self.ns_gaps, label='Nonsmooth Gaps')
        axs[2].semilogy(range(0, n_steps), np.sqrt(n_steps + 1) * (1.1 * avg_gaps[n_steps//2]) / np.sqrt(np.array(range(1, n_steps + 1))), '--', label=r'$O\left(k^{-1/2}\right)$')
        axs[2].set_title(r'Smoothed gaps $\sup_{s\in \mathcal{C}}\ \left\langle -\nabla \phi_k(x_k), s-x_k\right\rangle$')
        axs[2].set_xlabel('Iterations')
        axs[2].set_ylabel('FW Gaps')
        axs[2].legend()

        # Adjust spacing and alignment
        plt.tight_layout()
        fig.subplots_adjust(top=0.88)  # Adjust the top spacing

        # Add a common title for the entire figure
        fig.suptitle('Algorithm Convergence Analysis', fontsize=16, fontweight='bold')

        plt.show()