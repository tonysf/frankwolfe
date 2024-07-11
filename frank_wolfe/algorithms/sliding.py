import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from frank_wolfe.algorithms.base import FrankWolfe
from frank_wolfe.core.utils import line_search

class CondGradSliding(FrankWolfe):
    def __init__(self, objective_fn, lmo_fn, diam):
        super().__init__(objective_fn, lmo_fn)
        self.diam = diam

    def run(self, x0, n_steps=int(1e2)):
        self.x = x0
        self.func_vals = np.zeros(n_steps)
        self.gaps = np.zeros(n_steps)
        self.num_oracles = np.zeros(n_steps)
        x = np.copy(self.x)
        y = np.copy(self.x)
        for i in tqdm(range(n_steps), desc="Conditional Gradient Sliding Progress"):
            # 3.0 in numerator according to the Lan+Zhou paper
            step_size = 3.0 / (i + 3)
            beta = self.objective.lipschitz * 3/(i + 2)

            z = (1 - step_size) * y  + step_size * x
            grad = self.objective.gradient(z)

            inner_gap = np.inf
            # inner_tol is called eta_k in the paper
            inner_tol = self.objective.lipschitz * (self.diam ** 2)/((i + 1) * (i + 2))
            
            #########################
            # CndG loop to update x
            u = x
            k=0
            while inner_gap > inner_tol:
                inner_grad = grad + beta * (u - x)
                inner_direction = self.lmo(inner_grad)
                self.num_oracles[i] += 1
                inner_gap = np.sum(inner_grad.flatten() * (u-inner_direction).flatten())
                inner_step_size = inner_gap /(beta * (np.linalg.norm((inner_direction - u).flatten())**2))
                alpha = max(0, min(1, inner_step_size))
                u = (1 - alpha) * u + alpha * inner_direction
                k += 1
            x = u
            #
            #########################

            y = (1 - step_size) * y + step_size * x
            y_grad = self.objective.gradient(y)
            gap = np.sum(y_grad.flatten() *  (y - self.lmo(y_grad)).flatten())
            self.gaps[i] = gap

            func_val = self.objective.evaluate(y)
            self.func_vals[i] = func_val
        self.num_oracles = np.cumsum(self.num_oracles)
        self.x = y


    def plot_convergence(self):
        gaps = self.gaps
        n_steps = len(gaps)

        fig, axs = plt.subplots(1, 2, figsize=(20, 6))

        # Plot functional values as a function of iterations
        axs[0].semilogy(range(n_steps), self.func_vals - self.func_vals[-1], label=r'Functional values')
        axs[0].semilogy(range(0, n_steps), 15 * self.objective.lipschitz * (self.diam ** 2) / (2 * (np.array(range(1, n_steps + 1)) + 1) * (np.array(range(1, n_steps + 1)) + 2)), '--', label=r'$\frac{15 L \mathrm{diam}_{\mathcal{C}}^2}{k(k+1)}$')
        axs[0].set_title(r'Functional gap: $f(y_k) - \min_{x\in\mathcal{C}} f(x)$ vs gradient calls (iterations)')
        axs[0].set_xlabel('Iterations')
        axs[0].set_ylabel('Functional gap')
        axs[0].legend()

        # Plot functional values as a function of LMO calls
        axs[1].semilogy(self.gaps, self.func_vals - self.func_vals[-1], label='Functional values', linewidth=1)
        axs[1].semilogy(range(0, n_steps), 15 * np.sqrt(self.objective.lipschitz) * self.diam / (2 * (np.array(range(1, n_steps + 1)) + 1)), '--', label=r'$\frac{15 \sqrt{L} \mathrm{diam}_{\mathcal{C}}}{k+1}$')
        axs[1].set_title(r'Functional gap: $f(y_k) - \min_{x\in\mathcal{C}} f(x)$ vs LMO calls')
        axs[1].set_xlabel('LMO Calls')
        axs[1].set_title(r'Functional gap: $f(y_k) - \min_{x\in\mathcal{C}} f(x)$')
        axs[1].legend()

        # Adjust spacing and alignment
        plt.tight_layout()
        fig.subplots_adjust(top=0.88)  # Adjust the top spacing

        # Add a common title for the entire figure
        fig.suptitle('Conditional Gradient Sliding Algorithm', fontsize=16, fontweight='bold')

        plt.show()