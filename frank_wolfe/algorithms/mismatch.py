import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from frank_wolfe.algorithms.base import FrankWolfe
from frank_wolfe.core.utils import line_search

class MismatchFrankWolfe(FrankWolfe):
    def __init__(self, objective_fn, lmo_fn):
        super().__init__(objective_fn, lmo_fn)
        

    def run(self, x0, n_steps=int(1e2), mismatch=False, averaging=False):
        self.x = self.lmo(self.objective.gradient(x0))
        self.func_vals = np.zeros(n_steps)
        self.gaps = np.zeros(n_steps)
        self.num_oracles = np.zeros(n_steps)
        average_direction = np.zeros_like(x0)
        if mismatch:
            if averaging:
                for_string = '(Mismatch Frank-Wolfe with Averaging) Progress'
            else:
                for_string = '(Mismatch Frank-Wolfe) Progress'
        else:
            if averaging:
                for_string = '(Frank-Wolfe with Averaging) Progress'
            else:
                for_string = '(Frank-Wolfe) Progress'

        
        for i in tqdm(range(n_steps), desc=for_string):
            step_size = 2.0 / (i + 2)
            true_grad = self.objective.true_gradient(self.x)
            if mismatch:
                grad = self.objective.mismatch_gradient(self.x)
            elif mismatch == False:
                grad = true_grad
            
            direction = self.lmo(grad)
            self.num_oracles[i] += 1

            gap = np.sum(true_grad * (self.x - self.lmo(true_grad)))
            self.gaps[i] = gap

            func_val = self.objective.evaluate(self.x)
            self.func_vals[i] = func_val

            if averaging:
                average_direction = average_direction + step_size * (direction - average_direction)
                direction = average_direction

            self.x = (1 - step_size) * self.x + step_size * direction
        self.num_oracles = np.cumsum(self.num_oracles)


    def plot_convergence(self):
        gaps = self.gaps
        n_steps = len(gaps)

        fig, axs = plt.subplots(1, 2, figsize=(20, 6))

        # Plot functional values
        axs[0].semilogy(range(n_steps), self.func_vals, label=r'Functional values')
        axs[0].semilogy(range(0, n_steps), (n_steps + 1) * (1.1 * self.func_vals[n_steps//2]) / (np.array(range(1, n_steps + 1))), '--', label=r'$O\left(k^{-1}\right)$')
        axs[0].set_title(r'Functional values')
        axs[0].set_xlabel('Iterations')
        axs[0].set_ylabel('Functional Values')
        axs[0].legend()

        # Plot gaps
        axs[1].semilogy(range(n_steps), self.gaps, label='FW Gaps', linewidth=1)
        axs[1].semilogy(range(0, n_steps), (n_steps + 1) * (1.1 * gaps[n_steps//2]) / (np.array(range(1, n_steps + 1))), '--', label=r'$O\left(k^{-1}\right)$')
        axs[1].set_title(r'FW Gaps $\sup_{s\in \mathcal{C}}\ \left\langle -\nabla f(x_k), s-x_k\right\rangle$')
        axs[1].set_xlabel('Iterations')
        axs[1].set_ylabel('FW Gaps')
        axs[1].legend()

        # Adjust spacing and alignment
        plt.tight_layout()
        fig.subplots_adjust(top=0.88)  # Adjust the top spacing

        # Add a common title for the entire figure
        fig.suptitle('Algorithm Convergence Analysis', fontsize=16, fontweight='bold')

        plt.show()