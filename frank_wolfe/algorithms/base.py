import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class FrankWolfe:
    def __init__(self, objective_fn, lmo_fn):
        self.objective = objective_fn
        self.lmo = lmo_fn
        self.x = None
        self.func_vals = None
        self.gaps = None
        self.num_oracles = None

    def run(self, x0, n_steps=int(1e2)):
        self.x = self.lmo(self.objective.gradient(x0))
        self.func_vals = np.zeros(n_steps)
        self.gaps = np.zeros(n_steps)
        self.num_oracles = np.zeros(n_steps)
        for i in tqdm(range(n_steps), desc="Frank-Wolfe Progress"):
            step_size = 2.0 / (i + 2)
            grad = self.objective.gradient(self.x)
            direction = self.lmo(grad)
            self.num_oracles[i] += 1
            gap = np.sum(grad * (self.x - direction))
            self.gaps[i] = gap
            func_val = self.objective.evaluate(self.x)
            self.func_vals[i] = func_val
            self.x = (1 - step_size)*self.x + step_size*direction
        self.num_oracles = np.cumsum(self.num_oracles)

    def plot_convergence(self):
        n_steps = len(self.gaps)
        fig, axs = plt.subplots(1, 2, figsize=(20, 6))

        axs[0].semilogy(range(n_steps), self.func_vals - np.min(self.func_vals), label='Functional values')
        axs[0].set_title('Functional value gap')
        axs[0].set_xlabel('Iterations')
        axs[0].set_ylabel('Functional gap')
        axs[0].legend()

        axs[1].semilogy(range(n_steps), self.gaps, label='FW Gaps', linewidth=1)
        axs[1].set_title('FW Gaps')
        axs[1].set_xlabel('Iterations')
        axs[1].set_ylabel('FW Gaps')
        axs[1].legend()

        plt.tight_layout()
        plt.show()