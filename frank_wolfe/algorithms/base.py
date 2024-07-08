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

    def run(self, x0, n_steps=int(1e2)):
        raise NotImplementedError

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