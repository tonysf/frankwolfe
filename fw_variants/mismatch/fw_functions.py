from scipy.sparse.linalg import svds
from scipy.sparse.linalg import eigsh
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def general_lmo(gradient, radius, constraint_set):
    if constraint_set == "l1_ball":
        # Implement LMO for L1 ball
        # Handle both vector and matrix inputs
        index_flat = np.argmax(np.abs(gradient))
        index_multi = np.unravel_index(index_flat, gradient.shape)
        s = np.zeros_like(gradient)
        s[index_multi] = np.sign(gradient[index_multi])
        s = -radius * s
        return s
    elif constraint_set == "nuclear_norm_ball":
        # Implement LMO for nuclear norm ball
        # Assume gradient is a matrix
        u, _, vt = svds(gradient, k=1)
        s = -radius * np.outer(u, vt)
        return s
    elif constraint_set == "psd_trace":
        # Implement LMO for PSD matrices with bounded trace norm
        _, u = eigsh(gradient, k=1, which='LM')
        s = -radius * np.outer(u, u)
        return s
    elif constraint_set == "l2_ball":
        # Implement LMO for L2 ball
        # Handle both vector and matrix inputs
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm > 0:
            s = -radius * gradient / gradient_norm
        else:
            # Handle the case when the gradient is zero
            s = np.zeros_like(gradient)
        return s
    else:
        raise ValueError(f"Unsupported constraint set: {constraint_set}")

def create_lmo(radius, constraint_set):
    def lmo(gradient):
        return general_lmo(gradient, radius, constraint_set)
    return lmo

class ObjectiveFunction:
    def __init__(self):
        pass
    
    def evaluate(self, x):
        raise NotImplementedError
    
    def gradient(self, x):
        raise NotImplementedError

class MismatchFrankWolfe:
    def __init__(self, objective_fn, lmo_fn):
        self.objective = objective_fn
        self.lmo = lmo_fn
        self.x = None
        self.func_vals = None
        self.gaps = None

    def run(self, x0, n_steps = int(1e2), mismatch = True, averaging = False):
        self.x = x0
        self.func_vals = np.zeros(n_steps)
        self.gaps = np.zeros(n_steps)
        average_direction = np.zeros_like(x0)

        for i in tqdm(range(n_steps), desc="Frank-Wolfe Progress"):
            step_size = 2.0 / (i + 2)
            true_grad = self.objective.true_gradient(self.x)
            if mismatch:
                grad = self.objective.mismatch_gradient(self.x)
            elif mismatch == False:
                grad = true_grad
            
            direction = self.lmo(grad)

            gap = np.sum(true_grad * (self.x - self.lmo(true_grad)))
            self.gaps[i] = gap

            func_val = self.objective.evaluate(self.x)
            self.func_vals[i] = func_val

            if averaging:
                average_direction = average_direction + step_size * (direction - average_direction)
                direction = average_direction

            self.x = (1 - step_size) * self.x + step_size * direction

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