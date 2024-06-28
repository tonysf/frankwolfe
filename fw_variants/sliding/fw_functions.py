from scipy.sparse.linalg import svds
from scipy.sparse.linalg import eigsh
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def soft_thresh(U, beta):
    return np.sign(U) * np.maximum(np.abs(U) - beta, 0)

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
        self.lipschitz = lipschitz
    
    def evaluate(self, x):
        raise NotImplementedError
    
    def gradient(self, x):
        raise NotImplementedError
    
class NonsmoothObjectiveFunction:
    def __init__(self):
        pass

    def evaluate(self, x):
        raise NotImplementedError
    
    def moreau_gradient(self, x):
        raise NotImplementedError

class SlidingFrankWolfe:
    def __init__(self, objective_fn, lmo_fn, diam):
        self.objective = objective_fn
        self.lmo = lmo_fn
        self.x = None
        self.func_vals = None
        self.gaps = None
        self.diam = diam

    def run(self, x0, n_steps = int(1e2)):
        self.x = x0
        self.func_vals = np.zeros(n_steps)
        self.gaps = np.zeros(n_steps)
        x = np.copy(self.x)
        y = np.copy(self.x)
        for i in tqdm(range(n_steps), desc="Conditional Gradient Sliding Progress"):
            # 3.0 in numerator according to the Lan+Zhou paper
            step_size = 3.0 / (i + 3)
            beta = self.objective.lipschitz * 3/(i + 2)

            z = (1 - step_size) * y  + step_size * x
            grad = self.objective.grad(z)

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
                inner_gap = np.sum(inner_grad.flatten() * (u-inner_direction).flatten())
                inner_step_size = inner_gap /(beta * (np.linalg.norm((inner_direction - u).flatten())**2))
                alpha = max(0, min(1, inner_step_size))
                u = (1 - alpha) * u + alpha * inner_direction
                k += 1
            x = u
            #
            #########################

            y = (1 - step_size) * y + step_size * x

            # Compute the number of lmo calls
            self.gaps[i] = self.gaps[i-1] + k

            func_val = self.objective.evaluate(y)
            self.func_vals[i] = func_val
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


class NonsmoothSlidingFrankWolfe:
    def __init__(self, objective_fn, lmo_fn, diam):
        self.objective = objective_fn
        self.lmo = lmo_fn
        self.x = None
        self.func_vals = None
        self.gaps = None
        self.diam = diam

    def run(self, x0, n_steps = int(1e2), lam0 = 1.0):
        self.x = x0
        self.func_vals = np.zeros(n_steps)
        self.gaps = np.zeros(n_steps)
        x = np.copy(self.x)
        y = np.copy(self.x)
        for i in tqdm(range(n_steps), desc="Conditional Gradient Sliding Progress"):
            # 3.0 in numerator according to the Lan+Zhou paper
            step_size = 3.0 / (i + 3)
            lam = lam0 / np.log(i + 2)
            beta = (1.0/lam) * 3/(i + 2)

            z = (1 - step_size) * y  + step_size * x
            grad = self.objective.moreau_grad(z, lam)

            inner_gap = np.inf
            # inner_tol is called eta_k in the paper
            inner_tol = (1.0/lam) * self.objective.lipschitz * (self.diam ** 2)/((i + 1) * (i + 2))
            
            #########################
            # CndG loop to update x
            u = x
            k=0
            while inner_gap > inner_tol:
                inner_grad = grad + beta * (u - x)
                inner_direction = self.lmo(inner_grad)
                inner_gap = np.sum(inner_grad.flatten() * (u-inner_direction).flatten())
                inner_step_size = inner_gap /(beta * (np.linalg.norm((inner_direction - u).flatten())**2))
                alpha = max(0, min(1, inner_step_size))
                u = (1 - alpha) * u + alpha * inner_direction
                k += 1
            x = u
            #
            #########################

            y = (1 - step_size) * y + step_size * x

            # Compute the number of lmo calls
            self.gaps[i] = self.gaps[i-1] + k

            func_val = self.objective.evaluate(y)
            self.func_vals[i] = func_val
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