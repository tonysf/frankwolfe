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

class AwayFrankWolfe:
    def __init__(self, objective_fn, lmo_fn, L):
        self.objective = objective_fn
        self.lmo = lmo_fn
        self.x = None
        self.func_vals = None
        self.gaps = None
        self.active_set = None
        self.weights = None
        self.L = L

    def run(self, x0, n_steps=int(1e2), tol=1e-6, step_rule='LineSearch'):
        self.x = x0
        self.func_vals = np.zeros(n_steps)
        self.gaps = np.zeros(n_steps)
        
        # Initialize active set as a 2D numpy array
        self.active_set = np.array([x0.flatten()])
        self.weights = np.array([1.0])

        for i in tqdm(range(n_steps), desc="Away Frank-Wolfe Progress"):
            grad = self.objective.gradient(self.x)
            grad_flat = grad.flatten()

            # Forward step
            s = self.lmo(grad)
            d_fw = s - self.x
            gap_fw = np.dot(grad_flat, self.x.flatten() - s.flatten())

            # Away step
            if len(self.weights) > 1:
                # Vectorized computation of away vertex
                away_vertex_index = np.argmax(np.dot(self.active_set, grad_flat))
                away_vertex = self.active_set[away_vertex_index].reshape(self.x.shape)
                d_away = self.x - away_vertex
                gap_away = np.dot(grad_flat, away_vertex.flatten() - self.x.flatten())
            else:
                away_vertex_index = 0
                d_away = np.zeros_like(self.x)
                gap_away = 0

            # Choose step type
            if gap_fw >= gap_away:
                d = d_fw
                step_type = "FW"
                gamma_max = 1
            else:
                d = d_away
                step_type = "Away"
                gamma_max = self.weights[away_vertex_index] / (1 - self.weights[away_vertex_index])

            if step_rule == 'LineSearch':
                # Inline segment search
                left, right = self.x.copy(), self.x + gamma_max * d
                if np.dot(d, grad) * np.dot(d, self.objective.gradient(right)) >= 0:
                    if self.objective.evaluate(right) <= self.objective.evaluate(left):
                        x_min, gamma = right, gamma_max
                    else:
                        x_min, gamma = left, 0
                else:
                    gold = (1 + np.sqrt(5)) / 2
                    improv = np.inf
                    while improv > tol:
                        old_left, old_right = left.copy(), right.copy()
                        new = left + (right - left) / (1 + gold)
                        probe = new + (right - new) / 2
                        if self.objective.evaluate(probe) <= self.objective.evaluate(new):
                            left, right = new, right
                        else:
                            left, right = left, probe
                        improv = (np.linalg.norm(self.objective.evaluate(right) - self.objective.evaluate(old_right)) +
                                np.linalg.norm(self.objective.evaluate(left) - self.objective.evaluate(old_left)))
                    x_min = (left + right) / 2
                    for j in range(len(d)):
                        if d[j] != 0:
                            gamma = (x_min[j] - self.x[j]) / d[j]
                            break
                # Update x
                self.x = x_min
            else:
                # Implement the step size using Lipschitz constant
                gamma = min(gap_fw/(self.L * np.linalg.norm(d)**2), gamma_max)
                self.x = self.x + gamma * d

            if step_type == "FW":
                # Update weights
                self.weights *= (1 - gamma)
                
                # Check if s is in active set
                s_flat = s.flatten()
                s_in_active_set = np.any(np.all(self.active_set == s_flat, axis=1))
                
                if s_in_active_set:
                    idx = np.where(np.all(self.active_set == s_flat, axis=1))[0][0]
                    self.weights[idx] += gamma
                else:
                    self.active_set = np.vstack((self.active_set, s_flat))
                    self.weights = np.append(self.weights, gamma)
            else:  # Away step
                if gamma == gamma_max:
                    # Remove the atom from the active set and remove the weight
                    self.active_set = np.delete(self.active_set, away_vertex_index, axis=0)
                    self.weights = np.delete(self.weights, away_vertex_index)
                else:
                    # Active set stays the same, just update weights
                    self.weights *= (1 - gamma)
                    self.weights[away_vertex_index] += gamma

            # Record function value and gap
            self.func_vals[i] = self.objective.evaluate(self.x)
            self.gaps[i] = gap_fw

        # Reshape x to original shape if it was a matrix
        if len(self.x.shape) > 1:
            self.active_set = self.active_set.reshape(-1, *self.x.shape)



    def plot_convergence(self):
        gaps = self.gaps
        n_steps = len(gaps)

        fig, axs = plt.subplots(1, 2, figsize=(20, 6))

        # Plot functional values
        axs[0].semilogy(range(n_steps), self.func_vals - self.func_vals[-1], label='Functional values')
        axs[0].semilogy(range(0, n_steps), (n_steps + 1) * 1.1 * (self.func_vals[n_steps//2] - self.func_vals[-1]) / (np.array(range(1, n_steps + 1))), '--', label=r'$O\left(k^{-1}\right)$')
        axs[0].set_title('Functional values')
        axs[0].set_xlabel('Iterations')
        axs[0].set_ylabel('Functional Values')
        axs[0].legend()

        # Plot gaps
        axs[1].semilogy(range(n_steps), self.gaps, label='FW Gaps', linewidth=1)
        axs[1].semilogy(range(0, n_steps), (n_steps + 1) * (1.1 * gaps[n_steps//2]) / (np.array(range(1, n_steps + 1))), '--', label=r'$O\left(k^{-1}\right)$')
        axs[1].set_title(r'FW Gaps $\max\{g_{fw}, g_{away}\}$')
        axs[1].set_xlabel('Iterations')
        axs[1].set_ylabel('FW Gaps')
        axs[1].legend()

        # Adjust spacing and alignment
        plt.tight_layout()
        fig.subplots_adjust(top=0.88)

        # Add a common title for the entire figure
        fig.suptitle('Away Frank-Wolfe Convergence Analysis', fontsize=16, fontweight='bold')

        plt.show()
