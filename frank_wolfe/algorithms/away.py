import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from frank_wolfe.algorithms.base import FrankWolfe
from frank_wolfe.core.utils import segment_search

class AwayFrankWolfe(FrankWolfe):
    def __init__(self, objective_fn, lmo_fn):
        super().__init__(objective_fn, lmo_fn)
        self.active_set = None
        self.weights = None

    def run(self, x0, n_steps=int(1e2), tol=1e-6, step='LineSearch'):
        self.x = self.lmo(self.objective.gradient(x0))
        self.func_vals = np.zeros(n_steps)
        self.gaps = np.zeros(n_steps)
        self.num_oracles = np.zeros(n_steps)
        # Initialize active set as a 2D numpy array
        self.active_set = np.array([x0.flatten()])
        self.weights = np.array([1.0])

        for i in tqdm(range(n_steps), desc="Away Frank-Wolfe Progress"):
            grad = self.objective.gradient(self.x)
            grad_flat = grad.flatten()

            # Forward step
            s = self.lmo(grad)
            self.num_oracles[i] += 1
            d_fw = s - self.x
            gap_fw = np.dot(grad_flat, self.x.flatten() - s.flatten())
            # Record function value and gap
            self.func_vals[i] = self.objective.evaluate(self.x)
            self.gaps[i] = gap_fw
            # Away step
            if len(self.weights) > 1:
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

            if step == 'LineSearch':
                _, gamma = segment_search(self, self.x, self.x + d, tol=tol)
            elif step == 'Short':
                # Implement the step size using Lipschitz constant
                gamma = min(gap_fw / (self.objective.lipschitz * np.linalg.norm(d)**2), gamma_max)
            else:
                break

            gamma = min(gamma, gamma_max)
            self.x = self.x + gamma * d

            self._update_active_set(step_type, s, away_vertex_index, gamma)

        self.num_oracles = np.cumsum(self.num_oracles)

    def _update_active_set(self, step_type, s, away_vertex_index, gamma):
        if step_type == "FW":
            self.weights *= (1 - gamma)
            s_flat = s.flatten()
            s_in_active_set = np.any(np.all(self.active_set == s_flat, axis=1))
            
            if s_in_active_set:
                idx = np.where(np.all(self.active_set == s_flat, axis=1))[0][0]
                self.weights[idx] += gamma
            else:
                self.active_set = np.vstack((self.active_set, s_flat))
                self.weights = np.append(self.weights, gamma)
        else:  # Away step
            if gamma == self.weights[away_vertex_index] / (1 - self.weights[away_vertex_index]):
                # Remove the atom from the active set and remove the weight
                self.active_set = np.delete(self.active_set, away_vertex_index, axis=0)
                self.weights = np.delete(self.weights, away_vertex_index)
            else:
                # Active set stays the same, just update weights
                self.weights *= (1 - gamma)
                self.weights[away_vertex_index] += gamma

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
        axs[1].set_title(r'FW Gap $g_{fw}$')
        axs[1].set_xlabel('Iterations')
        axs[1].set_ylabel('FW Gaps')
        axs[1].legend()

        plt.tight_layout()
        fig.subplots_adjust(top=0.88)
        fig.suptitle('Away Frank-Wolfe Convergence Analysis', fontsize=16, fontweight='bold')
        plt.show()