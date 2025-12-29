import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from frank_wolfe.algorithms.base import FrankWolfe
from frank_wolfe.core.utils import segment_search, align

class BoostedFrankWolfe(FrankWolfe):
    def __init__(self, objective_fn, lmo_fn, diam):
        super().__init__(objective_fn, lmo_fn)
        self.diam = diam

    def _nnmp(self, x, grad, K, delta):
        d, Lambda, flag = np.zeros(len(x)), 0 , True
        G = grad + d
        align_d = align(-grad, d)
        for k in range(K):
            u = self.lmo(G) - x
            d_norm = np.linalg.norm(d)
            if d_norm > 0 and np.dot(G, -d/d_norm) < np.dot(G, u):
                u = -d/d_norm
                flag = False
            lambda_k = -np.dot(G, u)/np.linalg.norm(u)**2
            # If d == 0, then the above is just
            # gap(x_k)/(||s_k-x_k||^2) and so d_new is
            # just a scaled version of s_k-x_k
            d_new = d + lambda_k * u
            align_d_new = align(-grad, d_new)
            align_improve = align_d_new - align_d
            if align_improve > delta:
                d = d_new
                if flag:
                    Lambda = Lambda + lambda_k
                else:
                    Lambda = Lambda * (1-lambda_k/d_norm)
                G = grad + d
                align_d = align_d_new
                flag = True
            else:
                break
        return d/Lambda, k + 1, align_d

    def run(self, x0, n_steps=int(1e2), K=float('inf'), delta=1e-3, step='short'):
        self.x = self.lmo(self.objective.gradient(x0))
        # self.x = x0
        self.func_vals = np.zeros(n_steps)
        self.oracle_calls = np.zeros(n_steps)
        self.gaps = np.zeros(n_steps)
        self.num_oracles = np.zeros(n_steps)
        
        for i in tqdm(range(n_steps), desc="Boosted Frank-Wolfe Progress"):
            grad = self.objective.gradient(self.x)
            
            # Compute Frank-Wolfe gap (this lmo doesn't count towards num_oracles; it's not actually used in the algorithm, just insightful for us)
            v_fw = self.lmo(grad)
            self.gaps[i] = np.sum(grad.flatten() * (self.x - v_fw).flatten())
            # Record function value
            self.func_vals[i] = self.objective.evaluate(self.x)
            
            # Nonnegative Matching Pursuit
            g, num_oracles, align_g = self._nnmp(self.x, grad, K, delta)
            # Record number of oracle calls
            self.num_oracles[i] += num_oracles

            # Step size calculation
            if step == 'short':
                gamma = min(align_g*np.linalg.norm(grad) / (self.objective.lipschitz * np.linalg.norm(g)), 1)
                self.x = self.x + gamma * g
            elif step == 'linesearch':
                self.x, gamma = segment_search(self.objective, self.x, self.x + g)
            else:
                raise ValueError("Invalid step type. Choose 'short' or 'linesearch'.")
        self.num_oracles = np.cumsum(self.num_oracles)

    def plot_convergence(self):
        n_steps = len(self.func_vals)

        fig, axs = plt.subplots(2, 2, figsize=(12, 9))

        # Plot functional values as a function of iterations
        axs[0, 0].semilogy(range(n_steps), self.func_vals - self.func_vals[-1], label='Functional values')
        axs[0, 0].semilogy(range(n_steps), 0.5 * self.objective.lipschitz * (self.diam ** 2) / (np.array(range(1, n_steps + 1)) + 2), '--', label='Theoretical rate')
        axs[0, 0].set_title('Functional gap vs iterations')
        axs[0, 0].set_xlabel('Iterations')
        axs[0, 0].set_ylabel('Functional gap')
        axs[0, 0].legend()

        # Plot functional values as a function of LMO calls
        axs[0, 1].semilogy(self.oracle_calls, self.func_vals - self.func_vals[-1], label='Functional values')
        axs[0, 1].set_title('Functional gap vs LMO calls')
        axs[0, 1].set_xlabel('LMO Calls')
        axs[0, 1].set_ylabel('Functional gap')
        axs[0, 1].legend()

        # Plot Frank-Wolfe gaps as a function of iterations
        axs[1, 0].semilogy(range(n_steps), self.gaps, label='FW gaps')
        axs[1, 0].set_title('Frank-Wolfe gap vs iterations')
        axs[1, 0].set_xlabel('Iterations')
        axs[1, 0].set_ylabel('FW gap')
        axs[1, 0].legend()

        # Plot Frank-Wolfe gaps as a function of LMO calls
        axs[1, 1].semilogy(self.oracle_calls, self.gaps, label='FW gaps')
        axs[1, 1].set_title('Frank-Wolfe gap vs LMO calls')
        axs[1, 1].set_xlabel('LMO Calls')
        axs[1, 1].set_ylabel('FW gap')
        axs[1, 1].legend()

        plt.tight_layout()
        fig.suptitle('Boosted Frank-Wolfe Algorithm', fontsize=16, fontweight='bold')
        plt.show()
