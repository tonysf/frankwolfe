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
        d, Lambda, flag = np.zeros_like(x), 0, True
        G = grad + d
        align_d = align(-grad, d)

        num_oracles = 0
        while num_oracles < K:
            u = self.lmo(G) - x
            num_oracles += 1
            d_norm = np.linalg.norm(d)
            if d_norm > 0 and np.sum(G.flatten() * (-d / d_norm).flatten()) < np.sum(
                G.flatten() * u.flatten()
            ):
                u = -d / d_norm
                flag = False

            u_norm_sq = np.linalg.norm(u) ** 2
            if u_norm_sq <= 0:
                break

            lambda_k = -np.sum(G.flatten() * u.flatten()) / u_norm_sq
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
                    Lambda = Lambda * (1 - lambda_k / d_norm)
                G = grad + d
                align_d = align_d_new
                flag = True
            else:
                break

        if Lambda <= 0:
            fallback = self.lmo(grad) - x
            num_oracles += 1
            return fallback, num_oracles, align(-grad, fallback)

        return d / Lambda, num_oracles, align_d

    def run(self, x0, n_steps=int(1e2), K=float('inf'), delta=1e-3, step='Short'):
        self.x = self.lmo(self.objective.gradient(x0))
        self.func_vals = np.zeros(n_steps)
        self.oracle_calls = np.zeros(n_steps)
        self.gaps = np.zeros(n_steps)
        self.num_oracles = np.zeros(n_steps)
        step_name = step.lower().replace("_", "")
        
        for i in tqdm(range(n_steps), desc="Boosted Frank-Wolfe Progress"):
            grad = self.objective.gradient(self.x)
            
            # This gap is diagnostic and does not count toward oracle calls.
            v_fw = self.lmo(grad)
            self.gaps[i] = np.sum(grad.flatten() * (self.x - v_fw).flatten())
            self.func_vals[i] = self.objective.evaluate(self.x)
            
            g, num_oracles, align_g = self._nnmp(self.x, grad, K, delta)
            self.num_oracles[i] += num_oracles

            if step_name == "short":
                if self.objective.lipschitz is None:
                    raise ValueError("Short-step Boosted Frank-Wolfe requires objective.lipschitz.")
                denom = self.objective.lipschitz * np.linalg.norm(g)
                gamma = 0 if denom <= 0 else min(align_g * np.linalg.norm(grad) / denom, 1)
                self.x = self.x + gamma * g
            elif step_name in {"linesearch", "line"}:
                self.x, gamma = segment_search(self, self.x, self.x + g)
            else:
                raise ValueError("Invalid step type. Choose 'Short' or 'LineSearch'.")

        self.num_oracles = np.cumsum(self.num_oracles)
        self.oracle_calls = self.num_oracles.copy()

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
