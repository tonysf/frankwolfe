import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from frank_wolfe.algorithms.base import FrankWolfe
from frank_wolfe.core.utils import segment_search, align

class BoostedFrankWolfe(FrankWolfe):
    def __init__(self, objective_fn, lmo_fn, diam):
        super().__init__(objective_fn, lmo_fn)
        self.diam = diam
        self.oracle_calls = None
        self.fw_gaps = None

    def run(self, x0, n_steps=int(1e2), K=float('inf'), delta=1e-3, step_size_strategy='short'):
        self.x = x0
        self.func_vals = np.zeros(n_steps)
        self.oracle_calls = np.zeros(n_steps)
        self.fw_gaps = np.zeros(n_steps)
        x = np.copy(self.x)
        
        for t in tqdm(range(n_steps), desc="Boosted Frank-Wolfe Progress"):
            grad = self.objective.gradient(x)
            
            # Compute Frank-Wolfe gap
            v_fw = self.lmo(grad)
            self.fw_gaps[t] = np.sum(grad.flatten() * (x - v_fw).flatten())
            
            # Gradient pursuit procedure (NNMP)
            d = np.zeros_like(grad)
            Lambda = 0
            k = 0
            
            while k < K:
                r = -(grad + d)
                v = self.lmo(-r)
                v_minus_x = v - x
                if np.allclose(d, np.zeros_like(d)):
                    if np.sum(r.flatten() * v_minus_x.flatten()) > 0:
                        u = v_minus_x
                        v_minus_x_flag = True
                    else:
                        u = 0
                        v_minus_x_flag = False
                elif np.sum(r.flatten() * v_minus_x.flatten()) > np.sum(r.flatten() * -d.flatten() / np.linalg.norm(d)):
                    u = v_minus_x
                    v_minus_x_flag = True
                else:
                    u = -d / np.linalg.norm(d)
                    v_minus_x_flag = False
                
                lambda_k = np.sum(r.flatten() * u.flatten()) / (np.linalg.norm(u.flatten()) ** 2)
                d_new = d + lambda_k * u
                
                if align(-grad, d_new) - align(-grad, d) > delta:
                    d = d_new
                    if v_minus_x_flag:
                        Lambda += lambda_k
                    else:
                        Lambda *= (1 - lambda_k / np.linalg.norm(d))
                    k += 1
                else:
                    break
            
            g = d / Lambda
            
            # Step size calculation
            if step_size_strategy == 'Short':
                eta = align(-grad, g)
                gamma = min(eta * np.linalg.norm(grad) / (self.objective.lipschitz * np.linalg.norm(g)), 1)
                x = x + gamma * g
            elif step_size_strategy == 'LineSearch':
                x, _ = segment_search(self, x, x + g)
            else:
                raise ValueError("Invalid step_size_strategy. Choose 'Short' or 'LineSearch'.")
            
            # Record function value and oracle calls
            self.func_vals[t] = self.objective.evaluate(x)
            if t > 0:
                self.oracle_calls[t] = self.oracle_calls[t-1] + k + 1 
            else:
                self.oracle_calls[t] = k + 1
        
        self.x = x

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
        axs[1, 0].semilogy(range(n_steps), self.fw_gaps, label='FW gaps')
        axs[1, 0].set_title('Frank-Wolfe gap vs iterations')
        axs[1, 0].set_xlabel('Iterations')
        axs[1, 0].set_ylabel('FW gap')
        axs[1, 0].legend()

        # Plot Frank-Wolfe gaps as a function of LMO calls
        axs[1, 1].semilogy(self.oracle_calls, self.fw_gaps, label='FW gaps')
        axs[1, 1].set_title('Frank-Wolfe gap vs LMO calls')
        axs[1, 1].set_xlabel('LMO Calls')
        axs[1, 1].set_ylabel('FW gap')
        axs[1, 1].legend()

        plt.tight_layout()
        fig.suptitle('Boosted Frank-Wolfe Algorithm', fontsize=16, fontweight='bold')
        plt.show()

class oldBoostedFrankWolfe:
    def __init__(self, objective_fn, lmo_fn, diam):
        self.objective = objective_fn
        self.lmo = lmo_fn
        self.x = None
        self.func_vals = None
        self.diam = diam
        self.oracle_calls = None
        self.fw_gaps = None

    def run(self, x0, n_steps=int(1e2), K=float('inf'), delta=1e-3, step_size_strategy='short'):
        self.x = x0
        self.func_vals = np.zeros(n_steps)
        self.oracle_calls = np.zeros(n_steps)
        self.fw_gaps = np.zeros(n_steps)
        x = np.copy(self.x)
        
        for t in tqdm(range(n_steps), desc="Boosted Frank-Wolfe Progress"):
            grad = self.objective.gradient(x)
            
            # Compute Frank-Wolfe gap
            v_fw = self.lmo(grad)
            self.fw_gaps[t] = np.sum(grad.flatten() * (x - v_fw).flatten())
            
            # Gradient pursuit procedure (NNMP)
            d = np.zeros_like(grad)
            Lambda = 0
            k = 0
            
            while k < K:
                r = -(grad + d)
                v = self.lmo(-r)
                v_minus_x = v - x
                if np.allclose(d,np.zeros_like(d)):
                    if np.sum(r.flatten() * v_minus_x.flatten()) > 0:
                        u = v_minus_x
                        v_minus_x_flag = True
                    else:
                        u = 0
                elif np.sum(r.flatten() * v_minus_x.flatten()) > np.sum(r.flatten() * -d.flatten() / np.linalg.norm(d)):
                    u = v_minus_x
                    v_minus_x_flag = True
                else:
                    u = -d / np.linalg.norm(d)
                
                lambda_k = np.sum(r.flatten() * u.flatten()) / (np.linalg.norm(u.flatten()) ** 2)
                d_new = d + lambda_k * u
                
                if align(-grad, d_new) - align(-grad, d) > delta:
                    d = d_new
                    if v_minus_x_flag:
                        Lambda += lambda_k
                    else:
                        Lambda *= (1 - lambda_k / np.linalg.norm(d))
                    k += 1
                else:
                    break
            
            g = d / Lambda
            
            # Step size calculation
            if step_size_strategy == 'short':
                eta = align(-grad, g)
                gamma = min(eta * np.linalg.norm(grad) / (self.objective.lipschitz * np.linalg.norm(g)), 1)
            elif step_size_strategy == 'line_search':
                gamma = self.line_search(x, g)
            else:
                raise ValueError("Invalid step_size_strategy. Choose 'short' or 'line_search'.")
            
            # Update x
            x = x + gamma * g
            
            # Record function value and oracle calls
            self.func_vals[t] = self.objective.evaluate(x)
            if t > 0:
                self.oracle_calls[t] = self.oracle_calls[t-1] + k + 1 
            else:
                self.oracle_calls[t] = k + 1
        
        self.x = x

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
        axs[1, 0].semilogy(range(n_steps), self.fw_gaps, label='FW gaps')
        axs[1, 0].set_title('Frank-Wolfe gap vs iterations')
        axs[1, 0].set_xlabel('Iterations')
        axs[1, 0].set_ylabel('FW gap')
        axs[1, 0].legend()

        # Plot Frank-Wolfe gaps as a function of LMO calls
        axs[1, 1].semilogy(self.oracle_calls, self.fw_gaps, label='FW gaps')
        axs[1, 1].set_title('Frank-Wolfe gap vs LMO calls')
        axs[1, 1].set_xlabel('LMO Calls')
        axs[1, 1].set_ylabel('FW gap')
        axs[1, 1].legend()

        plt.tight_layout()
        fig.suptitle('Boosted Frank-Wolfe Algorithm', fontsize=16, fontweight='bold')
        plt.show()
