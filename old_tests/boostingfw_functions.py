from scipy.sparse.linalg import svds
from scipy.sparse.linalg import eigsh
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def align(d, d_hat):
    if np.linalg.norm(d_hat) == 0:
        return -1
    return np.dot(d, d_hat) / (np.linalg.norm(d) * np.linalg.norm(d_hat))

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
    def __init__(self, lipschitz):
        self.lipschitz = lipschitz
    
    def evaluate(self, x):
        raise NotImplementedError
    
    def gradient(self, x):
        raise NotImplementedError


class BoostedFrankWolfe:
    def __init__(self, objective_fn, lmo_fn, diam):
        self.objective = objective_fn
        self.lmo = lmo_fn
        self.x = None
        self.func_vals = None
        self.diam = diam
        self.oracle_calls = None
        self.fw_gaps = None

    def segment_search(self, x, y, tol=1e-15, stepsize=True):
        """
        Minimizes f over [x, y], i.e., f(x+gamma*(y-x)) as a function of scalar gamma in [0,1]
        """
        # restrict segment of search to [x, y]
        d = (y-x).copy()
        left, right = x.copy(), y.copy()
        
        # if the minimum is at an endpoint
        if np.dot(d, self.objective.gradient(x))*np.dot(d, self.objective.gradient(y)) >= 0:
            if self.objective.evaluate(y) <= self.objective.evaluate(x):
                return y, 1
            else:
                return x, 0
        
        # apply golden-section method to segment
        gold = (1+np.sqrt(5))/2
        improv = np.inf
        while improv > tol:
            old_left, old_right = left, right
            new = left+(right-left)/(1+gold)
            probe = new+(right-new)/2
            if self.objective.evaluate(probe) <= self.objective.evaluate(new):
                left, right = new, right
            else:
                left, right = left, probe
            improv = np.linalg.norm(self.objective.evaluate(right)-self.objective.evaluate(old_right))+np.linalg.norm(self.objective.evaluate(left)-self.objective.evaluate(old_left))
        
        x_min = (left+right)/2
        
        # compute step size gamma
        gamma = 0
        if stepsize == True:
            for i in range(len(d)):
                if d[i] != 0:
                    gamma = (x_min[i]-x[i])/d[i]
                    break
        
        return x_min, gamma
    
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
            
            ### BEING NNMP
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
            ### END NNMP

            # Step size calculation
            if step_size_strategy == 'Short':
                eta = align(-grad, g)
                gamma = min(eta * np.linalg.norm(grad) / (self.objective.lipschitz * np.linalg.norm(g)), 1)
                # Update x
                x = x + gamma * g
            elif step_size_strategy == 'LineSearch':
                x, gamma = self.segment_search(x, x + g)
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

class NonsmoothBoostedFrankWolfe:
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

    def line_search(self, x, g):
        # Implement line search here
        # This is a placeholder implementation
        return 1.0

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
