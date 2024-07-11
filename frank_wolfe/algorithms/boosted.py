import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from frank_wolfe.algorithms.base import FrankWolfe
from frank_wolfe.core.utils import segment_search, align

class BoostedFrankWolfe(FrankWolfe):
    def __init__(self, objective_fn, lmo_fn, diam):
        super().__init__(objective_fn, lmo_fn)
        self.diam = diam

    def _og_nnmp(self, x, grad, K, delta):
        d = np.zeros_like(grad)
        Lambda = 0
        k = 0
        align_d = align(-grad, d)
        while k < K:
            r = -(grad + d)
            v = self.lmo(-r)
            v_minus_x = v - x
            if np.allclose(d,np.zeros_like(d)):
                if np.sum(r.flatten() * v_minus_x.flatten()) > 0:
                    u = v_minus_x
                    v_minus_x_flag = True
                else:
                    u = np.zeros_like(x)
            
            elif np.sum(r.flatten() * v_minus_x.flatten()) > np.sum(r.flatten() * -d.flatten() / np.linalg.norm(d)):
                u = v_minus_x
                v_minus_x_flag = True
            # This is the first step in Cyrille's code
            else:
                u = -d / np.linalg.norm(d)
            
            lambda_k = np.sum(r.flatten() * u.flatten()) / (np.linalg.norm(u.flatten()) ** 2)
            d_new = d + lambda_k * u
            align_d_new = align(-grad, d_new)
            align_improve = align_d_new - align_d
            if align_improve > delta:
                d = d_new
                if v_minus_x_flag:
                    Lambda += lambda_k
                else:
                    Lambda *= (1 - lambda_k / np.linalg.norm(d))
                k += 1
            else:
                k += 1
                break

        return d, Lambda, k
    
    # This is an exact port of Cyrille's code.
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
            
            # Nonnegative Matching Pursuit
            # d, Lambda, num_oracles = self._nnmp(x, grad, K, delta)
            # g = d / Lambda
            g, num_oracles, align_g = self._nnmp(self.x, grad, K, delta)

            og_d, og_Lam, og_num_orac = self._og_nnmp(self.x, grad, K, delta)
            og_g = og_d / og_Lam

            # print('The two nnmp procedures used the same number of oracles:')
            # print(f'{og_num_orac=}')
            # print(f'{num_oracles=}')
            # print('The two nnmp procedures produced the same direction:')
            # print(f'{np.linalg.norm(og_g- g)}')

            # Step size calculation
            if step == 'Short':
                og_eta = align(-grad, g)
                og_gamma = min(og_eta * np.linalg.norm(grad) / (self.objective.lipschitz * np.linalg.norm(g)), 1)
                gamma = min(align_g*np.linalg.norm(grad) / (self.objective.lipschitz * np.linalg.norm(g)), 1)
                self.x = self.x + gamma * g
                # self.x = self.x + og_gamma * og_g
            elif step == 'LineSearch':
                self.x, gamma = segment_search(self.objective, self.x, self.x + g)
            else:
                raise ValueError("Invalid step type. Choose 'Short' or 'LineSearch'.")
            
            # Record function value and oracle calls
            self.func_vals[i] = self.objective.evaluate(self.x)
            self.num_oracles[i] += num_oracles
        self.num_oracles = np.cumsum(self.num_oracles)

    def test_run(self, x0, n_steps=int(1e2), K=float('inf'), delta=1e-3, step='short'):
        self.x = x0
        self.func_vals = np.zeros(n_steps)
        self.oracle_calls = np.zeros(n_steps)
        self.gaps = np.zeros(n_steps)
        x = np.copy(self.x)
        new_lmo = self.lmo

        def cyrille_nnmp(x, grad_f_x, align_tol, K):
            
            d, Lbd, flag = np.zeros(len(x)), 0, True
            
            G = grad_f_x+d
            align_d = align(-grad_f_x, d)
            
            for k in range(K):
                
                u = new_lmo(G)-x
                d_norm = np.linalg.norm(d)
                if d_norm > 0 and np.dot(G, -d/d_norm) < np.dot(G, u):
                    u = -d/d_norm
                    flag = False
                lbd = -np.dot(G, u)/np.linalg.norm(u)**2
                dd = d+lbd*u
                align_dd = align(-grad_f_x, dd)
                align_improv = align_dd-align_d
                
                if align_improv > align_tol:
                    d = dd
                    Lbd = Lbd+lbd if flag == True else Lbd*(1-lbd/d_norm)
                    G = grad_f_x+d
                    align_d = align_dd
                    flag = True
                    
                else:
                    break
                
            return d/Lbd, k, align_d
        
        for i in tqdm(range(n_steps), desc="Boosted Frank-Wolfe Progress"):
            grad = self.objective.gradient(x)
            
            # Compute Frank-Wolfe gap
            v_fw = self.lmo(grad)
            self.gaps[i] = np.sum(grad.flatten() * (x - v_fw).flatten())
            
            cyrille_g, cyrille_num_oracles, cyrille_align_g = cyrille_nnmp(x, grad, delta, K)
            if step == 'Short':
                cyrille_gamma = min(cyrille_align_g*np.linalg.norm(grad)/(self.objective.lipschitz*np.linalg.norm(cyrille_g)), 1)
                cyrille_x = x+cyrille_gamma*cyrille_g
            elif step == 'LineSearch':
                x, gamma = segment_search(f, grad_f, x, x+g)

            # Nonnegative Matching Pursuit
            # d, Lambda, num_oracles = self._nnmp(x, grad, K, delta)
            # g = d / Lambda
            g, num_oracles, align_g = self._nnmp(x, grad, K, delta)

            og_d, og_Lam, og_num_orac = self._og_nnmp(x, grad, K, delta)
            og_g = og_d/og_Lam

            # print('The three nnmp procedures used the same number of oracles:')
            # print(f'{og_num_orac=}')
            # print(f'{num_oracles=}')
            # print(f'{cyrille_num_oracles=}')
            # print('The three nnmp procedures produced the same direction:')
            # print('og_g - g')
            # print(f'{np.linalg.norm(og_g- g)}')
            # print('cyrille_g - g')
            # print(f'{np.linalg.norm(cyrille_g-g)}')

            # Step size calculation
            if step == 'Short':
                og_eta = align(-grad, g)
                og_gamma = min(og_eta * np.linalg.norm(grad) / (self.objective.lipschitz * np.linalg.norm(g)), 1)
                gamma = min(align_g*np.linalg.norm(grad)/(self.objective.lipschitz*np.linalg.norm(g)), 1)
                x = x + gamma * g
            elif step == 'LineSearch':
                x, gamma = segment_search(self.objective, x, x + g)
            else:
                raise ValueError("Invalid step type. Choose 'Short' or 'LineSearch'.")
            
            # Record function value and oracle calls
            self.func_vals[i] = self.objective.evaluate(x)
            if i > 0:
                self.oracle_calls[i] = self.oracle_calls[i-1] + num_oracles
            else:
                self.oracle_calls[i] = num_oracles
        
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

class oldBoostedFrankWolfe:
    def __init__(self, objective_fn, lmo_fn, diam):
        self.objective = objective_fn
        self.lmo = lmo_fn
        self.x = None
        self.func_vals = None
        self.diam = diam
        self.oracle_calls = None
        self.gaps = None

    def run(self, x0, n_steps=int(1e2), K=float('inf'), delta=1e-3, step='short'):
        self.x = x0
        self.func_vals = np.zeros(n_steps)
        self.oracle_calls = np.zeros(n_steps)
        self.gaps = np.zeros(n_steps)
        x = np.copy(self.x)
        
        for t in tqdm(range(n_steps), desc="Boosted Frank-Wolfe Progress"):
            grad = self.objective.gradient(x)
            
            # Compute Frank-Wolfe gap
            v_fw = self.lmo(grad)
            self.gaps[t] = np.sum(grad.flatten() * (x - v_fw).flatten())
            
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
            if step == 'short':
                eta = align(-grad, g)
                gamma = min(eta * np.linalg.norm(grad) / (self.objective.lipschitz * np.linalg.norm(g)), 1)
            elif step == 'line_search':
                gamma = self.line_search(x, g)
            else:
                raise ValueError("Invalid step type. Choose 'short' or 'line_search'.")
            
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
