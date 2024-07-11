import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frank_wolfe import *
from frank_wolfe.core.utils import *
import numpy as np
import matplotlib.pyplot as plt

# Import the old implementation
from boostingfw_functions import BoostedFrankWolfe as OldBoostedFrankWolfe

# New implementation
class BoostingObjective(ObjectiveFunction):
    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.lipschitz = np.linalg.norm(A.T @ A, ord=2)
    
    def evaluate(self, x):
        return 0.5 * np.linalg.norm(self.A @ x - self.b)**2
    
    def gradient(self, x):
        return self.A.T @ (self.A @ x - self.b)

# Set up the problem
n, m = 100, 10
np.random.seed(42)  # for reproducibility
A = 0.1 * (2 * np.random.rand(m, n) - 1)
x_true = np.random.rand(n)
x_true[x_true < 0.2] = 0  # Sparsify
b = A @ x_true

# Create objective and LMO for new implementation
new_obj = BoostingObjective(A, b)
radius = 1.0 * np.linalg.norm(x_true, ord=1)
new_lmo = create_lmo(radius, 'l1_ball')

# Initialize starting point
x0 = np.random.randn(n)
x0 = 0.8 * x0 / np.linalg.norm(x0, ord=1) * radius  # Project onto the L1 ball
n_steps = 450
n_K = 5
# Run new Boosted Frank-Wolfe
new_bfw = BoostedFrankWolfe(new_obj, new_lmo, 2*radius)
new_result = new_bfw.run(x0, n_steps=n_steps, K=n_K, delta=1e-3, step_size_strategy='Short')

# Run old Boosted Frank-Wolfe
old_bfw = OldBoostedFrankWolfe(new_obj, new_lmo, 2*radius)
old_result = old_bfw.run(x0, n_steps=n_steps, K=n_K, delta=1e-3, step_size_strategy='Short')


# Compare with Cyrille's code

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

def cyrille_boostfw(f, grad_f, L, x, step='ls', n_steps=1000, align_tol=1e-3, K=5):
    
    values, times, oracles, gaps = [], [0], [0], [np.dot(grad_f(x), x-new_lmo(grad_f(x)))]
    
    # x = new_lmo(grad_f(x))
    
    for k in range(n_steps):
        grad_f_x = grad_f(x)
        gaps.append(np.dot(grad_f_x, x-new_lmo(grad_f_x)))
        g, num_oracles, align_g = cyrille_nnmp(x, grad_f_x, align_tol, K)
        
        if step == 'Short':
            gamma = min(align_g*np.linalg.norm(grad_f_x)/(L*np.linalg.norm(g)), 1)
            x = x+gamma*g
        elif step == 'LineSearch':
            x, gamma = segment_search(f, grad_f, x, x+g)
        values.append(f(x))
        oracles.append(num_oracles)
    return x, values, oracles, gaps

cyrille_x, cyrille_values, cyrille_oracles, cyrille_gaps = cyrille_boostfw(new_obj.evaluate, new_obj.gradient, new_obj.lipschitz, x0, step='Short', n_steps=n_steps, K=n_K)



# Plot results
plt.figure(figsize=(12, 8))
plt.semilogy(new_bfw.func_vals, label='New Implementation')
plt.semilogy(old_bfw.func_vals, label='Old Implementation')
plt.semilogy(cyrille_values, label='Cyrille')
plt.title('Comparison of New and Old Boosted Frank-Wolfe Implementations')
plt.xlabel('Iterations')
plt.ylabel('Objective Value')
plt.legend()
plt.show()

# Print final results
print(f"New implementation - Final objective value: {new_bfw.func_vals[-1]}")
print(f"New implementation - L1 error: {np.linalg.norm(new_bfw.x - x_true, ord=1)}")
print(f"Old implementation - Final objective value: {old_bfw.func_vals[-1]}")
print(f"Old implementation - L1 error: {np.linalg.norm(old_bfw.x - x_true, ord=1)}")
print(f"Cyrille implementation - Final objective value: {cyrille_values[-1]}")
print(f"Cyrille implementation - L1 error: {np.linalg.norm(cyrille_x - x_true, ord=1)}")
print(f'Difference between solutions new-old: {np.linalg.norm(old_bfw.x - new_bfw.x)}')
print(f'Difference between solutions cyrille-new: {np.linalg.norm(cyrille_x - new_bfw.x)}')