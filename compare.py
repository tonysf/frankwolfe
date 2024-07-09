import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frank_wolfe import *
from frank_wolfe.core.utils import *
import numpy as np
import matplotlib.pyplot as plt

class MyObjective(ObjectiveFunction):
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
A = np.random.randn(m, n)
x_true = np.random.randn(n)
x_true[x_true < 0.5] = 0  # Sparsify
b = A @ x_true + 0.01 * np.random.randn(m)

# Create objective and LMO
obj = MyObjective(A, b)
radius = 1.1 * np.linalg.norm(x_true, ord=1)
lmo = create_lmo(radius, 'l1_ball')

# Initialize and run Vanilla Frank-Wolfe
x0 = np.zeros(n)
vfw = FrankWolfe(obj, lmo)
vfw.run(x0, n_steps=1000)

# Initialize and run Away-step Frank-Wolfe
x0 = np.zeros(n)
afw = AwayFrankWolfe(obj, lmo, obj.lipschitz)
afw.run(x0, n_steps=1000, step_rule = 'Short')

# Initialize and run Boosted Frank-Wolfe
x0 = np.zeros(n)
bfw = BoostedFrankWolfe(obj, lmo, 2*radius)
bfw.run(x0, n_steps=1000, K=5, delta=1e-3, step_size_strategy='Short')

# Initialize and run Conditional Gradient Sliding
x0 = np.zeros(n)
cgs = CondGradSliding(obj, lmo, obj.lipschitz, 2*radius)
cgs.run(x0, n_steps=1000)

# Compare with Cyrille's code

def cyrille_nnmp(x, grad_f_x, align_tol, K):
    
    d, Lbd, flag = np.zeros(len(x)), 0, True
    
    G = grad_f_x+d
    align_d = align(-grad_f_x, d)
    
    for k in range(K):
        
        u = lmo(G)-x
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
    
    values, times, oracles, gaps = [f(x)], [0], [0], [np.dot(grad_f(x), x-lmo(grad_f(x)))]
    
    x = lmo(grad_f(x))
    values.append(f(x))
    oracles.append(1)
    
    for k in range(n_steps):
        grad_f_x = grad_f(x)
        gaps.append(np.dot(grad_f_x, x-lmo(grad_f_x)))
        g, num_oracles, align_g = cyrille_nnmp(x, grad_f_x, align_tol, K)
        
        if step == 'Short':
            gamma = min(align_g*np.linalg.norm(grad_f_x)/(L*np.linalg.norm(g)), 1)
            x = x+gamma*g
        elif step == 'LineSearch':
            x, gamma = segment_search(f, grad_f, x, x+g)
        else:
            gamma = min(-np.dot(g, grad_f_x)/np.dot(g, (np.dot(step, g))), 1)
            x = x+gamma*g
        values.append(f(x))
        oracles.append(num_oracles)
    return x, values, oracles, gaps

cyrille_x, cyrille_values, cyrille_oracles, cyrille_gaps = cyrille_boostfw(obj.evaluate, obj.gradient, obj.lipschitz, x0, step='Short', n_steps=1000, K=5)





# Plot results
plt.figure(figsize=(12, 8))
plt.semilogy(vfw.func_vals, label='Vanilla FW')
plt.semilogy(afw.func_vals, label='Away-step FW')
plt.semilogy(bfw.func_vals, label='Boosted FW')
plt.semilogy(cgs.func_vals, label='Cond. Gradient Sliding')
plt.semilogy(cyrille_values, label='Cyrille')
plt.title('Comparison of Frank-Wolfe Variants')
plt.xlabel('Iterations')
plt.ylabel('Objective Value')
plt.legend()
plt.grid(True)
plt.show()

# Plot gaps
plt.figure(figsize=(12, 8))
plt.semilogy(vfw.gaps, label='Vanilla FW')
plt.semilogy(afw.gaps, label='Away-step FW')
plt.semilogy(bfw.gaps, label='Boosted FW')
plt.semilogy(cgs.gaps, label='Cond. Gradient Sliding')
plt.semilogy(cyrille_gaps, label='Cyrille')
plt.title('Comparison of Frank-Wolfe Gaps')
plt.xlabel('Iterations')
plt.ylabel('Gap')
plt.legend()
plt.grid(True)
plt.show()

# Print final results
print("Final Objective Values:")
print(f"Vanilla FW: {vfw.func_vals[-1]:.6f}")
print(f"Away-step FW: {afw.func_vals[-1]:.6f}")
print(f"Boosted FW: {bfw.func_vals[-1]:.6f}")
print(f"Cond. Gradient Sliding: {cgs.func_vals[-1]:.6f}")
print(f"Cyrille: {cyrille_func_vals[-1]:.6f}")

print("\nL1 Errors:")
print(f"Vanilla FW: {np.linalg.norm(vfw.x - x_true, ord=1):.6f}")
print(f"Away-step FW: {np.linalg.norm(afw.x - x_true, ord=1):.6f}")
print(f"Boosted FW: {np.linalg.norm(bfw.x - x_true, ord=1):.6f}")
print(f"Cond. Gradient Sliding: {np.linalg.norm(cgs.x - x_true, ord=1):.6f}")
print(f"Cyrille: {np.linalg.norm(cyrille_x - x_true, ord=1):.6f}")

# Plot the recovered signals
plt.figure(figsize=(12, 6))
plt.stem(x_true, markerfmt='ko', linefmt='k-', label='Ground Truth')
plt.stem(vfw.x, markerfmt='bo', linefmt='b-', label='Vanilla FW')
plt.stem(afw.x, markerfmt='ro', linefmt='r-', label='Away-step FW')
plt.stem(bfw.x, markerfmt='go', linefmt='g-', label='Boosted FW')
plt.stem(cgs.x, markerfmt='mo', linefmt='m-', label='Cond. Gradient Sliding')
plt.stem(cyrille_x, markerfmt='mo', linefmt='m-', label='Cyrille')
plt.title('Recovered Signals')
plt.legend()
plt.grid(True)
plt.show()