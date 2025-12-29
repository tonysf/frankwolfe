import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frank_wolfe import *
from frank_wolfe.core.utils import *
import numpy as np
import matplotlib.pyplot as plt

# 1/2 ||Ax-b||^2
class MyObjective(ObjectiveFunction):
    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.lipschitz = np.linalg.norm(A.T @ A, ord=2)
    
    # f(x)
    def evaluate(self, x):
        return 0.5 * np.linalg.norm(self.A @ x - self.b)**2
    
    # \nabla f(x)
    def gradient(self, x):
        return self.A.T @ (self.A @ x - self.b)
        # return logistic grad...

# Set up the problem
n, m = 1000, 750
np.random.seed(42)  # for reproducibility
A = np.random.randn(m, n)
x_true = np.random.randn(n)
x_true[np.abs(x_true) < 0.5] = 0  # Sparsify
b = A @ x_true + 0.01 * np.random.randn(m)
n_steps = 1000
n_K = 5

# Create objective and LMO
obj = MyObjective(A, b)
radius = 1.1 * np.linalg.norm(x_true, ord=1)
lmo = create_lmo(radius, 'l1_ball')

# Initialize and run Vanilla Frank-Wolfe
x0 = np.zeros(n)
vfw = FrankWolfe(obj, lmo)
vfw.run(x0, n_steps=n_steps)

# Initialize and run Away-step Frank-Wolfe
x0 = np.zeros(n)
afw = AwayFrankWolfe(obj, lmo)
afw.run(x0, n_steps=n_steps, step='short')

# Initialize and run Boosted Frank-Wolfe
x0 = np.zeros(n)
bfw = BoostedFrankWolfe(obj, lmo, 2*radius)
bfw.run(x0, n_steps=n_steps, K=n_K, delta=1e-3, step='short')

# Initialize and run Conditional Gradient Sliding
x0 = np.zeros(n)
cgs = CondGradSliding(obj, lmo, 2*radius)
cgs.run(x0, n_steps=n_steps)

# Plot results
plt.figure(figsize=(12, 8))
plt.semilogy(vfw.func_vals, label='Vanilla FW')
plt.semilogy(afw.func_vals, label='Away-step FW')
plt.semilogy(bfw.func_vals, label='Boosted FW')
plt.semilogy(cgs.func_vals, label='Cond. Gradient Sliding')
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
plt.title('Comparison of Frank-Wolfe Gaps')
plt.xlabel('Iterations')
plt.ylabel('Gap')
plt.legend()
plt.grid(True)
plt.show()

# Plot gaps
plt.figure(figsize=(12, 8))
plt.semilogy(vfw.num_oracles, vfw.gaps, label='Vanilla FW')
plt.semilogy(afw.num_oracles, afw.gaps, label='Away-step FW')
plt.semilogy(bfw.num_oracles, bfw.gaps, label='Boosted FW')
plt.semilogy(cgs.num_oracles, cgs.gaps, label='Cond. Gradient Sliding')
plt.title('Gaps vs Oracle Calls')
plt.xlabel('Oracle Calls')
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

print("\nL1 Errors:")
print(f"Vanilla FW: {np.linalg.norm(vfw.x - x_true, ord=1):.6f}")
print(f"Away-step FW: {np.linalg.norm(afw.x - x_true, ord=1):.6f}")
print(f"Boosted FW: {np.linalg.norm(bfw.x - x_true, ord=1):.6f}")
print(f"Cond. Gradient Sliding: {np.linalg.norm(cgs.x - x_true, ord=1):.6f}")

# Plot the recovered signals
plt.figure(figsize=(12, 6))
plt.stem(x_true, markerfmt='ko', linefmt='k-', label='Ground Truth')
plt.stem(vfw.x, markerfmt='bo', linefmt='b-', label='Vanilla FW')
plt.stem(afw.x, markerfmt='ro', linefmt='r-', label='Away-step FW')
plt.stem(bfw.x, markerfmt='go', linefmt='g-', label='Boosted FW')
plt.stem(cgs.x, markerfmt='mo', linefmt='m-', label='Cond. Gradient Sliding')
plt.title('Recovered Signals')
plt.legend()
plt.grid(True)
plt.show()