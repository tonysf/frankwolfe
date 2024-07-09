import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frank_wolfe import *
import numpy as np
import matplotlib.pyplot as plt

# Import the old implementation
from awayfw_functions import AwayFrankWolfe as OldAwayFrankWolfe

# New implementation
class L1ConstrainedLeastSquares(ObjectiveFunction):
    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.lipschitz = np.linalg.norm(A.T @ A, ord=2)
    
    def evaluate(self, x):
        return 0.5 * np.linalg.norm(self.A @ x - self.b)**2
    
    def gradient(self, x):
        return self.A.T @ (self.A @ x - self.b)

# Set up the problem
n, m = 100, 50
np.random.seed(42)  # for reproducibility
A = np.random.randn(m, n)
A /= np.linalg.norm(A, axis=1, keepdims=True)  # normalize rows
x_true = np.zeros(n)
k = int(0.2 * n)  # number of non-zero elements
non_zero_indices = np.random.choice(n, k, replace=False)
x_true[non_zero_indices] = np.random.randn(k)
b = A @ x_true + 0.01 * np.random.randn(m)  # add some noise

# Create objective and LMO
new_obj = L1ConstrainedLeastSquares(A, b)
radius = 1.1 * np.linalg.norm(x_true, ord=1)
new_lmo = create_lmo(radius, 'l1_ball')

# Initialize starting point
x0 = np.zeros(n)

# Run new Away-step Frank-Wolfe
new_afw = AwayFrankWolfe(new_obj, new_lmo, new_obj.lipschitz)
new_result = new_afw.run(x0, n_steps=100)

# Run old Away-step Frank-Wolfe
old_afw = OldAwayFrankWolfe(new_obj, new_lmo, new_obj.lipschitz)
old_result = old_afw.run(x0, n_steps=100)

# Plot results
plt.figure(figsize=(12, 8))
plt.semilogy(new_afw.func_vals, label='New Implementation')
plt.semilogy(old_afw.func_vals, label='Old Implementation')
plt.title('Comparison of New and Old Away-step Frank-Wolfe Implementations')
plt.xlabel('Iterations')
plt.ylabel('Objective Value')
plt.legend()
plt.show()

# Print final results
print(f"New implementation - Final objective value: {new_afw.func_vals[-1]}")
print(f"New implementation - L1 error: {np.linalg.norm(new_afw.x - x_true, ord=1)}")
print(f"Old implementation - Final objective value: {old_afw.func_vals[-1]}")
print(f"Old implementation - L1 error: {np.linalg.norm(old_afw.x - x_true, ord=1)}")
print(f'Difference between solutions: {np.linalg.norm(old_afw.x - new_afw.x)}')

# Compare gaps
plt.figure(figsize=(12, 8))
plt.semilogy(new_afw.gaps, label='New Implementation')
plt.semilogy(old_afw.gaps, label='Old Implementation')
plt.title('Comparison of Frank-Wolfe Gaps')
plt.xlabel('Iterations')
plt.ylabel('Gap')
plt.legend()
plt.show()

# Plot the recovered signals
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.stem(x_true, markerfmt='bo', linefmt='b-', label='Ground Truth')
plt.title('Ground Truth Signal')
plt.legend()

plt.subplot(2, 1, 2)
plt.stem(new_afw.x, markerfmt='ro', linefmt='r-', label='Recovered Signal (New)')
plt.stem(old_afw.x, markerfmt='go', linefmt='g-', label='Recovered Signal (Old)')
plt.title('Recovered Signals')
plt.legend()

plt.tight_layout()
plt.show()