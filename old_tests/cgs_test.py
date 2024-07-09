import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frank_wolfe import *
import numpy as np
import matplotlib.pyplot as plt

# Import the old implementation
from cgs_functions import SlidingFrankWolfe as OldSlidingFrankWolfe
from cgs_functions import ObjectiveFunction as OldObjectiveFunction
from cgs_functions import create_lmo as old_create_lmo

# New implementation
class SlidingObjective(ObjectiveFunction):
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
new_obj = SlidingObjective(A, b)
radius = 1.0 * np.linalg.norm(x_true, ord=1)
new_lmo = create_lmo(radius, 'l1_ball')

# Create objective and LMO for old implementation
class OldSlidingObjective(OldObjectiveFunction):
    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.lipschitz = np.linalg.norm(A.T @ A, ord=2)
    
    def evaluate(self, X):
        return 0.5 * (np.linalg.norm((self.A @ X) - self.b) ** 2)
    
    def grad(self, X):
        return self.A.T @ ((self.A @ X) - self.b)

old_obj = OldSlidingObjective(A, b)


# Initialize starting point
x0 = np.random.randn(n)
x0 = 0.8 * x0 / np.linalg.norm(x0, ord=1) * radius  # Project onto the L1 ball

# Run new Sliding Frank-Wolfe
diam = 2 * radius
new_cgs = CondGradSliding(new_obj, new_lmo, 2*radius, diam)
new_result = new_cgs.run(x0, n_steps=10)

# Run old Sliding Frank-Wolfe
old_cgs = OldSlidingFrankWolfe(old_obj, new_lmo, 2*radius)
old_result = old_cgs.run(x0, n_steps=10)

# Plot results
plt.figure(figsize=(12, 8))
plt.semilogy(new_cgs.func_vals, label='New Implementation')
plt.semilogy(old_cgs.func_vals, label='Old Implementation')
plt.title('Comparison of New and Old Conditional Gradient Sliding Implementations')
plt.xlabel('Iterations')
plt.ylabel('Objective Value')
plt.legend()
plt.show()

# Print final results
print(f"New implementation - Final objective value: {new_cgs.func_vals[-1]}")
print(f"New implementation - L1 error: {np.linalg.norm(new_cgs.x - x_true, ord=1)}")
print(f"Old implementation - Final objective value: {old_cgs.func_vals[-1]}")
print(f"Old implementation - L1 error: {np.linalg.norm(old_cgs.x - x_true, ord=1)}")
print(f'Difference between solutions: {np.linalg.norm(old_cgs.x - new_cgs.x)}')

# Compare LMO calls
plt.figure(figsize=(12, 8))
plt.plot(new_cgs.gaps, label='New Implementation')
plt.plot(old_cgs.gaps, label='Old Implementation')
plt.title('Comparison of LMO Calls')
plt.xlabel('Iterations')
plt.ylabel('Cumulative LMO Calls')
plt.legend()
plt.show()