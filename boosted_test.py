import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frank_wolfe import *
import numpy as np
import matplotlib.pyplot as plt

# Import the old implementation
from boostingfw_functions import BoostedFrankWolfe as OldBoostedFrankWolfe
from boostingfw_functions import ObjectiveFunction as OldObjectiveFunction
from boostingfw_functions import create_lmo as old_create_lmo

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

# # Old implementation
# class OldBoostingObjective(OldObjectiveFunction):
#     def __init__(self, A, b):
#         self.A = A
#         self.b = b
#         self.lipschitz = np.linalg.norm(A.T @ A, ord=2)
    
#     def evaluate(self, x):
#         return 0.5 * np.linalg.norm(self.A @ x - self.b)**2
    
#     def gradient(self, x):
#         return self.A.T @ (self.A @ x - self.b)

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

# Create objective and LMO for old implementation
# old_obj = OldBoostingObjective(A, b)
# Just use the same obj - does it work?
old_lmo = old_create_lmo(radius, 'l1_ball')

# Initialize starting point
x0 = np.random.randn(n)
x0 = 0.8 * x0 / np.linalg.norm(x0, ord=1) * radius  # Project onto the L1 ball

# Run new Boosted Frank-Wolfe
new_bfw = BoostedFrankWolfe(new_obj, new_lmo, 2*radius)
new_result = new_bfw.run(x0, n_steps=1000, K=5, delta=1e-3, step_size_strategy='LineSearch')

# Run old Boosted Frank-Wolfe
old_bfw = OldBoostedFrankWolfe(new_obj, old_lmo, 2*radius)
old_result = old_bfw.run(x0, n_steps=1000, K=5, delta=1e-3, step_size_strategy='LineSearch')

# Plot results
plt.figure(figsize=(12, 8))
plt.semilogy(new_bfw.func_vals, label='New Implementation')
plt.semilogy(old_bfw.func_vals, label='Old Implementation')
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
print(f'Difference between solutions: {np.linalg.norm(old_bfw.x - new_bfw.x)}')

# Compare oracle calls
plt.figure(figsize=(12, 8))
plt.plot(new_bfw.oracle_calls, label='New Implementation')
plt.plot(old_bfw.oracle_calls, label='Old Implementation')
plt.title('Comparison of Oracle Calls')
plt.xlabel('Iterations')
plt.ylabel('Cumulative Oracle Calls')
plt.legend()
plt.show()