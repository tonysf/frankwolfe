import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frank_wolfe import *
from frank_wolfe.core.utils import *
import numpy as np
import matplotlib.pyplot as plt

# Import the old implementation
from nonofw_functions import FrankWolfe as OldFrankWolfe

# Synthetic problem setup
height, width = 200, 200
np.random.seed(42)
true_image = np.random.rand(height, width)
observed_indices_mask = np.random.choice([True, False], size=(height, width), p=[0.8, 0.2])
observations = true_image[observed_indices_mask]

# Objective function
class RobustPCAObjective(ObjectiveFunction):
    def __init__(self, delta, observed_indices_mask, observations):
        self.delta = delta
        self.observed_indices_mask = observed_indices_mask
        self.observations = observations
    
    def evaluate(self, X):
        X_observed = X[self.observed_indices_mask]
        diff = X_observed - self.observations
        return (diff**2/(2*(self.delta**2) + diff**2)).sum()
    
    def gradient(self, X):
        X_observed = X[self.observed_indices_mask]
        diff = X_observed - self.observations
        grad_observed = 4 * diff * (self.delta ** 2) / ((2 * (self.delta ** 2) + (diff ** 2)) ** 2)
        grad = np.zeros_like(X)
        grad[self.observed_indices_mask] = grad_observed
        return grad
    
    def linear_operator(self, X):
        return X
    
    def linear_operator_adjoint(self, X):
        return X
    
    def minimal_norm_selection(self, X):
        return np.sign(X)

# Setup
delta = 0.1
X0 = np.zeros_like(true_image)
radius = 2.0 * np.linalg.norm(true_image, ord='nuc')
constraint_set = 'nuclear_norm_ball'

# New implementation setup
# Note that this objective works for the old implementation as well
new_obj = RobustPCAObjective(delta, observed_indices_mask, observations)
new_lmo = create_lmo(radius, constraint_set)

# Run new NoNo Frank-Wolfe
new_nono = NoNoFrankWolfe(new_obj, new_lmo, proj_cube, "indicator")
new_nono.run(X0, beta0=1e7, n_steps=500)
new_result = new_nono.x

# Run old NoNo Frank-Wolfe
old_nono = OldFrankWolfe(new_obj, new_lmo, proj_cube, "indicator")
old_nono.run(X0, beta0=1e7, n_steps=500)
old_result = old_nono.x

# Plot results
plt.figure(figsize=(12, 8))
plt.semilogy(new_nono.func_vals, label='New Implementation')
plt.semilogy(old_nono.func_vals, label='Old Implementation')
plt.title('Comparison of New and Old NoNo Frank-Wolfe Implementations')
plt.xlabel('Iterations')
plt.ylabel('Objective Value')
plt.legend()
plt.show()

# Print final results
print(f"New implementation - Final objective value: {new_nono.func_vals[-1]}")
print(f"Old implementation - Final objective value: {old_nono.func_vals[-1]}")
print(f'Difference between solutions: {np.linalg.norm(old_nono.x - new_nono.x)}')

# Compare LMO outputs
def compare_lmo_outputs(new_nono, old_nono, tolerance=1e-6):
    for i in range(min(len(new_nono.func_vals), len(old_nono.func_vals))):
        new_grad = new_nono.objective.gradient(new_nono.x)
        old_grad = old_nono.objective.gradient(old_nono.x)
        new_lmo_output = new_nono.lmo(new_grad)
        old_lmo_output = old_nono.lmo(old_grad)
        
        diff = np.linalg.norm(new_lmo_output - old_lmo_output)
        if diff > tolerance:
            print(f"LMO outputs diverge at iteration {i}")
            print(f"Difference norm: {diff}")
            break