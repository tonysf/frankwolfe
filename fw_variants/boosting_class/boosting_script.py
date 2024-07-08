import numpy as np
import fw_functions as fw
import matplotlib.pyplot as plt

class BoostingObjective(fw.ObjectiveFunction):
    def __init__(self, A, b, lipschitz):
        self.A = A
        self.b = b
        self.lipschitz = lipschitz
    
    def evaluate(self, x):
        return 0.5 * (np.linalg.norm((self.A@x - self.b).flatten()) ** 2)
    
    def gradient(self, x):
        return self.A.T@(self.A@x - self.b)

n = int(1e2)
m = int(1e1)
A = (2 * np.random.rand(m, n)) - 1
A = 0.1 * A
At = A.T
lipschitz = np.linalg.norm(A.T@A, ord=2)
X_ground_truth = np.random.rand(n)
X_small = X_ground_truth < 0.2
X_ground_truth[X_small] = 0
b = A @ X_ground_truth
X0 = np.random.rand(n)
X0 = (0.7 * X0 / (np.linalg.norm(X0, ord=1))) * np.linalg.norm(X_ground_truth, ord=1)

# Create the objective function and LMO
obj_fn = BoostingObjective(A, b, lipschitz)
radius = 1 * np.linalg.norm(X_ground_truth, ord=1)
constraint_set = 'l1_ball'
lmo_fn = fw.create_lmo(radius, constraint_set)

# Initialize the algorithm
x0 = np.random.randn(n)
x0 = 0.8 * x0 / np.linalg.norm(x0, ord=1) * radius  # Project onto the L1 ball

# Create and run the Boosted Frank-Wolfe algorithm
Boosted = fw.BoostedFrankWolfe(obj_fn, lmo_fn, 2*radius)
Boosted.run(x0, n_steps=1000, K=5, delta=1e-3, step_size_strategy='Short')

# Plot the results
Boosted.plot_convergence()

# Print the final objective value
print(f"Final objective value: {obj_fn.evaluate(Boosted.x)}")