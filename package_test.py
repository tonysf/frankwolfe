import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frank_wolfe import AwayFrankWolfe, create_lmo, ObjectiveFunction
import numpy as np

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
A = np.random.randn(m, n)
x_true = np.random.randn(n)
x_true[x_true < 0.5] = 0  # Sparsify
b = A @ x_true + 0.01 * np.random.randn(m)

# Create objective and LMO
obj = MyObjective(A, b)
radius = 1.1 * np.linalg.norm(x_true, ord=1)
lmo = create_lmo(radius, 'l1_ball')

# Initialize and run Away-step Frank-Wolfe
x0 = np.zeros(n)
afw = AwayFrankWolfe(obj, lmo, obj.lipschitz)
result = afw.run(x0, n_steps=1000)

# Plot results
afw.plot_convergence()

print(f"Final objective value: {obj.evaluate(result)}")
print(f"L1 error: {np.linalg.norm(result - x_true, ord=1)}")