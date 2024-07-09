import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frank_wolfe import *
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
afw.run(x0, n_steps=1000)
afw_data = afw.to_dict()

# Initialize and run Boosted Frank-Wolfe
x0 = np.zeros(n)
bfw = BoostingFrankWolfe(obj, lmo, obj.lipschitz)
bfw.run(x0, n_steps=1000)
bfw_data = bfw.to_dict()

# Initialize and run Conditional Gradient Sliding
x0 = np.zeros(n)
cgs = CondGradSliding(obj, lmo, obj.lipschitz)
cgs.run(x0, n_steps=1000)
cgs_data = cgs.to_dict()

# Plot results
