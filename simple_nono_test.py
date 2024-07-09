import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frank_wolfe import *
import numpy as np
from nonofw_functions import FrankWolfe as OldFrankWolfe

# Set up a simple problem
n = 10
A = np.random.randn(n, n)
b = np.random.randn(n)

class SimpleObjective(ObjectiveFunction):
    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.lipschitz = np.linalg.norm(A.T @ A, ord=2)
    
    def evaluate(self, x):
        return 0.5 * np.linalg.norm(self.A @ x - self.b)**2
    
    def gradient(self, x):
        return self.A.T @ (self.A @ x - self.b)
    
    def linear_operator(self, x):
        return x
    
    def linear_operator_adjoint(self, x):
        return x
    
    def minimal_norm_selection(self, x):
        return np.sign(x)

# Setup
obj = SimpleObjective(A, b)
radius = 1.0
lmo = create_lmo(radius, 'l1_ball')
prox_fn = lambda x, _: np.clip(x, 0, 1)  # simple box constraint

# Initialize
x0 = np.zeros(n)

# Run both implementations for a small number of steps
n_steps = 1000
new_nono = NoNoFrankWolfe(obj, lmo, prox_fn, "lipschitz")
old_nono = OldFrankWolfe(obj, lmo, prox_fn, "lipschitz")

print("Iteration | New Obj Value | Old Obj Value | Difference")
print("---------+---------------+---------------+------------")

for i in range(n_steps):
    new_result = new_nono.run(x0, beta0=1.0, n_steps=i+1)
    old_result = old_nono.run(x0, beta0=1.0, n_steps=i+1)
    
    new_obj_value = obj.evaluate(new_nono.x)
    old_obj_value = obj.evaluate(old_nono.x)
    difference = np.linalg.norm(new_nono.x - old_nono.x)
    
    print(f"{i+1:9d} | {new_obj_value:13.6f} | {old_obj_value:13.6f} | {difference:10.6f}")

print("\nFinal difference in x:", np.linalg.norm(new_nono.x - old_nono.x))
print("Final difference in objective value:", abs(new_obj_value - old_obj_value))

print(f'{new_nono.x}')
print(f'{old_nono.x}')
