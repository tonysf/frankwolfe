"""
Nonsmooth Frank-Wolfe example: Diagonal constraint problem

This example demonstrates nonsmooth Frank-Wolfe applied to the problem:

    min_{x} 1/2||x-y||_2^2 + iota_{C1 ∩ C2}(x)

where:
- y is a randomly generated vector
- C1 is an L1 ball of radius 2 centered at [1,0,...,0]
- C2 is an L1 ball of radius 2 centered at [-1,0,...,0]
- Their intersection C1 ∩ C2 is the unit L1 ball centered at origin

We solve this by lifting to a product space:

    min_{x1, x2} 1/4||x1-y||_2^2 + 1/4||x2-y||_2^2 + iota_V(x1,x2) + iota_{C1×C2}(x1,x2)

where V is the diagonal subspace {(x1,x2): x1=x2}, and we smooth iota_V
using the Moreau envelope.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from frank_wolfe.algorithms.base import FrankWolfe
from frank_wolfe.core.objective import ObjectiveFunction


class DiagonalConstraintObjective(ObjectiveFunction):
    """
    Objective function for the lifted problem with Moreau envelope smoothing.

    f(x1, x2) = 1/4||x1-y||^2 + 1/4||x2-y||^2 + (1/(2*beta))||x1-x2||^2

    The last term is the Moreau envelope of the indicator function of the
    diagonal subspace V = {(x1,x2): x1=x2}.
    """

    def __init__(self, y, beta):
        """
        Initialize the objective function.

        Args:
            y: Target vector (n-dimensional)
            beta: Moreau envelope parameter (smoothing parameter)
        """
        self.y = y
        self.beta = beta
        self.n = len(y)
        # Lipschitz constant for the smooth part
        self.lipschitz = 0.5 + 1.0/beta

    def evaluate(self, z):
        """
        Evaluate f(x1, x2) where z = [x1; x2] is the stacked vector.

        Args:
            z: Stacked vector of length 2n

        Returns:
            Function value
        """
        x1 = z[:self.n]
        x2 = z[self.n:]

        # Quadratic terms
        term1 = 0.25 * np.linalg.norm(x1 - self.y)**2
        term2 = 0.25 * np.linalg.norm(x2 - self.y)**2

        # Moreau envelope of diagonal constraint
        moreau_term = (1.0/(2*self.beta)) * np.linalg.norm(x1 - x2)**2

        return term1 + term2 + moreau_term

    def gradient(self, z):
        """
        Compute gradient of f(x1, x2).

        ∇_{x1} f = 1/2(x1-y) + (1/beta)(x1-x2)
        ∇_{x2} f = 1/2(x2-y) + (1/beta)(x2-x1)

        Args:
            z: Stacked vector of length 2n

        Returns:
            Gradient vector of length 2n
        """
        x1 = z[:self.n]
        x2 = z[self.n:]

        # Gradient w.r.t. x1
        grad_x1 = 0.5 * (x1 - self.y) + (1.0/self.beta) * (x1 - x2)

        # Gradient w.r.t. x2
        grad_x2 = 0.5 * (x2 - self.y) + (1.0/self.beta) * (x2 - x1)

        return np.concatenate([grad_x1, grad_x2])


def create_product_lmo(n, radius1=2.0, radius2=2.0, center1=None, center2=None):
    """
    Create LMO for the product space C1 × C2.

    Args:
        n: Dimension of each space
        radius1: Radius of C1 (L1 ball)
        radius2: Radius of C2 (L1 ball)
        center1: Center of C1 (default: [1,0,...,0])
        center2: Center of C2 (default: [-1,0,...,0])

    Returns:
        LMO function for product space
    """
    if center1 is None:
        center1 = np.zeros(n)
        center1[0] = 1.0

    if center2 is None:
        center2 = np.zeros(n)
        center2[0] = -1.0

    def product_lmo(grad):
        """
        LMO for C1 × C2 where each is a translated L1 ball.

        For a translated L1 ball {x: ||x-c|| <= r}, the LMO is:
            s = c - r * sign(grad) * e_i*
        where i* = argmax_i |grad_i|

        Args:
            grad: Gradient vector of length 2n

        Returns:
            Vertex of C1 × C2 minimizing <grad, (s1, s2)>
        """
        grad1 = grad[:n]
        grad2 = grad[n:]

        # LMO for C1 (L1 ball centered at center1)
        index1 = np.argmax(np.abs(grad1))
        s1 = center1.copy()
        s1[index1] -= radius1 * np.sign(grad1[index1])

        # LMO for C2 (L1 ball centered at center2)
        index2 = np.argmax(np.abs(grad2))
        s2 = center2.copy()
        s2[index2] -= radius2 * np.sign(grad2[index2])

        return np.concatenate([s1, s2])

    return product_lmo


def solve_ground_truth(y, radius=1.0):
    """
    Solve the original problem directly using projection.

    The solution to min_{x in L1 ball} 1/2||x-y||^2 is the L1 ball projection of y.

    Args:
        y: Target vector
        radius: Radius of L1 ball

    Returns:
        Optimal solution x*
    """
    # L1 ball projection using soft thresholding
    # This is a simplified version; full implementation would use sorting
    y_abs = np.abs(y)
    y_norm = np.sum(y_abs)

    if y_norm <= radius:
        return y.copy()

    # Binary search for threshold
    def compute_l1_norm(theta):
        return np.sum(np.maximum(y_abs - theta, 0))

    theta_low, theta_high = 0.0, np.max(y_abs)

    for _ in range(100):
        theta = (theta_low + theta_high) / 2
        current_norm = compute_l1_norm(theta)

        if current_norm > radius:
            theta_low = theta
        else:
            theta_high = theta

    return np.sign(y) * np.maximum(y_abs - theta, 0)


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    # Problem parameters
    n = 20  # Dimension
    beta = 0.1  # Moreau envelope parameter (smaller = tighter approximation)

    # Generate random target vector
    y = np.random.randn(n)
    print(f"Target vector y: {y[:5]}... (showing first 5 elements)")
    print(f"||y||_1 = {np.linalg.norm(y, ord=1):.4f}")
    print(f"||y||_2 = {np.linalg.norm(y, ord=2):.4f}")

    # Solve ground truth
    x_star = solve_ground_truth(y, radius=1.0)
    f_star = 0.5 * np.linalg.norm(x_star - y)**2
    print(f"\nGround truth solution:")
    print(f"||x*||_1 = {np.linalg.norm(x_star, ord=1):.4f}")
    print(f"f(x*) = {f_star:.6f}")

    # Create objective and LMO for lifted problem
    objective = DiagonalConstraintObjective(y, beta)
    lmo = create_product_lmo(n, radius1=2.0, radius2=2.0)

    # Initialize at LMO of zero gradient (center of constraints)
    z0 = np.concatenate([np.zeros(n), np.zeros(n)])
    z0[0] = 1.0  # x1 starts near center1
    z0[n] = -1.0  # x2 starts near center2

    # Run Frank-Wolfe on the smoothed lifted problem
    print(f"\nRunning Frank-Wolfe on smoothed lifted problem with beta={beta}...")
    fw = FrankWolfe(objective, lmo)
    fw.run(z0, n_steps=500)

    # Extract solution (average of x1 and x2)
    x1_final = fw.x[:n]
    x2_final = fw.x[n:]
    x_recovered = (x1_final + x2_final) / 2

    print(f"\nRecovered solution:")
    print(f"||x1 - x2||_2 = {np.linalg.norm(x1_final - x2_final):.6f} (should be small)")
    print(f"||x_recovered||_1 = {np.linalg.norm(x_recovered, ord=1):.4f}")
    print(f"||x_recovered - x*||_2 = {np.linalg.norm(x_recovered - x_star):.6f}")

    # Evaluate original objective at recovered solution
    f_recovered = 0.5 * np.linalg.norm(x_recovered - y)**2
    print(f"f(x_recovered) = {f_recovered:.6f}")
    print(f"Relative error: {(f_recovered - f_star)/f_star * 100:.4f}%")

    # Plot convergence
    fw.plot_convergence()

    # Plot comparison of solutions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Plot 1: Target and solutions
    indices = range(min(n, 20))
    axes[0].plot(indices, y[:len(indices)], 'ko-', label='Target y', linewidth=2)
    axes[0].plot(indices, x_star[:len(indices)], 'b^-', label='Ground truth x*', linewidth=2)
    axes[0].plot(indices, x_recovered[:len(indices)], 'r*-', label='Recovered x', linewidth=2)
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Comparison of Solutions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: x1 and x2 (should be close)
    axes[1].plot(indices, x1_final[:len(indices)], 'b.-', label='x1', linewidth=2)
    axes[1].plot(indices, x2_final[:len(indices)], 'r.-', label='x2', linewidth=2)
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Value')
    axes[1].set_title('Lifted Variables (x1 and x2)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Diagonal constraint violation over iterations
    n_plot = min(len(fw.func_vals), 500)
    axes[2].semilogy(range(n_plot), fw.gaps[:n_plot], 'g-', linewidth=2)
    axes[2].set_xlabel('Iterations')
    axes[2].set_ylabel('Frank-Wolfe Gap')
    axes[2].set_title('Convergence (FW Gap)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)
