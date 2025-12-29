"""
Step size strategies for Frank-Wolfe algorithms.

This module provides both fixed step size schedules (functions of iteration count)
and adaptive step size strategies (classes that compute step sizes based on problem data).
"""

import numpy as np
from frank_wolfe.core.utils import segment_search


# Fixed step size schedules (functions of iteration k, starting at k=0)

def diminishing_2_over_k2(k):
    """
    Standard Frank-Wolfe step size: 2/(k+2).

    First step (k=0) returns 1.0, guaranteeing full step toward LMO vertex.

    Args:
        k: Iteration number (0-indexed)

    Returns:
        Step size gamma_k = 2/(k+2)
    """
    return 2.0 / (k + 2)


def diminishing_2_over_k3(k):
    """
    Alternative diminishing step size: 2/(k+3).

    First step (k=0) returns 2/3, more conservative than standard schedule.

    Args:
        k: Iteration number (0-indexed)

    Returns:
        Step size gamma_k = 2/(k+3)
    """
    return 2.0 / (k + 3)


def polynomial_decay(p):
    """
    Create a polynomial decay step size function: 1/(k+1)^p.

    Args:
        p: Decay exponent in (0, 1). Larger p means faster decay.

    Returns:
        Step size function that takes iteration k and returns 1/(k+1)^p
    """
    def step_fn(k):
        return 1.0 / ((k + 1) ** p)
    return step_fn


# Adaptive step sizes (classes that take problem-dependent info)

class ShortStep:
    """
    Short step (a.k.a. Lipschitz step) using the Lipschitz constant.

    Computes: gamma = min(gap / (L * ||d||^2), gamma_max)

    This step size ensures sufficient decrease in the objective function
    without requiring a line search.
    """

    def __init__(self, lipschitz, gamma_max=1.0):
        """
        Initialize short step calculator.

        Args:
            lipschitz: Lipschitz constant L of the gradient
            gamma_max: Maximum allowed step size (default: 1.0)
        """
        self.L = lipschitz
        self.gamma_max = gamma_max

    def __call__(self, gap, direction):
        """
        Compute short step size.

        Args:
            gap: Frank-Wolfe gap at current point
            direction: Search direction vector

        Returns:
            Step size gamma in [0, gamma_max]
        """
        direction_norm_sq = np.linalg.norm(direction) ** 2
        if direction_norm_sq < 1e-15:
            return self.gamma_max
        return min(gap / (self.L * direction_norm_sq), self.gamma_max)


class AlignShortStep:
    """
    Aligned short step for boosted Frank-Wolfe.

    Computes: gamma = min(align_g * ||grad|| / (L * ||g||), gamma_max)

    Uses alignment between gradient and search direction for better step sizing.
    """

    def __init__(self, lipschitz, gamma_max=1.0):
        """
        Initialize aligned short step calculator.

        Args:
            lipschitz: Lipschitz constant L of the gradient
            gamma_max: Maximum allowed step size (default: 1.0)
        """
        self.L = lipschitz
        self.gamma_max = gamma_max

    def __call__(self, align_g, grad_norm, direction_norm):
        """
        Compute aligned short step size.

        Args:
            align_g: Alignment value between gradient and direction
            grad_norm: Norm of the gradient
            direction_norm: Norm of the search direction

        Returns:
            Step size gamma in [0, gamma_max]
        """
        if direction_norm < 1e-15:
            return self.gamma_max
        return min(align_g * grad_norm / (self.L * direction_norm), self.gamma_max)


class LineSearch:
    """
    Line search step size using golden section method.

    Performs exact line search to find the minimum of the objective
    along the line segment from x to x + direction.
    """

    def __init__(self, objective, tol=1e-6):
        """
        Initialize line search.

        Args:
            objective: Objective function with evaluate() and gradient() methods
            tol: Tolerance for line search convergence (default: 1e-6)
        """
        self.objective = objective
        self.tol = tol

    def __call__(self, x, direction):
        """
        Perform line search to find optimal step size.

        Args:
            x: Current point
            direction: Search direction (not necessarily normalized)

        Returns:
            Tuple (x_new, gamma) where x_new is the new point and gamma is the step size
        """
        return segment_search(self.objective, x, x + direction, tol=self.tol)


class DemyanovRubinovStep:
    """
    Demyanov-Rubinov step size for away-step Frank-Wolfe.

    Handles both forward and away steps with appropriate maximum step sizes.
    """

    def __init__(self, step_type='short', lipschitz=None, objective=None, tol=1e-6):
        """
        Initialize Demyanov-Rubinov step size.

        Args:
            step_type: Either 'short' or 'linesearch'
            lipschitz: Lipschitz constant (required if step_type='short')
            objective: Objective function (required if step_type='linesearch')
            tol: Tolerance for line search (default: 1e-6)
        """
        self.step_type = step_type
        if step_type == 'short':
            if lipschitz is None:
                raise ValueError("Lipschitz constant required for short step")
            self.lipschitz = lipschitz
        elif step_type == 'linesearch':
            if objective is None:
                raise ValueError("Objective function required for line search")
            self.objective = objective
            self.tol = tol
        else:
            raise ValueError(f"Unknown step type: {step_type}")

    def __call__(self, x, direction, gap, gamma_max):
        """
        Compute step size respecting gamma_max constraint.

        Args:
            x: Current point
            direction: Search direction
            gap: Frank-Wolfe gap
            gamma_max: Maximum allowed step size

        Returns:
            Step size gamma in [0, gamma_max]
        """
        if self.step_type == 'short':
            direction_norm_sq = np.linalg.norm(direction) ** 2
            if direction_norm_sq < 1e-15:
                return gamma_max
            gamma = min(gap / (self.lipschitz * direction_norm_sq), gamma_max)
        else:  # linesearch
            _, gamma = segment_search(self.objective, x, x + direction, tol=self.tol)
            gamma = min(gamma, gamma_max)

        return gamma
