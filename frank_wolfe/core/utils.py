import numpy as np

def line_search(x, d, objective_fn, max_step=1.0, tol=1e-6):
    """
    Perform a line search to find the step size that minimizes the objective function.
    
    Args:
    x (np.ndarray): Current point
    d (np.ndarray): Search direction
    objective_fn (callable): Objective function to minimize
    max_step (float): Maximum allowed step size
    tol (float): Tolerance for improvement in objective value
    
    Returns:
    float: Optimal step size
    """
    left, right = 0, max_step
    gold = (1 + np.sqrt(5)) / 2

    def obj(gamma):
        return objective_fn.evaluate(x + gamma * d)

    while right - left > tol:
        gamma1 = right - (right - left) / gold
        gamma2 = left + (right - left) / gold
        
        if obj(gamma1) < obj(gamma2):
            right = gamma2
        else:
            left = gamma1

    return (left + right) / 2

def align(d, d_hat):
    if np.linalg.norm(d_hat) == 0:
        return -1
    return np.dot(d, d_hat) / (np.linalg.norm(d) * np.linalg.norm(d_hat))

def proj_nonneg(U):
    """
    Projection onto the nonnegative orthon
    """
    return np.maximum(U, 0)

def proj_cube(U):
    return np.clip(U, 0, 1)

def soft_thresh(U, beta):
    return np.sign(U) * np.maximum(np.abs(U) - beta, 0)

def l1_minimal_norm_selection(U):
    """
    U is a numpy array representing a matrix that we are going to compute the minimal norm selection of the l1 subdifferential at.
    """
    return np.sign(U)