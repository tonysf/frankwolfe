import numpy as np

def segment_search(self, x, y, tol=1e-15, stepsize=True):
    """
    Minimizes f over [x, y], i.e., f(x+gamma*(y-x)) as a function of scalar gamma in [0,1]
    """
    # restrict segment of search to [x, y]
    d = (y-x).copy()
    left, right = x.copy(), y.copy()
    
    # NOTE: REWRITE THIS NOT TO USE GRADIENT
    # if the minimum is at an endpoint
    if np.dot(d, self.objective.gradient(x))*np.dot(d, self.objective.gradient(y)) >= 0:
        if self.objective.evaluate(y) <= self.objective.evaluate(x):
            return y, 1
        else:
            return x, 0
    
    # apply golden-section method to segment
    gold = (1+np.sqrt(5))/2
    improv = np.inf
    while improv > tol:
        old_left, old_right = left, right
        new = left+(right-left)/(1+gold)
        probe = new+(right-new)/2
        if self.objective.evaluate(probe) <= self.objective.evaluate(new):
            left, right = new, right
        else:
            left, right = left, probe
        improv = np.linalg.norm(self.objective.evaluate(right)-self.objective.evaluate(old_right))+np.linalg.norm(self.objective.evaluate(left)-self.objective.evaluate(old_left))
    
    x_min = (left+right)/2
    
    # compute step size gamma
    gamma = 0
    if stepsize == True:
        for i in range(len(d)):
            if d[i] != 0:
                gamma = (x_min[i]-x[i])/d[i]
                break
    
    return x_min, gamma

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
    if np.linalg.norm(d_hat) < 1e-15:
        return -1
    return np.dot(d, d_hat) / (np.linalg.norm(d) * np.linalg.norm(d_hat))

def proj_nonneg(U, beta):
    """
    Projection onto the nonnegative orthant
    """
    return np.maximum(U, 0)

def proj_cube(U, beta):
    """
    Projection onto the unit cube 
    """
    return np.clip(U, 0, 1)

def soft_thresh(U, beta):
    return np.sign(U) * np.maximum(np.abs(U) - beta, 0)

def l1_minimal_norm_selection(U):
    """
    U is a numpy array representing a matrix that we are going to compute the minimal norm selection of the l1 subdifferential at.
    """
    return np.sign(U)