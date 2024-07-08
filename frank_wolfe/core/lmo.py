import numpy as np
from scipy.sparse.linalg import svds, eigsh
from scipy.special import softmax

def general_lmo(gradient, radius, constraint_set):
    if constraint_set == "l1_ball":
        index_flat = np.argmax(np.abs(gradient))
        index_multi = np.unravel_index(index_flat, gradient.shape)
        s = np.zeros_like(gradient)
        s[index_multi] = np.sign(gradient[index_multi])
        return -radius * s
    elif constraint_set == "nuclear_norm_ball":
        u, _, vt = svds(gradient, k=1)
        return -radius * np.outer(u, vt)
    elif constraint_set == "psd_trace":
        _, u = eigsh(gradient, k=1, which='LM')
        return -radius * np.outer(u, u)
    elif constraint_set == "l2_ball":
        gradient_norm = np.linalg.norm(gradient)
        return -radius * gradient / gradient_norm if gradient_norm > 0 else np.zeros_like(gradient)
    elif constraint_set == "softmax_l1_ball":
        s = np.sign(gradient) * softmax(np.abs(gradient))
        return -radius * s
    else:
        raise ValueError(f"Unsupported constraint set: {constraint_set}")

def create_lmo(radius, constraint_set):
    return lambda gradient: general_lmo(gradient, radius, constraint_set)