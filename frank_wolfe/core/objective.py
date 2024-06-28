import numpy as np

class ObjectiveFunction:
    def __init__(self):
        self.lipschitz = None
    
    def evaluate(self, x):
        raise NotImplementedError
    
    def gradient(self, x):
        raise NotImplementedError
    
    def moreau_gradient(self, x, beta):
        raise NotImplementedError
    
    def subgradient(self, x):
        raise NotImplementedError
    
    def mismatch_gradient(self, x):
        raise NotImplementedError
    
    def linear_operator(self, x):
        raise NotImplementedError
    
    def linear_operator_adjoint(self, x):
        raise NotImplementedError
    
    def minimal_norm_selection(self, x):
        raise NotImplementedError