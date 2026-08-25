import numpy as np

class ObjectiveFunction:
    def __init__(self):
        self.lipschitz = None
    
    def evaluate(self, x):
        raise NotImplementedError
    
    def gradient(self, x):
        raise NotImplementedError

    def stochastic_gradient(self, x):
        """Return one stochastic gradient sample of the smooth term.

        The exact gradient is a valid zero-variance stochastic oracle, so this
        default keeps deterministic objectives usable by stochastic methods.
        Stochastic objectives should override this method.
        """
        return self.gradient(x)
    
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

    def linear_operator_adjoint_at(self, x, y):
        """Apply the operator adjoint at ``x``.

        Linear composite maps can ignore the point, so the default delegates
        to :meth:`linear_operator_adjoint`.  Objectives with a nonlinear map
        may override this hook with the adjoint of its Jacobian at ``x``.
        """
        del x
        return self.linear_operator_adjoint(y)
    
    def minimal_norm_selection(self, x):
        raise NotImplementedError
