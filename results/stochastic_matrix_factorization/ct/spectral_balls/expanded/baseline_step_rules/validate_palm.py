"""Check the PALM update against an independent dense-Hessian reference."""
import hashlib
import json
from pathlib import Path

import numpy as np

from methods import project_factor
from palm_paper import make_solver, parameters


class DenseProblem:
    def __init__(self):
        rng = np.random.default_rng(1234)
        self.train_ids = np.arange(7)
        self.operator = rng.normal(size=(7, 3, 12))
        self.y = rng.normal(size=(7, 3))

    def forward(self, image, ids):
        return self.operator[ids] @ image.ravel()

    def backward(self, observation, ids):
        return np.einsum('vdi,vd->i',self.operator[ids],observation).reshape(4,3)

    def grad(self, U, V, ids, *, block):
        residual = self.forward(U@V.T,ids)-self.y[ids]
        adjoint = self.backward(residual,ids)/residual.size
        return adjoint@V if block==0 else adjoint.T@U


def dense_block(problem,U,V,block):
    active = U if block==0 else V
    columns = []
    for k in range(active.size):
        direction = np.zeros_like(active)
        direction.flat[k] = 1
        image = direction@V.T if block==0 else U@direction.T
        columns.append(problem.forward(image,problem.train_ids).ravel())
    design = np.stack(columns,axis=1)
    residual = problem.forward(U@V.T,problem.train_ids)-problem.y
    gradient = (design.T@residual.ravel()/residual.size).reshape(active.shape)
    return design.T@design/residual.size,gradient


def main():
    p = DenseProblem()
    rng = np.random.default_rng(111)
    U,V = rng.uniform(.2,.6,(4,2)),rng.uniform(.2,.6,(3,2))
    params = parameters(7)
    seed = 100202
    solver = make_solver(p,U,V,(5.,5.),seed=seed,config=params)
    reference_rng = np.random.default_rng(np.random.SeedSequence([seed,0x50414C4D]))
    errors = []
    for iteration in range(4):
        reference_u,reference_v = solver.U.copy(),solver.V.copy()
        expected_curvatures = []
        for block in (0,1):
            active = reference_u if block==0 else reference_v
            hessian,gradient = dense_block(p,reference_u,reference_v,block)
            direction = reference_rng.standard_normal(active.shape).ravel()
            direction /= np.linalg.norm(direction)
            for _ in range(5):
                direction = hessian@direction
                direction /= np.linalg.norm(direction)
            estimate = np.linalg.norm(hessian@direction)
            expected_curvatures.append(estimate)
            candidate = project_factor(active-gradient/estimate,5.,iteration)
            if block==0:reference_u=candidate
            else:reference_v=candidate
        step = solver.step()
        errors.extend([float(np.max(np.abs(solver.U-reference_u))),float(np.max(np.abs(solver.V-reference_v)))])
        np.testing.assert_allclose(solver.U,reference_u,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(solver.V,reference_v,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose([step['curvature_u'],step['curvature_v']],expected_curvatures,atol=1e-12,rtol=1e-12)
        assert step['gradient_calls']==2 and step['curvature_operator_calls']==12 and step['oracle_calls']==14
        assert step['forward_views']==step['backward_views']==14*7
        assert step['curvature_forward_views']==step['curvature_backward_views']==12*7
        assert step['component_gradient_calls']==2*7 and step['projection_calls']==2
        assert step['curvature_sample_views_u']==step['curvature_sample_views_v']==7
    restored=make_solver(p,U,V,(5.,5.),seed=seed,config=params)
    restored.load_state_dict(solver.state_dict())
    for _ in range(3):
        assert solver.step()==restored.step()
        np.testing.assert_array_equal(solver.U,restored.U)
        np.testing.assert_array_equal(solver.V,restored.V)
    for bad in ({'step_coefficient':.5},{'power_iterations':4},{'curvature_batch_size':3}):
        try:parameters(7,bad)
        except ValueError:pass
        else:raise AssertionError('A non-paper PALM configuration was accepted')
    root=Path(__file__).resolve().parent
    names=('palm_paper.py','curvature_reference.py','methods.py','spectral_reference_baseline.py','spectral_reference_core.py')
    result=dict(status='passed',source_sha256={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names},
        checks=['explicit dense normal operators','full-data five-plus-one power estimates','sequential PALM updates with coefficient one',
                'inclusive curvature and gradient accounting','bitwise resumed continuation','invalid parameter rejection'],
        max_reference_update_error=max(errors),toy_complete_iterations=7)
    (root/'palm_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
