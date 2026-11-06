"""Reusable NumPy components for the stochastic-FRAMES experiments.

The classes in this module intentionally have small, explicit interfaces.  A
smooth term supplies values and (possibly batched) gradients, a composite map
supplies its forward operation and Jacobian adjoint, a penalty supplies a
proximal map, and a constraint supplies a linear minimization oracle (LMO).

All inner products used here are Euclidean/Frobenius inner products.  The
quadratic lift is the real-valued map ``U -> U @ U.T`` used by the factorized
correlation benchmark.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional, Protocol, runtime_checkable

import numpy as np


Array = np.ndarray


# ---------------------------------------------------------------------------
# Component contracts
# ---------------------------------------------------------------------------


@runtime_checkable
class SmoothTerm(Protocol):
    """Contract for a differentiable finite-sum or expectation term."""

    def value(self, x: Array) -> float:
        """Return the full smooth objective value at ``x``."""

    def gradient(self, x: Array) -> Array:
        """Return the full smooth gradient at ``x``."""

    def gradient_for_batch(self, x: Array, batch: Any) -> Array:
        """Return the mean stochastic gradient for ``batch``."""


@runtime_checkable
class CompositeMap(Protocol):
    """Contract for the map in a composite term ``g(h(x))``."""

    is_linear: bool

    def forward(self, x: Array) -> Array:
        """Evaluate ``h(x)``."""

    def jacobian_adjoint(self, x: Array, y: Array) -> Array:
        """Evaluate ``J_h(x)^*[y]``."""


@runtime_checkable
class Penalty(Protocol):
    """Contract for a proximable penalty."""

    kind: str

    def prox(self, y: Array, beta: float) -> Array:
        """Evaluate ``prox_{beta g}(y)``."""

    def value(self, y: Array) -> float:
        """Evaluate the penalty, possibly returning ``np.inf``."""


@runtime_checkable
class Constraint(Protocol):
    """Contract for a compact Frank--Wolfe constraint set."""

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return JSON-compatible descriptive metadata."""

    def lmo(self, gradient: Array) -> Array:
        """Minimize the linearization over the constraint set."""

    def contains(self, x: Array) -> bool:
        """Return whether ``x`` is feasible up to numerical tolerance."""


class CallableSmoothTerm:
    """Adapt three ordinary callables to :class:`SmoothTerm`.

    ``batch_gradient`` receives ``(x, batch)``.  If it is omitted, only
    ``batch=None`` is accepted and the full gradient is returned.
    """

    def __init__(
        self,
        value: Callable[[Array], float],
        gradient: Callable[[Array], Array],
        batch_gradient: Optional[Callable[[Array, Any], Array]] = None,
    ) -> None:
        if not callable(value) or not callable(gradient):
            raise TypeError("value and gradient must be callable.")
        if batch_gradient is not None and not callable(batch_gradient):
            raise TypeError("batch_gradient must be callable or None.")
        self._value = value
        self._gradient = gradient
        self._batch_gradient = batch_gradient

    def value(self, x: Array) -> float:
        return float(self._value(np.asarray(x)))

    def gradient(self, x: Array) -> Array:
        return np.asarray(self._gradient(np.asarray(x)))

    def gradient_for_batch(self, x: Array, batch: Any) -> Array:
        if self._batch_gradient is None:
            if batch is not None:
                raise ValueError(
                    "This smooth term has no batched-gradient callable."
                )
            return self.gradient(x)
        return np.asarray(self._batch_gradient(np.asarray(x), batch))


# ---------------------------------------------------------------------------
# Projection and proximal helpers
# ---------------------------------------------------------------------------


def _real_finite_array(value: Any, name: str) -> Array:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise TypeError(f"{name} must be real-valued.")
    try:
        finite = np.all(np.isfinite(array))
    except TypeError as error:
        raise TypeError(f"{name} must be a numeric array.") from error
    if not finite:
        raise ValueError(f"{name} must contain only finite values.")
    return np.asarray(array, dtype=np.result_type(array.dtype, np.float64))


def _nonnegative_scalar(value: Any, name: str) -> float:
    if (
        not np.isscalar(value)
        or isinstance(value, (str, bytes))
        or np.iscomplexobj(value)
    ):
        raise TypeError(f"{name} must be a finite nonnegative real number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(
            f"{name} must be a finite nonnegative real number."
        ) from error
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number.")
    return result


def _positive_scalar(value: Any, name: str) -> float:
    result = _nonnegative_scalar(value, name)
    if result == 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _validate_beta(beta: Any) -> float:
    return _nonnegative_scalar(beta, "beta")


def project_simplex(value: Array, radius: float = 1.0) -> Array:
    """Euclidean projection onto ``{x >= 0: sum(x) = radius}``.

    The entries of an arbitrary-shaped input are treated as one vector and the
    original shape is restored.  ``radius=0`` therefore projects to zero.
    """

    radius = _nonnegative_scalar(radius, "radius")
    array = _real_finite_array(value, "value")
    shape = array.shape
    vector = array.reshape(-1)
    if vector.size == 0:
        raise ValueError("Cannot project an empty array onto a simplex.")
    if radius == 0.0:
        return np.zeros_like(array)

    ordered = np.sort(vector)[::-1]
    shifted_cumsum = np.cumsum(ordered) - radius
    indices = np.arange(1, vector.size + 1, dtype=float)
    active = ordered - shifted_cumsum / indices > 0.0
    # For a positive radius the active set is mathematically nonempty.
    rho = int(np.flatnonzero(active)[-1])
    threshold = shifted_cumsum[rho] / float(rho + 1)
    projected = np.maximum(vector - threshold, 0.0)
    return projected.reshape(shape)


def project_l2_ball(value: Array, radius: float = 1.0) -> Array:
    """Euclidean projection onto a vector L2 ball."""

    radius = _nonnegative_scalar(radius, "radius")
    array = _real_finite_array(value, "value")
    norm = float(np.linalg.norm(array.reshape(-1)))
    if norm <= radius:
        return array.copy()
    if radius == 0.0:
        return np.zeros_like(array)
    return array * (radius / norm)


def project_frobenius_ball(value: Array, radius: float = 1.0) -> Array:
    """Euclidean projection onto a Frobenius-norm ball."""

    return project_l2_ball(value, radius)


def project_box(value: Array, lower: Any = 0.0, upper: Any = 1.0) -> Array:
    """Euclidean projection onto elementwise lower and upper bounds."""

    array = _real_finite_array(value, "value")
    lower_array = _real_finite_array(lower, "lower")
    upper_array = _real_finite_array(upper, "upper")
    try:
        lower_broadcast = np.broadcast_to(lower_array, array.shape)
        upper_broadcast = np.broadcast_to(upper_array, array.shape)
    except ValueError as error:
        raise ValueError("Box bounds must broadcast to the value shape.") from error
    if np.any(lower_broadcast > upper_broadcast):
        raise ValueError("Every lower box bound must be <= its upper bound.")
    return np.minimum(np.maximum(array, lower_broadcast), upper_broadcast)


def _project_nonnegative_l1_ball(value: Array, radius: float) -> Array:
    """Project a nonnegative vector onto ``sum(x) <= radius``."""

    if float(np.sum(value)) <= radius:
        return value.copy()
    return project_simplex(value, radius)


def project_nuclear_ball(value: Array, radius: float = 1.0) -> Array:
    """Euclidean projection of a matrix onto a nuclear-norm ball."""

    radius = _nonnegative_scalar(radius, "radius")
    matrix = _real_finite_array(value, "value")
    if matrix.ndim != 2:
        raise ValueError("Nuclear-ball projection requires a matrix.")
    if matrix.size == 0:
        return matrix.copy()
    left, singular_values, right_t = np.linalg.svd(matrix, full_matrices=False)
    projected_values = _project_nonnegative_l1_ball(singular_values, radius)
    return (left * projected_values) @ right_t


def soft_threshold(value: Array, threshold: float) -> Array:
    """Apply elementwise soft thresholding."""

    threshold = _nonnegative_scalar(threshold, "threshold")
    array = _real_finite_array(value, "value")
    return np.sign(array) * np.maximum(np.abs(array) - threshold, 0.0)


def project_nonnegative(value: Array) -> Array:
    """Project onto the nonnegative orthant."""

    array = _real_finite_array(value, "value")
    return np.maximum(array, 0.0)


def project_unit_diagonal(value: Array, target: Any = 1.0) -> Array:
    """Project a square matrix onto matrices with a prescribed diagonal."""

    matrix = _real_finite_array(value, "value")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Unit-diagonal projection requires a square matrix.")
    target_array = _real_finite_array(target, "target")
    try:
        diagonal = np.broadcast_to(target_array, (matrix.shape[0],))
    except ValueError as error:
        raise ValueError(
            "The diagonal target must be scalar or match the matrix size."
        ) from error
    projected = matrix.copy()
    projected[np.diag_indices(matrix.shape[0])] = diagonal
    return projected


# Verbose aliases make call sites read naturally and retain compatibility with
# early versions of the experiment branch.
project_onto_simplex = project_simplex
simplex_projection = project_simplex
project_onto_l2_ball = project_l2_ball
l2_ball_projection = project_l2_ball
project_onto_frobenius_ball = project_frobenius_ball
frobenius_ball_projection = project_frobenius_ball
project_onto_box = project_box
box_projection = project_box
project_onto_nuclear_ball = project_nuclear_ball
nuclear_ball_projection = project_nuclear_ball


# ---------------------------------------------------------------------------
# Composite maps
# ---------------------------------------------------------------------------


class IdentityMap:
    """The identity composite map."""

    is_linear = True

    def forward(self, x: Array) -> Array:
        return np.asarray(x)

    def adjoint(self, y: Array) -> Array:
        return np.asarray(y)

    def jacobian_adjoint(self, x: Array, y: Array) -> Array:
        del x
        return self.adjoint(y)

    __call__ = forward


class DenseLinearMap:
    """Linear map represented by a two-dimensional dense matrix."""

    is_linear = True

    def __init__(self, matrix: Array) -> None:
        matrix = _real_finite_array(matrix, "matrix")
        if matrix.ndim != 2:
            raise ValueError("A dense linear map requires a 2-D matrix.")
        self.matrix = matrix.copy()

    @property
    def input_dimension(self) -> int:
        return int(self.matrix.shape[1])

    @property
    def output_dimension(self) -> int:
        return int(self.matrix.shape[0])

    def forward(self, x: Array) -> Array:
        return self.matrix @ np.asarray(x)

    def adjoint(self, y: Array) -> Array:
        return self.matrix.T @ np.asarray(y)

    def jacobian_adjoint(self, x: Array, y: Array) -> Array:
        del x
        return self.adjoint(y)

    __call__ = forward


class FiniteDifferenceMap:
    """First forward difference ``x[..., i+1] - x[..., i]``.

    Differences are taken along ``axis``.  ``size`` is optional; when given it
    validates the length of that axis and lets :meth:`adjoint` validate the
    corresponding dual shape without seeing a primal point.
    """

    is_linear = True

    def __init__(
        self,
        size: Optional[int] = None,
        axis: int = -1,
        *,
        dimension: Optional[int] = None,
    ) -> None:
        if dimension is not None:
            if size is not None:
                raise TypeError("Specify size or dimension, not both.")
            size = dimension
        if size is not None:
            if not isinstance(size, (int, np.integer)) or size < 1:
                raise ValueError("size must be a positive integer or None.")
            size = int(size)
        if not isinstance(axis, (int, np.integer)):
            raise TypeError("axis must be an integer.")
        self.size = size
        self.dimension = size
        self.axis = int(axis)

    def _axis_for(self, ndim: int) -> int:
        if ndim == 0:
            raise ValueError("Finite differences require at least one dimension.")
        axis = self.axis + ndim if self.axis < 0 else self.axis
        if axis < 0 or axis >= ndim:
            raise np.AxisError(self.axis, ndim=ndim)
        return axis

    def forward(self, x: Array) -> Array:
        array = np.asarray(x)
        axis = self._axis_for(array.ndim)
        if self.size is not None and array.shape[axis] != self.size:
            raise ValueError(
                f"Expected size {self.size} along axis {axis}, got "
                f"{array.shape[axis]}."
            )
        return np.diff(array, axis=axis)

    @staticmethod
    def _adjoint_with_shape(y: Array, output_shape: tuple, axis: int) -> Array:
        dual = np.asarray(y)
        expected_shape = list(output_shape)
        expected_shape[axis] -= 1
        if tuple(expected_shape) != dual.shape:
            raise ValueError(
                "Finite-difference dual shape mismatch: expected "
                f"{tuple(expected_shape)}, got {dual.shape}."
            )
        dtype = np.result_type(dual.dtype, np.float64)
        result = np.zeros(output_shape, dtype=dtype)
        left = [slice(None)] * len(output_shape)
        right = [slice(None)] * len(output_shape)
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        result[tuple(left)] -= dual
        result[tuple(right)] += dual
        return result

    def adjoint(self, y: Array) -> Array:
        dual = np.asarray(y)
        axis = self._axis_for(dual.ndim)
        output_shape = list(dual.shape)
        output_shape[axis] += 1
        if self.size is not None and output_shape[axis] != self.size:
            raise ValueError(
                f"Expected a dual axis of length {self.size - 1}, got "
                f"{dual.shape[axis]}."
            )
        return self._adjoint_with_shape(dual, tuple(output_shape), axis)

    def jacobian_adjoint(self, x: Array, y: Array) -> Array:
        primal = np.asarray(x)
        axis = self._axis_for(primal.ndim)
        if self.size is not None and primal.shape[axis] != self.size:
            raise ValueError(
                f"Expected size {self.size} along axis {axis}, got "
                f"{primal.shape[axis]}."
            )
        return self._adjoint_with_shape(y, primal.shape, axis)

    __call__ = forward


class QuadraticLiftMap:
    """Experimental real-valued quadratic lift ``h(U) = U U^T``."""

    is_linear = False

    def forward(self, x: Array) -> Array:
        matrix = np.asarray(x)
        if matrix.ndim != 2:
            raise ValueError("The quadratic lift requires a 2-D matrix.")
        return matrix @ matrix.T

    def directional_derivative(self, x: Array, direction: Array) -> Array:
        matrix = np.asarray(x)
        tangent = np.asarray(direction)
        if matrix.ndim != 2 or tangent.shape != matrix.shape:
            raise ValueError("The direction must have the same 2-D shape as x.")
        return tangent @ matrix.T + matrix @ tangent.T

    def jacobian_adjoint(self, x: Array, y: Array) -> Array:
        matrix = np.asarray(x)
        dual = np.asarray(y)
        if matrix.ndim != 2:
            raise ValueError("The quadratic lift requires a 2-D matrix.")
        expected = (matrix.shape[0], matrix.shape[0])
        if dual.shape != expected:
            raise ValueError(
                f"The lift dual must have shape {expected}, got {dual.shape}."
            )
        return (dual + dual.T) @ matrix

    __call__ = forward


# ---------------------------------------------------------------------------
# Penalties
# ---------------------------------------------------------------------------


class ZeroPenalty:
    """The identically-zero penalty."""

    kind = "lipschitz"
    name = "zero"
    lipschitz_constant = 0.0

    def prox(self, y: Array, beta: float) -> Array:
        _validate_beta(beta)
        return np.asarray(y).copy()

    def value(self, y: Array) -> float:
        del y
        return 0.0

    def minimal_norm_subgradient(self, y: Array) -> Array:
        return np.zeros_like(np.asarray(y), dtype=np.result_type(y, np.float64))

    minimal_norm_selection = minimal_norm_subgradient


class L1Penalty:
    """Weighted entrywise L1 penalty ``weight * ||y||_1``."""

    kind = "lipschitz"
    name = "l1"

    def __init__(
        self,
        weight: float = 1.0,
        *,
        lam: Optional[float] = None,
        lambda_: Optional[float] = None,
    ) -> None:
        supplied = [value for value in (lam, lambda_) if value is not None]
        if len(supplied) > 1:
            raise TypeError("Specify at most one of lam and lambda_.")
        if supplied:
            if weight != 1.0:
                raise TypeError("Specify weight or lam/lambda_, not both.")
            weight = supplied[0]
        self.weight = _nonnegative_scalar(weight, "weight")
        self.lam = self.weight

    def prox(self, y: Array, beta: float) -> Array:
        beta = _validate_beta(beta)
        return soft_threshold(y, beta * self.weight)

    def value(self, y: Array) -> float:
        array = _real_finite_array(y, "y")
        return float(self.weight * np.sum(np.abs(array)))

    def minimal_norm_subgradient(self, y: Array) -> Array:
        array = _real_finite_array(y, "y")
        return self.weight * np.sign(array)

    minimal_norm_selection = minimal_norm_subgradient


class MaxPenalty:
    """Weighted maximum penalty ``weight * max_j y_j``."""

    kind = "lipschitz"
    name = "max"

    def __init__(self, weight: float = 1.0, *, kappa: Optional[float] = None) -> None:
        if kappa is not None:
            if weight != 1.0:
                raise TypeError("Specify weight or kappa, not both.")
            weight = kappa
        self.weight = _nonnegative_scalar(weight, "weight")
        self.kappa = self.weight
        self.lipschitz_constant = self.weight

    def prox(self, y: Array, beta: float) -> Array:
        beta = _validate_beta(beta)
        array = _real_finite_array(y, "y")
        scale = beta * self.weight
        if scale == 0.0:
            return array.copy()
        return array - scale * project_simplex(array / scale)

    def value(self, y: Array) -> float:
        array = _real_finite_array(y, "y")
        if array.size == 0:
            raise ValueError("The maximum penalty is undefined on an empty array.")
        return float(self.weight * np.max(array))

    def minimal_norm_subgradient(self, y: Array) -> Array:
        array = _real_finite_array(y, "y")
        if array.size == 0:
            raise ValueError("The maximum penalty is undefined on an empty array.")
        result = np.zeros_like(array)
        active = array == np.max(array)
        result[active] = self.weight / int(np.count_nonzero(active))
        return result

    minimal_norm_selection = minimal_norm_subgradient


class ProjectionIndicator:
    """Indicator penalty defined by a Euclidean projection.

    ``contains`` is optional.  When omitted, feasibility is detected by
    comparing a point with its projection using ``atol``.
    """

    kind = "indicator"

    def __init__(
        self,
        projection: Callable[[Array], Array],
        contains: Optional[Callable[[Array], bool]] = None,
        *,
        name: str = "indicator",
        atol: float = 1e-10,
    ) -> None:
        if not callable(projection):
            raise TypeError("projection must be callable.")
        if contains is not None and not callable(contains):
            raise TypeError("contains must be callable or None.")
        self._projection = projection
        self._contains = contains
        self.name = str(name)
        self.atol = _nonnegative_scalar(atol, "atol")

    def prox(self, y: Array, beta: float) -> Array:
        _validate_beta(beta)
        return np.asarray(self._projection(np.asarray(y)))

    def contains(self, y: Array) -> bool:
        array = np.asarray(y)
        if self._contains is not None:
            return bool(self._contains(array))
        projected = np.asarray(self._projection(array))
        return projected.shape == array.shape and bool(
            np.allclose(projected, array, rtol=0.0, atol=self.atol)
        )

    def value(self, y: Array) -> float:
        return 0.0 if self.contains(y) else float("inf")


class NonnegativeIndicator:
    """Indicator of the elementwise nonnegative orthant."""

    kind = "indicator"
    name = "nonnegative"

    def __init__(self, *, atol: float = 1e-10) -> None:
        self.atol = _nonnegative_scalar(atol, "atol")

    def prox(self, y: Array, beta: float) -> Array:
        _validate_beta(beta)
        return project_nonnegative(y)

    def contains(self, y: Array) -> bool:
        array = np.asarray(y)
        return bool(
            not np.iscomplexobj(array)
            and np.all(np.isfinite(array))
            and np.all(array >= -self.atol)
        )

    def value(self, y: Array) -> float:
        return 0.0 if self.contains(y) else float("inf")


class UnitDiagonalIndicator:
    """Indicator of square matrices whose diagonal equals ``target``."""

    kind = "indicator"
    name = "unit_diagonal"

    def __init__(self, target: Any = 1.0, *, atol: float = 1e-10) -> None:
        self.target = _real_finite_array(target, "target").copy()
        self.atol = _nonnegative_scalar(atol, "atol")

    def prox(self, y: Array, beta: float) -> Array:
        _validate_beta(beta)
        return project_unit_diagonal(y, self.target)

    def contains(self, y: Array) -> bool:
        matrix = np.asarray(y)
        if (
            np.iscomplexobj(matrix)
            or matrix.ndim != 2
            or matrix.shape[0] != matrix.shape[1]
            or not np.all(np.isfinite(matrix))
        ):
            return False
        try:
            target = np.broadcast_to(self.target, (matrix.shape[0],))
        except ValueError:
            return False
        return bool(
            np.allclose(np.diag(matrix), target, rtol=0.0, atol=self.atol)
        )

    def value(self, y: Array) -> float:
        return 0.0 if self.contains(y) else float("inf")


# ---------------------------------------------------------------------------
# Frank--Wolfe constraints
# ---------------------------------------------------------------------------


class SimplexConstraint:
    """Nonnegative simplex with a prescribed total mass."""

    def __init__(
        self,
        radius: float = 1.0,
        *,
        dimension: Optional[int] = None,
        atol: float = 1e-10,
    ) -> None:
        self.radius = _positive_scalar(radius, "radius")
        if dimension is not None:
            if not isinstance(dimension, (int, np.integer)) or dimension < 1:
                raise ValueError("dimension must be a positive integer or None.")
            dimension = int(dimension)
        self.dimension = dimension
        self.atol = _nonnegative_scalar(atol, "atol")

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "kind": "simplex",
            "radius": self.radius,
            "dimension": self.dimension,
        }

    def lmo(self, gradient: Array) -> Array:
        grad = _real_finite_array(gradient, "gradient")
        if grad.size == 0:
            raise ValueError("A simplex LMO requires a nonempty gradient.")
        if self.dimension is not None and grad.size != self.dimension:
            raise ValueError(
                f"Expected {self.dimension} gradient entries, got {grad.size}."
            )
        result = np.zeros_like(grad)
        flat_gradient = grad.reshape(-1)
        flat_result = result.reshape(-1)
        if np.all(flat_gradient == 0.0):
            flat_result[:] = self.radius / flat_result.size
        else:
            flat_result[int(np.argmin(flat_gradient))] = self.radius
        return result

    def contains(self, x: Array) -> bool:
        array = np.asarray(x)
        if np.iscomplexobj(array) or array.size == 0:
            return False
        try:
            finite = np.all(np.isfinite(array))
        except TypeError:
            return False
        if not finite or (self.dimension is not None and array.size != self.dimension):
            return False
        return bool(
            np.all(array >= -self.atol)
            and abs(float(np.sum(array)) - self.radius) <= self.atol
        )

    def project(self, x: Array) -> Array:
        return project_simplex(x, self.radius)


class L2BallConstraint:
    """Euclidean ball for vector-shaped decision variables."""

    norm_name = "l2"

    def __init__(self, radius: float = 1.0, *, atol: float = 1e-10) -> None:
        self.radius = _nonnegative_scalar(radius, "radius")
        self.atol = _nonnegative_scalar(atol, "atol")

    @property
    def metadata(self) -> Dict[str, Any]:
        return {"kind": f"{self.norm_name}_ball", "radius": self.radius}

    def lmo(self, gradient: Array) -> Array:
        grad = _real_finite_array(gradient, "gradient")
        norm = float(np.linalg.norm(grad.reshape(-1)))
        if norm == 0.0 or self.radius == 0.0:
            return np.zeros_like(grad)
        return -self.radius * grad / norm

    def contains(self, x: Array) -> bool:
        array = np.asarray(x)
        if np.iscomplexobj(array):
            return False
        try:
            norm = float(np.linalg.norm(array.reshape(-1)))
        except (TypeError, ValueError):
            return False
        return bool(np.isfinite(norm) and norm <= self.radius + self.atol)

    def project(self, x: Array) -> Array:
        return project_l2_ball(x, self.radius)


class FrobeniusBallConstraint(L2BallConstraint):
    """Frobenius ball for matrix-shaped decision variables."""

    norm_name = "frobenius"

    def lmo(self, gradient: Array) -> Array:
        grad = np.asarray(gradient)
        if grad.ndim != 2:
            raise ValueError("A Frobenius-ball LMO requires a matrix gradient.")
        return super().lmo(grad)

    def contains(self, x: Array) -> bool:
        return np.asarray(x).ndim == 2 and super().contains(x)

    def project(self, x: Array) -> Array:
        matrix = np.asarray(x)
        if matrix.ndim != 2:
            raise ValueError("Frobenius-ball projection requires a matrix.")
        return project_frobenius_ball(matrix, self.radius)


def _metadata_value(value: Array) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    return array.tolist()


class BoxConstraint:
    """Elementwise box constraint ``lower <= x <= upper``."""

    def __init__(
        self,
        lower: Any = 0.0,
        upper: Any = 1.0,
        *,
        atol: float = 1e-10,
    ) -> None:
        self.lower = _real_finite_array(lower, "lower").copy()
        self.upper = _real_finite_array(upper, "upper").copy()
        try:
            lower_broadcast, upper_broadcast = np.broadcast_arrays(
                self.lower, self.upper
            )
        except ValueError as error:
            raise ValueError("Box bounds must be mutually broadcastable.") from error
        if np.any(lower_broadcast > upper_broadcast):
            raise ValueError("Every lower box bound must be <= its upper bound.")
        self.atol = _nonnegative_scalar(atol, "atol")

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "kind": "box",
            "lower": _metadata_value(self.lower),
            "upper": _metadata_value(self.upper),
        }

    def _bounds_for(self, shape: tuple) -> tuple:
        try:
            lower = np.broadcast_to(self.lower, shape)
            upper = np.broadcast_to(self.upper, shape)
        except ValueError as error:
            raise ValueError("Box bounds must broadcast to the decision shape.") from error
        return lower, upper

    def lmo(self, gradient: Array) -> Array:
        grad = _real_finite_array(gradient, "gradient")
        lower, upper = self._bounds_for(grad.shape)
        zero_choice = np.minimum(np.maximum(np.zeros_like(grad), lower), upper)
        return np.where(grad > 0.0, lower, np.where(grad < 0.0, upper, zero_choice))

    def contains(self, x: Array) -> bool:
        array = np.asarray(x)
        if np.iscomplexobj(array):
            return False
        try:
            lower, upper = self._bounds_for(array.shape)
            finite = np.all(np.isfinite(array))
        except (TypeError, ValueError):
            return False
        return bool(
            finite
            and np.all(array >= lower - self.atol)
            and np.all(array <= upper + self.atol)
        )

    def project(self, x: Array) -> Array:
        return project_box(x, self.lower, self.upper)


class NuclearNormBallConstraint:
    """Nuclear-norm ball for matrix-shaped decision variables."""

    def __init__(self, radius: float = 1.0, *, atol: float = 1e-10) -> None:
        self.radius = _nonnegative_scalar(radius, "radius")
        self.atol = _nonnegative_scalar(atol, "atol")

    @property
    def metadata(self) -> Dict[str, Any]:
        return {"kind": "nuclear_norm_ball", "radius": self.radius}

    def lmo(self, gradient: Array) -> Array:
        grad = _real_finite_array(gradient, "gradient")
        if grad.ndim != 2:
            raise ValueError("A nuclear-ball LMO requires a matrix gradient.")
        if grad.size == 0 or self.radius == 0.0 or np.all(grad == 0.0):
            return np.zeros_like(grad)
        left, _, right_t = np.linalg.svd(grad, full_matrices=False)
        return -self.radius * np.outer(left[:, 0], right_t[0, :])

    def contains(self, x: Array) -> bool:
        matrix = np.asarray(x)
        if np.iscomplexobj(matrix) or matrix.ndim != 2:
            return False
        try:
            singular_values = np.linalg.svd(matrix, compute_uv=False)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            return False
        norm = float(np.sum(singular_values))
        return bool(np.isfinite(norm) and norm <= self.radius + self.atol)

    def project(self, x: Array) -> Array:
        return project_nuclear_ball(x, self.radius)


# Compatibility aliases used by concise benchmark definitions.
SmoothTermContract = SmoothTerm
CompositeMapContract = CompositeMap
PenaltyContract = Penalty
ConstraintContract = Constraint

IdentityCompositeMap = IdentityMap
LinearMap = DenseLinearMap
FirstDifferenceMap = FiniteDifferenceMap
QuadraticLift = QuadraticLiftMap

NoPenalty = ZeroPenalty
IndicatorPenalty = ProjectionIndicator
NonnegativeIndicatorPenalty = NonnegativeIndicator
UnitDiagonalIndicatorPenalty = UnitDiagonalIndicator
DiagonalOneIndicator = UnitDiagonalIndicator

Simplex = SimplexConstraint
L2Ball = L2BallConstraint
FrobeniusBall = FrobeniusBallConstraint
Box = BoxConstraint
NuclearNormBall = NuclearNormBallConstraint
NuclearBallConstraint = NuclearNormBallConstraint


__all__ = [
    "Array",
    "SmoothTerm",
    "SmoothTermContract",
    "CompositeMap",
    "CompositeMapContract",
    "Penalty",
    "PenaltyContract",
    "Constraint",
    "ConstraintContract",
    "CallableSmoothTerm",
    "project_simplex",
    "project_onto_simplex",
    "simplex_projection",
    "project_l2_ball",
    "project_onto_l2_ball",
    "l2_ball_projection",
    "project_frobenius_ball",
    "project_onto_frobenius_ball",
    "frobenius_ball_projection",
    "project_box",
    "project_onto_box",
    "box_projection",
    "project_nuclear_ball",
    "project_onto_nuclear_ball",
    "nuclear_ball_projection",
    "project_nonnegative",
    "project_unit_diagonal",
    "soft_threshold",
    "IdentityMap",
    "IdentityCompositeMap",
    "DenseLinearMap",
    "LinearMap",
    "FiniteDifferenceMap",
    "FirstDifferenceMap",
    "QuadraticLiftMap",
    "QuadraticLift",
    "ZeroPenalty",
    "NoPenalty",
    "L1Penalty",
    "MaxPenalty",
    "ProjectionIndicator",
    "IndicatorPenalty",
    "NonnegativeIndicator",
    "NonnegativeIndicatorPenalty",
    "UnitDiagonalIndicator",
    "UnitDiagonalIndicatorPenalty",
    "DiagonalOneIndicator",
    "SimplexConstraint",
    "Simplex",
    "L2BallConstraint",
    "L2Ball",
    "FrobeniusBallConstraint",
    "FrobeniusBall",
    "BoxConstraint",
    "Box",
    "NuclearNormBallConstraint",
    "NuclearNormBall",
    "NuclearBallConstraint",
]
