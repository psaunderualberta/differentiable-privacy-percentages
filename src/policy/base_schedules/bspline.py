import jax.lax as jlax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from scipy.interpolate import BSpline

from policy.base_schedules._registry import register
from policy.base_schedules.abstract import AbstractSchedule
from policy.base_schedules.config import BSplineScheduleConfig


def _compute_bspline_basis(T: int, num_control_points: int, degree: int) -> np.ndarray:
    """Compute clamped-uniform B-spline basis matrix of shape (T, num_control_points).

    Requires num_control_points >= degree + 1.
    """
    n, d = num_control_points, degree
    if n < d + 1:
        raise ValueError(f"num_control_points ({n}) must be >= degree + 1 ({d + 1}).")
    num_interior = n - d - 1
    interior = np.linspace(0, 1, num_interior + 2)[1:-1] if num_interior > 0 else np.array([])
    knots = np.concatenate([[0.0] * (d + 1), interior, [1.0] * (d + 1)])

    t_vals = np.linspace(0, 1, T)
    basis = np.zeros((T, n), dtype=np.float32)
    for i in range(n):
        c = np.zeros(n)
        c[i] = 1.0
        basis[:, i] = BSpline(knots, c, d)(t_vals).astype(np.float32)
    return basis


@register(BSplineScheduleConfig)
class BSplineSchedule(AbstractSchedule):
    """B-spline base schedule with positivity-constrained control points.

    Learnable parameters are ``control_points`` (in unconstrained space).
    ``get_valid_schedule()`` applies softplus to the control points then
    multiplies by the precomputed basis matrix.

    Attributes:
        control_points: Unconstrained learnable array of shape (num_control_points,).
        basis: Fixed B-spline basis matrix of shape (T, num_control_points).
        basis_pinv: Fixed pseudo-inverse of basis, shape (num_control_points, T).
    """

    control_points: Array
    basis: Array
    basis_pinv: Array

    def __init__(
        self,
        control_points: Array,
        basis: Array,
        basis_pinv: Array,
    ):
        self.control_points = control_points
        self.basis = basis
        self.basis_pinv = basis_pinv

    @classmethod
    def from_config(cls, conf: BSplineScheduleConfig, T: int) -> "BSplineSchedule":
        basis_np = _compute_bspline_basis(T, conf.num_control_points, conf.degree)
        basis = jnp.array(basis_np)
        basis_pinv = jnp.array(np.linalg.pinv(basis_np))

        control_points = jnp.full((conf.num_control_points,), conf.init_value, dtype=jnp.float32)
        return cls(control_points, basis, basis_pinv)

    def get_valid_schedule(self) -> Array:
        return jlax.stop_gradient(self.basis) @ self.control_points

    def get_raw_schedule(self) -> Array:
        return jlax.stop_gradient(self.basis) @ self.control_points

    @classmethod
    def from_projection(
        cls,
        schedule: "BSplineSchedule",
        projection: Array,
    ) -> "BSplineSchedule":
        # Least-squares fit: find pos_cp ≈ pinv(basis) @ projection
        pos_cp = jlax.stop_gradient(schedule.basis_pinv) @ projection
        return BSplineSchedule(
            control_points=pos_cp,
            basis=schedule.basis,
            basis_pinv=schedule.basis_pinv,
        )
