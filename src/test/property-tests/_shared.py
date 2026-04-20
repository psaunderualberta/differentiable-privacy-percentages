from hypothesis import HealthCheck, settings
from hypothesis import strategies as st

_EPS = 1.0
_DELTA = 0.126936737507
_P = 0.1
_T_FIXED = 10
_T_FIXED_LONG = 3000

_jax_settings = settings(
    max_examples=25,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)

_eps_st = st.floats(min_value=0.1, max_value=4.0, allow_nan=False, allow_infinity=False)
_delta_st = st.floats(min_value=0.01, max_value=0.9, allow_nan=False, allow_infinity=False)
_p_st = st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False)
_T_st = st.integers(min_value=1, max_value=30)
_weight_val_st = st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)
_schedule_val_st = st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)
