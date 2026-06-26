import pytest

from util.grad_guard import ConsecutiveNonfiniteGuard, NonfiniteGradError


def test_raises_after_limit_consecutive_nonfinite_steps():
    guard = ConsecutiveNonfiniteGuard(limit=3)
    guard.update(is_nonfinite=True)
    guard.update(is_nonfinite=True)
    with pytest.raises(NonfiniteGradError):
        guard.update(is_nonfinite=True)


def test_finite_step_resets_the_streak():
    guard = ConsecutiveNonfiniteGuard(limit=3)
    guard.update(is_nonfinite=True)
    guard.update(is_nonfinite=True)
    guard.update(is_nonfinite=False)  # resets
    # Two more non-finite steps must not trip, since they are not consecutive
    # with the earlier pair.
    guard.update(is_nonfinite=True)
    guard.update(is_nonfinite=True)


def test_non_positive_limit_disables_the_guard():
    guard = ConsecutiveNonfiniteGuard(limit=0)
    for _ in range(100):
        guard.update(is_nonfinite=True)  # must never raise when disabled
