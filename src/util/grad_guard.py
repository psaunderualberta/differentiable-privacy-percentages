"""Guard against a fully-diverged outer loop silently no-op'ing on zeroed grads.

The outer-loop robustness chain (``clip_by_global_norm -> zero_nans``) rewrites a
non-finite schedule gradient into a no-op so a *rare* divergent inner DP-SGD run
can't kill a multi-hour run. But if *every* step is non-finite — e.g. an inner
optimiser configuration that diverges deterministically — the chain zeros every
update and the schedule never trains, wasting the whole run. This guard counts
consecutive non-finite outer steps and aborts loudly once they exceed a limit.
"""


class NonfiniteGradError(RuntimeError):
    """Raised when the outer gradient has been non-finite for too many steps."""


class ConsecutiveNonfiniteGuard:
    """Track consecutive non-finite outer-loop gradients and abort past ``limit``."""

    def __init__(self, limit: int):
        self.limit = limit
        self.streak = 0

    def update(self, is_nonfinite: bool) -> None:
        self.streak = self.streak + 1 if is_nonfinite else 0
        if self.limit > 0 and self.streak >= self.limit:
            raise NonfiniteGradError(
                f"Outer gradient non-finite for {self.streak} consecutive steps "
                f"(limit {self.limit}); the inner DP-SGD is diverging every step and "
                f"the schedule is not training. Aborting."
            )
