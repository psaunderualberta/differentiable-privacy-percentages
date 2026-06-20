"""Tests for the chain-control logic in symbolic_regression.py.

Offline, pure-function coverage — no live cluster, no PySR/Julia. Covers
``should_resubmit`` (whether a finished synthesis job resubmits a chain
successor).

The former scratch→mirror tests (``_rsync`` / ``_start_mirror_daemon`` /
``_fit_with_mirror`` / ``should_restore_mirror``) and the ``run_regression``
construction tests were removed: that mirror-daemon layer is no longer part of
symbolic_regression.py, so the tests referenced functions that no longer exist.
"""

from symbolic_regression import should_resubmit

_TIMEOUT = 9900  # 2h45m
_PAD = 600  # 10m


class TestShouldResubmit:
    def test_natural_completion_does_not_resubmit(self):
        # fit() returned well before the timeout window -> synthesis is done.
        assert not should_resubmit(
            elapsed_seconds=100,
            timeout_seconds=_TIMEOUT,
            pad_seconds=_PAD,
            chain_depth=0,
            max_chain_jobs=16,
        )

    def test_timeout_resubmits_when_under_cap(self):
        # fit() ran to (near) the timeout -> not done -> resubmit a successor.
        assert should_resubmit(
            elapsed_seconds=_TIMEOUT,
            timeout_seconds=_TIMEOUT,
            pad_seconds=_PAD,
            chain_depth=3,
            max_chain_jobs=16,
        )

    def test_just_inside_pad_window_resubmits(self):
        # elapsed == timeout - pad is NOT a natural completion (boundary).
        assert should_resubmit(
            elapsed_seconds=_TIMEOUT - _PAD,
            timeout_seconds=_TIMEOUT,
            pad_seconds=_PAD,
            chain_depth=0,
            max_chain_jobs=16,
        )

    def test_just_under_pad_window_completes(self):
        assert not should_resubmit(
            elapsed_seconds=_TIMEOUT - _PAD - 1,
            timeout_seconds=_TIMEOUT,
            pad_seconds=_PAD,
            chain_depth=0,
            max_chain_jobs=16,
        )

    def test_depth_cap_stops_chain(self):
        # Hit the timeout but the chain has reached its depth cap -> stop.
        assert not should_resubmit(
            elapsed_seconds=_TIMEOUT,
            timeout_seconds=_TIMEOUT,
            pad_seconds=_PAD,
            chain_depth=16,
            max_chain_jobs=16,
        )

    def test_last_allowed_depth_resubmits(self):
        assert should_resubmit(
            elapsed_seconds=_TIMEOUT,
            timeout_seconds=_TIMEOUT,
            pad_seconds=_PAD,
            chain_depth=15,
            max_chain_jobs=16,
        )
