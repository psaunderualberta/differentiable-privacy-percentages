"""Retry the mandatory-online W&B calls that happen before ``wandb.init``.

Why this exists
---------------
``--wandb-conf.mode offline`` does not make *startup* offline-capable.  Two
calls must reach ``api.wandb.ai`` before anything is logged:

  1. ``conf/singleton_conf.get_wandb_run_conf`` — fetch the source run's config
     when ``restart_run_id`` / ``checkpoint_run_id`` is set.
  2. ``util/checkpointing.load_checkpoint`` — download the checkpoint artifact,
     and (on failure) probe whether one exists remotely.

Neither retried.  On 2026-07-30 a ~65-second outage of the ComputeCanada HTTP
proxy (503 tunnel failures and 20 s read timeouts) therefore killed 30 FirSweep
jobs outright: each died before ``wandb.init``, leaving a pristine seed run with
``_runtime: 0``, and nothing resubmits them — the chain-resubmit logic lives
inside ``main.py`` and ``--dependency=singleton`` does not requeue.

The defaults here keep retrying for longer than that outage lasted, so the same
blip costs a couple of minutes of startup instead of a whole job.

What is *not* retried
---------------------
Only failures that look like a network blip (see `is_transient`).  A missing
run or artifact raises ``ValueError``, and a bad argument raises ``UsageError``;
those fail fast, so the common "first job of a chain, no checkpoint yet" path
still starts immediately rather than burning the retry window on a certainty.
"""

import random
import time
from collections.abc import Callable
from typing import TypeVar

import requests
import urllib3

import wandb

T = TypeVar("T")

# Sum of the default delays is ~93 s, comfortably past the ~65 s outage that
# caused the FirSweep losses; the per-request timeouts add to that window.
DEFAULT_ATTEMPTS = 6
DEFAULT_BASE_DELAY_SECS = 3.0
DEFAULT_MAX_DELAY_SECS = 60.0
DEFAULT_JITTER = 0.25

# Depth cap for the __cause__/__context__ walk: enough for wandb's wrapping
# (raw OSError → urllib3 → requests → AuthenticationError) with room to spare,
# and it bounds the walk even if the chain is malformed.
_MAX_CHAIN_DEPTH = 10

_TRANSIENT_TYPES: tuple[type[BaseException], ...] = (
    requests.exceptions.RequestException,  # ProxyError, ReadTimeout, ConnectionError
    urllib3.exceptions.HTTPError,  # the layer underneath requests
    wandb.errors.AuthenticationError,  # raised when the API-key check can't reach the server
    wandb.errors.CommError,
    TimeoutError,
    ConnectionError,  # the builtin, not requests'
)


def is_transient(exc: BaseException) -> bool:
    """True if ``exc`` — or anything it wraps — looks like a network blip.

    The chain walk matters: wandb reports a timed-out API-key check as
    ``AuthenticationError`` and ``load_checkpoint`` re-raises fetch failures as
    ``RuntimeError``, so the outermost type alone is a poor signal.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    for _ in range(_MAX_CHAIN_DEPTH):
        if current is None or id(current) in seen:
            return False
        if isinstance(current, _TRANSIENT_TYPES):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def backoff_delays(
    attempts: int,
    base_delay: float,
    max_delay: float,
) -> list[float]:
    """The un-jittered wait before each retry: ``base * 2**i``, capped.

    Returns ``attempts - 1`` delays — nothing is slept after the final attempt.
    """
    return [min(max_delay, base_delay * 2**i) for i in range(max(0, attempts - 1))]


def retry_transient(
    fn: Callable[[], T],
    *,
    what: str,
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY_SECS,
    max_delay: float = DEFAULT_MAX_DELAY_SECS,
    jitter: float = DEFAULT_JITTER,
) -> T:
    """Call ``fn``, retrying transient failures with exponential backoff.

    Parameters
    ----------
    fn:
        The network call.  Retried as a whole, so it should include building
        the ``wandb.Api()`` object — that is where the API-key verification
        (and therefore the failure seen in the FirSweep logs) happens.
    what:
        Short description used in the progress messages, e.g.
        ``"fetch config for run 'abc123'"``.
    jitter:
        Fractional spread applied to each delay, so a burst of jobs launched
        seconds apart does not retry in lockstep.

    Raises
    ------
    The last exception, unchanged, once the attempts are exhausted — and any
    non-transient exception immediately, without retrying.
    """
    delays = backoff_delays(attempts, base_delay, max_delay)

    for attempt, delay in enumerate(delays, start=1):
        try:
            return fn()
        except Exception as e:
            if not is_transient(e):
                raise
            wait = delay * random.uniform(1.0 - jitter, 1.0 + jitter) if jitter else delay
            print(
                f"Transient W&B failure while trying to {what} "
                f"(attempt {attempt}/{attempts}): {e!r}. Retrying in {wait:.1f}s."
            )
            time.sleep(wait)

    # Final attempt: whatever it raises propagates to the caller.
    return fn()
