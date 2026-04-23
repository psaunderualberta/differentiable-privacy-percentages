from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from conf.config import Config


@dataclass(frozen=True)
class RunContext:
    config: Config
    # Future seam: rng_root_key, run_dir, logger handle, etc. can be added
    # here without touching any caller.


_CTX: ContextVar[RunContext] = ContextVar("run_ctx")


def current() -> RunContext:
    """Read the active RunContext. Call only at trace time, not inside JIT."""
    return _CTX.get()


@contextmanager
def using(ctx: RunContext):
    """Scope a RunContext for the duration of a block. Nests safely."""
    tok = _CTX.set(ctx)
    try:
        yield ctx
    finally:
        _CTX.reset(tok)
