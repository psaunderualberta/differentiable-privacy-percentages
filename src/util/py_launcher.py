"""How project scripts are invoked, so the launchers work with or without uv.

Every command this repo *emits* rather than runs directly — sbatch bodies, transfer
array manifests, job-chain resubmits — prefixes the script name with a launcher from
this module instead of hard-coding ``uv run``.

Three modes, selected by environment:

``uv`` (default)
    ``PY_LAUNCHER`` unset. Emits ``uv run --no-sync``. The original behaviour.

``venv``
    ``PY_LAUNCHER=/abs/path/to/venv/bin/python``. A persistent virtualenv, built once
    by ``cc/setup-venv.sh``. Simple, but the GPU stack lives on quota'd storage.

``bootstrap``
    ``PY_LAUNCHER_BOOTSTRAP=/abs/path/to/cc/job-prologue.sh``. Each job builds its own
    venv in ``$SLURM_TMPDIR`` and throws it away, so nothing large touches
    ``/project``. Because the interpreter path is only known *inside* the job, emitted
    commands must defer expansion to the job's own shell — hence the split below.

The distinction that matters:

* `python_launcher` — run something from *this* process, now. Reads the resolved
  interpreter out of the environment.
* `emitted_launcher` — goes into a script that runs *later*, in another shell, possibly
  on another node. In bootstrap mode this is the literal shell token ``"$PY_LAUNCHER"``,
  which the job expands after sourcing `job_prologue`.

Conflating the two breaks the job chain: a resubmit must spawn its successor with the
dying job's real interpreter, while the sbatch body it *writes* must stay deferred so
the successor builds a fresh venv in its own ``$SLURM_TMPDIR``.
"""

import os
import shlex
import sys

DEFAULT_LAUNCHER = "uv run --no-sync"

#: Resolved interpreter for the current process.
ENV_VAR = "PY_LAUNCHER"

#: Path to the prologue that builds a per-job venv; enables bootstrap mode.
BOOTSTRAP_VAR = "PY_LAUNCHER_BOOTSTRAP"

#: Shell token emitted in bootstrap mode, expanded by the job after sourcing the prologue.
_DEFERRED = '"$PY_LAUNCHER"'


def bootstrap_script() -> str | None:
    """Return the per-job venv prologue path, or None when not in bootstrap mode."""
    return os.environ.get(BOOTSTRAP_VAR, "").strip() or None


def python_launcher() -> str:
    """Return the shell prefix that runs a project script *in this environment, now*.

    In bootstrap mode with no resolved ``PY_LAUNCHER`` we are on the login node, inside
    the small launcher venv — so the running interpreter is the right one. Falling back
    to ``uv run`` there would name a tool that, by assumption, is not installed.
    """
    resolved = os.environ.get(ENV_VAR, "").strip()
    if resolved:
        return resolved
    if bootstrap_script():
        return shlex.quote(sys.executable)
    return DEFAULT_LAUNCHER


def python_launcher_argv() -> list[str]:
    """Return `python_launcher` split for use as a `subprocess` argv prefix."""
    return shlex.split(python_launcher())


def emitted_launcher() -> str:
    """Return the launcher to write into a generated script that runs later.

    Deferred in bootstrap mode, since `$SLURM_TMPDIR` is per-job; otherwise identical
    to `python_launcher`.
    """
    if bootstrap_script():
        return _DEFERRED
    return python_launcher()


def job_prologue() -> str:
    """Return shell lines to inject near the top of a generated sbatch body.

    Sets up whatever `emitted_launcher` assumed. Empty in uv mode, where the emitted
    command is self-contained.
    """
    script = bootstrap_script()
    if script:
        # Exports PY_LAUNCHER (this job's venv) and re-exports BOOTSTRAP_VAR, so an
        # in-job resubmit keeps emitting the deferred form for its successor.
        return f"source {shlex.quote(script)}"
    if os.environ.get(ENV_VAR, "").strip():
        # Persistent-venv mode: make the interpreter visible to in-job resubmits too.
        return f"export {ENV_VAR}={shlex.quote(python_launcher())}"
    return ""
