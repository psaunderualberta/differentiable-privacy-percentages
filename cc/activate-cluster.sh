# Source (don't execute) this before submitting jobs on a cluster without uv.
#
#   source cc/activate-cluster.sh --bootstrap   # per-job venv in $SLURM_TMPDIR (default)
#   source cc/activate-cluster.sh --full        # one persistent venv on /project
#
# Bootstrap mode keeps the multi-GB GPU stack off quota'd storage: the login node gets
# only the small launcher venv, and each job builds and discards its own full venv on
# node-local scratch. See src/util/py_launcher.py for how the two modes differ in what
# gets written into the generated sbatch scripts.

_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_MODE="${1:---bootstrap}"

if command -v module &>/dev/null; then
    module --force purge
    module load StdEnv/2023 "${PYTHON_MODULE:-python/3.11}" cuda arrow
fi

case "$_MODE" in
    --bootstrap)
        _VENV_DIR="${VENV_DIR:-$_PROJECT_ROOT/.venv-launcher}"
        # shellcheck disable=SC1091
        source "$_VENV_DIR/bin/activate"
        # Jobs resolve their own interpreter, so PY_LAUNCHER must NOT be baked into the
        # generated scripts — unset it and let the prologue below set it in-job.
        unset PY_LAUNCHER
        export PY_LAUNCHER_BOOTSTRAP="$_PROJECT_ROOT/cc/job-prologue.sh"
        echo "mode: bootstrap (per-job venv in \$SLURM_TMPDIR)"
        ;;
    --full)
        _VENV_DIR="${VENV_DIR:-$_PROJECT_ROOT/.venv-cluster}"
        # shellcheck disable=SC1091
        source "$_VENV_DIR/bin/activate"
        unset PY_LAUNCHER_BOOTSTRAP
        # Absolute: the emitted commands run on a compute node whose cwd is the job's
        # --chdir (src/), not the repo root.
        export PY_LAUNCHER="$_VENV_DIR/bin/python"
        echo "mode: full (persistent venv at $_VENV_DIR)"
        ;;
    *)
        echo "usage: source cc/activate-cluster.sh [--bootstrap|--full]" >&2
        return 2 2>/dev/null || exit 2
        ;;
esac

export PROJECT_ROOT="$_PROJECT_ROOT"
export PROJECT_SOURCE_ROOT="$_PROJECT_ROOT/src"
echo "launcher venv: $_VENV_DIR"
