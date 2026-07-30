# Sourced (not executed) at the top of every generated sbatch body in bootstrap mode.
#
# Builds the project venv inside $SLURM_TMPDIR — node-local scratch, wiped when the job
# ends — so the multi-GB GPU stack never occupies /project or /home quota. Exports
# PY_LAUNCHER so the job body (and any in-job resubmit) can find the interpreter.
#
# Runs on a COMPUTE NODE with no internet: every install must be satisfiable offline,
# i.e. from the cluster wheelhouse (`--no-index`) plus the small prefetched wheel cache
# in cc/wheels that cc/prefetch-wheels.sh populates on a login node.
#
# Failures MUST abort the job. The sbatch bodies that source this do not set -e, so a
# silent failure here would fall through to running main.py against the bare module
# python — which would either crash confusingly or, worse, half-work.

_prologue_fail() {
    echo "ERROR: job-prologue.sh: $1" >&2
    # Sourced from a script: `exit` ends the job, which is what we want. Interactively,
    # only bail out of the sourced file so we don't close the user's shell.
    if [[ $- == *i* ]]; then
        return 1
    fi
    exit 1
}

_PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
_REQ_FILE="$_PROJECT_ROOT/cc/requirements-cluster.txt"
_WHEELS_DIR="$_PROJECT_ROOT/cc/wheels"

# Fall back to a temp dir when run outside SLURM, so the script stays testable.
_VENV_DIR="${SLURM_TMPDIR:-$(mktemp -d)}/venv"

echo "==> building job venv in $_VENV_DIR"
_t0=$SECONDS

if command -v module &>/dev/null; then
    module --force purge
    module load StdEnv/2023 "${PYTHON_MODULE:-python/3.11}" cuda arrow \
        || _prologue_fail "could not load the required modules"
fi

[[ -f "$_REQ_FILE" ]] || _prologue_fail "missing $_REQ_FILE"

virtualenv --no-download "$_VENV_DIR" >/dev/null \
    || _prologue_fail "virtualenv creation failed in $_VENV_DIR (is \$SLURM_TMPDIR set?)"

# shellcheck disable=SC1091
source "$_VENV_DIR/bin/activate" || _prologue_fail "could not activate $_VENV_DIR"
python -m pip install --no-index --upgrade pip >/dev/null

# --find-links supplies the packages the wheelhouse lacks; --no-index keeps pip from
# reaching for PyPI, which would hang on a node with no route out.
_find_links=()
[[ -d "$_WHEELS_DIR" ]] && _find_links=(--find-links "$_WHEELS_DIR")

python -m pip install --no-index "${_find_links[@]}" -r "$_REQ_FILE" \
    || _prologue_fail "could not build the job venv offline — run 'bash cc/prefetch-wheels.sh' on a login node first"

export PY_LAUNCHER="$_VENV_DIR/bin/python"
# Re-export so an in-job resubmit still emits the DEFERRED launcher for its successor
# (which needs to build its own venv in its own $SLURM_TMPDIR), not this job's path.
export PY_LAUNCHER_BOOTSTRAP="$_PROJECT_ROOT/cc/job-prologue.sh"
export PROJECT_ROOT="$_PROJECT_ROOT"
export PROJECT_SOURCE_ROOT="$_PROJECT_ROOT/src"

echo "==> venv ready in $((SECONDS - _t0))s: $PY_LAUNCHER"
du -sh "$_VENV_DIR" 2>/dev/null || true
