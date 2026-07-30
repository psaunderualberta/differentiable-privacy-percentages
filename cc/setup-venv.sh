#!/bin/bash
# Build a virtualenv for clusters without uv (Alliance Canada et al).
#
# Run on a LOGIN node — compute nodes have no internet, so nothing can be installed
# from inside a job unless it is already in the wheelhouse or cc/wheels.
#
#   bash cc/setup-venv.sh --launcher   # small: just enough to submit jobs (bootstrap mode)
#   bash cc/setup-venv.sh --full       # everything, incl. the GPU stack (persistent mode)
#
# Prefer --launcher: with cc/job-prologue.sh each job builds its own full venv in
# $SLURM_TMPDIR, so the multi-GB stack never occupies /project or /home quota.
#
# Packages come from the cluster wheelhouse where possible (`pip --no-index`) and from
# PyPI only for the gaps, so the CUDA-linked builds stay the vendor's.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11}"

MODE="${1:---launcher}"
case "$MODE" in
    --launcher)
        REQ_FILE="$PROJECT_ROOT/cc/requirements-launcher.txt"
        VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv-launcher}"
        ;;
    --full)
        REQ_FILE="$PROJECT_ROOT/cc/requirements-cluster.txt"
        VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv-cluster}"
        ;;
    *)
        echo "usage: bash cc/setup-venv.sh [--launcher|--full]" >&2
        exit 2
        ;;
esac

echo "==> mode:   $MODE"
echo "==> reqs:   $REQ_FILE"
echo "==> venv:   $VENV_DIR"

# --- modules ---------------------------------------------------------------
if command -v module &>/dev/null; then
    module --force purge
    module load StdEnv/2023 "$PYTHON_MODULE" cuda arrow
else
    echo "WARNING: no 'module' command found; using whatever python3 is on PATH." >&2
fi

# --- venv ------------------------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
    # --no-download keeps the vendored pip, which knows about the wheelhouse.
    virtualenv --no-download "$VENV_DIR" 2>/dev/null || python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --no-index --upgrade pip 2>/dev/null || python -m pip install --upgrade pip

# --- dependencies ----------------------------------------------------------
# One resolver pass against the wheelhouse first: faster and gives a consistent set.
# On failure, retry per package so a single missing wheel doesn't block the rest.
echo "==> installing from cluster wheelhouse"
if ! python -m pip install --no-index -r "$REQ_FILE"; then
    echo "==> wheelhouse could not satisfy every package; retrying per-package"
    missing=()
    while read -r spec; do
        [[ -z "$spec" || "$spec" == \#* ]] && continue
        spec="${spec%%#*}"                     # strip trailing comments
        spec="$(echo "$spec" | xargs)"         # trim whitespace
        [[ -z "$spec" ]] && continue
        if ! python -m pip install --no-index "$spec" 2>/dev/null; then
            echo "    no cluster wheel for: $spec"
            missing+=("$spec")
        fi
    done < "$REQ_FILE"

    if (( ${#missing[@]} )); then
        echo "==> installing ${#missing[@]} package(s) from PyPI"
        python -m pip install "${missing[@]}"
    fi
fi

# --- verify ----------------------------------------------------------------
echo "==> verifying imports"
if [[ "$MODE" == "--full" ]]; then
    MODS="jax jaxlib equinox optax optimistix jaxtyping chex orbax.checkpoint numpy scipy pandas pyarrow sklearn datasets plotly matplotlib tyro wandb tqdm"
else
    MODS="tyro wandb tqdm numpy pandas pyarrow"
fi
MODS="$MODS" python - <<'PY'
import importlib, os
failed = []
for m in os.environ["MODS"].split():
    try:
        importlib.import_module(m)
    except Exception as exc:  # noqa: BLE001
        failed.append(f"{m}: {exc}")
if failed:
    raise SystemExit("MISSING:\n  " + "\n  ".join(failed))
print("all imports OK")
try:
    import jax
    print(f"jax {jax.__version__}; devices: {jax.devices()}")
except ImportError:
    pass
PY

echo
du -sh "$VENV_DIR"
cat <<EOF

==> done. Next:

  # bootstrap mode (recommended: per-job venv in \$SLURM_TMPDIR)
  bash cc/prefetch-wheels.sh                      # once, fills cc/wheels/
  source $PROJECT_ROOT/cc/activate-cluster.sh --bootstrap

  # persistent mode (needs --full above)
  source $PROJECT_ROOT/cc/activate-cluster.sh --full
EOF
