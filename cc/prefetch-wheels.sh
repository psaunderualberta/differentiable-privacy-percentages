#!/bin/bash
# Populate cc/wheels/ with the packages the cluster wheelhouse does NOT provide.
#
# Run on a LOGIN node (needs internet), once per cluster and again whenever
# cc/requirements-cluster.txt changes:
#
#   bash cc/prefetch-wheels.sh
#
# Compute nodes then build their venv fully offline from wheelhouse + this directory
# (see cc/job-prologue.sh). Only the gaps are stored here, so it stays small — the
# large CUDA-linked builds come from the wheelhouse and are never copied.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REQ_FILE="$PROJECT_ROOT/cc/requirements-cluster.txt"
WHEELS_DIR="$PROJECT_ROOT/cc/wheels"

if command -v module &>/dev/null; then
    module --force purge
    module load StdEnv/2023 "${PYTHON_MODULE:-python/3.11}" cuda arrow
fi

# A throwaway venv purely so the availability probe uses the same interpreter and ABI
# tags the jobs will. Built in a temp dir; nothing persists but the wheels.
PROBE_VENV="$(mktemp -d)/probe"
virtualenv --no-download "$PROBE_VENV" >/dev/null
# shellcheck disable=SC1091
source "$PROBE_VENV/bin/activate"

mkdir -p "$WHEELS_DIR"

missing=()
echo "==> probing wheelhouse"
while read -r spec; do
    [[ -z "$spec" || "$spec" == \#* ]] && continue
    if python -m pip install --no-index --dry-run "$spec" &>/dev/null; then
        echo "    wheelhouse: $spec"
    else
        echo "    MISSING:    $spec"
        missing+=("$spec")
    fi
done < "$REQ_FILE"

if (( ${#missing[@]} == 0 )); then
    echo "==> wheelhouse covers everything; cc/wheels/ not needed."
    exit 0
fi

echo "==> downloading ${#missing[@]} package(s) + deps into $WHEELS_DIR"
# No --no-deps: a package absent from the wheelhouse may also pull absent deps, and the
# compute node cannot fetch them later.
python -m pip download --dest "$WHEELS_DIR" "${missing[@]}"

echo
du -sh "$WHEELS_DIR"
echo "==> done. Commit or keep cc/wheels/ alongside the repo on this cluster."
