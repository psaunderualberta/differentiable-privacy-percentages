#!/bin/bash
# Throwaway probe for the account + chain-memory fixes.  Submit with the account on
# the command line, exactly as run-starter.py now does:
#
#     sbatch -A aip-nidhih cc/slurm/account-probe.sh
#
# Then, for the contrast, submit it WITHOUT the flag:
#
#     sbatch cc/slurm/account-probe.sh
#
# If a stray SBATCH_ACCOUNT is what mis-charged the sweep, the two runs report
# different accounts in section 1 and section 2 is non-empty.  Delete this file
# once you have the answer.
#SBATCH --job-name=acct-probe
#SBATCH --time=00:03:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --account=aip-nidhih
#SBATCH --output=%x-%j.log

echo "=== 1. account this job is actually charged to ==="
# scontrol reads the live record, so it needs no sacct-database lag.
scontrol show job "$SLURM_JOB_ID" | grep -oE '(Account|ReqTRES)=[^ ]*'

echo
echo "=== 2. account-setting env vars visible inside the job ==="
# Anything here also propagates to chain resubmits via --export=ALL.
env | grep -E 'SBATCH_|SLURM_ACCOUNT' || echo "(none — good)"

echo
echo "=== 3. chain resubmit argv ==="
# Fake the CHAIN_* context run-starter.py exports, then let job_chain build the
# continuation command.  CHAIN_RESUBMIT_SCRIPT points at a no-op, so the argv is
# printed and nothing is submitted.
cd "${SLURM_SUBMIT_DIR}/src" || exit 1
CHAIN_RESUBMIT_SCRIPT=/usr/bin/true \
CHAIN_WANDB_PROJ=Sweep \
CHAIN_JOBNAME=probe-run-abc \
CHAIN_ACCOUNT=aip-nidhih \
CHAIN_MEM_PER_GPU=8G \
uv run --no-sync python -c "
from util import job_chain
job_chain.request_shutdown()
job_chain.resubmit_if_requested('run-abc')
"

echo
echo "probe finished at $(date)"
