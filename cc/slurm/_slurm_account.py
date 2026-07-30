"""Shared account handling for the SLURM launchers (stdlib only).

Why this exists: `sbatch` resolves each option from the first of
  1. the command line,
  2. the `SBATCH_*` / `SLURM_*` environment variables,
  3. the `#SBATCH` directives inside the script,
so a login-shell `export SBATCH_ACCOUNT=...` (which DRAC's own docs suggest, and
which propagates into every job via `--export=ALL`, hence into chain resubmits
too) silently outranks `#SBATCH --account=` and charges the job to the wrong
allocation.  Every submission path must therefore pass the account on the
command line; the in-script directive stays for readability of the saved script.
"""


def account_argv(account: str) -> list[str]:
    """`['-A', account]`, or `[]` when no account was given.

    Empty is passed through as "no flag" rather than `-A ''`: the launchers join
    their argv with spaces and run it under a shell, where a blank value would
    make `-A` swallow the next flag.
    """
    account = account.strip()
    return ["-A", account] if account else []
