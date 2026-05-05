"""
patch_runner_for_fold_sharding.py — add a --fold_ids CLI flag to
benchmark_runner.py so each SLURM task can run only one fold.

This is needed for the full-N parallel BWORF run, where wallclock per
(seed, fold) is ~13-27 hours and we shard 50 ways (5 seeds * 10 folds)
to fit in the 7-day SLURM walltime ceiling.

The patch:
  1. Adds a --fold_ids CLI argument that takes a comma-separated list of
     fold indices (e.g., "0" or "0,3,7"). Default is empty, which means
     run all folds (existing behavior preserved).
  2. Inside the CV loop in run_dataset, if fold_ids is non-empty, only
     execute folds whose index is in the list.

Idempotent: safely re-runnable. Asserts each anchor exists exactly once
before applying. Detects and skips already-applied patches.

Backup before running:
    cp ~/external_benchmark/benchmark_runner.py \\
       ~/external_benchmark/benchmark_runner.py.bak_pre_fold_sharding

Run:
    cd ~/external_benchmark
    python patch_runner_for_fold_sharding.py
"""

from __future__ import annotations

import sys
from pathlib import Path

RUNNER_PATH = Path("benchmark_runner.py")


def main() -> int:
    if not RUNNER_PATH.exists():
        print(f"ERROR: {RUNNER_PATH} not found.", file=sys.stderr)
        return 1

    src = RUNNER_PATH.read_text(encoding="utf-8")

    # Patch 1: Add the --fold_ids CLI flag right after --bworf_n_tries.
    cli_anchor = '    p.add_argument("--bworf_n_tries", type=int, default=2)'
    cli_replacement = (
        '    p.add_argument("--bworf_n_tries", type=int, default=2)\n'
        '    p.add_argument("--fold_ids", type=str, default="",\n'
        '                   help="Comma-separated list of fold indices to run '
        '(e.g. \\"0,3,7\\"). Empty (default) runs all folds.")'
    )
    if "--fold_ids" in src:
        print("Skipping CLI patch: --fold_ids already in source.")
    else:
        n = src.count(cli_anchor)
        if n != 1:
            print(f"ERROR: CLI anchor found {n} times, expected 1.",
                  file=sys.stderr)
            return 1
        src = src.replace(cli_anchor, cli_replacement)
        print("Applied CLI patch: added --fold_ids.")

    # Patch 2: Skip folds not in fold_ids if the flag is non-empty.
    # The runner's CV loop iterates with `for fold_id, (train_idx, test_idx) in enumerate(...)`.
    # We add a continue-if-not-selected check at the top of the body.
    fold_anchor = (
        "            for fold_id, (train_idx, test_idx) in enumerate(splitter.split(X, y_enc)):"
    )
    if "_fold_ids_filter" in src:
        print("Skipping fold-skip patch: already in source.")
    else:
        n2 = src.count(fold_anchor)
        if n2 != 1:
            print(f"ERROR: fold-loop anchor found {n2} times, expected 1.",
                  file=sys.stderr)
            return 1
        # Insert two lines: parse fold_ids into a set, then continue-if-not-in.
        # We do it inside the loop body so we don't change the enumerate() call.
        fold_replacement = (
            "            _fold_ids_filter = set()\n"
            "            if getattr(args, \"fold_ids\", \"\"):\n"
            "                _fold_ids_filter = set(int(x) for x in args.fold_ids.split(\",\") if x.strip())\n"
            + fold_anchor
            + "\n                if _fold_ids_filter and fold_id not in _fold_ids_filter:\n"
            "                    continue"
        )
        src = src.replace(fold_anchor, fold_replacement)
        print("Applied fold-skip patch: per-task fold filtering enabled.")

    # Validate syntax before writing.
    try:
        compile(src, str(RUNNER_PATH), "exec")
    except SyntaxError as e:
        print(f"ERROR: patched source has syntax errors: {e}",
              file=sys.stderr)
        return 1

    RUNNER_PATH.write_text(src, encoding="utf-8")
    print(f"\nSuccessfully patched {RUNNER_PATH}.")
    print("Verify with:")
    print('  python -c "import ast; ast.parse(open(\'benchmark_runner.py\').read()); print(\'OK\')"')
    print("  grep -n 'fold_ids' benchmark_runner.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
