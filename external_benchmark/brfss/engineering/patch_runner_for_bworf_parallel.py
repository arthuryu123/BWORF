"""
patch_runner_for_bworf_parallel.py — apply the runner patch to register
the new bworf_parallel model name.

This script edits ~/external_benchmark/benchmark_runner.py in place to add:
  1. A new model name "bworf_parallel" recognized by --models.
  2. A builder function _build_bworf_parallel that imports
     BWORFParallelClassifier and constructs it with current args.
  3. The corresponding registry entry.

It is idempotent: running it twice doesn't double-patch (it asserts
each anchor is found exactly once before applying).

Run from the external_benchmark directory:
    cd ~/external_benchmark
    python patch_runner_for_bworf_parallel.py

Verification afterwards:
    python -c "import ast; ast.parse(open('benchmark_runner.py').read()); print('OK')"
    grep -n "bworf_parallel" benchmark_runner.py

Backup:
    cp benchmark_runner.py benchmark_runner.py.bak_pre_parallel_patch
    BEFORE running this patch. The script does NOT auto-backup.
"""

from __future__ import annotations

import sys
from pathlib import Path


RUNNER_PATH = Path("benchmark_runner.py")


PATCHES = [
    # -----------------------------------------------------------------
    # Patch 1: Add new CLI flag --bworf_parallel_n_jobs (controls workers
    # for the parallel BWORF; default -1 = all CPUs)
    # -----------------------------------------------------------------
    {
        "name": "Add bworf_parallel_n_jobs CLI flag",
        "anchor": (
            '    p.add_argument("--bworf_bootstrap_temperature", type=float, default=1.0)'
        ),
        "replacement": (
            '    p.add_argument("--bworf_bootstrap_temperature", type=float, default=1.0)\n'
            '    p.add_argument("--bworf_parallel_n_jobs", type=int, default=-1,\n'
            '                   help="n_jobs for BWORFParallelClassifier (default -1 = all CPUs)")'
        ),
    },

    # -----------------------------------------------------------------
    # Patch 2: Add the _build_bworf_parallel function right AFTER the
    # existing _build_bworf function. We anchor on the closing of
    # _build_bworf, which has a distinctive return statement.
    # -----------------------------------------------------------------
    {
        "name": "Add _build_bworf_parallel function",
        "anchor": (
            'def _build_bworf(seed: int, args: argparse.Namespace, **_kw):'
        ),
        "replacement": None,  # filled in below; we'll replace the whole body
    },
]


# Inserts the new builder function. We do this as a separate, clearly-
# delimited insertion so we don't have to second-guess the exact body
# of _build_bworf.
NEW_BUILDER_BLOCK = '''

def _build_bworf_parallel(seed: int, args: argparse.Namespace, **_kw):
    """Build BWORFParallelClassifier using the same hyperparameters
    as _build_bworf, plus the n_jobs argument.

    The parallel implementation is byte-equivalent to the serial one for
    the same random_state; see validate_parallel_bworf.py for the proof.
    Use this model name (--models bworf_parallel) only when fitting on
    large datasets where the serial version is the bottleneck.
    """
    try:
        # models/ is on sys.path; importing bworf_parallel pulls in
        # bworf_with_mi as a side effect via its own absolute import.
        from bworf_parallel import BWORFParallelClassifier
    except ImportError as e:
        raise ImportError(
            "BWORFParallelClassifier not available: "
            f"{e}. Make sure models/bworf_parallel.py is present "
            "and models/ is on sys.path."
        )

    return BWORFParallelClassifier(
        n_estimators=args.bworf_n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        l1_strength=args.bworf_l1_strength,
        random_state=seed,
        weighted_bootstrap=args.bworf_weighted_bootstrap,
        bootstrap_temperature=args.bworf_bootstrap_temperature,
        n_tries=args.bworf_n_tries,
        n_jobs=args.bworf_parallel_n_jobs,
        verbose=0,
    )

'''


def main() -> int:
    if not RUNNER_PATH.exists():
        print(f"ERROR: {RUNNER_PATH} not found in current directory.",
              file=sys.stderr)
        print("Run this script from ~/external_benchmark/", file=sys.stderr)
        return 1

    src = RUNNER_PATH.read_text(encoding="utf-8")

    # ----- Patch 1: CLI flag -----
    a = PATCHES[0]["anchor"]
    b = PATCHES[0]["replacement"]
    n = src.count(a)
    if n == 0:
        print(f"ERROR: anchor for patch 1 not found. Has the runner been "
              f"already patched, or has its CLI changed?", file=sys.stderr)
        return 1
    if n > 1:
        print(f"ERROR: anchor for patch 1 found {n} times, expected 1. "
              f"Cannot disambiguate.", file=sys.stderr)
        return 1
    if "bworf_parallel_n_jobs" in src:
        print("Skipping patch 1: --bworf_parallel_n_jobs already in source.")
    else:
        src = src.replace(a, b)
        print("Applied patch 1: added --bworf_parallel_n_jobs CLI flag.")

    # ----- Patch 2: Insert builder function before _build_bworf -----
    anchor2 = PATCHES[1]["anchor"]
    n2 = src.count(anchor2)
    if n2 == 0:
        print(f"ERROR: anchor for patch 2 not found.", file=sys.stderr)
        return 1
    if n2 > 1:
        print(f"ERROR: anchor for patch 2 found {n2} times.",
              file=sys.stderr)
        return 1
    if "_build_bworf_parallel" in src:
        print("Skipping patch 2: _build_bworf_parallel already in source.")
    else:
        # Insert the new builder right BEFORE _build_bworf.
        src = src.replace(anchor2, NEW_BUILDER_BLOCK.lstrip() + "\n\n" + anchor2)
        print("Applied patch 2: inserted _build_bworf_parallel function.")

    # ----- Patch 3: Add 'bworf_parallel' to the model registry -----
    # We look for the existing bworf entry in MODEL_BUILDERS and add a
    # parallel sibling. Anchors on the dictionary literal.
    if '"bworf_parallel": _build_bworf_parallel' in src:
        print("Skipping patch 3: model registry already has bworf_parallel.")
    else:
        # Find the existing bworf line in the MODEL_BUILDERS dict.
        anchor3 = '    "bworf": _build_bworf,'
        n3 = src.count(anchor3)
        if n3 == 0:
            print(f"ERROR: anchor for patch 3 not found "
                  f"(could not find '\"bworf\": _build_bworf,' line).",
                  file=sys.stderr)
            return 1
        if n3 > 1:
            print(f"ERROR: anchor for patch 3 found {n3} times.",
                  file=sys.stderr)
            return 1
        replacement3 = (
            '    "bworf": _build_bworf,\n'
            '    "bworf_parallel": _build_bworf_parallel,'
        )
        src = src.replace(anchor3, replacement3)
        print("Applied patch 3: added bworf_parallel to MODEL_BUILDERS.")

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
    print("  python -c \"import ast; ast.parse(open('benchmark_runner.py').read()); "
          "print('OK')\"")
    print("  grep -n 'bworf_parallel' benchmark_runner.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
