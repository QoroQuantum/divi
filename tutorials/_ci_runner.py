# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""CI runner for divi tutorials.

Replaces the old sed + GNU parallel approach with validated patching,
line-buffered output, and budget-aware timeouts.

Usage:
    uv run python -m tutorials._ci_runner [--dry-run] [--max-workers N]

Environment variables:
    DIVI_CI_MAX_SHOTS   Cap shots for get_backend() (read by _backend.py).
    DIVI_CI_JOB_TIMEOUT Total job budget in seconds (default 540).
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

TUTORIALS_DIR = Path(__file__).parent
REPO_ROOT = TUTORIALS_DIR.parent

JOB_START_TIME = time.monotonic()
JOB_TIMEOUT_SECONDS = int(os.environ.get("DIVI_CI_JOB_TIMEOUT", 540))
DEFAULT_TIMEOUT = 120

# ---------------------------------------------------------------------------
# Per-tutorial configuration
# ---------------------------------------------------------------------------
# "skip"       – tutorials that cannot run in CI (e.g. need API keys).
# "no_patches" – tutorials that run as-is (shots capped by DIVI_CI_MAX_SHOTS).
# "tutorials"  – tutorials that need source patches and/or custom timeouts.
#
# Every non-underscore .py file in tutorials/ MUST appear in exactly one of
# these three lists.  The runner validates this at startup.
# ---------------------------------------------------------------------------

SKIP = [
    "qasm_thru_service.py",  # requires API key
]

NO_PATCHES = [
    "backend_properties_conversion.py",
    "custom_vqa.py",
    "qaoa_binary_quadratic_model.py",
    "time_evolution.py",
    "time_evolution_trajectory.py",
    "viz_advanced_analysis.py",
    "viz_qaoa_pce_comparison.py",
    "vqe_h2_with_grouping.py",
    "vqe_h2_molecule.py",
]

TUTORIALS: dict[str, dict] = {
    "vqe_hyperparameter_sweep.py": {
        "timeout_seconds": 180,
        "patches": [
            ("shots=2000", "shots=100"),
            (
                "bond_modifiers=[-0.4, -0.25, 0, 0.25, 0.4]",
                "bond_modifiers=[-0.25, 0, 0.25]",
            ),
            (
                "ansatze=[HartreeFockAnsatz(), UCCSDAnsatz()]",
                "ansatze=[HartreeFockAnsatz()]",
            ),
            ("max_iterations=3", "max_iterations=2"),
            ("population_size=10", "population_size=5"),
        ],
    },
    "error_mitigation.py": {
        "timeout_seconds": 360,
        "patches": [
            (
                "QiskitSimulator(n_processes=8",
                "QiskitSimulator(n_processes=4, shots=500",
            ),
            ("max_iterations=20", "max_iterations=5"),
            ("scale_factors = [1.0, 3.0, 5.0]", "scale_factors = [1.0, 3.0]"),
        ],
    },
    "qaoa_qubo_partitioning.py": {
        "timeout_seconds": 180,
        "patches": [
            ("max_iterations=30", "max_iterations=5"),
            ("n_layers=2", "n_layers=1"),
            ("gnp_random_bqm(\n        25,", "gnp_random_bqm(\n        15,"),
            ("beam_width=3", "beam_width=2"),
            ("beam_width=5", "beam_width=2"),
            ("n_partition_candidates=5", "n_partition_candidates=3"),
            ("get_top_solutions(\n        n=5,", "get_top_solutions(\n        n=3,"),
        ],
    },
    "qaoa_max_weight_matching.py": {
        "timeout_seconds": 180,
        "patches": [
            ("n_layers=2", "n_layers=1"),
            ("max_iterations=20", "max_iterations=5"),
            ("max_iterations=10", "max_iterations=3"),
            ("gnm_random_graph(16, 30", "gnm_random_graph(8, 12"),
        ],
    },
    "qaoa_graph_partitioning.py": {
        "timeout_seconds": 180,
        "patches": [
            ("N_NODES = 30", "N_NODES = 10"),
            ("N_EDGES = 40", "N_EDGES = 15"),
            ("max_n_nodes_per_cluster=10", "max_n_nodes_per_cluster=5"),
            ("max_iterations=20", "max_iterations=5"),
        ],
    },
    "qaoa_qubo.py": {
        "patches": [
            ("n_layers=2", "n_layers=1"),
            ("max_iterations=10", "max_iterations=3"),
        ],
    },
    "qaoa_max_clique.py": {
        "patches": [
            ("n_layers=2", "n_layers=1"),
            ("max_iterations=5", "max_iterations=3"),
        ],
    },
    "qaoa_qdrift.py": {
        "patches": [
            ("N_NODES, N_EDGES = 12, 25", "N_NODES, N_EDGES = 8, 12"),
            ("max_iterations=5", "max_iterations=3"),
        ],
    },
    "checkpointing.py": {
        "patches": [
            ("max_iterations=3", "max_iterations=2"),
            ("max_iterations=5", "max_iterations=2"),
            ("vqe2.max_iterations = 6", "vqe2.max_iterations = 4"),
        ],
    },
    "pce_qubo.py": {
        "patches": [
            ("iters = 10", "iters = 3"),
            ("layers = 2", "layers = 1"),
        ],
    },
    "qaoa_hubo.py": {
        "timeout_seconds": 180,
        "patches": [
            ("max_iterations=15", "max_iterations=5"),
        ],
    },
    "iterative_qaoa.py": {
        "timeout_seconds": 180,
        "patches": [
            ("MAX_DEPTH = 8", "MAX_DEPTH = 3"),
            ("ITERS_PER_DEPTH = 15", "ITERS_PER_DEPTH = 5"),
            ("SHOTS = 10000", "SHOTS = 500"),
        ],
    },
    "ce_qaoa_tsp.py": {
        "patches": [
            ("grid_points=5", "grid_points=3"),
            ("max_iterations=5", "max_iterations=1"),
            ("pop_size = 10", "pop_size = 3"),
        ],
    },
    "ce_qaoa_cvrp.py": {
        "timeout_seconds": 180,
        "patches": [
            ("max_iterations=5", "max_iterations=1"),
            ("population_size=10", "population_size=3"),
        ],
    },
}

# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def discover_tutorials() -> set[str]:
    """Find all runnable tutorial .py files (exclude _ prefixed)."""
    return {p.name for p in TUTORIALS_DIR.glob("*.py") if not p.name.startswith("_")}


def validate_config(discovered: set[str]) -> list[str]:
    """Ensure every discovered tutorial is accounted for. Return errors."""
    configured = set(SKIP) | set(NO_PATCHES) | set(TUTORIALS.keys())
    missing = discovered - configured
    extra = configured - discovered

    errors: list[str] = []
    if missing:
        errors.append(
            f"Tutorial(s) not in _ci_runner.py config: {sorted(missing)}\n"
            "   Add to SKIP, NO_PATCHES, or TUTORIALS."
        )
    if extra:
        errors.append(
            f"Configured but not found on disk: {sorted(extra)}\n"
            "   Remove stale entries from _ci_runner.py."
        )
    return errors


def apply_patches(file_path: Path, patches: list[tuple[str, str]]) -> list[str]:
    """Apply validated string replacements. Return errors for unmatched patches."""
    content = file_path.read_text()
    errors: list[str] = []

    for original, replacement in patches:
        if original not in content:
            errors.append(
                f"  Patch failed in {file_path.name}: "
                f"'{original}' not found. Tutorial source may have changed."
            )
            continue
        content = content.replace(original, replacement)

    if not errors:
        file_path.write_text(content)
    return errors


def _stream_output(pipe, prefix: str) -> None:
    """Read from a pipe line by line, printing with a [prefix]."""
    for line in iter(pipe.readline, ""):
        print(f"[{prefix}] {line}", end="", flush=True)
    pipe.close()


def run_tutorial(
    file_path: Path,
    name: str,
    timeout: int,
) -> tuple[str, bool, float, str]:
    """Run a single tutorial with timeout and streamed output."""
    remaining = JOB_TIMEOUT_SECONDS - (time.monotonic() - JOB_START_TIME)
    effective_timeout = min(timeout, remaining - 30)
    if effective_timeout <= 0:
        return (name, False, 0.0, "SKIPPED (job budget exhausted)")

    print(f"[{name}] Starting (timeout {effective_timeout:.0f}s)", flush=True)
    start = time.monotonic()

    try:
        proc = subprocess.Popen(
            [sys.executable, str(file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
            cwd=str(REPO_ROOT),
        )
        reader = threading.Thread(
            target=_stream_output, args=(proc.stdout, name), daemon=True
        )
        reader.start()
        proc.wait(timeout=effective_timeout)
        reader.join(timeout=5)

        elapsed = time.monotonic() - start
        if proc.returncode == 0:
            return (name, True, elapsed, "")
        return (name, False, elapsed, f"exit code {proc.returncode}")

    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        elapsed = time.monotonic() - start
        return (name, False, elapsed, f"TIMEOUT after {effective_timeout:.0f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tutorials for CI.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and patches without running tutorials.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=os.cpu_count() or 2,
        help="Max parallel tutorial processes.",
    )
    args = parser.parse_args()

    # --- Discovery & validation ---
    discovered = discover_tutorials()
    errors = validate_config(discovered)
    if errors:
        print("Configuration errors:")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)

    # --- Build run list ---
    skip_set = set(SKIP)
    to_run: list[dict] = []

    for name in sorted(discovered):
        if name in skip_set:
            print(f"SKIP  {name}")
            continue
        patches = TUTORIALS.get(name, {}).get("patches", [])
        timeout = TUTORIALS.get(name, {}).get("timeout_seconds", DEFAULT_TIMEOUT)
        to_run.append({"name": name, "patches": patches, "timeout": timeout})

    print(f"\n{len(to_run)} tutorial(s) to run, {len(skip_set)} skipped.\n")

    # --- Copy to temp dir & apply patches ---
    tmp_dir = Path(tempfile.mkdtemp())
    tutorials_tmp = tmp_dir / "tutorials"

    try:
        shutil.copytree(TUTORIALS_DIR, tutorials_tmp)

        patch_errors: list[str] = []
        for t in to_run:
            if t["patches"]:
                errs = apply_patches(tutorials_tmp / t["name"], t["patches"])
                patch_errors.extend(errs)

        if patch_errors:
            print("Patch validation errors:")
            for e in patch_errors:
                print(e)
            sys.exit(1)

        print("All patches validated and applied.\n")

        if args.dry_run:
            print(
                f"Verification passed. {len(to_run)} tutorial(s) "
                "would be run (--dry-run)."
            )
            return

        # --- Run tutorials ---
        results: list[tuple[str, bool, float, str]] = []

        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {
                pool.submit(
                    run_tutorial,
                    tutorials_tmp / t["name"],
                    t["name"],
                    t["timeout"],
                ): t["name"]
                for t in to_run
            }
            for future in as_completed(futures):
                result = future.result()
                name, ok, elapsed, err = result
                status = "PASS" if ok else "FAIL"
                msg = f"  ({err})" if err else ""
                print(f"\n{status}  {name}  [{elapsed:.1f}s]{msg}", flush=True)
                results.append(result)

        # --- Summary ---
        results.sort(key=lambda r: r[0])
        failures = [r for r in results if not r[1]]

        print("\n" + "=" * 64)
        print(f"  {'Tutorial':<42} {'Time':>7}  {'Status'}")
        print("-" * 64)
        for name, ok, elapsed, err in results:
            status = "PASS" if ok else f"FAIL ({err})"
            print(f"  {name:<42} {elapsed:>6.1f}s  {status}")
        print("=" * 64)

        if failures:
            print(f"\n{len(failures)} tutorial(s) failed.")
            sys.exit(1)
        else:
            print(f"\nAll {len(results)} tutorial(s) passed.")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
