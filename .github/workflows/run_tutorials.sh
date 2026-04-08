#!/bin/bash
#
# Run tutorial scripts in CI with lighter parameters.
# Delegates to the Python CI runner; this wrapper just sets up the environment.
#
# Usage: ./run_tutorials.sh [--verify-only]
#
set -euo pipefail

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# Prevent matplotlib from trying to open GUI windows in CI.
export MPLBACKEND=Agg

# Cap shots for tutorials that use get_backend() (read by _backend.py).
export DIVI_CI_MAX_SHOTS=500

# Total job budget in seconds (10min GitHub timeout minus 60s buffer).
export DIVI_CI_JOB_TIMEOUT=540

uv run python -m tutorials._ci_runner "$@"
