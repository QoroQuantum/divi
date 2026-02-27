#!/bin/bash
#
# Run tutorial scripts in tutorials/ with CI-friendly tweaks (fewer shots,
# smaller problems). Used by GitHub Actions; run from repo root with Poetry
# available.
#
# Usage: ./run_tutorials.sh [--verify-only]
#   --verify-only    Run setup and verification only (no tutorial execution).
#   VERIFY_ONLY=1    Same as --verify-only.
#
# Requirements: bash, poetry, GNU parallel. Intended for Linux/CI.
#
set -euo pipefail

# --verify-only: run setup + verification only (no tutorial execution).
VERIFY_ONLY=0
[[ "${1:-}" == "--verify-only" ]] && VERIFY_ONLY=1
[[ "${VERIFY_ONLY:-0}" == "1" ]] && VERIFY_ONLY=1

# Create a temporary directory and copy tutorials into it
TEMP_TUTORIALS_DIR=$(mktemp -d)
cp -r tutorials/* "$TEMP_TUTORIALS_DIR/"
failures_file=$(mktemp)
parallel_log=$(mktemp)

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
# Timeout per tutorial script (seconds).
TUTORIAL_TIMEOUT_SECONDS=600
# Parallel jobs = nproc × this multiplier (GNU parallel).
PARALLEL_JOBS_MULTIPLIER=2

# Prevent matplotlib from trying to open GUI windows in CI.
export MPLBACKEND=Agg

cleanup() {
  rm -f "$failures_file" "$parallel_log"
  rm -rf "$TEMP_TUTORIALS_DIR"
}
trap cleanup EXIT

# Tutorials that are known to fail (e.g. require API keys or missing deps).
# They are skipped entirely (never run). Add here if a tutorial cannot run in CI.
SKIP_TUTORIALS=(
  qasm_thru_service.py
  circuit_cutting.py
)

# Tutorials that are run as-is; no sed tweaks needed (already fast/simple for CI).
# All other runnable tutorials must have at least one sed command below.
TUTORIALS_NO_SED_REQUIRED=(
  backend_properties_conversion.py
  custom_vqa.py
  qaoa_binary_quadratic_model.py
  time_evolution.py
  vqe_h2_with_grouping.py
)

# Return 0 if basename is skipped (helper module or in SKIP_TUTORIALS), 1 otherwise.
is_skipped() {
  local basename="$1"
  [[ "$basename" == _* ]] && return 0
  for skip in "${SKIP_TUTORIALS[@]}"; do
    [[ "$basename" == "$skip" ]] && return 0
  done
  return 1
}

# Return 0 if basename is in TUTORIALS_NO_SED_REQUIRED, 1 otherwise.
is_no_sed_required() {
  local basename="$1"
  for exempt in "${TUTORIALS_NO_SED_REQUIRED[@]}"; do
    [[ "$basename" == "$exempt" ]] && return 0
  done
  return 1
}

# Apply CI-friendly tweaks per tutorial (fewer shots/iterations, smaller problems).
sed -i \
    -e 's/shots=2000/shots=100/' \
    -e 's/\[-0.4, -0.25, 0, 0.25, 0.4\]/\[-0.25, 0, 0.25\]/' \
    -e 's/\[HartreeFockAnsatz(), UCCSDAnsatz()\]/\[HartreeFockAnsatz()\]/' \
    -e 's/max_iterations=3/max_iterations=2/' \
    "$TEMP_TUTORIALS_DIR"/vqe_hyperparameter_sweep.py
sed -i 's/n_processes=4/n_processes=4,shots=500/g' "$TEMP_TUTORIALS_DIR"/zne.py
sed -i \
    -e 's/max_iterations=30/max_iterations=5/' \
    -e 's/n_layers=2/n_layers=1/' \
    "$TEMP_TUTORIALS_DIR"/qaoa_qubo_partitioning.py
sed -i \
    -e 's/N_NODES = 30/N_NODES = 10/' \
    -e 's/N_EDGES = 40/N_EDGES = 15/' \
    -e 's/max_n_nodes_per_cluster=10/max_n_nodes_per_cluster=5/' \
    -e 's/max_iterations=20/max_iterations=5/' \
    -e 's/get_backend()/get_backend(shots=500)/' \
    "$TEMP_TUTORIALS_DIR"/qaoa_graph_partitioning.py
sed -i \
    -e 's/n_layers=2/n_layers=1/' \
    -e 's/max_iterations=10/max_iterations=3/' \
    -e 's/get_backend(shots=10000)/get_backend(shots=500)/' \
    "$TEMP_TUTORIALS_DIR"/qaoa_qubo.py
sed -i 's/get_backend()/get_backend(shots=500)/' "$TEMP_TUTORIALS_DIR"/vqe_h2_molecule.py
sed -i \
    -e 's/n_layers=2/n_layers=1/' \
    -e 's/max_iterations=5/max_iterations=3/' \
    -e 's/get_backend()/get_backend(shots=500)/' \
    "$TEMP_TUTORIALS_DIR"/qaoa_max_clique.py
sed -i \
    -e 's/N_NODES, N_EDGES = 12, 25/N_NODES, N_EDGES = 8, 12/' \
    -e 's/max_iterations=5/max_iterations=3/' \
    -e 's/shots=1000/shots=500/' \
    "$TEMP_TUTORIALS_DIR"/qaoa_qdrift.py
sed -i \
    -e 's/max_iterations=3/max_iterations=2/' \
    -e 's/max_iterations=5/max_iterations=2/' \
    -e 's/max_iterations=6/max_iterations=4/' \
    -e 's/get_backend()/get_backend(shots=500)/' \
    "$TEMP_TUTORIALS_DIR"/checkpointing.py
sed -i \
    -e 's/iters = 10/iters = 3/' \
    -e 's/layers = 2/layers = 1/' \
    -e 's/shots=10_000/shots=500/' \
    "$TEMP_TUTORIALS_DIR"/pce_qubo.py
sed -i \
    -e 's/max_iterations=15/max_iterations=5/' \
    -e 's/get_backend(shots=10000)/get_backend(shots=500)/' \
    "$TEMP_TUTORIALS_DIR"/qaoa_hubo.py

# Verification: ensure every runnable tutorial is either sed'd or in TUTORIALS_NO_SED_REQUIRED.
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SEDDED_TUTORIALS=$(grep -oE '"\$TEMP_TUTORIALS_DIR"/[^"*]+\.py' "$SCRIPT_PATH" | sed 's|.*/||' | sort -u)
missing=()
for f in "$TEMP_TUTORIALS_DIR"/*.py; do
  [[ -e "$f" ]] || continue
  basename="$(basename "$f")"
  is_skipped "$basename" && continue
  is_no_sed_required "$basename" && continue
  echo "$SEDDED_TUTORIALS" | grep -qx "$basename" || missing+=("$basename")
done
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "❌ Runnable tutorial(s) without a sed command in run_tutorials.sh:"
  printf '   - %s\n' "${missing[@]}"
  echo "   Add sed tweaks for CI, add to TUTORIALS_NO_SED_REQUIRED, or add to SKIP_TUTORIALS if expected to fail."
  exit 1
fi

# Collect list of tutorials to run (exclude helper modules and skipped tutorials).
tutorial_files=()
for f in "$TEMP_TUTORIALS_DIR"/*.py; do
  basename="$(basename "$f")"
  if is_skipped "$basename"; then
    [[ $VERIFY_ONLY -eq 0 ]] && echo "⏭️  Skipping $basename (known failure)"
    continue
  fi
  tutorial_files+=("$f")
done

if [[ ${#tutorial_files[@]} -eq 0 ]]; then
  echo "⚠️  No tutorial files to run."
  exit 0
fi

if [[ $VERIFY_ONLY -eq 1 ]]; then
  echo "✅ Verification passed. ${#tutorial_files[@]} tutorial(s) would be run (skipped with --verify-only)."
  exit 0
fi

run_test() {
  local file="$1"
  local original_file_path="tutorials/$(basename "$file")"
  echo "🔹 Running $original_file_path"

  local exit_code=0
  timeout --signal=TERM --kill-after=30s "${TUTORIAL_TIMEOUT_SECONDS}s" poetry run python "$file" || exit_code=$?

  if [[ $exit_code -eq 124 ]]; then
    echo "⏰ $original_file_path TIMED OUT after ${TUTORIAL_TIMEOUT_SECONDS}s"
    echo "$original_file_path (timed out)" >> "$failures_file"
  elif [[ $exit_code -ne 0 ]]; then
    echo "❌ $original_file_path failed with exit code $exit_code"
    echo "$original_file_path" >> "$failures_file"
  else
    echo "✅ $original_file_path passed"
  fi
}

export -f run_test
export failures_file
export TUTORIAL_TIMEOUT_SECONDS

# Run tutorials via GNU parallel; run_test is exported so parallel can invoke it.
# -j: nproc × multiplier; --halt never: let all jobs finish even if one fails.
parallel -j $(( $(nproc) * PARALLEL_JOBS_MULTIPLIER )) --halt never --joblog "$parallel_log" run_test {} ::: "${tutorial_files[@]}"

echo ""
if [[ -s "$failures_file" ]]; then
  echo "❌ Some scripts failed:"
  sed 's/^/   - /' "$failures_file"
  status=1
else
  echo "✅ All tutorial scripts passed."
  status=0
fi

echo ""
echo "📜 Parallel execution summary (from $parallel_log):"
cat "$parallel_log"

exit $status
