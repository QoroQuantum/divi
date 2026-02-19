#!/bin/bash

set -euo pipefail

# Create a temporary directory and copy tutorials into it
TEMP_TUTORIALS_DIR=$(mktemp -d)
cp -r tutorials/* "$TEMP_TUTORIALS_DIR/"
failures_file=$(mktemp)
parallel_log=$(mktemp)

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
# Default to 2 minutes per tutorial script; value can be overridden globally via env.
TUTORIAL_TIMEOUT_SECONDS="${TUTORIAL_TIMEOUT_SECONDS:-300}"

# Prevent matplotlib from trying to open GUI windows in CI.
export MPLBACKEND=Agg

cleanup() {
  rm -f "$failures_file" "$parallel_log"
  rm -rf "$TEMP_TUTORIALS_DIR"
}
trap cleanup EXIT

# Tutorials that are known to fail (e.g. require API keys or missing deps).
# These are skipped entirely rather than run-and-expected-to-fail, to avoid
# network hangs and wasted CI time.
SKIP_TUTORIALS="qasm_thru_service.py|circuit_cutting.py"

sed -i \
    -e 's/shots=2000/shots=100/' \
    -e 's/\[-0.4, -0.25, 0, 0.25, 0.4\]/\[-0.25, 0, 0.25\]/' \
    -e 's/\[HartreeFockAnsatz(), UCCSDAnsatz()\]/\[HartreeFockAnsatz()\]/' \
    -e 's/max_iterations=3/max_iterations=2/' \
    "$TEMP_TUTORIALS_DIR"/vqe_hyperparameter_sweep.py
sed -i 's/n_processes=4/n_processes=4,shots=500/g' "$TEMP_TUTORIALS_DIR"/zne.py
sed -i \
    -e 's/max_iterations=10/max_iterations=3/' \
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

# Collect tutorial files, excluding helper modules (_*.py) and skipped tutorials.
tutorial_files=()
for f in "$TEMP_TUTORIALS_DIR"/*.py; do
  basename="$(basename "$f")"

  # Skip private helper modules (e.g. _backend.py).
  [[ "$basename" == _* ]] && continue

  # Skip tutorials that are known to fail.
  if echo "$basename" | grep -qE "^(${SKIP_TUTORIALS})$"; then
    echo "⏭️  Skipping $basename (known failure)"
    continue
  fi

  tutorial_files+=("$f")
done

if [[ ${#tutorial_files[@]} -eq 0 ]]; then
  echo "⚠️  No tutorial files to run."
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

# Run tests in parallel: 2× cores, logs grouped per job.
# --halt never: don't abort remaining jobs when one fails (let all finish).
parallel -j $(( $(nproc) * 2 )) --halt never --joblog "$parallel_log" run_test {} ::: "${tutorial_files[@]}"

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
