#!/bin/bash

set -euo pipefail

# Create a temporary directory and copy tutorials into it
TEMP_TUTORIALS_DIR=$(mktemp -d)
cp -r tutorials/* "$TEMP_TUTORIALS_DIR/"
failures_file=$(mktemp)
parallel_log=$(mktemp)

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
# Default to 2 minutes per tutorial script; value can be overridden globally via env.
TUTORIAL_TIMEOUT_SECONDS="${TUTORIAL_TIMEOUT_SECONDS:-120}"

cleanup() {
  rm -f "$failures_file" "$parallel_log"
  rm -rf "$TEMP_TUTORIALS_DIR"
}
trap cleanup EXIT

# Convert array to string for easier export/import
EXPECTED_FAILURES_STR="tutorials/qasm_thru_service.py|tutorials/circuit_cutting.py"

sed -i \
    -e 's/shots=2000/shots=100/' \
    -e 's/\[-0.4, -0.25, 0, 0.25, 0.4\]/\[-0.25, 0, 0.25\]/' \
    -e 's/\[HartreeFockAnsatz(), UCCSDAnsatz()\]/\[HartreeFockAnsatz()\]/' \
    -e 's/max_iterations=3/max_iterations=2/' \
    "$TEMP_TUTORIALS_DIR"/vqe_hyperparameter_sweep.py
sed -i 's/n_processes=4/n_processes=4,shots=500/g' "$TEMP_TUTORIALS_DIR"/zne_local.py
sed -i \
    -e 's/max_iterations=10/max_iterations=3/' \
    -e 's/n_layers=2/n_layers=1/' \
    "$TEMP_TUTORIALS_DIR"/qaoa_qubo_partitioning.py
sed -i \
    -e 's/N_NODES = 30/N_NODES = 10/' \
    -e 's/N_EDGES = 40/N_EDGES = 15/' \
    -e 's/max_n_nodes_per_cluster=10/max_n_nodes_per_cluster=5/' \
    -e 's/max_iterations=20/max_iterations=5/' \
    -e 's/ParallelSimulator()/ParallelSimulator(shots=500)/' \
    "$TEMP_TUTORIALS_DIR"/qaoa_graph_partitioning.py
sed -i \
    -e 's/n_layers=2/n_layers=1/' \
    -e 's/max_iterations=10/max_iterations=3/' \
    -e 's/ParallelSimulator(shots=10000)/ParallelSimulator(shots=500)/' \
    "$TEMP_TUTORIALS_DIR"/qaoa_qubo.py
sed -i 's/ParallelSimulator()/ParallelSimulator(shots=500)/' "$TEMP_TUTORIALS_DIR"/vqe_h2_molecule_local.py
sed -i \
    -e 's/n_layers=2/n_layers=1/' \
    -e 's/max_iterations=10/max_iterations=3/' \
    -e 's/ParallelSimulator()/ParallelSimulator(shots=500)/' \
    "$TEMP_TUTORIALS_DIR"/qaoa_max_clique_local.py
sed -i \
    -e 's/N_NODES, N_EDGES = 12, 25/N_NODES, N_EDGES = 8, 12/' \
    -e 's/max_iterations=5/max_iterations=3/' \
    -e 's/shots=1000/shots=500/' \
    "$TEMP_TUTORIALS_DIR"/qaoa_qdrift_local.py

run_test() {
  local file="$1"
  # Extract the original relative path for display and failure checking
  local original_file_path="tutorials/$(basename "$file")"
  echo "🔹 Running $original_file_path"

  # Check if this file is expected to fail
  local expected=0
  if [[ "|${EXPECTED_FAILURES_STR}|" == *"|${original_file_path}|"* ]]; then
    expected=1
  fi

  if [[ $expected -eq 1 ]]; then
    echo "⚠️ Expecting failure for $original_file_path"
    if timeout --signal=TERM --kill-after=30s "${TUTORIAL_TIMEOUT_SECONDS}s" poetry run python "$file"; then
      echo "❌ $original_file_path was expected to fail but passed"
      echo "$original_file_path (unexpected success)" >> "$failures_file"
    else
      echo "✅ $original_file_path failed as expected"
    fi
  else
    if ! timeout --signal=TERM --kill-after=30s "${TUTORIAL_TIMEOUT_SECONDS}s" poetry run python "$file"; then
      echo "❌ $original_file_path failed unexpectedly"
      echo "$original_file_path (unexpected failure)" >> "$failures_file"
    else
      echo "✅ $original_file_path passed"
    fi
  fi
}

export -f run_test
export failures_file
export EXPECTED_FAILURES_STR
export TUTORIAL_TIMEOUT_SECONDS

# Run tests in parallel: 2× cores, logs grouped per job
parallel -j $(( $(nproc) * 2 )) --joblog "$parallel_log" run_test {} ::: "$TEMP_TUTORIALS_DIR"/*.py

echo ""
if [[ -s "$failures_file" ]]; then
  echo "❌ Some scripts failed:"
  sed 's/^/   - /' "$failures_file"
  status=1
else
  echo "✅ All tutorials scripts behaved as expected."
  status=0
fi

echo ""
echo "📜 Parallel execution summary (from $parallel_log):"
cat "$parallel_log"

exit $status
