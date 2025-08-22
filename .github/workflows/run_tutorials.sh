#!/bin/bash

set -e

export PYTHONPATH=$(pwd):$PYTHONPATH

expected_failures=(
  tutorials/qasm_thru_service.py
  tutorials/circuit_cutting.py
)

sed -i \
    -e 's/shots=2000/shots=100/' \
    -e 's/\[-0.4, -0.25, 0, 0.25, 0.4\]/\[-0.25, 0, 0.25\]/' \
    -e 's/max_iterations=3/max_iterations=2/' \
    tutorials/vqe_hyperparameter_sweep.py
sed -i 's/n_processes=4/n_processes=4,shots=500/g' tutorials/zne_local.py
sed -i \
    -e 's/max_iterations=10/max_iterations=3/' \
    -e 's/n_layers=2/n_layers=1/' \
    tutorials/qaoa_qubo_partitioning.py
sed -i \
    -e 's/N_NODES = 30/N_NODES = 10/' \
    -e 's/N_EDGES = 40/N_EDGES = 15/' \
    -e 's/max_n_nodes_per_cluster=10/max_n_nodes_per_cluster=5/' \
    -e 's/max_iterations=20/max_iterations=5/' \
    -e 's/ParallelSimulator()/ParallelSimulator(shots=500)/' \
    tutorials/qaoa_graph_partitioning.py
sed -i \
    -e 's/n_layers=2/n_layers=1/' \
    -e 's/max_iterations=10/max_iterations=3/' \
    -e 's/ParallelSimulator(shots=10000)/ParallelSimulator(shots=500)/' \
    tutorials/qaoa_qubo.py
sed -i 's/ParallelSimulator()/ParallelSimulator(shots=500)/' tutorials/vqe_h2_molecule_local.py
sed -i \
    -e 's/n_layers=2/n_layers=1/' \
    -e 's/max_iterations=10/max_iterations=3/' \
    -e 's/ParallelSimulator()/ParallelSimulator(shots=500)/' \
    tutorials/qaoa_max_clique_local.py

failures_file=$(mktemp)

run_test() {
  local file="$1"
  echo "🔹 Running $file"

  # Inline expectation check
  local expected=0
  for exp in "${expected_failures[@]}"; do
    if [[ "$file" == "$exp" ]]; then
      expected=1
      break
    fi
  done

  if [[ $expected -eq 1 ]]; then
    echo "⚠️ Expecting failure for $file"
    if poetry run python "$file"; then
      echo "❌ $file was expected to fail but passed"
      echo "$file (unexpected success)" >> "$failures_file"
    else
      echo "✅ $file failed as expected"
    fi
  else
    if ! poetry run python "$file"; then
      echo "❌ $file failed unexpectedly"
      echo "$file (unexpected failure)" >> "$failures_file"
    else
      echo "✅ $file passed"
    fi
  fi
}

export -f run_test
export failures_file
export expected_failures

# Run tests in parallel: 2× cores, logs grouped per job
ls tutorials/*.py | parallel -j $(( $(nproc) * 2 )) --joblog parallel.log run_test {}

echo ""
if [[ -s "$failures_file" ]]; then
  echo "❌ Some scripts failed:"
  sed 's/^/   - /' "$failures_file"
  rm -f "$failures_file"
  status=1
else
  echo "✅ All tutorials scripts behaved as expected."
  rm -f "$failures_file"
  status=0
fi

echo ""
echo "📜 Parallel execution summary (from parallel.log):"
cat parallel.log

exit $status
