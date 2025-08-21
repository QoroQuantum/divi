#!/bin/bash

set -e

export PYTHONPATH=$(pwd):$PYTHONPATH

expected_failures=(
  tutorials/qasm_thru_service.py
  tutorials/circuit_cutting.py
)

sed -i -e 's/shots=2000/shots=100/' -e 's/\[-0.4, -0.25, 0, 0.25, 0.4\]/\[-0.25, 0, 0.25\]/' tutorials/vqe_hyperparameter_sweep.py
sed -i 's/n_processes=4/n_processes=4,shots=500/g' tutorials/zne_local.py
sed -i -e 's/max_iterations=10/max_iterations=3/' -e 's/n_layers=2/n_layers=1/' tutorials/qaoa_qubo_partitioning.py
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

is_expected_to_fail() {
  for expected in "${expected_failures[@]}"; do
    if [[ "$1" == "$expected" ]]; then
      return 0
    fi
  done
  return 1
}

failed=()

for file in tutorials/*.py; do
  echo "🔹 Running $file"

  if is_expected_to_fail "$file"; then
    echo "⚠️ Expecting failure for $file"
    if poetry run python "$file"; then
      echo "❌ $file was expected to fail but passed"
      failed+=("$file (unexpected success)")
    else
      echo "✅ $file failed as expected"
    fi
  else
    if ! poetry run python "$file"; then
      echo "❌ $file failed unexpectedly"
      failed+=("$file (unexpected failure)")
    else
      echo "✅ $file passed"
    fi
  fi
done

echo ""
if [ ${#failed[@]} -ne 0 ]; then
  echo "❌ Some scripts failed:"
  for f in "${failed[@]}"; do
    echo "   - $f"
  done
  exit 1
else
  echo "✅ All tutorials scripts behaved as expected."
fi
