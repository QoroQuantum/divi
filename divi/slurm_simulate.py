import argparse
import json
from pathlib import Path

from divi.parallel_simulator import ParallelSimulator


def simulate_qasm_circuits(circuits, shots, n_processes=2):
    # Placeholder logic
    return ParallelSimulator(shots=shots, n_processes=n_processes).submit_circuits(
        circuits
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir", required=True, help="Directory containing job data."
    )
    parser.add_argument(
        "--shots", type=int, default=1024, help="Number of shots for simulation."
    )
    parser.add_argument(
        "--n-processes",
        type=int,
        default=1,
        help="Number of parallel processes to use for simulation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of circuits to process in each batch.",
    )
    args = parser.parse_args()

    base = Path(args.input_dir)
    circuit_map = json.loads((base / "circuit_map.json").read_text())

    circuits = {}
    results = []
    batch_size = 100
    for i, circuit_label in enumerate(circuit_map.keys()):
        qasm_file = base / f"circuit_{circuit_label}.qasm"
        qasm = qasm_file.read_text()
        circuits[circuit_label] = qasm
        if i % batch_size == 0:
            results.extend(
                simulate_qasm_circuits(circuits, args.shots, args.n_processes)
            )
            circuits = {}

    if circuits.keys():
        results.extend(simulate_qasm_circuits(circuits, args.shots, args.n_processes))

    for result in results:
        circuit_label = result["label"]
        result_file = base / f"result_{circuit_label}.json"

        # Save the result to a file
        with open(result_file, "w") as f:
            json.dump(result["results"], f)

    done_file = Path(base / "done.flag")
    done_file.write_text("OK")
