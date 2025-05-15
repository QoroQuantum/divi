import matplotlib.pyplot as plt
import numpy as np

from divi.services import QoroService
from divi.parallel_simulator import ParallelSimulator

from circuit_generator import CircuitGenerator
from qiskit.result import marginal_counts


# This one is live
q_service = QoroService("3ce4a6bdaa01a6ada69a5809a0dad69306adf995")


def normalize_counts(counts):
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def compute_distribution_difference(res1, res2):
    """Compute the difference between two distributions"""
    res1_total_shots = sum(res1.values())
    res2_total_shots = sum(res2.values())

    res1_distribution = {k: v / res1_total_shots for k, v in res1.items()}
    res2_distribution = {k: v / res2_total_shots for k, v in res2.items()}

    diff = 0
    for key in set(res1_distribution.keys()).union(res2_distribution.keys()):
        res1_prob = res1_distribution.get(key, 0)
        res2_prob = res2_distribution.get(key, 0)
        diff += abs(res1_prob - res2_prob)
    return diff


if __name__ == "__main__":
    NUM_QUBITS = 8

    cg = CircuitGenerator(num_qubits=NUM_QUBITS)
    qasm_str = cg.hea_ansatz()

    jobs = q_service.send_circuit_cut_job(qasm_str)
    q_service.poll_job_status(jobs, loop_until_complete=True)
    results = q_service.get_job_results(jobs)

    res1 = results[0]["results"]

    # Need to reverse the keys to match Qiskit's ordering
    res1 = {k[::-1]: v for k, v in res1.items()}
    res1 = marginal_counts(res1, range(NUM_QUBITS))

    res2 = ParallelSimulator.simulate_circuit((f"circuit", qasm_str), shots=1_000_000)[
        "results"
    ]
    diff = compute_distribution_difference(res1, res2)
    print(f"The difference between the two distributions is: {round(diff, 5)}")

    res1_norm = normalize_counts(res1)
    res2_norm = normalize_counts(res2)

    # Get all possible keys
    all_keys = sorted(set(res1_norm.keys()).union(res2_norm.keys()))

    # Prepare data for plotting
    res1_vals = [res1_norm.get(k, 0) for k in all_keys]
    res2_vals = [res2_norm.get(k, 0) for k in all_keys]

    x = np.arange(len(all_keys))
    width = 0.35

    plt.figure(figsize=(16, 6))
    plt.bar(x - width / 2, res1_vals, width, label="QoroService")
    plt.bar(x + width / 2, res2_vals, width, label="ParallelSimulator")

    plt.xlabel("Bitstring")
    plt.ylabel("Probability")
    plt.title("Normalized Histogram of Results")
    plt.xticks(x, all_keys, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()
