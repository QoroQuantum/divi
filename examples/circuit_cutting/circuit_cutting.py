from divi.services import QoroService
from divi.parallel_simulator import ParallelSimulator

from circuit_generator import CircuitGenerator
from qiskit.result import marginal_counts

# This one is live
q_service = QoroService("3ce4a6bdaa01a6ada69a5809a0dad69306adf995")


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
    NUM_QUBITS = 12

    cg = CircuitGenerator(num_qubits=NUM_QUBITS)
    qasm_str = cg.hea_ansatz()

    jobs = q_service.send_circuit_cut_job(qasm_str)
    q_service.poll_job_status(jobs, loop_until_complete=True)
    results = q_service.get_job_results(jobs)

    res1 = results[0]["results"]

    # Need to reverse the keys to match Qiskit's ordering
    res1 = {k[::-1]: v for k, v in res1.items()}
    res1 = marginal_counts(res1, range(NUM_QUBITS))

    res2 = ParallelSimulator.simulate_circuit((f"circuit", qasm_str), shots=100_000)[
        "results"
    ]

    print(compute_distribution_difference(res1, res2))
