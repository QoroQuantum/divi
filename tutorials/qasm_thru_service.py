# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from divi.backends import JobConfig, JobType, QoroService

if __name__ == "__main__":
    # Initialize the service. This will use a default configuration.
    service = QoroService()

    # Test if QoroService is initialized correctly
    service.test_connection()

    circuit = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\ncreg c[4];\nx q[0];\nx q[1];\nry(0) q[2];\ncx q[2],q[3];\ncx q[2],q[0];\ncx q[3],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\nmeasure q[3] -> c[3];\n'

    # ============================================================
    # Example 1: Standard Circuit Simulation
    # ============================================================
    print("\n=== Example 1: Standard Circuit Simulation ===")

    circuits = {}
    for i in range(10):
        circuits[f"circuit_{i}"] = circuit

    # We can override the default configuration for a specific job submission.
    # Here, we increase the number of shots to 2000.
    override = JobConfig(shots=2000)
    job_id = service.submit_circuits(circuits, override_config=override)

    print(f"Job submitted with ID: {job_id}")
    service.poll_job_status(job_id, loop_until_complete=True)
    results = service.get_job_results(job_id)
    print(f"Results: {results}")

    # ============================================================
    # Example 2: Expectation Value Calculation
    # ============================================================
    print("\n=== Example 2: Expectation Value Calculation ===")

    # Define Hamiltonian operators to measure
    # For a 4-qubit system, each operator has 4 Pauli terms (one per qubit)
    # Example: "IIII;ZZZZ;XXXX;YYYY" measures:
    #   - Identity (IIII) - no measurement
    #   - All Z on all qubits (ZZZZ)
    #   - All X on all qubits (XXXX)
    #   - All Y on all qubits (YYYY)
    ham_ops = "IIII;ZZZZ;XXXX;YYYY"

    # For expectation value jobs, we only need one circuit instance
    single_circuit = {"circuit_0": circuit}

    # Submit with expectation job type and Hamiltonian operators
    # Note: When ham_ops is provided, shots parameter is not used
    job_id_expectation = service.submit_circuits(
        circuits=single_circuit, ham_ops=ham_ops, job_type=JobType.EXPECTATION
    )

    print(f"Expectation job submitted with ID: {job_id_expectation}")
    service.poll_job_status(job_id_expectation, loop_until_complete=True)
    expectation_results = service.get_job_results(job_id_expectation)
    print(f"Expectation value results: {expectation_results}")
