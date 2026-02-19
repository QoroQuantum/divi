# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from divi.backends import (
    ExecutionConfig,
    JobConfig,
    JobType,
    QoroService,
    SimulationMethod,
    Simulator,
)

if __name__ == "__main__":
    # Example 1: Initialize the service with default configuration.
    service = QoroService()
    service.test_connection()

    circuit = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\ncreg c[4];\nx q[0];\nx q[1];\nry(0) q[2];\ncx q[2],q[3];\ncx q[2],q[0];\ncx q[3],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\nmeasure q[3] -> c[3];\n'

    # Example 2: Initialize the service with a custom default JobConfig.
    default_config = JobConfig(
        shots=500,
        qpu_system="qoro_maestro",
        use_circuit_packing=True,
        tag="tutorial_default",
    )
    service_with_config = QoroService(config=default_config)

    # Example 3: Submit a job with an override JobConfig.
    print("\n" + "=" * 60)
    print("=== Example 1: Submit a job with an override JobConfig ===")
    print("=" * 60)

    circuits = {f"circuit_{i}": circuit for i in range(10)}

    # Override the default number of shots and tag for this submission.
    override = JobConfig(shots=2000, tag="example_3_override")
    execution_result = service_with_config.submit_circuits(
        circuits, override_config=override
    )

    print(f"Job submitted with ID: {execution_result.job_id}")
    service_with_config.poll_job_status(execution_result, loop_until_complete=True)
    completed_result = service_with_config.get_job_results(execution_result)
    results = completed_result.results
    print("Results received.")

    # Verify that the total shots in the histogram matches the override.
    if results:
        total_shots = sum(results[0]["results"].values())
        print(f"Verified total shots in histogram: {total_shots}")

    # Example 4: Submit an expectation value calculation job.
    print("\n" + "=" * 60)
    print("=== Example 2: Submit an expectation value calculation job ===")
    print("=" * 60)

    ham_ops = "IIII;ZZZZ;XXXX;YYYY;ZIII;IZII;IIZI;IIIZ"
    single_circuit = {"circuit_0": circuit}

    # The service will use the default tag 'tutorial_default' for this job.
    execution_result_expectation = service_with_config.submit_circuits(
        circuits=single_circuit, ham_ops=ham_ops, job_type=JobType.EXPECTATION
    )

    print(f"Expectation job submitted with ID: {execution_result_expectation.job_id}")
    service_with_config.poll_job_status(
        execution_result_expectation, loop_until_complete=True
    )
    completed_expectation_result = service_with_config.get_job_results(
        execution_result_expectation
    )
    expectation_results = completed_expectation_result.results
    print(f"Expectation value results: {expectation_results}")

    # Example 5: Set execution configuration on a PENDING job.
    print("\n" + "=" * 60)
    print("=== Example 3: Set execution configuration on a PENDING job ===")
    print("=" * 60)

    # Submit a job — it starts in PENDING status.
    exec_result = service.submit_circuits({"circuit_0": circuit})
    print(f"Job submitted with ID: {exec_result.job_id}")

    # Attach an execution configuration while the job is still PENDING.
    exec_config = ExecutionConfig(
        bond_dimension=16,
        simulator=Simulator.QCSim,
        simulation_method=SimulationMethod.MatrixProductState,
        api_meta={"optimization_level": 2},
    )
    response = service.set_execution_config(exec_result, exec_config)
    print(f"Execution config set: {response['status']}")

    # Retrieve the config to confirm the round-trip.
    retrieved_config = service.get_execution_config(exec_result)
    print(f"Retrieved bond_dimension: {retrieved_config.bond_dimension}")
    print(f"Retrieved simulator: {retrieved_config.simulator}")
    print(f"Retrieved simulation_method: {retrieved_config.simulation_method}")
    print(f"Retrieved api_meta: {retrieved_config.api_meta}")

    # Complete the job and clean up.
    service.poll_job_status(exec_result, loop_until_complete=True)
    service.delete_job(exec_result)
    print("Job completed and cleaned up.")
