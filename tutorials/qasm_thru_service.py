# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from divi.backends import JobConfig, JobType, QoroService

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
    job_id = service_with_config.submit_circuits(circuits, override_config=override)

    print(f"Job submitted with ID: {job_id}")
    service_with_config.poll_job_status(job_id, loop_until_complete=True)
    results = service_with_config.get_job_results(job_id)
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
    job_id_expectation = service_with_config.submit_circuits(
        circuits=single_circuit, ham_ops=ham_ops, job_type=JobType.EXPECTATION
    )

    print(f"Expectation job submitted with ID: {job_id_expectation}")
    service_with_config.poll_job_status(job_id_expectation, loop_until_complete=True)
    expectation_results = service_with_config.get_job_results(job_id_expectation)
    print(f"Expectation value results: {expectation_results}")
