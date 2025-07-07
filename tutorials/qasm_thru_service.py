from divi import QoroService

if __name__ == "__main__":
    api_token = "6a539a765fe0b20f409b3c0bbd5d46875598f230"
    service = QoroService(api_token)

    # Test if QoroService is initialized correctly
    service.test_connection()

    circuit = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[4];\ncreg c[4];\nx q[0];\nx q[1];\nry(0) q[2];\ncx q[2],q[3];\ncx q[2],q[0];\ncx q[3],q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];\nmeasure q[2] -> c[2];\nmeasure q[3] -> c[3];\n'
    circuits = {}
    for i in range(10):
        circuits[f"circuit_{i}"] = circuit

    job_id = service.submit_circuits(circuits)

    print(job_id)

    service.poll_job_status(job_id, loop_until_complete=True)

    print(service.get_job_results(job_id))
