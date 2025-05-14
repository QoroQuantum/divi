from divi.services import QoroService
from tests.benchmarking import Benchmarking

# This one is live
# from divi.services import QoroService
q_service = QoroService("3ce4a6bdaa01a6ada69a5809a0dad69306adf995")

# This uses local sim
# q_service = None


if __name__ == "__main__":

    qasm_str = Benchmarking(num_qubits=9).hea_ansatz(depth=1, entanglement='linear')
    print(qasm_str)

    jobs = q_service.send_circuit_cut_job(qasm_str)
    q_service.poll_job_status(jobs, loop_until_complete=True)
    results = q_service.get_job_results(jobs)
    print(f"Results: {results}")
