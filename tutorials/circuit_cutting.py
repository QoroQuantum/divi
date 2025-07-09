from divi import QoroService
from divi.qoro_service import JobTypes

if __name__ == "__main__":

    qasm_str = """OPENQASM 2.0;
                    include "qelib1.inc";
                    qreg q4[9];
                    creg c0[9];
                    h q4[0];
                    h q4[1];
                    h q4[2];
                    cx q4[0],q4[8];
                    measure q4[0] -> c0[0];
                    measure q4[1] -> c0[1];
                    measure q4[2] -> c0[2];
                    measure q4[8] -> c0[8];"""

    q_service = QoroService("41049da587f31f66b92dd769a591ac03164e4421")

    jobs = q_service.submit_circuits(
        {"sample_cut_circuit": qasm_str}, job_type=JobTypes.CIRCUIT_CUT
    )
    q_service.poll_job_status(jobs, loop_until_complete=True)

    results = q_service.get_job_results(jobs)
    print(f"Results: {results}")
