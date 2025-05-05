from divi.services import QoroService

# This one is live
# from divi.services import QoroService
q_service = QoroService("cf668392271cc93409eb0004037788eb86594277")

# This uses local sim
# q_service = None


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

    jobs = q_service.send_circuit_cut_job(qasm_str)
    q_service.poll_job_status(jobs, loop_until_complete=True)
    results = q_service.get_job_results(jobs)
    print(f"Results: {results}")
