# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from divi.backends import JobType, QoroService

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

    q_service = QoroService()

    execution_result = q_service.submit_circuits(
        {"sample_cut_circuit": qasm_str}, job_type=JobType.CIRCUIT_CUT
    )
    q_service.poll_job_status(execution_result, loop_until_complete=True)

    completed_result = q_service.get_job_results(execution_result)
    print(f"Results: {completed_result.results}")
