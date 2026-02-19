# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Cross-backend endianness tests.

Run the same circuits on ParallelSimulator and QoroService and verify result
bitstrings (and for deterministic circuits, full counts) are equivalent.
ParallelSimulator is the ground truth.
"""

import pytest

from divi.backends import JobConfig, JobStatus, ParallelSimulator, QoroService


def _circuit_deterministic_2q(bitstring: str) -> str:
    """Circuit that yields exactly one outcome for 2 qubits (e.g. \"01\" or \"10\")."""
    if bitstring == "01":
        x_line = "x q[0];"
    elif bitstring == "10":
        x_line = "x q[1];"
    else:
        raise ValueError(f"Unsupported 2-qubit bitstring: {bitstring!r}")
    return f"""
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    {x_line}
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    """


def _circuit_deterministic_3q(bitstring: str) -> str:
    """Circuit that yields exactly one outcome for 3 qubits (e.g. \"001\", \"010\", \"100\")."""
    if bitstring not in ("001", "010", "100"):
        raise ValueError(f"Unsupported 3-qubit bitstring: {bitstring!r}")
    idx = bitstring.index("1")
    x_line = f"x q[{idx}];"
    return f"""
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg c[3];
    {x_line}
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    measure q[2] -> c[2];
    """


def _circuit_superposition_2q() -> str:
    """Circuit that yields \"00\", \"01\", \"10\", \"11\" with equal probability."""
    return """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    h q[0];
    cx q[0], q[1];
    measure q[0] -> c[0];
    measure q[1] -> c[1];
    """


@pytest.mark.requires_api_key
def test_qoro_service_and_parallel_simulator_produce_equivalent_bitstrings(api_key):
    """Run the same circuits on both backends; verify bitstrings and counts match."""
    circuits = {
        "det_01": _circuit_deterministic_2q("01"),
        "det_10": _circuit_deterministic_2q("10"),
        "det_001": _circuit_deterministic_3q("001"),
        "det_010": _circuit_deterministic_3q("010"),
        "det_100": _circuit_deterministic_3q("100"),
        "super_2q": _circuit_superposition_2q(),
    }
    shots = 1000

    sim_result = ParallelSimulator(
        shots=shots,
        simulation_seed=42,
        _deterministic_execution=True,
    ).submit_circuits(circuits)

    qoro_service = QoroService(
        auth_token=api_key,
        config=JobConfig(qpu_system="qoro_maestro", shots=shots),
    )
    qoro_result = qoro_service.submit_circuits(circuits)
    status = qoro_service.poll_job_status(qoro_result, loop_until_complete=True)
    assert status == JobStatus.COMPLETED
    qoro_result = qoro_service.get_job_results(qoro_result)

    assert sim_result.results is not None
    assert qoro_result.results is not None
    assert len(sim_result.results) == len(qoro_result.results)

    deterministic_labels = {"det_01", "det_10", "det_001", "det_010", "det_100"}

    for sim_item, qoro_item in zip(sim_result.results, qoro_result.results):
        assert sim_item["label"] == qoro_item["label"], "Result order/labels must match"
        sim_counts = sim_item["results"]
        qoro_counts = qoro_item["results"]

        if sim_item["label"] in deterministic_labels:
            assert sim_counts == qoro_counts, (
                f"Deterministic circuit {sim_item['label']!r}: counts must match exactly. "
                f"ParallelSimulator {sim_counts} vs QoroService {qoro_counts}"
            )
        else:
            sim_bitstrings = set(sim_counts.keys())
            qoro_bitstrings = set(qoro_counts.keys())
            assert sim_bitstrings == qoro_bitstrings, (
                f"Bitstring sets differ for {sim_item['label']!r}: "
                f"ParallelSimulator {sim_bitstrings} vs QoroService {qoro_bitstrings}"
            )

    qoro_service.delete_job(qoro_result)
