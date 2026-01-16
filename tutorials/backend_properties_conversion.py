# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""
Tutorial: Converting BackendProperties to Qiskit 2.x BackendV2

This tutorial demonstrates how to convert BackendProperties dictionaries
to Qiskit 2.0+ BackendV2 instances and shows the quantitative
impact of backend properties (gate errors, T1/T2) on simulation results.
"""

import datetime

import qiskit.qasm2
from qiskit import QuantumCircuit

from divi.backends import (
    ParallelSimulator,
    create_backend_from_properties,
)

if __name__ == "__main__":
    # Create a Bell state circuit to test
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    qasm_str = qiskit.qasm2.dumps(qc)

    # Noiseless baseline
    print("Noiseless simulation:")
    noiseless_sim = ParallelSimulator(shots=10000, n_processes=2, simulation_seed=42)
    noiseless_result = noiseless_sim.submit_circuits({"bell": qasm_str})
    noiseless_counts = noiseless_result.results[0]["results"]

    total_shots = sum(noiseless_counts.values())
    prob_00_noiseless = noiseless_counts.get("00", 0) / total_shots
    prob_11_noiseless = noiseless_counts.get("11", 0) / total_shots
    print(f"  {noiseless_counts}")
    print(f"  P(|00⟩)={prob_00_noiseless:.3f}, P(|11⟩)={prob_11_noiseless:.3f}")

    # Simplified properties dictionary - missing fields will be filled automatically
    example_properties = {
        "backend_name": "example_backend",
        "backend_version": "1.0.0",
        "last_update_date": datetime.datetime(2025, 5, 21, 3, 29, 4),
        "qubits": [
            [
                {
                    "name": "T1",
                    "unit": "us",  # Can be omitted - will default to "us" for T1/T2
                    "value": 100.0,
                },
                {
                    "name": "T2",
                    "unit": "us",
                    "value": 80.0,
                },
            ],
            [
                {
                    "name": "T1",
                    "value": 120.0,
                },
                {
                    "name": "T2",
                    "value": 90.0,
                },
            ],
        ],
        "gates": [
            {
                "gate": "rz",
                "qubits": [0],
                "parameters": [
                    {
                        "name": "gate_error",
                        # unit will be automatically set to "" (dimensionless)
                        "value": 0.0,
                    },
                    {
                        "name": "gate_length",
                        # unit will be automatically set to "ns"
                        "value": 0.0,
                    },
                ],
            },
            {
                "gate": "sx",
                "qubits": [0],
                "parameters": [
                    {
                        "name": "gate_error",
                        "value": 0.01,
                    },
                    {
                        "name": "gate_length",
                        "value": 35.0,
                    },
                ],
            },
            {
                "gate": "rz",
                "qubits": [1],
                "parameters": [
                    {
                        "name": "gate_error",
                        "value": 0.0,
                    },
                    {
                        "name": "gate_length",
                        "value": 0.0,
                    },
                ],
            },
            {
                "gate": "sx",
                "qubits": [1],
                "parameters": [
                    {
                        "name": "gate_error",
                        "value": 0.01,
                    },
                    {
                        "name": "gate_length",
                        "value": 35.0,
                    },
                ],
            },
            {
                "gate": "cx",
                "qubits": [0, 1],
                "parameters": [
                    {
                        "name": "gate_error",
                        "value": 0.05,
                    },
                    {
                        "name": "gate_length",
                        "value": 250.0,
                    },
                ],
            },
        ],
    }

    backend = create_backend_from_properties(example_properties)

    # Run simulation with the converted backend
    print("\nSimulation with converted backend:")
    custom_sim = ParallelSimulator(
        qiskit_backend=backend,
        shots=10000,
        n_processes=2,
        simulation_seed=42,
    )
    custom_result = custom_sim.submit_circuits({"bell": qasm_str})
    custom_counts = custom_result.results[0]["results"]

    total_shots_custom = sum(custom_counts.values())
    prob_00_custom = custom_counts.get("00", 0) / total_shots_custom
    prob_11_custom = custom_counts.get("11", 0) / total_shots_custom
    prob_01_custom = custom_counts.get("01", 0) / total_shots_custom
    prob_10_custom = custom_counts.get("10", 0) / total_shots_custom
    print(f"  {custom_counts}")
    print(f"  P(|00⟩)={prob_00_custom:.3f}, P(|11⟩)={prob_11_custom:.3f}")

    # Quantitative comparison
    print("\nImpact analysis:")
    deviation = abs(prob_00_noiseless - prob_00_custom)
    noiseless_error_rate = (
        noiseless_counts.get("01", 0) + noiseless_counts.get("10", 0)
    ) / total_shots
    custom_error_rate = prob_01_custom + prob_10_custom
    print(f"  |00⟩ deviation: {deviation:.3f}")
    print(f"  Error rate: {noiseless_error_rate:.4f} -> {custom_error_rate:.4f}")

    # Inspect backend properties
    conv_target = backend.target
    conv_durations = conv_target.durations()
    if "sx" in conv_target.operation_names:
        sx_dur = conv_durations.get("sx", (0,), unit="s")
        print(f"  SX duration: {sx_dur*1e9:.1f}ns")
    if "cx" in conv_target.operation_names:
        cx_dur = conv_durations.get("cx", (0, 1), unit="s")
        print(f"  CX duration: {cx_dur*1e9:.1f}ns")

    conv_props = conv_target.qubit_properties
    if conv_props and conv_props[0]:
        print(
            f"  Qubit 0: T1={conv_props[0].t1*1e6:.1f}μs, T2={conv_props[0].t2*1e6:.1f}μs"
        )
