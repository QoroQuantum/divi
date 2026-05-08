# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tutorial: Standalone pipelines with PennyLane, QNode, and Qiskit inputs.

Demonstrates how to use Divi's circuit pipeline directly — without a
QuantumProgram — to execute circuits from PennyLane (QuantumScript and
QNode) and Qiskit (QuantumCircuit).

Pipeline results are returned as a ``PipelineResult`` dict.  For a
single-circuit pipeline, use ``result.value`` to get the measurement data
directly.
"""

import numpy as np
import pennylane as qp
from qiskit import QuantumCircuit

from divi.pipeline import CircuitPipeline, PipelineEnv
from divi.pipeline.stages import (
    MeasurementStage,
    ParameterBindingStage,
    PennyLaneSpecStage,
    QiskitSpecStage,
)
from tutorials._backend import get_backend


def _print_probs(result):
    """Pretty-print probability results from a single-circuit pipeline."""
    probs = result.value
    for bitstring, prob in sorted(probs.items()):
        print(f"  |{bitstring}> : {prob:.4f}")


def run_pennylane_script(backend):
    """Example 1: Run a PennyLane QuantumScript through a standalone pipeline."""
    print("=" * 60)
    print("Example 1: PennyLane QuantumScript (Bell state)")
    print("=" * 60)

    qscript = qp.tape.QuantumScript(
        ops=[qp.Hadamard(0), qp.CNOT(wires=[0, 1])],
        measurements=[qp.probs()],
    )

    pipeline = CircuitPipeline(
        stages=[
            PennyLaneSpecStage(),
            MeasurementStage(),
        ]
    )

    env = PipelineEnv(backend=backend)
    result = pipeline.run(initial_spec=qscript, env=env)

    print("Probabilities:")
    _print_probs(result)
    print()


def run_pennylane_qnode(backend):
    """Example 2: Run a PennyLane QNode with parameter binding."""
    print("=" * 60)
    print("Example 2: PennyLane QNode (parametric circuit)")
    print("=" * 60)

    dev = qp.device("default.qubit", wires=1)

    @qp.qnode(dev)
    def circuit(x, y):
        qp.RX(x, wires=0)
        qp.RZ(y, wires=0)
        return qp.expval(qp.Z(0))

    pipeline = CircuitPipeline(
        stages=[
            PennyLaneSpecStage(),
            MeasurementStage(),
            ParameterBindingStage(),
        ]
    )

    param_sets = np.array([[0.5, 0.3]])
    env = PipelineEnv(backend=backend, param_sets=param_sets)
    result = pipeline.run(initial_spec=circuit, env=env)

    expval = result.value
    print(f"Parameters: {param_sets[0].tolist()}")
    print(f"Expectation value: {expval:.4f}")
    print()


def run_qiskit_circuit(backend):
    """Example 3: Run a Qiskit QuantumCircuit through a standalone pipeline."""
    print("=" * 60)
    print("Example 3: Qiskit QuantumCircuit (Bell state)")
    print("=" * 60)

    qc = QuantumCircuit(2, 2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure([0, 1], [0, 1])

    pipeline = CircuitPipeline(
        stages=[
            QiskitSpecStage(),
            MeasurementStage(),
        ]
    )

    env = PipelineEnv(backend=backend)
    result = pipeline.run(initial_spec=qc, env=env)

    print("Probabilities:")
    _print_probs(result)
    print()


if __name__ == "__main__":
    backend = get_backend(shots=5000)

    run_pennylane_script(backend)
    run_pennylane_qnode(backend)
    run_qiskit_circuit(backend)
