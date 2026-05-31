# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tutorial: CustomVQA — bring your own circuit, from toy to QNN.

CustomVQA wraps a PennyLane ``QuantumScript``/``QNode`` or a Qiskit
``QuantumCircuit`` and optimizes its trainable parameters. This tutorial
builds up to the workflows that actually exercise it:

1. A PennyLane ``QNode`` with a non-trivial tensor-product observable.
2. A Qiskit circuit with **data binding** (some parameters fed from a
   classical feature batch, the rest trained) via ``data_param_indices``.
3. A **multi-argument QNN**: a PennyLane template feature map
   (``AngleEmbedding``) plus a ``StronglyEntanglingLayers`` ansatz, ingested
   with ``arg_shapes`` + ``data_arg`` so the inputs bind from the batch and
   only the weights are optimized.
4. The same QNN with a **nonlinear** ``IQPEmbedding`` feature map, showing
   that templates whose gate angles are products of inputs ingest faithfully.

Examples 3-4 mirror the structure of ``qml.qnn.TorchLayer`` +
``qml.batch_input``: ``arg_shapes`` declares the weight shape and ``data_arg``
names the batched input argument.
"""

import numpy as np
import pennylane as qp
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from divi.qprog import CustomVQA
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from tutorials._backend import get_backend

N_QUBITS = 3


def build_qnode():
    """A QNode minimizing a two-qubit tensor-product observable ``Z⊗Z``.

    Two ``RY`` rotations plus an entangler can drive ``<Z⊗Z>`` to its minimum
    of -1, so the optimizer has a clear gradient to follow.
    """
    dev = qp.device("default.qubit", wires=2)

    @qp.qnode(dev)
    def circuit(theta, phi):
        qp.RY(theta, wires=0)
        qp.RY(phi, wires=1)
        qp.CNOT(wires=[0, 1])
        return qp.expval(qp.Z(0) @ qp.Z(1))

    return circuit


def build_qiskit_data_circuit() -> tuple[QuantumCircuit, Parameter]:
    """A Qiskit circuit with one data parameter (``x``) and two weights."""
    x = Parameter("x")
    w0 = Parameter("w0")
    w1 = Parameter("w1")
    qc = QuantumCircuit(2, 2)
    qc.ry(x, 0)
    qc.ry(w0, 1)
    qc.cx(0, 1)
    qc.rz(w1, 0)
    qc.measure(0, 0)
    qc.measure(1, 1)
    return qc, x


def build_template_qnn(feature_map):
    """A two-argument QNN: ``feature_map(inputs)`` + StronglyEntanglingLayers.

    ``inputs`` is the data axis (bound per sample); ``weights`` are the
    trainable ansatz parameters of shape ``(1, N_QUBITS, 3)``.
    """
    dev = qp.device("default.qubit", wires=N_QUBITS)

    @qp.qnode(dev)
    def circuit(inputs, weights):
        feature_map(inputs)
        qp.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return qp.expval(qp.Z(0) @ qp.Z(1) @ qp.Z(2))

    return circuit


if __name__ == "__main__":
    optimizer = ScipyOptimizer(method=ScipyMethod.COBYLA)
    # Toy feature batch: 4 samples of N_QUBITS features each.
    feature_batch = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
            [1.0, 1.1, 1.2],
        ]
    )

    # Example 1: PennyLane QNode with a tensor-product observable.
    print("Example 1: PennyLane QNode (observable Z⊗Z)")
    program1 = CustomVQA(
        qscript=build_qnode(),
        param_shape=(2,),
        optimizer=optimizer,
        max_iterations=8,
        seed=1997,
        backend=get_backend(),
    )
    program1.run(perform_final_computation=False)
    print(f"  Best loss: {program1.best_loss:.4f}")
    print(f"  Best params (theta, phi): {program1.best_params}\n")

    # Example 2: Qiskit circuit + data binding via integer indices.
    # `x` is bound from the feature batch (one feature per sample); `w0`, `w1`
    # are trained. best_params reports only the weights.
    print("Example 2: Qiskit circuit with data binding (data_param_indices)")
    qc, x = build_qiskit_data_circuit()
    program2 = CustomVQA(
        qscript=qc,
        data_param_indices=[list(qc.parameters).index(x)],
        feature_batch=feature_batch[:, :1],  # one data feature
        loss_reduction="mean",
        optimizer=optimizer,
        max_iterations=8,
        backend=get_backend(),
    )
    program2.run(perform_final_computation=False)
    print(
        f"  Best loss (mean over {len(feature_batch)} samples): {program2.best_loss:.4f}"
    )
    print(f"  Best weights (w0, w1): {program2.best_params}\n")

    # Example 3: multi-argument QNN — AngleEmbedding feature map + SEL ansatz.
    # arg_shapes declares the (1, N_QUBITS, 3) weight tensor; data_arg names the
    # batched input. Only the 9 SEL weights are optimized.
    print("Example 3: QNN with AngleEmbedding feature map + StronglyEntanglingLayers")
    angle_qnn = build_template_qnn(
        lambda inp: qp.AngleEmbedding(inp, wires=range(N_QUBITS), rotation="Y")
    )
    program3 = CustomVQA(
        qscript=angle_qnn,
        arg_shapes={"weights": (1, N_QUBITS, 3)},
        data_arg="inputs",
        feature_batch=feature_batch,
        loss_reduction="mean",
        optimizer=optimizer,
        max_iterations=8,
        backend=get_backend(),
    )
    program3.run(perform_final_computation=False)
    print(
        f"  Optimizer sees {program3.n_params_per_layer} weights (data bound separately)"
    )
    print(
        f"  Best loss (mean over {len(feature_batch)} samples): {program3.best_loss:.4f}\n"
    )

    # Example 4: nonlinear feature map. IQPEmbedding's entangling angles are
    # products of inputs (x_i * x_j); they ingest as ParameterExpressions and
    # bind correctly per sample.
    print("Example 4: QNN with nonlinear IQPEmbedding feature map")
    iqp_qnn = build_template_qnn(
        lambda inp: qp.IQPEmbedding(inp, wires=range(N_QUBITS))
    )
    program4 = CustomVQA(
        qscript=iqp_qnn,
        arg_shapes={"weights": (1, N_QUBITS, 3)},
        data_arg="inputs",
        feature_batch=feature_batch,
        loss_reduction="mean",
        optimizer=optimizer,
        max_iterations=8,
        backend=get_backend(),
    )
    program4.run(perform_final_computation=False)
    print(
        f"  Best loss (mean over {len(feature_batch)} samples): {program4.best_loss:.4f}"
    )
