# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tutorial: QNN — train a small supervised quantum classifier.

This tutorial shows how to train a quantum neural network on a labeled
classical feature batch. The circuit factors into:

* an :class:`~divi.qprog.AngleEmbedding` feature map that encodes each
  feature vector ``x_i`` into single-qubit rotations,
* an :class:`~divi.qprog.GenericLayerAnsatz` with trainable weights.

The :class:`~divi.qprog.QNN` algorithm evaluates the cost observable on the
composed circuit once per sample to get a per-sample prediction in
``[-1, 1]``. With ``labels`` supplied, each prediction is scored against its
label by ``loss_fn`` (squared error here), and those per-sample losses are
averaged into one scalar loss per weight candidate — i.e. mean-squared error.
The optimizer never sees the data axis. Drop ``labels`` to train the
unsupervised objective (minimize the observable) instead.
"""

import numpy as np
from qiskit.circuit.library import CXGate, RYGate, RZGate

from divi.qprog import QNN, AngleEmbedding, GenericLayerAnsatz
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from tutorials._backend import get_backend


def make_toy_dataset(seed: int = 1997) -> tuple[np.ndarray, np.ndarray]:
    """Four 2D feature vectors in two clusters, with ``-1`` / ``+1`` labels.

    The labels are encoded to match the parity observable's ``[-1, 1]``
    readout: cluster A maps to ``-1``, cluster B to ``+1``.
    """
    rng = np.random.default_rng(seed)
    cluster_a = rng.uniform(low=0.1, high=0.6, size=(2, 2))
    cluster_b = rng.uniform(low=2.0, high=2.6, size=(2, 2))
    features = np.vstack([cluster_a, cluster_b])
    labels = np.array([-1.0, -1.0, 1.0, 1.0])
    return features, labels


if __name__ == "__main__":
    n_qubits = 2
    X_train, y_train = make_toy_dataset()

    program = QNN(
        n_qubits=n_qubits,
        feature_map=AngleEmbedding(rotation="Y"),
        feature_batch=X_train,
        labels=y_train,
        loss_fn="squared_error",
        loss_reduction="mean",
        ansatz=GenericLayerAnsatz(
            gate_sequence=[RYGate, RZGate],
            entangler=CXGate,
            entangling_layout="linear",
        ),
        n_layers=2,
        optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
        max_iterations=10,
        backend=get_backend(),
        seed=1997,
    )

    # The "final computation" step is a sampling-mode re-execution used by
    # combinatorial VQAs to extract solution bitstrings. QNNs return their
    # answer as the trained weights themselves, so the step is unnecessary.
    program.run(perform_final_computation=False)

    print(f"Best loss: {program.best_loss:.4f}")
    print(f"Best weights: {program.best_params}")
    print(f"Total circuits executed: {program.total_circuit_count}")

    # Inference: predict() binds the trained weights, measures the readout per
    # sample, and returns the sign as a {-1, +1} class label.
    predictions = program.predict(X_train)
    accuracy = float(np.mean(predictions == y_train))
    print(f"Predictions: {predictions}")
    print(f"Training accuracy: {accuracy:.2%}")
