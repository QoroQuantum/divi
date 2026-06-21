# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Quantum Neural Network (QNN) algorithm.

A QNN learns weights of a parameterized quantum circuit so that the
expectation value of a chosen observable, averaged over a batch of classical
feature vectors, is minimized. The circuit factors into two layers:

* a :class:`~divi.qprog.algorithms.FeatureMap` that encodes each feature
  vector ``x_i`` into circuit parameters bound from data — not optimized;
* an :class:`~divi.qprog.algorithms.Ansatz` (any existing Divi ansatz) whose
  parameters are the trainable weights.

Data parameters and weight parameters are kept disjoint: the optimizer
only sees weights, and the data axis is handled internally by
:class:`~divi.pipeline.stages.DataBindingStage`.
"""

from warnings import warn

import numpy as np
import numpy.typing as npt
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import MetaCircuit
from divi.hamiltonians._term_ops import _clean_hamiltonian_spo
from divi.pipeline.stages import LossReductionFn, SampleLossFn
from divi.qprog.algorithms._ansatze import Ansatz
from divi.qprog.algorithms._data_binding import (
    _LOSS_FN_IGNORED_MSG,
    DataBindingMixin,
)
from divi.qprog.algorithms._feature_maps import FeatureMap
from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm


class QNN(DataBindingMixin, VariationalQuantumAlgorithm):
    """Quantum Neural Network trained on a classical feature batch.

    Composes a :class:`~divi.qprog.algorithms.FeatureMap` (data-binding
    layer) with an :class:`~divi.qprog.algorithms.Ansatz` (trainable
    layer) into a single parameterized circuit. At each optimization
    step, the framework evaluates the cost observable on the composed
    circuit once per sample in ``feature_batch``, reduces the per-sample
    expectation values along the sample axis (mean by default), and
    reports one scalar loss per weight candidate to the optimizer. The
    optimizer never sees the data axis.

    The data fan-out and reduction live in
    :class:`~divi.pipeline.stages.DataBindingStage`, so the per-sample
    axis shows up cleanly in
    :meth:`~divi.qprog.QuantumProgram.dry_run` alongside the param-set
    axis — no custom evaluator on the QNN class itself.

    Attributes:
        n_qubits (int): Number of qubits in the circuit.
        feature_map: Classical → quantum encoder
            (:class:`~divi.qprog.algorithms.FeatureMap`).
        ansatz: Trainable variational layer
            (:class:`~divi.qprog.algorithms.Ansatz`).
        n_layers (int): Number of ansatz layers.
        feature_batch (numpy.ndarray): Shape ``(n_samples, n_data_params)``
            classical feature batch fed into the feature map every step.
        labels (numpy.ndarray or None): Shape ``(n_samples,)`` supervised
            targets when training a supervised loss; ``None`` for the
            unsupervised objective.
        cost_hamiltonian: The observable being minimized
            (:class:`~qiskit.quantum_info.SparsePauliOp`).
        loss_constant (float): Constant term extracted from the observable.
        loss_reduction: User-facing aggregation across samples —
            ``"mean"``, ``"sum"``, or a callable ``(n_samples,) → float``.
            The resolved callable is stored privately and forwarded to
            :class:`~divi.pipeline.stages.DataBindingStage`.
        optimizer: Classical optimizer for weight updates.
        max_iterations (int): Maximum number of optimization iterations.
        current_iteration (int): Current optimization iteration.
    """

    def __init__(
        self,
        n_qubits: int,
        feature_map: FeatureMap,
        ansatz: Ansatz,
        feature_batch: npt.ArrayLike,
        *,
        observable: SparsePauliOp | None = None,
        n_layers: int = 1,
        labels: npt.ArrayLike | None = None,
        loss_fn: SampleLossFn = "squared_error",
        loss_reduction: LossReductionFn = "mean",
        max_iterations: int = 10,
        **kwargs,
    ) -> None:
        """Initialize a QNN.

        Args:
            n_qubits: Number of qubits in the circuit.
            feature_map: Encoder applied first; its parameters are bound from
                ``feature_batch`` at execution time.
            ansatz: Trainable variational block applied after the feature map.
            feature_batch: Classical feature batch of shape
                ``(n_samples, n_data_params)``, where ``n_data_params``
                equals ``feature_map.n_params(n_qubits)``.
            observable: Cost observable as a ``SparsePauliOp``. Must act on
                ``n_qubits`` qubits. Defaults to the all-qubit parity
                operator ``Z ⊗ Z ⊗ … ⊗ Z``, which gives a single readout in
                ``[-1, 1]`` and uses information from every qubit.
            n_layers: Number of ansatz layers. Defaults to 1.
            labels: Optional supervised targets of shape ``(n_samples,)``,
                aligned with ``feature_batch``'s rows. When given, the QNN
                trains a *supervised* loss: each sample's prediction (the cost
                observable's expectation value, in ``[-1, 1]`` for the default
                parity observable) is compared to its label via ``loss_fn``,
                and those per-sample losses are aggregated by
                ``loss_reduction``. When ``None`` (default) the QNN minimizes
                the bare expectation value (unsupervised).
            loss_fn: Per-sample supervised loss ``(prediction, label) ->
                float``, used only when ``labels`` is given. ``"squared_error"``
                (default) with the default ``"mean"`` reduction yields
                mean-squared error; pass a callable for a custom loss. A custom
                callable must return a finite value — a NaN/Inf loss propagates
                to the optimizer with no diagnostic.
            loss_reduction: How to aggregate the per-sample values (predictions
                when unsupervised, or per-sample losses when ``labels`` is set)
                into the scalar the optimizer sees. ``"mean"`` (default),
                ``"sum"``, or a callable ``np.ndarray (n_samples,) -> float``.
            max_iterations: Maximum number of optimization iterations.
            **kwargs: Forwarded to
                :class:`~divi.qprog.VariationalQuantumAlgorithm`
                (e.g. ``backend``, ``optimizer``, ``seed``).

        Raises:
            TypeError: If ``feature_map`` or ``ansatz`` are not the expected
                base types, or if ``observable`` is not a ``SparsePauliOp``.
            ValueError: On shape mismatches, non-positive layer counts, or
                degenerate Hamiltonians.
        """
        super().__init__(**kwargs)

        if not isinstance(feature_map, FeatureMap):
            raise TypeError("feature_map must be a FeatureMap instance.")
        if not isinstance(ansatz, Ansatz):
            raise TypeError("ansatz must be an Ansatz instance.")
        if n_qubits <= 0:
            raise ValueError(f"n_qubits must be positive; got {n_qubits}.")
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive; got {n_layers}.")

        default_observable = observable is None
        if observable is None:
            observable = SparsePauliOp.from_list([("Z" * n_qubits, 1.0)])
        elif not isinstance(observable, SparsePauliOp):
            raise TypeError("observable must be a SparsePauliOp.")
        elif observable.num_qubits != n_qubits:
            raise ValueError(
                f"observable acts on {observable.num_qubits} qubits, "
                f"but n_qubits is {n_qubits}."
            )

        self.n_qubits = n_qubits
        self.feature_map = feature_map
        self.ansatz = ansatz
        self.n_layers = n_layers
        self.max_iterations = max_iterations
        self.current_iteration = 0

        self._n_data_params = feature_map.n_params(n_qubits)
        self._n_weight_params = ansatz.n_params_per_layer(n_qubits) * n_layers

        self.feature_batch = self._validate_feature_batch(
            feature_batch, self._n_data_params
        )
        self._set_loss_reduction(loss_reduction)
        self.labels, self._sample_loss_fn = self._resolve_supervision(
            labels, loss_fn, self.feature_batch.shape[0]
        )
        if labels is None and loss_fn != "squared_error":
            warn(_LOSS_FN_IGNORED_MSG, UserWarning, stacklevel=2)

        if self.labels is not None and default_observable:
            # The default parity observable reads out in [-1, 1]; labels outside
            # that band can never be matched, so squared error floors above zero.
            if np.any(np.abs(self.labels) > 1.0):
                warn(
                    "labels fall outside [-1, 1] but the default parity observable "
                    "reads out in [-1, 1]; the supervised loss cannot reach zero. "
                    "Encode labels in [-1, 1] (e.g. -1/+1) or pass an observable "
                    "whose range matches your labels.",
                    UserWarning,
                    stacklevel=2,
                )

        data_params = tuple(ParameterVector("x", self._n_data_params))
        weight_params = tuple(ParameterVector("w", self._n_weight_params))

        feature_circuit = feature_map.build(
            np.asarray(data_params, dtype=object), n_qubits
        )
        ansatz_circuit = ansatz.build(
            np.asarray(weight_params, dtype=object), n_qubits, n_layers
        )
        if feature_circuit.num_qubits != n_qubits:
            raise ValueError(
                f"feature_map produced a {feature_circuit.num_qubits}-qubit "
                f"circuit, expected {n_qubits}."
            )
        if ansatz_circuit.num_qubits != n_qubits:
            raise ValueError(
                f"ansatz produced a {ansatz_circuit.num_qubits}-qubit "
                f"circuit, expected {n_qubits}."
            )

        composed = QuantumCircuit(n_qubits)
        composed.compose(feature_circuit, inplace=True)
        composed.compose(ansatz_circuit, inplace=True)
        self._composed_circuit = composed

        self._data_symbols = data_params
        self._weight_symbols = weight_params
        # The optimizer only ever sees the weight parameters.
        self._param_symbols = np.asarray(weight_params, dtype=object)

        self.cost_hamiltonian, self.loss_constant = _clean_hamiltonian_spo(
            observable, raise_on_constant=True
        )

    # ------------------------------------------------------------------ #
    # Shape contracts the VQA base reads
    # ------------------------------------------------------------------ #

    @property
    def n_params_per_layer(self) -> int:
        """Trainable parameters per ansatz layer (data params are excluded)."""
        return self.ansatz.n_params_per_layer(self.n_qubits)

    # ------------------------------------------------------------------ #
    # Plumbing
    # ------------------------------------------------------------------ #

    def _create_cost_circuit(self) -> MetaCircuit:
        """Single MetaCircuit carrying both parameter groups.

        ``DataBindingStage`` swaps in per-sample variants (data substituted
        as floats) and rewrites ``parameters`` to the weight-only tuple
        before downstream stages run, so the parametric IR we hand to the
        spec stage is the full ``(data + weights)`` parameterization.
        """
        return self._cost_meta_circuit(self._data_symbols + self._weight_symbols)
