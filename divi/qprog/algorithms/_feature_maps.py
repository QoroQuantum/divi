# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Built-in QNN feature maps.

A :class:`FeatureMap` encodes a classical feature vector into a parametric
quantum circuit. The :class:`QNN` algorithm composes a feature map
(data-binding layer) with an :class:`~divi.qprog.algorithms.Ansatz`
(trainable layer) to form the hybrid classical/quantum circuit.

Unlike :class:`~divi.qprog.algorithms.Ansatz`, feature maps are not layered:
a single application encodes the feature vector once. Users who want true
data re-uploading should build a custom feature map that interleaves
encoding with variational layers.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Literal

import numpy as np
from qiskit.circuit import QuantumCircuit

from divi.qprog.algorithms._ansatze import (
    _emit_rx,
    _emit_ry,
    _emit_rz,
    _emit_two_qubit_pauli_rot,
)

_RotationEmitter = Callable[[QuantumCircuit, object, int], None]

_ROTATION_EMITTERS: dict[str, _RotationEmitter] = {
    "X": _emit_rx,
    "Y": _emit_ry,
    "Z": _emit_rz,
}


class FeatureMap(ABC):
    """Abstract base class for QNN feature maps (classical → quantum encoders)."""

    @property
    def name(self) -> str:
        """Human-readable name of the feature map."""
        return self.__class__.__name__

    @staticmethod
    @abstractmethod
    def n_params(n_qubits: int, **kwargs) -> int:
        """Number of data parameters consumed on ``n_qubits`` qubits."""
        raise NotImplementedError

    @abstractmethod
    def build(self, features, n_qubits: int, **kwargs) -> QuantumCircuit:
        """Build the feature-map circuit.

        Args:
            features: Flat parameter array of length ``n_params(n_qubits)``.
                Entries are Qiskit ``Parameter`` objects bound from
                classical data at execution time.
            n_qubits: Number of qubits.

        Returns:
            QuantumCircuit: Qiskit circuit implementing the encoding.
        """
        raise NotImplementedError


class AngleEmbedding(FeatureMap):
    """Encode features as single-qubit rotation angles.

    For an ``n_qubits``-qubit register and an ``n_qubits``-element feature
    vector ``x``, applies ``R(x_i)`` to qubit ``i`` for the chosen rotation
    axis.

    Args:
        rotation: Rotation axis: ``"X"``, ``"Y"``, or ``"Z"``. Defaults to ``"Y"``.
    """

    def __init__(self, rotation: Literal["X", "Y", "Z"] = "Y") -> None:
        if rotation not in _ROTATION_EMITTERS:
            raise ValueError(
                f"rotation must be one of 'X', 'Y', 'Z'; got {rotation!r}."
            )
        self.rotation = rotation
        self._emit = _ROTATION_EMITTERS[rotation]

    @staticmethod
    def n_params(n_qubits: int, **kwargs) -> int:
        """One feature per qubit."""
        return n_qubits

    def build(self, features, n_qubits: int, **kwargs) -> QuantumCircuit:
        feature_arr = np.asarray(features, dtype=object).reshape(n_qubits)
        qc = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            self._emit(qc, feature_arr[q], q)
        return qc


class ZZFeatureMap(FeatureMap):
    """ZZ entangling encoding (Havlíček et al., 2019).

    Applies Hadamards on every qubit, ``RZ(2 * x_i)`` per qubit, then
    ``RZZ(2 * (π − x_i)(π − x_j))`` on every pair from the entangling
    layout.

    Args:
        entangling_layout: Pair pattern for the ZZ interactions. ``"linear"``
            (``(i, i+1)``), ``"circular"`` (linear + wrap-around),
            or ``"all-to-all"`` (all unordered pairs). Defaults to ``"linear"``.
            For ``n_qubits == 2``, ``"circular"`` is equivalent to ``"linear"``:
            the single pair already connects both qubits and ``RZZ`` is symmetric.
    """

    def __init__(
        self,
        entangling_layout: Literal["linear", "circular", "all-to-all"] = "linear",
    ) -> None:
        if entangling_layout not in ("linear", "circular", "all-to-all"):
            raise ValueError(
                f"entangling_layout must be 'linear', 'circular', or "
                f"'all-to-all'; got {entangling_layout!r}."
            )
        self.entangling_layout = entangling_layout

    @staticmethod
    def _require_min_qubits(n_qubits: int) -> None:
        if n_qubits < 2:
            raise ValueError(
                "ZZFeatureMap requires at least 2 qubits for the ZZ entangling "
                f"layer; got n_qubits={n_qubits}. Use AngleEmbedding for "
                "single-qubit encoding."
            )

    @staticmethod
    def n_params(n_qubits: int, **kwargs) -> int:
        """One feature per qubit (re-used inside the ZZ pair terms)."""
        ZZFeatureMap._require_min_qubits(n_qubits)
        return n_qubits

    def _pair_iter(self, n_qubits: int) -> list[tuple[int, int]]:
        if self.entangling_layout == "linear":
            return [(i, i + 1) for i in range(n_qubits - 1)]
        if self.entangling_layout == "circular":
            pairs = [(i, i + 1) for i in range(n_qubits - 1)]
            if n_qubits > 2:
                pairs.append((n_qubits - 1, 0))
            return pairs
        return [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]

    def build(self, features, n_qubits: int, **kwargs) -> QuantumCircuit:
        self._require_min_qubits(n_qubits)
        feature_arr = np.asarray(features, dtype=object).reshape(n_qubits)
        pairs = self._pair_iter(n_qubits)

        qc = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            qc.h(q)
        for q in range(n_qubits):
            qc.rz(2.0 * feature_arr[q], q)
        for a, b in pairs:
            angle = 2.0 * (np.pi - feature_arr[a]) * (np.pi - feature_arr[b])
            _emit_two_qubit_pauli_rot(qc, "ZZ", angle, a, b)
        return qc
