# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

import pennylane as qml


class Ansatz(ABC):
    """Abstract base class for all VQE ansaetze."""

    @property
    def name(self) -> str:
        """Returns the human-readable name of the ansatz."""
        return self.__class__.__name__

    @staticmethod
    @abstractmethod
    def n_params(n_qubits: int, **kwargs) -> int:
        """Returns the number of parameters required by the ansatz for one layer."""
        raise NotImplementedError

    @abstractmethod
    def build(self, params, n_qubits: int, n_layers: int, **kwargs) -> None:
        """
        Builds the ansatz circuit.

        Args:
            params (array): The parameters (weights) for the ansatz.
            n_qubits (int): The number of qubits.
            n_layers (int): The number of layers.
            **kwargs: Additional arguments like n_electrons for chemistry ansaetze.
        """
        raise NotImplementedError


# --- Template Ansaetze ---


class RYAnsatz(Ansatz):
    @staticmethod
    def n_params(n_qubits: int, **kwargs) -> int:
        return n_qubits

    def build(self, params, n_qubits: int, n_layers: int, **kwargs) -> None:
        qml.layer(
            qml.AngleEmbedding,
            n_layers,
            params.reshape(n_layers, -1),
            wires=range(n_qubits),
            rotation="Y",
        )


class RYRZAnsatz(Ansatz):
    @staticmethod
    def n_params(n_qubits: int, **kwargs) -> int:
        return 2 * n_qubits

    def build(self, params, n_qubits: int, n_layers: int, **kwargs) -> None:
        def _ryrz_layer(layer_params, wires):
            ry_rots, rz_rots = layer_params.reshape(2, -1)
            qml.AngleEmbedding(ry_rots, wires=wires, rotation="Y")
            qml.AngleEmbedding(rz_rots, wires=wires, rotation="Z")

        qml.layer(
            _ryrz_layer,
            n_layers,
            params.reshape(n_layers, -1),
            wires=range(n_qubits),
        )


class QAOAAnsatz(Ansatz):
    @staticmethod
    def n_params(n_qubits: int, **kwargs) -> int:
        return qml.QAOAEmbedding.shape(n_layers=1, n_wires=n_qubits)[1]

    def build(self, params, n_qubits: int, n_layers: int, **kwargs) -> None:
        qml.QAOAEmbedding(
            features=[],
            weights=params.reshape(n_layers, -1),
            wires=range(n_qubits),
        )


class HardwareEfficientAnsatz(Ansatz):
    @staticmethod
    def n_params(n_qubits: int, **kwargs) -> int:
        raise NotImplementedError("HardwareEfficientAnsatz is not yet implemented.")

    def build(self, params, n_qubits: int, n_layers: int, **kwargs) -> None:
        raise NotImplementedError("HardwareEfficientAnsatz is not yet implemented.")


# --- Chemistry Ansaetze ---


class UCCSDAnsatz(Ansatz):
    @staticmethod
    def n_params(n_qubits: int, n_electrons: int, **kwargs) -> int:
        singles, doubles = qml.qchem.excitations(n_electrons, n_qubits)
        s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
        return len(s_wires) + len(d_wires)

    def build(
        self, params, n_qubits: int, n_layers: int, n_electrons: int, **kwargs
    ) -> None:
        singles, doubles = qml.qchem.excitations(n_electrons, n_qubits)
        s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
        hf_state = qml.qchem.hf_state(n_electrons, n_qubits)

        qml.UCCSD(
            params.reshape(n_layers, -1),
            wires=range(n_qubits),
            s_wires=s_wires,
            d_wires=d_wires,
            init_state=hf_state,
            n_repeats=n_layers,
        )


class HartreeFockAnsatz(Ansatz):
    @staticmethod
    def n_params(n_qubits: int, n_electrons: int, **kwargs) -> int:
        singles, doubles = qml.qchem.excitations(n_electrons, n_qubits)
        return len(singles) + len(doubles)

    def build(self, params, n_qubits: int, n_layers: int, n_electrons: int, **kwargs):
        singles, doubles = qml.qchem.excitations(n_electrons, n_qubits)
        hf_state = qml.qchem.hf_state(n_electrons, n_qubits)

        qml.layer(
            qml.AllSinglesDoubles,
            n_layers,
            params.reshape(n_layers, -1),
            wires=range(n_qubits),
            hf_state=hf_state,
            singles=singles,
            doubles=doubles,
        )

        # Reset the BasisState operations after the first layer
        # for behaviour similar to UCCSD ansatz
        for op in qml.QueuingManager.active_context().queue[1:]:
            op._hyperparameters["hf_state"] = 0
