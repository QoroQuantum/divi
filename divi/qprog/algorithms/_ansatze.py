# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from itertools import tee
from typing import Literal, Sequence
from warnings import warn

import pennylane as qml


class Ansatz(ABC):
    """Abstract base class for all VQE ansätze."""

    @property
    def name(self) -> str:
        """Returns the human-readable name of the ansatz."""
        return self.__class__.__name__

    @staticmethod
    @abstractmethod
    def n_params_per_layer(n_qubits: int, **kwargs) -> int:
        """Returns the number of parameters required by the ansatz for one layer."""
        raise NotImplementedError

    @abstractmethod
    def build(self, params, n_qubits: int, n_layers: int, **kwargs) -> None:
        """
        Builds the ansatz circuit by adding operations to the active PennyLane
        quantum function context.

        Note: This method is called within a PennyLane quantum function context
        (qml.qnode or qml.tape.make_qscript). Operations are added via side effects
        to the active quantum tape/script.

        Returns:
            None: Operations are added to the active PennyLane context
        """
        raise NotImplementedError


# --- Template Ansaetze ---


class GenericLayerAnsatz(Ansatz):
    """
    A flexible ansatz alternating single-qubit gates with optional entanglers.
    """

    def __init__(
        self,
        gate_sequence: list[qml.operation.Operator],
        entangler: qml.operation.Operator | None = None,
        entangling_layout: (
            Literal["linear", "brick", "circular", "all-to-all"]
            | Sequence[tuple[int, int]]
            | None
        ) = None,
    ):
        """
        Args:
            gate_sequence (list[Callable]): List of one-qubit gate classes (e.g., qml.RY, qml.Rot).
            entangler (Callable): Two-qubit entangling gate class (e.g., qml.CNOT, qml.CZ).
                                  If None, no entanglement is applied.
            entangling_layout (str): Layout for entangling layer ("linear", "all_to_all", etc.).
        """
        if not all(
            issubclass(g, qml.operation.Operator) and g.num_wires == 1
            for g in gate_sequence
        ):
            raise ValueError(
                "All elements in gate_sequence must be PennyLane one-qubit gate classes."
            )
        self.gate_sequence = gate_sequence

        if entangler not in (None, qml.CNOT, qml.CZ):
            raise ValueError("Only qml.CNOT and qml.CZ are supported as entanglers.")
        self.entangler = entangler

        self.entangling_layout = entangling_layout
        if entangler is None and self.entangling_layout is not None:
            warn("`entangling_layout` provided but `entangler` is None.")
        match self.entangling_layout:
            case None | "linear":
                self.entangling_layout = "linear"

                self._layout_fn = lambda n_qubits: zip(
                    range(n_qubits), range(1, n_qubits)
                )
            case "brick":
                self._layout_fn = lambda n_qubits: [
                    (i, i + 1) for r in range(2) for i in range(r, n_qubits - 1, 2)
                ]
            case "circular":
                self._layout_fn = lambda n_qubits: zip(
                    range(n_qubits), [(i + 1) % n_qubits for i in range(n_qubits)]
                )
            case "all_to_all":
                self._layout_fn = lambda n_qubits: (
                    (i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)
                )
            case _:
                if not all(
                    isinstance(ent, tuple)
                    and len(ent) == 2
                    and isinstance(ent[0], int)
                    and isinstance(ent[1], int)
                    for ent in entangling_layout
                ):
                    raise ValueError(
                        "entangling_layout must be 'linear', 'circular', "
                        "'all_to_all', or a Sequence of tuples of integers."
                    )

                self._layout_fn = lambda _: entangling_layout

    def n_params_per_layer(self, n_qubits: int, **kwargs) -> int:
        """Total parameters = sum of gate.num_params per qubit per layer."""
        per_qubit = sum(getattr(g, "num_params", 1) for g in self.gate_sequence)
        return per_qubit * n_qubits

    def build(self, params, n_qubits: int, n_layers: int, **kwargs) -> None:
        # calculate how many params each gate needs per qubit
        gate_param_counts = [getattr(g, "num_params", 1) for g in self.gate_sequence]
        per_qubit = sum(gate_param_counts)

        # reshape into [layers, qubits, per_qubit]
        params = params.reshape(n_layers, n_qubits, per_qubit)
        layout_gen = iter(tee(self._layout_fn(n_qubits), n_layers))

        def _layer(layer_params, wires):
            for w, qubit_params in zip(wires, layer_params):
                idx = 0
                for gate, n_p in zip(self.gate_sequence, gate_param_counts):
                    theta = qubit_params[idx : idx + n_p]
                    gate(*theta, wires=w)
                    idx += n_p

            if self.entangler is not None:
                for wire_a, wire_b in next(layout_gen):
                    self.entangler(wires=[wire_a, wire_b])

        qml.layer(_layer, n_layers, params, wires=range(n_qubits))


class QAOAAnsatz(Ansatz):
    """
    QAOA-style ansatz using PennyLane's QAOAEmbedding.

    Implements a parameterized ansatz based on the Quantum Approximate Optimization
    Algorithm structure, alternating between problem and mixer Hamiltonians.
    """

    @staticmethod
    def n_params_per_layer(n_qubits: int, **kwargs) -> int:
        """
        Calculate the number of parameters per layer for QAOA ansatz.

        Args:
            n_qubits (int): Number of qubits in the circuit.
            **kwargs: Additional unused arguments.

        Returns:
            int: Number of parameters needed per layer.
        """
        return qml.QAOAEmbedding.shape(n_layers=1, n_wires=n_qubits)[1]

    def build(self, params, n_qubits: int, n_layers: int, **kwargs) -> None:
        """
        Build the QAOA ansatz circuit.

        Args:
            params: Parameter array to use for the ansatz.
            n_qubits (int): Number of qubits.
            n_layers (int): Number of QAOA layers.
            **kwargs: Additional unused arguments.
        """
        qml.QAOAEmbedding(
            features=[],
            weights=params.reshape(n_layers, -1),
            wires=range(n_qubits),
        )


class HardwareEfficientAnsatz(Ansatz):
    """
    Hardware-efficient ansatz (not yet implemented).

    This ansatz is designed to be easily implementable on near-term quantum hardware,
    typically using native gate sets and connectivity patterns.

    Note:
        This class is a placeholder for future implementation.
    """

    @staticmethod
    def n_params_per_layer(n_qubits: int, **kwargs) -> int:
        """Not yet implemented."""
        raise NotImplementedError("HardwareEfficientAnsatz is not yet implemented.")

    def build(self, params, n_qubits: int, n_layers: int, **kwargs) -> None:
        """Not yet implemented."""
        raise NotImplementedError("HardwareEfficientAnsatz is not yet implemented.")


# --- Chemistry Ansaetze ---


class UCCSDAnsatz(Ansatz):
    """
    Unitary Coupled Cluster Singles and Doubles (UCCSD) ansatz.

    This ansatz is specifically designed for quantum chemistry calculations,
    implementing the UCCSD approximation which includes all single and double
    electron excitations from a reference state.
    """

    @staticmethod
    def n_params_per_layer(n_qubits: int, n_electrons: int, **kwargs) -> int:
        """
        Calculate the number of parameters per layer for UCCSD ansatz.

        Args:
            n_qubits (int): Number of qubits in the circuit.
            n_electrons (int): Number of electrons in the system.
            **kwargs: Additional unused arguments.

        Returns:
            int: Number of parameters (number of single + double excitations).
        """
        singles, doubles = qml.qchem.excitations(n_electrons, n_qubits)
        s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
        return len(s_wires) + len(d_wires)

    def build(self, params, n_qubits: int, n_layers: int, **kwargs) -> None:
        """
        Build the UCCSD ansatz circuit.

        Args:
            params: Parameter array for excitation amplitudes.
            n_qubits (int): Number of qubits.
            n_layers (int): Number of UCCSD layers (repeats).
            **kwargs: Additional arguments:
                n_electrons (int): Number of electrons in the system (required).
        """
        n_electrons = kwargs.pop("n_electrons")
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
    """
    Hartree-Fock-based ansatz for quantum chemistry.

    This ansatz prepares the Hartree-Fock reference state and applies
    parameterized single and double excitation gates. It's a simplified
    alternative to UCCSD, often used as a starting point for VQE calculations.
    """

    @staticmethod
    def n_params_per_layer(n_qubits: int, n_electrons: int, **kwargs) -> int:
        """
        Calculate the number of parameters per layer for Hartree-Fock ansatz.

        Args:
            n_qubits (int): Number of qubits in the circuit.
            n_electrons (int): Number of electrons in the system.
            **kwargs: Additional unused arguments.

        Returns:
            int: Number of parameters (number of single + double excitations).
        """
        singles, doubles = qml.qchem.excitations(n_electrons, n_qubits)
        return len(singles) + len(doubles)

    def build(self, params, n_qubits: int, n_layers: int, **kwargs) -> None:
        """
        Build the Hartree-Fock ansatz circuit.

        Args:
            params: Parameter array for excitation amplitudes.
            n_qubits (int): Number of qubits.
            n_layers (int): Number of ansatz layers.
            **kwargs: Additional arguments:
                n_electrons (int): Number of electrons in the system (required).
        """
        n_electrons = kwargs.pop("n_electrons")
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
