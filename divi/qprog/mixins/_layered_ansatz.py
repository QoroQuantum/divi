# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Layered-ansatz capability for variational programs."""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from qiskit import transpile
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import MetaCircuit
from divi.circuits._conversions import _QISKIT_TO_QASM2

if TYPE_CHECKING:
    from divi.qprog.algorithms import Ansatz, ElectronCounts


class LayeredAnsatzMixin:
    """Builds a variational program's circuit from ``n_layers`` of an
    :class:`~divi.qprog.algorithms.Ansatz`.

    Mix in before :class:`~divi.qprog.VariationalQuantumAlgorithm`. The host
    sets ``n_qubits`` and builds its cost circuit with
    :meth:`_cost_meta_from_ansatz`. Every ansatz call receives the reference
    from :meth:`_ansatz_reference`, so the parameter count and the circuit
    always agree.
    """

    n_qubits: int
    _precision: int

    def __init__(
        self,
        *args,
        ansatz: "Ansatz",
        n_layers: int = 1,
        **kwargs,
    ) -> None:
        """
        Args:
            ansatz: The layered circuit to train.
            n_layers: Number of ansatz layers. Defaults to 1.
            ``*args``, ``**kwargs``: Forwarded to the next class in the MRO
                (the host program).
        """
        super().__init__(*args, **kwargs)
        self.ansatz = ansatz
        self.n_layers = n_layers

    def _ansatz_reference(self) -> "ElectronCounts | None":
        """The electron counts every ansatz call receives, if the problem has any."""
        return None

    @property
    def n_params_per_layer(self) -> int:
        """Number of trainable parameters per ansatz layer."""
        return self.ansatz.n_params_per_layer(self.n_qubits, self._ansatz_reference())

    @property
    def _serialized_ansatz_type(self) -> str | None:
        """The ansatz class and options recorded in checkpoints."""
        return repr(self.ansatz)

    def _parameter_frequencies(self) -> Sequence[tuple[float, int]] | None:
        """The ansatz's per-layer frequencies, repeated across layers."""
        per_layer = self.ansatz.parameter_frequencies(
            self.n_qubits, self._ansatz_reference()
        )
        return None if per_layer is None else list(per_layer) * self.n_layers

    def _cost_meta_from_ansatz(
        self,
        observable: SparsePauliOp | tuple[SparsePauliOp, ...],
        prefix: QuantumCircuit | None = None,
    ) -> MetaCircuit:
        """Wrap ``prefix`` plus the layered ansatz into a cost ``MetaCircuit``
        measuring ``observable``."""
        reference = self._ansatz_reference()
        n_params = self.ansatz.n_params_per_layer(self.n_qubits, reference)
        weights = np.array(
            [ParameterVector(f"w_{i}", n_params) for i in range(self.n_layers)],
            dtype=object,
        )

        qc = QuantumCircuit(self.n_qubits)
        if prefix is not None:
            qc.compose(prefix, inplace=True)
        qc.compose(
            self.ansatz.build(
                weights,
                n_qubits=self.n_qubits,
                n_layers=self.n_layers,
                reference=reference,
            ),
            inplace=True,
        )

        # Ansatzes such as LUCJAnsatz emit gates outside the QASM emitter's
        # basis; optimization_level=0 is a cheap gate-by-gate substitution.
        qc = transpile(
            qc,
            basis_gates=list(_QISKIT_TO_QASM2.keys()),
            optimization_level=0,
        )

        return MetaCircuit(
            circuit_bodies=(((), circuit_to_dag(qc)),),
            parameters=tuple(weights.flatten()),
            observable=observable,
            precision=self._precision,
        )
