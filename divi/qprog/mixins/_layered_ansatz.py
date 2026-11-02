# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Layered-ansatz capability for variational programs."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from qiskit import transpile
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import MetaCircuit
from divi.circuits._conversions import _QISKIT_TO_QASM2

if TYPE_CHECKING:
    from divi.qprog.algorithms import Ansatz


class LayeredAnsatzMixin:
    """Builds a variational program's circuit from ``n_layers`` of an
    :class:`~divi.qprog.algorithms.Ansatz`.

    Mix in before :class:`~divi.qprog.VariationalQuantumAlgorithm`. The host
    sets ``n_qubits`` and builds its cost circuit with
    :meth:`_cost_meta_from_ansatz`. Every ansatz call receives the keywords
    from :meth:`_ansatz_context` plus the user's ``ansatz_kwargs``, so the
    parameter count and the circuit always agree.
    """

    n_qubits: int
    _precision: int

    def __init__(
        self,
        *args,
        ansatz: "Ansatz",
        n_layers: int = 1,
        ansatz_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """
        Args:
            ansatz: The layered circuit to train.
            n_layers: Number of ansatz layers. Defaults to 1.
            ansatz_kwargs: Ansatz-specific options, forwarded to both
                ``n_params_per_layer`` and ``build``. Defaults to None.
            ``*args``, ``**kwargs``: Forwarded to the next class in the MRO
                (the host program).
        """
        super().__init__(*args, **kwargs)
        self.ansatz = ansatz
        self.n_layers = n_layers
        self._ansatz_kwargs: dict[str, Any] = dict(ansatz_kwargs or {})

    def _ansatz_context(self) -> dict[str, Any]:
        """Problem-derived keywords every ansatz call receives."""
        return {}

    def _ansatz_call_kwargs(self) -> dict[str, Any]:
        """The context keywords, overridden by the user's ``ansatz_kwargs``."""
        return {**self._ansatz_context(), **self._ansatz_kwargs}

    @property
    def n_params_per_layer(self) -> int:
        """Number of trainable parameters per ansatz layer."""
        return self.ansatz.n_params_per_layer(
            self.n_qubits, **self._ansatz_call_kwargs()
        )

    @property
    def _serialized_ansatz_type(self) -> str | None:
        """The ansatz class name recorded in checkpoints."""
        return type(self.ansatz).__name__

    def _parameter_frequencies(self) -> Sequence[tuple[float, int]] | None:
        """The ansatz's per-layer frequencies, repeated across layers."""
        per_layer = self.ansatz.parameter_frequencies(
            self.n_qubits, **self._ansatz_call_kwargs()
        )
        return None if per_layer is None else list(per_layer) * self.n_layers

    def _cost_meta_from_ansatz(
        self,
        observable: SparsePauliOp | tuple[SparsePauliOp, ...],
        prefix: QuantumCircuit | None = None,
    ) -> MetaCircuit:
        """Wrap ``prefix`` plus the layered ansatz into a cost ``MetaCircuit``
        measuring ``observable``."""
        call_kwargs = self._ansatz_call_kwargs()
        n_params = self.ansatz.n_params_per_layer(self.n_qubits, **call_kwargs)
        weights = np.array(
            [ParameterVector(f"w_{i}", n_params) for i in range(self.n_layers)],
            dtype=object,
        )

        qc = QuantumCircuit(self.n_qubits)
        if prefix is not None:
            qc.compose(prefix, inplace=True)
        qc.compose(
            self.ansatz.build(
                weights, n_qubits=self.n_qubits, n_layers=self.n_layers, **call_kwargs
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
