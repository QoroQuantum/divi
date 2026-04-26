# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Spec stage that converts Qiskit QuantumCircuit(s) into a pipeline batch."""

from collections.abc import Mapping, Sequence
from warnings import warn

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from divi.circuits import MetaCircuit
from divi.pipeline.abc import MetaCircuitBatch, PipelineEnv, StageToken
from divi.pipeline.stages._circuit_spec_stage import CircuitSpecStage


class QiskitSpecStage(CircuitSpecStage):
    """SpecStage that converts Qiskit QuantumCircuit(s) into MetaCircuit(s).

    Accepts three input shapes:

    - A single ``QuantumCircuit`` → one-element batch
    - A ``Sequence[QuantumCircuit]`` → indexed by position
    - A ``Mapping[str, QuantumCircuit]`` → indexed by key name

    Qiskit ``measure`` instructions are converted to
    :func:`~pennylane.probs` on the measured wires.  If the circuit has
    no measurements, all wires are measured by default (with a warning).

    Qiskit ``Parameter`` objects flow through directly — the migrated
    pipeline is Qiskit-native, so there's no sympy detour.
    ``ParameterExpression`` objects (e.g. ``2 * theta``) are preserved on
    the resulting DAG, and resolved by
    :class:`~divi.pipeline.stages.ParameterBindingStage` at bind time.
    """

    # pyrefly: ignore[bad-override]
    def expand(
        self,
        batch: QuantumCircuit | Sequence[QuantumCircuit] | Mapping[str, QuantumCircuit],
        env: PipelineEnv,
    ) -> tuple[MetaCircuitBatch, StageToken]:
        """Convert QuantumCircuit(s) to MetaCircuit(s) and build a keyed batch."""
        return super().expand(self._convert(batch), env)

    @staticmethod
    def _qiskit_to_meta(qc: QuantumCircuit) -> MetaCircuit:
        """Convert a single Qiskit QuantumCircuit into a MetaCircuit.

        Extracts measured qubits, strips ``measure`` instructions, turns
        the resulting parametric circuit into a DAG, and stores parameters
        + measured-wire indices on the MetaCircuit.  The circuit is
        assumed to produce a probability distribution (``probs``); expval
        flows go through other paths.
        """
        measured_wires = sorted(
            {
                qc.qubits.index(qubit)
                for instruction in qc.data
                if instruction.operation.name == "measure"
                for qubit in instruction.qubits
            }
        )
        if not measured_wires:
            warn(
                "Provided QuantumCircuit has no measurement operations. "
                "Defaulting to all wires.",
                UserWarning,
                stacklevel=2,
            )
            measured_wires = list(range(len(qc.qubits)))

        qc_no_measure = QuantumCircuit(*qc.qregs, *qc.cregs)
        for instruction in qc.data:
            if instruction.operation.name != "measure":
                qc_no_measure.append(
                    instruction.operation, instruction.qubits, instruction.clbits
                )

        return MetaCircuit(
            circuit_bodies=(((), circuit_to_dag(qc_no_measure)),),
            parameters=tuple(qc_no_measure.parameters),
            measured_wires=tuple(measured_wires),
        )

    @staticmethod
    def _convert(
        items: QuantumCircuit | Sequence[QuantumCircuit] | Mapping[str, QuantumCircuit],
    ) -> MetaCircuit | list[MetaCircuit] | dict[str, MetaCircuit]:
        """Dispatch input shape and convert each QuantumCircuit to MetaCircuit."""
        if isinstance(items, QuantumCircuit):
            return QiskitSpecStage._qiskit_to_meta(items)
        if isinstance(items, str):
            raise TypeError(
                f"QiskitSpecStage expects a QuantumCircuit, sequence, or mapping, "
                f"got str"
            )
        if isinstance(items, Mapping):
            return {k: QiskitSpecStage._qiskit_to_meta(v) for k, v in items.items()}
        if isinstance(items, Sequence):
            return [QiskitSpecStage._qiskit_to_meta(v) for v in items]
        raise TypeError(
            f"QiskitSpecStage expects a QuantumCircuit, sequence, or mapping, "
            f"got {type(items).__name__}"
        )
