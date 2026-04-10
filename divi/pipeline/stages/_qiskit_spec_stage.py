# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Spec stage that converts Qiskit QuantumCircuit(s) into a pipeline batch."""

from collections.abc import Callable, Mapping, Sequence
from warnings import warn

import numpy as np
import pennylane as qml
import sympy as sp
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression

from divi.circuits import MetaCircuit
from divi.pipeline.abc import MetaCircuitBatch, PipelineEnv, StageToken
from divi.pipeline.stages._circuit_spec_stage import CircuitSpecStage


def qiskit_to_pennylane(
    qc: QuantumCircuit,
    measurement_fn: Callable[[list[int]], qml.measurements.MeasurementProcess],
) -> qml.tape.QuantumScript:
    """Convert a Qiskit QuantumCircuit to a PennyLane QuantumScript.

    Extracts measured qubits, strips ``measure`` instructions, converts the
    circuit via :func:`pennylane.from_qiskit`, and appends the measurement
    returned by *measurement_fn*.

    Args:
        qc: Qiskit QuantumCircuit.
        measurement_fn: ``(measured_wires) → PennyLane measurement``.
            Called with the sorted list of measured wire indices.

    Returns:
        PennyLane QuantumScript with the specified measurement.
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

    qfunc = qml.from_qiskit(qc_no_measure)
    params = [qml.numpy.array(0.0, requires_grad=True) for _ in qc.parameters]

    def qfunc_with_measurement(*p):
        qfunc(*p)
        return measurement_fn(measured_wires)

    return qml.tape.make_qscript(qfunc_with_measurement)(*params)


def _bind_qiskit_expressions(
    qscript: qml.tape.QuantumScript,
    qc: QuantumCircuit,
) -> tuple[qml.tape.QuantumScript, np.ndarray]:
    """Bind sympy expressions from Qiskit gate params into a QuantumScript.

    Iterates all gate parameters in circuit order, converts each
    ``ParameterExpression`` to its sympy equivalent via ``sympify()``,
    and binds them at the correct parameter indices — skipping float
    constants so they remain untouched.

    Returns:
        ``(bound_script, base_symbols)`` where *base_symbols* are the
        ``sympy.Symbol`` objects for each Qiskit ``Parameter``
        (alphabetical order, matching ``qc.parameters``).
    """
    base_symbols = np.array([sp.Symbol(p.name) for p in qc.parameters], dtype=object)
    if len(base_symbols) == 0:
        return qscript, base_symbols

    gate_exprs: list[sp.Basic] = []
    expr_indices: list[int] = []
    total_params = 0
    for instruction in qc.data:
        if instruction.operation.name == "measure":
            continue
        for param in instruction.operation.params:
            if isinstance(param, ParameterExpression):
                gate_exprs.append(param.sympify())
                expr_indices.append(total_params)
            total_params += 1

    qs_param_count = len(qscript.get_parameters())
    if total_params != qs_param_count:
        raise RuntimeError(
            f"Gate parameter count mismatch: Qiskit circuit has "
            f"{total_params} gate parameters but the converted "
            f"QuantumScript has {qs_param_count}. This may indicate "
            f"that PennyLane decomposed or reordered gates during "
            f"conversion."
        )

    bound = qscript.bind_new_parameters(gate_exprs, expr_indices)
    return bound, base_symbols


class QiskitSpecStage(CircuitSpecStage):
    """SpecStage that converts Qiskit QuantumCircuit(s) into MetaCircuit(s).

    Accepts three input shapes:

    - A single ``QuantumCircuit`` → one-element batch
    - A ``Sequence[QuantumCircuit]`` → indexed by position
    - A ``Mapping[str, QuantumCircuit]`` → indexed by key name

    Qiskit ``measure`` instructions are converted to
    :func:`~pennylane.probs` on the measured wires.  If the circuit has
    no measurements, all wires are measured by default (with a warning).

    Qiskit ``Parameter`` objects are converted to ``sympy.Symbol``
    instances.  ``ParameterExpression`` objects (e.g. ``2 * theta``)
    are converted to the corresponding ``sympy`` expression, preserving
    arithmetic relationships.  This enables downstream parameter binding
    via :class:`~divi.pipeline.stages.ParameterBindingStage`.
    """

    def expand(
        self,
        items: QuantumCircuit | Sequence[QuantumCircuit] | Mapping[str, QuantumCircuit],
        env: PipelineEnv,
    ) -> tuple[MetaCircuitBatch, StageToken]:
        """Convert QuantumCircuit(s) to MetaCircuit(s) and build a keyed batch."""
        return super().expand(self._convert(items), env)

    @staticmethod
    def _qiskit_to_meta(qc: QuantumCircuit) -> MetaCircuit:
        """Convert a single Qiskit QuantumCircuit into a MetaCircuit."""
        qscript = qiskit_to_pennylane(qc, lambda wires: qml.probs(wires=wires))
        qscript, base_symbols = _bind_qiskit_expressions(qscript, qc)
        return MetaCircuit(source_circuit=qscript, symbols=base_symbols)

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
