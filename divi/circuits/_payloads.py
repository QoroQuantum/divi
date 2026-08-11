# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Circuit payloads: one parametric circuit plus the values to bind."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from qiskit import QuantumCircuit, qasm2
from qiskit.circuit import Parameter
from qiskit.qasm2 import QASM2ExportError

#: Already-resolved circuits: label → circuit, or a sequence labelled by index.
CircuitBatch = Mapping[str, str | QuantumCircuit] | Sequence[str | QuantumCircuit]


def _to_qasm(circuit: str | QuantumCircuit, label: str) -> str:
    if isinstance(circuit, str):
        return circuit
    if isinstance(circuit, QuantumCircuit):
        try:
            return qasm2.dumps(circuit)
        except QASM2ExportError as e:
            raise ValueError(
                f"Circuit '{label}' cannot be exported to OpenQASM 2: {e}"
            ) from e
    raise TypeError(
        f"Circuit '{label}' must be an OpenQASM string or a QuantumCircuit, "
        f"got {type(circuit).__name__}."
    )


@dataclass(frozen=True)
class CircuitPayload:
    """One parametric circuit together with the values to bind into it.

    Shaped like Qiskit's
    :class:`~qiskit.primitives.containers.SamplerPub`, with per-row labels
    added for result routing.  Replaces N near-identical bound circuits with
    one circuit and a parameter matrix.

    A non-parametric circuit is the degenerate case: ``parameters`` is empty
    and ``parameter_sets`` holds a single row with no values.
    """

    circuit: str
    """QASM 2.0 template text with named placeholders, including measurement."""

    parameters: tuple[Parameter, ...]
    """Ordered parameters. Position matches the column order of each
    ``parameter_sets`` row."""

    parameter_sets: tuple[tuple[str, tuple[float, ...]], ...]
    """``(label, values)`` rows; each produces one resolved circuit. Labels
    are the pipeline branch keys results are routed back by."""

    def __post_init__(self):
        """Shape validation only — the caller owns circuit correctness."""
        if not self.parameter_sets:
            raise ValueError("CircuitPayload requires at least one parameter set.")

        expected = len(self.parameters)
        for label, values in self.parameter_sets:
            if len(values) != expected:
                raise ValueError(
                    f"Parameter set '{label}' has {len(values)} value(s) but "
                    f"the circuit declares {expected} parameter(s). Row order "
                    "must match CircuitPayload.parameters."
                )

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Placeholder names, in ``parameters`` order."""
        return tuple(parameter.name for parameter in self.parameters)


def is_bound(payloads: Sequence[CircuitPayload]) -> bool:
    """Whether every payload is already resolved — no parameters left to bind."""
    return all(not payload.parameters for payload in payloads)


def bound_circuits(
    payloads: Sequence[CircuitPayload] | CircuitBatch,
) -> dict[str, str]:
    """Flatten already-bound QASM payloads to the ``label -> circuit`` mapping.

    Also accepts a plain collection of resolved circuits — a ``{label: circuit}``
    mapping or a sequence labelled by index — so a backend needs no separate
    normalisation step.

    Raises:
        ValueError: If any payload still carries parameters. Every row of a
            parametric payload shares one unresolved template, so they would
            all map to the same circuit rather than to their own.
    """
    payloads = as_payloads(payloads)
    if not is_bound(payloads):
        raise ValueError(
            "bound_circuits needs resolved payloads, but one still carries free "
            "parameters. Its rows share a single unresolved template, so the "
            "mapping would submit the same circuit under every label."
        )
    return {
        label: payload.circuit
        for payload in payloads
        for label, _values in payload.parameter_sets
    }


def bound_payloads(circuits: CircuitBatch) -> list[CircuitPayload]:
    """Wrap resolved circuits as single-row payloads — inverse of :func:`bound_circuits`.

    Sequences are labelled by positional index; Qiskit circuits are exported to
    OpenQASM 2.
    """
    labelled = (
        circuits.items()
        if isinstance(circuits, Mapping)
        else ((str(i), circuit) for i, circuit in enumerate(circuits))
    )
    return [
        CircuitPayload(
            circuit=_to_qasm(circuit, label),
            parameters=(),
            parameter_sets=((label, ()),),
        )
        for label, circuit in labelled
    ]


def as_payloads(
    submitted: Sequence[CircuitPayload] | CircuitBatch,
) -> Sequence[CircuitPayload]:
    """Normalise a submission argument to :class:`CircuitPayload` objects.

    A payload sequence passes through. Anything else is shorthand for
    already-resolved circuits and becomes one single-row payload each.

    Raises:
        TypeError: If a single circuit is passed instead of a collection, or a
            sequence mixes payloads with bare circuits.
    """
    if isinstance(submitted, (str, bytes, QuantumCircuit)):
        raise TypeError(
            "submit_circuits expects a collection of circuits; wrap a single "
            "circuit in a list. A bare circuit string is itself a Sequence and "
            "would otherwise be iterated character by character."
        )
    if isinstance(submitted, Mapping):
        return bound_payloads(submitted)

    items = list(submitted)
    payloads = [item for item in items if isinstance(item, CircuitPayload)]
    if not payloads:
        return bound_payloads(cast(CircuitBatch, items))
    if len(payloads) != len(items):
        raise TypeError(
            "A sequence must hold either CircuitPayloads or circuits, not both."
        )
    # The original object, not ``payloads``: callers rely on pass-through.
    return cast(Sequence[CircuitPayload], submitted)
