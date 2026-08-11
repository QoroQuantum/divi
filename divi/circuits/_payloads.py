# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Circuit payloads: one parametric circuit plus the values to bind."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from qiskit.circuit import Parameter


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
    payloads: Sequence[CircuitPayload] | Mapping[str, str],
) -> dict[str, str]:
    """Flatten already-bound QASM payloads to the ``label -> circuit`` mapping.

    Accepts the ``{label: circuit}`` shorthand and returns it unchanged, so a
    backend needs no separate normalization step.

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


def bound_payloads(circuits: Mapping[str, str]) -> list[CircuitPayload]:
    """Wrap resolved circuits as single-row payloads — inverse of :func:`bound_circuits`."""
    return [
        CircuitPayload(circuit=qasm, parameters=(), parameter_sets=((label, ()),))
        for label, qasm in circuits.items()
    ]


def as_payloads(
    submitted: Sequence[CircuitPayload] | Mapping[str, str],
) -> Sequence[CircuitPayload]:
    """Normalize a submission argument to :class:`CircuitPayload` objects.

    A ``{label: circuit}`` mapping is shorthand for already-resolved circuits
    and becomes one single-row payload each; a payload sequence passes
    through.
    """
    if isinstance(submitted, Mapping):
        return bound_payloads(submitted)
    if isinstance(submitted, (str, bytes)):
        raise TypeError(
            "submit_circuits takes payloads or a {label: circuit} mapping, not a "
            'bare circuit string. Wrap it: {"my_circuit": qasm}.'
        )
    return submitted
