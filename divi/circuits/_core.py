# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Self

import numpy as np
import numpy.typing as npt
import pennylane as qml

from divi.circuits._qasm_conversion import circuit_body_to_qasm
from divi.circuits._qasm_template import QASMTemplate, build_template
from divi.typing import QASMTag


@dataclass(frozen=True)
class MetaCircuit:
    """
    Logical circuit IR.

    MetaCircuit stores circuit intent and symbols. The circuit body QASM is
    computed once in __post_init__ and remains stable until QEM (e.g. folding)
    modifies it.
    """

    source_circuit: qml.tape.QuantumScript
    """The PennyLane quantum circuit with symbolic parameters."""
    symbols: npt.NDArray[np.object_]
    """Flat 1D array of sympy symbols used as circuit parameters.
    Multi-dimensional inputs are flattened in ``__post_init__``."""
    precision: int = 8
    """Number of decimal places for parameter values in QASM conversion."""
    circuit_body_qasms: tuple[tuple[QASMTag, str], ...] = field(init=False)
    """OpenQASM 2.0 body. Computed in __post_init__ when None; QEMStage overrides via replace()."""
    measurement_qasms: tuple[tuple[QASMTag, str], ...] = field(init=False)
    """OpenQASM 2.0 body. Computed in __post_init__ when None; QEMStage overrides via replace()."""
    measurement_groups: tuple[tuple[object, ...], ...] = field(init=False)
    """Cached grouped observables set by MeasurementStage."""
    circuit_body_templates: tuple[tuple[QASMTag, QASMTemplate], ...] = field(init=False)
    """Pre-split QASM templates for fast parameter binding."""

    def __post_init__(self):
        """Validate logical circuit shape and compute circuit body QASM when not set."""
        if len(self.source_circuit.measurements) != 1:
            raise ValueError(
                f"MetaCircuit requires a circuit with exactly one measurement, "
                f"but {len(self.source_circuit.measurements)} were found."
            )

        # Normalise symbols to a flat 1D array so downstream code can use
        # len(self.symbols) and iterate directly without extra flattening.
        object.__setattr__(
            self,
            "symbols",
            np.asarray(self.symbols, dtype=object).flatten(),
        )

        body = circuit_body_to_qasm(self.source_circuit, precision=self.precision)
        self.set_circuit_bodies((((), body),))
        self._build_templates()
        self.set_measurement_bodies(())
        self.set_measurement_groups(())

    def _build_templates(self):
        """Build :class:`QASMTemplate` instances from current bodies and symbols."""
        symbol_names = tuple(str(s) for s in self.symbols)
        templates = tuple(
            (tag, build_template(body, symbol_names))
            for tag, body in self.circuit_body_qasms
        )
        object.__setattr__(self, "circuit_body_templates", templates)

    def set_circuit_bodies(
        self,
        bodies: tuple[tuple[QASMTag, str], ...],
        symbol_names: tuple[str, ...] | None = None,
    ) -> Self:
        """Set circuit body QASMs, optionally rebuilding templates.

        Args:
            bodies: Tagged QASM body strings.
            symbol_names: If provided, the bodies still contain symbolic
                parameters and templates are rebuilt for fast parameter
                binding.  If ``None``, templates are left unchanged (use
                this when the bodies are already fully bound).
        """
        object.__setattr__(self, "circuit_body_qasms", bodies)
        if symbol_names is not None:
            templates = tuple(
                (tag, build_template(body, symbol_names)) for tag, body in bodies
            )
            object.__setattr__(self, "circuit_body_templates", templates)
        return self

    def set_measurement_bodies(self, bodies: tuple[tuple[QASMTag, str], ...]) -> Self:
        object.__setattr__(self, "measurement_qasms", bodies)
        return self

    def set_measurement_groups(
        self, measurement_groups: tuple[tuple[object, ...], ...]
    ) -> Self:
        object.__setattr__(self, "measurement_groups", measurement_groups)
        return self
