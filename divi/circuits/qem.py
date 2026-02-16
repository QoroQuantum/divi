# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from functools import partial

import cirq
import numpy as np
from cirq.circuits.circuit import Circuit
from mitiq.zne import combine_results, construct_circuits
from mitiq.zne.inference import Factory

from divi.circuits._cirq import ExtendedQasmParser
from divi.typing import QASMTag


class QEMProtocol(ABC):
    """
    Abstract Base Class for Quantum Error Mitigation (QEM) protocols.

    All concrete QEM protocols should inherit from this class and implement
    the abstract methods and properties. This ensures a consistent interface
    across different mitigation techniques.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def modify_circuit(self, cirq_circuit: Circuit) -> Sequence[Circuit]:
        """
        Modifies a given Cirq circuit into one or more new circuits
        required by the QEM protocol.

        For example, a Zero Noise Extrapolation (ZNE) protocol might
        produce multiple scaled versions of the input circuit. A simple
        mitigation protocol might return the original circuit unchanged.

        Args:
            cirq_circuit (cirq.Circuit): The input quantum circuit to be modified.

        Returns:
            Sequence[cirq.Circuit]: A sequence (e.g., list or tuple) of
                                    Cirq circuits to be executed.
        """
        pass

    @abstractmethod
    def postprocess_results(self, results: Sequence[float]) -> float:
        """
        Applies post-processing (e.g., extrapolation, filtering) to the
        results obtained from executing the modified circuits.

        This method takes the raw output from quantum circuit executions
        (typically a sequence of expectation values or probabilities) and
        applies the core error mitigation logic to produce a single,
        mitigated result.

        Args:
            results (Sequence[float]): A sequence of floating-point results,
                                       corresponding to the executions of the
                                       circuits returned by `modify_circuit`.

        Returns:
            float: The single, mitigated result after post-processing.
        """
        pass


class _NoMitigation(QEMProtocol):
    """
    A dummy default mitigation protocol.
    """

    @property
    def name(self) -> str:
        return "NoMitigation"

    def modify_circuit(self, cirq_circuit: Circuit) -> Sequence[Circuit]:
        # Identity, do nothing
        return [cirq_circuit]

    def postprocess_results(self, results: Sequence[float]) -> float:
        """
        Returns the single result provided, ensuring only one result is given.

        If multiple results are provided, it raises a RuntimeError, as this
        protocol expects a single measurement outcome for its input circuit.

        Args:
            results (Sequence[float]): A sequence containing a single floating-point result.

        Returns:
            float: The single result from the sequence.

        Raises:
            RuntimeError: If more than one result is provided.
        """
        if len(results) > 1:
            raise RuntimeError("NoMitigation class received multiple partial results.")

        return results[0]


class ZNE(QEMProtocol):
    """
    Implements the Zero Noise Extrapolation (ZNE) quantum error mitigation protocol.

    This protocol uses `Mitiq`'s functionalities to construct noise-scaled
    circuits and then extrapolate to the zero-noise limit based on the
    obtained results.
    """

    def __init__(
        self,
        scale_factors: Sequence[float],
        folding_fn: Callable,
        extrapolation_factory: Factory,
    ):
        """
        Initializes a ZNE protocol instance.

        Args:
            scale_factors (Sequence[float]): A sequence of noise scale factors
                                             to be applied to the circuits. These
                                             factors typically range from 1.0 upwards.
            folding_fn (Callable): A callable (e.g., a `functools.partial` object)
                                   that defines how the circuit should be "folded"
                                   to increase noise. This function must accept
                                   a `cirq.Circuit` and a `float` (scale factor)
                                   as its first two arguments.
            extrapolation_factory (mitiq.zne.inference.Factory): An instance of
                                                                `Mitiq`'s `Factory`
                                                                class, which provides
                                                                the extrapolation method.

        Raises:
            ValueError: If `scale_factors` is not a sequence of numbers,
                        `folding_fn` is not callable, or `extrapolation_factory`
                        is not an instance of `mitiq.zne.inference.Factory`.
        """
        if (
            not isinstance(scale_factors, Sequence)
            or not all(isinstance(elem, (int, float)) for elem in scale_factors)
            or not all(elem >= 1.0 for elem in scale_factors)
        ):
            raise ValueError(
                "scale_factors is expected to be a sequence of real numbers >=1."
            )

        if not isinstance(folding_fn, partial):
            raise ValueError(
                "folding_fn is expected to be of type partial with all parameters "
                "except for the circuit object and the scale factor already set."
            )

        if not isinstance(extrapolation_factory, Factory):
            raise ValueError("extrapolation_fn is expected to be of Factory.")

        self._scale_factors = scale_factors
        self._folding_fn = folding_fn
        self._extrapolation_factory = extrapolation_factory

    @property
    def name(self) -> str:
        return "zne"

    @property
    def scale_factors(self) -> Sequence[float]:
        return self._scale_factors

    @property
    def folding_fn(self):
        return self._folding_fn

    @property
    def extrapolation_factory(self):
        return self._extrapolation_factory

    def modify_circuit(self, cirq_circuit: Circuit) -> Sequence[Circuit]:
        return construct_circuits(
            cirq_circuit,
            scale_factors=self._scale_factors,
            scale_method=self._folding_fn,
        )

    def postprocess_results(self, results: Sequence[float]) -> float:
        return combine_results(
            scale_factors=self._scale_factors,
            results=results,
            extrapolation_method=self._extrapolation_factory.extrapolate,
        )


def _inject_input_declarations(qasm_body: str, symbols: Sequence) -> str:
    """Insert OpenQASM 3.0-style input angle declarations so the parser accepts symbolic parameters."""
    flat = []
    for s in symbols:
        if isinstance(s, np.ndarray):
            flat.extend(s.flatten())
        else:
            flat.append(s)

    if not flat:
        return qasm_body

    decls = "".join(f"input angle[32] {s};\n" for s in flat)

    # Insert after the include line (parser expects declarations before qreg/creg)
    include_marker = 'include "qelib1.inc";'
    if include_marker not in qasm_body:
        return qasm_body
    idx = qasm_body.index(include_marker) + len(include_marker)
    if qasm_body[idx : idx + 1] == "\n":
        idx += 1
    return qasm_body[:idx] + decls + qasm_body[idx:]


def normalize_qasm_after_cirq(qasm_str: str) -> str:
    """
    Normalize QASM string produced by Cirq export for downstream use.

    Collapses repeated newlines, strips line comments, and ensures
    a classical register declaration follows the quantum register.
    """
    qasm_str = re.sub(r"\n+", "\n", qasm_str)
    qasm_str = re.sub(r"^//.*\n?", "", qasm_str, flags=re.MULTILINE)
    qasm_str = re.sub(r"qreg q\[(\d+)\];", r"qreg q[\1];creg c[\1];", qasm_str)
    return qasm_str


def apply_protocol_to_qasm(
    qasm_body: tuple[tuple[QASMTag, str], ...] | tuple[QASMTag, str] | str,
    protocol: QEMProtocol,
    *,
    axis_name: str = "qem",
    symbols: Sequence | None = None,
    parser: type[ExtendedQasmParser] | None = None,
) -> tuple[tuple[QASMTag, str], ...]:
    """
    Apply a QEM protocol to tagged QASM body/bodies and return updated tagged bodies.

    Parses the QASM to Cirq, runs the protocol's modify_circuit, then exports
    each modified circuit back to QASM and normalizes the output.

    When the QASM contains symbolic parameters (e.g. w_0, w_1), pass ``symbols``
    so that input angle declarations are injected before parsing; the parser
    then recognizes those names in gate arguments.

    Args:
        qasm_body: OpenQASM payload as either:
            - ``((tag_tuple, body), ...)`` for one or more tagged bodies, or
            - ``(tag_tuple, body)`` where ``tag_tuple`` is metadata and ``body`` is OpenQASM 2.0, or
            - ``body`` (legacy).
        protocol: QEM protocol whose modify_circuit is applied.
        symbols: Optional sequence of sympy symbols (or array of) used in the circuit.
            When provided, corresponding ``input angle[32] name;`` lines are inserted
            into the QASM so the parser can parse symbolic parameters.
        parser: Parser class to use for QASM → Cirq. Defaults to ExtendedQasmParser.

    Returns:
        Tuple of ``(updated_tag, updated_body)`` pairs.
    """
    if isinstance(qasm_body, str):
        tagged_bodies: tuple[tuple[QASMTag, str], ...] = (((), qasm_body),)
    elif (
        isinstance(qasm_body, tuple)
        and len(qasm_body) == 2
        and isinstance(qasm_body[0], tuple)
        and isinstance(qasm_body[1], str)
    ):
        tagged_bodies = (qasm_body,)
    else:
        tagged_bodies = tuple(qasm_body)

    cls = parser or ExtendedQasmParser
    updated_bodies: list[tuple[QASMTag, str]] = []

    for existing_tag, body in tagged_bodies:
        parsed_body = (
            _inject_input_declarations(body, symbols) if symbols is not None else body
        )
        cirq_circuit = cls().parse(parsed_body).circuit
        modified_circuits = list(protocol.modify_circuit(cirq_circuit))
        if len(modified_circuits) == 0:
            raise RuntimeError(
                f"QEM protocol '{protocol.name}' returned no circuits from modify_circuit."
            )

        for qem_idx, modified_circuit in enumerate(modified_circuits):
            updated_tag = (
                *existing_tag,
                (axis_name, qem_idx),
            )
            updated_body = normalize_qasm_after_cirq(cirq.qasm(modified_circuit))
            updated_bodies.append((updated_tag, updated_body))

    return tuple(updated_bodies)
