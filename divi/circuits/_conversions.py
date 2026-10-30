# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Qiskit DAG conversion helpers and parametric QASM2 emission."""

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import sympy as sp
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import SparsePauliOp

from divi.circuits._core import DEFAULT_PRECISION

# QASM2 gate name per Qiskit instruction name, for the body emitter.  Matches
# the old OPENQASM_GATES values.  Any instruction outside this map is an
# error — the caller is responsible for ensuring the DAG only contains
# supported gates (typically by running PennyLane decompose beforehand).
_QISKIT_TO_QASM2 = {
    "id": "id",
    "x": "x",
    "y": "y",
    "z": "z",
    "h": "h",
    "s": "s",
    "sdg": "sdg",
    "sx": "sx",
    "sxdg": "sxdg",
    "t": "t",
    "tdg": "tdg",
    "rx": "rx",
    "ry": "ry",
    "rz": "rz",
    "p": "u1",  # QASM2's qelib1.inc spells PhaseShift as u1
    "u2": "u2",
    "u": "u3",  # QASM2's qelib1.inc spells U(θ,φ,λ) as u3
    "cx": "cx",
    "cz": "cz",
    "crx": "crx",
    "cry": "cry",
    "crz": "crz",
    "swap": "swap",
    "rzz": "rzz",
    "rxx": "rxx",
    "ccx": "ccx",
    "cswap": "cswap",
}


def _sympy_to_qiskit(
    expr: sp.Expr,
    mapping: Mapping[sp.Symbol, Parameter],
) -> ParameterExpression | float:
    """Convert a sympy expression into a Qiskit ``ParameterExpression`` / float.

    Qiskit's ``ParameterExpression`` constructor accepts a string
    expression and resolves parameter names via a symbol map.
    ``str(sympy_expr)`` produces syntax that Qiskit's internal parser
    understands — arithmetic, powers, and transcendentals all
    round-trip cleanly.
    """
    if isinstance(expr, (int, float, np.floating, np.integer)):
        return float(expr)
    free_symbols = cast(set[sp.Symbol], expr.free_symbols)
    if not free_symbols:
        return float(expr)
    try:
        name_map = {mapping[s].name: mapping[s] for s in free_symbols}
    except KeyError:
        missing = free_symbols - mapping.keys()
        raise ValueError(
            f"Unmapped sympy symbol(s) {missing!r}; mapping covers "
            f"{list(mapping.keys())}"
        ) from None
    try:
        return ParameterExpression(name_map, str(expr))
    except (RuntimeError, TypeError) as e:
        raise NotImplementedError(
            f"Cannot convert sympy expression {expr!r} to a Qiskit "
            f"ParameterExpression — its parser rejected it: {e}"
        ) from e


def _format_gate_param(
    param: ParameterExpression | float | int,
    precision: int,
) -> str:
    """Format a gate parameter for a body-only parametric QASM2 string.

    Numeric values are rejected if non-finite — this is the universal leaf for
    DAG-to-QASM serialisation (the slow/eager binding paths and circuit-literal
    angles), so guarding here makes finiteness enforcement uniform alongside the
    ingestion-boundary :func:`_assert_finite`.
    """
    if isinstance(param, ParameterExpression):
        # str() gives Qiskit's own serialisation, which renders bare
        # Parameters as their name and composite expressions using standard
        # arithmetic syntax (QASM2-compatible: +, -, *, /, **, sin, cos…).
        return str(param)
    value = float(param)
    if not np.isfinite(value):
        raise ValueError(
            f"Cannot serialise non-finite gate parameter {value!r} to QASM; "
            f"check the circuit for NaN or Inf angles."
        )
    return _format_bound_param(value, precision)


def _real_float_array(values, *, source: str) -> np.ndarray:
    """``values`` as a float64 array, rejecting a non-zero imaginary part.

    A complex array whose imaginary part is zero is accepted and cast; anything
    else complex raises rather than being silently truncated to its real part.
    """
    array = np.asarray(values)
    if np.iscomplexobj(array):
        if np.any(array.imag != 0):
            raise ValueError(
                f"{source} must be real; got complex values with a non-zero "
                "imaginary part."
            )
        array = array.real
    return array.astype(np.float64, copy=False)


def _assert_finite(values: np.ndarray, *, source: str) -> None:
    """Reject NaN/Inf gate parameters at the value-ingestion boundary.

    Run on a binding stage's incoming value matrix (``env.param_sets`` or
    ``env.feature_batch``) before it is fanned across circuit bodies. Validating
    here — rather than in any single render leaf — means every downstream path
    (template, fast, slow/eager DAG, and backend templates) rejects non-finite
    gate parameters uniformly.
    """
    if not np.isfinite(values).all():
        raise ValueError(
            f"Cannot bind non-finite gate parameters: {source} contains NaN or "
            f"Inf. Check the feature batch / parameter values for missing data, "
            f"divide-by-zero, or overflow in preprocessing."
        )


_FIXED_POINT_MIN = 0.1


def _format_bound_param(value: float, precision: int) -> str:
    """Format a numeric gate parameter (a radian angle) for QASM, per
    :attr:`~divi.circuits.MetaCircuit.precision`, stripping trailing
    fractional zeros and rendering negative zero as ``"0"``.

    Finiteness is enforced at the binding-stage ingestion boundary
    (:func:`_assert_finite` over ``param_sets``/``feature_batch``), so the
    env-sourced values this renders are finite; it adds no per-value guard of
    its own. DAG-serialised values are guarded separately in
    :func:`_format_gate_param`.
    """
    value = float(value)
    if value != 0 and abs(value) < _FIXED_POINT_MIN:
        return f"{value:.{precision}g}"
    s = f"{value:.{precision}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def _bind_op_params(op, substitution: dict):
    """Return ``op`` with any of ``substitution``'s parameters bound in its
    expressions; the original is returned untouched when none appear."""
    if not op.params:
        return op
    changed = False
    new_params = []
    for param in op.params:
        if isinstance(param, ParameterExpression):
            shared = substitution.keys() & set(param.parameters)
            if shared:
                param = param.bind({k: float(substitution[k]) for k in shared})
                changed = True
        new_params.append(param)
    if not changed:
        return op
    bound_op = op.copy()
    bound_op.params = new_params
    return bound_op


def bind_parameters_in_dag(dag: DAGCircuit, substitution: dict) -> DAGCircuit:
    """Rebuild ``dag`` with the parameters in ``substitution`` bound to values,
    leaving every other parameter symbolic.

    Walks the DAG node-by-node and binds each gate's
    :class:`~qiskit.circuit.ParameterExpression` in place — no round-trip
    through a :class:`~qiskit.circuit.QuantumCircuit`. ``substitution`` maps
    :class:`~qiskit.circuit.Parameter` to the value to bind.
    """
    bound = dag.copy_empty_like()
    for node in dag.topological_op_nodes():
        bound.apply_operation_back(
            _bind_op_params(node.op, substitution), node.qargs, node.cargs
        )
    return bound


def dag_to_qasm_body(dag: DAGCircuit, precision: int = DEFAULT_PRECISION) -> str:
    """Emit a body-only parametric OpenQASM 2.0 string from a DAG.

    No preamble, no ``qreg``/``creg`` declarations — just gate instructions,
    one per line.  Parametric gate parameters are rendered via their
    :class:`~qiskit.circuit.ParameterExpression` ``str()`` form, producing
    identifier placeholders that
    :class:`~divi.circuits.QASMTemplate` substitutes at bind time.  Numeric
    parameters render at *precision*.

    Args:
        dag: Qiskit DAG containing only gates from the internal
            ``_QISKIT_TO_QASM2`` whitelist (single quantum register assumed).
        precision: See :attr:`~divi.circuits.MetaCircuit.precision`.

    Raises:
        ValueError: if *dag* contains an instruction outside the
            ``_QISKIT_TO_QASM2`` whitelist.
    """
    qubit_index = {q: i for i, q in enumerate(dag.qubits)}
    parts: list[str] = []
    for node in dag.topological_op_nodes():
        inst_name = node.op.name
        try:
            gate = _QISKIT_TO_QASM2[inst_name]
        except KeyError as e:
            hint = (
                " `barrier` is emitted by QuantumCircuit.measure_all(); use "
                "explicit `measure(i, i)` on a circuit with a classical register "
                "instead."
                if inst_name == "barrier"
                else " Decompose to basis gates before calling dag_to_qasm_body."
            )
            raise ValueError(
                f"Instruction {inst_name!r} not supported by the QASM body "
                f"emitter.{hint}"
            ) from e
        if node.op.params:
            args = (
                "("
                + ",".join(_format_gate_param(p, precision) for p in node.op.params)
                + ")"
            )
        else:
            args = ""
        qubits = ",".join(f"q[{qubit_index[q]}]" for q in node.qargs)
        parts.append(f"{gate}{args} {qubits};\n")
    return "".join(parts)


_PAULI_CHAR_LOOKUP = np.array(list("IXZY"), dtype="U1")


def _sparse_pauli_op_to_ham_string(op: SparsePauliOp) -> str:
    """Render a :class:`~qiskit.quantum_info.SparsePauliOp` as the ``;``-separated dense Pauli
    string format used by backend ``ham_ops`` artifacts.

    The backend contract is big-endian (qubit 0 on the left).  Coefficients
    are intentionally dropped — the backend computes ``<ψ|P|ψ>`` per term and
    the caller recombines with coefficients.

    Builds the dense Pauli strings directly from the SPO's symplectic
    ``(x, z)`` arrays — qubit ``q`` indexed as character ``q`` (big-endian).
    Skips ``PauliList.to_labels`` (Python-level, ~3μs/term for wide
    observables) and the subsequent per-string reverse.
    """
    x_arr = op.paulis.x  # bool[N_terms, n_qubits]
    z_arr = op.paulis.z
    n_terms, n_qubits = x_arr.shape
    if n_terms == 0:
        return ""
    # I=0, X=1, Z=2, Y=3 — encoded as (z<<1 | x) so a single uint8 lookup
    # yields the right character per (term, qubit) cell.
    indices = (z_arr.astype(np.uint8) << 1) | x_arr.astype(np.uint8)
    chars = np.ascontiguousarray(_PAULI_CHAR_LOOKUP[indices])
    rows = chars.view(f"U{n_qubits}").reshape(-1)
    return ";".join(rows)


def measurement_qasms_from_groups(
    measurement_groups: Sequence[tuple[str, ...]],
    n_qubits: int,
    measure_all: bool = True,
) -> list[str]:
    """Emit body-only measurement QASM per commuting observable group.

    For each QWC group, determines the measurement basis per qubit from
    the big-endian Pauli labels and emits the appropriate diagonalising
    gates (H for X, Sdg+H for Y, nothing for Z/I) followed by
    ``measure q[i] -> c[i]`` instructions.  No PennyLane dependency.

    Args:
        measurement_groups: Tuple of tuples of big-endian Pauli label
            strings, one tuple per commuting group.
        n_qubits: Total qubit count.
        measure_all: If ``True``, measure all qubits.  If ``False``,
            restrict to qubits active in the group; an all-identity group
            (no active qubit) raises ``ValueError``, since it has nothing to
            measure and should not reach this function in the normal pipeline.
    """
    qasms: list[str] = []
    measure_all_qasm = "".join(f"measure q[{q}] -> c[{q}];\n" for q in range(n_qubits))
    for group in measurement_groups:
        for label in group:
            if len(label) > n_qubits:
                raise ValueError(
                    f"Pauli label {label!r} spans {len(label)} qubits but the "
                    f"register has only {n_qubits}."
                )

        # Per-qubit basis from the last non-I Pauli on that qubit; QWC guarantees
        # each qubit has at most one non-I Pauli across the group's labels.
        active = np.empty(0, dtype=np.intp)
        basis = np.empty(0, dtype=np.uint8)
        if group:
            chars = np.frombuffer(
                "".join(label.ljust(n_qubits, "I") for label in group).encode(),
                dtype=np.uint8,
            ).reshape(len(group), n_qubits)
            non_identity = chars != ord("I")
            active = np.flatnonzero(non_identity.any(axis=0))
            last_row = len(group) - 1 - np.argmax(non_identity[::-1], axis=0)
            basis = chars[last_row[active], active]

        # Emit diagonalising gates; Z and I need no rotation.
        diag_parts: list[str] = []
        for q, b in zip(active.tolist(), basis.tolist()):
            if b == ord("X"):
                diag_parts.append(f"h q[{q}];\n")
            elif b == ord("Y"):
                diag_parts.append(f"sdg q[{q}];\nh q[{q}];\n")
        diag_qasm = "".join(diag_parts)

        if measure_all:
            measure_qasm = measure_all_qasm
        elif not len(active):
            # Constants are stripped upstream, so this only happens on a direct
            # misuse (a constant-only group).
            raise ValueError(
                "all-identity group with measure_all=False has no qubit to "
                "measure; pass measure_all=True or drop the constant term."
            )
        else:
            measure_qasm = "".join(
                f"measure q[{q}] -> c[{q}];\n" for q in active.tolist()
            )
        qasms.append(diag_qasm + measure_qasm)
    return qasms
