# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""PennyLane QuantumScript → Qiskit DAGCircuit conversion and DAG → parametric QASM2 emission."""

from collections.abc import Mapping

import numpy as np
import pennylane as qml
import sympy as sp
from pennylane.tape import QuantumScript
from pennylane_qiskit.converter import QISKIT_OPERATION_MAP, circuit_to_qiskit
from qiskit import transpile
from qiskit.circuit import Parameter, ParameterExpression
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import SparsePauliOp

from divi.circuits._core import MetaCircuit

# Supported PennyLane op.name set, taken from ``pennylane-qiskit``'s gate
# table.  Using this directly (instead of maintaining our own mirror) means
# any gate ``pennylane-qiskit`` adds is automatically supported here, and
# stays symmetric with ``_qiskit_spec_stage.py`` which uses the inverse
# direction via ``qml.from_qiskit``.
_PL_TO_QISKIT_GATE = QISKIT_OPERATION_MAP

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
    (backed by symengine) understands — arithmetic, powers, and
    transcendentals all round-trip cleanly.
    """
    if isinstance(expr, (int, float, np.floating, np.integer)):
        return float(expr)
    if not expr.free_symbols:
        return float(expr)
    missing = expr.free_symbols - mapping.keys()
    if missing:
        raise ValueError(
            f"Unmapped sympy symbol(s) {missing!r}; mapping covers "
            f"{list(mapping.keys())}"
        )
    name_map = {mapping[s].name: mapping[s] for s in expr.free_symbols}
    try:
        return ParameterExpression(name_map, str(expr))
    except (RuntimeError, TypeError) as e:
        raise NotImplementedError(
            f"Cannot convert sympy expression {expr!r} to a Qiskit "
            f"ParameterExpression — symengine rejected it: {e}"
        ) from e


def _qscript_to_dag(
    qscript: QuantumScript,
) -> tuple[DAGCircuit, tuple[Parameter, ...], dict | None]:
    """Convert a PennyLane ``QuantumScript`` into a Qiskit ``DAGCircuit``.

    The qscript is first decomposed down to gates present in
    :data:`pennylane_qiskit.converter.QISKIT_OPERATION_MAP`.  Sympy-parametric
    gate parameters are converted to Qiskit
    :class:`~qiskit.circuit.ParameterExpression` via :func:`_sympy_to_qiskit`.

    Returns:
        ``(dag, ordered_parameters, wire_map)``.  ``ordered_parameters``
        preserves the order of symbols as they first appear in the
        qscript's flattened parameter list.  ``wire_map`` is
        ``{original_label: int}`` when non-integer or non-contiguous
        wire labels were remapped; ``None`` when wires were already
        0-indexed ints.
    """
    # Discover parameters in first-appearance order.  Supports both:
    #  - Qiskit Parameter/ParameterExpression (from factories that create
    #    Parameters directly — no sympy, no param_map needed)
    #  - sympy symbols (from PennyLaneSpecStage / QNode probing)
    ordered_qiskit_params: list[Parameter] = []
    ordered_sympy_symbols: list[sp.Symbol] = []
    seen_qk: set[Parameter] = set()
    seen_sp: set[sp.Symbol] = set()
    for p in qscript.get_parameters():
        if isinstance(p, ParameterExpression):
            for qk_param in p.parameters:
                if qk_param not in seen_qk:
                    seen_qk.add(qk_param)
                    ordered_qiskit_params.append(qk_param)
        elif isinstance(p, sp.Expr):
            for s in p.free_symbols:
                if s not in seen_sp:
                    seen_sp.add(s)
                    ordered_sympy_symbols.append(s)

    # Build sympy → Qiskit Parameter mapping for sympy-parametric tapes.
    param_map: dict[sp.Symbol, Parameter] | None = None
    if ordered_sympy_symbols:
        param_map = {s: Parameter(str(s)) for s in ordered_sympy_symbols}

    # Workaround: pennylane_qiskit's circuit_to_qiskit indexes into a
    # QuantumRegister by wire label, which only accepts ints.  Non-integer
    # wires (strings, tuples — common in graph problems) must be remapped
    # to 0-indexed ints before the conversion call.
    wires = qscript.wires
    needs_wire_map = any(not isinstance(w, int) for w in wires) or set(wires) != set(
        range(len(wires))
    )
    wire_map: dict | None = None
    if needs_wire_map:
        wire_map = {w: i for i, w in enumerate(wires)}
        mapped_qscripts, _ = qml.map_wires(qscript, wire_map=wire_map)
        qscript = mapped_qscripts[0]

    # Decompose gates outside the supported set.  Mirror of
    # _circuit_body_to_qasm's qml.transforms.decompose call — pennylane-qiskit
    # itself doesn't auto-decompose, so we do it up front so the subsequent
    # circuit_to_qiskit call always sees gates it recognises.
    just_ops = QuantumScript(qscript.operations)
    [decomposed_qscript], _ = qml.transforms.decompose(
        just_ops, stopping_condition=lambda obj: obj.name in _PL_TO_QISKIT_GATE
    )

    # Substitute sympy params with Qiskit Parameters.  Skipped entirely
    # when the qscript already carries Qiskit Parameter objects (no sympy).
    if ordered_sympy_symbols and param_map:
        new_values: list = []
        indices: list[int] = []
        for i, p in enumerate(decomposed_qscript.get_parameters()):
            if isinstance(p, sp.Expr) and not p.is_Number:
                new_values.append(_sympy_to_qiskit(p, param_map))
                indices.append(i)
        if indices:
            decomposed_qscript = decomposed_qscript.bind_new_parameters(
                new_values, indices
            )

    # Delegate the actual PL-op → Qiskit-gate translation to pennylane-qiskit.
    qc = circuit_to_qiskit(
        decomposed_qscript,
        register_size=len(qscript.wires),
        measure=False,
        diagonalize=False,
    )

    # Decompose any gates outside the QASM2 basis (e.g. rxx/ryy/rzz from
    # Trotter decompositions) so the downstream dag_to_qasm_body emitter
    # never sees instructions it doesn't recognise.  Optimization level 0
    # keeps the pass cheap — just a gate-by-gate substitution.
    qc = transpile(
        qc,
        basis_gates=list(_QISKIT_TO_QASM2.keys()),
        optimization_level=0,
    )

    # Combine both parameter sources: Qiskit-native first, then sympy-converted.
    # In practice only one source is active per call.
    if ordered_sympy_symbols and param_map:
        sympy_params = tuple(param_map[s] for s in ordered_sympy_symbols)
    else:
        sympy_params = ()

    ordered_params = tuple(ordered_qiskit_params) + sympy_params
    return circuit_to_dag(qc), ordered_params, wire_map


def _format_gate_param(
    param: ParameterExpression | float | int,
    precision: int,
) -> str:
    """Format a gate parameter for a body-only parametric QASM2 string."""
    if isinstance(param, ParameterExpression):
        # str() gives Qiskit's own serialisation, which renders bare
        # Parameters as their name and composite expressions using standard
        # arithmetic syntax (QASM2-compatible: +, -, *, /, **, sin, cos…).
        return str(param)
    return f"{float(param):.{precision}f}"


def dag_to_qasm_body(dag: DAGCircuit, precision: int = 8) -> str:
    """Emit a body-only parametric OpenQASM 2.0 string from a DAG.

    No preamble, no ``qreg``/``creg`` declarations — just gate instructions,
    one per line.  Parametric gate parameters are rendered via their
    :class:`~qiskit.circuit.ParameterExpression` ``str()`` form, producing
    identifier placeholders that
    :class:`~divi.circuits.QASMTemplate` substitutes at bind time.  Numeric
    parameters are formatted to *precision* decimal places.

    Args:
        dag: Qiskit DAG containing only gates from the internal
            ``_QISKIT_TO_QASM2`` whitelist (single quantum register assumed).
        precision: Decimal places used for numeric gate parameters.

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
            raise ValueError(
                f"Instruction {inst_name!r} not supported by the QASM body "
                f"emitter — did you forget to decompose before calling "
                f"dag_to_qasm_body?"
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


def observable_to_sparse_pauli_op(
    obs: qml.operation.Operator,
    wires,
) -> SparsePauliOp:
    """Convert a PennyLane observable to a Qiskit :class:`SparsePauliOp`.

    Handles arbitrary wire labels (strings, tuples, non-contiguous ints)
    by resolving through the provided *wires* register.
    ``pennylane_qiskit.mp_to_pauli`` assumes 0-indexed integer wires, so
    we keep a custom implementation here for the general case.

    Coefficients are stored as real floats.  A warning is emitted if any
    coefficient has a non-negligible imaginary part (>1e-10), which would
    indicate a non-Hermitian observable.
    """
    import warnings

    pauli_rep = obs.pauli_rep
    if pauli_rep is None:
        raise ValueError(
            f"Observable {obs!r} has no Pauli representation; cannot "
            f"convert to SparsePauliOp."
        )
    wire_list = list(wires)
    num_qubits = len(wire_list)

    sparse: list[tuple[str, list[int], float]] = []
    for pauli_word, coeff in pauli_rep.items():
        c = complex(coeff)
        if abs(c.imag) > 1e-10:
            warnings.warn(
                f"Observable coefficient {c} has non-negligible imaginary "
                f"part ({c.imag:.2e}); dropping it. This may indicate a "
                f"non-Hermitian observable.",
                stacklevel=2,
            )
        pw_dict = dict(pauli_word)
        if not pw_dict:
            sparse.append(("", [], c.real))
        else:
            pauli_chars = "".join(pw_dict[w] for w in pw_dict)
            qubit_indices = [wire_list.index(w) for w in pw_dict]
            sparse.append((pauli_chars, qubit_indices, c.real))

    return SparsePauliOp.from_sparse_list(sparse, num_qubits=num_qubits)


def sparse_pauli_op_to_pl_observable(
    op: SparsePauliOp,
    wires,
) -> qml.operation.Operator:
    """Convert a Qiskit :class:`SparsePauliOp` back into a PennyLane observable.

    ``pennylane_qiskit.load_pauli_op`` exists but always produces complex
    coefficients (because ``SparsePauliOp`` stores ``complex128``), which
    then propagate through the pipeline and break downstream code that
    expects real floats.  We keep a custom implementation that extracts
    real coefficients explicitly.
    """
    pauli_cls = {"I": qml.Identity, "X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
    wire_list = list(wires)
    terms: list[qml.operation.Operator] = []
    for pauli_str, coeff in zip(op.paulis.to_labels(), op.coeffs):
        ops_for_term = [
            pauli_cls[char](wire_list[i])
            for i, char in enumerate(reversed(pauli_str))
            if char != "I"
        ]
        if not ops_for_term:
            term = qml.Identity(wire_list[0])
        elif len(ops_for_term) == 1:
            term = ops_for_term[0]
        else:
            term = qml.prod(*ops_for_term)

        c = float(np.real(coeff))
        terms.append(c * term if c != 1.0 else term)

    return terms[0] if len(terms) == 1 else qml.sum(*terms)


def sparse_pauli_op_to_ham_string(op: SparsePauliOp) -> str:
    """Render a :class:`SparsePauliOp` as the ``;``-separated dense Pauli
    string format used by backend ``ham_ops`` artifacts.

    The backend contract is big-endian (qubit 0 on the left), matching the
    PennyLane-based :func:`~divi.hamiltonians.convert_hamiltonian_to_pauli_string`
    output.  Coefficients are intentionally dropped — the backend computes
    ``<ψ|P|ψ>`` per term and the caller recombines with coefficients.
    """
    return ";".join(label[::-1] for label in op.paulis.to_labels())


def qscript_to_meta(
    qscript: QuantumScript,
    precision: int = 8,
    parameter_order: tuple[Parameter, ...] | None = None,
):
    """Shared helper: convert a PennyLane ``QuantumScript`` to a ``MetaCircuit``.

    Used by :class:`~divi.pipeline.stages.PennyLaneSpecStage` and by the
    program-layer factories in ``divi.qprog.algorithms``.  Builds the
    circuit body from the qscript and derives the measurement observable
    (:class:`~qiskit.quantum_info.SparsePauliOp`) or measured-wire tuple
    from the qscript's single measurement.

    Args:
        qscript: PennyLane ``QuantumScript`` with exactly one measurement
            (``expval``, ``probs``, or ``counts``).
        precision: ``MetaCircuit.precision`` for numeric gate parameters.
        parameter_order: Explicit parameter ordering for the resulting
            ``MetaCircuit``.  Use when the qscript's first-appearance order
            doesn't match ``env.param_sets`` columns (e.g. ansatz builds
            gates in a different order than the flat weight array).
            When ``None``, ordering is inferred from the qscript
            (first appearance).
    """
    measurement = qscript.measurements[0] if qscript.measurements else None

    dag, inferred_params, _ = _qscript_to_dag(qscript)

    params = parameter_order if parameter_order is not None else inferred_params

    observable = None
    measured_wires = None
    if isinstance(measurement, qml.measurements.ExpectationMP):
        observable = observable_to_sparse_pauli_op(measurement.obs, qscript.wires)
    elif isinstance(
        measurement,
        (qml.measurements.ProbabilityMP, qml.measurements.CountsMP),
    ):
        target_wires = measurement.wires if len(measurement.wires) else qscript.wires
        measured_wires = tuple(qscript.wires.index(w) for w in target_wires)

    return MetaCircuit(
        circuit_bodies=(((), dag),),
        parameters=params,
        observable=observable,
        measured_wires=measured_wires,
        precision=precision,
    )


def measurement_qasms_from_groups(
    measurement_groups: tuple[tuple[str, ...], ...],
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
            restrict to qubits active in the group.
    """
    qasms: list[str] = []
    for group in measurement_groups:
        # Determine per-qubit basis from labels. QWC guarantees each qubit
        # has at most one non-I Pauli across all labels in the group.
        basis = ["I"] * n_qubits
        for label in group:
            for q, char in enumerate(label):
                if char != "I":
                    basis[q] = char

        # Emit diagonalising gates.
        diag_parts: list[str] = []
        for q, b in enumerate(basis):
            if b == "X":
                diag_parts.append(f"h q[{q}];\n")
            elif b == "Y":
                diag_parts.append(f"sdg q[{q}];\nh q[{q}];\n")
            # Z and I: no rotation needed.
        diag_qasm = "".join(diag_parts)

        if measure_all:
            measured = range(n_qubits)
        else:
            measured = [q for q in range(n_qubits) if basis[q] != "I"]

        measure_qasm = "".join(f"measure q[{q}] -> c[{q}];\n" for q in measured)
        qasms.append(diag_qasm + measure_qasm)
    return qasms


__all__ = [
    "dag_to_qasm_body",
    "observable_to_sparse_pauli_op",
    "sparse_pauli_op_to_pl_observable",
    "sparse_pauli_op_to_ham_string",
    "qscript_to_meta",
    "measurement_qasms_from_groups",
]
