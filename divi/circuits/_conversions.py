# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""PennyLane QuantumScript → Qiskit DAGCircuit conversion and DAG → parametric QASM2 emission."""

from collections.abc import Mapping
from typing import cast

import numpy as np
import pennylane as qp
import sympy as sp
from pennylane.tape import QuantumScript
from qiskit import transpile
from qiskit.circuit import (
    Parameter,
    ParameterExpression,
    QuantumCircuit,
    QuantumRegister,
)
from qiskit.circuit.library import (
    CCXGate,
    CRXGate,
    CRYGate,
    CRZGate,
    CSwapGate,
    CXGate,
    CZGate,
    HGate,
    IGate,
    PhaseGate,
    RXGate,
    RYGate,
    RZGate,
    SdgGate,
    SGate,
    StatePreparation,
    SwapGate,
    SXdgGate,
    SXGate,
    TdgGate,
    TGate,
    U2Gate,
    UGate,
    UnitaryGate,
    XGate,
    YGate,
    ZGate,
)
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import DEFAULT_PRECISION, MetaCircuit
from divi.hamiltonians import to_spo

# Supported PennyLane op.name set for local QuantumScript → Qiskit conversion.
_PL_TO_QISKIT_GATE = {
    "Identity": IGate,
    "PauliX": XGate,
    "PauliY": YGate,
    "PauliZ": ZGate,
    "Hadamard": HGate,
    "S": SGate,
    "Adjoint(S)": SdgGate,
    "SX": SXGate,
    "Adjoint(SX)": SXdgGate,
    "T": TGate,
    "Adjoint(T)": TdgGate,
    "RX": RXGate,
    "RY": RYGate,
    "RZ": RZGate,
    "PhaseShift": PhaseGate,
    "U2": U2Gate,
    "U3": UGate,
    "CNOT": CXGate,
    "CZ": CZGate,
    "CRX": CRXGate,
    "CRY": CRYGate,
    "CRZ": CRZGate,
    "SWAP": SwapGate,
    "Toffoli": CCXGate,
    "CSWAP": CSwapGate,
    "QubitUnitary": UnitaryGate,
    "StatePrep": StatePreparation,
}

# Wire-reversal applies to ops that take a (state)vector indexed by qubit
# ordering — Qiskit and PennyLane disagree on endianness for these.
_REVERSE_WIRES = {"QubitUnitary", "StatePrep"}


def _qscript_to_qiskit_circuit(
    qscript: QuantumScript, register_size: int
) -> QuantumCircuit:
    """Build a Qiskit ``QuantumCircuit`` from a fully-decomposed ``QuantumScript``.

    Every op in ``qscript.operations`` must be in :data:`_PL_TO_QISKIT_GATE`.
    """
    reg = QuantumRegister(register_size)
    qc = QuantumCircuit(reg)
    for op in qscript.operations:
        params = op.parameters
        for idx, p in enumerate(params):
            if isinstance(p, np.ndarray):
                params[idx] = p.tolist()
        qubits = [reg[w] for w in op.wires.labels]
        if op.name in _REVERSE_WIRES:
            qubits.reverse()
        # pyrefly: ignore[bad-argument-type]
        gate = _PL_TO_QISKIT_GATE[op.name](*params)
        qc.append(gate, qubits)
    return qc


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


def _symbolize_trainable_subset(qscript: QuantumScript) -> QuantumScript:
    """Symbolize concrete gate slots named by an explicit, proper-subset
    ``trainable_params`` so the conversion treats them as bindable.

    PennyLane defaults ``trainable_params`` to *all* gate-data slots; a proper
    subset is a deliberate signal that only those slots are the knobs. When such
    a subset points at concrete (non-symbolic) values, those slots are replaced
    with sympy symbols (``p0``, ``p1``, ...) in trainable-index order so the
    downstream conversion exposes them as bindable parameters.

    Full-default trainable sets and already-symbolic slots are left untouched,
    so concrete tapes intended to be evaluated as-is are unaffected.
    """
    all_values = qscript.get_parameters(trainable_only=False)
    trainable = list(qscript.trainable_params)
    if not trainable or len(trainable) >= len(all_values):
        return qscript
    # Operation parameters occupy the leading slots of the flat parameter list;
    # measured-observable (Hamiltonian) coefficients follow them. Only gate
    # slots are bindable knobs — never symbolize an observable's coefficient, or
    # the measured operator would silently change.
    n_op_params = len(
        qscript.get_parameters(trainable_only=False, operations_only=True)
    )
    concrete = [
        i
        for i in trainable
        if i < n_op_params
        and not isinstance(all_values[i], (sp.Expr, ParameterExpression))
    ]
    if not concrete:
        return qscript
    symbols = _fresh_symbols(len(concrete), all_values)
    # PennyLane's bind_new_parameters stub types params as TensorLike; divi binds
    # sympy symbols here — a supported runtime path the stub does not model.
    return qscript.bind_new_parameters(cast(list, symbols), concrete)


def _fresh_symbols(n: int, existing_values: list) -> list[sp.Symbol]:
    """Return ``n`` sympy symbols named ``p0``, ``p1``, … skipping any name
    already present among ``existing_values`` (sympy expressions or Qiskit
    ``ParameterExpression``), so injected symbols never alias a pre-existing
    parameter and get merged into one column by the template renderer."""
    taken: set[str] = set()
    for value in existing_values:
        if isinstance(value, sp.Expr):
            taken |= {str(s) for s in value.free_symbols}
        elif isinstance(value, ParameterExpression):
            taken |= {p.name for p in value.parameters}
    fresh: list[sp.Symbol] = []
    counter = 0
    while len(fresh) < n:
        name = f"p{counter}"
        if name not in taken:
            fresh.append(sp.Symbol(name))
        counter += 1
    return fresh


def _qscript_to_dag(
    qscript: QuantumScript,
) -> tuple[DAGCircuit, tuple[Parameter, ...], dict | None]:
    """Convert a PennyLane ``QuantumScript`` into a Qiskit ``DAGCircuit``.

    The qscript is first decomposed down to locally-supported Qiskit gates.
    Sympy-parametric gate parameters are converted to Qiskit
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
    # Only operation data is walked; measurement observables in divi's
    # pipeline carry float coefficients and contribute no parameters.
    ordered_qiskit_params: list[Parameter] = []
    ordered_sympy_symbols: list[sp.Symbol] = []
    seen_qk: set[Parameter] = set()
    seen_sp: set[sp.Symbol] = set()
    for op in qscript.operations:
        for p in op.data:
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

    # Qiskit's QuantumRegister indexes only accept ints. Non-integer wires
    # (strings, tuples — common in graph problems) must be remapped to
    # 0-indexed ints before conversion.
    wires = qscript.wires
    needs_wire_map = any(not isinstance(w, int) for w in wires) or set(wires) != set(
        range(len(wires))
    )
    wire_map: dict | None = None
    if needs_wire_map:
        wire_map = {w: i for i, w in enumerate(wires)}
        mapped_qscripts, _ = qp.map_wires(qscript, wire_map=wire_map)
        qscript = mapped_qscripts[0]

    # Decompose gates outside the supported set.  Mirror of
    # _circuit_body_to_qasm's qp.transforms.decompose call. Do it up front so
    # the local mapper always sees gates it recognises.
    just_ops = QuantumScript(qscript.operations)
    [decomposed_qscript], _ = qp.transforms.decompose(
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

    qc = _qscript_to_qiskit_circuit(
        decomposed_qscript, register_size=len(qscript.wires)
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
    return f"{value:.{precision}f}"


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


def _format_bound_param(value: float, precision: int) -> str:
    """Format a bound numeric gate parameter (a radian angle) for QASM substitution.

    Renders to *precision* decimal places, strips trailing zeros and dots, and
    normalises negative zero to ``"0"``. Angles below ``10 ** -precision`` round
    toward ``"0"`` (≈5e-9 rad at the default 8 places — physically negligible);
    scale features to O(1) if sub-precision magnitudes must be represented.

    Finiteness is enforced at the binding-stage ingestion boundary
    (:func:`_assert_finite` over ``param_sets``/``feature_batch``), so the
    env-sourced values this renders are finite; it adds no per-value guard of
    its own. DAG-serialised values are guarded separately in
    :func:`_format_gate_param`.
    """
    value = float(value)
    s = f"{value:.{precision}f}".rstrip("0").rstrip(".")
    return "0" if s in {"-0", ""} else s


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
                param = param.bind({k: substitution[k] for k in shared})
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


def qscript_to_meta(
    qscript: QuantumScript,
    precision: int = DEFAULT_PRECISION,
    parameter_order: tuple[Parameter, ...] | None = None,
    was_multi_obs: bool | None = None,
):
    """Shared helper: convert a PennyLane ``QuantumScript`` to a ``MetaCircuit``.

    Used by :class:`~divi.pipeline.stages.PennyLaneSpecStage` and by the
    program-layer factories in ``divi.qprog.algorithms``.  Builds the
    circuit body from the qscript and derives the measurement observable
    (:class:`~qiskit.quantum_info.SparsePauliOp`) or measured-wire tuple
    from the qscript's single measurement.

    Args:
        qscript: PennyLane ``QuantumScript``.  Accepts:

            * a single ``probs``/``counts`` measurement;
            * one or more ``expval`` measurements — sets ``observable`` to
              a ``tuple[SparsePauliOp, ...]``, one entry per measurement.
              Mixing ``expval`` with ``probs``/``counts`` in one
              ``QuantumScript`` is not supported.
        precision: ``MetaCircuit.precision`` for numeric gate parameters.
        parameter_order: Explicit parameter ordering for the resulting
            ``MetaCircuit``.  Use when the qscript's first-appearance order
            doesn't match ``env.param_sets`` columns (e.g. ansatz builds
            gates in a different order than the flat weight array).
            When ``None``, ordering is inferred from the qscript
            (first appearance).
        was_multi_obs: Optional override for the resulting MetaCircuit's
            ``_was_multi_obs`` flag.  When ``None`` (default), inferred
            from the script: more than one ``expval`` measurement → ``True``,
            otherwise ``False``.  Pass ``True`` when the higher-level
            caller knows the user opted into the multi-observable API
            (e.g. ``observable=[O]`` in
            :class:`~divi.qprog.algorithms.TimeEvolution`) even when only
            one expval ends up in the script.
    """
    measurements = list(qscript.measurements)

    qscript = _symbolize_trainable_subset(qscript)
    dag, inferred_params, _ = _qscript_to_dag(qscript)

    params = parameter_order if parameter_order is not None else inferred_params

    observable: tuple[SparsePauliOp, ...] | None = None
    measured_wires = None

    expval_measurements = [
        m for m in measurements if isinstance(m, qp.measurements.ExpectationMP)
    ]
    if expval_measurements:
        if len(expval_measurements) != len(measurements):
            raise ValueError(
                "qscript_to_meta: mixing `expval` with `probs`/`counts` "
                "measurements in a single QuantumScript is not supported."
            )
        ops: list[SparsePauliOp] = []
        for m in expval_measurements:
            if m.obs is None:
                raise ValueError(
                    "ExpectationMP without an observable is not supported."
                )
            ops.append(to_spo(m.obs, wires=qscript.wires))
        observable = tuple(ops)
    elif measurements:
        first = measurements[0]
        if isinstance(first, (qp.measurements.ProbabilityMP, qp.measurements.CountsMP)):
            target_wires = first.wires if len(first.wires) else qscript.wires
            measured_wires = tuple(qscript.wires.index(w) for w in target_wires)

    if was_multi_obs is None:
        was_multi_obs = len(expval_measurements) > 1

    return MetaCircuit(
        circuit_bodies=(((), dag),),
        parameters=params,
        observable=observable,
        measured_wires=measured_wires,
        precision=precision,
        _was_multi_obs=was_multi_obs,
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
