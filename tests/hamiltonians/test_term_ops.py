# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for term-manipulation primitives in divi.hamiltonians._term_ops."""

import numpy as np
import pennylane as qp
import pytest
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.quantum_info import Operator, SparsePauliOp

from divi.hamiltonians import _term_ops as term_ops_module
from divi.hamiltonians._term_ops import (
    _clean_hamiltonian_spo,
    _sort_hamiltonian_terms_spo,
    _spo_to_basis_gate_ops,
    _spo_to_qiskit_basis_gates,
    _spo_to_qiskit_basis_gates_numeric,
    _spo_to_qiskit_basis_gates_symbolic,
    _spo_wires,
    generate_empty_spo,
    to_spo,
)


@pytest.fixture
def simple_pl_hamiltonian() -> qp.operation.Operator:
    """Three-term PennyLane Hamiltonian (coefficients 1.0, 2.0, 3.0)."""
    return (1.0 * qp.Z(0) + 2.0 * qp.Z(1) + 3.0 * (qp.Z(0) @ qp.Z(1))).simplify()


@pytest.fixture
def simple_spo() -> SparsePauliOp:
    """Three-term ``SparsePauliOp`` over 2 qubits (coefficients 1, 2, 3)."""
    # Qiskit big-endian: rightmost char is qubit 0.
    return SparsePauliOp.from_list([("IZ", 1.0), ("ZI", 2.0), ("ZZ", 3.0)])


class TestEmptySpo:
    @pytest.mark.parametrize("num_qubits", [0, 1, 4, 17])
    def test_size_is_zero(self, num_qubits):
        """Returned SPO has no rows regardless of register size."""
        spo = generate_empty_spo(num_qubits)
        assert spo.size == 0
        assert spo.num_qubits == num_qubits


class TestSpoConversion:
    def test_to_spo_passthrough(self, simple_spo):
        """An SPO input is returned as-is."""
        assert to_spo(simple_spo) is simple_spo

    def test_to_spo_rejects_non_hermitian_spo(self):
        """Direct SPO inputs cannot bypass observable validation."""
        with pytest.raises(ValueError, match="Hermitian"):
            to_spo(SparsePauliOp.from_list([("Y", 1.0j)]))

    def test_to_spo_from_pl(self, simple_pl_hamiltonian, simple_spo):
        """A PL Hamiltonian converts to the equivalent SPO (modulo simplify)."""
        spo = to_spo(simple_pl_hamiltonian)
        assert spo.simplify() == simple_spo.simplify()

    def test_spo_wires_for_spo(self, simple_spo):
        """Wires for an SPO input are ``range(num_qubits)``."""
        assert _spo_wires(simple_spo) == (0, 1)

    def test_spo_wires_for_pl_uses_op_wires(self):
        """A PL operator's canonical wires are taken from ``op.wires``."""
        op = qp.PauliZ(2) @ qp.PauliX(5)
        assert _spo_wires(op) == tuple(op.wires)


class TestCleanHamiltonianSpo:
    def test_no_identity_returns_input_with_zero_constant(self, simple_spo):
        """A purely non-identity SPO is returned unchanged with constant 0."""
        spo, constant = _clean_hamiltonian_spo(simple_spo)
        assert spo.simplify() == simple_spo.simplify()
        assert constant == 0.0

    def test_identity_only_returns_empty_with_constant(self):
        """An all-identity SPO yields an empty SPO and the summed constant."""
        spo = SparsePauliOp.from_list([("II", 2.5), ("II", 1.5)])
        cleaned, constant = _clean_hamiltonian_spo(spo)
        assert cleaned.size == 0
        assert constant == pytest.approx(4.0)

    def test_mixed_partitions_constant_from_non_identity(self):
        """Mixed input: constant is summed, non-identity rows survive."""
        spo = SparsePauliOp.from_list([("IX", 2.0), ("II", 3.0), ("ZI", 1.0)])
        cleaned, constant = _clean_hamiltonian_spo(spo)
        assert constant == pytest.approx(3.0)
        expected = SparsePauliOp.from_list([("IX", 2.0), ("ZI", 1.0)])
        assert cleaned.simplify() == expected.simplify()

    def test_complex_imaginary_constant_dropped(self):
        """Identity rows with imaginary coefficient: only the real part is kept."""
        spo = SparsePauliOp.from_list([("II", 4.0 + 0.0j), ("ZI", 1.0)])
        _, constant = _clean_hamiltonian_spo(spo)
        assert constant == pytest.approx(4.0)

    def test_returns_empty_for_empty_input(self):
        """Clean of an empty SPO is the empty SPO with zero constant."""
        spo, constant = _clean_hamiltonian_spo(generate_empty_spo(3))
        assert spo.size == 0
        assert constant == 0.0


class TestSortHamiltonianTermsSpo:
    def test_absolute_order_sorts_by_signed_coefficient(self):
        """``order='absolute'`` sorts by the literal coefficient (ascending)."""
        spo = SparsePauliOp.from_list([("IIZ", 0.5), ("IZI", -0.3), ("ZII", 0.1)])
        result = _sort_hamiltonian_terms_spo(spo, order="absolute")
        assert list(result.coeffs.real) == pytest.approx([-0.3, 0.1, 0.5])

    def test_magnitude_order_sorts_by_absolute_value(self):
        """``order='magnitude'`` sorts by ``|coeff|`` (ascending)."""
        spo = SparsePauliOp.from_list([("IIZ", 0.5), ("IZI", -0.3), ("ZII", 0.1)])
        result = _sort_hamiltonian_terms_spo(spo, order="magnitude")
        assert [abs(c) for c in result.coeffs.real] == pytest.approx([0.1, 0.3, 0.5])

    def test_single_row_passes_through(self):
        """A single-row SPO is returned unchanged."""
        spo = SparsePauliOp.from_list([("Z", 7.0)])
        result = _sort_hamiltonian_terms_spo(spo)
        assert result is spo

    def test_negative_coefficients_preserved(self):
        """Sorting preserves coefficient signs in signed-ascending order."""
        spo = SparsePauliOp.from_list([("IZ", -2.0), ("ZI", 1.0)])
        result = _sort_hamiltonian_terms_spo(spo, order="absolute")
        assert list(result.coeffs.real) == [-2.0, 1.0]


class TestSpoToBasisGateOps:
    def test_single_z_emits_rz(self):
        """A single ``Z`` term emits one ``RZ`` rotation on the right wire."""
        spo = SparsePauliOp.from_list([("Z", 0.5)])
        ops = _spo_to_basis_gate_ops(spo, time=0.7, wires=[3])
        assert len(ops) == 1
        assert isinstance(ops[0], qp.RZ)
        assert ops[0].wires.tolist() == [3]
        # PauliRot decomposition uses theta = 2 * time * c.
        assert float(ops[0].parameters[0]) == pytest.approx(2 * 0.7 * 0.5)

    def test_single_x_emits_rx(self):
        spo = SparsePauliOp.from_list([("X", 1.0)])
        ops = _spo_to_basis_gate_ops(spo, time=0.4, wires=[0])
        assert len(ops) == 1
        assert isinstance(ops[0], qp.RX)

    def test_zz_decomposes_to_cnot_staircase(self):
        """A two-qubit ZZ term decomposes to CNOT–RZ–CNOT."""
        spo = SparsePauliOp.from_list([("ZZ", 1.0)])
        ops = _spo_to_basis_gate_ops(spo, time=0.3, wires=[0, 1])
        gate_names = [op.name for op in ops]
        assert gate_names == ["CNOT", "RZ", "CNOT"]

    def test_identity_term_skipped(self):
        """All-identity rows produce no gates."""
        spo = SparsePauliOp.from_list([("II", 1.0)])
        ops = _spo_to_basis_gate_ops(spo, time=0.3, wires=[0, 1])
        assert ops == []

    def test_single_y_emits_ry(self):
        """A single ``Y`` term emits one ``RY`` rotation on the right wire."""
        spo = SparsePauliOp.from_list([("Y", 0.5)])
        ops = _spo_to_basis_gate_ops(spo, time=0.7, wires=[2])
        assert len(ops) == 1
        assert isinstance(ops[0], qp.RY)
        assert ops[0].wires.tolist() == [2]
        assert float(ops[0].parameters[0]) == pytest.approx(2 * 0.7 * 0.5)

    @pytest.mark.parametrize(
        "label,coeff,time,wires",
        [
            ("Y", 0.7, 0.3, [0]),
            ("YZ", 0.4, 0.5, [0, 1]),
            ("YY", 0.3, 0.5, [0, 1]),
            ("XYZ", 0.2, 0.25, [0, 1, 2]),
        ],
    )
    def test_unitary_matches_pauli_rot(self, label, coeff, time, wires):
        """Emitted basis-gate sequence matches ``qp.PauliRot`` unitary."""
        spo = SparsePauliOp.from_list([(label, coeff)])
        ops = _spo_to_basis_gate_ops(spo, time=time, wires=wires)
        actual = np.eye(2 ** len(wires), dtype=complex)
        for op in ops:
            actual = qp.matrix(op, wire_order=wires) @ actual
        # Qiskit big-endian: rightmost char is qubit 0; PauliRot reads left-to-right.
        pl_label = label[::-1]
        expected = qp.matrix(
            qp.PauliRot(2 * time * coeff, pl_label, wires=wires),
            wire_order=wires,
        )
        assert np.allclose(actual, expected)


def _build_numeric(
    spo: SparsePauliOp, time: float, qubits, n_qubits: int
) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    _spo_to_qiskit_basis_gates_numeric(qc, spo, time, qubits)
    return qc


def _build_symbolic(spo: SparsePauliOp, time, qubits, n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    _spo_to_qiskit_basis_gates_symbolic(qc, spo, time, qubits)
    return qc


class TestSpoToQiskitBasisGatesDispatch:
    """Verify the public dispatcher routes by ``time`` type."""

    @pytest.fixture
    def spo(self) -> SparsePauliOp:
        return SparsePauliOp.from_list([("ZZ", 0.5)])

    @pytest.mark.parametrize(
        "time",
        [0.5, 1, np.float32(0.3), np.float64(0.7), np.int64(2)],
        ids=["python_float", "python_int", "np_float32", "np_float64", "np_int64"],
    )
    def test_numeric_time_routes_to_numeric_path(self, mocker, spo, time):
        spy_numeric = mocker.spy(term_ops_module, "_spo_to_qiskit_basis_gates_numeric")
        spy_legacy = mocker.spy(term_ops_module, "_spo_to_qiskit_basis_gates_symbolic")
        qc = QuantumCircuit(2)
        _spo_to_qiskit_basis_gates(qc, spo, time, [0, 1])
        assert spy_numeric.call_count == 1
        assert spy_legacy.call_count == 0

    def test_parameter_routes_to_legacy_path(self, mocker, spo):
        spy_numeric = mocker.spy(term_ops_module, "_spo_to_qiskit_basis_gates_numeric")
        spy_legacy = mocker.spy(term_ops_module, "_spo_to_qiskit_basis_gates_symbolic")
        qc = QuantumCircuit(2)
        _spo_to_qiskit_basis_gates(qc, spo, Parameter("t"), [0, 1])
        assert spy_numeric.call_count == 0
        assert spy_legacy.call_count == 1

    def test_parameter_expression_routes_to_legacy_path(self, mocker, spo):
        spy_legacy = mocker.spy(term_ops_module, "_spo_to_qiskit_basis_gates_symbolic")
        t = Parameter("t")
        qc = QuantumCircuit(2)
        _spo_to_qiskit_basis_gates(qc, spo, 2 * t + 1, [0, 1])
        assert spy_legacy.call_count == 1


class TestSpoToQiskitBasisGatesNumericEdgeCases:
    """Edge cases for the numeric path."""

    def test_empty_spo_is_noop(self):
        qc = _build_numeric(generate_empty_spo(2), 0.5, [0, 1], n_qubits=2)
        assert len(qc.data) == 0

    def test_identity_only_row_is_skipped(self):
        # II contributes only a global phase; gate emission should match a
        # circuit built from the non-identity terms alone.
        spo_with_identity = SparsePauliOp.from_list([("II", 1.0), ("ZZ", 0.5)])
        spo_without = SparsePauliOp.from_list([("ZZ", 0.5)])
        qc_with = _build_numeric(spo_with_identity, 0.3, [0, 1], n_qubits=2)
        qc_without = _build_numeric(spo_without, 0.3, [0, 1], n_qubits=2)
        assert Operator(qc_with).equiv(Operator(qc_without), atol=1e-10)

    def test_zero_time_is_identity(self):
        spo = SparsePauliOp.from_list([("ZZ", 0.5), ("XI", 1.0), ("IY", 0.3)])
        qc = _build_numeric(spo, 0.0, [0, 1], n_qubits=2)
        assert Operator(qc).equiv(Operator(QuantumCircuit(2)), atol=1e-10)

    @pytest.mark.parametrize("pauli", ["X", "Y", "Z"])
    def test_single_qubit_branches(self, pauli):
        spo = SparsePauliOp.from_list([(pauli, 1.0)])
        qc = _build_numeric(spo, 0.5, [0], n_qubits=1)
        expected = QuantumCircuit(1)
        method = {"X": expected.rx, "Y": expected.ry, "Z": expected.rz}[pauli]
        method(2 * 0.5 * 1.0, 0)
        assert Operator(qc).equiv(Operator(expected), atol=1e-10)

    def test_non_contiguous_qubits(self):
        # Place a ZZ rotation on qubits [3, 7] of a 10-qubit circuit.
        spo = SparsePauliOp.from_list([("ZZ", 1.0)])
        qc = _build_numeric(spo, 0.3, [3, 7], n_qubits=10)
        # Every two-qubit gate in qc must straddle qubits 3 and 7 only.
        for instr in qc.data:
            if instr.operation.num_qubits == 2:
                qubits = sorted(qc.find_bit(q).index for q in instr.qubits)
                assert qubits == [3, 7]

    def test_two_qubit_pauli_emits_basis_gates(self):
        # ``pauli_evolution`` emits ``rxx``/``ryy``/``rzz`` for 2-qubit Paulis;
        # the helper must decompose those into our QASM2 basis.
        spo = SparsePauliOp.from_list([("XX", 1.0), ("YY", 1.0), ("ZZ", 1.0)])
        qc = _build_numeric(spo, 0.3, [0, 1], n_qubits=2)
        gate_names = {instr.operation.name for instr in qc.data}
        assert "rxx" not in gate_names
        assert "ryy" not in gate_names
        assert "rzz" not in gate_names


class TestSpoToQiskitBasisGatesParity:
    """The numeric and symbolic paths must produce equivalent unitaries
    on the same SPO + numeric time.

    Cases are capped at 4 qubits — ``Operator(qc)`` materialises a
    ``2^n × 2^n`` complex128 dense matrix, so wider cases pay
    quadratic memory. If a wider case is genuinely needed, switch the
    comparison to ``Statevector`` (2^n) on a fixed initial state.
    """

    _MAX_PARITY_QUBITS = 4

    @pytest.mark.parametrize(
        "spo, qubits",
        [
            (SparsePauliOp.from_list([("Z", 1.0)]), [0]),
            (SparsePauliOp.from_list([("X", 1.0)]), [0]),
            (SparsePauliOp.from_list([("Y", 1.0)]), [0]),
            (SparsePauliOp.from_list([("ZZ", 0.5)]), [0, 1]),
            (SparsePauliOp.from_list([("XX", 0.7), ("YY", 0.7), ("ZZ", 0.3)]), [0, 1]),
            # X₀ and Z₀Z₁ anticommute on qubit 0 — a future change to
            # ``pauli_evolution``'s internal term ordering would break this
            # parity but pass the commuting cases above.
            (SparsePauliOp.from_list([("IX", 0.3), ("ZZ", 0.5)]), [0, 1]),
            (SparsePauliOp.from_list([("ZIZ", 0.5), ("XII", 0.4)]), [0, 1, 2]),
            (SparsePauliOp.from_list([("YYY", 1.0)]), [0, 1, 2]),
            (
                SparsePauliOp.from_list([("ZIIZ", 0.5), ("IXIY", 0.3), ("YZIZ", -0.2)]),
                [0, 1, 2, 3],
            ),
        ],
        ids=[
            "1q_Z",
            "1q_X",
            "1q_Y",
            "2q_ZZ",
            "2q_mixed_XYZ_commuting",
            "2q_non_commuting_X_and_ZZ",
            "3q_with_identity_gap",
            "3q_all_Y",
            "4q_mixed_full",
        ],
    )
    def test_numeric_matches_symbolic(self, spo, qubits):
        """Numeric path (Rust accelerator) and symbolic path (CX-RZ-CX) must
        produce unitarily equivalent circuits on the same SPO + numeric time."""
        assert len(qubits) <= self._MAX_PARITY_QUBITS, (
            f"Parity test cases must stay ≤{self._MAX_PARITY_QUBITS} qubits "
            f"so ``Operator.equiv`` doesn't materialise an oversized dense "
            f"matrix. Got {len(qubits)} qubits."
        )
        n = len(qubits)
        time = 0.37
        qc_numeric = _build_numeric(spo, time, qubits, n_qubits=n)
        qc_symbolic = _build_symbolic(spo, time, qubits, n_qubits=n)
        assert Operator(qc_numeric).equiv(Operator(qc_symbolic), atol=1e-10)

    def test_parity_with_non_contiguous_qubits(self):
        spo = SparsePauliOp.from_list([("ZZ", 0.5), ("XI", 0.3)])
        qc_numeric = _build_numeric(spo, 0.4, [1, 3], n_qubits=5)
        qc_symbolic = _build_symbolic(spo, 0.4, [1, 3], n_qubits=5)
        # 5-qubit dense matrix is 32×32 — still well within budget.
        assert Operator(qc_numeric).equiv(Operator(qc_symbolic), atol=1e-10)


class TestSpoToQiskitBasisGatesSymbolicEdgeCases:
    """Edge cases specific to the symbolic-angle path."""

    def test_empty_spo_is_noop(self):
        qc = _build_symbolic(generate_empty_spo(2), Parameter("t"), [0, 1], n_qubits=2)
        assert len(qc.data) == 0

    def test_identity_only_row_is_skipped(self):
        spo_with_identity = SparsePauliOp.from_list([("II", 1.0), ("ZZ", 0.5)])
        spo_without = SparsePauliOp.from_list([("ZZ", 0.5)])
        t = Parameter("t")
        qc_with = _build_symbolic(spo_with_identity, t, [0, 1], n_qubits=2)
        qc_without = _build_symbolic(spo_without, t, [0, 1], n_qubits=2)
        # Bind ``t`` to a concrete value, then compare.
        bound_with = qc_with.assign_parameters({t: 0.3})
        bound_without = qc_without.assign_parameters({t: 0.3})
        assert Operator(bound_with).equiv(Operator(bound_without), atol=1e-10)

    def test_parameter_expression_threads_through_gates(self):
        spo = SparsePauliOp.from_list([("Z", 1.0)])
        t = Parameter("t")
        qc = _build_symbolic(spo, 2 * t + 1, [0], n_qubits=1)
        # The single emitted RZ should carry a ParameterExpression involving ``t``.
        rz_instrs = [instr for instr in qc.data if instr.operation.name == "rz"]
        assert len(rz_instrs) == 1
        params = rz_instrs[0].operation.params
        assert any(hasattr(p, "parameters") and t in p.parameters for p in params)
