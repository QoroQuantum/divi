# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for term-manipulation primitives in divi.hamiltonians._term_ops."""

import numpy as np
import pennylane as qp
import pytest
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians._term_ops import (
    _clean_hamiltonian_spo,
    _clean_hamiltonian_via_spo,
    _empty_spo,
    _from_spo,
    _sort_hamiltonian_terms_spo,
    _spo_to_basis_gate_ops,
    _spo_wires,
    _to_spo,
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


# ---------------------------------------------------------------------------
# _empty_spo
# ---------------------------------------------------------------------------


class TestEmptySpo:
    @pytest.mark.parametrize("num_qubits", [0, 1, 4, 17])
    def test_size_is_zero(self, num_qubits):
        """Returned SPO has no rows regardless of register size."""
        spo = _empty_spo(num_qubits)
        assert spo.size == 0
        assert spo.num_qubits == num_qubits


# ---------------------------------------------------------------------------
# _to_spo / _spo_wires / _from_spo round-trip
# ---------------------------------------------------------------------------


class TestSpoConversion:
    def test_to_spo_passthrough(self, simple_spo):
        """An SPO input is returned as-is."""
        assert _to_spo(simple_spo) is simple_spo

    def test_to_spo_from_pl(self, simple_pl_hamiltonian, simple_spo):
        """A PL Hamiltonian converts to the equivalent SPO (modulo simplify)."""
        spo = _to_spo(simple_pl_hamiltonian)
        assert spo.simplify() == simple_spo.simplify()

    def test_spo_wires_for_spo(self, simple_spo):
        """Wires for an SPO input are ``range(num_qubits)``."""
        assert _spo_wires(simple_spo) == (0, 1)

    def test_spo_wires_for_pl_uses_op_wires(self):
        """A bare PL operator without cached wires falls back to ``op.wires``."""
        op = qp.PauliZ(2) @ qp.PauliX(5)
        assert _spo_wires(op) == tuple(op.wires)

    def test_from_spo_records_canonical_mapping(self, simple_spo):
        """``_from_spo`` records the SPO and canonical wires for round-trip recovery."""
        pl = _from_spo(simple_spo, range(2))
        assert _to_spo(pl) is simple_spo
        assert _spo_wires(pl) == (0, 1)

    def test_from_spo_to_spo_roundtrip_short_circuits(self, simple_spo):
        """``_to_spo`` on a ``_from_spo`` result returns the original cached SPO."""
        pl = _from_spo(simple_spo, range(2))
        assert _to_spo(pl) is simple_spo

    def test_from_spo_preserves_canonical_wires_after_simplify(self):
        """Even when simplify reorders/drops wires, ``_spo_wires`` returns canonical."""
        # SPO with qubit 0 carrying only identity; simplify drops it from .wires.
        spo = SparsePauliOp.from_list([("ZI", 1.0), ("XI", 2.0)])
        pl = _from_spo(spo, range(2))
        assert _spo_wires(pl) == (0, 1)
        # The simplified PL op may have a smaller .wires set:
        assert set(pl.wires).issubset({0, 1})

    def test_from_spo_empty_returns_pl_empty_hamiltonian(self):
        """Empty SPO maps to ``qp.Hamiltonian([], [])``."""
        pl = _from_spo(_empty_spo(3), range(3))
        assert isinstance(pl, qp.Hamiltonian)
        assert len(pl) == 0

    def test_from_spo_simplify_false_skips_simplify(self):
        """``simplify=False`` returns a non-simplified Sum/SProd structure."""
        spo = SparsePauliOp.from_list([("Z", 1.0)])
        # With simplify=True the trivial SProd(1, Z) collapses to bare Z.
        simplified = _from_spo(spo, range(1))
        assert isinstance(simplified, qp.PauliZ)
        # With simplify=False the SProd wrapper survives.
        unsimplified = _from_spo(spo, range(1), simplify=False)
        assert not isinstance(unsimplified, qp.PauliZ)


# ---------------------------------------------------------------------------
# _clean_hamiltonian_spo
# ---------------------------------------------------------------------------


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
        spo, constant = _clean_hamiltonian_spo(_empty_spo(3))
        assert spo.size == 0
        assert constant == 0.0


# ---------------------------------------------------------------------------
# _clean_hamiltonian_via_spo (PL boundary)
# ---------------------------------------------------------------------------


class TestCleanHamiltonianViaSpo:
    @pytest.mark.parametrize(
        "ham, expected_pl, expected_constant",
        [
            (
                qp.sum(qp.s_prod(2, qp.PauliX(0)), qp.PauliZ(1)),
                qp.sum(qp.s_prod(2, qp.PauliX(0)), qp.PauliZ(1)),
                0.0,
            ),
            (
                qp.sum(qp.s_prod(2.5, qp.Identity(0)), qp.s_prod(1.5, qp.Identity(1))),
                qp.Hamiltonian([], []),
                4.0,
            ),
            (
                qp.sum(
                    qp.s_prod(2, qp.PauliX(0)),
                    qp.s_prod(3, qp.Identity(0)),
                    qp.PauliZ(1),
                ),
                qp.sum(qp.s_prod(2, qp.PauliX(0)), qp.PauliZ(1)),
                3.0,
            ),
            (qp.Identity(0), qp.Hamiltonian([], []), 1.0),
            (qp.s_prod(5.0, qp.Identity(0)), qp.Hamiltonian([], []), 5.0),
            (qp.PauliZ(0), qp.PauliZ(0), 0.0),
            (qp.Hamiltonian([], []), qp.Hamiltonian([], []), 0.0),
        ],
    )
    def test_partition_matches_expected(self, ham, expected_pl, expected_constant):
        cleaned, constant = _clean_hamiltonian_via_spo(ham)
        assert constant == pytest.approx(expected_constant)
        # Compare via SPO equality so simplify/order differences don't bite.
        if isinstance(expected_pl, qp.Hamiltonian) and len(expected_pl) == 0:
            assert isinstance(cleaned, qp.Hamiltonian) and len(cleaned) == 0
        else:
            assert _to_spo(cleaned).simplify() == _to_spo(expected_pl).simplify()

    def test_empty_input_carries_zero_constant(self):
        cleaned, constant = _clean_hamiltonian_via_spo(qp.Hamiltonian([], []))
        assert constant == 0.0
        assert isinstance(cleaned, qp.Hamiltonian)
        assert len(cleaned) == 0


# ---------------------------------------------------------------------------
# _sort_hamiltonian_terms_spo
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# _spo_to_basis_gate_ops
# ---------------------------------------------------------------------------


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
