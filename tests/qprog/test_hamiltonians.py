# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import warnings

import numpy as np
import pennylane as qml
import pytest
import scipy.sparse as sps

from divi.qprog._hamiltonians import (
    ExactTrotterization,
    QDrift,
    _clean_hamiltonian,
    _is_sanitized,
    convert_hamiltonian_to_pauli_string,
    convert_qubo_matrix_to_pennylane_ising,
)


@pytest.mark.parametrize(
    "matrix, expected",
    [
        (np.array([[1, 2], [2, 3]]), True),  # Symmetric dense
        (np.array([[1, 2], [0, 3]]), True),  # Upper triangular dense
        (np.array([[1, 2], [1, 3]]), False),  # Not sanitized dense
        (sps.csr_matrix([[1, 2], [2, 3]]), True),  # Symmetric sparse
        (sps.csr_matrix([[1, 2], [0, 3]]), True),  # Upper triangular sparse
        (sps.csr_matrix([[1, 2], [1, 3]]), False),  # Not sanitized sparse
    ],
)
def test_is_sanitized(matrix, expected):
    """
    Tests the _is_sanitized function with various matrix types.
    """
    assert _is_sanitized(matrix) == expected


class TestQuboToIsingConversion:
    """
    A test class for the convert_qubo_matrix_to_pennylane_ising function.
    """

    def test_symmetric_dense_qubo(self):
        """
        Tests conversion with a simple symmetric dense QUBO matrix.
        """
        qubo_matrix = np.array([[-1.0, 2.0], [2.0, -1.0]])
        hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(qubo_matrix)

        expected_hamiltonian = 0.5 * (qml.Z(0) @ qml.Z(1))

        assert np.isclose(constant, -0.5)
        assert qml.equal(hamiltonian, expected_hamiltonian)

    def test_upper_triangular_dense_qubo(self):
        """
        Tests conversion with an upper triangular dense QUBO matrix.
        The result should be the same as the symmetric equivalent.
        """
        qubo_matrix = np.array([[-1.0, 4.0], [0.0, -1.0]])
        hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(qubo_matrix)

        # Symmetrized version is [[-1, 2], [2, -1]]
        expected_hamiltonian = 0.5 * (qml.Z(0) @ qml.Z(1))

        assert np.isclose(constant, -0.5)
        assert qml.equal(hamiltonian, expected_hamiltonian)

    def test_non_sanitized_qubo_raises_warning(self):
        """
        Tests that a non-sanitized QUBO matrix raises a warning and is correctly
        symmetrized.
        """
        qubo_matrix = np.array([[-1.0, 3.0], [1.0, -1.0]])

        with pytest.warns(UserWarning, match="neither symmetric nor upper triangular"):
            hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(qubo_matrix)

        # Symmetrized version is [[-1, 2], [2, -1]]
        expected_hamiltonian = 0.5 * (qml.Z(0) @ qml.Z(1))

        assert np.isclose(constant, -0.5)
        assert qml.equal(hamiltonian, expected_hamiltonian)

    def test_diagonal_qubo(self):
        """
        Tests a purely diagonal QUBO matrix, which should result in only Z terms.
        """
        qubo_matrix = np.array([[1.0, 0.0], [0.0, -2.0]])
        hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(qubo_matrix)

        expected_hamiltonian = -0.5 * qml.Z(0) + 1.0 * qml.Z(1)
        expected_constant = -0.5

        assert np.isclose(constant, expected_constant)
        assert qml.equal(hamiltonian, expected_hamiltonian)

    def test_sparse_qubo(self):
        """
        Tests conversion with a sparse QUBO matrix.
        """
        qubo_matrix = sps.csr_matrix([[-1.0, 2.0], [2.0, -1.0]])
        hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(qubo_matrix)

        expected_hamiltonian = 0.5 * (qml.Z(0) @ qml.Z(1))

        assert np.isclose(constant, -0.5)
        assert qml.equal(hamiltonian, expected_hamiltonian)

    def test_3x3_qubo(self):
        """
        Tests a larger 3x3 QUBO matrix to ensure correct term summation.
        """
        qubo_matrix = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]], dtype=float)

        hamiltonian, constant = convert_qubo_matrix_to_pennylane_ising(qubo_matrix)

        expected_constant = 8.0
        expected_hamiltonian = (
            0.5 * (qml.Z(0) @ qml.Z(1))
            + 0.75 * (qml.Z(0) @ qml.Z(2))
            + 1.25 * (qml.Z(1) @ qml.Z(2))
            - 1.75 * qml.Z(0)
            - 3.75 * qml.Z(1)
            - 5.0 * qml.Z(2)
        )

        assert np.isclose(constant, expected_constant)

        hamiltonian = hamiltonian.simplify()
        expected_hamiltonian = expected_hamiltonian.simplify()

        h_coeffs, h_ops = hamiltonian.terms()
        e_coeffs, e_ops = expected_hamiltonian.terms()

        # Create dictionaries for easy lookup and comparison for robust comparison
        h_map = {op.hash: coeff for coeff, op in zip(h_coeffs, h_ops)}
        e_map = {op.hash: coeff for coeff, op in zip(e_coeffs, e_ops)}

        assert h_map.keys() == e_map.keys()
        for key in h_map:
            assert np.isclose(h_map[key], e_map[key])


class TestCleanHamiltonian:
    """
    A test class for the clean_hamiltonian function.
    """

    @pytest.mark.parametrize(
        "hamiltonian, expected_op, expected_constant",
        [
            # Test case 1: Hamiltonian with no constant part
            (
                qml.sum(qml.s_prod(2, qml.PauliX(0)), qml.PauliZ(1)),
                qml.sum(qml.s_prod(2, qml.PauliX(0)), qml.PauliZ(1)),
                0.0,
            ),
            # Test case 2: Hamiltonian with only constant parts
            (
                qml.sum(
                    qml.s_prod(2.5, qml.Identity(0)), qml.s_prod(1.5, qml.Identity(1))
                ),
                qml.Hamiltonian([], []),
                4.0,
            ),
            # Test case 3: Mixed Hamiltonian
            (
                qml.sum(
                    qml.s_prod(2, qml.PauliX(0)),
                    qml.s_prod(3, qml.Identity(0)),
                    qml.PauliZ(1),
                ),
                qml.sum(qml.s_prod(2, qml.PauliX(0)), qml.PauliZ(1)),
                3.0,
            ),
            # Test case 4: Single Identity operator
            (qml.Identity(0), qml.Hamiltonian([], []), 1.0),
            # Test case 5: Single scaled Identity operator
            (qml.s_prod(5.0, qml.Identity(0)), qml.Hamiltonian([], []), 5.0),
            # Test case 6: Product of Identities
            (qml.prod(qml.Identity(0), qml.Identity(1)), qml.Hamiltonian([], []), 1.0),
            # Test case 7: Single non-constant operator
            (qml.PauliZ(0), qml.PauliZ(0), 0.0),
            # Test case 8: Empty Hamiltonian
            (qml.Hamiltonian([], []), qml.Hamiltonian([], []), 0.0),
        ],
    )
    def test_clean_hamiltonian_various_cases(
        self, hamiltonian, expected_op, expected_constant
    ):
        """
        Tests the clean_hamiltonian function with various scenarios.
        """
        new_hamiltonian, constant = _clean_hamiltonian(hamiltonian)

        assert np.isclose(constant, expected_constant)

        try:
            # qml.equal can be sensitive, so we compare terms for robustness
            new_coeffs, new_ops = new_hamiltonian.terms()
            exp_coeffs, exp_ops = expected_op.terms()

            assert len(new_coeffs) == len(exp_coeffs)

            if len(new_coeffs) > 0:
                # Create dictionaries for easy lookup and comparison
                new_map = {op.hash: coeff for coeff, op in zip(new_coeffs, new_ops)}
                exp_map = {op.hash: coeff for coeff, op in zip(exp_coeffs, exp_ops)}

                assert new_map.keys() == exp_map.keys()
                for key in new_map:
                    assert np.isclose(new_map[key], exp_map[key])
        except qml.operation.TermsUndefinedError:
            # Fallback for operators that don't have .terms()
            assert qml.equal(new_hamiltonian, expected_op)


class TestHamiltonianToPauliString:
    """
    A test class for the hamiltonian_to_pauli_string function.
    """

    @pytest.mark.parametrize(
        "hamiltonian, n_qubits, expected",
        [
            # Test single Pauli operators
            (qml.Hamiltonian([1.0], [qml.PauliX(0)]), 1, "X"),
            (qml.Hamiltonian([1.0], [qml.PauliY(0)]), 1, "Y"),
            (qml.Hamiltonian([1.0], [qml.PauliZ(0)]), 1, "Z"),
            # Test multiple qubits
            (
                qml.Hamiltonian([1.0], [qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliZ(2)]),
                3,
                "XYZ",
            ),
            # Test multiple terms separated by semicolon
            (
                qml.Hamiltonian([1.0, 1.0], [qml.PauliX(0), qml.PauliY(1)]),
                2,
                "XI;IY",
            ),
            # Test scaled operator (SProd)
            (
                qml.Hamiltonian([1.0], [qml.s_prod(2.5, qml.PauliX(0))]),
                1,
                "X",
            ),
            # Test product operator
            (
                qml.Hamiltonian([1.0], [qml.prod(qml.PauliX(0), qml.PauliZ(1))]),
                2,
                "XZ",
            ),
            # Test identity fills remaining qubits
            (
                qml.Hamiltonian([1.0], [qml.PauliX(2)]),
                3,
                "IIX",
            ),
            # Test complex Hamiltonian with multiple terms
            (
                qml.Hamiltonian(
                    [1.0, 1.0, 1.0],
                    [
                        qml.PauliX(0) @ qml.PauliY(1),
                        qml.PauliZ(1) @ qml.PauliZ(2),
                        qml.PauliX(2),
                    ],
                ),
                3,
                "XYI;IZZ;IIX",
            ),
            # Test a single operator not wrapped in a Hamiltonian
            (qml.PauliZ(0), 1, "Z"),
            # Test with qml.sum
            (qml.sum(qml.PauliX(0), qml.PauliZ(1)), 2, "XI;IZ"),
            # Test with nested s_prod (coefficients are ignored)
            (qml.s_prod(2.0, qml.s_prod(1.5, qml.PauliX(0))), 1, "X"),
            # Test empty Hamiltonian
            (qml.Hamiltonian([], []), 3, ""),
            # Test that qml.Identity is simplified away in a product
            (qml.PauliX(0) @ qml.Identity(1), 2, "XI"),
            # Test sum of prods
            (
                qml.sum(
                    qml.prod(qml.PauliX(0), qml.PauliY(1)),
                    qml.prod(qml.PauliZ(1), qml.PauliZ(2)),
                ),
                3,
                "XYI;IZZ",
            ),
            # Test that Identity is handled correctly
            (qml.Identity(0), 2, "II"),
            (qml.Hamiltonian([1.0], [qml.Identity(0)]), 1, "I"),
        ],
    )
    def test_hamiltonian_conversions(self, hamiltonian, n_qubits, expected):
        """Test various Hamiltonian conversion scenarios."""
        result = convert_hamiltonian_to_pauli_string(hamiltonian, n_qubits=n_qubits)
        assert result == expected

    def test_unknown_pauli_operator_raises_error(self):
        """Test that unknown Pauli operators raise ValueError."""

        # Create a mock operator with an unknown name
        class UnknownPauli(qml.operation.Operator):
            def __init__(self, wires):
                super().__init__(wires=wires)
                self.name = "UnknownOperator"

        unknown_op = UnknownPauli(wires=[0])
        hamiltonian = qml.Hamiltonian([1.0], [unknown_op])

        with pytest.raises(ValueError, match="Unknown Pauli operator"):
            convert_hamiltonian_to_pauli_string(hamiltonian, n_qubits=1)

    def test_pauli_with_no_wires_raises_error(self):
        """Test that Pauli operators with no wires raise ValueError."""

        # Create a mock operator with no wires
        class NoWirePauli(qml.operation.Operator):
            def __init__(self):
                super().__init__(wires=[])
                self.name = "PauliX"

        no_wire_op = NoWirePauli()
        hamiltonian = qml.Hamiltonian([1.0], [no_wire_op])

        with pytest.raises(ValueError, match="has no wires"):
            convert_hamiltonian_to_pauli_string(hamiltonian, n_qubits=1)

    @pytest.mark.parametrize(
        "wire, n_qubits",
        [
            (-1, 1),  # Negative wire index
            (2, 2),  # Wire index >= n_qubits
            (5, 3),  # Wire index much larger than n_qubits
        ],
    )
    def test_wire_index_out_of_range_raises_error(self, wire, n_qubits):
        """Test that wire indices out of range raise ValueError."""
        hamiltonian = qml.Hamiltonian([1.0], [qml.PauliX(wire)])
        with pytest.raises(ValueError, match="Wire index.*out of range"):
            convert_hamiltonian_to_pauli_string(hamiltonian, n_qubits=n_qubits)


@pytest.fixture
def simple_hamiltonian():
    """Hamiltonian with three terms (coefficients 1.0, 2.0, 3.0) for Trotterization tests."""
    return (1.0 * qml.Z(0) + 2.0 * qml.Z(1) + 3.0 * (qml.Z(0) @ qml.Z(1))).simplify()


class TestExactTrotterization:
    """
    Tests for ExactTrotterization strategy (public API and specified behavior).
    """

    def test_both_keep_fraction_and_keep_top_n_raises(self):
        """At most one of keep_fraction or keep_top_n may be provided."""
        with pytest.raises(
            ValueError, match="At most one of keep_fraction or keep_top_n"
        ):
            ExactTrotterization(keep_fraction=0.5, keep_top_n=2)

    @pytest.mark.parametrize("keep_fraction", [-0.1, 0, 1.5])
    def test_keep_fraction_out_of_range_raises(self, keep_fraction):
        """keep_fraction must be in (0, 1]."""
        with pytest.raises(ValueError, match="keep_fraction must be in \\(0, 1\\]"):
            ExactTrotterization(keep_fraction=keep_fraction)

    @pytest.mark.parametrize("keep_top_n", [-1, 0, 0.5])
    def test_keep_top_n_invalid_raises(self, keep_top_n):
        """keep_top_n must be a positive integer (>= 1)."""
        with pytest.raises(ValueError, match="keep_top_n must be a positive integer"):
            ExactTrotterization(keep_top_n=keep_top_n)

    def test_no_truncation_returns_simplified_hamiltonian(self, simple_hamiltonian):
        """When both keep_fraction and keep_top_n are None, returns Hamiltonian unchanged."""
        result = ExactTrotterization().process_hamiltonian(simple_hamiltonian)
        assert qml.equal(result, simple_hamiltonian.simplify())

    def test_keep_fraction_one_returns_full_and_warns(self, simple_hamiltonian):
        """When keep_fraction is 1.0, returns full Hamiltonian and warns."""
        with pytest.warns(UserWarning, match="keep_fraction is 1.0.*no truncation"):
            result = ExactTrotterization(keep_fraction=1.0).process_hamiltonian(
                simple_hamiltonian
            )
        assert qml.equal(result, simple_hamiltonian.simplify())

    def test_keep_top_n_at_least_terms_returns_full_and_warns(self, simple_hamiltonian):
        """When keep_top_n >= number of terms, returns full Hamiltonian and warns."""
        with pytest.warns(UserWarning, match="keep_top_n is greater than or equal"):
            result = ExactTrotterization(keep_top_n=10).process_hamiltonian(
                simple_hamiltonian
            )
        assert qml.equal(result, simple_hamiltonian.simplify())

    def test_keep_top_n_reduces_term_count(self, simple_hamiltonian):
        """keep_top_n keeps only that many largest-magnitude terms (plus constant)."""
        result = ExactTrotterization(keep_top_n=1).process_hamiltonian(
            simple_hamiltonian
        )
        coeffs, _ = result.terms()
        assert len(coeffs) == 1
        # simple_hamiltonian has 1.0*Z(0), 2.0*Z(1), 3.0*Z(0)@Z(1); largest magnitude is 3.0
        expected = (3.0 * (qml.Z(0) @ qml.Z(1))).simplify()
        assert qml.equal(result, expected)

    def test_keep_top_n_two_keeps_two_terms(self, simple_hamiltonian):
        """keep_top_n=2 keeps the two terms with largest coefficient magnitude."""
        result = ExactTrotterization(keep_top_n=2).process_hamiltonian(
            simple_hamiltonian
        )
        coeffs, _ = result.terms()
        assert len(coeffs) == 2
        # Largest two by magnitude: 2.0*Z(1) and 3.0*Z(0)@Z(1)
        expected = (2.0 * qml.Z(1) + 3.0 * (qml.Z(0) @ qml.Z(1))).simplify()
        assert qml.equal(result, expected)

    def test_keep_fraction_reduces_term_count(self, simple_hamiltonian):
        """keep_fraction < 1 yields fewer terms; kept terms have total |coeff| >= fraction of full."""
        result = ExactTrotterization(keep_fraction=0.5).process_hamiltonian(
            simple_hamiltonian
        )
        full_coeffs, _ = simple_hamiltonian.terms()
        result_coeffs, _ = result.terms()
        assert len(result_coeffs) <= len(full_coeffs)
        # For keep_fraction=0.5, total |coeff| in simple_hamiltonian is 6; we keep terms summing to >= 3
        full_sum_abs = np.sum(np.abs(full_coeffs))
        result_sum_abs = np.sum(np.abs(result_coeffs))
        assert result_sum_abs >= 0.5 * full_sum_abs
        # With 0.5 we keep exactly the largest term (3.0*Z(0)@Z(1))
        expected = (3.0 * (qml.Z(0) @ qml.Z(1))).simplify()
        assert qml.equal(result, expected)

    def test_result_is_simplified_and_same_type(self, simple_hamiltonian):
        """process_hamiltonian returns a simplified operator of the same structural type."""
        result = ExactTrotterization(keep_top_n=2).process_hamiltonian(
            simple_hamiltonian
        )
        assert hasattr(result, "simplify")
        result.simplify()  # idempotent when already simplified

    def test_exact_trotterization_stateful_is_false(self):
        """ExactTrotterization reports stateful=False (no caching)."""
        assert ExactTrotterization(keep_top_n=2).stateful is False


class TestQDrift:
    """
    Tests for QDrift strategy (public API and specified behavior).
    """

    def test_invalid_sampling_strategy_raises(self):
        """sampling_strategy must be 'uniform' or 'weighted'."""
        with pytest.raises(ValueError, match="Invalid sampling_strategy"):
            QDrift(sample_budget=5, sampling_strategy="invalid")

    def test_seed_non_int_raises(self):
        """seed must be an integer when provided."""
        with pytest.raises(ValueError, match="seed must be an integer"):
            QDrift(sample_budget=5, seed=1.5)

    def test_all_none_returns_unchanged_and_warns(self, simple_hamiltonian):
        """When keep_fraction, keep_top_n and sample_budget are all None, returns Hamiltonian unchanged."""
        with pytest.warns(UserWarning, match="Neither keep_fraction nor sample_budget"):
            result = QDrift().process_hamiltonian(simple_hamiltonian)
        assert qml.equal(result, simple_hamiltonian.simplify())

    def test_sample_budget_only_returns_valid_hamiltonian(self, simple_hamiltonian):
        """When only sample_budget is set (no keep_*), result is a valid operator with terms."""
        result = QDrift(sample_budget=4, seed=42).process_hamiltonian(
            simple_hamiltonian
        )
        coeffs, ops = result.terms()
        assert len(coeffs) >= 1
        assert len(coeffs) == len(ops)

    def test_seed_gives_reproducible_result(self, simple_hamiltonian):
        """Same seed yields identical Hamiltonian from process_hamiltonian."""
        strategy = QDrift(sample_budget=5, seed=123, sampling_strategy="uniform")
        r1 = strategy.process_hamiltonian(simple_hamiltonian)
        r2 = strategy.process_hamiltonian(simple_hamiltonian)
        assert qml.equal(r1, r2)

    def test_with_keep_fraction_and_sample_budget(self, simple_hamiltonian):
        """QDrift with keep_fraction and sample_budget returns keep terms + sampled terms."""
        result = QDrift(
            keep_fraction=0.5, sample_budget=3, seed=0, sampling_strategy="weighted"
        ).process_hamiltonian(simple_hamiltonian)
        coeffs, _ = result.terms()
        # Kept terms (from 0.5 fraction) + 3 sampled from the rest
        assert len(coeffs) >= 3

    def test_keep_fraction_one_and_sample_budget_warns_no_terms_to_sample(
        self, simple_hamiltonian
    ):
        """When keep_fraction=1.0 and sample_budget set, all terms kept so no sampling; warns."""
        with pytest.warns(UserWarning) as record:
            result = QDrift(
                keep_fraction=1.0, sample_budget=5, seed=0
            ).process_hamiltonian(simple_hamiltonian)
        # ExactTrotterization may warn "keep_fraction is 1.0..."; QDrift warns "no terms left to sample"
        messages = [str(w.message) for w in record]
        assert any("no terms left to sample" in m for m in messages)
        assert qml.equal(result, simple_hamiltonian)

    def test_sample_budget_none_equivalent_to_exact_trotterization(
        self, simple_hamiltonian
    ):
        """When sample_budget is None but keep_fraction set, result equals ExactTrotterization."""
        with pytest.warns(UserWarning, match="sample_budget is not set"):
            qdrift = QDrift(keep_fraction=0.5)
        exact = ExactTrotterization(keep_fraction=0.5)
        qdrift_result = qdrift.process_hamiltonian(simple_hamiltonian)
        exact_result = exact.process_hamiltonian(simple_hamiltonian)
        assert qml.equal(qdrift_result, exact_result)

    def test_qdrift_caches_keep_hamiltonian(self, mocker, simple_hamiltonian):
        """Repeated process_hamiltonian with same Hamiltonian uses cache; ExactTrotterization called once."""
        spy = mocker.spy(ExactTrotterization, "process_hamiltonian")
        strategy = QDrift(
            keep_fraction=0.5, sample_budget=3, seed=42, sampling_strategy="weighted"
        )
        strategy.process_hamiltonian(simple_hamiltonian)
        strategy.process_hamiltonian(simple_hamiltonian)
        assert spy.call_count == 1

    def test_keep_fraction_one_warns_only_on_first_call(self, simple_hamiltonian):
        """'No terms left to sample' warning is emitted only on first call, not on cached calls."""
        strategy = QDrift(keep_fraction=1.0, sample_budget=5, seed=0)
        with pytest.warns(UserWarning) as first_record:
            first_result = strategy.process_hamiltonian(simple_hamiltonian)
        first_messages = [str(w.message) for w in first_record]
        assert any("no terms left to sample" in m for m in first_messages)
        assert qml.equal(first_result, simple_hamiltonian)

        with warnings.catch_warnings(record=True) as second_record:
            warnings.simplefilter("always")
            second_result = strategy.process_hamiltonian(simple_hamiltonian)
        second_messages = [str(w.message) for w in second_record]
        assert not any("no terms left to sample" in m for m in second_messages)
        assert qml.equal(second_result, simple_hamiltonian)

    def test_qdrift_stateful_is_true(self):
        """QDrift reports stateful=True (caches intermediate results)."""
        assert QDrift(sample_budget=5, seed=0).stateful is True
