# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from qiskit.quantum_info import Pauli, SparsePauliOp

from divi.hamiltonians.trotterization import _pauli_bits as bits
from divi.qprog.initial_states import CustomPerQubitState, DickeState, OnesState
from tests._helpers import initial_state_amplitudes


def _random_paulis(n_qubits: int, count: int, seed: int) -> SparsePauliOp:
    rng = np.random.default_rng(seed)
    labels = ["".join(rng.choice(list("IXYZ"), n_qubits)) for _ in range(count)]
    return SparsePauliOp(labels, rng.normal(size=count))


def _anticommuting(left: SparsePauliOp, right: SparsePauliOp) -> set[tuple[int, int]]:
    return {
        (a, b)
        for a, p in enumerate(left.paulis)
        for b, q in enumerate(right.paulis)
        if not Pauli(p).commutes(Pauli(q))
    }


def _dense_expectation(initial_state, n_qubits):
    psi = initial_state_amplitudes(initial_state, n_qubits)
    return lambda operator: psi.conj() @ operator.to_matrix() @ psi


_LEFT = _random_paulis(4, 12, seed=1)
_RIGHT = _random_paulis(4, 15, seed=2)


@pytest.mark.parametrize("n_qubits", [5, 70], ids=["one_word", "two_words"])
def test_commutators_are_those_of_every_anticommuting_pair(n_qubits):
    left = _random_paulis(n_qubits, 15, seed=3)
    right = _random_paulis(n_qubits, 15, seed=4)
    found = {}
    for a, b, x, z, factor in bits.commutators(bits.pack(left), bits.pack(right)):
        strings = bits.to_paulis(x, z, n_qubits)
        for k, pair in enumerate(zip(a.tolist(), b.tolist())):
            found[pair] = SparsePauliOp(strings[k], [factor[k]])

    assert found.keys() == _anticommuting(left, right)
    for (a, b), commutator in found.items():
        p, q = SparsePauliOp(left.paulis[a]), SparsePauliOp(right.paulis[b])
        np.testing.assert_allclose(
            (p @ q - q @ p - commutator).simplify().coeffs, 0, atol=1e-12
        )


def test_combine_merges_repeated_strings_and_drops_cancelled_ones():
    packed = bits.pack(SparsePauliOp(["XY", "ZI", "XY", "ZI"], [1.0, 0.5, 2.0, -0.5]))
    combined = bits.combine(packed.x, packed.z, packed.coeffs)
    assert bits.to_paulis(combined.x, combined.z, 2).to_labels() == ["XY"]
    np.testing.assert_allclose(combined.coeffs, [3.0])


def test_pack_folds_each_phase_into_its_coefficient():
    packed = bits.pack(SparsePauliOp(Pauli("-iZZ"), [2.0]))
    np.testing.assert_allclose(packed.coeffs, [-2j])
    assert bits.to_paulis(packed.x, packed.z, 2).to_labels() == ["ZZ"]


@pytest.mark.parametrize(
    "initial_state",
    [CustomPerQubitState("0+-1"), OnesState(), DickeState(2)],
    ids=["per_qubit", "ones", "dicke"],
)
def test_expectations_match_the_state(initial_state):
    strings = _random_paulis(4, 30, seed=5)
    packed = bits.pack(strings)
    expectation = _dense_expectation(initial_state, 4)
    np.testing.assert_allclose(
        bits.evaluate(
            initial_state, 4, lambda source: source.values(packed.x, packed.z)
        ),
        [expectation(SparsePauliOp(p)) for p in strings.paulis],
        atol=1e-10,
    )


def test_row_index_tells_apart_rows_with_equal_hashes():
    # Word 1 is rotated by 17 bits before the XOR, so both rows hash to 1 << 17.
    rows = np.array([[0, 1], [1 << 17, 0], [5, 7]], dtype=np.uint64)
    index = bits._RowIndex.of(rows)
    np.testing.assert_array_equal(index.locate(rows[[2, 1, 0, 1]]), [2, 1, 0, 1])
    with pytest.raises(KeyError, match="missing from the index"):
        index.locate(np.array([[1, 1]], dtype=np.uint64))


@pytest.mark.parametrize(
    "initial_state",
    [CustomPerQubitState("0+-1"), CustomPerQubitState("+-+-"), DickeState(2)],
    ids=["mixed_axes", "x_axes", "dicke"],
)
def test_commutator_values_cover_every_non_zero_pair(initial_state):
    expectation = _dense_expectation(initial_state, 4)
    expected = {}
    for a, b in _anticommuting(_LEFT, _RIGHT):
        p, q = SparsePauliOp(_LEFT.paulis[a]), SparsePauliOp(_RIGHT.paulis[b])
        value = expectation(p @ q - q @ p)
        if abs(value) > 1e-12:
            expected[(a, b)] = value
    found = {}
    blocks = bits.evaluate(
        initial_state,
        4,
        lambda source: list(
            bits.commutator_values(bits.pack(_LEFT), bits.pack(_RIGHT), source)
        ),
    )
    for a, b, values in blocks:
        for pair, value in zip(zip(a.tolist(), b.tolist()), values):
            if abs(value) > 1e-12:
                found[pair] = value

    assert found.keys() == expected.keys()
    np.testing.assert_allclose(
        [found[pair] for pair in expected], list(expected.values()), atol=1e-10
    )


@pytest.mark.parametrize(
    "initial_state",
    [CustomPerQubitState("0+-1"), DickeState(2)],
    ids=["product", "dicke"],
)
def test_double_commutator_sums_match_dense_matrices(initial_state):
    psi = initial_state_amplitudes(initial_state, 4)
    left = _LEFT.to_matrix()

    def nested(inner, right):
        middle = inner @ right - right @ inner
        return psi.conj() @ (left @ middle - middle @ left) @ psi

    per_inner, per_right = bits.evaluate(
        initial_state,
        4,
        lambda source: bits.double_commutator_sums(
            bits.pack(_LEFT), bits.pack(_RIGHT), source
        ),
    )
    np.testing.assert_allclose(
        per_inner,
        [nested(p.to_matrix(), _RIGHT.to_matrix()) for p in _LEFT.paulis],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        per_right,
        [nested(left, q.to_matrix()) for q in _RIGHT.paulis],
        atol=1e-9,
    )


def test_shift_sums_count_minus_two_per_anticommuting_weight():
    weights = np.random.default_rng(6).normal(size=_RIGHT.size)
    weights[::3] = 0.0
    expected = np.zeros(_LEFT.size)
    for a, b in _anticommuting(_LEFT, _RIGHT):
        expected[a] -= 2 * weights[b]
    np.testing.assert_allclose(
        bits.shift_sums(bits.pack(_LEFT), bits.pack(_RIGHT), weights).real,
        expected,
        atol=1e-12,
    )
