# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from qiskit.quantum_info import PauliList, SparsePauliOp

from divi.circuits._core import flatten_observable_tuple
from divi.pipeline._shadows import (
    _compute_cover_probabilities,
    _encode_paulis,
    _fit_biased_bases,
    _sample_shadow_covers,
    _sum_squared_coefficients,
    _uniform_bases,
)


def _codes(labels):
    return _encode_paulis(PauliList(labels))


def test_uniform_bases_cover_a_term_with_probability_three_to_minus_its_weight():
    codes = _codes(["ZII", "XYI", "III"])

    np.testing.assert_allclose(
        _compute_cover_probabilities(codes, _uniform_bases(3)), [1 / 3, 1 / 9, 1.0]
    )


@pytest.mark.parametrize(
    "labels",
    [["ZI", "XI", "IZ", "ZZ", "XZ"], ["ZZZZ"]],
    ids=["overlapping-terms", "rarely-covered-term"],
)
def test_covers_fit_one_basis_and_account_for_every_draw(labels):
    covers, counts, empty_draws = _sample_shadow_covers(
        _codes(labels), _uniform_bases(len(labels[0])), 500, np.random.default_rng(3)
    )

    assert counts.sum() + empty_draws == 500
    assert covers.any(axis=1).all()
    for pattern in covers:
        covered = [label for label, hit in zip(labels, pattern) if hit]
        for qubit in range(len(labels[0])):
            assert len({label[qubit] for label in covered} - {"I"}) <= 1


@pytest.mark.parametrize("biased", [False, True], ids=["uniform", "biased"])
def test_terms_are_covered_at_their_cover_probability(biased):
    codes = _codes(["ZI", "XY", "ZZ"])
    bases = (
        _fit_biased_bases(codes, np.array([4.0, 1.0, 2.0]))
        if biased
        else _uniform_bases(2)
    )

    covers, counts, _ = _sample_shadow_covers(
        codes, bases, 90_000, np.random.default_rng(5)
    )

    hits = (covers * counts[:, None]).sum(axis=0) / 90_000
    np.testing.assert_allclose(
        hits, _compute_cover_probabilities(codes, bases), rtol=0.03
    )


@pytest.mark.parametrize(
    ("labels", "weights", "expected"),
    [
        pytest.param(
            ["X", "Z"], [1.0, 4.0], [1.0, 1 / 3, 2 / 3, 0.0], id="sqrt-weights"
        ),
        pytest.param(
            ["X", "Y", "Z"], [1.0, 1.0, 1.0], [1.0, 1 / 3, 1 / 3, 1 / 3], id="uniform"
        ),
        pytest.param(["Z", "X"], [1.0, 0.0], [1.0, 0.0, 1.0, 0.0], id="zero-weight"),
    ],
)
def test_single_qubit_bases_follow_the_square_root_of_each_letters_weight(
    labels, weights, expected
):
    bases = _fit_biased_bases(_codes(labels), np.array(weights))

    np.testing.assert_allclose(bases[0], expected, atol=1e-9)


def test_fit_stays_finite_when_cover_probabilities_underflow():
    n_qubits = 700  # 3**-700 underflows a float64.
    labels = ["Z" * n_qubits] + [
        "I" * qubit + "X" + "I" * (n_qubits - qubit - 1) for qubit in range(n_qubits)
    ]

    bases = _fit_biased_bases(_codes(labels), np.ones(len(labels)))

    assert np.isfinite(bases).all()
    np.testing.assert_allclose(bases[:, 1:].sum(axis=1), 1.0)


def test_fitted_bases_lower_the_variance_bound_below_uniform():
    rng = np.random.default_rng(0)
    labels = ["".join(rng.choice(list("IXYZ"), 6)) for _ in range(40)]
    codes = _codes(labels)
    weights = rng.random(len(labels)) ** 4
    fitted = codes.any(axis=1)

    def bound(bases):
        return np.sum(
            weights[fitted] / _compute_cover_probabilities(codes[fitted], bases)
        )

    assert bound(_fit_biased_bases(codes, weights)) < bound(_uniform_bases(6))


def test_squared_coefficients_merge_repeats_within_an_observable():
    observables = (
        SparsePauliOp.from_list([("Z", 0.5), ("Z", 0.25)]),
        SparsePauliOp.from_list([("Z", -1.0)]),
    )
    union, per_obs_term_indices = flatten_observable_tuple(observables)

    np.testing.assert_allclose(
        _sum_squared_coefficients(observables, per_obs_term_indices, len(union)),
        [0.75**2 + 1.0],
    )
