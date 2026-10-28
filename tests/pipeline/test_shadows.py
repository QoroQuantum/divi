# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from divi.pipeline._shadows import _sample_shadow_covers, _shadow_term_weights


def test_term_weights_are_inverse_cover_probabilities():
    np.testing.assert_allclose(
        _shadow_term_weights(["ZII", "XYI", "III"]), [3.0, 9.0, 1.0]
    )


@pytest.mark.parametrize(
    "labels",
    [["ZI", "XI", "IZ", "ZZ", "XZ"], ["ZZZZ"]],
    ids=["overlapping-terms", "rarely-covered-term"],
)
def test_covers_fit_one_basis_and_account_for_every_draw(labels):
    covers, counts, empty_draws = _sample_shadow_covers(
        labels, 500, np.random.default_rng(3)
    )

    assert counts.sum() + empty_draws == 500
    assert covers.any(axis=1).all()
    for pattern in covers:
        covered = [label for label, hit in zip(labels, pattern) if hit]
        for qubit in range(len(labels[0])):
            assert len({label[qubit] for label in covered} - {"I"}) <= 1


def test_terms_are_covered_at_their_inverse_weight_rate():
    labels = ["ZI", "XY"]
    covers, counts, _ = _sample_shadow_covers(labels, 90_000, np.random.default_rng(5))

    hits = (covers * counts[:, None]).sum(axis=0) / 90_000
    np.testing.assert_allclose(hits, 1 / _shadow_term_weights(labels), rtol=0.03)
