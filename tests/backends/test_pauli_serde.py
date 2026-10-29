# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ham_ops wire-format serialisation (divi.backends._pauli_serde)."""

import base64
import gzip

import pytest

from divi.backends._pauli_serde import (
    _dense_to_sparse,
    compress_ham_ops,
    encode_ham_ops,
    group_ham_ops,
    ham_ops_terms_for_circuit,
    ham_ops_terms_per_circuit,
    pad_ham_ops,
)
from tests._helpers import exact_match
from tests.backends._helpers import padding_warning


def _too_short(term: str, widths: list[int]) -> str:
    return f"Observables [{term!r}] are too short for circuits of widths {widths}."


@pytest.mark.parametrize(
    "circuit_index, ham_ops, circuit_ham_map, expected",
    [
        (0, "ZI;IZ;XX", None, ["ZI", "IZ", "XX"]),
        (0, "ZI;IZ|XX;YY", None, ["ZI", "IZ", "XX", "YY"]),
        (0, "ZI;IZ|XX;YY", [[0, 3], [3, 5]], ["ZI", "IZ"]),
        (2, "ZI;IZ|XX;YY", [[0, 3], [3, 5]], ["ZI", "IZ"]),
        (3, "ZI;IZ|XX;YY", [[0, 3], [3, 5]], ["XX", "YY"]),
        (4, "ZI;IZ|XX;YY", [[0, 3], [3, 5]], ["XX", "YY"]),
        (10, "ZI|XX", [[0, 2], [2, 4]], ["ZI", "XX"]),
    ],
    ids=[
        "no-map-flat",
        "no-map-pipe-delimited",
        "map-first-group-start",
        "map-first-group-end",
        "map-second-group-start",
        "map-second-group-end",
        "index-outside-all-ranges",
    ],
)
def test_ham_ops_terms_for_circuit(circuit_index, ham_ops, circuit_ham_map, expected):
    assert (
        ham_ops_terms_for_circuit(circuit_index, ham_ops, circuit_ham_map) == expected
    )


@pytest.mark.parametrize(
    "n_circuits, ham_ops, circuit_ham_map",
    [
        (3, "ZI;IZ;XX", None),
        (2, "ZI;IZ|XX;YY", None),
        (5, "ZI;IZ|XX;YY", [[0, 3], [3, 5]]),
        (8, "ZI;IZ|XX;YY", [[0, 3], [3, 5]]),
        (4, "ZI|XX|YY", [[2, 3], [0, 2], [3, 4]]),
        (0, "ZI|XX", [[0, 2], [2, 4]]),
    ],
    ids=[
        "no-map-flat",
        "no-map-pipe-delimited",
        "map-covers-every-circuit",
        "circuits-outside-every-range",
        "unsorted-map",
        "no-circuits",
    ],
)
def test_ham_ops_terms_per_circuit_matches_per_circuit_lookup(
    n_circuits, ham_ops, circuit_ham_map
):
    assert ham_ops_terms_per_circuit(n_circuits, ham_ops, circuit_ham_map) == [
        ham_ops_terms_for_circuit(i, ham_ops, circuit_ham_map)
        for i in range(n_circuits)
    ]


@pytest.mark.parametrize(
    "observables, expected",
    [
        (["ZI;IZ"], ("ZI;IZ", [[0, 1]])),
        (["ZI", "ZI", "XX"], ("ZI|XX", [[0, 2], [2, 3]])),
        (["XX", "ZZ", "XX", "XX"], ("XX|ZZ|XX", [[0, 1], [1, 2], [2, 4]])),
    ],
    ids=["one-circuit", "leading-run", "recurring-observable"],
)
def test_group_ham_ops_gives_each_circuit_its_observable(observables, expected):
    ham_ops, circuit_ham_map = group_ham_ops(observables)

    assert (ham_ops, circuit_ham_map) == expected
    assert [
        ";".join(terms)
        for terms in ham_ops_terms_per_circuit(
            len(observables), ham_ops, circuit_ham_map
        )
    ] == observables


class TestPadHamOps:
    """Tests for padding terms shorter than their circuits."""

    def test_full_length_terms_are_unchanged(self, recwarn):
        assert pad_ham_ops("ZZI;IXX", None, [3, 3]) == "ZZI;IXX"
        assert not recwarn.list

    @pytest.mark.parametrize(
        "ham_ops, widths, short, padded, expected",
        [
            ("ZZ;XX", [5], "ZZ", "ZZIII", "ZZIII;XXIII"),
            ("ZZZ;Z", [3], "Z", "ZII", "ZZZ;ZII"),
        ],
        ids=[
            "short_terms_act_on_first_qubits",
            "warning_example_is_the_first_short_term",
        ],
    )
    def test_short_terms_are_padded_with_one_warning(
        self, ham_ops, widths, short, padded, expected
    ):
        with pytest.warns(
            UserWarning, match=exact_match(padding_warning(short, padded))
        ):
            assert pad_ham_ops(ham_ops, None, widths) == expected

    def test_each_group_is_padded_to_its_own_circuits_in_one_warning(self):
        with pytest.warns(
            UserWarning, match=exact_match(padding_warning("Z", "ZI"))
        ) as record:
            padded = pad_ham_ops("Z|X", [[0, 1], [1, 2]], [2, 3])
        assert padded == "ZI|XII"
        assert len(record) == 1

    def test_short_group_on_circuits_of_different_widths_raises(self):
        with pytest.raises(ValueError, match=exact_match(_too_short("ZZ", [3, 5]))):
            pad_ham_ops("ZZ", None, [3, 5])

    def test_circuit_outside_every_range_is_measured_on_every_group(self):
        """The 3-qubit circuit outside both ranges widens both 2-qubit groups."""
        with pytest.raises(ValueError, match=exact_match(_too_short("ZZ", [2, 3]))):
            pad_ham_ops("ZZ|XX", [[0, 1], [1, 2]], [2, 2, 3])


class TestCompressedObservables:
    """Tests for sparse+gzip observable compression."""

    # -- _dense_to_sparse ------------------------------------------------

    @pytest.mark.parametrize(
        "dense, expected",
        [
            ("Z", "Z0"),
            ("IZ", "Z1"),
            ("ZZ", "Z0Z1"),
            ("XYZ", "X0Y1Z2"),
            ("IIII", "I"),
            ("ZIIZ", "Z0Z3"),
            ("XI", "X0"),
        ],
    )
    def test_dense_to_sparse(self, dense, expected):
        assert _dense_to_sparse(dense) == expected

    # -- encode_ham_ops --------------------------------------------------

    def test_encoded_body_decodes_to_the_sparse_terms(self):
        encoded = encode_ham_ops("ZZII;IZIZ;IIII")
        prefix, body = encoded.split(":", 1)
        assert prefix == "@gzs4"
        assert gzip.decompress(base64.b64decode(body)).decode() == "Z0Z1;Z1Z3;I"

    @pytest.mark.parametrize(
        "dense, message",
        [
            ("", "dense_ham_ops must be a non-empty semicolon-separated Pauli string"),
            ("ZZ;Z", "All Pauli terms must have the same length; got lengths {1, 2}"),
        ],
        ids=["empty", "ragged"],
    )
    def test_encode_rejects_malformed_input(self, dense, message):
        with pytest.raises(ValueError, match=exact_match(message)):
            encode_ham_ops(dense)

    def test_encode_large_qubit_count(self):
        prefix, body = encode_ham_ops("I" * 63 + "Z").split(":", 1)
        assert prefix == "@gzs64"
        assert gzip.decompress(base64.b64decode(body)).decode() == "Z63"

    def test_compress_ham_ops_multi_group(self):
        """Pipe-delimited groups are each independently compressed."""
        group_a = "ZZII;IZIZ"
        group_b = "XXII;IYIZ"
        compressed = compress_ham_ops(f"{group_a}|{group_b}")
        parts = compressed.split("|")
        assert len(parts) == 2
        assert parts[0] == encode_ham_ops(group_a)
        assert parts[1] == encode_ham_ops(group_b)

    def test_compression_ratio(self):
        """Encoding should produce a string shorter than the dense input for large Hamiltonians."""
        # 64-qubit Hamiltonian with 100 sparse terms
        terms = []
        for i in range(100):
            paulis = ["I"] * 64
            paulis[i % 64] = "Z"
            paulis[(i + 1) % 64] = "Z"
            terms.append("".join(paulis))
        dense = ";".join(terms)
        encoded = encode_ham_ops(dense)
        assert len(encoded) < len(dense)
