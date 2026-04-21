# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for ham_ops wire-format serialisation (divi.backends._pauli_serde)."""

import pytest

from divi.backends._pauli_serde import (
    _dense_to_sparse,
    compress_ham_ops,
    encode_ham_ops,
)


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

    def test_encode_prefix(self):
        encoded = encode_ham_ops("ZZII;IZIZ;IIII")
        assert encoded.startswith("@gzs4:")

    def test_encode_large_qubit_count(self):
        dense = "Z" + "I" * 63
        encoded = encode_ham_ops(dense)
        assert encoded.startswith("@gzs64:")

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
