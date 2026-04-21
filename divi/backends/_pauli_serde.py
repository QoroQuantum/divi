# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Wire-format serialisation for ``ham_ops`` payloads sent to the Qoro service.

``ham_ops`` is a semicolon-separated dense Pauli string (``;`` between terms,
``|`` between groups) produced upstream by
:func:`~divi.circuits.sparse_pauli_op_to_ham_string`.  This module compresses
that string for transport via sparse encoding + gzip + base64.  Not an
operator encoding (Jordan-Wigner, Bravyi-Kitaev) — purely an I/O concern
scoped to the Qoro backend.
"""

import base64
import gzip


def _dense_to_sparse(term: str) -> str:
    """Convert a dense Pauli string to sparse notation.

    Only non-Identity positions are encoded as ``<Pauli><index>`` pairs.
    An all-Identity term becomes ``I``.

    Example::

        >>> _dense_to_sparse("ZIIZIIII")
        'Z0Z3'
        >>> _dense_to_sparse("IIIIIIII")
        'I'
    """
    parts = []
    for i, ch in enumerate(term):
        if ch != "I":
            parts.append(f"{ch}{i}")
    return "".join(parts) if parts else "I"


def encode_ham_ops(dense_ham_ops: str) -> str:
    """Compress a semicolon-separated dense Pauli string for transport.

    Applies sparse Pauli encoding (only non-Identity positions) followed by
    gzip + base64 compression.  The result is prefixed with
    ``@gzs<n_qubits>:`` so the receiver can detect and decode it.

    Args:
        dense_ham_ops: Semicolon-separated dense Pauli strings,
            e.g. ``"ZZII;IZIZ;IIII"``.

    Returns:
        str: Encoded string of the form ``@gzs<n>:<base64_of_gzipped_sparse>``.

    Example::

        >>> encoded = encode_ham_ops("ZZII;IZIZ;IIII")
        >>> encoded.startswith("@gzs4:")
        True
    """
    if not dense_ham_ops:
        raise ValueError(
            "dense_ham_ops must be a non-empty semicolon-separated Pauli string"
        )
    terms = dense_ham_ops.split(";")
    n_qubits = len(terms[0])
    lengths = {len(t) for t in terms}
    if len(lengths) > 1:
        raise ValueError(
            f"All Pauli terms must have the same length; got lengths {lengths}"
        )
    sparse_str = ";".join(_dense_to_sparse(t) for t in terms)
    compressed = base64.b64encode(gzip.compress(sparse_str.encode("utf-8"))).decode(
        "ascii"
    )
    return f"@gzs{n_qubits}:{compressed}"


def compress_ham_ops(ham_ops: str) -> str:
    """Compress a ham_ops string for transport, handling ``|``-delimited groups.

    Each ``|``-delimited group is independently encoded via :func:`encode_ham_ops`.

    Args:
        ham_ops: Dense Pauli string, optionally with ``|``-delimited groups.

    Returns:
        Compressed string with each group prefixed by ``@gzs<n>:``.
    """
    groups = ham_ops.split("|")
    return "|".join(encode_ham_ops(g) for g in groups)
