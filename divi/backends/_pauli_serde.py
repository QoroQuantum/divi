# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""The ``ham_ops`` string format: its wire serialisation and its group layout.

``ham_ops`` is a semicolon-separated dense Pauli string (``;`` between terms,
``|`` between groups) produced upstream by
:func:`~divi.circuits._sparse_pauli_op_to_ham_string`.  This module compresses
that string for transport via sparse encoding + gzip + base64, and resolves
which ``|``-group a given circuit was measured for.  Not an operator encoding
(Jordan-Wigner, Bravyi-Kitaev) — purely an I/O and layout concern.
"""

import base64
import gzip


def ham_ops_group_for_circuit(
    circuit_index: int,
    ham_ops: str,
    circuit_ham_map: list[list[int]] | None,
) -> str:
    """The ``|``-group of *ham_ops* covering *circuit_index*, else all of it.

    Falling back to the whole string matters for auxiliary circuits submitted
    outside the per-group ranges (overlap circuits, for instance): they are
    evaluated against every observable rather than none.
    """
    if circuit_ham_map is None:
        return ham_ops

    groups = ham_ops.split("|")
    for group_index, (start, end) in enumerate(circuit_ham_map):
        if start <= circuit_index < end:
            return groups[group_index]

    return ham_ops


def ham_ops_terms_for_circuit(
    circuit_index: int,
    ham_ops: str,
    circuit_ham_map: list[list[int]] | None,
) -> list[str]:
    """The individual Pauli terms *circuit_index* was measured for.

    A matched group holds no ``|``, so flattening only affects the fall-back
    case where every group applies to this circuit.
    """
    group = ham_ops_group_for_circuit(circuit_index, ham_ops, circuit_ham_map)
    return group.replace("|", ";").split(";")


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
