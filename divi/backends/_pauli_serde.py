# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""The ``ham_ops`` string format: its wire serialisation and its group layout.

``ham_ops`` is a semicolon-separated dense Pauli string (``;`` between terms,
``|`` between groups) produced upstream by
:func:`~divi.circuits._sparse_pauli_op_to_ham_string`.  This module compresses
that string for transport via sparse encoding + gzip + base64, and resolves
which ``|``-group a given circuit was measured for.  :func:`compress_text` is
also the cloud encoding for circuits.  Not an operator encoding
(Jordan-Wigner, Bravyi-Kitaev) — purely an I/O and layout concern.
"""

import base64
import gzip
import warnings
from collections.abc import Sequence


def _assign_circuit_groups(
    n_circuits: int, n_groups: int, circuit_ham_map: Sequence[list[int]] | None
) -> list[int] | None:
    """Each circuit's ``|``-group index, or None when every group applies to all.

    Raises:
        ValueError: If ``circuit_ham_map`` does not hold one ``[start, end)``
            range per group, or its ranges do not cover every circuit exactly once.
    """
    if circuit_ham_map is None:
        return None
    if len(circuit_ham_map) != n_groups:
        raise ValueError(
            f"circuit_ham_map length ({len(circuit_ham_map)}) must match "
            f"number of ham_ops groups ({n_groups})."
        )
    group_of: dict[int, int] = {}
    for group, (start, end) in enumerate(circuit_ham_map):
        if not 0 <= start < end <= n_circuits:
            raise ValueError(
                f"circuit_ham_map range [{start}, {end}) is empty or out of bounds "
                f"for {n_circuits} circuits."
            )
        for index in range(start, end):
            if index in group_of:
                raise ValueError(
                    f"circuit_ham_map range [{start}, {end}) overlaps an earlier "
                    f"range at circuit {index}."
                )
            group_of[index] = group
    if len(group_of) < n_circuits:
        missing = [index for index in range(n_circuits) if index not in group_of]
        raise ValueError(
            f"circuit_ham_map does not cover every circuit; missing indices {missing}."
        )
    return [group_of[index] for index in range(n_circuits)]


def pad_ham_ops(
    ham_ops: str,
    circuit_ham_map: Sequence[list[int]] | None,
    n_qubits: Sequence[int],
    stacklevel: int = 3,
) -> str:
    """Pad terms shorter than their circuits onto the circuits' first qubits.

    ``ZZ`` measured on a 5-qubit circuit becomes ``ZZIII``, and results are
    keyed by the padded term. ``n_qubits`` holds each circuit's width, in
    submission order. Warns when anything is padded.

    Raises:
        ValueError: If a term is empty or holds anything but I, X, Y, Z, a
            group's terms differ in length after padding, short terms are
            measured on circuits of different widths, or ``circuit_ham_map``
            is malformed.
    """
    groups = ham_ops.split("|")
    for group in groups:
        terms = group.split(";")
        if not all(terms):
            raise ValueError(
                "Hamiltonian operators must be non-empty semicolon-separated strings."
            )
        if not set("".join(terms)) <= {"I", "X", "Y", "Z"}:
            raise ValueError(
                "Hamiltonian operators must contain only I, X, Y, Z characters."
            )

    circuit_groups = _assign_circuit_groups(len(n_qubits), len(groups), circuit_ham_map)
    if circuit_groups is None:
        widths = [set(n_qubits) for _ in groups]
    else:
        widths = [set() for _ in groups]
        for group, n in zip(circuit_groups, n_qubits):
            widths[group].add(n)

    padded, example = [], None
    for group, group_widths in zip(groups, widths):
        terms = group.split(";")
        if group_widths and min(map(len, terms)) < max(group_widths):
            if len(group_widths) > 1:
                raise ValueError(
                    f"Observables {terms} are too short for circuits of widths "
                    f"{sorted(group_widths)}."
                )
            (width,) = group_widths
            short = next(t for t in terms if len(t) < width)
            example = example or (short, short.ljust(width, "I"))
            terms = [t.ljust(width, "I") for t in terms]
        if len(set(map(len, terms))) > 1:
            raise ValueError("All Hamiltonian operators must have the same length.")
        padded.append(";".join(terms))

    if example:
        warnings.warn(
            "Observables shorter than their circuit are padded onto its first "
            f"qubits, e.g. {example[0]!r} -> {example[1]!r}.",
            stacklevel=stacklevel,
        )
    return "|".join(padded)


def split_ham_ops_per_circuit(
    n_circuits: int,
    ham_ops: str,
    circuit_ham_map: Sequence[list[int]] | None,
) -> list[str]:
    """Each circuit's ``ham_ops`` group, or all groups ``;``-joined without a map."""
    groups = ham_ops.split("|")
    circuit_groups = _assign_circuit_groups(n_circuits, len(groups), circuit_ham_map)
    if circuit_groups is None:
        return [";".join(groups)] * n_circuits
    return [groups[group] for group in circuit_groups]


def split_ham_ops_terms_per_circuit(
    n_circuits: int,
    ham_ops: str,
    circuit_ham_map: Sequence[list[int]] | None,
) -> list[list[str]]:
    """:func:`split_ham_ops_per_circuit` as term lists; lists are shared within
    a group, so do not mutate them."""
    per_circuit = split_ham_ops_per_circuit(n_circuits, ham_ops, circuit_ham_map)
    terms = {group: group.split(";") for group in set(per_circuit)}
    return [terms[group] for group in per_circuit]


def group_ham_ops(observables: Sequence[str]) -> tuple[str, list[list[int]]]:
    """``ham_ops`` and ``circuit_ham_map`` giving each circuit its observable.

    Each run of consecutive circuits sharing an observable becomes one group,
    so an observable that recurs later in the list appears in more than one.
    """
    groups: list[str] = []
    circuit_ham_map: list[list[int]] = []
    for index, observable in enumerate(observables):
        if groups and groups[-1] == observable:
            circuit_ham_map[-1][1] = index + 1
        else:
            groups.append(observable)
            circuit_ham_map.append([index, index + 1])
    return "|".join(groups), circuit_ham_map


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
    return "".join(f"{ch}{i}" for i, ch in enumerate(term) if ch != "I") or "I"


def compress_text(text: str) -> str:
    """Gzip then base64, the cloud's encoding for circuits and observable groups."""
    compressed = gzip.compress(text.encode("utf-8"), mtime=0)
    return base64.b64encode(compressed).decode("ascii")


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
    return f"@gzs{n_qubits}:{compress_text(sparse_str)}"


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
