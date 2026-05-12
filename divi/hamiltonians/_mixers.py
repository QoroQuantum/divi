# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""QAOA mixer and driver Hamiltonian builders backed by ``SparsePauliOp``."""

import itertools
from collections.abc import Iterable, Sequence

import networkx as nx
from qiskit.quantum_info import SparsePauliOp

# ---------------------------------------------------------------------------
# Big-endian Pauli-label builders for ``SparsePauliOp.from_list``.
# Position ``n_qubits - 1 - qubit`` in the label corresponds to qubit ``qubit``.
# ---------------------------------------------------------------------------


def single_pauli_label(n_qubits: int, qubit: int, pauli: str) -> str:
    """Big-endian label with ``pauli`` at ``qubit`` and ``I`` elsewhere."""
    label = ["I"] * n_qubits
    label[n_qubits - 1 - qubit] = pauli
    return "".join(label)


def two_pauli_label(n_qubits: int, left: int, right: int, pauli: str) -> str:
    """Big-endian label with ``pauli`` at both ``left`` and ``right`` qubits."""
    label = ["I"] * n_qubits
    label[n_qubits - 1 - left] = pauli
    label[n_qubits - 1 - right] = pauli
    return "".join(label)


def multi_pauli_label(n_qubits: int, qubit_paulis: Sequence[tuple[int, str]]) -> str:
    """Big-endian label with the given ``(qubit, pauli)`` assignments."""
    label = ["I"] * n_qubits
    for qubit, pauli in qubit_paulis:
        label[n_qubits - 1 - qubit] = pauli
    return "".join(label)


def _validate_int_nodes(nodes: Iterable[int], builder: str) -> None:
    """Raise if any node is not a non-negative integer."""
    for node in nodes:
        if not isinstance(node, int):
            raise TypeError(f"{builder} requires integer qubit nodes.")
        if node < 0:
            raise ValueError(f"{builder} requires non-negative qubit nodes.")


def x_mixer_spo(n_qubits: int) -> SparsePauliOp:
    """Return the standard QAOA X mixer ``sum_i X_i``."""
    if n_qubits < 0:
        raise ValueError("n_qubits must be non-negative.")
    if n_qubits == 0:
        return SparsePauliOp.from_list([("", 0.0)])
    return SparsePauliOp.from_list(
        [(single_pauli_label(n_qubits, qubit, "X"), 1.0) for qubit in range(n_qubits)]
    )


def xy_mixer_spo(
    graph: nx.Graph | Iterable[tuple[int, int]], *, n_qubits: int | None = None
) -> SparsePauliOp:
    """Return the XY mixer ``0.5 * sum_(i,j) (X_i X_j + Y_i Y_j)``.

    The graph nodes must be integer qubit indices. ``n_qubits`` may be
    provided to preserve isolated trailing qubits.
    """
    nodes: set[int]
    if isinstance(graph, nx.Graph):
        edges = list(graph.edges())
        nodes = set(graph.nodes())  # type: ignore[bad-assignment]
    else:
        edges = list(graph)
        nodes = {n for edge in edges for n in edge}
    _validate_int_nodes(nodes, "xy_mixer_spo")

    inferred = max(nodes, default=-1) + 1
    n_qubits = inferred if n_qubits is None else n_qubits
    if n_qubits < inferred:
        raise ValueError("n_qubits is smaller than the largest graph node.")
    if n_qubits == 0:
        return SparsePauliOp.from_list([("", 0.0)])

    terms: list[tuple[str, float]] = []
    for left, right in edges:
        terms.append((two_pauli_label(n_qubits, left, right, "X"), 0.5))
        terms.append((two_pauli_label(n_qubits, left, right, "Y"), 0.5))
    if not terms:
        return SparsePauliOp.from_list([("I" * n_qubits, 0.0)])
    return SparsePauliOp.from_list(terms)


def bit_driver_spo(n_qubits: int, b: int) -> SparsePauliOp:
    """Bit-driver Hamiltonian ``(-1)^(b+1) * sum_i Z_i``.

    ``b=1`` returns ``+sum_i Z_i`` (rewards bitstrings with majority 1s);
    ``b=0`` returns ``-sum_i Z_i`` (rewards bitstrings with majority 0s).
    """
    if b not in (0, 1):
        raise ValueError(f"'b' must be either 0 or 1, got {b}")
    if n_qubits < 0:
        raise ValueError("n_qubits must be non-negative.")
    if n_qubits == 0:
        return SparsePauliOp.from_list([("", 0.0)])
    coeff = 1.0 if b == 1 else -1.0
    return SparsePauliOp.from_list(
        [(single_pauli_label(n_qubits, qubit, "Z"), coeff) for qubit in range(n_qubits)]
    )


def edge_driver_spo(
    graph: nx.Graph | Iterable[tuple[int, int]],
    reward: list[str],
    *,
    n_qubits: int | None = None,
) -> SparsePauliOp:
    """Edge-driver Hamiltonian rewarding edges whose endpoints match ``reward``.

    Mirrors the formulations in
    `pennylane.qaoa.cost.edge_driver
    <https://docs.pennylane.ai/en/stable/code/api/pennylane.qaoa.cost.edge_driver.html>`__:

    * ``reward`` ⊆ ``{"00","01","10","11"}``; ``"01"`` and ``"10"`` are
      treated as the undirected pair (must appear together or not at all).
    * ``len(reward) ∈ {0, 4}`` reduces to a constant identity term.
    * Otherwise the difference between rewarded and penalised energies is
      always ``1``.

    The graph nodes must be integer qubit indices. ``n_qubits`` may be
    provided to preserve isolated trailing qubits.
    """
    allowed = {"00", "01", "10", "11"}
    bad = [e for e in reward if e not in allowed]
    if bad:
        raise ValueError(
            f"Encountered invalid entry in 'reward', expected 2-bit "
            f"bitstrings; got {bad}."
        )
    if ("01" in reward) ^ ("10" in reward):
        raise ValueError(
            "'reward' cannot contain either '10' or '01' alone; must contain "
            "neither or both."
        )

    nodes: set[int]
    if isinstance(graph, nx.Graph):
        edges = list(graph.edges())
        nodes = set(graph.nodes())  # type: ignore[bad-assignment]
    else:
        edges = list(graph)
        nodes = {n for edge in edges for n in edge}
    _validate_int_nodes(nodes, "edge_driver_spo")

    inferred = max(nodes, default=-1) + 1
    n_qubits = inferred if n_qubits is None else n_qubits
    if n_qubits < inferred:
        raise ValueError("n_qubits is smaller than the largest graph node.")
    if n_qubits == 0:
        return SparsePauliOp.from_list([("", 0.0)])

    # Constant Hamiltonian (no preference).
    if len(reward) == 0 or len(reward) == 4:
        return SparsePauliOp.from_list(
            [(single_pauli_label(n_qubits, qubit, "I"), 1.0) for qubit in nodes]
            or [("I" * n_qubits, 0.0)]
        )

    # Collapse undirected edges by removing "01" duplicate.
    canonical = list(set(reward) - {"01"})
    sign = -1.0
    if len(canonical) == 2:
        canonical = list({"00", "10", "11"} - set(canonical))
        sign = 1.0
    selector = canonical[0]

    terms: list[tuple[str, float]] = []
    for left, right in edges:
        zz = two_pauli_label(n_qubits, left, right, "Z")
        zl = single_pauli_label(n_qubits, left, "Z")
        zr = single_pauli_label(n_qubits, right, "Z")
        if selector == "00":
            terms.extend([(zz, 0.25 * sign), (zl, 0.25 * sign), (zr, 0.25 * sign)])
        elif selector == "10":
            terms.append((zz, -0.5 * sign))
        elif selector == "11":
            terms.extend([(zz, 0.25 * sign), (zl, -0.25 * sign), (zr, -0.25 * sign)])

    if not terms:
        return SparsePauliOp.from_list([("I" * n_qubits, 0.0)])
    return SparsePauliOp.from_list(terms)


def bit_flip_mixer_spo(graph: nx.Graph, b: int) -> SparsePauliOp:
    """Bit-flip mixer over ``graph`` (per Hadfield et al. 2019).

    For each vertex ``v`` with neighbour set ``N(v)`` of degree ``d(v)``:

    .. math::
        \\frac{1}{2^{d(v)}} X_v \\prod_{w \\in N(v)} (I + (-1)^b Z_w)

    expanded into ``2^{d(v)}`` Pauli terms per vertex.

    The graph nodes must be non-negative integer qubit indices.
    """
    if b not in (0, 1):
        raise ValueError(f"'b' must be either 0 or 1, got {b}")
    if not isinstance(graph, nx.Graph):
        raise TypeError("bit_flip_mixer_spo requires a networkx.Graph.")
    nodes = list(graph.nodes())
    _validate_int_nodes(nodes, "bit_flip_mixer_spo")

    n_qubits = (max(nodes) + 1) if nodes else 0
    if n_qubits == 0:
        return SparsePauliOp.from_list([("", 0.0)])

    sign = 1.0 if b == 0 else -1.0
    terms: list[tuple[str, float]] = []
    for vertex in nodes:
        neighbours = list(graph.neighbors(vertex))
        degree = len(neighbours)
        scale = 0.5**degree
        # Each neighbour contributes either I or sign*Z.
        for choice in itertools.product([(None, 1.0), ("Z", sign)], repeat=degree):
            qubit_paulis: list[tuple[int, str]] = [(vertex, "X")]
            coeff = scale
            for neighbour, (op, sgn) in zip(neighbours, choice):
                coeff *= sgn
                if op is not None:
                    qubit_paulis.append((neighbour, op))
            terms.append((multi_pauli_label(n_qubits, qubit_paulis), coeff))

    if not terms:
        return SparsePauliOp.from_list([("I" * n_qubits, 0.0)])
    return SparsePauliOp.from_list(terms)
