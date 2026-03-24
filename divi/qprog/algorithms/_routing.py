# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Routing problem classes and utilities for QAOA (TSP, CVRP)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pennylane as qml
import pennylane.qaoa as pqaoa
import sympy
from scipy.optimize import linear_sum_assignment

from divi.hamiltonians import qubo_to_ising
from divi.qprog.algorithms._initial_state import (
    InitialState,
    SuperpositionState,
    WState,
    build_block_xy_mixer_graph,
)
from divi.qprog.algorithms._problem import Problem

# --- TSP utilities ---


def create_tsp_qubo(
    cost_matrix: npt.NDArray[np.floating],
    start_city: int = 0,
    constraint_penalty: float = 4.0,
    objective_weight: float = 1.0,
    reduced: bool = True,
) -> dict[tuple[int, ...], float]:
    """Generate a QUBO for TSP with a fixed start city.

    Uses the standard node-ordering (assignment) encoding where binary
    variable ``x_{i,t}`` indicates that city *i* is visited at time step *t*.
    The start city is fixed, reducing the problem from *n²* to *(n-1)²* qubits.

    By default the **reduced phase operator** from CE-QAOA
    (`arXiv:2511.14296 <https://arxiv.org/abs/2511.14296>`_, Eq. 13) is
    used: row constraints (each time step has exactly one city) are
    **omitted** because they are enforced structurally by the W-state
    initialisation and XY mixer.  Only column constraints and the
    objective remain.

    Args:
        cost_matrix: Symmetric (n × n) distance/cost matrix.
        start_city: Index of the fixed start/end city. Defaults to 0.
        constraint_penalty: Penalty strength for constraint violations.
        objective_weight: Weight for the objective function.
        reduced: If ``True`` (default), omit row penalties — the correct
            choice when using CE-QAOA (W-state + XY mixer) since row
            one-hot constraints are enforced by the ansatz.

            Set to ``False`` only if you are using this QUBO with a
            **standard QAOA** ansatz (Hadamard init + X mixer) or any
            other method that does **not** structurally enforce row
            constraints.  In that case both row and column penalties are
            included so the Hamiltonian alone encodes feasibility.

    Returns:
        Upper-triangular QUBO matrix of shape ``(m², m²)`` where
        ``m = n - 1``, compatible with ``QUBOProblemTypes``.

    Raises:
        ValueError: If cost_matrix is not square or start_city is out of range.
    """
    n = cost_matrix.shape[0]
    if cost_matrix.shape != (n, n):
        raise ValueError(f"cost_matrix must be square, got shape {cost_matrix.shape}.")
    if not (0 <= start_city < n):
        raise ValueError(f"start_city {start_city} out of range [0, {n}).")

    m = n - 1  # reduced dimension (cities excluding start)
    cities = [c for c in range(n) if c != start_city]

    # Variable ordering: x_{i,t} -> qubit index i*m + t
    I_m = np.eye(m)
    J_m = np.ones((m, m))

    if reduced:
        # Row penalties OMITTED — enforced by W-state + XY mixer (CE-QAOA).
        Q = np.zeros((m * m, m * m))
    else:
        # Row penalties: A * sum_t (1 - sum_i x_{i,t})^2
        Q = constraint_penalty * np.kron(J_m - I_m, I_m)

    # --- Column penalties: A * sum_i (1 - sum_t x_{i,t})^2 ---
    Q += constraint_penalty * np.kron(I_m, J_m - I_m)

    # Linear term: each variable appears in column penalty (and row if not reduced)
    n_penalties = 1 if reduced else 2
    np.fill_diagonal(Q, np.diag(Q) - n_penalties * constraint_penalty)

    # --- Objective: consecutive timeslot costs ---
    W_reduced = cost_matrix[np.ix_(cities, cities)]
    S = np.zeros((m, m))
    for t in range(m - 1):
        S[t, t + 1] = 1.0
    Q += objective_weight * np.kron(W_reduced, S)

    # --- Depot edges (linear terms) ---
    depot_to_first = cost_matrix[start_city, cities]
    for j in range(m):
        Q[j * m, j * m] += objective_weight * depot_to_first[j]

    last_to_depot = cost_matrix[cities, start_city]
    for i in range(m):
        Q[i * m + (m - 1), i * m + (m - 1)] += objective_weight * last_to_depot[i]

    # Return upper-triangular matrix
    return np.triu(Q + Q.T - np.diag(np.diag(Q)))


def is_valid_tsp_tour(bitstring: str, n_cities: int) -> bool:
    """Check if a bitstring represents a valid TSP tour.

    A valid tour has the assignment matrix (reshaped from the bitstring)
    as a permutation matrix: every row and column sums to exactly 1.

    Args:
        bitstring: Binary string of length ``(n_cities - 1)²``.
        n_cities: Total number of cities (including the fixed start city).

    Returns:
        True if the bitstring encodes a valid permutation.
    """
    m = n_cities - 1
    if len(bitstring) != m * m:
        return False

    mat = np.array([int(b) for b in bitstring]).reshape(m, m)
    return bool(np.all(mat.sum(axis=0) == 1) and np.all(mat.sum(axis=1) == 1))


def decode_tsp_solution(
    bitstring: str,
    n_cities: int,
    start_city: int = 0,
) -> list[int] | None:
    """Decode a bitstring into a city tour.

    Args:
        bitstring: Binary string of length ``(n_cities - 1)²``.
        n_cities: Total number of cities.
        start_city: Index of the fixed start/end city.

    Returns:
        Ordered list of city indices forming the tour
        (starting and ending at *start_city*), or None if infeasible.
    """
    if not is_valid_tsp_tour(bitstring, n_cities):
        return None

    m = n_cities - 1
    cities = [c for c in range(n_cities) if c != start_city]
    mat = np.array([int(b) for b in bitstring]).reshape(m, m)

    tour = [start_city]
    for t in range(m):
        city_idx = int(np.argmax(mat[:, t]))
        tour.append(cities[city_idx])
    tour.append(start_city)

    return tour


def tour_cost(
    tour: list[int],
    cost_matrix: npt.NDArray[np.floating],
) -> float:
    """Compute the total travel cost of a tour.

    Args:
        tour: Ordered list of city indices (first and last should match
            for a round trip).
        cost_matrix: Distance/cost matrix.

    Returns:
        Total tour cost.
    """
    total = 0.0
    for i in range(len(tour) - 1):
        total += cost_matrix[tour[i], tour[i + 1]]
    return float(total)


def repair_tsp_solution(
    bitstring: str,
    n_cities: int,
    start_city: int,
    cost_matrix: npt.NDArray[np.floating],
) -> tuple[str, list[int], float]:
    """Repair an infeasible bitstring to the nearest valid permutation.

    Uses the Hungarian algorithm (``scipy.optimize.linear_sum_assignment``)
    to find the permutation matrix closest in Hamming distance to the
    (possibly infeasible) assignment matrix.

    If the bitstring is already feasible, it is returned unchanged.

    Args:
        bitstring: Binary string of length ``(n_cities - 1)²``.
        n_cities: Total number of cities.
        start_city: Index of the fixed start/end city.
        cost_matrix: Distance/cost matrix (used for cost evaluation).

    Returns:
        Tuple of (repaired_bitstring, tour, cost).
    """
    m = n_cities - 1
    cities = [c for c in range(n_cities) if c != start_city]

    # Parse the bitstring into a (soft) assignment matrix
    mat = np.array([int(b) for b in bitstring], dtype=np.float64).reshape(m, m)

    # Hungarian algorithm minimises cost; we want the assignment closest
    # to the soft matrix, so we use (1 - mat) as the cost matrix
    # (maximise overlap with the measured matrix).
    row_ind, col_ind = linear_sum_assignment(1.0 - mat)

    # Build the repaired permutation matrix
    repaired = np.zeros((m, m), dtype=int)
    repaired[row_ind, col_ind] = 1

    repaired_bitstring = "".join(str(x) for x in repaired.flatten())

    # Decode tour
    tour = [start_city]
    for t in range(m):
        city_idx = int(np.argmax(repaired[:, t]))
        tour.append(cities[city_idx])
    tour.append(start_city)

    cost = tour_cost(tour, cost_matrix)

    return repaired_bitstring, tour, cost


# --- CVRP utilities (one-hot) ---


def create_cvrp_qubo(
    cost_matrix: npt.NDArray[np.floating],
    demands: npt.NDArray[np.floating],
    capacity: float,
    n_vehicles: int,
    depot: int = 0,
    constraint_penalty: float = 4.0,
    objective_weight: float = 1.0,
    capacity_penalty: float = 4.0,
) -> dict[tuple[int, ...], float]:
    """Generate a QUBO for CVRP with a fixed depot.

    Uses one-hot block encoding where each vehicle *v* has a block of
    ``n_customers × max_steps`` qubits. The depot is fixed and not encoded.

    Constraints enforced via init + mixer (row one-hot per vehicle block):
    - Each vehicle visits at most one customer per time step.

    Constraints in the phase operator (penalties):
    - **Column one-hot** (constraint_penalty): each customer visited exactly once
      across all vehicles.
    - **Capacity** (capacity_penalty): each vehicle route does not exceed capacity.
    - **Objective** (objective_weight): minimise total travel distance.

    Variable ordering: ``x_{v,i,t}`` → qubit ``v * K * T + t * K + i``
    where K = n_customers, T = max_steps = K.

    The QUBO matrix is built via Kronecker products and vectorised
    numpy operations.

    Args:
        cost_matrix: Distance matrix of shape ``(n_nodes, n_nodes)``.
        demands: Customer demands, shape ``(n_nodes,)``. Depot demand should be 0.
        capacity: Vehicle capacity.
        n_vehicles: Number of vehicles.
        depot: Index of the depot node. Defaults to 0.
        constraint_penalty: Penalty for customer-visit constraints.
        objective_weight: Weight for the objective function.
        capacity_penalty: Penalty for capacity constraint violations.

    Returns:
        Upper-triangular QUBO matrix compatible with ``QUBOProblemTypes``.

    Raises:
        ValueError: If inputs are inconsistent.
    """
    n_nodes = cost_matrix.shape[0]
    if cost_matrix.shape != (n_nodes, n_nodes):
        raise ValueError(f"cost_matrix must be square, got shape {cost_matrix.shape}.")
    if len(demands) != n_nodes:
        raise ValueError(
            f"demands length ({len(demands)}) must match "
            f"cost_matrix size ({n_nodes})."
        )

    customers = [c for c in range(n_nodes) if c != depot]
    K = len(customers)  # n_customers
    V = n_vehicles
    T = K  # max_steps = n_customers (worst case)
    N = V * K * T  # total qubits

    # Variable ordering: x_{v,i,t} -> qubit v*K*T + t*K + i
    # Reshape views use (V, T, K) -> flatten to N

    I_K = np.eye(K)
    J_T = np.ones((T, T))
    I_V = np.eye(V)
    J_V = np.ones((V, V))

    Q = np.zeros((N, N))

    # --- Customer visit constraints ---
    # For each customer i: (sum_{v,t} x_{v,i,t} - 1)^2
    # The sum runs over all (v,t) pairs for a fixed customer i.
    # Off-diagonal: 2A between all pairs of qubits sharing the same customer i
    #   across different (v,t) slots.
    # In Kronecker form with ordering (V, T, K):
    #   same-customer coupling = I_K, all (v,t) pairs = J_V ⊗ J_T
    #   off-diagonal part: (J_V ⊗ J_T - I_{VT}) ⊗ I_K  (exclude self-pairs)
    visit_offdiag = np.kron(np.kron(J_V, J_T) - np.eye(V * T), I_K)
    Q += constraint_penalty * visit_offdiag

    # Linear: -A per variable (from -2*x + x^2 = -x per constraint)
    # Each variable participates in exactly one customer constraint
    np.fill_diagonal(Q, np.diag(Q) - constraint_penalty)

    # --- Capacity constraints ---
    # For each vehicle v: (sum_{i,t} d_i * x_{v,i,t} - C)^2
    # = sum_{i,t} sum_{j,s} d_i*d_j * x_{v,i,t} * x_{v,j,s}
    #   - 2C * sum_{i,t} d_i * x_{v,i,t} + C^2
    #
    # The quadratic part for a single vehicle is:
    #   d ⊗ d^T ⊗ J_T  (all timestep pairs, weighted by demand products)
    # where d is the customer demand vector.
    d = demands[customers]
    dd = np.outer(d, d)  # K×K demand product matrix

    # Per-vehicle capacity quadratic: (J_T ⊗ dd) for each vehicle
    # Across vehicles: block-diagonal (I_V)
    cap_quad_per_vehicle = np.kron(J_T, dd)  # (KT × KT)
    Q += capacity_penalty * np.kron(I_V, cap_quad_per_vehicle)

    # Capacity linear: -2C * d_i per variable (+ d_i^2 from diagonal of quad)
    # The d_i^2 is already in the diagonal of cap_quad.
    # We need: -2C * d_i additionally on the diagonal.
    # d_i for each qubit: tile d across all (V, T) slots
    d_per_qubit = np.tile(d, V * T)
    np.fill_diagonal(Q, np.diag(Q) - 2.0 * capacity_penalty * capacity * d_per_qubit)

    # --- Objective: travel cost ---
    W = cost_matrix[np.ix_(customers, customers)]  # K×K reduced cost matrix

    # Consecutive steps: x_{v,i,t} * x_{v,j,t+1} with weight W[i,j]
    # Shift matrix S[t, t+1] = 1
    S = np.zeros((T, T))
    for t in range(T - 1):
        S[t, t + 1] = 1.0

    # Per-vehicle: W ⊗ S (customer-to-customer across consecutive timesteps)
    # Across vehicles: block-diagonal (I_V)
    obj_consecutive = np.kron(S, W)  # (KT × KT)
    Q += objective_weight * np.kron(I_V, obj_consecutive)

    # Depot edges (linear terms)
    depot_to_first = cost_matrix[depot, customers]  # K vector
    last_to_depot = cost_matrix[customers, depot]  # K vector

    for v in range(V):
        base = v * K * T
        # depot -> first timestep (t=0): weight on x_{v,i,0}
        for i in range(K):
            Q[base + i, base + i] += objective_weight * depot_to_first[i]
        # last timestep (t=T-1) -> depot: weight on x_{v,i,T-1}
        for i in range(K):
            q = base + (T - 1) * K + i
            Q[q, q] += objective_weight * last_to_depot[i]

    # Return upper-triangular matrix
    return np.triu(Q + Q.T - np.diag(np.diag(Q)))


def cvrp_block_structure(
    n_customers: int,
    n_vehicles: int,
) -> tuple[int, int]:
    """Return the block structure for CVRP CE-QAOA.

    Args:
        n_customers: Number of customers (excluding depot).
        n_vehicles: Number of vehicles.

    Returns:
        Tuple of (block_size, n_blocks) where:
        - block_size = n_customers (one qubit per customer per time step)
        - n_blocks = n_vehicles * n_customers (one block per vehicle × time step)
    """
    max_steps = n_customers
    return n_customers, n_vehicles * max_steps


def is_valid_cvrp_solution(
    bitstring: str,
    n_customers: int,
    n_vehicles: int,
    demands: npt.NDArray[np.floating],
    capacity: float,
    depot: int = 0,
) -> bool:
    """Check if a bitstring represents a valid CVRP solution.

    Validates:
    - Each customer is visited exactly once across all vehicles.
    - Each vehicle's total demand does not exceed capacity.

    Args:
        bitstring: Binary string.
        n_customers: Number of customers (excluding depot).
        n_vehicles: Number of vehicles.
        demands: Customer demands (length = total nodes including depot).
        capacity: Vehicle capacity.
        depot: Depot index.

    Returns:
        True if the solution is feasible.
    """
    max_steps = n_customers
    expected_len = n_vehicles * n_customers * max_steps
    if len(bitstring) != expected_len:
        return False

    customers = [c for c in range(len(demands)) if c != depot]
    bits = np.array([int(b) for b in bitstring]).reshape(
        n_vehicles, max_steps, n_customers
    )

    # Check each customer visited exactly once
    total_visits = bits.sum(axis=(0, 1))
    if not np.all(total_visits == 1):
        return False

    # Check capacity per vehicle
    for v in range(n_vehicles):
        vehicle_demand = 0.0
        for t in range(max_steps):
            for i in range(n_customers):
                if bits[v, t, i] == 1:
                    vehicle_demand += demands[customers[i]]
        if vehicle_demand > capacity + 1e-9:
            return False

    return True


def decode_cvrp_solution(
    bitstring: str,
    n_customers: int,
    n_vehicles: int,
    depot: int = 0,
    n_nodes: int | None = None,
) -> list[list[int]] | None:
    """Decode a bitstring into vehicle routes.

    Args:
        bitstring: Binary string.
        n_customers: Number of customers (excluding depot).
        n_vehicles: Number of vehicles.
        depot: Depot index.
        n_nodes: Total number of nodes (if None, inferred as n_customers + 1).

    Returns:
        List of routes, where each route is ``[depot, c1, c2, ..., depot]``,
        or None if the bitstring cannot be decoded.
    """
    if n_nodes is None:
        n_nodes = n_customers + 1

    max_steps = n_customers
    expected_len = n_vehicles * n_customers * max_steps
    if len(bitstring) != expected_len:
        return None

    customers = [c for c in range(n_nodes) if c != depot]
    bits = np.array([int(b) for b in bitstring]).reshape(
        n_vehicles, max_steps, n_customers
    )

    routes = []
    for v in range(n_vehicles):
        route = [depot]
        for t in range(max_steps):
            assigned = np.where(bits[v, t] == 1)[0]
            if len(assigned) == 1:
                route.append(customers[assigned[0]])
        route.append(depot)
        routes.append(route)

    return routes


def repair_cvrp_solution(
    bitstring: str,
    n_customers: int,
    n_vehicles: int,
    cost_matrix: npt.NDArray[np.floating],
    demands: npt.NDArray[np.floating],
    capacity: float,
    depot: int = 0,
) -> tuple[str, list[list[int]], float]:
    """Repair an infeasible CVRP bitstring.

    Uses a greedy assignment strategy:
    1. Parse the soft assignment matrix.
    2. Use the Hungarian algorithm on a cost matrix derived from the
       soft assignments to find the best customer-to-(vehicle, timestep) mapping.
    3. Verify capacity constraints; reassign overflows greedily.

    Args:
        bitstring: Binary string.
        n_customers: Number of customers (excluding depot).
        n_vehicles: Number of vehicles.
        cost_matrix: Distance matrix.
        demands: Customer demands.
        capacity: Vehicle capacity.
        depot: Depot index.

    Returns:
        Tuple of (repaired_bitstring, routes, total_cost).
    """
    max_steps = n_customers
    n_nodes = len(demands)
    customers = [c for c in range(n_nodes) if c != depot]

    bits = np.array([int(b) for b in bitstring], dtype=np.float64).reshape(
        n_vehicles * max_steps, n_customers
    )

    # Use Hungarian algorithm to find best assignment of customers
    # to (vehicle, timestep) slots
    # Cost matrix: 1 - soft_assignment (to maximise overlap)
    row_ind, col_ind = linear_sum_assignment(1.0 - bits)

    # Build repaired assignment
    repaired = np.zeros((n_vehicles, max_steps, n_customers), dtype=int)
    slot_to_customer = {}
    for slot, cust in zip(row_ind, col_ind):
        if slot < n_vehicles * max_steps and cust < n_customers:
            v = slot // max_steps
            t = slot % max_steps
            repaired[v, t, cust] = 1
            slot_to_customer[(v, t)] = cust

    # Check capacity and greedily reassign overflows
    for v in range(n_vehicles):
        vehicle_demand = 0.0
        for t in range(max_steps):
            assigned = np.where(repaired[v, t] == 1)[0]
            for cust_idx in assigned:
                cust_demand = demands[customers[cust_idx]]
                if vehicle_demand + cust_demand > capacity + 1e-9:
                    # Try to move to another vehicle with remaining capacity
                    repaired[v, t, cust_idx] = 0
                    placed = False
                    for v2 in range(n_vehicles):
                        if v2 == v:
                            continue
                        v2_demand = sum(
                            demands[customers[i]]
                            for t2 in range(max_steps)
                            for i in range(n_customers)
                            if repaired[v2, t2, i] == 1
                        )
                        if v2_demand + cust_demand <= capacity + 1e-9:
                            # Find empty slot
                            for t2 in range(max_steps):
                                if repaired[v2, t2].sum() == 0:
                                    repaired[v2, t2, cust_idx] = 1
                                    placed = True
                                    break
                            if placed:
                                break
                    if not placed:
                        # Last resort: put it back
                        repaired[v, t, cust_idx] = 1
                        vehicle_demand += cust_demand
                else:
                    vehicle_demand += cust_demand

    repaired_bitstring = "".join(str(x) for x in repaired.flatten())
    routes = decode_cvrp_solution(
        repaired_bitstring, n_customers, n_vehicles, depot, n_nodes
    )

    # Compute total cost
    total_cost = 0.0
    if routes is not None:
        for route in routes:
            for i in range(len(route) - 1):
                total_cost += cost_matrix[route[i], route[i + 1]]

    return repaired_bitstring, routes or [], float(total_cost)


# --- CVRP utilities (binary encoding) ---


@dataclass(frozen=True)
class BinaryBlockConfig:
    """Configuration for binary-encoded CE-QAOA blocks.

    Attributes:
        n_customers: Number of customers (excluding depot).
        n_vehicles: Number of vehicles.
        max_steps: Maximum route length per vehicle.
        bits_per_slot: Qubits per slot (ceil(log₂(n_customers + 1))).
        n_slots: Total number of (vehicle, timestep) slots.
        n_qubits: Total qubit count.
    """

    n_customers: int
    n_vehicles: int
    max_steps: int
    bits_per_slot: int
    n_slots: int
    n_qubits: int


def binary_block_config(
    n_customers: int,
    n_vehicles: int,
    max_steps: int | None = None,
) -> BinaryBlockConfig:
    """Compute the binary encoding layout.

    Args:
        n_customers: Number of customers (excluding depot).
        n_vehicles: Number of vehicles.
        max_steps: Maximum route length per vehicle. Defaults to n_customers
            (worst case: all customers on one vehicle).

    Returns:
        :class:`BinaryBlockConfig` with the encoding parameters.
    """
    if max_steps is None:
        max_steps = n_customers

    # Need to encode values 0..n_customers (0 = empty slot)
    bits_per_slot = math.ceil(math.log2(n_customers + 1)) if n_customers > 0 else 1
    n_slots = n_vehicles * max_steps
    n_qubits = n_slots * bits_per_slot

    return BinaryBlockConfig(
        n_customers=n_customers,
        n_vehicles=n_vehicles,
        max_steps=max_steps,
        bits_per_slot=bits_per_slot,
        n_slots=n_slots,
        n_qubits=n_qubits,
    )


def create_cvrp_hubo_binary(
    cost_matrix: npt.NDArray[np.floating],
    demands: npt.NDArray[np.floating],
    capacity: float,
    n_vehicles: int,
    depot: int = 0,
    constraint_penalty: float = 4.0,
    objective_weight: float = 1.0,
    capacity_penalty: float = 4.0,
    max_steps: int | None = None,
) -> tuple[dict[tuple[int, ...], float], BinaryBlockConfig]:
    """Generate a HUBO (Higher-Order Binary Optimization) for CVRP with binary encoding.

    Each (vehicle, timestep) slot is encoded as a ``bits_per_slot``-bit
    integer representing which customer (1..N) is assigned, or 0 for
    an empty slot.

    The variable ``x_{s,b}`` denotes bit *b* of slot *s*. The integer value
    of slot *s* is ``sum_b x_{s,b} * 2^b``.

    Constraints:
    - **Customer visit**: each customer j in 1..N appears exactly once
      across all slots.
    - **Capacity**: each vehicle's route total demand ≤ capacity.
    - **Objective**: minimise total travel distance.

    Note: This produces a HUBO (terms up to degree ``bits_per_slot``)
    which requires quadratization before use in standard QAOA.

    Args:
        cost_matrix: Distance matrix, shape ``(n_nodes, n_nodes)``.
        demands: Node demands, shape ``(n_nodes,)``. Depot demand = 0.
        capacity: Vehicle capacity.
        n_vehicles: Number of vehicles.
        depot: Depot node index.
        constraint_penalty: Customer-visit constraint penalty.
        objective_weight: Objective weight.
        capacity_penalty: Capacity penalty.
        max_steps: Max route steps per vehicle (default: n_customers).

    Returns:
        Tuple of (hubo_dict, config) where hubo_dict maps tuples of
        variable indices to coefficients.
    """
    n_nodes = cost_matrix.shape[0]
    customers = [c for c in range(n_nodes) if c != depot]
    n_cust = len(customers)

    config = binary_block_config(n_cust, n_vehicles, max_steps)
    B = config.bits_per_slot

    def slot_bits(slot_idx: int) -> list[int]:
        """Return qubit indices for a given slot."""
        start = slot_idx * B
        return list(range(start, start + B))

    def vehicle_slot(v: int, t: int) -> int:
        """Map (vehicle, timestep) to slot index."""
        return v * config.max_steps + t

    hubo: dict[tuple[int, ...], float] = {}

    def _add(key: tuple[int, ...], val: float) -> None:
        key = tuple(sorted(key))
        hubo[key] = hubo.get(key, 0.0) + val

    # Helper: build the "indicator" polynomial for slot s == customer j
    # In binary: slot value = sum_b x_{s,b} * 2^b
    # Indicator I(s = j) = product over bits b of:
    #   x_{s,b}     if bit b of j is 1
    #   (1 - x_{s,b}) if bit b of j is 0
    #
    # This is a multilinear polynomial of degree B in the slot's bits.
    # We expand it into a dict of {frozen_set_of_variables: coefficient}.

    def indicator_terms(
        slot_idx: int, customer_val: int
    ) -> dict[frozenset[int], float]:
        """Expand the indicator I(slot == customer_val) into polynomial terms.

        Returns dict mapping frozensets of variable indices to coefficients.
        """
        bits = slot_bits(slot_idx)
        # Start with constant 1
        terms: dict[frozenset[int], float] = {frozenset(): 1.0}

        for b in range(B):
            bit_is_set = (customer_val >> b) & 1
            new_terms: dict[frozenset[int], float] = {}

            for vars_set, coeff in terms.items():
                if bit_is_set:
                    # Multiply by x_{bits[b]}
                    new_vars = vars_set | {bits[b]}
                    new_terms[new_vars] = new_terms.get(new_vars, 0.0) + coeff
                else:
                    # Multiply by (1 - x_{bits[b]})
                    # = coeff * 1 - coeff * x_{bits[b]}
                    new_terms[vars_set] = new_terms.get(vars_set, 0.0) + coeff
                    new_vars = vars_set | {bits[b]}
                    new_terms[new_vars] = new_terms.get(new_vars, 0.0) - coeff

            terms = new_terms

        # Remove near-zero terms
        return {k: v for k, v in terms.items() if abs(v) > 1e-15}

    # --- Customer visit constraints ---
    # For each customer j: (sum_s I(s == j) - 1)^2
    # = sum_s I(s==j) * sum_s' I(s'==j) - 2 * sum_s I(s==j) + 1
    for cust_idx in range(n_cust):
        cust_val = cust_idx + 1  # binary value (0 is "empty")

        slot_indicators = []
        for v in range(n_vehicles):
            for t in range(config.max_steps):
                s = vehicle_slot(v, t)
                ind = indicator_terms(s, cust_val)
                slot_indicators.append(ind)

        # -2 * sum_s I(s==j)  (linear part of (sum-1)^2)
        for ind in slot_indicators:
            for vars_set, coeff in ind.items():
                _add(
                    tuple(sorted(vars_set)),
                    -2.0 * constraint_penalty * coeff,
                )

        # sum_s sum_{s'} I(s==j) * I(s'==j)  (quadratic part)
        # = sum_s I(s==j)^2 + 2 * sum_{s<s'} I(s==j)*I(s'==j)
        # Since I^2 = I (indicator is 0 or 1), the diagonal is just sum_s I(s==j)
        for ind in slot_indicators:
            for vars_set, coeff in ind.items():
                _add(tuple(sorted(vars_set)), constraint_penalty * coeff)

        # Cross terms: 2 * sum_{s<s'} I(s==j) * I(s'==j)
        for i in range(len(slot_indicators)):
            for j_idx in range(i + 1, len(slot_indicators)):
                for vars_i, coeff_i in slot_indicators[i].items():
                    for vars_j, coeff_j in slot_indicators[j_idx].items():
                        combined = vars_i | vars_j
                        _add(
                            tuple(sorted(combined)),
                            2.0 * constraint_penalty * coeff_i * coeff_j,
                        )

    # --- Objective: travel distance ---
    # For each vehicle v:
    #   depot -> slot(v,0) + slot(v,t) -> slot(v,t+1) + slot(v,last) -> depot
    for v in range(n_vehicles):
        ms = config.max_steps

        for t in range(ms):
            s = vehicle_slot(v, t)

            for cust_idx in range(n_cust):
                cust_val = cust_idx + 1
                cust_node = customers[cust_idx]
                ind = indicator_terms(s, cust_val)

                if t == 0:
                    # Depot to first customer
                    weight = cost_matrix[depot, cust_node]
                    for vars_set, coeff in ind.items():
                        _add(
                            tuple(sorted(vars_set)),
                            objective_weight * weight * coeff,
                        )

                if t == ms - 1:
                    # Last customer to depot
                    weight = cost_matrix[cust_node, depot]
                    for vars_set, coeff in ind.items():
                        _add(
                            tuple(sorted(vars_set)),
                            objective_weight * weight * coeff,
                        )

            # Consecutive: slot(v,t) -> slot(v,t+1)
            if t < ms - 1:
                s_next = vehicle_slot(v, t + 1)
                for ci in range(n_cust):
                    ci_val = ci + 1
                    ci_node = customers[ci]
                    ind_i = indicator_terms(s, ci_val)
                    for cj in range(n_cust):
                        cj_val = cj + 1
                        cj_node = customers[cj]
                        weight = cost_matrix[ci_node, cj_node]
                        if abs(weight) < 1e-15:
                            continue
                        ind_j = indicator_terms(s_next, cj_val)
                        for vars_i, coeff_i in ind_i.items():
                            for vars_j, coeff_j in ind_j.items():
                                combined = vars_i | vars_j
                                _add(
                                    tuple(sorted(combined)),
                                    objective_weight * weight * coeff_i * coeff_j,
                                )

    # --- Capacity constraints ---
    # For each vehicle v: (sum_{t,j} demand_j * I(slot(v,t)==j) - C)^2
    # Simplified: capacity_penalty * (sum - C)^2
    # We only add the quadratic expansion terms (linear + cross)
    for v in range(n_vehicles):
        # Collect all demand-weighted indicator terms for this vehicle
        demand_terms: list[tuple[dict[frozenset[int], float], float]] = []
        for t in range(config.max_steps):
            s = vehicle_slot(v, t)
            for ci in range(n_cust):
                ci_val = ci + 1
                d = demands[customers[ci]]
                if abs(d) < 1e-15:
                    continue
                ind = indicator_terms(s, ci_val)
                demand_terms.append((ind, d))

        # -2C * sum of demand_j * I(slot==j)
        for ind, d in demand_terms:
            for vars_set, coeff in ind.items():
                _add(
                    tuple(sorted(vars_set)),
                    capacity_penalty * d * (-2.0 * capacity) * coeff,
                )

        # (sum demand_j * I(slot==j))^2
        # = sum_i d_i^2 * I_i + 2 * sum_{i<j} d_i * d_j * I_i * I_j
        for i, (ind_i, d_i) in enumerate(demand_terms):
            # Diagonal: d_i^2 * I_i (since I_i^2 = I_i)
            for vars_set, coeff in ind_i.items():
                _add(
                    tuple(sorted(vars_set)),
                    capacity_penalty * d_i * d_i * coeff,
                )
            # Cross terms
            for j_idx in range(i + 1, len(demand_terms)):
                ind_j, d_j = demand_terms[j_idx]
                for vars_i, coeff_i in ind_i.items():
                    for vars_j, coeff_j in ind_j.items():
                        combined = vars_i | vars_j
                        _add(
                            tuple(sorted(combined)),
                            2.0 * capacity_penalty * d_i * d_j * coeff_i * coeff_j,
                        )

    return hubo, config


def build_binary_superposition_ops(
    config: BinaryBlockConfig,
    wires: Sequence[int],
) -> list[qml.operation.Operator]:
    """Prepare a uniform superposition over valid slot values.

    For each slot of ``bits_per_slot`` qubits, prepares a Hadamard
    superposition. This initialises each slot in a uniform superposition
    over all 2^B basis states. Values > n_customers represent "empty"
    or invalid states; the cost Hamiltonian penalises these.

    For the basic version, this is simply Hadamard on every qubit.
    (A more refined version could use amplitude encoding to restrict
    to values 0..n_customers only.)

    Args:
        config: Binary block configuration.
        wires: Ordered sequence of wire labels.

    Returns:
        List of PennyLane operations.
    """
    ops: list[qml.operation.Operator] = []
    for w in wires:
        ops.append(qml.Hadamard(wires=w))
    return ops


def build_binary_mixer_ops(
    beta: sympy.Expr | float,
    config: BinaryBlockConfig,
    wires: Sequence[int],
) -> list[qml.operation.Operator]:
    """Build a mixer for binary-encoded slots.

    Uses an X-mixer (RX rotation) on every qubit. This is the standard
    transverse-field mixer that explores the full binary space.

    For a more structured mixer that respects the valid-value subspace
    (0..n_customers), a Grover-style mixer could be used instead.

    Args:
        beta: Mixer angle parameter.
        config: Binary block configuration.
        wires: Ordered sequence of wire labels.

    Returns:
        List of PennyLane operations.
    """
    ops: list[qml.operation.Operator] = []
    for w in wires:
        ops.append(qml.RX(phi=2 * beta, wires=w))
    return ops


def decode_binary_cvrp_solution(
    bitstring: str,
    config: BinaryBlockConfig,
    depot: int = 0,
    n_nodes: int | None = None,
) -> list[list[int]] | None:
    """Decode a binary-encoded CVRP bitstring into vehicle routes.

    Args:
        bitstring: Binary string of length ``config.n_qubits``.
        config: Binary block configuration.
        depot: Depot node index.
        n_nodes: Total number of nodes.

    Returns:
        List of routes or None if decoding fails.
    """
    if n_nodes is None:
        n_nodes = config.n_customers + 1
    if len(bitstring) != config.n_qubits:
        return None

    B = config.bits_per_slot
    customers = [c for c in range(n_nodes) if c != depot]

    routes = []
    for v in range(config.n_vehicles):
        route = [depot]
        for t in range(config.max_steps):
            slot_idx = v * config.max_steps + t
            start = slot_idx * B
            slot_bits = bitstring[start : start + B]
            # Convert binary to integer (little-endian: bit 0 is LSB)
            val = sum(int(slot_bits[b]) * (2**b) for b in range(B))
            if 1 <= val <= config.n_customers:
                route.append(customers[val - 1])
        route.append(depot)
        routes.append(route)

    return routes


def is_valid_binary_cvrp(
    bitstring: str,
    config: BinaryBlockConfig,
    demands: npt.NDArray[np.floating],
    capacity: float,
    depot: int = 0,
) -> bool:
    """Check if a binary-encoded CVRP solution is feasible.

    Validates:
    - Each customer appears exactly once across all slots.
    - No slot has an out-of-range value (> n_customers).
    - Each vehicle's route demand ≤ capacity.

    Args:
        bitstring: Binary string.
        config: Binary block configuration.
        demands: Node demands.
        capacity: Vehicle capacity.
        depot: Depot index.

    Returns:
        True if feasible.
    """
    if len(bitstring) != config.n_qubits:
        return False

    B = config.bits_per_slot
    n_nodes = len(demands)
    customers = [c for c in range(n_nodes) if c != depot]

    # Track which customers are visited
    visited = set()
    vehicle_demands = [0.0] * config.n_vehicles

    for v in range(config.n_vehicles):
        for t in range(config.max_steps):
            slot_idx = v * config.max_steps + t
            start = slot_idx * B
            slot_bits = bitstring[start : start + B]
            val = sum(int(slot_bits[b]) * (2**b) for b in range(B))

            if val == 0:
                continue  # empty slot
            if val > config.n_customers:
                return False  # out of range

            cust_node = customers[val - 1]
            if val in visited:
                return False  # duplicate visit
            visited.add(val)
            vehicle_demands[v] += demands[cust_node]

    # All customers must be visited
    if len(visited) != config.n_customers:
        return False

    # Capacity check
    for v in range(config.n_vehicles):
        if vehicle_demands[v] > capacity + 1e-9:
            return False

    return True


# --- VRP file parser ---


@dataclass
class VRPInstance:
    """Parsed VRP/TSP instance data.

    Attributes:
        name: Instance name from the file header.
        comment: Instance comment (may contain optimal cost).
        problem_type: ``"CVRP"`` or ``"TSP"``.
        dimension: Total number of nodes (including depot).
        capacity: Vehicle capacity (CVRP only, 0 for TSP).
        n_vehicles: Number of vehicles (extracted from name ``-kN`` pattern
            or from comment, 1 for TSP).
        coords: Node coordinates, shape ``(dimension, 2)``.
        demands: Node demands, shape ``(dimension,)``. Depot demand is 0.
        depot: Depot node index (0-based).
        cost_matrix: Euclidean distance matrix, shape ``(dimension, dimension)``.
        optimal_cost: Optimal cost from comment, if available.
    """

    name: str = ""
    comment: str = ""
    problem_type: str = "CVRP"
    dimension: int = 0
    capacity: int = 0
    n_vehicles: int = 1
    coords: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    demands: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    depot: int = 0
    cost_matrix: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([]))
    optimal_cost: float | None = None

    @property
    def n_customers(self) -> int:
        """Number of customers (excluding depot)."""
        return self.dimension - 1


def parse_vrp_file(path: str | Path) -> VRPInstance:
    """Parse a TSPLIB/CVRPLIB format `.vrp` or `.tsp` file.

    Supports:
    - ``TYPE: CVRP`` and ``TYPE: TSP``
    - ``EDGE_WEIGHT_TYPE: EUC_2D`` (Euclidean 2D distances, rounded to int)
    - ``NODE_COORD_SECTION``, ``DEMAND_SECTION``, ``DEPOT_SECTION``

    Args:
        path: Path to the `.vrp` or `.tsp` file.

    Returns:
        Parsed :class:`VRPInstance`.

    Raises:
        ValueError: If the file format is unsupported or malformed.
    """
    path = Path(path)
    lines = path.read_text().splitlines()

    inst = VRPInstance()
    section = None
    coords_list: list[tuple[float, float]] = []
    demands_list: list[float] = []
    depot_nodes: list[int] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Header key-value pairs
        if ":" in line and section is None:
            key, _, value = line.partition(":")
            key = key.strip().upper()
            value = value.strip().strip('"')

            if key == "NAME":
                inst.name = value
                # Try to extract n_vehicles from name like "XSH-n20-k4-01"
                for part in value.split("-"):
                    if part.startswith("k") and part[1:].isdigit():
                        inst.n_vehicles = int(part[1:])
            elif key == "COMMENT":
                inst.comment = value
                # Try to extract optimal cost
                lower = value.lower()
                for prefix in [
                    "optimal cost:",
                    "optimal cost :",
                    "optimal:",
                ]:
                    if prefix in lower:
                        cost_str = lower.split(prefix)[1].strip().rstrip('"')
                        try:
                            inst.optimal_cost = float(cost_str)
                        except ValueError:
                            pass
            elif key == "TYPE":
                inst.problem_type = value.upper()
            elif key == "DIMENSION":
                inst.dimension = int(value)
            elif key == "CAPACITY":
                inst.capacity = int(value)
            elif key == "EDGE_WEIGHT_TYPE":
                if value.upper() != "EUC_2D":
                    raise ValueError(
                        f"Unsupported EDGE_WEIGHT_TYPE: {value}. "
                        f"Only EUC_2D is supported."
                    )
            continue

        # Section headers
        upper = line.upper()
        if upper == "NODE_COORD_SECTION":
            section = "COORDS"
            continue
        elif upper == "DEMAND_SECTION":
            section = "DEMANDS"
            continue
        elif upper == "DEPOT_SECTION":
            section = "DEPOT"
            continue
        elif upper == "EOF":
            break

        # Section data
        if section == "COORDS":
            parts = line.split()
            if len(parts) >= 3:
                coords_list.append((float(parts[1]), float(parts[2])))
        elif section == "DEMANDS":
            parts = line.split()
            if len(parts) >= 2:
                demands_list.append(float(parts[1]))
        elif section == "DEPOT":
            val = int(line)
            if val == -1:
                section = None
            else:
                depot_nodes.append(val)

    # Build arrays
    if coords_list:
        inst.coords = np.array(coords_list, dtype=np.float64)
    if demands_list:
        inst.demands = np.array(demands_list, dtype=np.float64)
    if depot_nodes:
        # TSPLIB uses 1-based indexing
        inst.depot = depot_nodes[0] - 1

    # Compute Euclidean distance matrix (TSPLIB rounds to nearest int)
    if len(coords_list) > 0:
        n = len(coords_list)
        cost = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                dx = coords_list[i][0] - coords_list[j][0]
                dy = coords_list[i][1] - coords_list[j][1]
                # TSPLIB EUC_2D: nint(sqrt(dx^2 + dy^2))
                d = round(math.sqrt(dx * dx + dy * dy))
                cost[i, j] = d
                cost[j, i] = d
        inst.cost_matrix = cost

    # Default demands for TSP
    if inst.problem_type == "TSP" and len(demands_list) == 0:
        inst.demands = np.zeros(inst.dimension, dtype=np.float64)

    return inst


def parse_vrp_solution(
    path: str | Path,
) -> tuple[list[list[int]], float]:
    """Parse a CVRPLIB solution file.

    Expects format::

        Route #1: 15 2 18 1 8
        Route #2: 10 11 7 9 6
        ...
        Cost 646

    Customer IDs are 1-based (TSPLIB convention). The returned routes
    use 0-based indexing with the depot prepended and appended.

    Args:
        path: Path to the `.opt.sol` or `.bst.sol` file.

    Returns:
        Tuple of (routes, cost) where routes is a list of routes
        and each route is ``[depot, c1, c2, ..., depot]`` with 0-based indices.
        The depot index is assumed to be 0.
    """
    path = Path(path)
    lines = path.read_text().splitlines()

    routes: list[list[int]] = []
    cost = 0.0

    for line in lines:
        line = line.strip()
        if line.startswith("Route"):
            # "Route #1: 15 2 18 1 8" -> customers [15, 2, 18, 1, 8] (1-based)
            _, _, customer_str = line.partition(":")
            customers_1based = [int(x) for x in customer_str.split()]
            # Convert to 0-based
            route = [0] + [c - 1 for c in customers_1based] + [0]
            routes.append(route)
        elif line.startswith("Cost"):
            cost = float(line.split()[1])

    return routes, cost


# --- Routing problem base ---


class _RoutingProblemBase(Problem):
    """Shared base for routing problems.

    Stores the :class:`IsingResult` and mixer after subclass constructors
    call :meth:`_init_ising`.
    """

    def _init_ising(
        self,
        qubo,
        *,
        block_size: int,
        n_blocks: int,
        hamiltonian_builder: str = "native",
        use_xy_mixer: bool = True,
    ) -> None:
        """Shared Ising conversion + mixer setup."""
        self._block_size = block_size
        self._n_blocks = n_blocks
        self._ising = qubo_to_ising(qubo, hamiltonian_builder=hamiltonian_builder)

        if use_xy_mixer:
            graph = build_block_xy_mixer_graph(
                block_size, n_blocks, range(self._ising.n_qubits)
            )
            self._mixer_hamiltonian = pqaoa.xy_mixer(graph)
        else:
            self._mixer_hamiltonian = pqaoa.x_mixer(range(self._ising.n_qubits))

    @property
    def cost_hamiltonian(self) -> qml.operation.Operator:
        return self._ising.cost_hamiltonian

    @property
    def mixer_hamiltonian(self) -> qml.operation.Operator:
        return self._mixer_hamiltonian

    @property
    def loss_constant(self) -> float:
        return self._ising.loss_constant


# --- TSPProblem ---


class TSPProblem(_RoutingProblemBase):
    """Traveling Salesman Problem for QAOA.

    Generates a QUBO from the cost matrix, converts to an Ising
    Hamiltonian, and uses block W-state initialisation with an XY mixer
    that preserves the one-hot constraint within each time-step block.

    Args:
        cost_matrix: Symmetric distance/cost matrix of shape ``(n, n)``.
        start_city: Index of the fixed start city. Defaults to 0.
        constraint_penalty: Constraint penalty strength. Defaults to 4.0.
        objective_weight: Objective weight. Defaults to 1.0.
    """

    def __init__(
        self,
        cost_matrix: npt.NDArray[np.floating],
        *,
        start_city: int = 0,
        constraint_penalty: float = 4.0,
        objective_weight: float = 1.0,
    ):
        self._cost_matrix = np.asarray(cost_matrix, dtype=np.float64)
        self._start_city = start_city
        self._n_cities = self._cost_matrix.shape[0]
        m = self._n_cities - 1

        qubo = create_tsp_qubo(
            self._cost_matrix, start_city, constraint_penalty, objective_weight
        )
        self._init_ising(qubo, block_size=m, n_blocks=m)

    @property
    def recommended_initial_state(self) -> InitialState:
        return WState(self._block_size, self._n_blocks)

    @property
    def decode_fn(self) -> Callable[[str], Any]:
        n_cities = self._n_cities
        start = self._start_city
        return lambda bs: decode_tsp_solution(bs, n_cities, start)

    @property
    def feasible_dimension(self) -> int:
        """Size of the feasible subspace: (n-1)!"""
        m = self._n_cities - 1
        result = 1
        for i in range(1, m + 1):
            result *= i
        return result

    def is_feasible(self, bitstring: str) -> bool:
        return is_valid_tsp_tour(bitstring, self._n_cities)

    def repair(self, bitstring: str) -> tuple[str, Any, float]:
        return repair_tsp_solution(
            bitstring, self._n_cities, self._start_city, self._cost_matrix
        )

    def compute_energy(self, bitstring: str) -> float | None:
        t = decode_tsp_solution(bitstring, self._n_cities, self._start_city)
        return tour_cost(t, self._cost_matrix) if t is not None else None


# --- CVRPProblem ---


class CVRPProblem(_RoutingProblemBase):
    """Capacitated Vehicle Routing Problem for QAOA.

    Supports two encodings:

    * ``"one_hot"`` — block one-hot with W-state + XY mixer.
    * ``"binary"`` — compact binary (HUBO) with Hadamard + X mixer,
      reducing qubit count from O(N) to O(log N) per slot.

    Args:
        cost_matrix: Symmetric distance/cost matrix of shape ``(n, n)``.
        demands: Customer demands (length = n_nodes, depot demand = 0).
        capacity: Vehicle capacity.
        n_vehicles: Number of vehicles.
        depot: Index of the depot node. Defaults to 0.
        encoding: ``"one_hot"`` or ``"binary"``. Defaults to ``"one_hot"``.
        constraint_penalty: Constraint penalty strength. Defaults to 4.0.
        objective_weight: Objective weight. Defaults to 1.0.
        capacity_penalty: Capacity penalty strength. Defaults to 4.0.
    """

    def __init__(
        self,
        cost_matrix: npt.NDArray[np.floating],
        *,
        demands: npt.NDArray[np.floating],
        capacity: float,
        n_vehicles: int,
        depot: int = 0,
        encoding: Literal["one_hot", "binary"] = "one_hot",
        constraint_penalty: float = 4.0,
        objective_weight: float = 1.0,
        capacity_penalty: float = 4.0,
    ):
        self._cost_matrix = np.asarray(cost_matrix, dtype=np.float64)
        self._demands = demands
        self._capacity = capacity
        self._n_vehicles = n_vehicles
        self._depot = depot
        self._encoding = encoding
        self._n_cities = self._cost_matrix.shape[0]
        self._binary_config: BinaryBlockConfig | None = None

        n_cust = self._n_cities - 1

        if encoding == "binary":
            hubo, self._binary_config = create_cvrp_hubo_binary(
                self._cost_matrix,
                demands=demands,
                capacity=capacity,
                n_vehicles=n_vehicles,
                depot=depot,
                constraint_penalty=constraint_penalty,
                objective_weight=objective_weight,
                capacity_penalty=capacity_penalty,
            )
            self._init_ising(
                hubo,
                block_size=self._binary_config.bits_per_slot,
                n_blocks=self._binary_config.n_slots,
                hamiltonian_builder="quadratized",
                use_xy_mixer=False,
            )
        else:
            qubo = create_cvrp_qubo(
                self._cost_matrix,
                demands=demands,
                capacity=capacity,
                n_vehicles=n_vehicles,
                depot=depot,
                constraint_penalty=constraint_penalty,
                objective_weight=objective_weight,
                capacity_penalty=capacity_penalty,
            )
            bs, nb = cvrp_block_structure(n_cust, n_vehicles)
            self._init_ising(qubo, block_size=bs, n_blocks=nb)

    @property
    def recommended_initial_state(self) -> InitialState:
        if self._encoding == "binary":
            return SuperpositionState()
        return WState(self._block_size, self._n_blocks)

    @property
    def decode_fn(self) -> Callable[[str], Any]:
        if self._encoding == "binary" and self._binary_config is not None:
            cfg = self._binary_config
            dep, nn = self._depot, self._n_cities
            return lambda bs: decode_binary_cvrp_solution(bs, cfg, dep, nn)
        nc, nv, dep, nn = (
            self._n_cities - 1,
            self._n_vehicles,
            self._depot,
            self._n_cities,
        )
        return lambda bs: decode_cvrp_solution(bs, nc, nv, dep, nn)

    @property
    def encoding(self) -> str:
        return self._encoding

    @property
    def binary_config(self) -> BinaryBlockConfig | None:
        return self._binary_config

    def is_feasible(self, bitstring: str) -> bool:
        if self._encoding == "binary" and self._binary_config is not None:
            return is_valid_binary_cvrp(
                bitstring,
                self._binary_config,
                self._demands,
                self._capacity,
                self._depot,
            )
        return is_valid_cvrp_solution(
            bitstring,
            n_customers=self._n_cities - 1,
            n_vehicles=self._n_vehicles,
            demands=self._demands,
            capacity=self._capacity,
            depot=self._depot,
        )

    def repair(self, bitstring: str) -> tuple[str, Any, float]:
        return repair_cvrp_solution(
            bitstring,
            n_customers=self._n_cities - 1,
            n_vehicles=self._n_vehicles,
            cost_matrix=self._cost_matrix,
            demands=self._demands,
            capacity=self._capacity,
            depot=self._depot,
        )

    def compute_energy(self, bitstring: str) -> float | None:
        routes = decode_cvrp_solution(
            bitstring,
            self._n_cities - 1,
            self._n_vehicles,
            self._depot,
            self._n_cities,
        )
        if routes is None:
            return None
        return sum(tour_cost(route, self._cost_matrix) for route in routes)
