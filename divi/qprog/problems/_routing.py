# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Routing problem classes and utilities for QAOA (TSP, CVRP)."""

import math
import numbers
import re
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache, cached_property
from heapq import heappop, heappush
from itertools import combinations, pairwise, product
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import linear_sum_assignment

from divi.hamiltonians import IsingResult, qubo_to_ising, x_mixer, xy_mixer
from divi.qprog.initial_states import (
    InitialState,
    SuperpositionState,
    WState,
    build_block_xy_mixer_graph,
)

from ._base import QAOAProblem
from ._binary import BinaryOptimizationProblem
from ._constraints import LinearConstraint

# --- TSP utilities ---


def create_tsp_qubo(
    cost_matrix: npt.NDArray[np.floating],
    start_city: int = 0,
    penalty_weight: float = 4.0,
    objective_weight: float = 1.0,
    reduced: bool = True,
) -> npt.NDArray[np.floating]:
    """Generate a QUBO for TSP with a fixed start city.

    Uses the standard node-ordering (assignment) encoding where binary
    variable ``x_{i,t}`` indicates that city *i* is visited at time step *t*.
    The start city is fixed, reducing the problem from *n²* to *(n-1)²* qubits.

    The penalty that each time step holds exactly one city is always
    included. By default the **reduced phase operator** from CE-QAOA
    (`arXiv:2511.14296 <https://arxiv.org/abs/2511.14296>`_, Eq. 13) is
    used: the penalty that each city takes exactly one time step is omitted,
    because the W-state initialisation and XY mixer, which act on the blocks of
    qubits belonging to one city, keep it satisfied.

    Penalty constants are dropped: a valid tour has energy
    ``objective_weight * tour_cost - n_penalties * penalty_weight * (n - 1)``,
    with ``n_penalties`` equal to 1 if ``reduced`` else 2.

    Args:
        cost_matrix: Symmetric (n × n) distance/cost matrix.
        start_city: Index of the fixed start/end city. Defaults to 0.
        penalty_weight: Penalty strength for constraint violations.
        objective_weight: Weight for the objective function.
        reduced: If ``True`` (default), omit the per-city penalties. Only valid
            for states that keep every city in exactly one time step, as with
            CE-QAOA's W-state initialisation and XY mixer. If ``False``, the
            per-city penalties are included as well, so the Hamiltonian alone
            encodes feasibility under a standard QAOA ansatz (Hadamard
            initialisation, X mixer).

    Returns:
        Upper-triangular QUBO matrix of shape ``(m², m²)`` where
        ``m = n - 1``, compatible with ``QUBOProblemTypes``.

    Raises:
        ValueError: If cost_matrix is not square or start_city is out of range.
    """
    n = _check_routing_inputs(cost_matrix)
    if not (0 <= start_city < n):
        raise ValueError(f"start_city {start_city} out of range [0, {n}).")

    m = n - 1  # reduced dimension (cities excluding start)
    cities = [c for c in range(n) if c != start_city]

    # Variable ordering: x_{i,t} -> qubit index i*m + t
    I_m = np.eye(m)
    J_m = np.ones((m, m))

    # Per-step penalties: A * sum_t (1 - sum_i x_{i,t})^2
    Q = penalty_weight * np.kron(J_m - I_m, I_m)

    if not reduced:
        # Per-city penalties: A * sum_i (1 - sum_t x_{i,t})^2. Omitted when reduced:
        # the W-state + XY mixer blocks (one per city) enforce them.
        Q += penalty_weight * np.kron(I_m, J_m - I_m)

    # Linear term: each variable appears in the per-step penalty (and per-city if not reduced)
    n_penalties = 1 if reduced else 2
    np.fill_diagonal(Q, np.diag(Q) - n_penalties * penalty_weight)

    # --- Objective: consecutive timeslot costs ---
    W_reduced = cost_matrix[np.ix_(cities, cities)]
    S = np.eye(m, k=1)
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


def compute_tour_cost(
    tour: Sequence[int],
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
    for src, dst in pairwise(tour):
        total += cost_matrix[src, dst]
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

    cost = compute_tour_cost(tour, cost_matrix)

    return repaired_bitstring, tour, cost


# --- CVRP utilities (one-hot) ---


def _check_routing_inputs(
    cost_matrix: npt.NDArray[np.floating],
    demands: npt.NDArray[np.floating] | None = None,
) -> int:
    """Return the node count after checking the cost-matrix shape and demand length."""
    n_nodes = cost_matrix.shape[0]
    if cost_matrix.shape != (n_nodes, n_nodes):
        raise ValueError(f"cost_matrix must be square, got shape {cost_matrix.shape}.")
    if demands is not None and len(demands) != n_nodes:
        raise ValueError(
            f"demands length ({len(demands)}) must match cost_matrix size ({n_nodes})."
        )
    return n_nodes


def _check_encoding(encoding: str) -> None:
    if encoding not in ("one_hot", "binary"):
        raise ValueError(f"encoding must be 'one_hot' or 'binary', got {encoding!r}.")


def _check_integer(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}.")


def _check_node_index(name: str, value: Any, n_nodes: int) -> None:
    _check_integer(name, value)
    if not (0 <= value < n_nodes):
        raise ValueError(f"{name} {value} out of range [0, {n_nodes}).")


def _check_cvrp_inputs(
    cost_matrix: npt.NDArray[np.float64],
    demands: npt.NDArray[np.float64],
    capacity: float,
    n_vehicles: int,
    depot: int,
    encoding: str,
) -> None:
    if demands.ndim != 1:
        raise ValueError(f"demands must be one-dimensional, got shape {demands.shape}.")
    n_nodes = _check_routing_inputs(cost_matrix, demands)
    if n_nodes < 2:
        raise ValueError("cost_matrix must cover the depot and at least one customer.")
    if not np.isfinite(cost_matrix).all():
        raise ValueError("cost_matrix must be finite.")
    _check_encoding(encoding)
    _check_node_index("depot", depot, n_nodes)
    _check_integer("n_vehicles", n_vehicles)
    if n_vehicles < 1:
        raise ValueError(f"n_vehicles must be at least 1, got {n_vehicles}.")
    if not (math.isfinite(capacity) and capacity > 0):
        raise ValueError(f"capacity must be finite and positive, got {capacity}.")
    if not np.isfinite(demands).all() or (demands < 0).any():
        raise ValueError("demands must be finite and non-negative.")
    if demands[depot] != 0:
        raise ValueError(f"demand at the depot must be zero, got {demands[depot]}.")
    if demands.max() > capacity + 1e-9:
        raise ValueError(
            f"every customer demand must be at most capacity {capacity}, "
            f"got {demands.max()}."
        )


def create_cvrp_qubo(
    cost_matrix: npt.NDArray[np.floating],
    n_vehicles: int,
    depot: int = 0,
    penalty_weight: float = 4.0,
    objective_weight: float = 1.0,
) -> npt.NDArray[np.floating]:
    """Generate a QUBO for CVRP with a fixed depot.

    Uses one-hot block encoding where ``x_{i,v,t}`` indicates that customer *i*
    is served by vehicle *v* at step *t*. The depot is fixed and not encoded.

    Enforced by the W-state initialisation and XY mixer, which act on each
    customer's block of ``n_vehicles × max_steps`` qubits (reduced phase
    operator, as in :func:`create_tsp_qubo`):
    - Each customer is visited exactly once across all vehicles and steps.

    Penalties in the phase operator:
    - **Slot occupancy** (penalty_weight): penalises two customers in one
      (vehicle, step) slot.
    - **Compactness** (penalty_weight): penalises an occupied step that follows
      an empty one.
    - **Objective** (objective_weight): minimise total travel distance. The
      return edge is charged at the last occupied step.

    Variable ordering: ``x_{i,v,t}`` → qubit ``i * V * T + v * T + t``
    where K = n_customers, V = n_vehicles, T = max_steps = K.

    Capacity is not penalised; it is enforced when decoding, filtering and
    repairing solutions.

    The QUBO matrix is built via Kronecker products and vectorised
    numpy operations.

    Args:
        cost_matrix: Distance matrix of shape ``(n_nodes, n_nodes)``.
        n_vehicles: Number of vehicles.
        depot: Index of the depot node. Defaults to 0.
        penalty_weight: Penalty for slot-occupancy and compactness violations.
        objective_weight: Weight for the objective function.

    Returns:
        Upper-triangular QUBO matrix compatible with ``QUBOProblemTypes``.

    Raises:
        ValueError: If ``cost_matrix`` is not square.
    """
    n_nodes = _check_routing_inputs(cost_matrix)

    customers = [c for c in range(n_nodes) if c != depot]
    K = len(customers)  # n_customers
    V = n_vehicles
    T = K  # max_steps = n_customers (worst case)

    # Variable ordering: x_{i,v,t} -> qubit i*V*T + v*T + t
    I_K = np.eye(K)
    J_K = np.ones((K, K))

    # Slot occupancy: A * s * (s - 1) with s = sum_i x_{i,v,t}, no linear term.
    Q = penalty_weight * np.kron(J_K - I_K, np.eye(V * T))

    # Objective: x_{i,v,t} * x_{j,v,t+1} costs W[i,j]; the return edge
    # c_i * x_{i,v,t} * (1 - s_{v,t+1}) is charged at the last occupied step.
    W = cost_matrix[np.ix_(customers, customers)]
    to_depot = cost_matrix[customers, depot]
    from_depot = cost_matrix[depot, customers]
    S = np.eye(T, k=1)
    # Compactness: B * sum_t s_{v,t+1} * (1 - s_{v,t}) keeps routes in steps 0..len-1.
    step_coupling = (
        objective_weight * (W - np.outer(to_depot, np.ones(K))) - penalty_weight * J_K
    )
    Q += np.kron(step_coupling, np.kron(np.eye(V), S))
    linear = objective_weight * np.repeat(to_depot, V * T)
    linear += penalty_weight * np.tile(np.arange(T) >= 1, K * V)
    linear[::T] += objective_weight * np.repeat(from_depot, V)
    np.fill_diagonal(Q, np.diag(Q) + linear)

    # Return upper-triangular matrix
    return np.triu(Q + Q.T - np.diag(np.diag(Q)))


def build_cvrp_block_structure(
    n_customers: int,
    n_vehicles: int,
) -> tuple[int, int]:
    """Return the block structure for CVRP CE-QAOA.

    Args:
        n_customers: Number of customers (excluding depot).
        n_vehicles: Number of vehicles.

    Returns:
        tuple[int, int]: Pair ``(block_size, n_blocks)`` where
        ``block_size = n_vehicles * n_customers`` (one qubit per vehicle × time
        step slot) and ``n_blocks = n_customers`` (one block per customer).
    """
    max_steps = n_customers
    return n_vehicles * max_steps, n_customers


def _reshape_cvrp_bits(
    bitstring: str, n_customers: int, n_vehicles: int
) -> npt.NDArray[np.int_]:
    """Bits as an array indexed ``[vehicle, step, customer]``."""
    bits = np.array([int(b) for b in bitstring])
    return bits.reshape(n_customers, n_vehicles, n_customers).transpose(1, 2, 0)


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
    - Each (vehicle, step) slot holds at most one customer.
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
    bits = _reshape_cvrp_bits(bitstring, n_customers, n_vehicles)

    if not np.all(bits.sum(axis=(0, 1)) == 1) or np.any(bits.sum(axis=2) > 1):
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
    bits = _reshape_cvrp_bits(bitstring, n_customers, n_vehicles)

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


_MAX_EXACT_REPAIR_CUSTOMERS = 10


def _plan_cvrp_vehicle_routes(
    match: npt.NDArray[np.floating],
    from_depot: npt.NDArray[np.floating],
    between: npt.NDArray[np.floating],
    to_depot: npt.NDArray[np.floating],
    empty_cost: float,
) -> dict[int, tuple[tuple[int, float], tuple[int, ...]]]:
    """Best ``((-matches, cost), order)`` for every customer subset on one vehicle.

    ``match[t, i]`` is 1 when customer ``i`` at step ``t`` agrees with the
    input bits. Dynamic programming over subsets and last customers.
    """
    n = len(from_depot)
    table: dict[tuple[int, int], tuple[int, float, int]] = {}
    best: dict[int, tuple[tuple[int, float], tuple[int, ...]]] = {
        0: ((0, empty_cost), ())
    }
    for mask in range(1, 1 << n):
        step = mask.bit_count() - 1
        finals = []
        for j in range(n):
            if not mask >> j & 1:
                continue
            prev = mask ^ (1 << j)
            gain = -int(match[step, j])
            if prev == 0:
                entry = (gain, float(from_depot[j]), -1)
            else:
                entry = min(
                    (
                        table[prev, i][0] + gain,
                        table[prev, i][1] + float(between[i, j]),
                        i,
                    )
                    for i in range(n)
                    if prev >> i & 1
                )
            table[mask, j] = entry
            finals.append((entry[0], entry[1] + float(to_depot[j]), j))
        neg_matches, cost, last = min(finals)
        order = []
        covered, j = mask, last
        while j >= 0:
            order.append(j)
            covered, j = covered ^ (1 << j), table[covered, j][2]
        best[mask] = ((neg_matches, cost), tuple(reversed(order)))
    return best


def _search_feasible_cvrp_routes(
    bits: npt.NDArray[np.floating],
    cost_matrix: npt.NDArray[np.floating],
    customer_demands: npt.NDArray[np.floating],
    capacity: float,
    customers: Sequence[int],
    n_vehicles: int,
    depot: int,
) -> npt.NDArray[np.int_] | None:
    """Capacity-feasible compact assignment as a ``[vehicle, step, customer]`` array.

    Picks, over every split of the customers into vehicle loads within
    ``capacity``, the assignment that agrees with the most input ``bits``, then
    has the lowest route cost. Returns ``None`` when no split fits.
    """
    n = len(customers)
    from_depot = cost_matrix[depot, customers]
    between = cost_matrix[np.ix_(customers, customers)]
    to_depot = cost_matrix[customers, depot]
    loads = [0.0] * (1 << n)
    for mask in range(1, 1 << n):
        low = mask & -mask
        loads[mask] = loads[mask ^ low] + float(customer_demands[low.bit_length() - 1])

    full = (1 << n) - 1
    reached: dict[int, tuple[int, float]] = {0: (0, 0.0)}
    stages = []
    for v in range(n_vehicles):
        plans = _plan_cvrp_vehicle_routes(
            bits[v * n : (v + 1) * n],
            from_depot,
            between,
            to_depot,
            float(cost_matrix[depot, depot]),
        )
        extended: dict[int, tuple[int, float]] = {}
        picks: dict[int, int] = {}
        for covered, (neg_matches, cost) in reached.items():
            rest = full ^ covered
            sub = rest
            while True:
                if loads[sub] <= capacity + 1e-9:
                    (sub_matches, sub_cost), _ = plans[sub]
                    total = covered | sub
                    candidate = (neg_matches + sub_matches, cost + sub_cost)
                    if total not in extended or candidate < extended[total]:
                        extended[total] = candidate
                        picks[total] = sub
                if sub == 0:
                    break
                sub = (sub - 1) & rest
        reached = extended
        stages.append((picks, plans))
    if full not in reached:
        return None

    packed = np.zeros((n_vehicles, n, n), dtype=int)
    covered = full
    for v in reversed(range(n_vehicles)):
        picks, plans = stages[v]
        sub = picks[covered]
        for step, customer in enumerate(plans[sub][1]):
            packed[v, step, customer] = 1
        covered ^= sub
    return packed


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

    1. Parse the soft assignment matrix.
    2. Use the Hungarian algorithm on a cost matrix derived from the
       soft assignments to find the best customer-to-(vehicle, timestep) mapping.
    3. Verify capacity constraints; reassign overflows greedily.
    4. If a vehicle is still over capacity and there are at most
       ``_MAX_EXACT_REPAIR_CUSTOMERS`` customers, search every split of the
       customers into vehicle loads within capacity. Among the compact
       assignments of those splits, the one agreeing with the most input bits
       wins, then the lowest route cost. When no split fits, the result of
       step 3 is returned.

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

    bits = (
        np.array([int(b) for b in bitstring], dtype=np.float64)
        .reshape(n_customers, n_vehicles * max_steps)
        .T
    )

    # Use Hungarian algorithm to find best assignment of customers
    # to (vehicle, timestep) slots
    # Cost matrix: 1 - soft_assignment (to maximise overlap)
    row_ind, col_ind = linear_sum_assignment(1.0 - bits)

    # Build repaired assignment
    repaired = np.zeros((n_vehicles, max_steps, n_customers), dtype=int)
    repaired.reshape(n_vehicles * max_steps, n_customers)[row_ind, col_ind] = 1

    customer_demands = np.asarray(demands)[customers]
    vehicle_loads = np.einsum("vtc,c->v", repaired, customer_demands)
    empty_slots = [
        [int(slot) for slot in np.flatnonzero(repaired[v].sum(axis=1) == 0)]
        for v in range(n_vehicles)
    ]

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
                    vehicle_loads[v] -= cust_demand
                    placed = False
                    for v2 in range(n_vehicles):
                        if v2 == v:
                            continue
                        if (
                            empty_slots[v2]
                            and vehicle_loads[v2] + cust_demand <= capacity + 1e-9
                        ):
                            t2 = heappop(empty_slots[v2])
                            repaired[v2, t2, cust_idx] = 1
                            vehicle_loads[v2] += cust_demand
                            heappush(empty_slots[v], t)
                            placed = True
                            break
                    if not placed:
                        # Last resort: put it back
                        repaired[v, t, cust_idx] = 1
                        vehicle_loads[v] += cust_demand
                        vehicle_demand += cust_demand
                else:
                    vehicle_demand += cust_demand

    if (
        0 < n_customers <= _MAX_EXACT_REPAIR_CUSTOMERS
        and np.einsum("vtc,c->v", repaired, customer_demands).max() > capacity + 1e-9
    ):
        packed = _search_feasible_cvrp_routes(
            bits,
            cost_matrix,
            customer_demands,
            capacity,
            customers,
            n_vehicles,
            depot,
        )
        if packed is not None:
            repaired = packed

    repaired_bitstring = "".join(str(x) for x in repaired.transpose(2, 0, 1).flatten())
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


def compute_binary_block_config(
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
        :class:`~divi.qprog.problems.BinaryBlockConfig` with the encoding parameters.
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
    penalty_weight: float = 4.0,
    objective_weight: float = 1.0,
    capacity_penalty_weight: float = 4.0,
    max_steps: int | None = None,
) -> tuple[dict[tuple[int, ...], float], BinaryBlockConfig]:
    """Generate a HUBO (Higher-Order Binary Optimisation) for CVRP with binary encoding.

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
        penalty_weight: Customer-visit constraint penalty.
        objective_weight: Objective weight.
        capacity_penalty_weight: Capacity penalty.
        max_steps: Max route steps per vehicle (default: n_customers).

    Returns:
        Tuple of (hubo_dict, config) where hubo_dict maps tuples of
        variable indices to coefficients.

    Raises:
        ValueError: If inputs are inconsistent.
    """
    n_nodes = _check_routing_inputs(cost_matrix, demands)
    customers = [c for c in range(n_nodes) if c != depot]
    n_cust = len(customers)

    config = compute_binary_block_config(n_cust, n_vehicles, max_steps)
    B = config.bits_per_slot

    def slot_bits(slot_idx: int) -> list[int]:
        """Return qubit indices for a given slot."""
        start = slot_idx * B
        return list(range(start, start + B))

    def compute_vehicle_slot(v: int, t: int) -> int:
        """Map (vehicle, timestep) to slot index."""
        return v * config.max_steps + t

    hubo: dict[tuple[int, ...], float] = {}

    def _add(key: tuple[int, ...], val: float) -> None:
        hubo[key] = hubo.get(key, 0.0) + val

    # Helper: build the "indicator" polynomial for slot s == customer j
    # In binary: slot value = sum_b x_{s,b} * 2^b
    # Indicator I(s = j) = product over bits b of:
    #   x_{s,b}     if bit b of j is 1
    #   (1 - x_{s,b}) if bit b of j is 0
    #
    # This is a multilinear polynomial of degree B in the slot's bits.
    # We expand it into a dict of {frozen_set_of_variables: coefficient}.

    @cache
    def expand_indicator_terms(
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

        slot_indicators = [
            expand_indicator_terms(s, cust_val) for s in range(config.n_slots)
        ]

        # -2 * sum_s I(s==j)  (linear part of (sum-1)^2)
        for ind in slot_indicators:
            for vars_set, coeff in ind.items():
                _add(
                    tuple(sorted(vars_set)),
                    -2.0 * penalty_weight * coeff,
                )

        # sum_s sum_{s'} I(s==j) * I(s'==j)  (quadratic part)
        # = sum_s I(s==j)^2 + 2 * sum_{s<s'} I(s==j)*I(s'==j)
        # Since I^2 = I (indicator is 0 or 1), the diagonal is just sum_s I(s==j)
        for ind in slot_indicators:
            for vars_set, coeff in ind.items():
                _add(tuple(sorted(vars_set)), penalty_weight * coeff)

        # Cross terms: 2 * sum_{s<s'} I(s==j) * I(s'==j)
        for ind_a, ind_b in combinations(slot_indicators, 2):
            for (vars_i, coeff_i), (vars_j, coeff_j) in product(
                ind_a.items(), ind_b.items()
            ):
                combined = vars_i | vars_j
                _add(
                    tuple(sorted(combined)),
                    2.0 * penalty_weight * coeff_i * coeff_j,
                )

    # --- Objective: travel distance ---
    # For each vehicle v:
    #   depot -> slot(v,0) + slot(v,t) -> slot(v,t+1) + slot(v,last) -> depot
    for v in range(n_vehicles):
        ms = config.max_steps

        for t in range(ms):
            s = compute_vehicle_slot(v, t)

            for cust_idx in range(n_cust):
                cust_val = cust_idx + 1
                cust_node = customers[cust_idx]
                ind = expand_indicator_terms(s, cust_val)

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
                s_next = compute_vehicle_slot(v, t + 1)
                for ci in range(n_cust):
                    ci_val = ci + 1
                    ci_node = customers[ci]
                    ind_i = expand_indicator_terms(s, ci_val)
                    for cj in range(n_cust):
                        cj_val = cj + 1
                        cj_node = customers[cj]
                        weight = cost_matrix[ci_node, cj_node]
                        if abs(weight) < 1e-15:
                            continue
                        ind_j = expand_indicator_terms(s_next, cj_val)
                        for (vars_i, coeff_i), (vars_j, coeff_j) in product(
                            ind_i.items(), ind_j.items()
                        ):
                            combined = vars_i | vars_j
                            _add(
                                tuple(sorted(combined)),
                                objective_weight * weight * coeff_i * coeff_j,
                            )

    # --- Capacity constraints ---
    # For each vehicle v: (sum_{t,j} demand_j * I(slot(v,t)==j) - C)^2
    # Simplified: capacity_penalty_weight * (sum - C)^2
    # We only add the quadratic expansion terms (linear + cross)
    for v in range(n_vehicles):
        # Collect all demand-weighted indicator terms for this vehicle
        demand_terms: list[tuple[dict[frozenset[int], float], float]] = []
        for t in range(config.max_steps):
            s = compute_vehicle_slot(v, t)
            for ci in range(n_cust):
                ci_val = ci + 1
                d = demands[customers[ci]]
                if abs(d) < 1e-15:
                    continue
                ind = expand_indicator_terms(s, ci_val)
                demand_terms.append((ind, d))

        # -2C * sum of demand_j * I(slot==j)
        for ind, d in demand_terms:
            for vars_set, coeff in ind.items():
                _add(
                    tuple(sorted(vars_set)),
                    capacity_penalty_weight * d * (-2.0 * capacity) * coeff,
                )

        # (sum demand_j * I(slot==j))^2
        # = sum_i d_i^2 * I_i + 2 * sum_{i<j} d_i * d_j * I_i * I_j
        for i, (ind_i, d_i) in enumerate(demand_terms):
            # Diagonal: d_i^2 * I_i (since I_i^2 = I_i)
            for vars_set, coeff in ind_i.items():
                _add(
                    tuple(sorted(vars_set)),
                    capacity_penalty_weight * d_i * d_i * coeff,
                )
            # Cross terms
            for j_idx in range(i + 1, len(demand_terms)):
                ind_j, d_j = demand_terms[j_idx]
                for (vars_i, coeff_i), (vars_j, coeff_j) in product(
                    ind_i.items(), ind_j.items()
                ):
                    combined = vars_i | vars_j
                    _add(
                        tuple(sorted(combined)),
                        2.0 * capacity_penalty_weight * d_i * d_j * coeff_i * coeff_j,
                    )

    return hubo, config


def create_tsp_hubo_binary(
    cost_matrix: npt.NDArray[np.floating],
    start_city: int = 0,
    penalty_weight: float = 4.0,
    objective_weight: float = 1.0,
) -> tuple[dict[tuple[int, ...], float], BinaryBlockConfig]:
    """Generate a binary-encoded HUBO for TSP.

    TSP is the K=1 special case of CVRP with no demands and ``max_steps``
    locked to ``n_customers``. The customer-visit penalty alone already
    rejects any configuration with fewer than ``n_customers`` customer
    slots, but when ``n_customers + 1`` is not a power of 2 some bit
    patterns decode to values outside ``[0, n_customers]``. We add an
    explicit penalty on those out-of-range patterns so the Hamiltonian
    landscape directly reflects the feasibility constraint, instead of
    relying on a counting argument.

    Args:
        cost_matrix: Symmetric distance matrix, shape ``(n, n)``.
        start_city: Index of the fixed start/end city.
        penalty_weight: Customer-visit + slot-validity penalty strength.
        objective_weight: Tour-length weight.

    Returns:
        Tuple of (hubo_dict, config) — same shape as
        :func:`create_cvrp_hubo_binary` but with ``n_vehicles=1`` and
        ``max_steps=n_customers``.

    Raises:
        ValueError: If cost_matrix is not square.
    """
    n_nodes = _check_routing_inputs(cost_matrix)
    n_cust = n_nodes - 1
    hubo, config = create_cvrp_hubo_binary(
        cost_matrix=cost_matrix,
        demands=np.zeros(n_nodes, dtype=np.float64),
        capacity=1.0,  # any value; demands=0 makes capacity terms vacuous.
        n_vehicles=1,
        depot=start_city,
        penalty_weight=penalty_weight,
        objective_weight=objective_weight,
        capacity_penalty_weight=0.0,
        max_steps=n_cust,
    )

    # Slot-validity penalty: for each slot s and each value v in
    # ``(n_cust + 1, ..., 2^B - 1)``, add ``penalty_weight * I(s == v)``
    # to the HUBO. Empty when 2^B == n_cust + 1 (no out-of-range values).
    B = config.bits_per_slot
    for slot in range(config.n_slots):
        slot_bits = list(range(slot * B, slot * B + B))
        for v in range(n_cust + 1, 1 << B):
            # Indicator polynomial I(slot == v) expanded over bits.
            terms: dict[frozenset[int], float] = {frozenset(): 1.0}
            for b in range(B):
                bit_set = (v >> b) & 1
                new_terms: dict[frozenset[int], float] = {}
                for vars_set, coeff in terms.items():
                    if bit_set:
                        key = vars_set | {slot_bits[b]}
                        new_terms[key] = new_terms.get(key, 0.0) + coeff
                    else:
                        new_terms[vars_set] = new_terms.get(vars_set, 0.0) + coeff
                        key = vars_set | {slot_bits[b]}
                        new_terms[key] = new_terms.get(key, 0.0) - coeff
                terms = new_terms
            for vars_set, coeff in terms.items():
                key = tuple(sorted(vars_set))
                hubo[key] = hubo.get(key, 0.0) + penalty_weight * coeff

    return hubo, config


def decode_binary_tsp_solution(
    bitstring: str,
    config: BinaryBlockConfig,
    n_nodes: int,
    start_city: int = 0,
) -> list[int] | None:
    """Decode a binary-encoded TSP bitstring into a tour.

    Returns ``[start_city, c1, ..., c_{n-1}, start_city]`` if the
    bitstring decodes to a valid TSP tour (all customers, each exactly
    once, no empty slot), else ``None``.
    """
    routes = decode_binary_cvrp_solution(bitstring, config, start_city, n_nodes)
    if routes is None or len(routes) != 1:
        return None
    tour = routes[0]
    # decode_binary_cvrp_solution prepends/appends depot; expect depot + n_cust customers + depot.
    if len(tour) != n_nodes + 1:
        return None
    if len(set(tour[1:-1])) != n_nodes - 1:
        return None
    return tour


def is_valid_binary_tsp(
    bitstring: str, config: BinaryBlockConfig, n_nodes: int, start_city: int = 0
) -> bool:
    return (
        decode_binary_tsp_solution(bitstring, config, n_nodes, start_city) is not None
    )


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
            val = int(slot_bits[::-1], 2)
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
            val = int(slot_bits[::-1], 2)

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
    return not any(d > capacity + 1e-9 for d in vehicle_demands)


# --- VRP file parser ---


@dataclass
class RoutingInstance:
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


_SUPPORTED_EWT = {"EUC_2D", "GEO", "EXPLICIT"}
_SUPPORTED_EXPLICIT_FORMATS = {
    "LOWER_DIAG_ROW",
    "LOWER_ROW",
    "UPPER_ROW",
    "UPPER_DIAG_ROW",
    "FULL_MATRIX",
}


# TSPLIB's C reference uses PI = 3.141592 rather than math.pi. Mismatching
# the constant shifts integer-truncated GEO distances by ±1 against published
# benchmarks, so we mirror the reference exactly.
_TSPLIB_PI = 3.141592


def _nint(x: float) -> int:
    """TSPLIB's nint: round half-away-from-zero (NOT Python's banker's round)."""
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


def _euc2d_matrix(coords: Sequence[tuple[float, float]]) -> npt.NDArray[np.float64]:
    n = len(coords)
    cost = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            d = _nint(math.sqrt(dx * dx + dy * dy))
            cost[i, j] = cost[j, i] = d
    return cost


def _geo_matrix(coords: Sequence[tuple[float, float]]) -> npt.NDArray[np.float64]:
    """TSPLIB GEO: spherical distance using the radius from the reference spec.

    Mirrors the C reference exactly: ``PI = 3.141592``, the (1±q1)/(1∓q1)
    cosine identity, integer-truncated `(int)(R*acos(...) + 1.0)`. The
    ``acos`` argument is clamped to ``[-1, 1]`` to absorb 1-ulp drift from
    the trig identity; without it, near-identical points can raise.
    """

    # TSPLIB encodes coords as DDD.MM where MM is minutes — convert to radians.
    def to_rad(x: float) -> float:
        deg = int(x)
        minutes = x - deg
        return _TSPLIB_PI * (deg + 5.0 * minutes / 3.0) / 180.0

    rad = [(to_rad(lat), to_rad(lon)) for lat, lon in coords]
    n = len(coords)
    R = 6378.388
    cost = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            q1 = math.cos(rad[i][1] - rad[j][1])
            q2 = math.cos(rad[i][0] - rad[j][0])
            q3 = math.cos(rad[i][0] + rad[j][0])
            arg = max(-1.0, min(1.0, 0.5 * ((1 + q1) * q2 - (1 - q1) * q3)))
            d = int(R * math.acos(arg) + 1.0)
            cost[i, j] = cost[j, i] = d
    return cost


def _unpack_explicit(
    values: Sequence[float], n: int, fmt: str
) -> npt.NDArray[np.float64]:
    cost = np.zeros((n, n), dtype=np.float64)
    it = iter(values)
    if fmt == "LOWER_DIAG_ROW":
        for i in range(n):
            for j in range(i + 1):
                cost[i, j] = cost[j, i] = next(it)
    elif fmt == "LOWER_ROW":
        for i in range(1, n):
            for j in range(i):
                cost[i, j] = cost[j, i] = next(it)
    elif fmt == "UPPER_ROW":
        for i in range(n):
            for j in range(i + 1, n):
                cost[i, j] = cost[j, i] = next(it)
    elif fmt == "UPPER_DIAG_ROW":
        for i in range(n):
            for j in range(i, n):
                cost[i, j] = cost[j, i] = next(it)
    elif fmt == "FULL_MATRIX":
        for i in range(n):
            for j in range(n):
                cost[i, j] = next(it)
    else:
        raise ValueError(f"unsupported EDGE_WEIGHT_FORMAT: {fmt}")
    return cost


def parse_tsplib_file(path: str | Path) -> RoutingInstance:
    """Parse a TSPLIB/CVRPLIB format `.vrp` or `.tsp` file.

    Supports:
    - ``TYPE: CVRP`` and ``TYPE: TSP``
    - ``EDGE_WEIGHT_TYPE``: ``EUC_2D``, ``GEO``, ``EXPLICIT``
    - ``EDGE_WEIGHT_FORMAT`` (for EXPLICIT): ``LOWER_DIAG_ROW``, ``UPPER_ROW``, ``FULL_MATRIX``
    - ``NODE_COORD_SECTION``, ``EDGE_WEIGHT_SECTION``, ``DEMAND_SECTION``, ``DEPOT_SECTION``

    Args:
        path: Path to the `.vrp` or `.tsp` file.

    Returns:
        Parsed :class:`~divi.qprog.problems.RoutingInstance`.

    Raises:
        ValueError: If the file format is unsupported or malformed.
    """
    path = Path(path)
    lines = path.read_text().splitlines()

    inst = RoutingInstance()
    section = None
    coords_list: list[tuple[float, float]] = []
    demands_list: list[float] = []
    depot_nodes: list[int] = []
    edge_weights: list[float] = []
    edge_weight_type = "EUC_2D"
    edge_weight_format: str | None = None

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
                    "optimal value:",
                    "optimal value :",
                    "optimal:",
                ]:
                    if prefix in lower:
                        tail = lower.split(prefix, 1)[1]
                        # Accept ints, decimals (with or without leading zero),
                        # and scientific notation. Anchored to the start of
                        # ``tail`` so we don't pick up unrelated numbers.
                        match = re.search(
                            r"-?(?:\d+\.\d+|\.\d+|\d+)(?:[eE][+-]?\d+)?",
                            tail,
                        )
                        if match:
                            inst.optimal_cost = float(match.group(0))
            elif key == "TYPE":
                inst.problem_type = value.upper()
            elif key == "DIMENSION":
                inst.dimension = int(value)
            elif key == "CAPACITY":
                inst.capacity = int(value)
            elif key == "EDGE_WEIGHT_TYPE":
                edge_weight_type = value.upper()
                if edge_weight_type not in _SUPPORTED_EWT:
                    raise ValueError(
                        f"Unsupported EDGE_WEIGHT_TYPE: {value}. "
                        f"Supported: {sorted(_SUPPORTED_EWT)}."
                    )
            elif key == "EDGE_WEIGHT_FORMAT":
                edge_weight_format = value.upper()
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
        elif upper == "EDGE_WEIGHT_SECTION":
            section = "EDGE_WEIGHTS"
            continue
        elif upper == "EOF":
            break
        elif upper in {"DISPLAY_DATA_SECTION", "FIXED_EDGES_SECTION"}:
            # Skip auxiliary sections we don't consume.
            section = "SKIP"
            continue

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
        elif section == "EDGE_WEIGHTS":
            edge_weights.extend(float(tok) for tok in line.split())

    # Build arrays
    if coords_list:
        inst.coords = np.array(coords_list, dtype=np.float64)
    if demands_list:
        inst.demands = np.array(demands_list, dtype=np.float64)
    if depot_nodes:
        # TSPLIB uses 1-based indexing
        inst.depot = depot_nodes[0] - 1

    # Compute distance matrix
    if edge_weight_type == "EXPLICIT":
        if edge_weight_format is None:
            raise ValueError("EXPLICIT requires EDGE_WEIGHT_FORMAT.")
        if edge_weight_format not in _SUPPORTED_EXPLICIT_FORMATS:
            raise ValueError(
                f"unsupported EDGE_WEIGHT_FORMAT: {edge_weight_format}. "
                f"Supported: {sorted(_SUPPORTED_EXPLICIT_FORMATS)}."
            )
        inst.cost_matrix = _unpack_explicit(
            edge_weights, inst.dimension, edge_weight_format
        )
    elif coords_list:
        if edge_weight_type == "GEO":
            inst.cost_matrix = _geo_matrix(coords_list)
        else:  # EUC_2D
            inst.cost_matrix = _euc2d_matrix(coords_list)

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


class _RoutingProblemBase(QAOAProblem):
    """Base for the QUBO/HUBO-backed :class:`CVRPProblem`.

    The constructor calls :meth:`_set_hamiltonian_spec`; the Ising conversion
    and mixer are built on first use.
    """

    def _set_hamiltonian_spec(
        self,
        qubo,
        *,
        block_size: int,
        n_blocks: int,
        hamiltonian_builder: Literal["native", "quadratized"] = "native",
        use_constrained_mixer: bool = True,
    ) -> None:
        self._qubo = qubo
        self._block_size = block_size
        self._n_blocks = n_blocks
        self._hamiltonian_builder = hamiltonian_builder
        self._use_constrained_mixer = use_constrained_mixer

    @cached_property
    def _ising(self) -> IsingResult:
        return qubo_to_ising(self._qubo, hamiltonian_builder=self._hamiltonian_builder)

    @property
    def cost_hamiltonian(self) -> SparsePauliOp:
        return self._ising.cost_hamiltonian

    @cached_property
    def mixer_hamiltonian(self) -> SparsePauliOp:
        if not self._use_constrained_mixer:
            return x_mixer(self._ising.n_qubits)
        n_qubits = self._block_size * self._n_blocks
        graph = build_block_xy_mixer_graph(
            self._block_size, self._n_blocks, range(n_qubits)
        )
        return xy_mixer(graph, n_qubits=n_qubits)


# --- TSP decomposition ---


def _split_cities_into_clusters(
    cost_matrix: npt.NDArray[np.floating], max_size: int
) -> list[list[int]]:
    """Group cities into clusters of at most ``max_size`` by average-linkage
    agglomeration, merging the closest pair whose union still fits.
    Deterministic; clusters are sorted city lists ordered by first city."""
    n = len(cost_matrix)
    sums = np.array((cost_matrix + cost_matrix.T) / 2, dtype=np.float64)
    sizes = np.ones(n)
    members = [[city] for city in range(n)]
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    while True:
        fits = upper & (sizes[:, None] + sizes[None, :] <= max_size)
        if not fits.any():
            break
        average = np.where(fits, sums / np.outer(sizes, sizes), np.inf)
        a, b = divmod(int(np.argmin(average)), n)
        sums[a] += sums[b]
        sums[:, a] += sums[:, b]
        sizes[a] += sizes[b]
        sizes[b] = max_size + 1
        members[a] = sorted(members[a] + members[b])
        members[b] = []
    return sorted((cluster for cluster in members if cluster), key=lambda c: c[0])


def _patch_cycles_into_tour(
    cycle_groups: Sequence[Sequence[int]], cost_matrix: npt.NDArray[np.floating]
) -> list[int]:
    """Merge city cycles into one, repeatedly swapping the edge pair whose
    exchange between two cycles adds the least cost.

    Raises:
        ValueError: If there are no cycles or a cycle is empty.
    """
    cycles: list[list[int]] = [list(cycle) for cycle in cycle_groups]
    if not cycles or not all(cycles):
        raise ValueError("cycles must be non-empty and each hold at least one city.")
    while len(cycles) > 1:
        best = (math.inf, 0, 1, 0, 0, False)
        for ia, ib in combinations(range(len(cycles)), 2):
            a, b = cycles[ia], cycles[ib]
            for i, j in product(range(len(a)), range(len(b))):
                a0, a1 = a[i], a[(i + 1) % len(a)]
                b0, b1 = b[j], b[(j + 1) % len(b)]
                removed = cost_matrix[a0, a1] + cost_matrix[b0, b1]
                for reverse, added in (
                    (False, cost_matrix[a0, b1] + cost_matrix[b0, a1]),
                    (True, cost_matrix[a0, b0] + cost_matrix[b1, a1]),
                ):
                    if added - removed < best[0]:
                        best = (added - removed, ia, ib, i, j, reverse)
        _, ia, ib, i, j, reverse = best
        a, b = cycles[ia], cycles[ib]
        spliced = b[j::-1] + b[:j:-1] if reverse else b[j + 1 :] + b[: j + 1]
        merged = a[: i + 1] + spliced + a[i + 1 :]
        cycles = [c for k, c in enumerate(cycles) if k not in (ia, ib)] + [merged]
    return cycles[0]


# --- TSPProblem ---


def _drop_zero_terms(
    terms: Mapping[tuple[int, ...], float],
) -> dict[tuple[Hashable, ...], float]:
    return {key: coeff for key, coeff in terms.items() if coeff != 0.0}


def _compute_nearest_neighbour_cost(
    cost_matrix: npt.NDArray[np.floating], start_city: int
) -> float:
    """Cost of the round trip from ``start_city`` that always moves to the closest unvisited city."""
    unvisited = set(range(len(cost_matrix))) - {start_city}
    city, total = start_city, 0.0
    while unvisited:
        nearest = min(unvisited, key=cost_matrix[city].__getitem__)
        total += cost_matrix[city, nearest]
        unvisited.remove(nearest)
        city = nearest
    return float(total + cost_matrix[city, start_city])


def _check_default_penalty_inputs(
    cost_matrix: npt.NDArray[np.floating], objective_weight: float
) -> None:
    if objective_weight <= 0 or (cost_matrix < 0).any():
        raise ValueError(
            "penalty_weight must be given when objective_weight is not positive "
            "or cost_matrix has negative entries."
        )


def _compute_default_tsp_penalty(
    cost_matrix: npt.NDArray[np.floating],
    start_city: int,
    objective_weight: float,
    encoding: str,
) -> float:
    """Penalty weight: ``objective_weight`` times the nearest-neighbour tour cost,
    doubled for ``"binary"``."""
    _check_default_penalty_inputs(cost_matrix, objective_weight)
    scale = objective_weight * (
        _compute_nearest_neighbour_cost(cost_matrix, start_city) or 1.0
    )
    return scale if encoding == "one_hot" else 2.0 * scale


def _compute_default_cvrp_penalty(
    cost_matrix: npt.NDArray[np.floating], depot: int, objective_weight: float
) -> float:
    """One-hot CVRP penalty weight: ``objective_weight`` times the nearest-neighbour
    tour cost from the depot plus ``max(n_customers - 2, 1)`` times the largest
    cost entry, or ``objective_weight`` when both are zero."""
    _check_default_penalty_inputs(cost_matrix, objective_weight)
    n_customers = len(cost_matrix) - 1
    scale = _compute_nearest_neighbour_cost(cost_matrix, depot) + max(
        n_customers - 2, 1
    ) * float(cost_matrix.max())
    return objective_weight * (scale or 1.0)


class TSPProblem(BinaryOptimizationProblem):
    """Travelling Salesman Problem for QAOA and PCE.

    A :class:`~divi.qprog.problems.BinaryOptimizationProblem` over the
    assignment variables ``x_{i,t}`` (city *i* at time step *t*, qubit
    ``i*(n-1) + t``) whose objective is the tour length.

    Supports two encodings:

    * ``"one_hot"`` — each city takes one time step and each step holds one
      city, both encoded as constraints, so ``constraints`` is populated
      and :class:`~divi.qprog.algorithms.PCE`, ``relaxed_energy`` and
      :func:`~divi.backends.characterization.characterize_and_validate` use
      both. QAOA starts from a W-state with an XY mixer on each city's block
      of qubits, and its cost Hamiltonian omits the per-city penalty the mixer
      enforces. Qubit count: ``(n-1)²``.
    * ``"binary"`` — log-encoded slot bits with Hadamard initialisation and a
      transverse-field mixer. Qubit count: ``(n-1) · ⌈log₂(n)⌉``. No linear
      constraints; the Hamiltonian omits the constant of each city's visit
      penalty.

    The Hamiltonians are built on first use, and the constructor copies
    ``cost_matrix``.

    Args:
        cost_matrix: Symmetric, finite distance/cost matrix of shape ``(n, n)``
            with ``n >= 3``.
        start_city: Integer index of the fixed start city. Defaults to 0.
        encoding: ``"one_hot"`` or ``"binary"``. Defaults to ``"one_hot"``.
        penalty_weight: Constraint penalty strength. ``None`` (default) picks
            ``objective_weight`` times the nearest-neighbour tour cost (twice
            that for ``"binary"``). Needs non-negative costs and a positive
            ``objective_weight``.
        objective_weight: Objective weight. Defaults to 1.0.
        max_cities_per_partition: Enables :meth:`decompose`. Cities are
            clustered into groups of at most this many (at least 3; ``n`` or
            more gives one group). Each group of three or more becomes a
            sub-:class:`TSPProblem`, and aggregation patches the sub-tours by
            repeatedly merging the pair whose edge exchange adds the least
            cost. Invalid one-hot sub-solutions are Hungarian-repaired; invalid
            binary ones fall back to index order. Requires a symmetric cost
            matrix.

    Raises:
        ValueError: If ``cost_matrix`` is not square, has fewer than 3 cities or
            a non-finite entry, if ``encoding`` is not ``"one_hot"`` or
            ``"binary"``, or if ``start_city`` is out of range.
        TypeError: If ``start_city`` is not an integer.
    """

    def __init__(
        self,
        cost_matrix: npt.NDArray[np.floating],
        *,
        start_city: int = 0,
        encoding: Literal["one_hot", "binary"] = "one_hot",
        penalty_weight: float | None = None,
        objective_weight: float = 1.0,
        max_cities_per_partition: int | None = None,
    ):
        if max_cities_per_partition is not None and max_cities_per_partition < 3:
            raise ValueError(
                f"max_cities_per_partition must be at least 3, got {max_cities_per_partition}."
            )
        cost_matrix = np.array(cost_matrix, dtype=np.float64)
        n = _check_routing_inputs(cost_matrix)
        if n < 3:
            raise ValueError(f"cost_matrix must cover at least 3 cities, got {n}.")
        if not np.isfinite(cost_matrix).all():
            raise ValueError("cost_matrix must be finite.")
        _check_encoding(encoding)
        _check_node_index("start_city", start_city, n)
        start_city = int(start_city)
        self._cost_matrix = cost_matrix
        self._start_city = start_city
        self._n_cities = n
        self._encoding = encoding
        self._objective_weight = objective_weight
        self._explicit_penalty_weight = penalty_weight
        self._max_cities_per_partition = max_cities_per_partition
        self._binary_config: BinaryBlockConfig | None = None
        m = n - 1
        if penalty_weight is None:
            penalty_weight = _compute_default_tsp_penalty(
                cost_matrix, start_city, objective_weight, encoding
            )

        if encoding == "binary":
            objective, self._binary_config = create_tsp_hubo_binary(
                cost_matrix, start_city, 0.0, objective_weight
            )
            penalty, _ = create_tsp_hubo_binary(cost_matrix, start_city, 1.0, 0.0)
            super().__init__(
                _drop_zero_terms(objective),
                penalty=_drop_zero_terms(penalty),
                penalty_weight=penalty_weight,
            )
        else:
            steps: list[dict[Hashable, float]] = [
                {m * city + t: 1.0 for city in range(m)} for t in range(m)
            ]
            cities: list[dict[Hashable, float]] = [
                {m * city + t: 1.0 for t in range(m)} for city in range(m)
            ]
            super().__init__(
                create_tsp_qubo(cost_matrix, start_city, 0.0, objective_weight),
                constraints=[
                    LinearConstraint(coefficients, "==", 1)
                    for coefficients in (*steps, *cities)
                ],
                penalty_weight=penalty_weight,
            )

    @property
    def encoding(self) -> str:
        return self._encoding

    @property
    def binary_config(self) -> BinaryBlockConfig | None:
        return self._binary_config

    @cached_property
    def _reduced_ising(self) -> IsingResult:
        return qubo_to_ising(
            create_tsp_qubo(
                self._cost_matrix,
                self._start_city,
                self._penalty_weight,
                self._objective_weight,
            )
        )

    @property
    def cost_hamiltonian(self) -> SparsePauliOp:
        """Reduced-penalty Hamiltonian for ``"one_hot"``, the full one for ``"binary"``."""
        if self._encoding == "binary":
            return super().cost_hamiltonian
        return self._reduced_ising.cost_hamiltonian

    @property
    def mixer_hamiltonian(self) -> SparsePauliOp:
        """XY mixer on each city's block of qubits, or the X mixer for binary encoding."""
        if self._encoding == "binary":
            return super().mixer_hamiltonian
        if self._mixer_cache is None:
            m = self._n_cities - 1
            idx = self._canonical_problem.variable_to_idx
            graph = build_block_xy_mixer_graph(
                m, m, [idx[qubit] for qubit in range(m * m)]
            )
            self._mixer_cache = xy_mixer(graph, n_qubits=m * m)
        return self._mixer_cache

    @property
    def recommended_initial_state(self) -> InitialState:
        if self._encoding == "binary":
            return SuperpositionState()
        return WState(self._n_cities - 1, self._n_cities - 1)

    def _decode_tour(self, bitstring: str) -> list[int] | None:
        if self._encoding == "binary" and self._binary_config is not None:
            return decode_binary_tsp_solution(
                bitstring, self._binary_config, self._n_cities, self._start_city
            )
        return decode_tsp_solution(bitstring, self._n_cities, self._start_city)

    @property
    def decode_fn(self) -> Callable[[str], Any]:
        return self._decode_tour

    def _solution_key(self, bitstring: str) -> Hashable:
        return bitstring

    @property
    def feasible_dimension(self) -> int:
        """Size of the feasible subspace, equal to ``(n-1)!``."""
        return math.factorial(self._n_cities - 1)

    def is_feasible(self, bitstring: str) -> bool:
        if self._encoding == "binary" and self._binary_config is not None:
            return is_valid_binary_tsp(
                bitstring, self._binary_config, self._n_cities, self._start_city
            )
        return is_valid_tsp_tour(bitstring, self._n_cities)

    def repair_infeasible_bitstring(self, bitstring: str) -> tuple[str, Any, float]:
        if self._encoding == "binary":
            raise NotImplementedError(
                "repair_infeasible_bitstring is not implemented for binary TSP."
            )
        return repair_tsp_solution(
            bitstring, self._n_cities, self._start_city, self._cost_matrix
        )

    def compute_energy(self, bitstring: str) -> float:
        """Cost of the tour ``bitstring`` encodes.

        Raises:
            ValueError: If ``bitstring`` is not a valid tour.
        """
        tour = self._decode_tour(bitstring)
        if tour is None:
            raise ValueError(f"Bitstring {bitstring!r} does not encode a valid tour.")
        return compute_tour_cost(tour, self._cost_matrix)

    @cached_property
    def _clusters(self) -> list[list[int]]:
        if self._max_cities_per_partition is None:
            raise ValueError(
                "Cannot decompose: max_cities_per_partition was not set at construction."
            )
        cost = self._cost_matrix
        if not np.allclose(cost, cost.T):
            raise ValueError("Cannot decompose: cost_matrix must be symmetric.")
        return _split_cities_into_clusters(cost, self._max_cities_per_partition)

    @cached_property
    def _partition_clusters(self) -> dict[Hashable, list[int]]:
        partitions: dict[Hashable, list[int]] = {
            (f"P{k}", len(cluster)): cluster
            for k, cluster in enumerate(self._clusters)
            if len(cluster) >= 3
        }
        if not partitions:
            raise ValueError(
                "No city cluster has three or more cities, leaving nothing to solve; "
                "raise max_cities_per_partition or solve the problem without "
                "partitioning."
            )
        return partitions

    def decompose(self) -> dict[Hashable, QAOAProblem]:
        return {
            prog_id: _PartitionTSPProblem(
                self._cost_matrix[np.ix_(cluster, cluster)],
                encoding=self._encoding,
                penalty_weight=self._explicit_penalty_weight,
                objective_weight=self._objective_weight,
            )
            for prog_id, cluster in self._partition_clusters.items()
        }

    def _has_reproducible_decomposition(self) -> bool:
        return True

    def initial_solution_size(self) -> int:
        return self._n_cities

    def extend_solution(
        self,
        current_solution: Sequence[int],
        prog_id: Hashable,
        candidate_decoded: Sequence[int] | None,
    ) -> list[int]:
        """Record each city's 1-based tour position in its cluster; ``0`` is unset.

        A ``None`` candidate, which only a binary-encoding sub-problem yields
        for an invalid tour, puts the cluster in index order.
        """
        extended = list(current_solution)
        cluster = self._partition_clusters[prog_id]
        order = (
            range(len(cluster)) if candidate_decoded is None else candidate_decoded[:-1]
        )
        for city in cluster:
            extended[city] = 0
        for position, local in enumerate(order, start=1):
            extended[cluster[local]] = position
        return extended

    def _build_tour(self, solution: Sequence[int]) -> list[int]:
        """The patched round trip from ``start_city`` for a global solution vector."""
        cycles = [
            sorted(cluster, key=solution.__getitem__) for cluster in self._clusters
        ]
        tour = _patch_cycles_into_tour(cycles, self._cost_matrix)
        start = tour.index(self._start_city)
        return [*tour[start:], *tour[:start], self._start_city]

    def evaluate_global_solution(self, solution: Sequence[int]) -> float:
        return compute_tour_cost(self._build_tour(solution), self._cost_matrix)

    def postprocess_candidates(
        self, candidates: Sequence[tuple[float, list[int]]], *, strict: bool = False
    ) -> list[tuple[Any, float]]:
        """Distinct ``(tour, cost)`` pairs, cheapest first; ``strict`` has no effect."""
        tours = {tuple(self._build_tour(solution)) for _, solution in candidates}
        return sorted(
            (
                (list(tour), compute_tour_cost(tour, self._cost_matrix))
                for tour in tours
            ),
            key=lambda result: result[1],
        )


class _PartitionTSPProblem(TSPProblem):
    """Sub-problem of a partitioned TSP: a one-hot bitstring that is not a tour decodes to its Hungarian repair."""

    @property
    def decode_fn(self) -> Callable[[str], Any]:
        if self._encoding == "binary":
            return super().decode_fn
        return self._decode_repaired_tour

    def _decode_repaired_tour(self, bitstring: str) -> list[int]:
        tour = self._decode_tour(bitstring)
        if tour is not None:
            return tour
        return self.repair_infeasible_bitstring(bitstring)[1]


# --- CVRPProblem ---


class CVRPProblem(_RoutingProblemBase):
    """Capacitated Vehicle Routing Problem for QAOA.

    The Hamiltonians are built on first use. Supports two encodings:

    * ``"one_hot"`` — variable ``x_{i,v,t}`` (customer *i*, vehicle *v*, step
      *t*) on qubit ``i*V*T + v*T + t``. QAOA starts from a W-state with an XY
      mixer on each customer's block of ``V*T`` qubits. The phase operator
      penalises two customers sharing a slot and a gap in a route; capacity is
      enforced by :meth:`is_feasible` and the ``"filter"`` and ``"repair"``
      rankings. Qubit count: ``V * n_customers²``.
    * ``"binary"`` — compact binary (HUBO) with Hadamard + X mixer,
      reducing qubit count from O(N) to O(log N) per slot.

    The constructor copies ``cost_matrix`` and ``demands``.

    Args:
        cost_matrix: Finite distance/cost matrix of shape ``(n, n)`` with at
            least one customer besides the depot.
        demands: Finite, non-negative customer demands (length = n_nodes, depot
            demand = 0), each at most ``capacity``.
        capacity: Positive vehicle capacity.
        n_vehicles: Number of vehicles, at least 1.
        depot: Integer index of the depot node. Defaults to 0.
        encoding: ``"one_hot"`` or ``"binary"``. Defaults to ``"one_hot"``.
        penalty_weight: Constraint penalty strength, finite and positive.
            ``None`` (default) picks, for ``"one_hot"``, ``objective_weight``
            times the sum of the nearest-neighbour tour cost from the depot and
            ``max(n_customers - 2, 1)`` times the largest cost entry
            (``objective_weight`` if that sum is zero), and for ``"binary"``
            4.0. The one-hot default needs non-negative costs and a positive
            ``objective_weight``.
        objective_weight: Objective weight. Defaults to 1.0.
        capacity_penalty_weight: Capacity penalty strength (``"binary"``
            encoding only; ``"one_hot"`` ignores it). Defaults to 4.0.
        max_steps: Maximum route length per vehicle (``"binary"`` encoding
            only). ``None`` (default) uses ``n_customers``. Qubit count and
            HUBO term count both scale with ``n_vehicles * max_steps``;
            lowering ``max_steps`` below a vehicle's actual stop count
            renders the instance infeasible.

    Raises:
        ValueError: If ``cost_matrix`` is not square or has a non-finite entry,
            ``demands`` do not match it or violate the bounds above,
            ``encoding``, ``depot``, ``n_vehicles``, ``capacity`` or
            ``penalty_weight`` is out of range, or ``max_steps`` is given with
            the ``"one_hot"`` encoding.
        TypeError: If ``depot`` or ``n_vehicles`` is not an integer.
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
        penalty_weight: float | None = None,
        objective_weight: float = 1.0,
        capacity_penalty_weight: float = 4.0,
        max_steps: int | None = None,
    ):
        cost_matrix = np.array(cost_matrix, dtype=np.float64)
        demands = np.array(demands, dtype=np.float64)
        _check_cvrp_inputs(cost_matrix, demands, capacity, n_vehicles, depot, encoding)
        if penalty_weight is None:
            penalty_weight = (
                _compute_default_cvrp_penalty(cost_matrix, depot, objective_weight)
                if encoding == "one_hot"
                else 4.0
            )
        elif not (math.isfinite(penalty_weight) and penalty_weight > 0):
            raise ValueError("penalty_weight must be finite and positive.")
        n_vehicles, depot = int(n_vehicles), int(depot)
        self._cost_matrix = cost_matrix
        self._demands = demands
        self._capacity = capacity
        self._n_vehicles = n_vehicles
        self._depot = depot
        self._encoding = encoding
        self._penalty_weight = float(penalty_weight)
        self._n_cities = cost_matrix.shape[0]
        self._binary_config: BinaryBlockConfig | None = None

        n_cust = self._n_cities - 1

        if encoding == "binary":
            hubo, self._binary_config = create_cvrp_hubo_binary(
                self._cost_matrix,
                demands=demands,
                capacity=capacity,
                n_vehicles=n_vehicles,
                depot=depot,
                penalty_weight=penalty_weight,
                objective_weight=objective_weight,
                capacity_penalty_weight=capacity_penalty_weight,
                max_steps=max_steps,
            )
            self._set_hamiltonian_spec(
                hubo,
                block_size=self._binary_config.bits_per_slot,
                n_blocks=self._binary_config.n_slots,
                hamiltonian_builder="native",
                use_constrained_mixer=False,
            )
        else:
            if max_steps is not None:
                raise ValueError("max_steps is only supported for encoding='binary'.")
            qubo = create_cvrp_qubo(
                self._cost_matrix,
                n_vehicles=n_vehicles,
                depot=depot,
                penalty_weight=penalty_weight,
                objective_weight=objective_weight,
            )
            bs, nb = build_cvrp_block_structure(n_cust, n_vehicles)
            self._set_hamiltonian_spec(qubo, block_size=bs, n_blocks=nb)

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

    @property
    def penalty_weight(self) -> float:
        """Penalty strength used in the Hamiltonian."""
        return self._penalty_weight

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

    def repair_infeasible_bitstring(
        self, bitstring: str
    ) -> tuple[str, Any, float | None]:
        if self._encoding == "binary":
            # Binary repair not yet implemented; return unchanged.
            return bitstring, None, None
        return repair_cvrp_solution(
            bitstring,
            n_customers=self._n_cities - 1,
            n_vehicles=self._n_vehicles,
            cost_matrix=self._cost_matrix,
            demands=self._demands,
            capacity=self._capacity,
            depot=self._depot,
        )

    def _solution_key(self, bitstring: str) -> Hashable:
        routes = self.decode_fn(bitstring)
        if routes is None:
            return bitstring
        return tuple(tuple(route) for route in routes)

    def compute_energy(self, bitstring: str) -> float | None:
        if self._encoding == "binary" and self._binary_config is not None:
            routes = decode_binary_cvrp_solution(
                bitstring, self._binary_config, self._depot, self._n_cities
            )
        else:
            routes = decode_cvrp_solution(
                bitstring,
                self._n_cities - 1,
                self._n_vehicles,
                self._depot,
                self._n_cities,
            )
        if routes is None:
            return None
        return sum(compute_tour_cost(route, self._cost_matrix) for route in routes)
