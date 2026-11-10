# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import divi.qprog.problems._binary as binary_module
import divi.qprog.problems._routing as routing_module
from divi.hamiltonians import qubo_to_ising, x_mixer, xy_mixer
from divi.qprog.algorithms import PCE, QAOA, ProgressivePCE
from divi.qprog.checkpointing import CheckpointConfig
from divi.qprog.ensembles import PartitioningProgramEnsemble
from divi.qprog.initial_states import SuperpositionState, build_block_xy_mixer_graph
from divi.qprog.optimizers import (
    GridSearchOptimizer,
    MonteCarloOptimizer,
    ScipyMethod,
    ScipyOptimizer,
)
from divi.qprog.problems import (
    BinaryOptimizationProblem,
    CVRPProblem,
    RoutingInstance,
    TSPProblem,
    build_cvrp_block_structure,
    compute_binary_block_config,
    compute_tour_cost,
    is_valid_tsp_tour,
    parse_tsplib_file,
    parse_vrp_solution,
)
from divi.qprog.problems import parse_tsplib_file as parse_tsplib_file_public
from divi.qprog.problems._routing import (
    _nint,
    _patch_cycles_into_tour,
    _split_cities_into_clusters,
    _unpack_explicit,
    create_cvrp_hubo_binary,
    create_cvrp_qubo,
    create_tsp_hubo_binary,
    create_tsp_qubo,
    decode_binary_cvrp_solution,
    decode_binary_tsp_solution,
    decode_cvrp_solution,
    decode_tsp_solution,
    is_valid_binary_cvrp,
    is_valid_cvrp_solution,
    repair_cvrp_solution,
    repair_tsp_solution,
)

CVRP_COST = np.array(
    [[0, 10, 15, 20], [10, 0, 25, 30], [15, 25, 0, 12], [20, 30, 12, 0]],
    dtype=float,
)
CVRP_DEMANDS = np.array([0, 3, 4, 2], dtype=float)


@pytest.fixture
def three_city_cost():
    return np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]], dtype=float)


@pytest.fixture
def four_city_cost():
    return np.array(
        [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]],
        dtype=float,
    )


@pytest.fixture
def two_bowties_problem():
    return TSPProblem(_TWO_BOWTIES, max_cities_per_partition=4)


def build_euclidean_cost(points):
    xy = np.asarray(points, dtype=float)
    return np.linalg.norm(xy[:, None] - xy[None, :], axis=2)


# Index order 0-1-2-3 crosses itself (cost 2 + 2*sqrt(2)); the optimal tour costs 4.
_BOWTIE = [(0, 0), (1, 1), (1, 0), (0, 1)]
_TWO_BOWTIES = build_euclidean_cost([*_BOWTIE, *((x + 50, y) for x, y in _BOWTIE)])
_TWO_BOWTIES_CLUSTERS = [[0, 1, 2, 3], [4, 5, 6, 7]]


def compute_best_local_tour(cost):
    """Cheapest round trip through every city of ``cost``, starting and ending at 0."""
    return min(
        ([0, *order, 0] for order in itertools.permutations(range(1, len(cost)))),
        key=lambda tour: compute_tour_cost(tour, cost),
    )


def extend_with_tours(problem, choose_tour):
    """Global solution vector holding ``choose_tour(sub_cost)`` for every partition."""
    solution = [0] * problem.initial_solution_size()
    for prog_id, cluster in zip(problem.decompose(), _TWO_BOWTIES_CLUSTERS):
        sub_cost = _TWO_BOWTIES[np.ix_(cluster, cluster)]
        solution = problem.extend_solution(solution, prog_id, choose_tour(sub_cost))
    return solution


def compute_diagonal_energies(problem, bits):
    """Cost-Hamiltonian energy of each row of ``bits``."""
    hamiltonian = problem.cost_hamiltonian
    z = 1 - 2 * bits
    energies = np.zeros(len(bits))
    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        energies += coeff.real * np.prod(z[:, pauli.z], axis=1)
    return energies


def count_zz_terms(hamiltonian):
    return sum(int(pauli.z.sum() == 2) for pauli in hamiltonian.paulis)


def enumerate_route_splits(demands, capacity, n_vehicles):
    """Every tuple of ``n_vehicles`` routes within ``capacity`` that serves each customer once."""
    n_customers = len(demands) - 1
    for order in itertools.permutations(range(1, len(demands))):
        for cuts in itertools.combinations_with_replacement(
            range(n_customers + 1), n_vehicles - 1
        ):
            bounds = (0, *cuts, n_customers)
            routes = tuple(order[a:b] for a, b in itertools.pairwise(bounds))
            if all(sum(demands[c] for c in route) <= capacity for route in routes):
                yield routes


def enumerate_one_hot_per_block_states(n_blocks, block_size):
    """Every bit row with exactly one set qubit in each of ``n_blocks`` contiguous blocks."""
    slots = np.array(list(itertools.product(range(block_size), repeat=n_blocks)))
    bits = np.zeros((len(slots), n_blocks * block_size), dtype=int)
    bits[np.arange(len(slots))[:, None], np.arange(n_blocks) * block_size + slots] = 1
    return bits


def enumerate_cvrp_manifold(problem, n_customers, n_vehicles):
    """Bits, Hamiltonian energies and compactness of every one-slot-per-customer state."""
    bits = enumerate_one_hot_per_block_states(n_customers, n_customers * n_vehicles)
    occupancy = bits.reshape(len(bits), n_customers, n_vehicles, n_customers).sum(
        axis=1
    )
    compact = (occupancy <= 1).all(axis=(1, 2)) & (
        occupancy[:, :, 1:] <= occupancy[:, :, :-1]
    ).all(axis=(1, 2))
    return bits, compute_diagonal_energies(problem, bits), compact


def replace_entry(cost, index, value):
    changed = np.array(cost, dtype=float)
    changed[index] = value
    return changed


def build_random_cost(rng, n_nodes):
    cost = rng.random((n_nodes, n_nodes))
    np.fill_diagonal(cost, 0.0)
    return cost


def compute_route_cost(routes, cost):
    return sum(compute_tour_cost(route, cost) for route in routes)


def seed_best_bitstrings(ensemble, bitstrings):
    """Give each program of ``ensemble``, in order, a sampled distribution
    concentrated on one bitstring, bypassing circuit execution."""
    for program, bitstring in zip(ensemble.programs.values(), bitstrings):
        program._results["best_probs"] = {"tag": {bitstring: 1.0}}
        program._losses_history = [{"dummy_loss": 0.0}]


def _write_file(tmp_path, text, name="instance.tsp"):
    p = tmp_path / name
    p.write_text(text)
    return p


def _all_bitstrings(n_qubits):
    """Rows are bit arrays whose position ``i`` is character ``i`` of the bitstring."""
    rows = np.arange(2**n_qubits)[:, None]
    return ((rows >> np.arange(n_qubits - 1, -1, -1)) & 1).astype(bool)


def _hubo_energies(hubo, n_qubits):
    x = _all_bitstrings(n_qubits)
    energies = np.zeros(len(x))
    for key, coeff in hubo.items():
        energies += coeff * x[:, list(key)].all(axis=1)
    return energies


def _slot_values(x, bits_per_slot):
    """Little-endian integer value of every slot, shape ``(rows, n_slots)``."""
    weights = 1 << np.arange(bits_per_slot)
    return x.reshape(len(x), -1, bits_per_slot).astype(int) @ weights


def _binary_cvrp_reference_energy(
    slot_vals, cost, demands, capacity, n_vehicles, max_steps, weight=4.0, depot=0
):
    customers = [c for c in range(len(cost)) if c != depot]
    n_cust = len(customers)
    energy = weight * sum(
        (np.count_nonzero(slot_vals == j) - 1) ** 2 - 1 for j in range(1, n_cust + 1)
    )
    for v in range(n_vehicles):
        route = [
            customers[val - 1] if 1 <= val <= n_cust else None
            for val in slot_vals[v * max_steps : (v + 1) * max_steps]
        ]
        load = sum(demands[node] for node in route if node is not None)
        energy += weight * ((load - capacity) ** 2 - capacity**2)
        legs = [(depot, route[0]), (route[-1], depot), *zip(route, route[1:])]
        energy += sum(cost[a, b] for a, b in legs if a is not None and b is not None)
    return energy


def _cvrp_one_hot_bits(assignments, n_vehicles=2, n_customers=3):
    """One-hot CVRP bitstring with customer ``i`` in slot ``(v, t)`` for each ``(v, t, i)``."""
    bits = np.zeros((n_customers, n_vehicles, n_customers), dtype=int)
    for v, t, i in assignments:
        bits[i, v, t] = 1
    return "".join(str(x) for x in bits.flatten())


def _tsp_assignment_bits(order):
    """One-hot TSP bit vector visiting reduced city ``order[t]`` at step ``t``."""
    m = len(order)
    mat = np.zeros((m, m), dtype=int)
    mat[list(order), range(m)] = 1
    return mat.flatten()


class TestCreateTspQubo:
    def test_reduced_matrix_values(self, three_city_cost):
        np.testing.assert_array_equal(
            create_tsp_qubo(three_city_cost),
            [[6, 0, 8, 20], [0, 6, 20, 8], [0, 0, 11, 0], [0, 0, 0, 11]],
        )

    def test_unreduced_matrix_values(self, three_city_cost):
        np.testing.assert_array_equal(
            create_tsp_qubo(three_city_cost, reduced=False),
            [[2, 8, 8, 20], [0, 2, 20, 8], [0, 0, 7, 8], [0, 0, 0, 7]],
        )

    @pytest.mark.parametrize("reduced, n_penalties", [(True, 1), (False, 2)])
    @pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
    def test_permutation_energy_is_tour_cost_minus_penalty_offset(
        self, four_city_cost, reduced, n_penalties, order
    ):
        x = _tsp_assignment_bits(order)
        Q = create_tsp_qubo(four_city_cost, reduced=reduced)
        tour = decode_tsp_solution("".join(map(str, x)), 4)
        assert (
            x @ Q @ x == compute_tour_cost(tour, four_city_cost) - n_penalties * 4.0 * 3
        )

    @pytest.mark.parametrize("start_city", [3, 5])
    def test_out_of_range_start_city_raises(self, three_city_cost, start_city):
        with pytest.raises(ValueError, match="out of range"):
            create_tsp_qubo(three_city_cost, start_city=start_city)


class TestIsValidTspTour:
    def test_accepts_feasible_assignments(self):
        assert is_valid_tsp_tour("1001", 3) is True
        assert is_valid_tsp_tour("0110", 3) is True

    def test_rejects_infeasible_assignments(self):
        assert is_valid_tsp_tour("1100", 3) is False
        assert is_valid_tsp_tour("0000", 3) is False
        assert is_valid_tsp_tour("101", 3) is False
        assert is_valid_tsp_tour("1010", 3) is False


class TestDecodeTspSolution:
    @pytest.mark.parametrize(
        "bitstring, expected", [("1001", [0, 1, 2, 0]), ("0110", [0, 2, 1, 0])]
    )
    def test_default_start_city_tour(self, bitstring, expected):
        assert decode_tsp_solution(bitstring, 3) == expected

    def test_infeasible_returns_none(self):
        assert decode_tsp_solution("1100", 3, start_city=0) is None


def test_undirected_matrix_same_cost_both_directions(three_city_cost):
    assert compute_tour_cost([0, 1, 2, 0], three_city_cost) == 45.0
    assert compute_tour_cost([0, 2, 1, 0], three_city_cost) == 45.0


class TestRepairTspSolution:
    def test_feasible_returns_full_tuple(self, three_city_cost):
        assert repair_tsp_solution("0110", 3, 0, three_city_cost) == (
            "0110",
            [0, 2, 1, 0],
            45.0,
        )

    def test_infeasible_repaired(self, three_city_cost):
        repaired_bs, tour, cost_val = repair_tsp_solution("0000", 3, 0, three_city_cost)
        assert is_valid_tsp_tour(repaired_bs, 3)
        assert cost_val > 0

    def test_repair_produces_valid_4cities(self):
        cost = np.array(
            [[0, 10, 15, 20], [10, 0, 25, 30], [15, 25, 0, 35], [20, 30, 35, 0]]
        )
        repaired_bs, _, _ = repair_tsp_solution("110100010", 4, 0, cost)
        assert is_valid_tsp_tour(repaired_bs, 4)


SAMPLE_VRP = """\
# Comment line
NAME : TEST-n4-k2-01
COMMENT : "Test instance; Optimal cost: 100"
TYPE : CVRP
DIMENSION : 5
EDGE_WEIGHT_TYPE : EUC_2D
CAPACITY : 10
NODE_COORD_SECTION
1    0   0
2    3   0
3    0   4
4    3   4
5    6   0
DEMAND_SECTION
1    0
2    3
3    4
4    2
5    5
DEPOT_SECTION
1
-1
EOF
"""

TINY_K3_VRP = """\
NAME : "tiny-k3"
COMMENT : "Optimal cost : 42"
TYPE : CVRP
DIMENSION : 3
EDGE_WEIGHT_TYPE : EUC_2D
NODE_COORD_SECTION
# comment line with tokens
1 1 2
2 4 6
3 1 6
DISPLAY_DATA_SECTION
1 9 9
2 9 9
3 9 9
DEPOT_SECTION
2
-1
CAPACITY : 5
DEMAND_SECTION
1 1
2 0
3 4
FIXED_EDGES_SECTION
1 3
-1
EOF
"""

SAMPLE_SOL = """\
Route #1: 2 3
Route #2: 4 5
Cost 100
"""


@pytest.fixture
def vrp_file(tmp_path):
    return _write_file(tmp_path, SAMPLE_VRP, "test.vrp")


@pytest.fixture
def sol_file(tmp_path):
    return _write_file(tmp_path, SAMPLE_SOL, "test.opt.sol")


class TestParseVrpFile:
    def test_sample_vrp_parses_all_fields(self, vrp_file):
        inst = parse_tsplib_file(vrp_file)
        assert inst.name == "TEST-n4-k2-01"
        assert inst.problem_type == "CVRP"
        assert inst.dimension == 5
        assert inst.capacity == 10
        assert inst.n_vehicles == 2
        assert inst.depot == 0
        assert inst.optimal_cost == 100.0
        assert inst.n_customers == 4
        assert inst.coords.shape == (5, 2)
        np.testing.assert_array_equal(inst.coords[0], [0, 0])
        assert inst.demands.shape == (5,)
        assert inst.demands[0] == 0
        assert inst.cost_matrix.shape == (5, 5)
        assert inst.cost_matrix[0, 1] == 3.0
        assert inst.cost_matrix[0, 2] == 4.0
        assert inst.cost_matrix[0, 3] == 5.0
        assert inst.cost_matrix[1, 0] == inst.cost_matrix[0, 1]

    def test_qoblib_instance(self):
        qoblib_path = Path(__file__).parent / "fixtures" / "XSH-n20-k4-01.vrp"
        inst = parse_tsplib_file(qoblib_path)
        assert inst.dimension == 21
        assert inst.n_customers == 20
        assert inst.n_vehicles == 4
        assert inst.capacity == 231
        assert inst.optimal_cost == 646.0

    def test_non_first_depot_and_auxiliary_sections(self, tmp_path):
        inst = parse_tsplib_file(_write_file(tmp_path, TINY_K3_VRP, "tiny.vrp"))
        assert inst.name == "tiny-k3"
        assert inst.comment == "Optimal cost : 42"
        assert inst.problem_type == "CVRP"
        assert inst.dimension == 3
        assert inst.capacity == 5
        assert inst.n_vehicles == 3
        assert inst.depot == 1
        assert inst.optimal_cost == 42.0
        np.testing.assert_array_equal(inst.coords, [[1, 2], [4, 6], [1, 6]])
        np.testing.assert_array_equal(inst.demands, [1, 0, 4])
        np.testing.assert_array_equal(
            inst.cost_matrix, [[0, 5, 4], [5, 0, 3], [4, 3, 0]]
        )

    def test_public_import_surface(self, vrp_file):
        # Guards against __init__.py regressions: parse_tsplib_file and
        # RoutingInstance must remain importable from divi.qprog.problems.
        inst = parse_tsplib_file_public(vrp_file)
        assert isinstance(inst, RoutingInstance)
        assert inst.dimension == 5


class TestParseTsplibFormats:
    """Coverage for the EWT / EWF dispatch added beyond EUC_2D."""

    @pytest.fixture
    def explicit_lower_diag(self, tmp_path):
        # 4×4 symmetric with diagonal zero. LOWER_DIAG_ROW reads, row by row,
        # entries (i, j) for j <= i: (0)(1,0)(0)(2,0)(2,1)(0)(3,0)(3,1)(3,2)(0).
        body = """\
NAME: tiny
TYPE: TSP
DIMENSION: 4
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: LOWER_DIAG_ROW
EDGE_WEIGHT_SECTION
0
10 0
20 30 0
40 50 60 0
EOF
"""
        return _write_file(tmp_path, body, "tiny_ldr.tsp")

    @pytest.fixture
    def explicit_upper_row(self, tmp_path):
        # Strict upper triangle of a 4×4 symmetric matrix with the same
        # off-diagonal entries as the LOWER_DIAG_ROW fixture.
        body = """\
NAME: tiny
TYPE: TSP
DIMENSION: 4
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: UPPER_ROW
EDGE_WEIGHT_SECTION
10 20 40
30 50
60
EOF
"""
        return _write_file(tmp_path, body, "tiny_ur.tsp")

    @pytest.fixture
    def explicit_lower_row(self, tmp_path):
        # Strict lower triangle (no diagonal) of the same 4×4 matrix.
        body = """\
NAME: tiny
TYPE: TSP
DIMENSION: 4
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: LOWER_ROW
EDGE_WEIGHT_SECTION
10
20 30
40 50 60
EOF
"""
        return _write_file(tmp_path, body, "tiny_lr.tsp")

    @pytest.fixture
    def explicit_upper_diag_row(self, tmp_path):
        # Upper triangle including diagonal of the same 4×4 matrix.
        body = """\
NAME: tiny
TYPE: TSP
DIMENSION: 4
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: UPPER_DIAG_ROW
EDGE_WEIGHT_SECTION
0 10 20 40
0 30 50
0 60
0
EOF
"""
        return _write_file(tmp_path, body, "tiny_udr.tsp")

    @pytest.fixture
    def explicit_full_matrix(self, tmp_path):
        body = """\
NAME: tiny
TYPE: TSP
DIMENSION: 3
EDGE_WEIGHT_TYPE: EXPLICIT
EDGE_WEIGHT_FORMAT: FULL_MATRIX
EDGE_WEIGHT_SECTION
0 7 8
7 0 9
8 9 0
EOF
"""
        return _write_file(tmp_path, body, "tiny_fm.tsp")

    @pytest.fixture
    def geo_burma14(self):
        return Path(__file__).parent / "fixtures" / "burma14.tsp"

    _SYMMETRIC_4X4 = np.array(
        [[0, 10, 20, 40], [10, 0, 30, 50], [20, 30, 0, 60], [40, 50, 60, 0]],
        dtype=float,
    )

    @pytest.mark.parametrize(
        "fixture_name",
        [
            "explicit_lower_diag",
            "explicit_upper_row",
            "explicit_lower_row",
            "explicit_upper_diag_row",
        ],
    )
    def test_explicit_formats_reconstruct_symmetric_matrix(self, fixture_name, request):
        inst = parse_tsplib_file(request.getfixturevalue(fixture_name))
        np.testing.assert_array_equal(inst.cost_matrix, self._SYMMETRIC_4X4)

    def test_explicit_full_matrix(self, explicit_full_matrix):
        inst = parse_tsplib_file(explicit_full_matrix)
        expected = np.array([[0, 7, 8], [7, 0, 9], [8, 9, 0]], dtype=float)
        np.testing.assert_array_equal(inst.cost_matrix, expected)

    def test_unsupported_ewt_raises(self, tmp_path):
        p = _write_file(
            tmp_path,
            "NAME: bad\nTYPE: TSP\nDIMENSION: 3\nEDGE_WEIGHT_TYPE: ATT\n"
            "NODE_COORD_SECTION\n1 0 0\n2 1 0\n3 0 1\nEOF\n",
        )
        with pytest.raises(ValueError, match="Unsupported EDGE_WEIGHT_TYPE"):
            parse_tsplib_file(p)

    def test_unsupported_ewf_raises(self, tmp_path):
        p = _write_file(
            tmp_path,
            "NAME: bad\nTYPE: TSP\nDIMENSION: 3\nEDGE_WEIGHT_TYPE: EXPLICIT\n"
            "EDGE_WEIGHT_FORMAT: WEIRD_FORMAT\nEDGE_WEIGHT_SECTION\n0 1 2 3 4 5\nEOF\n",
        )
        with pytest.raises(ValueError, match="unsupported EDGE_WEIGHT_FORMAT"):
            parse_tsplib_file(p)

    def test_explicit_without_format_raises(self, tmp_path):
        p = _write_file(
            tmp_path,
            "NAME: t\nTYPE: TSP\nDIMENSION: 2\nEDGE_WEIGHT_TYPE: EXPLICIT\n"
            "EDGE_WEIGHT_SECTION\n0 1 1 0\nEOF\n",
        )
        with pytest.raises(
            ValueError, match=r"^EXPLICIT requires EDGE_WEIGHT_FORMAT\.$"
        ):
            parse_tsplib_file(p)

    def test_unpack_explicit_rejects_unknown_format(self):
        with pytest.raises(
            ValueError, match=r"^unsupported EDGE_WEIGHT_FORMAT: WEIRD$"
        ):
            _unpack_explicit([0.0], 1, "WEIRD")

    def test_geo_real_instance(self, geo_burma14):
        inst = parse_tsplib_file(geo_burma14)
        assert inst.dimension == 14
        c = inst.cost_matrix
        assert c.shape == (14, 14)
        assert (c == c.T).all()
        assert (c.diagonal() == 0).all()
        assert (c >= 0).all()
        assert c[0].tolist() == [
            0,
            153,
            510,
            706,
            966,
            581,
            455,
            70,
            160,
            372,
            157,
            567,
            342,
            398,
        ]

    def test_geo_one_degree_along_equator(self, tmp_path):
        p = _write_file(
            tmp_path,
            "NAME: eq\nTYPE: TSP\nDIMENSION: 2\nEDGE_WEIGHT_TYPE: GEO\n"
            "NODE_COORD_SECTION\n1 0.0 0.0\n2 0.0 1.0\nEOF\n",
        )
        assert parse_tsplib_file(p).cost_matrix[0, 1] == 112.0

    @pytest.mark.parametrize(
        "demand_section, expected",
        [("", [0.0, 0.0]), ("DEMAND_SECTION\n1 0\n2 7\n", [0.0, 7.0])],
    )
    def test_tsp_demands(self, tmp_path, demand_section, expected):
        p = _write_file(
            tmp_path,
            "NAME: t\nTYPE: TSP\nDIMENSION: 2\nEDGE_WEIGHT_TYPE: EUC_2D\n"
            f"NODE_COORD_SECTION\n1 0 0\n2 3 4\n{demand_section}EOF\n",
        )
        inst = parse_tsplib_file(p)
        assert inst.problem_type == "TSP"
        np.testing.assert_array_equal(inst.demands, expected)

    def test_geo_does_not_raise_on_identical_points(self, tmp_path):
        body = """\
NAME: degenerate
TYPE: TSP
DIMENSION: 3
EDGE_WEIGHT_TYPE: GEO
NODE_COORD_SECTION
1 49.30 6.10
2 49.30 6.10
3 50.00 6.00
EOF
"""
        inst = parse_tsplib_file(_write_file(tmp_path, body, "degenerate_geo.tsp"))
        # TSPLIB's ``int(R * acos(1) + 1.0)`` is exactly 1.
        assert inst.cost_matrix[0, 1] == 1.0
        assert inst.cost_matrix[0, 2] > 0

    def test_euc2d_uses_half_away_from_zero(self, tmp_path):
        body = """\
NAME: half_int
TYPE: TSP
DIMENSION: 2
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
1 0.0 0.0
2 1.5 2.0
EOF
"""
        inst = parse_tsplib_file(_write_file(tmp_path, body, "half_int.tsp"))
        # Distance is exactly 2.5, which TSPLIB rounds away from zero to 3.
        assert inst.cost_matrix[0, 1] == 3.0

    @pytest.mark.parametrize(
        "comment, expected",
        [
            ('"Optimal cost: 1.5e3"', 1500.0),
            ('"Optimal value: .25"', 0.25),
            ('"Optimal value: -7"', -7.0),
            ('"Optimal cost: 42"', 42.0),
            ('"(Augerat et al, No of trucks: 8, Optimal value: 450)"', 450.0),
            ('"Optimal value : 7"', 7.0),
            ('"Optimal: 9"', 9.0),
            ('"optimal: (see optimal: 12)"', 12.0),
            ('"Optimal: 5 (optimal: 6)"', 5.0),
        ],
    )
    def test_optimal_cost_regex_variants(self, tmp_path, comment, expected):
        p = _write_file(
            tmp_path,
            f"NAME: x\nTYPE: TSP\nDIMENSION: 2\nCOMMENT: {comment}\n"
            "EDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n1 0 0\n2 1 0\nEOF\n",
        )
        assert parse_tsplib_file(p).optimal_cost == expected


@pytest.mark.parametrize(
    "x, expected",
    [
        (2.5, 3),
        (3.5, 4),  # Python round(3.5) -> 4; round(2.5) -> 2 (banker's)
        (-2.5, -3),
        (-3.5, -4),
        (0.0, 0),
        (0.49999, 0),
        (0.5, 1),
        (-0.5, -1),
    ],
)
def test_half_away_from_zero(x, expected):
    """``_nint`` mirrors TSPLIB's half-away-from-zero rounding."""
    assert _nint(x) == expected


class TestParseVrpSolution:
    def test_basic_solution(self, sol_file):
        assert parse_vrp_solution(sol_file) == ([[0, 1, 2, 0], [0, 3, 4, 0]], 100.0)

    def test_missing_cost_defaults_to_zero(self, tmp_path):
        p = _write_file(tmp_path, "Route #1: 2 3\n", "nocost.sol")
        assert parse_vrp_solution(p) == ([[0, 1, 2, 0]], 0.0)

    def test_qoblib_solution(self):
        sol_path = Path(__file__).parent / "fixtures" / "XSH-n20-k4-01.opt.sol"
        routes, cost = parse_vrp_solution(sol_path)
        assert cost == 646.0
        assert len(routes) == 4
        all_customers = set()
        for route in routes:
            all_customers.update(route[1:-1])
        assert len(all_customers) == 20


@pytest.mark.parametrize(
    "n_customers, n_vehicles, max_steps, bits_per_slot, n_slots, n_qubits",
    [
        (0, 1, None, 1, 0, 0),
        (1, 1, None, 1, 1, 1),
        (3, 2, None, 2, 6, 12),
        (20, 4, None, 5, 80, 400),
        (20, 4, 5, 5, 20, 100),
        (20, 4, 6, 5, 24, 120),
        (20, 4, 7, 5, 28, 140),
    ],
)
def test_binary_block_config(
    n_customers, n_vehicles, max_steps, bits_per_slot, n_slots, n_qubits
):
    config = compute_binary_block_config(n_customers, n_vehicles, max_steps=max_steps)
    assert (config.bits_per_slot, config.n_slots, config.n_qubits) == (
        bits_per_slot,
        n_slots,
        n_qubits,
    )


class TestDecodeBinaryCvrp:
    def test_default_depot_and_node_count(self):
        bitstring = "10" + "01" + "00" + "11" + "00" + "00"
        assert decode_binary_cvrp_solution(
            bitstring, compute_binary_block_config(3, 2)
        ) == [
            [0, 1, 2, 0],
            [0, 3, 0],
        ]

    def test_wrong_length(self):
        assert (
            decode_binary_cvrp_solution("010", compute_binary_block_config(3, 2))
            is None
        )


class TestIsValidBinaryCvrp:
    def test_valid_solution(self):
        config = compute_binary_block_config(3, 2)
        demands = np.array([0, 3, 4, 2], dtype=float)
        assert is_valid_binary_cvrp(
            "01" + "10" + "00" + "11" + "00" + "00", config, demands, 10.0, depot=0
        )

    def test_rejects_invalid_solutions(self):
        config = compute_binary_block_config(3, 2)
        demands = np.array([0, 3, 4, 2], dtype=float)
        assert not is_valid_binary_cvrp(
            "01" + "10" + "00" + "00" + "00" + "00", config, demands, 10.0, depot=0
        )
        assert not is_valid_binary_cvrp(
            "01" + "01" + "00" + "11" + "00" + "00", config, demands, 10.0, depot=0
        )
        assert not is_valid_binary_cvrp(
            "01" + "10" + "11" + "00" + "00" + "00", config, demands, 5.0, depot=0
        )

        small_config = compute_binary_block_config(2, 1)
        small_demands = np.array([0, 3, 4], dtype=float)
        assert not is_valid_binary_cvrp(
            "01" + "11", small_config, small_demands, 10.0, depot=0
        )

    @pytest.mark.parametrize(
        "bitstring, capacity, kwargs, expected",
        [
            ("10" + "00" + "11" + "01" + "00" + "00", 5.0, {"depot": 0}, True),
            ("10" + "00" + "11" + "01" + "00" + "00", 4.5, {}, False),
            ("0" * 11, 10.0, {}, False),
        ],
        ids=["load-at-capacity", "load-over-capacity", "short-bitstring"],
    )
    def test_boundaries(self, bitstring, capacity, kwargs, expected):
        demands = np.array([0, 3, 4, 2], dtype=float)
        config = compute_binary_block_config(3, 2)
        assert (
            is_valid_binary_cvrp(bitstring, config, demands, capacity, **kwargs)
            is expected
        )


def test_decode_binary_tsp_solution():
    cfg = compute_binary_block_config(3, 1)
    assert decode_binary_tsp_solution("100111", cfg, 4) == [0, 1, 2, 3, 0]
    assert decode_binary_tsp_solution("0", cfg, 4) is None


def build_tour_bits(tour):
    """One-hot bitstring of a closed tour over local cities ``0..m`` that starts at 0."""
    return "".join(map(str, _tsp_assignment_bits([city - 1 for city in tour[1:-1]])))


def build_binary_tour_bits(tour):
    """Binary-encoded slot bits (2 bits per slot, little-endian) of a 4-city tour from 0."""
    return "".join(f"{city & 1}{city >> 1}" for city in tour[1:-1])


def drop_first_one(bitstring):
    return bitstring.replace("1", "0", 1)


_BOWTIE_BEST_TOUR = compute_best_local_tour(_TWO_BOWTIES[:4, :4])


class TestTSPProblem:
    @pytest.mark.parametrize(
        "cost_fixture, n_free_cities",
        [
            pytest.param("three_city_cost", 2, id="three_cities"),
            pytest.param("four_city_cost", 3, id="four_cities"),
        ],
    )
    def test_init_sizes_one_hot_blocks(self, request, cost_fixture, n_free_cities):
        problem = TSPProblem(request.getfixturevalue(cost_fixture), start_city=0)
        state = problem.recommended_initial_state
        assert state.block_size == n_free_cities
        assert state.n_blocks == n_free_cities
        assert problem.cost_hamiltonian.num_qubits == n_free_cities**2

    def test_feasible_dimension(self, three_city_cost):
        assert TSPProblem(three_city_cost, start_city=0).feasible_dimension == 2

    def test_is_feasible(self, three_city_cost):
        problem = TSPProblem(three_city_cost, start_city=0)
        assert problem.is_feasible("1001") is True
        assert problem.is_feasible("1100") is False

    def test_repair(self, three_city_cost):
        problem = TSPProblem(three_city_cost, start_city=0)
        repaired_bs, _, cost = problem.repair_infeasible_bitstring("0000")
        assert problem.is_feasible(repaired_bs)
        assert cost > 0

    def test_compute_energy(self, three_city_cost):
        problem = TSPProblem(three_city_cost, start_city=0)
        assert problem.compute_energy("1001") == 45.0  # 10 + 20 + 15
        with pytest.raises(ValueError, match="does not encode a valid tour"):
            problem.compute_energy("1100")

    @pytest.mark.parametrize(
        "encoding, bitstring", [("one_hot", "100010001"), ("binary", "100111")]
    )
    def test_compute_energy_non_default_start_city(
        self, four_city_cost, encoding, bitstring
    ):
        problem = TSPProblem(four_city_cost, start_city=2, encoding=encoding)
        assert problem.compute_energy(bitstring) == 80.0

    def test_decode_fn(self, three_city_cost):
        problem = TSPProblem(three_city_cost, start_city=0)
        tour = problem.decode_fn("1001")
        assert tour is not None
        assert tour[0] == 0 and tour[-1] == 0
        assert problem.decode_fn("1100") is None

    @pytest.mark.parametrize(
        "make_optimizer",
        [
            lambda: MonteCarloOptimizer(population_size=3, n_best_sets=1),
            lambda: GridSearchOptimizer(
                param_ranges=[(0, 2 * np.pi), (0, np.pi)], grid_points=3
            ),
        ],
        ids=["monte-carlo", "grid-search"],
    )
    def test_runs_via_qaoa(
        self, three_city_cost, default_test_simulator, make_optimizer
    ):
        qaoa = QAOA(
            TSPProblem(three_city_cost, start_city=0),
            backend=default_test_simulator,
            max_iterations=1,
            n_layers=1,
            optimizer=make_optimizer(),
        )
        qaoa.run()
        assert qaoa.total_circuit_count > 0
        assert len(qaoa.losses_history) == 1

    def test_optimal_has_lowest_energy(self):
        """Verify the QUBO assigns lower energy to the optimal tour."""
        cost = np.array([[0, 1, 10], [1, 0, 10], [10, 10, 0]], dtype=float)
        problem = TSPProblem(cost, start_city=0)
        # Tour 0->1->2->0 costs 1+10+10=21, tour 0->2->1->0 costs 10+10+1=21
        # Both are optimal (symmetric), both should have energy
        e1 = problem.compute_energy("1001")  # city1@t0, city2@t1
        e2 = problem.compute_energy("0110")  # city2@t0, city1@t1
        assert e1 is not None and e2 is not None
        assert e1 == e2  # symmetric cost matrix

    @pytest.mark.parametrize("start_city", [-1, 4])
    def test_out_of_range_start_city_rejected(self, four_city_cost, start_city):
        with pytest.raises(ValueError, match="out of range"):
            TSPProblem(four_city_cost, start_city=start_city)

    @pytest.mark.parametrize(
        "cost_change, kwargs, error, message",
        [
            ((0, 1), {"value": np.nan}, ValueError, "cost_matrix must be finite"),
            ((2, 3), {"value": np.inf}, ValueError, "cost_matrix must be finite"),
            (None, {"encoding": "dense"}, ValueError, "encoding must be"),
            (None, {"start_city": 1.0}, TypeError, "start_city must be an integer"),
            (None, {"start_city": True}, TypeError, "start_city must be an integer"),
        ],
        ids=["nan", "inf", "encoding", "float-start-city", "bool-start-city"],
    )
    def test_invalid_arguments_are_rejected(
        self, four_city_cost, cost_change, kwargs, error, message
    ):
        cost = four_city_cost
        if cost_change is not None:
            cost = replace_entry(cost, cost_change, kwargs.pop("value"))
        with pytest.raises(error, match=message):
            TSPProblem(cost, **kwargs)

    @pytest.mark.parametrize("n_cities", [1, 2])
    def test_fewer_than_three_cities_rejected(self, n_cities):
        with pytest.raises(ValueError, match="at least 3 cities"):
            TSPProblem(np.ones((n_cities, n_cities)) - np.eye(n_cities))

    def test_non_square_cost_rejected(self):
        with pytest.raises(ValueError, match="must be square"):
            TSPProblem(np.ones((3, 4)))

    @pytest.mark.parametrize("encoding", ["one_hot", "binary"])
    @pytest.mark.parametrize("integer_type", [np.int32, np.int64])
    def test_numpy_integer_start_city_is_accepted(
        self, four_city_cost, encoding, integer_type
    ):
        problem = TSPProblem(
            four_city_cost, start_city=integer_type(2), encoding=encoding
        )
        bits = "100010001" if encoding == "one_hot" else "100111"
        assert problem.compute_energy(bits) == 80.0

    @pytest.mark.parametrize("encoding", ["one_hot", "binary"])
    def test_mutating_the_cost_matrix_after_construction_changes_nothing(
        self, four_city_cost, encoding
    ):
        bits = "100010001" if encoding == "one_hot" else "100111"
        cost = four_city_cost.copy()
        problem = TSPProblem(cost, encoding=encoding)
        expected = TSPProblem(four_city_cost, encoding=encoding)
        cost[0, 1] = 999.0
        assert problem.compute_energy(bits) == expected.compute_energy(bits)
        assert problem.penalty_weight == expected.penalty_weight
        n_variables = len(problem.canonical_problem.variable_order)
        point = np.full(n_variables, 0.5)
        assert problem.relaxed_energy(point) == expected.relaxed_energy(point)
        _assert_same_ising(problem, expected, exact=encoding == "one_hot")

    @pytest.mark.parametrize("encoding", ["one_hot", "binary"])
    def test_is_binary_optimization_problem(self, four_city_cost, encoding):
        problem = TSPProblem(four_city_cost, encoding=encoding)
        assert isinstance(problem, BinaryOptimizationProblem)
        assert problem.canonical_problem.variable_order == tuple(
            range(problem.binary_config.n_qubits if encoding == "binary" else 9)
        )
        assert len(problem.constraints) == (6 if encoding == "one_hot" else 0)

    def test_one_hot_constraints_cover_every_step_and_city(self, four_city_cost):
        problem = TSPProblem(four_city_cost)
        groups = {frozenset(c.coefficients) for c in problem.constraints}
        steps = {frozenset(range(t, 9, 3)) for t in range(3)}
        cities = {frozenset(range(3 * i, 3 * i + 3)) for i in range(3)}
        assert groups == steps | cities
        assert {(c.sense, c.bound) for c in problem.constraints} == {("==", 1.0)}

    @pytest.mark.parametrize(
        "encoding, expected", [("one_hot", 80.0), ("binary", 160.0)]
    )
    def test_default_penalty_scales_with_the_nearest_neighbour_tour(
        self, four_city_cost, encoding, expected
    ):
        # Nearest-neighbour tour 0-1-3-2-0 costs 10 + 25 + 30 + 15.
        problem = TSPProblem(four_city_cost, encoding=encoding)
        assert problem.penalty_weight == expected

    def test_default_penalty_follows_the_objective_weight(self, four_city_cost):
        assert TSPProblem(four_city_cost, objective_weight=0.5).penalty_weight == 40.0

    def test_default_penalty_for_all_zero_costs_is_one(self):
        assert TSPProblem(np.zeros((4, 4))).penalty_weight == 1.0

    def test_explicit_penalty_wins(self, four_city_cost):
        assert TSPProblem(four_city_cost, penalty_weight=7.0).penalty_weight == 7.0

    @pytest.mark.parametrize("penalty_weight", [0.0, -1.0, math.inf, math.nan])
    def test_invalid_explicit_penalty_rejected(self, four_city_cost, penalty_weight):
        with pytest.raises(ValueError, match="finite and positive"):
            TSPProblem(four_city_cost, penalty_weight=penalty_weight)

    @pytest.mark.parametrize(
        "kwargs, negative_cost",
        [({"objective_weight": 0.0}, False), ({}, True)],
        ids=["zero-objective-weight", "negative-cost"],
    )
    def test_default_penalty_needs_a_positive_objective_and_costs(
        self, four_city_cost, kwargs, negative_cost
    ):
        cost = four_city_cost.copy()
        if negative_cost:
            cost[0, 1] = cost[1, 0] = -1.0
        with pytest.raises(ValueError, match="penalty_weight must be given"):
            TSPProblem(cost, **kwargs)
        assert TSPProblem(cost, penalty_weight=5.0, **kwargs).penalty_weight == 5.0

    @pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
    @pytest.mark.parametrize(
        "n_cities, start_city", [(4, 0), (4, 2), (4, 3), (5, 0), (5, 2), (5, 4)]
    )
    def test_per_city_one_hot_subspace_ground_state_is_a_valid_tour(
        self, n_cities, start_city, scale
    ):
        """The XY mixer keeps each city in one step, so with the default penalty
        the ground state of the cost Hamiltonian within that subspace must not
        collide two cities."""
        cost = scale * _M5_COST[:n_cities, :n_cities]
        problem = TSPProblem(cost, start_city=start_city)
        m = n_cities - 1
        bits = enumerate_one_hot_per_block_states(m, m)
        energies = compute_diagonal_energies(problem, bits)
        feasible = np.array([problem.is_feasible("".join(map(str, b))) for b in bits])
        assert feasible.sum() == math.factorial(m)
        assert feasible[np.argmin(energies)]
        assert energies[~feasible].min() > energies[feasible].min() + 1e-9 * scale
        others = [c for c in range(n_cities) if c != start_city]
        tour_costs = [
            compute_tour_cost([start_city, *order, start_city], cost)
            for order in itertools.permutations(others)
        ]
        np.testing.assert_allclose(
            np.sort(energies[feasible]) + problem.penalty_weight * m,
            np.sort(tour_costs),
        )

    def test_reduced_and_full_formulations_differ_by_a_constant_on_tours(
        self, four_city_cost
    ):
        problem = TSPProblem(four_city_cost)
        bits = enumerate_one_hot_per_block_states(3, 3)
        tours = bits[[problem.is_feasible("".join(map(str, b))) for b in bits]]
        reduced = compute_diagonal_energies(problem, tours)
        full = problem.relaxed_energy(tours.T.astype(float))
        np.testing.assert_allclose(reduced - full, -problem.penalty_weight * 3)

    def test_qaoa_cost_has_only_the_reduced_penalty_terms(self):
        problem = TSPProblem(_M5_COST)
        args = (_M5_COST, 0, problem.penalty_weight, 1.0)
        reduced = qubo_to_ising(create_tsp_qubo(*args))
        full = qubo_to_ising(create_tsp_qubo(*args, reduced=False))
        assert problem.cost_hamiltonian == reduced.cost_hamiltonian
        assert count_zz_terms(problem.cost_hamiltonian) < count_zz_terms(
            full.cost_hamiltonian
        )


class TestTSPFeasibilityRanking:
    @pytest.fixture
    def qaoa(self, four_city_cost, dummy_simulator, default_optimizer):
        qaoa = QAOA(
            TSPProblem(four_city_cost),
            n_layers=1,
            backend=dummy_simulator,
            optimizer=default_optimizer,
        )
        valid = build_tour_bits([0, 1, 3, 2, 0])
        qaoa._results["best_probs"] = {
            0: {valid: 0.4, drop_first_one(valid): 0.3, "0" * 9: 0.2, "1" * 9: 0.1}
        }
        qaoa._losses_history = [{0: -1.0}]
        return qaoa

    def test_filter_keeps_only_the_sampled_tour(self, qaoa, four_city_cost):
        (entry,) = qaoa.get_top_solutions(feasibility="filter", include_decoded=True)
        assert entry.decoded == [0, 1, 3, 2, 0]
        assert entry.energy == compute_tour_cost(entry.decoded, four_city_cost)
        assert entry.prob == pytest.approx(0.4)

    def test_repair_turns_every_sample_into_a_costed_tour(self, qaoa, four_city_cost):
        top = qaoa.get_top_solutions(feasibility="repair", include_decoded=True)
        assert [e.energy for e in top] == sorted(e.energy for e in top)
        assert sum(e.prob for e in top) == pytest.approx(1.0)
        for entry in top:
            assert qaoa.problem.is_feasible(entry.bitstring)
            assert entry.energy == compute_tour_cost(entry.decoded, four_city_cost)


class TestTSPLazyHamiltonians:
    @pytest.fixture
    def spies(self, mocker):
        return SimpleNamespace(
            reduced=mocker.spy(routing_module, "qubo_to_ising"),
            full=mocker.spy(binary_module, "qubo_to_ising"),
            xy_mixer=mocker.spy(routing_module, "xy_mixer"),
        )

    @pytest.mark.parametrize("encoding", ["one_hot", "binary"])
    def test_construction_builds_no_ising(self, four_city_cost, spies, encoding):
        TSPProblem(four_city_cost, encoding=encoding)
        spies.reduced.assert_not_called()
        spies.full.assert_not_called()

    def test_one_hot_problem_data_needs_no_ising(self, four_city_cost, spies):
        problem = TSPProblem(four_city_cost)
        _ = problem.recommended_initial_state
        _ = problem.decode_fn
        _ = problem.constraints
        _ = problem.canonical_problem
        problem.is_feasible("0" * 9)
        problem.compute_energy("100010001")
        problem.relaxed_energy(np.zeros(9))
        problem._solution_key("0" * 9)
        spies.reduced.assert_not_called()
        spies.full.assert_not_called()

    def test_one_hot_mixer_is_built_once_and_needs_no_ising(
        self, four_city_cost, spies
    ):
        problem = TSPProblem(four_city_cost)
        assert problem.mixer_hamiltonian is problem.mixer_hamiltonian
        spies.xy_mixer.assert_called_once()
        spies.reduced.assert_not_called()
        spies.full.assert_not_called()

    def test_one_hot_cost_builds_only_the_reduced_ising_once(
        self, four_city_cost, spies
    ):
        problem = TSPProblem(four_city_cost)
        for _ in range(2):
            _ = problem.cost_hamiltonian
        spies.reduced.assert_called_once()
        spies.full.assert_not_called()

    def test_full_ising_is_built_separately_on_demand(self, four_city_cost, spies):
        problem = TSPProblem(four_city_cost)
        _ = problem.cost_hamiltonian
        _ = problem.metadata
        spies.reduced.assert_called_once()
        spies.full.assert_called_once()

    def test_binary_cost_builds_the_full_ising_once(self, four_city_cost, spies):
        problem = TSPProblem(four_city_cost, encoding="binary")
        for _ in range(2):
            _ = problem.cost_hamiltonian
        spies.full.assert_called_once()
        spies.reduced.assert_not_called()

    @pytest.mark.parametrize("encoding", ["one_hot", "binary"])
    def test_decompose_builds_no_ising(self, spies, encoding):
        problem = TSPProblem(
            _TWO_BOWTIES, encoding=encoding, max_cities_per_partition=4
        )
        assert len(problem.decompose()) == 2
        spies.reduced.assert_not_called()
        spies.full.assert_not_called()


class TestSplitCitiesIntoClusters:
    def test_recovers_separated_groups(self):
        assert _split_cities_into_clusters(_TWO_BOWTIES, 4) == _TWO_BOWTIES_CLUSTERS

    @pytest.mark.parametrize("seed", range(5))
    @pytest.mark.parametrize("max_size", [3, 4])
    def test_packs_within_max_size(self, seed, max_size):
        cost = build_euclidean_cost(np.random.default_rng(seed).random((9, 2)))
        clusters = _split_cities_into_clusters(cost, max_size)
        assert sorted(c for cluster in clusters for c in cluster) == list(range(9))
        sizes = [len(cluster) for cluster in clusters]
        assert max(sizes) <= max_size
        assert all(a + b > max_size for a, b in itertools.combinations(sizes, 2))
        assert all(cluster == sorted(cluster) for cluster in clusters)
        assert clusters == sorted(clusters, key=lambda cluster: cluster[0])
        assert clusters == _split_cities_into_clusters(cost, max_size)


class TestPatchCyclesIntoTour:
    @pytest.mark.parametrize("cycles", [[], [[0, 1], []]], ids=["none", "empty-cycle"])
    def test_rejects_missing_cities(self, cycles):
        with pytest.raises(ValueError, match="non-empty"):
            _patch_cycles_into_tour(cycles, _TWO_BOWTIES)

    def test_merges_cycles_into_one_covering_tour(self):
        tour = _patch_cycles_into_tour(_TWO_BOWTIES_CLUSTERS, _TWO_BOWTIES)
        assert sorted(tour) == list(range(8))


class TestTSPPartitioning:
    def test_decompose_requires_max_cities(self, three_city_cost):
        with pytest.raises(ValueError, match="max_cities_per_partition was not set"):
            TSPProblem(three_city_cost).decompose()

    def test_max_cities_below_three_rejected(self, four_city_cost):
        with pytest.raises(ValueError, match="must be at least 3"):
            TSPProblem(four_city_cost, max_cities_per_partition=2)

    def test_decompose_skips_clusters_under_three_cities(self):
        cost = build_euclidean_cost([(x, 0) for x in (0, 1, 2, 50, 51, 52, 100)])
        problem = TSPProblem(cost, max_cities_per_partition=3)
        sub_problems = problem.decompose()
        assert sorted(sub_problems) == [("P0", 3), ("P1", 3)]
        assert problem.initial_solution_size() == 7

    def test_decompose_with_cap_at_least_n_gives_one_partition(self, four_city_cost):
        problem = TSPProblem(four_city_cost, max_cities_per_partition=10)
        assert list(problem.decompose()) == [("P0", 4)]

    def test_decompose_rejects_when_no_cluster_has_three_cities(self):
        cost = build_euclidean_cost([(0, 0), (1, 0), (100, 0), (101, 0)])
        problem = TSPProblem(cost, max_cities_per_partition=3)
        with pytest.raises(ValueError, match="No city cluster has three or more"):
            problem.decompose()

    def test_decompose_rejects_asymmetric_cost(self):
        cost = _TWO_BOWTIES.copy()
        cost[0, 1] = 99.0
        problem = TSPProblem(cost, max_cities_per_partition=4)
        with pytest.raises(ValueError, match="must be symmetric"):
            problem.decompose()

    @pytest.mark.parametrize("encoding", ["one_hot", "binary"])
    def test_sub_problems_are_tsp_problems_of_the_same_encoding(self, encoding):
        problem = TSPProblem(
            _TWO_BOWTIES, encoding=encoding, max_cities_per_partition=4
        )
        for sub in problem.decompose().values():
            assert isinstance(sub, TSPProblem)
            assert isinstance(sub, BinaryOptimizationProblem)
            assert sub.encoding == encoding
            assert (sub.binary_config is None) is (encoding == "one_hot")

    def test_default_penalty_is_recomputed_per_sub_problem(self, two_bowties_problem):
        sub_cost = _TWO_BOWTIES[:4, :4]
        for sub in two_bowties_problem.decompose().values():
            assert sub.penalty_weight == TSPProblem(sub_cost).penalty_weight
            assert sub.penalty_weight != two_bowties_problem.penalty_weight

    def test_explicit_penalty_is_forwarded_to_sub_problems(self):
        problem = TSPProblem(
            _TWO_BOWTIES, penalty_weight=7.0, max_cities_per_partition=4
        )
        assert {sub.penalty_weight for sub in problem.decompose().values()} == {7.0}

    def test_extend_solution_records_cluster_positions(self, two_bowties_problem):
        solution = two_bowties_problem.extend_solution(
            [0] * 8, ("P1", 4), [0, 2, 1, 3, 0]
        )
        assert solution == [0, 0, 0, 0, 1, 3, 2, 4]

    def test_extend_solution_without_a_tour_uses_index_order(self, two_bowties_problem):
        solution = two_bowties_problem.extend_solution(
            [0] * 8, ("P1", 4), [0, 2, 1, 3, 0]
        )
        assert two_bowties_problem.extend_solution(solution, ("P1", 4), None) == [
            0,
            0,
            0,
            0,
            1,
            2,
            3,
            4,
        ]

    def test_all_none_candidates_aggregate_to_index_order_tour(
        self, two_bowties_problem
    ):
        fallback = extend_with_tours(two_bowties_problem, lambda _: None)
        assert fallback == [1, 2, 3, 4] * 2
        ((tour, tour_cost),) = two_bowties_problem.postprocess_candidates(
            [(0.0, fallback)]
        )
        assert tour_cost == compute_tour_cost(tour, _TWO_BOWTIES)

    def test_aggregated_tour_is_valid_and_costed(self):
        problem = TSPProblem(_TWO_BOWTIES, start_city=5, max_cities_per_partition=4)
        solution = extend_with_tours(problem, compute_best_local_tour)
        ((tour, tour_cost),) = problem.postprocess_candidates([(0.0, solution)])
        assert tour[0] == tour[-1] == 5
        assert sorted(tour[:-1]) == list(range(8))
        assert tour_cost == compute_tour_cost(tour, _TWO_BOWTIES)
        assert problem.evaluate_global_solution(solution) == tour_cost

    def test_optimal_sub_tours_beat_index_order_fallback(self, two_bowties_problem):
        optimal = extend_with_tours(two_bowties_problem, compute_best_local_tour)
        fallback = extend_with_tours(two_bowties_problem, lambda _: None)
        assert two_bowties_problem.evaluate_global_solution(
            optimal
        ) < two_bowties_problem.evaluate_global_solution(fallback)

    def test_postprocess_returns_distinct_tours_cheapest_first(
        self, two_bowties_problem
    ):
        optimal = extend_with_tours(two_bowties_problem, compute_best_local_tour)
        fallback = extend_with_tours(two_bowties_problem, lambda _: None)
        results = two_bowties_problem.postprocess_candidates(
            [(0.0, fallback), (0.0, optimal), (0.0, optimal)]
        )
        assert len(results) == 2
        assert results[0][1] < results[1][1]

    def test_sub_problem_decodes_an_infeasible_bitstring_to_its_repair(
        self, two_bowties_problem
    ):
        sub = two_bowties_problem.decompose()[("P0", 4)]
        infeasible = drop_first_one(build_tour_bits(_BOWTIE_BEST_TOUR))
        assert sub.is_feasible(infeasible) is False
        assert sub.decode_fn(infeasible) == _BOWTIE_BEST_TOUR
        with pytest.raises(ValueError, match="does not encode a valid tour"):
            sub.compute_energy(infeasible)

    def test_standalone_problem_still_decodes_an_infeasible_bitstring_to_none(self):
        infeasible = drop_first_one(build_tour_bits(_BOWTIE_BEST_TOUR))
        assert TSPProblem(_TWO_BOWTIES[:4, :4]).decode_fn(infeasible) is None

    def test_repaired_candidate_extends_the_global_solution_with_its_tour(
        self, two_bowties_problem
    ):
        sub = two_bowties_problem.decompose()[("P0", 4)]
        repaired = sub.decode_fn(drop_first_one(build_tour_bits(_BOWTIE_BEST_TOUR)))
        solution = two_bowties_problem.extend_solution([0] * 8, ("P0", 4), repaired)
        assert solution == two_bowties_problem.extend_solution(
            [0] * 8, ("P0", 4), _BOWTIE_BEST_TOUR
        )
        assert solution[:4] != [1, 2, 3, 4]

    @pytest.mark.parametrize("damage", [str, drop_first_one], ids=["valid", "repaired"])
    def test_aggregation_follows_the_sampled_sub_tours(
        self, dummy_simulator, two_bowties_problem, damage
    ):
        ensemble = PartitioningProgramEnsemble(
            problem=two_bowties_problem,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            backend=dummy_simulator,
        )
        ensemble.create_programs()
        seed_best_bitstrings(ensemble, [damage(build_tour_bits(_BOWTIE_BEST_TOUR))] * 2)

        tour, tour_cost = ensemble.aggregate_results()

        sampled = extend_with_tours(two_bowties_problem, lambda _: _BOWTIE_BEST_TOUR)
        index_order = extend_with_tours(two_bowties_problem, lambda _: None)
        assert (tour, tour_cost) == two_bowties_problem.postprocess_candidates(
            [(0.0, sampled)]
        )[0]
        assert tour_cost < two_bowties_problem.evaluate_global_solution(index_order)

    def test_binary_sub_problem_keeps_none_for_an_invalid_tour(self):
        problem = TSPProblem(
            _TWO_BOWTIES, encoding="binary", max_cities_per_partition=4
        )
        sub = problem.decompose()[("P0", 4)]
        assert sub.decode_fn("0" * 6) is None
        assert sub.decode_fn(build_binary_tour_bits(_BOWTIE_BEST_TOUR)) == (
            _BOWTIE_BEST_TOUR
        )

    def test_binary_partitioning_aggregates_to_a_valid_tour(self, dummy_simulator):
        problem = TSPProblem(
            _TWO_BOWTIES, encoding="binary", max_cities_per_partition=4
        )
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            backend=dummy_simulator,
        )
        ensemble.create_programs()
        seed_best_bitstrings(
            ensemble, [build_binary_tour_bits(_BOWTIE_BEST_TOUR), "0" * 6]
        )

        tour, tour_cost = ensemble.aggregate_results()

        expected = problem.extend_solution(
            problem.extend_solution([0] * 8, ("P0", 4), _BOWTIE_BEST_TOUR),
            ("P1", 4),
            None,
        )
        assert (tour, tour_cost) == problem.postprocess_candidates([(0.0, expected)])[0]
        assert tour[0] == tour[-1] == 0
        assert sorted(tour[:-1]) == list(range(8))
        assert tour_cost == compute_tour_cost(tour, _TWO_BOWTIES)

    @pytest.mark.parametrize("routine", ["pce", "progressive_pce"])
    def test_partitions_build_pce_programs(
        self, dummy_simulator, two_bowties_problem, routine
    ):
        ensemble = PartitioningProgramEnsemble(
            problem=two_bowties_problem,
            n_layers=1,
            quantum_routine=routine,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            backend=dummy_simulator,
        )
        ensemble.create_programs()
        assert sorted(ensemble.programs) == [("P0", 4), ("P1", 4)]
        assert all(
            isinstance(program, PCE | ProgressivePCE)
            for program in ensemble.programs.values()
        )

    def test_partitioned_run_accepts_checkpointing(
        self, dummy_simulator, two_bowties_problem, tmp_path
    ):
        ensemble = PartitioningProgramEnsemble(
            problem=two_bowties_problem,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=1,
            backend=dummy_simulator,
        )
        ensemble.run(checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path))
        assert (tmp_path / "round_001" / "round_completion.json").is_file()


@pytest.mark.e2e
class TestTSPPartitioningE2E:
    @pytest.mark.parametrize("routine", ["qaoa", "pce"])
    def test_partitioned_ensemble_returns_valid_tour(
        self, default_test_simulator, two_bowties_problem, routine
    ):
        ensemble = PartitioningProgramEnsemble(
            problem=two_bowties_problem,
            n_layers=1,
            quantum_routine=routine,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=3,
            backend=default_test_simulator,
            seed=1997,
        )
        ensemble.run()
        tour, tour_cost = ensemble.aggregate_results()
        assert tour[0] == tour[-1] == 0
        assert sorted(tour[:-1]) == list(range(8))
        assert tour_cost == compute_tour_cost(tour, _TWO_BOWTIES)


class TestTSPProblemPCE:
    @pytest.fixture
    def pce(self, four_city_cost, dummy_simulator, default_optimizer):
        return PCE(
            TSPProblem(four_city_cost),
            n_layers=1,
            backend=dummy_simulator,
            optimizer=default_optimizer,
        )

    def test_encodes_every_assignment_variable(self, pce):
        assert pce.n_vars == 9
        assert pce.problem.penalty_weight == 80.0

    @staticmethod
    def sample_every_state(pce):
        n_states = 2**pce.n_qubits
        pce._results["best_probs"] = {
            0: {format(i, f"0{pce.n_qubits}b"): 1 / n_states for i in range(n_states)}
        }
        pce._losses_history = [{0: -1.0}]

    def test_filter_ranking_drops_states_that_are_not_tours(self, pce):
        self.sample_every_state(pce)
        assert pce.get_top_solutions(feasibility="filter") == []

    def test_repair_ranking_returns_costed_tours_cheapest_first(
        self, pce, four_city_cost
    ):
        self.sample_every_state(pce)

        top = pce.get_top_solutions(feasibility="repair", include_decoded=True)

        assert top
        for entry in top:
            assert pce.problem.is_feasible(entry.bitstring)
            assert entry.decoded == pce.problem.decode_fn(entry.bitstring)
            assert entry.energy == pytest.approx(
                compute_tour_cost(entry.decoded, four_city_cost)
            )
        assert [e.energy for e in top] == sorted(e.energy for e in top)
        assert sum(e.prob for e in top) == pytest.approx(1.0)


class TestTSPProblemBinary:
    """Binary CE-QAOA encoding for TSP — log-encoded slot bits + transverse mixer."""

    def test_qubit_layout(self, four_city_cost):
        # 4 cities, start fixed -> 3 customers, 3 slots, 2 bits/slot = 6 logical qubits.
        problem = TSPProblem(four_city_cost, start_city=0, encoding="binary")
        cfg = problem.binary_config
        assert cfg is not None
        assert cfg.n_customers == 3
        assert cfg.n_vehicles == 1
        assert cfg.max_steps == 3
        assert cfg.bits_per_slot == 2
        assert cfg.n_qubits == 6
        # Quadratization adds ancillas; total physical qubits >= logical.
        assert problem.cost_hamiltonian.num_qubits >= cfg.n_qubits

    def test_initial_state_is_superposition(self, four_city_cost):
        problem = TSPProblem(four_city_cost, encoding="binary")
        assert isinstance(problem.recommended_initial_state, SuperpositionState)

    def test_feasibility_roundtrip(self, four_city_cost):
        # 4 cities, start=0, customers = [1,2,3] -> slot values 1,2,3.
        # Tour 0->1->2->3->0 encodes as slots (1, 2, 3); little-endian bits:
        # slot 0 = "10", slot 1 = "01", slot 2 = "11".
        problem = TSPProblem(four_city_cost, start_city=0, encoding="binary")
        bs = "10" + "01" + "11"
        assert problem.is_feasible(bs) is True
        tour = problem.decode_fn(bs)
        assert tour == [0, 1, 2, 3, 0]

    def test_compute_energy_matches_tour_cost(self, four_city_cost):
        problem = TSPProblem(four_city_cost, start_city=0, encoding="binary")
        bs = "10" + "01" + "11"  # tour 0-1-2-3-0
        # 0->1: 10, 1->2: 35, 2->3: 30, 3->0: 20  => 95
        assert problem.compute_energy(bs) == 95.0

    def test_infeasible_returns_none(self, four_city_cost):
        problem = TSPProblem(four_city_cost, start_city=0, encoding="binary")
        # Slot value 0 = "empty" — not allowed in a TSP tour of length n_cust.
        bs = "00" + "01" + "11"
        assert problem.is_feasible(bs) is False
        with pytest.raises(ValueError, match="does not encode a valid tour"):
            problem.compute_energy(bs)

    def test_repair_not_implemented(self, four_city_cost):
        problem = TSPProblem(four_city_cost, encoding="binary")
        with pytest.raises(
            NotImplementedError,
            match=r"^repair_infeasible_bitstring is not implemented for binary TSP\.$",
        ):
            problem.repair_infeasible_bitstring("000000")

    @pytest.mark.parametrize(
        "kwargs",
        [{}, dict(start_city=2, penalty_weight=3.0, objective_weight=0.5)],
        ids=["defaults", "custom"],
    )
    def test_slot_validity_penalty_counts_out_of_range_slots(self, kwargs):
        cost = np.array(
            [
                [0, 3, 4, 5, 6],
                [3, 0, 7, 8, 9],
                [4, 7, 0, 10, 11],
                [5, 8, 10, 0, 12],
                [6, 9, 11, 12, 0],
            ],
            dtype=float,
        )
        penalty_weight = kwargs.get("penalty_weight", 4.0)
        hubo_tsp, cfg = create_tsp_hubo_binary(cost, **kwargs)
        hubo_cvrp, _ = create_cvrp_hubo_binary(
            cost,
            np.zeros(5),
            1.0,
            1,
            depot=kwargs.get("start_city", 0),
            penalty_weight=penalty_weight,
            objective_weight=kwargs.get("objective_weight", 1.0),
            capacity_penalty_weight=0.0,
            max_steps=4,
        )
        n_out_of_range = (
            _slot_values(_all_bitstrings(cfg.n_qubits), cfg.bits_per_slot) > 4
        ).sum(axis=1)
        np.testing.assert_allclose(
            _hubo_energies(hubo_tsp, cfg.n_qubits)
            - _hubo_energies(hubo_cvrp, cfg.n_qubits),
            penalty_weight * n_out_of_range,
            atol=1e-9,
        )

    def test_no_slot_validity_penalty_when_bit_width_is_tight(self):
        # n=4 cities -> n_cust=3, B=ceil(log2(4))=2 (values 0..3), no invalid values.
        # The TSP HUBO should match the K=1 CVRP HUBO term-for-term.
        cost = np.array(
            [[0, 1, 2, 3], [1, 0, 4, 5], [2, 4, 0, 6], [3, 5, 6, 0]], dtype=float
        )
        hubo_tsp, _ = create_tsp_hubo_binary(cost, start_city=0)
        hubo_cvrp, _ = create_cvrp_hubo_binary(
            cost,
            demands=np.zeros(4, dtype=np.float64),
            capacity=1.0,
            n_vehicles=1,
            depot=0,
            capacity_penalty_weight=0.0,
            max_steps=3,
        )
        assert hubo_tsp == hubo_cvrp


class TestCreateCvrpQubo:
    def test_matrix_values(self):
        Q = create_cvrp_qubo(
            np.array([[0, 1, 2], [3, 0, 4], [5, 6, 0.0]]), n_vehicles=2
        )
        np.testing.assert_array_equal(
            Q,
            [
                [4, -7, 0, 0, 8, -3, 0, 0],
                [0, 7, 0, 0, -3, 8, 0, 0],
                [0, 0, 4, -7, 0, 0, 8, -3],
                [0, 0, 0, 7, 0, 0, -3, 8],
                [0, 0, 0, 0, 7, -9, 0, 0],
                [0, 0, 0, 0, 0, 9, 0, 0],
                [0, 0, 0, 0, 0, 0, 7, -9],
                [0, 0, 0, 0, 0, 0, 0, 9],
            ],
        )

    def test_energy_is_penalties_and_route_cost(self):
        cost = np.array([[0, 1, 2], [3, 0, 4], [5, 6, 0.0]])
        Q = create_cvrp_qubo(cost, n_vehicles=2)

        def reference(x):
            energy = 0.0
            for v in range(2):
                occupancy = x[:, v, :].sum(axis=0)
                energy += sum(4 * s * (s - 1) for s in occupancy)
                energy += 4 * occupancy[1] * (1 - occupancy[0])
                for i in range(2):
                    energy += cost[0, i + 1] * x[i, v, 0]
                    energy += cost[i + 1, 0] * x[i, v, 0] * (1 - occupancy[1])
                    energy += cost[i + 1, 0] * x[i, v, 1]
                    for j in range(2):
                        energy += cost[i + 1, j + 1] * x[i, v, 0] * x[j, v, 1]
            return energy

        def route_cost(x):
            total = 0.0
            for v in range(2):
                steps = [np.flatnonzero(x[:, v, t]) for t in range(2)]
                route = [0, *(int(s[0]) + 1 for s in steps if len(s)), 0]
                total += compute_tour_cost(route, cost)
            return total

        n_compact = 0
        for bits in itertools.product([0, 1], repeat=8):
            x = np.array(bits, dtype=float)
            energy = x @ Q @ x
            grid = x.reshape(2, 2, 2)  # customer, vehicle, step
            assert energy == pytest.approx(reference(grid))
            occupancy = grid.sum(axis=0)
            if np.all(occupancy <= 1) and np.all(occupancy[:, 1] <= occupancy[:, 0]):
                n_compact += 1
                assert energy == pytest.approx(route_cost(grid))
        assert n_compact == 7**2

    def test_energy_of_full_routes_is_the_tour_cost(self):
        rng = np.random.default_rng(3)
        cost = rng.integers(1, 20, (4, 4)).astype(float)
        np.fill_diagonal(cost, 0.0)
        Q = create_cvrp_qubo(cost, n_vehicles=1)
        offsets = set()
        for order in itertools.permutations(range(3)):
            x = np.zeros((3, 3))
            x[order, np.arange(3)] = 1
            x = x.ravel()
            route = [0, *(c + 1 for c in order), 0]
            offsets.add(round(float(x @ Q @ x) - compute_tour_cost(route, cost), 9))
        assert offsets == {0.0}


def _build_cvrp_qubo(cost, _demands):
    return create_cvrp_qubo(cost, n_vehicles=2)


def _build_cvrp_hubo(cost, demands):
    return create_cvrp_hubo_binary(cost, demands, 10.0, n_vehicles=2)


def _build_tsp_qubo(cost, _demands):
    return create_tsp_qubo(cost, start_city=0)


def _build_tsp_hubo(cost, _demands):
    return create_tsp_hubo_binary(cost)


_NON_SQUARE = np.array([[0, 1, 2], [1, 0, 3.0]])


@pytest.mark.parametrize(
    "builder",
    [_build_cvrp_qubo, _build_cvrp_hubo, _build_tsp_qubo, _build_tsp_hubo],
    ids=["cvrp-qubo", "cvrp-hubo", "tsp-qubo", "tsp-hubo"],
)
@pytest.mark.parametrize(
    "cost",
    [_NON_SQUARE, _NON_SQUARE.T, np.zeros(3)],
    ids=["wide", "tall", "one-dimensional"],
)
def test_routing_builders_reject_non_square_cost(builder, cost):
    with pytest.raises(ValueError, match="must be square"):
        builder(cost, np.zeros(len(cost)))


@pytest.mark.parametrize("n_demands", [2, 5], ids=["short", "long"])
def test_cvrp_hubo_builder_rejects_demand_length_mismatch(n_demands):
    with pytest.raises(ValueError, match="demands length"):
        _build_cvrp_hubo(CVRP_COST, np.zeros(n_demands))


@pytest.mark.parametrize("encoding", ["one_hot", "binary"])
@pytest.mark.parametrize("n_demands", [2, 5], ids=["short", "long"])
def test_cvrp_problem_rejects_demand_length_mismatch(encoding, n_demands):
    with pytest.raises(ValueError, match="demands length"):
        CVRPProblem(
            CVRP_COST,
            demands=np.zeros(n_demands),
            capacity=10.0,
            n_vehicles=2,
            encoding=encoding,
        )


def test_basic():
    bs, nb = build_cvrp_block_structure(3, 2)
    assert bs == 6  # 2 vehicles * 3 steps per customer
    assert nb == 3  # one block per customer


@pytest.mark.parametrize(
    "is_valid",
    [
        lambda demands, cap: is_valid_cvrp_solution("1", 1, 1, demands, cap),
        lambda demands, cap: is_valid_binary_cvrp(
            "1", compute_binary_block_config(1, 1), demands, cap
        ),
    ],
    ids=["one-hot", "binary"],
)
@pytest.mark.parametrize("load, expected", [(1e-9, True), (2e-9, False)])
def test_capacity_check_tolerance(is_valid, load, expected):
    assert is_valid(np.array([0.0, load]), 0.0) is expected


_SPLIT_ROUTES_BITS = _cvrp_one_hot_bits([(0, 0, 0), (0, 1, 1), (1, 0, 2)])
_CUSTOMER_PER_VEHICLE_BITS = _cvrp_one_hot_bits([(0, 0, 0), (0, 1, 2), (1, 0, 1)])

_M5_COST = np.array(
    [
        [0, 1, 2, 3, 4],
        [1, 0, 5, 6, 7],
        [2, 5, 0, 8, 9],
        [3, 6, 8, 0, 10],
        [4, 7, 9, 10, 0],
    ],
    dtype=float,
)


class TestCvrpSolutionUtils:
    @pytest.mark.parametrize(
        "bitstring, capacity, expected",
        [
            ("010", 10.0, False),
            (_SPLIT_ROUTES_BITS, 7.0, True),
            (_SPLIT_ROUTES_BITS, 6.0, False),
        ],
        ids=["wrong-length", "load-at-capacity", "load-over-capacity"],
    )
    def test_validity_boundaries(self, bitstring, capacity, expected):
        assert (
            is_valid_cvrp_solution(bitstring, 3, 2, CVRP_DEMANDS, capacity) is expected
        )

    def test_invalid_assignments(self):
        missing_customer = _cvrp_one_hot_bits([(0, 0, 0), (0, 1, 1)])
        assert not is_valid_cvrp_solution(
            missing_customer, 3, 2, CVRP_DEMANDS, 10.0, depot=0
        )

        shared_slot = _cvrp_one_hot_bits([(0, 0, 0), (0, 0, 1), (1, 0, 2)])
        assert not is_valid_cvrp_solution(shared_slot, 3, 2, CVRP_DEMANDS, 10.0)

        capacity_violation = _cvrp_one_hot_bits([(0, 0, 0), (0, 1, 1), (0, 2, 2)])
        assert not is_valid_cvrp_solution(
            capacity_violation, 3, 2, CVRP_DEMANDS, 5.0, depot=0
        )

    def test_decode_wrong_length(self):
        assert decode_cvrp_solution("010", 3, 2, depot=0) is None

    def test_decode_defaults(self):
        assert decode_cvrp_solution(_SPLIT_ROUTES_BITS, 3, 2) == [
            [0, 1, 2, 0],
            [0, 3, 0],
        ]

    def test_repair_moves_overflow_into_tolerance_headroom(self):
        assert repair_cvrp_solution(
            _cvrp_one_hot_bits([(0, 0, 0), (1, 0, 0)], n_customers=2),
            2,
            2,
            np.zeros((3, 3)),
            np.array([0, 1e-9, 1e-9]),
            0.0,
        ) == (
            _cvrp_one_hot_bits([(0, 0, 0), (1, 0, 1)], n_customers=2),
            [[0, 1, 0], [0, 2, 0]],
            0.0,
        )

    def test_repair_feasible_with_defaults(self):
        assert repair_cvrp_solution(
            _SPLIT_ROUTES_BITS, 3, 2, CVRP_COST, CVRP_DEMANDS, 10.0
        ) == (_SPLIT_ROUTES_BITS, [[0, 1, 2, 0], [0, 3, 0]], 90.0)

    @pytest.mark.parametrize(
        "assignments, n_customers, n_vehicles, demands, capacity, repaired, routes, cost",
        [
            (
                [(0, 0, 0), (0, 1, 1), (0, 2, 2)],
                3,
                2,
                [0, 3, 1, 3],
                3.0,
                [(0, 0, 0), (0, 2, 2), (1, 0, 1)],
                [[0, 1, 3, 0], [0, 2, 0]],
                90.0,
            ),
            (
                [(0, 0, 0), (0, 2, 1), (1, 1, 2)],
                3,
                2,
                [0, 4, 2, 1],
                3.0,
                [(0, 0, 0), (1, 0, 1), (1, 1, 2)],
                [[0, 1, 0], [0, 2, 3, 0]],
                67.0,
            ),
            (
                [(0, 0, 0), (1, 0, 1), (1, 2, 2)],
                3,
                2,
                [0, 4, 1, 2],
                2.0,
                [(0, 0, 0), (1, 0, 1), (1, 2, 2)],
                [[0, 1, 0], [0, 2, 3, 0]],
                67.0,
            ),
            (
                [(1, 0, 0), (1, 1, 1), (1, 2, 2)],
                3,
                2,
                [0, 1, 2, 1],
                3.0,
                [(0, 0, 2), (1, 0, 0), (1, 1, 1)],
                [[0, 3, 0], [0, 1, 2, 0]],
                90.0,
            ),
            (
                [(0, 2, 0), (0, 3, 1), (1, 0, 2), (1, 1, 3)],
                4,
                2,
                [0, 3, 1, 3, 1],
                3.0,
                [(0, 2, 0), (0, 3, 1), (1, 0, 2), (1, 1, 3)],
                [[0, 1, 2, 0], [0, 3, 4, 0]],
                25.0,
            ),
            (
                [(0, 1, 0), (0, 3, 1), (1, 1, 2), (1, 3, 3)],
                4,
                3,
                [0, 2, 3, 3, 2],
                4.0,
                [(0, 0, 3), (0, 1, 0), (1, 1, 2), (2, 0, 1)],
                [[0, 4, 1, 0], [0, 3, 0], [0, 2, 0]],
                22.0,
            ),
            (
                [(0, 3, 0), (1, 0, 1), (1, 2, 2), (1, 3, 3)],
                4,
                2,
                [0, 3, 4, 2, 3],
                5.0,
                [(0, 0, 2), (0, 3, 0), (1, 0, 1), (1, 3, 3)],
                [[0, 3, 1, 0], [0, 2, 4, 0]],
                25.0,
            ),
            (
                [(0, 0, 0), (0, 2, 1), (0, 3, 2), (1, 1, 3)],
                4,
                2,
                [0, 3, 2, 1, 3],
                4.0,
                [(0, 0, 0), (0, 2, 1), (1, 0, 2), (1, 1, 3)],
                [[0, 1, 2, 0], [0, 3, 4, 0]],
                25.0,
            ),
        ],
    )
    def test_repair_capacity_reassignment(
        self,
        assignments,
        n_customers,
        n_vehicles,
        demands,
        capacity,
        repaired,
        routes,
        cost,
    ):
        assert repair_cvrp_solution(
            _cvrp_one_hot_bits(assignments, n_vehicles, n_customers),
            n_customers,
            n_vehicles,
            CVRP_COST if n_customers == 3 else _M5_COST,
            np.array(demands, dtype=float),
            capacity,
            depot=0,
        ) == (_cvrp_one_hot_bits(repaired, n_vehicles, n_customers), routes, cost)

    def test_repair_produces_valid(self):
        # All zeros -> infeasible
        bitstring = "0" * 18
        repaired_bs, routes, cost = repair_cvrp_solution(
            bitstring, 3, 2, CVRP_COST, CVRP_DEMANDS, 10.0, depot=0
        )
        assert len(repaired_bs) == 18
        assert routes is not None
        # Every customer served exactly once, within capacity — the decoder wraps
        # each route in the depot regardless, so the wrapper proves nothing.
        assert is_valid_cvrp_solution(repaired_bs, 3, 2, CVRP_DEMANDS, 10.0, depot=0)

    def test_repair_swaps_customers_when_moving_one_cannot_fit(self):
        cost = build_random_cost(np.random.default_rng(0), 4)
        problem = CVRPProblem(
            cost, demands=np.array([0, 2, 1, 3.0]), capacity=3.0, n_vehicles=2
        )
        repaired, routes, repaired_cost = problem.repair_infeasible_bitstring(
            "100000000100100000"
        )
        assert repaired == _cvrp_one_hot_bits([(0, 0, 2), (1, 0, 1), (1, 1, 0)])
        assert routes == [[0, 3, 0], [0, 2, 1, 0]]
        assert repaired_cost == compute_route_cost(routes, cost)

    def test_repair_makes_every_manifold_state_feasible_when_a_packing_exists(self):
        cost = build_random_cost(np.random.default_rng(1), 4)
        demands = np.array([0, 2, 1, 3.0])
        for bits in enumerate_one_hot_per_block_states(3, 6):
            repaired, routes, repaired_cost = repair_cvrp_solution(
                "".join(map(str, bits)), 3, 2, cost, demands, 3.0
            )
            assert is_valid_cvrp_solution(repaired, 3, 2, demands, 3.0)
            assert repaired_cost == pytest.approx(compute_route_cost(routes, cost))

    @pytest.mark.parametrize("seed", range(6))
    def test_repair_is_feasible_exactly_when_a_packing_exists(self, seed):
        rng = np.random.default_rng(seed)
        for _ in range(12):
            n_customers = int(rng.integers(1, 5))
            n_vehicles = int(rng.integers(1, 4))
            cost = build_random_cost(rng, n_customers + 1)
            demands = np.concatenate([[0.0], rng.integers(0, 6, n_customers)])
            capacity = float(rng.integers(1, 9))
            packing_exists = any(
                all(
                    sum(demands[1 + i] for i in range(n_customers) if v_of[i] == v)
                    <= capacity
                    for v in range(n_vehicles)
                )
                for v_of in itertools.product(range(n_vehicles), repeat=n_customers)
            )
            n_bits = n_vehicles * n_customers**2
            for _ in range(40):
                bitstring = "".join(map(str, rng.integers(0, 2, n_bits)))
                repaired, routes, repaired_cost = repair_cvrp_solution(
                    bitstring, n_customers, n_vehicles, cost, demands, capacity
                )
                assert (
                    is_valid_cvrp_solution(
                        repaired, n_customers, n_vehicles, demands, capacity
                    )
                    is packing_exists
                )
                assert repaired_cost == pytest.approx(compute_route_cost(routes, cost))

    def test_repair_leaves_feasible_manifold_states_unchanged(self):
        cost = build_random_cost(np.random.default_rng(2), 4)
        demands = np.array([0, 2, 1, 3.0])
        for bits in enumerate_one_hot_per_block_states(3, 6):
            bitstring = "".join(map(str, bits))
            if is_valid_cvrp_solution(bitstring, 3, 2, demands, 3.0):
                assert (
                    repair_cvrp_solution(bitstring, 3, 2, cost, demands, 3.0)[0]
                    == bitstring
                )

    def test_repair_is_idempotent(self):
        cost = build_random_cost(np.random.default_rng(3), 4)
        demands = np.array([0, 2, 1, 3.0])
        for bits in enumerate_one_hot_per_block_states(3, 6):
            once = repair_cvrp_solution(
                "".join(map(str, bits)), 3, 2, cost, demands, 3.0
            )
            twice = repair_cvrp_solution(once[0], 3, 2, cost, demands, 3.0)
            assert twice == once

    def test_repair_returns_the_over_capacity_result_when_no_packing_exists(self):
        demands = np.array([0, 2, 2, 2.0])
        bitstring = _cvrp_one_hot_bits([(0, 0, 0), (0, 1, 1), (0, 2, 2)])
        repaired, _, _ = repair_cvrp_solution(bitstring, 3, 2, CVRP_COST, demands, 3.0)
        assert not is_valid_cvrp_solution(repaired, 3, 2, demands, 3.0)


class TestCVRPProblem:
    def test_one_hot_construction(self):
        """3 customers × 2 vehicles × 3 steps: one 6-way block per customer."""
        problem = CVRPProblem(
            CVRP_COST,
            demands=CVRP_DEMANDS,
            capacity=6.0,
            n_vehicles=2,
            encoding="one_hot",
        )
        assert problem.cost_hamiltonian.num_qubits == 18
        state = problem.recommended_initial_state
        assert (state.block_size, state.n_blocks) == (6, 3)
        assert (state.block_size, state.n_blocks) == build_cvrp_block_structure(3, 2)

    def test_hamiltonians_are_built_on_first_use(self, mocker):
        spy_ising = mocker.spy(routing_module, "qubo_to_ising")
        spy_mixer = mocker.spy(routing_module, "xy_mixer")
        problem = CVRPProblem(
            CVRP_COST, demands=CVRP_DEMANDS, capacity=6.0, n_vehicles=2
        )
        _ = problem.recommended_initial_state
        _ = problem.decode_fn
        _ = problem.is_feasible("0" * 18)
        spy_ising.assert_not_called()
        spy_mixer.assert_not_called()
        _ = problem.mixer_hamiltonian
        spy_mixer.assert_called_once()
        spy_ising.assert_not_called()
        _ = problem.cost_hamiltonian
        spy_ising.assert_called_once()

    @pytest.mark.parametrize("encoding", ["one_hot", "binary"])
    def test_ising_and_mixer_are_built_once(self, mocker, encoding):
        spy_ising = mocker.spy(routing_module, "qubo_to_ising")
        spy_mixer = mocker.spy(
            routing_module, "xy_mixer" if encoding == "one_hot" else "x_mixer"
        )
        problem = CVRPProblem(
            CVRP_COST,
            demands=CVRP_DEMANDS,
            capacity=6.0,
            n_vehicles=2,
            encoding=encoding,
        )
        spy_ising.assert_not_called()
        for _ in range(2):
            _ = problem.cost_hamiltonian
            _ = problem.mixer_hamiltonian
        spy_ising.assert_called_once()
        spy_mixer.assert_called_once()

    @pytest.mark.parametrize(
        "n_vehicles, capacity", [(1, 9.0), (2, 6.0), (3, 6.0)], ids=["V1", "V2", "V3"]
    )
    def test_reachable_subspace_holds_the_optimal_routing(self, n_vehicles, capacity):
        """W-state support and XY-mixer sectors hold one slot per customer. The
        Hamiltonian ignores capacity, so its minimum over the feasible states of
        that subspace is the optimal capacitated routing, while its overall
        ground state is a compact assignment that may exceed capacity."""
        problem = CVRPProblem(
            CVRP_COST,
            demands=CVRP_DEMANDS,
            capacity=capacity,
            n_vehicles=n_vehicles,
            penalty_weight=60.0,
        )
        bits = enumerate_one_hot_per_block_states(3, 3 * n_vehicles)
        assert len(bits) == (3 * n_vehicles) ** 3
        state = problem.recommended_initial_state
        assert (state.n_blocks, state.block_size) == (3, 3 * n_vehicles)
        energies = compute_diagonal_energies(problem, bits)
        bitstrings = ["".join(map(str, b)) for b in bits]
        feasible = np.array([problem.is_feasible(s) for s in bitstrings])
        assert feasible.any()

        optimum = min(
            sum(compute_tour_cost([0, *route, 0], CVRP_COST) for route in split)
            for split in enumerate_route_splits(CVRP_DEMANDS, capacity, n_vehicles)
        )
        assert energies[feasible].min() == pytest.approx(optimum)

        ground = np.argmin(energies)
        occupancy = bits[ground].reshape(3, n_vehicles, 3).sum(axis=0)
        assert np.all(occupancy <= 1)
        assert np.all(occupancy[:, 1:] <= occupancy[:, :-1])
        assert is_valid_cvrp_solution(
            bitstrings[ground], 3, n_vehicles, CVRP_DEMANDS, math.inf
        )

    def test_over_capacity_assignment_is_infeasible_until_repaired(self):
        problem = CVRPProblem(
            CVRP_COST, demands=CVRP_DEMANDS, capacity=6.0, n_vehicles=2
        )
        over_capacity = _cvrp_one_hot_bits([(0, 0, 0), (0, 1, 1), (0, 2, 2)])
        assert problem.is_feasible(over_capacity) is False
        repaired, _, _ = problem.repair_infeasible_bitstring(over_capacity)
        assert problem.is_feasible(repaired) is True

    def test_solution_key_is_the_decoded_routes(self):
        problem = CVRPProblem(
            CVRP_COST, demands=CVRP_DEMANDS, capacity=10.0, n_vehicles=2
        )
        compact = _cvrp_one_hot_bits([(0, 0, 0), (0, 1, 1), (1, 0, 2)])
        gapped = _cvrp_one_hot_bits([(0, 0, 0), (0, 2, 1), (1, 0, 2)])
        other = _cvrp_one_hot_bits([(0, 0, 0), (1, 0, 1), (1, 1, 2)])
        assert problem._solution_key(compact) == ((0, 1, 2, 0), (0, 3, 0))
        assert problem._solution_key(gapped) == problem._solution_key(compact)
        assert problem._solution_key(other) != problem._solution_key(compact)

    def test_filter_ranking_merges_bitstrings_with_the_same_routes(
        self, dummy_simulator, default_optimizer
    ):
        problem = CVRPProblem(
            CVRP_COST, demands=CVRP_DEMANDS, capacity=10.0, n_vehicles=2
        )
        compact = _cvrp_one_hot_bits([(0, 0, 0), (0, 1, 1), (1, 0, 2)])
        gapped = _cvrp_one_hot_bits([(0, 0, 0), (0, 2, 1), (1, 0, 2)])
        qaoa = QAOA(
            problem, n_layers=1, backend=dummy_simulator, optimizer=default_optimizer
        )
        qaoa._results["best_probs"] = {0: {compact: 0.3, gapped: 0.2}}
        qaoa._losses_history = [{0: -1.0}]

        (merged,) = qaoa.get_top_solutions(feasibility="filter")

        assert merged.prob == pytest.approx(0.5)

    def test_is_feasible(self):
        problem = CVRPProblem(
            CVRP_COST, demands=CVRP_DEMANDS, capacity=6.0, n_vehicles=2
        )
        assert problem.is_feasible(_CUSTOMER_PER_VEHICLE_BITS)
        assert not problem.is_feasible("0" * 18)

    def test_compute_energy(self):
        problem = CVRPProblem(
            CVRP_COST, demands=CVRP_DEMANDS, capacity=6.0, n_vehicles=2
        )
        # Routes 0-1-3-0 (10 + 30 + 20) and 0-2-0 (15 + 15).
        assert problem.compute_energy(_CUSTOMER_PER_VEHICLE_BITS) == 90.0

    def test_decode_fn(self):
        problem = CVRPProblem(
            CVRP_COST, demands=CVRP_DEMANDS, capacity=6.0, n_vehicles=2
        )
        routes = problem.decode_fn(_CUSTOMER_PER_VEHICLE_BITS)
        assert routes is not None
        assert len(routes) == 2

    def test_runs_via_qaoa(self, default_test_simulator):
        problem = CVRPProblem(
            CVRP_COST,
            demands=CVRP_DEMANDS,
            capacity=6.0,
            n_vehicles=2,
        )
        qaoa = QAOA(
            problem,
            backend=default_test_simulator,
            max_iterations=1,
            n_layers=1,
            optimizer=MonteCarloOptimizer(population_size=3, n_best_sets=1),
        )
        qaoa.run()
        assert qaoa.total_circuit_count > 0
        assert len(qaoa.losses_history) == 1

    def test_binary_default_max_steps_matches_n_customers(self):
        problem = CVRPProblem(
            CVRP_COST,
            demands=CVRP_DEMANDS,
            capacity=6.0,
            n_vehicles=2,
            encoding="binary",
        )
        # CVRP_COST has 4 nodes (depot + 3 customers); default worst case
        # is "all customers on one vehicle" → max_steps == n_customers.
        assert problem.binary_config.max_steps == 3

    def test_binary_max_steps_reduces_qubit_count(self):
        baseline = CVRPProblem(
            CVRP_COST,
            demands=CVRP_DEMANDS,
            capacity=6.0,
            n_vehicles=2,
            encoding="binary",
        )
        tightened = CVRPProblem(
            CVRP_COST,
            demands=CVRP_DEMANDS,
            capacity=6.0,
            n_vehicles=2,
            encoding="binary",
            max_steps=2,
        )
        assert tightened.binary_config.max_steps == 2
        # n_qubits scales as n_vehicles * max_steps * bits_per_slot, so
        # tightening from max_steps=3 to max_steps=2 must strictly reduce.
        assert (
            tightened.cost_hamiltonian.num_qubits < baseline.cost_hamiltonian.num_qubits
        )

    @pytest.mark.parametrize("encoding", ["one_hot", "binary"])
    @pytest.mark.parametrize(
        "overrides, error, message",
        [
            (
                {"cost_matrix": replace_entry(CVRP_COST, (0, 1), np.nan)},
                ValueError,
                "cost_matrix must be finite",
            ),
            (
                {"cost_matrix": replace_entry(CVRP_COST, (2, 3), np.inf)},
                ValueError,
                "cost_matrix must be finite",
            ),
            (
                {"cost_matrix": np.zeros((1, 1)), "demands": np.zeros(1)},
                ValueError,
                "at least one customer",
            ),
            ({"encoding": "dense"}, ValueError, "encoding must be"),
            ({"depot": -1}, ValueError, "depot -1 out of range"),
            ({"depot": 4}, ValueError, "depot 4 out of range"),
            ({"depot": 0.0}, TypeError, "depot must be an integer"),
            ({"n_vehicles": 0}, ValueError, "n_vehicles must be at least 1"),
            ({"n_vehicles": 1.5}, TypeError, "n_vehicles must be an integer"),
            ({"capacity": 0.0}, ValueError, "capacity must be finite and positive"),
            ({"capacity": -2.0}, ValueError, "capacity must be finite and positive"),
            ({"capacity": np.nan}, ValueError, "capacity must be finite and positive"),
            ({"capacity": np.inf}, ValueError, "capacity must be finite and positive"),
            (
                {"demands": np.array([0, np.nan, 4, 2.0])},
                ValueError,
                "demands must be finite and non-negative",
            ),
            (
                {"demands": np.array([0, -1, 4, 2.0])},
                ValueError,
                "demands must be finite and non-negative",
            ),
            (
                {"demands": np.array([1, 3, 4, 2.0])},
                ValueError,
                "demand at the depot must be zero",
            ),
            (
                {"demands": np.array([0, 7, 4, 2.0])},
                ValueError,
                "every customer demand must be at most capacity",
            ),
            ({"demands": np.zeros((4, 1))}, ValueError, "must be one-dimensional"),
            ({"penalty_weight": 0.0}, ValueError, "penalty_weight must be finite"),
            ({"penalty_weight": -1.0}, ValueError, "penalty_weight must be finite"),
            ({"penalty_weight": np.nan}, ValueError, "penalty_weight must be finite"),
            ({"penalty_weight": np.inf}, ValueError, "penalty_weight must be finite"),
        ],
    )
    def test_invalid_arguments_are_rejected(self, encoding, overrides, error, message):
        with pytest.raises(error, match=message):
            _make_problem("cvrp", **{"encoding": encoding, **overrides})

    @pytest.mark.parametrize("encoding", ["one_hot", "binary"])
    def test_numpy_integer_depot_and_vehicle_count_are_accepted(self, encoding):
        problem = CVRPProblem(
            CVRP_COST,
            **{
                **_DEPOT_ONE_CVRP,
                "depot": np.int64(1),
                "n_vehicles": np.int32(2),
            },
            encoding=encoding,
        )
        bits = _CUSTOMER_PER_VEHICLE_BITS if encoding == "one_hot" else "101100010000"
        assert problem.compute_energy(bits) == 110.0

    @pytest.mark.parametrize("encoding", ["one_hot", "binary"])
    def test_mutating_the_inputs_after_construction_changes_nothing(self, encoding):
        bits = (
            _CUSTOMER_PER_VEHICLE_BITS
            if encoding == "one_hot"
            else "10" + "00" + "11" + "01" + "00" + "00"
        )
        cost, demands = CVRP_COST.copy(), CVRP_DEMANDS.copy()
        problem = _make_problem(
            "cvrp", encoding=encoding, cost_matrix=cost, demands=demands
        )
        expected = _make_problem("cvrp", encoding=encoding)
        cost[0, 1] = 999.0
        demands[1] = 99.0
        assert expected.is_feasible(bits) is True
        assert problem.is_feasible(bits) is True
        assert problem.compute_energy(bits) == expected.compute_energy(bits)
        assert problem.penalty_weight == expected.penalty_weight
        _assert_same_ising(problem, expected)

    def test_default_penalty_is_the_cost_derived_value_for_one_hot(self):
        # Nearest-neighbour tour 0-1-2-3-0 costs 10 + 25 + 12 + 20; the largest entry is 30.
        problem = _make_problem("cvrp")
        assert problem.penalty_weight == 67.0 + 30.0

    def test_default_penalty_follows_the_objective_weight_and_cost_scale(self):
        base = _make_problem("cvrp").penalty_weight
        assert _make_problem("cvrp", objective_weight=0.5).penalty_weight == base / 2
        scaled = _make_problem("cvrp", cost_matrix=1000.0 * CVRP_COST)
        assert scaled.penalty_weight == pytest.approx(1000.0 * base)

    def test_default_penalty_scales_the_largest_entry_with_the_customer_count(self):
        # Nearest-neighbour tour 0-1-2-3-4-0 costs 1 + 5 + 8 + 10 + 4; the largest entry is 10.
        problem = CVRPProblem(_M5_COST, demands=np.zeros(5), capacity=1.0, n_vehicles=2)
        assert problem.penalty_weight == 28.0 + 2 * 10.0

    def test_default_penalty_for_all_zero_costs_is_the_objective_weight(self):
        problem = CVRPProblem(
            np.zeros((4, 4)), demands=CVRP_DEMANDS, capacity=6.0, n_vehicles=2
        )
        assert problem.penalty_weight == 1.0

    def test_binary_default_penalty_is_fixed(self):
        assert _make_problem("cvrp", encoding="binary").penalty_weight == 4.0
        scaled = _make_problem(
            "cvrp", encoding="binary", cost_matrix=1000.0 * CVRP_COST
        )
        assert scaled.penalty_weight == 4.0

    @pytest.mark.parametrize("encoding", ["one_hot", "binary"])
    def test_explicit_penalty_wins(self, encoding):
        problem = _make_problem("cvrp", encoding=encoding, penalty_weight=7.0)
        assert problem.penalty_weight == 7.0

    @pytest.mark.parametrize(
        "kwargs, negative_cost",
        [({"objective_weight": 0.0}, False), ({}, True)],
        ids=["zero-objective-weight", "negative-cost"],
    )
    def test_default_penalty_needs_a_positive_objective_and_costs(
        self, kwargs, negative_cost
    ):
        cost = replace_entry(CVRP_COST, (1, 2), -1.0) if negative_cost else CVRP_COST
        with pytest.raises(ValueError, match="penalty_weight must be given"):
            _make_problem("cvrp", cost_matrix=cost, **kwargs)
        problem = _make_problem("cvrp", cost_matrix=cost, penalty_weight=5.0, **kwargs)
        assert problem.penalty_weight == 5.0

    @pytest.mark.parametrize(
        "cost, demands, capacity, n_vehicles, n_below_old_default",
        [
            pytest.param(
                [[0, 13, 19], [10, 0, 20], [15, 13, 0]],
                [0, 1, 1],
                10.0,
                1,
                1,
                id="two-customers-one-vehicle",
            ),
            pytest.param(
                [[0, 18, 10], [14, 0, 10], [1, 14, 0]],
                [0, 1, 1],
                10.0,
                2,
                2,
                id="two-customers-two-vehicles",
            ),
            pytest.param(CVRP_COST, CVRP_DEMANDS, 6.0, 2, 66, id="tutorial"),
            pytest.param(
                [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]],
                [0, 3, 4, 2],
                10.0,
                2,
                118,
                id="docs",
            ),
        ],
    )
    def test_default_penalty_makes_the_manifold_ground_state_compact(
        self, cost, demands, capacity, n_vehicles, n_below_old_default
    ):
        """Among states with one slot per customer, a compact assignment is the
        ground state and compact energies are the independent route costs;
        penalty 4.0 left states below the best compact one."""
        n_customers = len(demands) - 1

        def enumerate_manifold(**kwargs):
            problem = CVRPProblem(
                np.array(cost, dtype=float),
                demands=np.array(demands, dtype=float),
                capacity=capacity,
                n_vehicles=n_vehicles,
                **kwargs,
            )
            return problem, *enumerate_cvrp_manifold(problem, n_customers, n_vehicles)

        _, _, old_energies, old_compact = enumerate_manifold(penalty_weight=4.0)
        assert (old_energies < old_energies[old_compact].min() - 1e-9).sum() == (
            n_below_old_default
        )

        problem, bits, energies, compact = enumerate_manifold()
        assert compact[np.argmin(energies)]
        assert energies[~compact].min() > energies[compact].min() + 1e-9
        route_costs = [
            problem.compute_energy("".join(map(str, row))) for row in bits[compact]
        ]
        np.testing.assert_allclose(energies[compact], route_costs)

    def test_one_hot_rejects_max_steps(self):
        with pytest.raises(
            ValueError, match=r"^max_steps is only supported for encoding='binary'\.$"
        ):
            CVRPProblem(
                CVRP_COST,
                demands=CVRP_DEMANDS,
                capacity=6.0,
                n_vehicles=2,
                encoding="one_hot",
                max_steps=2,
            )


def test_cvrp_hubo_energy_matches_penalised_route_cost_on_every_bitstring():
    cost = np.array(
        [[0, 2, 3, 4], [5, 0, 1, 6], [7, 8, 0, 9], [10, 11, 12, 0]], dtype=float
    )
    demands = np.array([0, 0, 0.5, 2])
    hubo, cfg = create_cvrp_hubo_binary(cost, demands, 3.0, 2, max_steps=2)
    assert cfg.n_qubits == 8
    slot_vals = _slot_values(_all_bitstrings(cfg.n_qubits), cfg.bits_per_slot)
    expected = [
        _binary_cvrp_reference_energy(row, cost, demands, 3.0, 2, 2)
        for row in slot_vals
    ]
    np.testing.assert_allclose(_hubo_energies(hubo, cfg.n_qubits), expected, atol=1e-9)


_CVRP_ARGS = dict(demands=CVRP_DEMANDS, capacity=6.0, n_vehicles=2)
_DEFAULT_WEIGHTS = dict(penalty_weight=4.0, objective_weight=1.0)
_CUSTOM_WEIGHTS = dict(penalty_weight=3.0, objective_weight=0.5)


def _make_problem(kind, **kwargs):
    cost = kwargs.pop("cost_matrix", CVRP_COST)
    if kind == "tsp":
        return TSPProblem(cost, **kwargs)
    return CVRPProblem(cost, **{**_CVRP_ARGS, **kwargs})


def _expected_ising(kind, encoding, **kwargs):
    if kind == "tsp":
        builder = create_tsp_qubo if encoding == "one_hot" else create_tsp_hubo_binary
        poly = builder(CVRP_COST, **kwargs)
    elif encoding == "one_hot":
        kwargs.pop("capacity_penalty_weight", None)
        kwargs.pop("demands", None)
        poly = create_cvrp_qubo(CVRP_COST, _CVRP_ARGS["n_vehicles"], **kwargs)
    else:
        poly = create_cvrp_hubo_binary(CVRP_COST, **{**_CVRP_ARGS, **kwargs})
    return qubo_to_ising(poly if encoding == "one_hot" else poly[0])


def _assert_same_ising(problem, expected, *, exact=True):
    if exact:
        assert problem.cost_hamiltonian == expected.cost_hamiltonian
        return
    difference = (problem.cost_hamiltonian - expected.cost_hamiltonian).simplify()
    np.testing.assert_allclose(difference.coeffs, 0.0, atol=1e-9)


_KINDS = pytest.mark.parametrize("kind", ["tsp", "cvrp"])
_ENCODINGS = pytest.mark.parametrize("encoding", ["one_hot", "binary"])


def _construction_extras(kind, custom):
    if kind == "tsp":
        return dict(start_city=1 if custom else 0)
    return (
        dict(
            depot=1,
            demands=_DEPOT_ONE_CVRP["demands"],
            capacity_penalty_weight=2.0,
        )
        if custom
        else dict(depot=0, capacity_penalty_weight=4.0)
    )


@_KINDS
@pytest.mark.parametrize("encoding", [None, "one_hot", "binary"])
@pytest.mark.parametrize("custom", [False, True], ids=["defaults", "custom"])
def test_problem_builds_ising_from_construction_arguments(kind, encoding, custom):
    weights = _CUSTOM_WEIGHTS if custom else _DEFAULT_WEIGHTS
    extra = _construction_extras(kind, custom)
    passed = {**weights, **extra} if custom else {}
    if encoding is not None:
        passed["encoding"] = encoding
    problem = _make_problem(kind, **passed)
    expected_encoding = encoding or "one_hot"
    assert problem.encoding == expected_encoding
    assert (problem.binary_config is None) is (expected_encoding == "one_hot")
    if not custom and (kind == "tsp" or expected_encoding == "one_hot"):
        weights = {**weights, "penalty_weight": problem.penalty_weight}
    _assert_same_ising(
        problem,
        _expected_ising(kind, expected_encoding, **weights, **extra),
        exact=not (kind == "tsp" and expected_encoding == "binary"),
    )


@_KINDS
@_ENCODINGS
def test_problem_mixer(kind, encoding):
    problem = _make_problem(kind, encoding=encoding)
    n = problem.cost_hamiltonian.num_qubits
    if encoding == "binary":
        expected = x_mixer(n)
    else:
        block_size, n_blocks = (3, 3) if kind == "tsp" else (6, 3)
        graph = build_block_xy_mixer_graph(block_size, n_blocks, range(n))
        expected = xy_mixer(graph, n_qubits=n)
        for pauli in problem.mixer_hamiltonian.paulis:
            touched = np.flatnonzero(pauli.x | pauli.z)
            assert len(set(touched // block_size)) == 1
    assert problem.mixer_hamiltonian == expected


_DEPOT_ONE_CVRP = dict(
    demands=np.array([3, 0, 4, 2], dtype=float), capacity=6.0, n_vehicles=2, depot=1
)


def test_binary_cvrp_non_default_depot():
    problem = CVRPProblem(CVRP_COST, encoding="binary", **_DEPOT_ONE_CVRP)
    assert problem.is_feasible("101100010000") is True
    assert problem.is_feasible("100100110000") is False
    assert problem.compute_energy("101100010000") == 110.0


def test_one_hot_cvrp_non_default_depot():
    problem = CVRPProblem(CVRP_COST, **_DEPOT_ONE_CVRP)
    assert problem.is_feasible(_CUSTOMER_PER_VEHICLE_BITS) is True
    assert problem.is_feasible(_SPLIT_ROUTES_BITS) is False
    assert problem.compute_energy(_CUSTOMER_PER_VEHICLE_BITS) == 110.0
