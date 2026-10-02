# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import itertools
from pathlib import Path

import numpy as np
import pytest

from divi.hamiltonians import qubo_to_ising, x_mixer, xy_mixer
from divi.qprog import QAOA
from divi.qprog.algorithms import SuperpositionState
from divi.qprog.algorithms._initial_state import build_block_xy_mixer_graph
from divi.qprog.optimizers import GridSearchOptimizer, MonteCarloOptimizer
from divi.qprog.problems import (
    CVRPProblem,
    RoutingInstance,
    TSPProblem,
    binary_block_config,
    cvrp_block_structure,
    is_valid_tsp_tour,
)
from divi.qprog.problems import parse_tsplib_file
from divi.qprog.problems import parse_tsplib_file as parse_tsplib_file_public
from divi.qprog.problems import (
    parse_vrp_solution,
    tour_cost,
)
from divi.qprog.problems._routing import (
    _nint,
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
    """One-hot CVRP bitstring with ``x[v, t, i] = 1`` for each ``(v, t, i)``."""
    bits = np.zeros((n_vehicles, n_customers, n_customers), dtype=int)
    for v, t, i in assignments:
        bits[v, t, i] = 1
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
            [[6, 8, 0, 20], [0, 6, 20, 0], [0, 0, 11, 8], [0, 0, 0, 11]],
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
        assert x @ Q @ x == tour_cost(tour, four_city_cost) - n_penalties * 4.0 * 3

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
    assert tour_cost([0, 1, 2, 0], three_city_cost) == 45.0
    assert tour_cost([0, 2, 1, 0], three_city_cost) == 45.0


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
    config = binary_block_config(n_customers, n_vehicles, max_steps=max_steps)
    assert (config.bits_per_slot, config.n_slots, config.n_qubits) == (
        bits_per_slot,
        n_slots,
        n_qubits,
    )


class TestDecodeBinaryCvrp:
    def test_default_depot_and_node_count(self):
        bitstring = "10" + "01" + "00" + "11" + "00" + "00"
        assert decode_binary_cvrp_solution(bitstring, binary_block_config(3, 2)) == [
            [0, 1, 2, 0],
            [0, 3, 0],
        ]

    def test_wrong_length(self):
        assert decode_binary_cvrp_solution("010", binary_block_config(3, 2)) is None


class TestIsValidBinaryCvrp:
    def test_valid_solution(self):
        config = binary_block_config(3, 2)
        demands = np.array([0, 3, 4, 2], dtype=float)
        assert is_valid_binary_cvrp(
            "01" + "10" + "00" + "11" + "00" + "00", config, demands, 10.0, depot=0
        )

    def test_rejects_invalid_solutions(self):
        config = binary_block_config(3, 2)
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

        small_config = binary_block_config(2, 1)
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
        config = binary_block_config(3, 2)
        assert (
            is_valid_binary_cvrp(bitstring, config, demands, capacity, **kwargs)
            is expected
        )


def test_decode_binary_tsp_solution():
    cfg = binary_block_config(3, 1)
    assert decode_binary_tsp_solution("100111", cfg, 4) == [0, 1, 2, 3, 0]
    assert decode_binary_tsp_solution("0", cfg, 4) is None


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
        assert problem.compute_energy("1100") is None  # infeasible

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
        assert problem.compute_energy(bs) is None

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
            np.array([[0, 1, 2], [3, 0, 4], [5, 6, 0.0]]),
            np.array([0, 1, 2.0]),
            capacity=2.0,
            n_vehicles=2,
        )
        np.testing.assert_array_equal(
            Q,
            [
                [-15, 16, 16, 20, 8, 0, 8, 0],
                [0, -18, 22, 40, 0, 8, 0, 8],
                [0, 0, -13, 16, 8, 0, 8, 0],
                [0, 0, 0, -15, 0, 8, 0, 8],
                [0, 0, 0, 0, -15, 16, 16, 20],
                [0, 0, 0, 0, 0, -18, 22, 40],
                [0, 0, 0, 0, 0, 0, -13, 16],
                [0, 0, 0, 0, 0, 0, 0, -15],
            ],
        )

    def test_energy_of_full_routes_is_tour_cost_plus_a_constant(self):
        rng = np.random.default_rng(3)
        cost = rng.integers(1, 20, (4, 4)).astype(float)
        np.fill_diagonal(cost, 0.0)
        Q = create_cvrp_qubo(cost, np.array([0, 1, 1, 1.0]), 10.0, n_vehicles=1)
        offsets = set()
        for order in itertools.permutations(range(3)):
            x = np.zeros((3, 3))
            x[np.arange(3), order] = 1
            x = x.ravel()
            route = [0, *(c + 1 for c in order), 0]
            offsets.add(round(float(x @ Q @ x) - tour_cost(route, cost), 9))
        assert len(offsets) == 1


def _build_cvrp_qubo(cost, demands):
    return create_cvrp_qubo(cost, demands, 10.0, n_vehicles=2)


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


@pytest.mark.parametrize(
    "builder", [_build_cvrp_qubo, _build_cvrp_hubo], ids=["cvrp-qubo", "cvrp-hubo"]
)
@pytest.mark.parametrize("n_demands", [2, 5], ids=["short", "long"])
def test_cvrp_builders_reject_demand_length_mismatch(builder, n_demands):
    with pytest.raises(ValueError, match="demands length"):
        builder(CVRP_COST, np.zeros(n_demands))


def test_basic():
    bs, nb = cvrp_block_structure(3, 2)
    assert bs == 3
    assert nb == 6  # 2 vehicles * 3 steps


@pytest.mark.parametrize(
    "is_valid",
    [
        lambda demands, cap: is_valid_cvrp_solution("1", 1, 1, demands, cap),
        lambda demands, cap: is_valid_binary_cvrp(
            "1", binary_block_config(1, 1), demands, cap
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
            "10001000", 2, 2, np.zeros((3, 3)), np.array([0, 1e-9, 1e-9]), 0.0
        ) == ("10000100", [[0, 1, 0], [0, 2, 0]], 0.0)

    def test_repair_feasible_with_defaults(self):
        assert repair_cvrp_solution(
            _SPLIT_ROUTES_BITS, 3, 2, CVRP_COST, CVRP_DEMANDS, 10.0
        ) == ("100010000001000000", [[0, 1, 2, 0], [0, 3, 0]], 90.0)

    @pytest.mark.parametrize(
        "bitstring, n_customers, n_vehicles, demands, capacity, expected",
        [
            (
                "100010001000000000",
                3,
                2,
                [0, 3, 1, 3],
                3.0,
                ("100000001010000000", [[0, 1, 3, 0], [0, 2, 0]], 90.0),
            ),
            (
                "100000010000001000",
                3,
                2,
                [0, 4, 2, 1],
                3.0,
                ("100000000010001000", [[0, 1, 0], [0, 2, 3, 0]], 67.0),
            ),
            (
                "100000000010000001",
                3,
                2,
                [0, 4, 1, 2],
                2.0,
                ("100000000010000001", [[0, 1, 0], [0, 2, 3, 0]], 67.0),
            ),
            (
                "000000000100010001",
                3,
                2,
                [0, 1, 2, 1],
                3.0,
                ("001000000100010000", [[0, 3, 0], [0, 1, 2, 0]], 90.0),
            ),
            (
                "00000000100001000010000100000000",
                4,
                2,
                [0, 3, 1, 3, 1],
                3.0,
                (
                    "00000000100001000010000100000000",
                    [[0, 1, 2, 0], [0, 3, 4, 0]],
                    25.0,
                ),
            ),
            (
                "000010000000010000000010000000010000000000000000",
                4,
                3,
                [0, 2, 3, 3, 2],
                4.0,
                (
                    "000110000000000000000010000000000100000000000000",
                    [[0, 4, 1, 0], [0, 3, 0], [0, 2, 0]],
                    22.0,
                ),
            ),
            (
                "00000000000010000100000000100001",
                4,
                2,
                [0, 3, 4, 2, 3],
                5.0,
                (
                    "00100000000010000100000000000001",
                    [[0, 3, 1, 0], [0, 2, 4, 0]],
                    25.0,
                ),
            ),
            (
                "10000000010000100000000100000000",
                4,
                2,
                [0, 3, 2, 1, 3],
                4.0,
                (
                    "10000000010000000010000100000000",
                    [[0, 1, 2, 0], [0, 3, 4, 0]],
                    25.0,
                ),
            ),
        ],
    )
    def test_repair_capacity_reassignment(
        self, bitstring, n_customers, n_vehicles, demands, capacity, expected
    ):
        cost = CVRP_COST if n_customers == 3 else _M5_COST
        assert (
            repair_cvrp_solution(
                bitstring,
                n_customers,
                n_vehicles,
                cost,
                np.array(demands, dtype=float),
                capacity,
                depot=0,
            )
            == expected
        )

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


class TestCVRPProblem:
    def test_one_hot_construction(self):
        """3 customers × 2 vehicles × 3 steps: one 3-way block per (vehicle, step)."""
        problem = CVRPProblem(
            CVRP_COST,
            demands=CVRP_DEMANDS,
            capacity=6.0,
            n_vehicles=2,
            encoding="one_hot",
        )
        assert problem.cost_hamiltonian.num_qubits == 18
        state = problem.recommended_initial_state
        assert (state.block_size, state.n_blocks) == cvrp_block_structure(3, 2)

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
    if kind == "tsp":
        return TSPProblem(CVRP_COST, **kwargs)
    return CVRPProblem(CVRP_COST, **_CVRP_ARGS, **kwargs)


def _expected_ising(kind, encoding, **kwargs):
    if kind == "tsp":
        builder = create_tsp_qubo if encoding == "one_hot" else create_tsp_hubo_binary
        poly = builder(CVRP_COST, **kwargs)
    else:
        builder = create_cvrp_qubo if encoding == "one_hot" else create_cvrp_hubo_binary
        poly = builder(CVRP_COST, **_CVRP_ARGS, **kwargs)
    return qubo_to_ising(poly if encoding == "one_hot" else poly[0])


def _assert_same_ising(problem, expected):
    assert problem.cost_hamiltonian == expected.cost_hamiltonian
    assert problem.loss_constant == expected.loss_constant


_KINDS = pytest.mark.parametrize("kind", ["tsp", "cvrp"])
_ENCODINGS = pytest.mark.parametrize("encoding", ["one_hot", "binary"])


def _construction_extras(kind, custom):
    if kind == "tsp":
        return dict(start_city=1 if custom else 0)
    return (
        dict(depot=1, capacity_penalty_weight=2.0)
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
    _assert_same_ising(
        problem, _expected_ising(kind, expected_encoding, **weights, **extra)
    )


@_KINDS
@_ENCODINGS
def test_problem_mixer(kind, encoding):
    problem = _make_problem(kind, encoding=encoding)
    n = problem.cost_hamiltonian.num_qubits
    if encoding == "binary":
        expected = x_mixer(n)
    else:
        block_size, n_blocks = (3, 3) if kind == "tsp" else cvrp_block_structure(3, 2)
        graph = build_block_xy_mixer_graph(block_size, n_blocks, range(n))
        expected = xy_mixer(graph, n_qubits=n)
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
