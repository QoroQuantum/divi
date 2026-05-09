# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest

from divi.qprog import QAOA
from divi.qprog.algorithms import SuperpositionState
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
    create_cvrp_hubo_binary,
    create_cvrp_qubo,
    create_tsp_hubo_binary,
    create_tsp_qubo,
    decode_binary_cvrp_solution,
    decode_cvrp_solution,
    decode_tsp_solution,
    is_valid_binary_cvrp,
    is_valid_cvrp_solution,
    repair_cvrp_solution,
    repair_tsp_solution,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def three_city_cost():
    return np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]], dtype=float)


@pytest.fixture
def four_city_cost():
    return np.array(
        [[0, 10, 15, 20], [10, 0, 35, 25], [15, 35, 0, 30], [20, 25, 30, 0]],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# TSP utilities
# ---------------------------------------------------------------------------


class TestCreateTspQubo:
    def test_three_cities(self):
        cost = np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]])
        Q = create_tsp_qubo(cost, start_city=0)
        # 3 cities, fixed start -> 2x2 = 4 qubits
        assert Q.shape == (4, 4)
        assert np.any(np.diag(Q) != 0)  # has linear terms
        assert np.any(np.triu(Q, k=1) != 0)  # has quadratic terms

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            create_tsp_qubo(np.array([[1, 2, 3], [4, 5, 6]]), start_city=0)

    def test_invalid_start_city_raises(self):
        with pytest.raises(ValueError, match="out of range"):
            create_tsp_qubo(np.eye(3), start_city=5)


class TestIsValidTspTour:
    def test_valid_permutation_3cities(self):
        assert is_valid_tsp_tour("1001", 3) is True

    def test_valid_permutation_swap(self):
        assert is_valid_tsp_tour("0110", 3) is True

    def test_invalid_double_assignment(self):
        assert is_valid_tsp_tour("1100", 3) is False

    def test_invalid_no_assignment(self):
        assert is_valid_tsp_tour("0000", 3) is False

    def test_wrong_length(self):
        assert is_valid_tsp_tour("101", 3) is False


class TestDecodeTspSolution:
    def test_identity_3cities(self):
        tour = decode_tsp_solution("1001", 3, start_city=0)
        assert tour is not None
        assert tour[0] == 0 and tour[-1] == 0
        assert set(tour[1:-1]) == {1, 2}

    def test_swap_3cities(self):
        tour = decode_tsp_solution("0110", 3, start_city=0)
        assert tour is not None
        assert tour[0] == 0 and tour[-1] == 0

    def test_infeasible_returns_none(self):
        assert decode_tsp_solution("1100", 3, start_city=0) is None


class TestTourCost:
    def test_simple_tour(self):
        cost = np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]])
        assert tour_cost([0, 1, 2, 0], cost) == 45.0

    def test_reverse_tour(self):
        cost = np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]])
        assert tour_cost([0, 2, 1, 0], cost) == 45.0


class TestRepairTspSolution:
    def test_feasible_unchanged(self):
        cost = np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]])
        repaired_bs, tour, _ = repair_tsp_solution("1001", 3, 0, cost)
        assert is_valid_tsp_tour(repaired_bs, 3)
        assert tour[0] == 0 and tour[-1] == 0

    def test_infeasible_repaired(self):
        cost = np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]])
        repaired_bs, tour, cost_val = repair_tsp_solution("0000", 3, 0, cost)
        assert is_valid_tsp_tour(repaired_bs, 3)
        assert cost_val > 0

    def test_repair_produces_valid_4cities(self):
        cost = np.array(
            [[0, 10, 15, 20], [10, 0, 25, 30], [15, 25, 0, 35], [20, 30, 35, 0]]
        )
        repaired_bs, _, _ = repair_tsp_solution("110100010", 4, 0, cost)
        assert is_valid_tsp_tour(repaired_bs, 4)


# ---------------------------------------------------------------------------
# VRP parser
# ---------------------------------------------------------------------------

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

SAMPLE_SOL = """\
Route #1: 2 3
Route #2: 4 5
Cost 100
"""


@pytest.fixture
def vrp_file(tmp_path):
    p = tmp_path / "test.vrp"
    p.write_text(SAMPLE_VRP)
    return p


@pytest.fixture
def sol_file(tmp_path):
    p = tmp_path / "test.opt.sol"
    p.write_text(SAMPLE_SOL)
    return p


class TestParseVrpFile:
    def test_basic_fields(self, vrp_file):
        inst = parse_tsplib_file(vrp_file)
        assert inst.name == "TEST-n4-k2-01"
        assert inst.problem_type == "CVRP"
        assert inst.dimension == 5
        assert inst.capacity == 10
        assert inst.n_vehicles == 2
        assert inst.depot == 0

    def test_optimal_cost_extracted(self, vrp_file):
        assert parse_tsplib_file(vrp_file).optimal_cost == 100.0

    def test_coordinates(self, vrp_file):
        inst = parse_tsplib_file(vrp_file)
        assert inst.coords.shape == (5, 2)
        np.testing.assert_array_equal(inst.coords[0], [0, 0])

    def test_demands(self, vrp_file):
        inst = parse_tsplib_file(vrp_file)
        assert inst.demands.shape == (5,)
        assert inst.demands[0] == 0

    def test_cost_matrix_euclidean(self, vrp_file):
        inst = parse_tsplib_file(vrp_file)
        assert inst.cost_matrix.shape == (5, 5)
        assert inst.cost_matrix[0, 1] == 3.0
        assert inst.cost_matrix[0, 2] == 4.0
        assert inst.cost_matrix[0, 3] == 5.0
        assert inst.cost_matrix[1, 0] == inst.cost_matrix[0, 1]

    def test_n_customers(self, vrp_file):
        assert parse_tsplib_file(vrp_file).n_customers == 4

    def test_qoblib_instance(self):
        qoblib_path = Path(__file__).parent / "fixtures" / "XSH-n20-k4-01.vrp"
        inst = parse_tsplib_file(qoblib_path)
        assert inst.dimension == 21
        assert inst.n_customers == 20
        assert inst.n_vehicles == 4
        assert inst.capacity == 231
        assert inst.optimal_cost == 646.0

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
        p = tmp_path / "tiny_ldr.tsp"
        p.write_text(body)
        return p

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
        p = tmp_path / "tiny_ur.tsp"
        p.write_text(body)
        return p

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
        p = tmp_path / "tiny_lr.tsp"
        p.write_text(body)
        return p

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
        p = tmp_path / "tiny_udr.tsp"
        p.write_text(body)
        return p

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
        p = tmp_path / "tiny_fm.tsp"
        p.write_text(body)
        return p

    @pytest.fixture
    def geo_burma14(self):
        # Real GEO instance from the TSPLIB95 distribution.
        path = Path(__file__).parent / "fixtures" / "burma14.tsp"
        # Lazy fixture: not all checkouts ship this fixture; skip if missing.
        if not path.exists():
            pytest.skip("burma14.tsp fixture not available")
        return path

    def test_explicit_lower_diag_row(self, explicit_lower_diag):
        inst = parse_tsplib_file(explicit_lower_diag)
        expected = np.array(
            [[0, 10, 20, 40], [10, 0, 30, 50], [20, 30, 0, 60], [40, 50, 60, 0]],
            dtype=float,
        )
        np.testing.assert_array_equal(inst.cost_matrix, expected)

    def test_explicit_upper_row(self, explicit_upper_row):
        inst = parse_tsplib_file(explicit_upper_row)
        expected = np.array(
            [[0, 10, 20, 40], [10, 0, 30, 50], [20, 30, 0, 60], [40, 50, 60, 0]],
            dtype=float,
        )
        np.testing.assert_array_equal(inst.cost_matrix, expected)

    def test_explicit_lower_row(self, explicit_lower_row):
        inst = parse_tsplib_file(explicit_lower_row)
        expected = np.array(
            [[0, 10, 20, 40], [10, 0, 30, 50], [20, 30, 0, 60], [40, 50, 60, 0]],
            dtype=float,
        )
        np.testing.assert_array_equal(inst.cost_matrix, expected)

    def test_explicit_upper_diag_row(self, explicit_upper_diag_row):
        inst = parse_tsplib_file(explicit_upper_diag_row)
        expected = np.array(
            [[0, 10, 20, 40], [10, 0, 30, 50], [20, 30, 0, 60], [40, 50, 60, 0]],
            dtype=float,
        )
        np.testing.assert_array_equal(inst.cost_matrix, expected)

    def test_explicit_full_matrix(self, explicit_full_matrix):
        inst = parse_tsplib_file(explicit_full_matrix)
        expected = np.array([[0, 7, 8], [7, 0, 9], [8, 9, 0]], dtype=float)
        np.testing.assert_array_equal(inst.cost_matrix, expected)

    def test_unsupported_ewt_raises(self, tmp_path):
        p = tmp_path / "bad.tsp"
        p.write_text(
            "NAME: bad\nTYPE: TSP\nDIMENSION: 3\nEDGE_WEIGHT_TYPE: ATT\n"
            "NODE_COORD_SECTION\n1 0 0\n2 1 0\n3 0 1\nEOF\n"
        )
        with pytest.raises(ValueError, match="Unsupported EDGE_WEIGHT_TYPE"):
            parse_tsplib_file(p)

    def test_unsupported_ewf_raises(self, tmp_path):
        p = tmp_path / "bad.tsp"
        p.write_text(
            "NAME: bad\nTYPE: TSP\nDIMENSION: 3\nEDGE_WEIGHT_TYPE: EXPLICIT\n"
            "EDGE_WEIGHT_FORMAT: WEIRD_FORMAT\nEDGE_WEIGHT_SECTION\n0 1 2 3 4 5\nEOF\n"
        )
        with pytest.raises(ValueError, match="EDGE_WEIGHT_FORMAT"):
            parse_tsplib_file(p)

    def test_geo_real_instance(self, geo_burma14):
        inst = parse_tsplib_file(geo_burma14)
        assert inst.dimension == 14
        c = inst.cost_matrix
        assert c.shape == (14, 14)
        assert (c == c.T).all()
        assert (c.diagonal() == 0).all()
        assert (c >= 0).all()

    def test_geo_does_not_raise_on_identical_points(self, tmp_path):
        # acos arg must be clamped to [-1, 1]; without clamp the trig
        # identity can drift past 1.0 by 1 ulp for coincident points.
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
        p = tmp_path / "degenerate_geo.tsp"
        p.write_text(body)
        # Parsing must not raise from an acos domain error on coincident coords.
        inst = parse_tsplib_file(p)
        # TSPLIB's `int(R*acos(arg)+1.0)` for arg=1 yields exactly 1.
        assert inst.cost_matrix[0, 1] == 1.0
        assert inst.cost_matrix[0, 2] > 0

    def test_euc2d_uses_half_away_from_zero(self, tmp_path):
        # Coordinates whose Euclidean distance is exactly 2.5: TSPLIB nint
        # rounds to 3, Python's banker round() returns 2. Catches the bug.
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
        p = tmp_path / "half_int.tsp"
        p.write_text(body)
        inst = parse_tsplib_file(p)
        # sqrt(1.5² + 2.0²) = sqrt(6.25) = 2.5  →  nint(2.5) = 3
        assert inst.cost_matrix[0, 1] == 3.0

    @pytest.mark.parametrize(
        "comment, expected",
        [
            ('"Optimal cost: 1.5e3"', 1500.0),
            ('"Optimal value: .25"', 0.25),
            ('"Optimal value: -7"', -7.0),
            ('"Optimal cost: 42"', 42.0),
            ('"(Augerat et al, No of trucks: 8, Optimal value: 450)"', 450.0),
        ],
    )
    def test_optimal_cost_regex_variants(self, tmp_path, comment, expected):
        p = tmp_path / "opt.tsp"
        p.write_text(
            f"NAME: x\nTYPE: TSP\nDIMENSION: 2\nCOMMENT: {comment}\n"
            "EDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n1 0 0\n2 1 0\nEOF\n"
        )
        assert parse_tsplib_file(p).optimal_cost == expected


class TestNint:
    """``_nint`` mirrors TSPLIB's half-away-from-zero rounding."""

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
    def test_half_away_from_zero(self, x, expected):
        assert _nint(x) == expected


class TestParseVrpSolution:
    def test_basic_solution(self, sol_file):
        routes, cost = parse_vrp_solution(sol_file)
        assert cost == 100.0
        assert len(routes) == 2
        assert all(r[0] == 0 and r[-1] == 0 for r in routes)

    def test_qoblib_solution(self):
        sol_path = Path(__file__).parent / "fixtures" / "XSH-n20-k4-01.opt.sol"
        routes, cost = parse_vrp_solution(sol_path)
        assert cost == 646.0
        assert len(routes) == 4
        all_customers = set()
        for route in routes:
            all_customers.update(route[1:-1])
        assert len(all_customers) == 20


# ---------------------------------------------------------------------------
# Binary encoding
# ---------------------------------------------------------------------------


class TestBinaryBlockConfig:
    def test_small_instance(self):
        config = binary_block_config(3, 2)
        assert config.bits_per_slot == 2
        assert config.n_slots == 6
        assert config.n_qubits == 12

    def test_qoblib_size(self):
        config = binary_block_config(20, 4)
        assert config.bits_per_slot == 5
        assert config.n_slots == 80
        assert config.n_qubits == 400

    def test_qoblib_reduced_steps(self):
        assert binary_block_config(20, 4, max_steps=6).n_qubits == 120

    def test_one_customer(self):
        assert binary_block_config(1, 1).n_qubits == 1


class TestDecodeBinaryCvrp:
    def test_simple_decode(self):
        config = binary_block_config(3, 2)
        bitstring = "10" + "01" + "00" + "11" + "00" + "00"
        routes = decode_binary_cvrp_solution(bitstring, config, depot=0, n_nodes=4)
        assert routes is not None
        assert routes[0] == [0, 1, 2, 0]
        assert routes[1] == [0, 3, 0]

    def test_wrong_length(self):
        assert decode_binary_cvrp_solution("010", binary_block_config(3, 2)) is None


class TestIsValidBinaryCvrp:
    def test_valid_solution(self):
        config = binary_block_config(3, 2)
        demands = np.array([0, 3, 4, 2], dtype=float)
        assert is_valid_binary_cvrp(
            "01" + "10" + "00" + "11" + "00" + "00", config, demands, 10.0, depot=0
        )

    def test_missing_customer(self):
        config = binary_block_config(3, 2)
        demands = np.array([0, 3, 4, 2], dtype=float)
        assert not is_valid_binary_cvrp(
            "01" + "10" + "00" + "00" + "00" + "00", config, demands, 10.0, depot=0
        )

    def test_duplicate_customer(self):
        config = binary_block_config(3, 2)
        demands = np.array([0, 3, 4, 2], dtype=float)
        assert not is_valid_binary_cvrp(
            "01" + "01" + "00" + "11" + "00" + "00", config, demands, 10.0, depot=0
        )

    def test_capacity_violation(self):
        config = binary_block_config(3, 2)
        demands = np.array([0, 3, 4, 2], dtype=float)
        assert not is_valid_binary_cvrp(
            "01" + "10" + "11" + "00" + "00" + "00", config, demands, 5.0, depot=0
        )

    def test_out_of_range_value(self):
        config = binary_block_config(2, 1)
        demands = np.array([0, 3, 4], dtype=float)
        assert not is_valid_binary_cvrp("01" + "11", config, demands, 10.0, depot=0)


class TestBinaryVsOneHotQubitCount:
    def test_qoblib_savings(self):
        assert binary_block_config(20, 4).n_qubits < 1600
        assert binary_block_config(20, 4, max_steps=6).n_qubits == 120

    def test_paper_claim_133_qubits(self):
        assert binary_block_config(20, 4, max_steps=5).n_qubits == 100
        assert binary_block_config(20, 4, max_steps=7).n_qubits == 140


# ---------------------------------------------------------------------------
# TSPProblem
# ---------------------------------------------------------------------------


class TestTSPProblem:
    def test_basic_init(self, three_city_cost):
        problem = TSPProblem(three_city_cost, start_city=0)
        state = problem.recommended_initial_state
        assert state.block_size == 2
        assert state.n_blocks == 2
        assert len(problem.cost_hamiltonian.wires) == 4

    def test_four_cities(self, four_city_cost):
        problem = TSPProblem(four_city_cost, start_city=0)
        state = problem.recommended_initial_state
        assert state.block_size == 3
        assert state.n_blocks == 3
        assert len(problem.cost_hamiltonian.wires) == 9

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
        energy = problem.compute_energy("1001")
        assert energy is not None
        assert energy > 0
        assert problem.compute_energy("1100") is None  # infeasible

    def test_decode_fn(self, three_city_cost):
        problem = TSPProblem(three_city_cost, start_city=0)
        tour = problem.decode_fn("1001")
        assert tour is not None
        assert tour[0] == 0 and tour[-1] == 0
        assert problem.decode_fn("1100") is None

    def test_runs_via_qaoa(self, three_city_cost, default_test_simulator):
        qaoa = QAOA(
            TSPProblem(three_city_cost, start_city=0),
            backend=default_test_simulator,
            max_iterations=1,
            n_layers=1,
            optimizer=MonteCarloOptimizer(population_size=3, n_best_sets=1),
        )
        qaoa.run()
        assert qaoa.total_circuit_count > 0

    def test_with_grid_search(self, three_city_cost, default_test_simulator):
        qaoa = QAOA(
            TSPProblem(three_city_cost, start_city=0),
            backend=default_test_simulator,
            max_iterations=1,
            n_layers=1,
            optimizer=GridSearchOptimizer(
                param_ranges=[(0, 2 * np.pi), (0, np.pi)], grid_points=3
            ),
        )
        qaoa.run()
        assert qaoa.total_circuit_count > 0

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
        assert len(problem.cost_hamiltonian.wires) >= cfg.n_qubits

    def test_initial_state_is_superposition(self, four_city_cost):
        problem = TSPProblem(four_city_cost, encoding="binary")
        assert isinstance(problem.recommended_initial_state, SuperpositionState)

    def test_encoding_property(self, four_city_cost):
        oh = TSPProblem(four_city_cost, encoding="one_hot")
        bn = TSPProblem(four_city_cost, encoding="binary")
        assert oh.encoding == "one_hot"
        assert oh.binary_config is None
        assert bn.encoding == "binary"
        assert bn.binary_config is not None

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
        with pytest.raises(NotImplementedError):
            problem.repair_infeasible_bitstring("000000")

    def test_slot_validity_penalty_added_when_bit_width_overshoots(self):
        # n=6 cities -> n_cust=5, B=ceil(log2(6))=3 (values 0..7), invalid: {6, 7}.
        # CVRP K=1 does not penalize bit patterns that decode to 6 or 7; the TSP
        # HUBO must — so coefficients on the valid-customer indicators differ.
        cost = np.arange(36, dtype=float).reshape(6, 6)
        cost = (cost + cost.T) / 2.0
        np.fill_diagonal(cost, 0.0)

        hubo_tsp, _ = create_tsp_hubo_binary(cost, start_city=0)
        hubo_cvrp, _ = create_cvrp_hubo_binary(
            cost,
            demands=np.zeros(6, dtype=np.float64),
            capacity=1.0,
            n_vehicles=1,
            depot=0,
            capacity_penalty=0.0,
            max_steps=5,
        )
        diff = {
            k
            for k in hubo_tsp.keys() | hubo_cvrp.keys()
            if abs(hubo_tsp.get(k, 0.0) - hubo_cvrp.get(k, 0.0)) > 1e-12
        }
        assert diff, "expected slot-validity penalty to alter HUBO coefficients"

        # An "all-invalid" bitstring (every slot decoded as 7 = '111') should
        # carry strictly higher HUBO energy under TSP than under CVRP-K=1.
        cfg = binary_block_config(5, 1, max_steps=5)
        all_invalid_bits = "111" * cfg.n_slots

        def _eval(hubo, bits):
            # Skip the constant key () — `all([])` is vacuously True, which would
            # spuriously credit the offset to the state-dependent energy.
            energy = 0.0
            for key, coeff in hubo.items():
                if key and all(bits[i] == "1" for i in key):
                    energy += coeff
            return energy

        assert _eval(hubo_tsp, all_invalid_bits) > _eval(hubo_cvrp, all_invalid_bits)

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
            capacity_penalty=0.0,
            max_steps=3,
        )
        assert hubo_tsp == hubo_cvrp


# ---------------------------------------------------------------------------
# CVRPProblem
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CVRP utilities
# ---------------------------------------------------------------------------

CVRP_COST = np.array(
    [[0, 10, 15, 20], [10, 0, 25, 30], [15, 25, 0, 12], [20, 30, 12, 0]],
    dtype=float,
)
CVRP_DEMANDS = np.array([0, 3, 4, 2], dtype=float)


class TestCreateCvrpQubo:
    def test_returns_matrix(self):
        Q = create_cvrp_qubo(CVRP_COST, CVRP_DEMANDS, capacity=6.0, n_vehicles=2)
        assert isinstance(Q, np.ndarray)
        assert Q.ndim == 2
        assert Q.shape[0] == Q.shape[1]

    def test_qubit_count(self):
        Q = create_cvrp_qubo(CVRP_COST, CVRP_DEMANDS, capacity=6.0, n_vehicles=2)
        # 3 customers, 2 vehicles, max_steps=3 -> 2*3*3 = 18 qubits
        assert Q.shape[0] == 18

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            create_cvrp_qubo(np.array([[1, 2, 3]]), np.array([0, 1]), 10.0, 1)

    def test_demands_mismatch_raises(self):
        with pytest.raises(ValueError, match="demands length"):
            create_cvrp_qubo(CVRP_COST, np.array([0, 1]), 10.0, 2)


class TestCvrpBlockStructure:
    def test_basic(self):
        bs, nb = cvrp_block_structure(3, 2)
        assert bs == 3
        assert nb == 6  # 2 vehicles * 3 steps


class TestCvrpSolutionUtils:
    def test_valid_solution(self):
        # 3 customers, 2 vehicles, 18 qubits
        # Vehicle 0: customer 0 at step 0, customer 1 at step 1
        # Vehicle 1: customer 2 at step 0
        bits = np.zeros((2, 3, 3), dtype=int)
        bits[0, 0, 0] = 1  # v0, t0, cust0
        bits[0, 1, 1] = 1  # v0, t1, cust1
        bits[1, 0, 2] = 1  # v1, t0, cust2
        bitstring = "".join(str(x) for x in bits.flatten())
        assert is_valid_cvrp_solution(bitstring, 3, 2, CVRP_DEMANDS, 10.0, depot=0)

    def test_invalid_missing_customer(self):
        bits = np.zeros((2, 3, 3), dtype=int)
        bits[0, 0, 0] = 1
        bits[0, 1, 1] = 1
        # customer 2 not visited
        bitstring = "".join(str(x) for x in bits.flatten())
        assert not is_valid_cvrp_solution(bitstring, 3, 2, CVRP_DEMANDS, 10.0, depot=0)

    def test_invalid_capacity_violation(self):
        # All 3 customers on vehicle 0: demand = 3+4+2 = 9, capacity = 5
        bits = np.zeros((2, 3, 3), dtype=int)
        bits[0, 0, 0] = 1
        bits[0, 1, 1] = 1
        bits[0, 2, 2] = 1
        bitstring = "".join(str(x) for x in bits.flatten())
        assert not is_valid_cvrp_solution(bitstring, 3, 2, CVRP_DEMANDS, 5.0, depot=0)

    def test_decode_wrong_length(self):
        assert decode_cvrp_solution("010", 3, 2, depot=0) is None

    def test_decode_returns_routes(self):
        bits = np.zeros((2, 3, 3), dtype=int)
        bits[0, 0, 0] = 1
        bits[0, 1, 1] = 1
        bits[1, 0, 2] = 1
        bitstring = "".join(str(x) for x in bits.flatten())
        routes = decode_cvrp_solution(bitstring, 3, 2, depot=0, n_nodes=4)
        assert routes is not None
        assert len(routes) == 2
        assert all(r[0] == 0 and r[-1] == 0 for r in routes)

    def test_repair_produces_valid(self):
        # All zeros -> infeasible
        bitstring = "0" * 18
        repaired_bs, routes, cost = repair_cvrp_solution(
            bitstring, 3, 2, CVRP_COST, CVRP_DEMANDS, 10.0, depot=0
        )
        assert len(repaired_bs) == 18
        assert routes is not None
        assert all(r[0] == 0 and r[-1] == 0 for r in routes)


# ---------------------------------------------------------------------------
# CVRPProblem
# ---------------------------------------------------------------------------


class TestCVRPProblem:
    def test_missing_params_raises(self, three_city_cost):
        with pytest.raises(TypeError):
            CVRPProblem(three_city_cost)

    def test_one_hot_construction(self):
        problem = CVRPProblem(
            CVRP_COST,
            demands=CVRP_DEMANDS,
            capacity=6.0,
            n_vehicles=2,
            encoding="one_hot",
        )
        assert problem.cost_hamiltonian is not None
        state = problem.recommended_initial_state
        assert hasattr(state, "block_size")

    def test_binary_construction(self):
        problem = CVRPProblem(
            CVRP_COST,
            demands=CVRP_DEMANDS,
            capacity=6.0,
            n_vehicles=2,
            encoding="binary",
        )
        assert problem.encoding == "binary"
        assert problem.binary_config is not None

    def test_is_feasible(self):
        problem = CVRPProblem(
            CVRP_COST, demands=CVRP_DEMANDS, capacity=6.0, n_vehicles=2
        )
        # Build a valid bitstring: v0 gets cust0+cust2 (demand 3+2=5), v1 gets cust1 (demand 4)
        bits = np.zeros((2, 3, 3), dtype=int)
        bits[0, 0, 0] = 1
        bits[0, 1, 2] = 1
        bits[1, 0, 1] = 1
        valid_bs = "".join(str(x) for x in bits.flatten())
        assert problem.is_feasible(valid_bs)
        assert not problem.is_feasible("0" * 18)

    def test_compute_energy(self):
        problem = CVRPProblem(
            CVRP_COST, demands=CVRP_DEMANDS, capacity=6.0, n_vehicles=2
        )
        bits = np.zeros((2, 3, 3), dtype=int)
        bits[0, 0, 0] = 1
        bits[0, 1, 2] = 1
        bits[1, 0, 1] = 1
        valid_bs = "".join(str(x) for x in bits.flatten())
        energy = problem.compute_energy(valid_bs)
        assert energy is not None
        assert energy > 0

    def test_decode_fn(self):
        problem = CVRPProblem(
            CVRP_COST, demands=CVRP_DEMANDS, capacity=6.0, n_vehicles=2
        )
        bits = np.zeros((2, 3, 3), dtype=int)
        bits[0, 0, 0] = 1
        bits[0, 1, 2] = 1
        bits[1, 0, 1] = 1
        valid_bs = "".join(str(x) for x in bits.flatten())
        routes = problem.decode_fn(valid_bs)
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


# ---------------------------------------------------------------------------
# CVRP HOBO (binary encoding)
# ---------------------------------------------------------------------------


class TestCreateCvrpHuboBinary:
    def test_returns_tuple(self):
        hubo, config = create_cvrp_hubo_binary(
            CVRP_COST, CVRP_DEMANDS, capacity=6.0, n_vehicles=2
        )
        assert isinstance(hubo, dict)
        assert config.bits_per_slot == 2  # ceil(log2(3+1)) = 2
        assert config.n_qubits == config.n_slots * config.bits_per_slot

    def test_has_terms(self):
        hubo, _ = create_cvrp_hubo_binary(
            CVRP_COST, CVRP_DEMANDS, capacity=6.0, n_vehicles=2
        )
        assert len(hubo) > 0
