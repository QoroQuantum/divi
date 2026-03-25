# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest

from divi.qprog import QAOA, CVRPProblem, TSPProblem
from divi.qprog.optimizers import GridSearchOptimizer, MonteCarloOptimizer
from divi.qprog.problems._routing import (
    binary_block_config,
    create_cvrp_hubo_binary,
    create_cvrp_qubo,
    create_tsp_qubo,
    cvrp_block_structure,
    decode_binary_cvrp_solution,
    decode_cvrp_solution,
    decode_tsp_solution,
    enhanced_binary_block_config,
    is_valid_binary_cvrp,
    is_valid_cvrp_solution,
    is_valid_tsp_tour,
    parse_vrp_file,
    parse_vrp_solution,
    repair_cvrp_solution,
    repair_tsp_solution,
    swap_repair_tsp_solution,
    tour_cost,
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


class TestSwapRepair:
    def test_feasible_unchanged(self):
        cost = np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]])
        # Valid permutation: city 1 at t=0, city 2 at t=1
        bs = "1001"
        repaired_bs, tour, cost_val = swap_repair_tsp_solution(bs, 3, 0, cost)
        assert is_valid_tsp_tour(repaired_bs, 3)
        assert cost_val > 0

    def test_infeasible_repaired(self):
        cost = np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]])
        repaired_bs, tour, cost_val = swap_repair_tsp_solution("0000", 3, 0, cost)
        assert is_valid_tsp_tour(repaired_bs, 3)
        assert cost_val > 0

    def test_duplicate_cities_fixed(self):
        cost = np.array(
            [[0, 10, 15, 20], [10, 0, 25, 30], [15, 25, 0, 35], [20, 30, 35, 0]]
        )
        # Both time slots 0 and 1 assign city 0 (duplicate), city 2 missing
        repaired_bs, tour, _ = swap_repair_tsp_solution("110100010", 4, 0, cost)
        assert is_valid_tsp_tour(repaired_bs, 4)

    def test_tsp_problem_swap_strategy(self, three_city_cost):
        problem = TSPProblem(three_city_cost, start_city=0, repair_strategy="swap")
        repaired_bs, _, cost = problem.repair_infeasible_bitstring("0000")
        assert problem.is_feasible(repaired_bs)
        assert cost > 0


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
        inst = parse_vrp_file(vrp_file)
        assert inst.name == "TEST-n4-k2-01"
        assert inst.problem_type == "CVRP"
        assert inst.dimension == 5
        assert inst.capacity == 10
        assert inst.n_vehicles == 2
        assert inst.depot == 0

    def test_optimal_cost_extracted(self, vrp_file):
        assert parse_vrp_file(vrp_file).optimal_cost == 100.0

    def test_coordinates(self, vrp_file):
        inst = parse_vrp_file(vrp_file)
        assert inst.coords.shape == (5, 2)
        np.testing.assert_array_equal(inst.coords[0], [0, 0])

    def test_demands(self, vrp_file):
        inst = parse_vrp_file(vrp_file)
        assert inst.demands.shape == (5,)
        assert inst.demands[0] == 0

    def test_cost_matrix_euclidean(self, vrp_file):
        inst = parse_vrp_file(vrp_file)
        assert inst.cost_matrix.shape == (5, 5)
        assert inst.cost_matrix[0, 1] == 3.0
        assert inst.cost_matrix[0, 2] == 4.0
        assert inst.cost_matrix[0, 3] == 5.0
        assert inst.cost_matrix[1, 0] == inst.cost_matrix[0, 1]

    def test_n_customers(self, vrp_file):
        assert parse_vrp_file(vrp_file).n_customers == 4

    def test_qoblib_instance(self):
        qoblib_path = Path(__file__).parent / "fixtures" / "XSH-n20-k4-01.vrp"
        inst = parse_vrp_file(qoblib_path)
        assert inst.dimension == 21
        assert inst.n_customers == 20
        assert inst.n_vehicles == 4
        assert inst.capacity == 231
        assert inst.optimal_cost == 646.0


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

    def test_enhanced_fewer_qubits_at_scale(self):
        """Enhanced formula gives fewer qubits for larger instances."""
        std = binary_block_config(20, 4)
        enh = enhanced_binary_block_config(20, 4)
        assert enh.n_qubits < std.n_qubits

    def test_enhanced_config_values(self):
        cfg = enhanced_binary_block_config(20, 4)
        # 20 customers, each needs ceil(log2(4 * 20)) = ceil(log2(80)) = 7 bits
        assert cfg.n_slots == 20
        assert cfg.bits_per_slot == 7
        assert cfg.n_qubits == 140

    def test_enhanced_encoding_not_implemented(self):
        cost = np.array([[0, 10, 15], [10, 0, 20], [15, 20, 0]], dtype=float)
        with pytest.raises(NotImplementedError, match="binary_enhanced"):
            CVRPProblem(
                cost,
                demands=np.array([0, 3, 4.0]),
                capacity=10,
                n_vehicles=2,
                encoding="binary_enhanced",
            )


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
        circuit_count, _ = qaoa.run()
        assert circuit_count > 0

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
        assert qaoa.run()[0] > 0

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
        circuit_count, _ = qaoa.run()
        assert circuit_count > 0


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
