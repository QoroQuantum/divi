# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import dimod
import hybrid
import numpy as np
import pennylane as qp
import pytest
import scipy.sparse as sps

import divi.qprog.problems._binary as binary_module
from divi.backends import CircuitRunner
from divi.hamiltonians import BinaryPolynomialProblem
from divi.qprog import (
    PCE,
    QAOA,
    BatchConfig,
    MonteCarloOptimizer,
    ScipyMethod,
    ScipyOptimizer,
)
from divi.qprog.algorithms import GenericLayerAnsatz
from divi.qprog.checkpointing import CheckpointConfig
from divi.qprog.problems import BinaryOptimizationProblem
from divi.qprog.problems._binary import _merge_substates, _sanitize_problem_input
from divi.qprog.workflows import PartitioningProgramEnsemble
from tests.qprog.problems._helpers import (
    HUBO_CUBIC,
    QUBO_MATRIX,
    QUBO_SOLUTION,
    exact_hubo_minima,
    make_bqm_maximize,
    make_bqm_minimize,
)
from tests.qprog.qprog_contracts import verify_metacircuit_dict

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_qubo_matrix():
    """Provides a simple 4x4 QUBO matrix for testing."""
    return np.array(
        [
            [-1, 2, 0, 0],
            [2, -1, 1, 0],
            [0, 1, -1, 3],
            [0, 0, 3, -1],
        ]
    )


# ---------------------------------------------------------------------------
# BinaryOptimizationProblem construction / validation
# ---------------------------------------------------------------------------


class TestBinaryOptimizationProblem:
    def test_invalid_hamiltonian_builder_raises(self):
        with pytest.raises(
            ValueError,
            match="hamiltonian_builder must be either 'native' or 'quadratized'",
        ):
            BinaryOptimizationProblem(
                QUBO_MATRIX.tolist(),
                hamiltonian_builder="custom",  # type: ignore[arg-type]
            )

    def test_non_square_qubo_fails(self):
        with pytest.raises(
            ValueError,
            match=r"Invalid QUBO matrix\. Got array of shape \(3, 2\)\. Must be a square matrix\.",
        ):
            BinaryOptimizationProblem(np.array([[1, 2], [3, 4], [5, 6]]))

    def test_non_symmetrical_qubo_normalizes_problem(self):
        # [[1, 2], [3, 4]] → diagonal: {(0,): 1, (1,): 4}, off-diag summed: {(0,1): 2+3=5}
        test_array = np.array([[1, 2], [3, 4]])
        expected_terms = {(0,): 1.0, (1,): 4.0, (0, 1): 5.0}

        qubo_problem = BinaryOptimizationProblem(test_array)

        assert isinstance(qubo_problem.canonical_problem, BinaryPolynomialProblem)
        assert qubo_problem.canonical_problem.terms == expected_terms

        # Sparse input produces the same result
        qubo_problem_sparse = BinaryOptimizationProblem(sps.csc_matrix(test_array))
        assert qubo_problem_sparse.canonical_problem.terms == expected_terms

    def test_binary_quadratic_model_with_spin_raises_error(self):
        # Create a BQM with SPIN vartype (non-binary)
        bqm = dimod.BinaryQuadraticModel(
            {"x": -1, "y": -2, "z": 3}, {}, 0.0, dimod.Vartype.SPIN
        )

        with pytest.raises(
            ValueError,
            match=r"BinaryQuadraticModel must have vartype='BINARY', got Vartype\.SPIN",
        ):
            BinaryOptimizationProblem(bqm)


class TestLazyIsingInit:
    """The Ising conversion and X-mixer are computed lazily on first access.

    For workflows like ``QUBOPartitioningQAOA`` the parent's full Ising is
    never used after decomposition, so eager construction was wasted work.
    These tests pin both the laziness and the per-property memoization.
    """

    def test_constructor_does_not_call_qubo_to_ising(self, mocker):
        spy = mocker.spy(binary_module, "qubo_to_ising")
        BinaryOptimizationProblem(QUBO_MATRIX)
        spy.assert_not_called()

    def test_constructor_does_not_build_x_mixer(self, mocker):
        spy = mocker.spy(binary_module.pqaoa, "x_mixer")
        BinaryOptimizationProblem(QUBO_MATRIX)
        spy.assert_not_called()

    def test_cost_hamiltonian_triggers_ising_build(self, mocker):
        spy = mocker.spy(binary_module, "qubo_to_ising")
        problem = BinaryOptimizationProblem(QUBO_MATRIX)
        spy.assert_not_called()

        _ = problem.cost_hamiltonian
        spy.assert_called_once()

    def test_ising_cached_across_dependent_properties(self, mocker):
        """All properties backed by the Ising share one ``qubo_to_ising`` call."""
        spy = mocker.spy(binary_module, "qubo_to_ising")
        problem = BinaryOptimizationProblem(QUBO_MATRIX)

        _ = problem.cost_hamiltonian
        _ = problem.loss_constant
        _ = problem.metadata
        _ = problem.decode_fn
        _ = problem.mixer_hamiltonian  # also touches Ising for n_qubits

        spy.assert_called_once()

    def test_mixer_cached_independently(self, mocker):
        spy = mocker.spy(binary_module.pqaoa, "x_mixer")
        problem = BinaryOptimizationProblem(QUBO_MATRIX)

        first = problem.mixer_hamiltonian
        second = problem.mixer_hamiltonian
        assert first is second
        spy.assert_called_once()

    def test_lazy_init_passes_constructor_kwargs(self, mocker):
        """Builder + quadratization strength are forwarded on the deferred call."""
        spy = mocker.spy(binary_module, "qubo_to_ising")
        problem = BinaryOptimizationProblem(
            QUBO_MATRIX,
            hamiltonian_builder="quadratized",
            quadratization_strength=7.5,
        )
        _ = problem.cost_hamiltonian

        assert spy.call_args.kwargs["hamiltonian_builder"] == "quadratized"
        assert spy.call_args.kwargs["quadratization_strength"] == 7.5


# ---------------------------------------------------------------------------
# _sanitize_problem_input
# ---------------------------------------------------------------------------


class TestSanitizeProblemInput:
    def test_with_bqm_input(self):
        bqm = dimod.BinaryQuadraticModel({"a": 1}, {("a", "b"): -1}, "BINARY")
        orig, sanitized = _sanitize_problem_input(bqm)
        assert orig is bqm
        assert sanitized is bqm

    def test_with_numpy_array(self, sample_qubo_matrix):
        orig, bqm = _sanitize_problem_input(sample_qubo_matrix)
        assert orig is sample_qubo_matrix
        assert isinstance(bqm, dimod.BinaryQuadraticModel)
        assert len(bqm.variables) == 4

    @pytest.mark.parametrize(
        "fmt",
        [sps.coo_matrix, sps.csr_matrix, sps.csc_matrix, sps.lil_matrix],
        ids=["COO", "CSR", "CSC", "LIL"],
    )
    def test_with_sparse_matrix(self, sample_qubo_matrix, fmt):
        sparse_matrix = fmt(sample_qubo_matrix)
        orig, bqm = _sanitize_problem_input(sparse_matrix)
        assert orig is sparse_matrix
        assert isinstance(bqm, dimod.BinaryQuadraticModel)
        assert len(bqm.variables) == 4

    def test_raises_on_non_square_matrix(self):
        non_square = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError, match="Only square matrices"):
            _sanitize_problem_input(non_square)

    def test_raises_on_unsupported_type(self):
        unsupported = {"a": 1, "b": 2}
        with pytest.raises(ValueError, match="Got an unsupported QUBO input format"):
            _sanitize_problem_input(unsupported)


# ---------------------------------------------------------------------------
# evaluate_global_solution
# ---------------------------------------------------------------------------


class TestEvaluateSolution:
    """Tests for BinaryOptimizationProblem.evaluate_global_solution."""

    @staticmethod
    def _make_problem(qubo):
        """Create a minimal BinaryOptimizationProblem for unit-testing only."""
        decomposer = hybrid.EnergyImpactDecomposer(size=2)
        return BinaryOptimizationProblem(qubo, decomposer=decomposer)

    def test_known_qubo_optimal(self):
        """Verify energy for the known optimal solution [1,1,0,0]."""
        qubo = {
            (0, 0): -0.5,
            (1, 1): 1,
            (0, 1): -2,
            (2, 2): 1,
            (3, 3): 1,
            (2, 3): 2,
        }
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
        problem = self._make_problem(bqm)
        energy = problem.evaluate_global_solution([1, 1, 0, 0])
        assert energy == pytest.approx(-1.5)

    def test_all_zeros(self):
        """All-zero solution has energy 0 for a QUBO with no constant offset."""
        qubo = {
            (0, 0): -0.5,
            (1, 1): 1,
            (0, 1): -2,
        }
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
        problem = self._make_problem(bqm)
        energy = problem.evaluate_global_solution([0, 0])
        assert energy == pytest.approx(0.0)

    def test_diagonal_qubo(self):
        """Diagonal QUBO (only linear terms): energy = sum of selected biases."""
        qubo = np.diag([-1.0, 2.0, -3.0])
        problem = self._make_problem(qubo)
        # x = [1,0,1] -> energy = -1 + 0 + (-3) = -4
        energy = problem.evaluate_global_solution([1, 0, 1])
        assert energy == pytest.approx(-4.0)

    def test_lower_energy_is_better(self):
        """Verify that optimal solution has lowest energy."""
        qubo = {
            (0, 0): -0.5,
            (1, 1): 1,
            (0, 1): -2,
            (2, 2): 1,
            (3, 3): 1,
            (2, 3): 2,
        }
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
        problem = self._make_problem(bqm)
        optimal = problem.evaluate_global_solution([1, 1, 0, 0])
        suboptimal = problem.evaluate_global_solution([1, 0, 0, 0])
        all_ones = problem.evaluate_global_solution([1, 1, 1, 1])
        assert optimal < suboptimal
        assert optimal < all_ones


# ---------------------------------------------------------------------------
# _merge_substates
# ---------------------------------------------------------------------------


class TestMergeSubstates:
    """Tests for _merge_substates helper."""

    def test_merges_two_sample_sets(self):
        """_merge_substates horizontally stacks subsamples from two states.

        hstack_samplesets merges variables from both states into one sample,
        not adding rows.
        """
        ss1 = dimod.SampleSet.from_samples({"a": 0}, "BINARY", -1.0)
        ss2 = dimod.SampleSet.from_samples({"b": 1}, "BINARY", -0.5)

        state1 = hybrid.State(subsamples=ss1)
        state2 = hybrid.State(subsamples=ss2)

        merged = _merge_substates(None, (state1, state2))

        # hstack merges variables: result has both 'a' and 'b'
        merged_vars = set(merged.subsamples.variables)
        assert "a" in merged_vars
        assert "b" in merged_vars


# ---------------------------------------------------------------------------
# decompose
# ---------------------------------------------------------------------------


class TestDecomposeQUBO:
    @staticmethod
    def _make_problem(qubo, decomposer=None):
        return BinaryOptimizationProblem(qubo, decomposer=decomposer)

    def test_decompose_returns_sub_problems(self, sample_qubo_matrix):
        decomposer = hybrid.EnergyImpactDecomposer(size=2)
        problem = self._make_problem(sample_qubo_matrix, decomposer=decomposer)

        sub_problems = problem.decompose()

        assert len(sub_problems) >= 1
        for sub_problem in sub_problems.values():
            assert isinstance(sub_problem, BinaryOptimizationProblem)

    def test_decompose_populates_variable_maps(self, sample_qubo_matrix):
        decomposer = hybrid.EnergyImpactDecomposer(size=2)
        problem = self._make_problem(sample_qubo_matrix, decomposer=decomposer)

        problem.decompose()

        assert len(problem._variable_maps) >= 1
        for prog_id in problem._variable_maps:
            assert isinstance(problem._variable_maps[prog_id], list)

    def test_decompose_raises_without_decomposer(self, sample_qubo_matrix):
        problem = self._make_problem(sample_qubo_matrix)

        with pytest.raises(ValueError, match="Cannot decompose"):
            problem.decompose()

    def test_decompose_identifies_trivial_subproblems(self):
        trivial_qubo = np.array(
            [
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, -1],
            ]
        )
        decomposer = hybrid.EnergyImpactDecomposer(size=2)
        problem = self._make_problem(trivial_qubo, decomposer=decomposer)

        problem.decompose()

        assert len(problem._trivial_program_ids) >= 1


# ---------------------------------------------------------------------------
# initial_solution_size
# ---------------------------------------------------------------------------


class TestInitialSolutionSizeQUBO:
    def test_returns_number_of_variables(self, sample_qubo_matrix):
        decomposer = hybrid.EnergyImpactDecomposer(size=2)
        problem = BinaryOptimizationProblem(sample_qubo_matrix, decomposer=decomposer)

        assert problem.initial_solution_size() == 4


# ---------------------------------------------------------------------------
# extend_solution
# ---------------------------------------------------------------------------


class TestExtendSolutionQUBO:
    """Tests for BinaryOptimizationProblem.extend_solution."""

    @staticmethod
    def _make_problem(qubo):
        decomposer = hybrid.EnergyImpactDecomposer(size=2)
        return BinaryOptimizationProblem(qubo, decomposer=decomposer)

    def test_maps_local_bits_to_global_positions(self):
        """Candidate's decoded bits appear at the correct global indices."""
        qubo = {
            (0, 0): -0.5,
            (1, 1): 1,
            (0, 1): -2,
            (2, 2): 1,
            (3, 3): 1,
            (2, 3): 2,
        }
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
        problem = self._make_problem(bqm)
        problem.decompose()

        prog_id = list(problem._variable_maps.keys())[0]
        global_indices = problem._variable_maps[prog_id]
        n_local = len(global_indices)

        decoded = np.ones(n_local, dtype=np.int32)
        result = problem.extend_solution([0, 0, 0, 0], prog_id, decoded)

        # Only the positions mapped by this program should be set to 1
        for local_idx, global_idx in enumerate(global_indices):
            assert result[global_idx] == 1
        # Other positions should remain 0
        other_positions = set(range(4)) - set(global_indices)
        for idx in other_positions:
            assert result[idx] == 0

    def test_does_not_mutate_input(self):
        """extend_solution returns a new list, not a mutation of the input."""
        qubo = np.diag([-1.0, -1.0, -1.0])
        problem = self._make_problem(qubo)
        problem.decompose()

        prog_id = list(problem._variable_maps.keys())[0]
        global_indices = problem._variable_maps[prog_id]
        n_local = len(global_indices)

        original = [0, 0, 0]
        decoded = np.ones(n_local, dtype=np.int32)
        result = problem.extend_solution(original, prog_id, decoded)

        assert result is not original
        assert original == [0, 0, 0]

    def test_overwrites_previous_partition_values(self):
        """Extending with zeros overwrites previous ones at mapped positions."""
        qubo = {
            (0, 0): -0.5,
            (1, 1): 1,
            (0, 1): -2,
            (2, 2): 1,
            (3, 3): 1,
            (2, 3): 2,
        }
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
        problem = self._make_problem(bqm)
        problem.decompose()

        prog_id = list(problem._variable_maps.keys())[0]
        global_indices = problem._variable_maps[prog_id]
        n_local = len(global_indices)

        # Start with all-ones solution
        decoded_zeros = np.zeros(n_local, dtype=np.int32)
        result = problem.extend_solution([1, 1, 1, 1], prog_id, decoded_zeros)

        # Mapped positions should now be 0
        for global_idx in global_indices:
            assert result[global_idx] == 0
        # Other positions should remain 1
        other_positions = set(range(4)) - set(global_indices)
        for idx in other_positions:
            assert result[idx] == 1


# ---------------------------------------------------------------------------
# _compose_solution
# ---------------------------------------------------------------------------


class TestComposeSolutionQUBO:
    """Tests for BinaryOptimizationProblem._compose_solution."""

    @staticmethod
    def _make_problem(qubo):
        decomposer = hybrid.EnergyImpactDecomposer(size=2)
        return BinaryOptimizationProblem(qubo, decomposer=decomposer)

    def test_optimal_solution_has_correct_energy(self):
        """_compose_solution returns (solution, energy) matching the QUBO energy."""
        qubo = {
            (0, 0): -0.5,
            (1, 1): 1,
            (0, 1): -2,
            (2, 2): 1,
            (3, 3): 1,
            (2, 3): 2,
        }
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
        problem = self._make_problem(bqm)
        problem.decompose()

        # [1,1,0,0] is the optimal solution with energy -1.5
        solution_array, energy = problem._compose_solution([1, 1, 0, 0])

        assert isinstance(solution_array, np.ndarray)
        assert energy == pytest.approx(-1.5)
        np.testing.assert_array_equal(solution_array, [1, 1, 0, 0])

    def test_all_zeros_returns_zero_energy(self):
        """All-zero solution has zero energy for a QUBO with no constant offset."""
        qubo = {
            (0, 0): -0.5,
            (1, 1): 1,
            (0, 1): -2,
        }
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
        problem = self._make_problem(bqm)
        problem.decompose()

        solution_array, energy = problem._compose_solution([0, 0])

        assert energy == pytest.approx(0.0)

    def test_does_not_mutate_subproblem_states(self):
        """_compose_solution operates on copies and doesn't alter shared state."""
        qubo = {
            (0, 0): -0.5,
            (1, 1): 1,
            (0, 1): -2,
            (2, 2): 1,
            (3, 3): 1,
            (2, 3): 2,
        }
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
        problem = self._make_problem(bqm)
        problem.decompose()

        # Snapshot subproblem state identities before call
        state_ids_before = {k: id(v) for k, v in problem._bqm_subproblem_states.items()}

        problem._compose_solution([1, 1, 0, 0])

        # Original state objects should not have been replaced
        for k, v in problem._bqm_subproblem_states.items():
            assert id(v) == state_ids_before[k]


# ---------------------------------------------------------------------------
# finalize_solution
# ---------------------------------------------------------------------------


class TestFinalizeSolutionQUBO:
    def test_returns_array_and_energy(self, sample_qubo_matrix):
        decomposer = hybrid.EnergyImpactDecomposer(size=2)
        problem = BinaryOptimizationProblem(sample_qubo_matrix, decomposer=decomposer)
        problem.decompose()

        result = problem.finalize_solution(-1.0, [1, 1, 0, 0])

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], float)


# ---------------------------------------------------------------------------
# format_top_solutions
# ---------------------------------------------------------------------------


class TestFormatTopSolutionsQUBO:
    def test_formats_and_sorts_by_energy(self, sample_qubo_matrix):
        decomposer = hybrid.EnergyImpactDecomposer(size=2)
        problem = BinaryOptimizationProblem(sample_qubo_matrix, decomposer=decomposer)
        problem.decompose()

        results = [(-1.0, [1, 1, 0, 0]), (-0.5, [0, 0, 1, 1])]
        formatted = problem.format_top_solutions(results)

        assert len(formatted) == 2
        # Should be sorted by energy (second element of each tuple)
        energies = [entry[1] for entry in formatted]
        assert energies == sorted(energies)


# ---------------------------------------------------------------------------
# QAOA + BinaryOptimizationProblem integration tests
# ---------------------------------------------------------------------------

QUBO_FORMATS_TO_TEST = {
    "argvalues": [
        QUBO_MATRIX.tolist(),
        QUBO_MATRIX,
        sps.csc_matrix(QUBO_MATRIX),
        sps.csr_matrix(QUBO_MATRIX),
        sps.coo_matrix(QUBO_MATRIX),
        sps.lil_matrix(QUBO_MATRIX),
    ],
    "ids": ["List", "Numpy", "CSC", "CSR", "COO", "LIL"],
}


class TestQUBOInput:
    """Test suite for QUBO problem inputs in QAOA."""

    @pytest.mark.parametrize("input_qubo", **QUBO_FORMATS_TO_TEST)
    def test_qubo_basic_initialization(self, input_qubo, default_test_simulator):
        qubo_problem = BinaryOptimizationProblem(input_qubo)
        qaoa_problem = QAOA(
            qubo_problem,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=10,
            backend=default_test_simulator,
        )

        assert isinstance(qaoa_problem.backend, CircuitRunner)
        assert qaoa_problem.backend.shots == 5000
        assert isinstance(qaoa_problem.optimizer, ScipyOptimizer)
        assert qaoa_problem.optimizer.method == ScipyMethod.NELDER_MEAD
        assert qaoa_problem.max_iterations == 10
        assert isinstance(qaoa_problem.problem, BinaryOptimizationProblem)
        assert qaoa_problem.n_layers == 1
        assert isinstance(qubo_problem.canonical_problem, BinaryPolynomialProblem)
        assert qubo_problem.canonical_problem.n_vars == 3

        # Hand-computed from QUBO_MATRIX [[-3,4,0],[0,2,0],[0,0,-3]]
        assert qubo_problem.canonical_problem.variable_order == (0, 1, 2)
        assert qubo_problem.canonical_problem.terms == {
            (0,): -3.0,
            (1,): 2.0,
            (2,): -3.0,
            (0, 1): 4.0,
        }

        assert len(qaoa_problem.cost_hamiltonian) == 4
        assert all(
            isinstance(op, (qp.Z, qp.ops.Prod))
            for op in qaoa_problem.cost_hamiltonian.terms()[1]
        )
        assert len(qaoa_problem.problem.mixer_hamiltonian) == 3
        assert all(
            isinstance(op, qp.X)
            for op in qaoa_problem.problem.mixer_hamiltonian.terms()[1]
        )

        verify_metacircuit_dict(qaoa_problem, ["cost_circuit", "meas_circuit"])

    def test_hubo_dict_initialization_native_builder(self, dummy_simulator):
        hubo = {
            ("x0",): -1.0,
            ("x0", "x1"): 2.0,
            ("x0", "x1", "x2"): 1.5,
            (): 0.25,
        }

        qubo_problem = BinaryOptimizationProblem(hubo)
        qaoa_problem = QAOA(
            qubo_problem,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=dummy_simulator,
        )

        assert isinstance(qaoa_problem.problem, BinaryOptimizationProblem)
        assert qaoa_problem.n_qubits == 3
        assert qubo_problem.canonical_problem.n_vars == 3
        assert qaoa_problem.problem_metadata["strategy"] == "native"

    def test_hubo_dict_quadratized_builder_adds_ancillas(self, dummy_simulator):
        hubo = {("x0", "x1", "x2"): 1.0}
        qubo_problem = BinaryOptimizationProblem(
            hubo,
            hamiltonian_builder="quadratized",
            quadratization_strength=5.0,
        )
        qaoa_problem = QAOA(
            qubo_problem,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=dummy_simulator,
        )

        assert qaoa_problem.problem_metadata["strategy"] == "quadratized"
        assert qaoa_problem.problem_metadata["ancilla_count"] >= 1
        assert qaoa_problem.n_qubits >= qubo_problem.canonical_problem.n_vars

    def test_quadratized_decode_excludes_ancillas(self, mocker, dummy_simulator):
        hubo = {("x0", "x1", "x2"): 1.0}
        qubo_problem = BinaryOptimizationProblem(
            hubo,
            hamiltonian_builder="quadratized",
            quadratization_strength=5.0,
        )
        qaoa_problem = QAOA(
            qubo_problem,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=dummy_simulator,
        )

        best_bitstring = "1" * qaoa_problem.n_qubits
        qaoa_problem._best_probs = {"0_NoMitigation:0_0": {best_bitstring: 1.0}}
        mocker.patch.object(qaoa_problem, "_run_solution_measurement_for")

        qaoa_problem._perform_final_computation()

        sol = qaoa_problem.solution
        assert isinstance(sol, dict)
        assert set(sol.keys()) == set(qubo_problem.canonical_problem.variable_order)
        assert all(v == 1 for v in sol.values())

    @pytest.mark.e2e
    @pytest.mark.parametrize("builder", ["native", "quadratized"])
    def test_hubo_e2e_recovers_exact_minimum(self, builder, default_test_simulator):
        default_test_simulator.set_seed(1997)
        _, exact_minima = exact_hubo_minima(HUBO_CUBIC, n_vars=3)

        qaoa_problem = QAOA(
            BinaryOptimizationProblem(
                HUBO_CUBIC,
                hamiltonian_builder=builder,
                quadratization_strength=5.0,
            ),
            n_layers=2,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=18,
            backend=default_test_simulator,
            seed=1997,
        )

        qaoa_problem.run()
        assert any(np.array_equal(qaoa_problem.solution, x) for x in exact_minima)

    @pytest.mark.e2e
    def test_qubo_returns_correct_solution(self, default_test_simulator):
        default_test_simulator.set_seed(1997)

        qaoa_problem = QAOA(
            BinaryOptimizationProblem(QUBO_MATRIX),
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=12,
            backend=default_test_simulator,
            seed=1997,
        )

        qaoa_problem.run()

        np.testing.assert_equal(qaoa_problem.solution, QUBO_SOLUTION)

    @pytest.mark.e2e
    def test_qubo_e2e_checkpointing_resume(self, default_test_simulator, tmp_path):
        """Test QAOA QUBO e2e with checkpointing and resume functionality."""
        optimizer = MonteCarloOptimizer(population_size=10, n_best_sets=3)

        checkpoint_dir = tmp_path / "checkpoint_test"
        default_test_simulator.set_seed(1997)

        qubo_problem = BinaryOptimizationProblem(QUBO_MATRIX)

        qaoa_problem1 = QAOA(
            qubo_problem,
            n_layers=1,
            optimizer=optimizer,
            max_iterations=6,
            backend=default_test_simulator,
            seed=1997,
        )

        qaoa_problem1.run(
            checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir)
        )

        assert checkpoint_dir.exists()
        checkpoint_path = checkpoint_dir / "checkpoint_006"
        assert checkpoint_path.exists()
        assert (checkpoint_path / "program_state.json").exists()

        first_run_iteration = qaoa_problem1.current_iteration
        first_run_losses_count = len(qaoa_problem1.losses_history)

        qaoa_problem2 = QAOA.load_state(
            checkpoint_dir,
            backend=default_test_simulator,
            problem=qubo_problem,
            n_layers=1,
        )

        assert qaoa_problem2.current_iteration == first_run_iteration
        assert len(qaoa_problem2.losses_history) == first_run_losses_count

        qaoa_problem2.max_iterations = 12
        qaoa_problem2.run()

        np.testing.assert_equal(qaoa_problem2.solution, QUBO_SOLUTION)
        assert qaoa_problem2.current_iteration == 12

    @pytest.fixture
    def binary_quadratic_model(self):
        bqm = dimod.BinaryQuadraticModel(
            {"x": -1, "y": -2, "z": 3}, {}, 0.0, dimod.Vartype.BINARY
        )
        return bqm

    @pytest.fixture
    def bqm_minimize(self):
        """BQM for minimization test: x=1, y=-2, z=3, w=-1"""
        return make_bqm_minimize()

    @pytest.fixture
    def bqm_maximize(self):
        """BQM for maximization test (negated for minimization): x=-1, y=2, z=-3, w=1"""
        return make_bqm_maximize()

    def test_binary_quadratic_model_initialization(
        self, binary_quadratic_model, default_test_simulator
    ):
        qaoa_problem = QAOA(
            BinaryOptimizationProblem(binary_quadratic_model),
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=10,
            backend=default_test_simulator,
        )

        assert isinstance(qaoa_problem.backend, CircuitRunner)
        assert qaoa_problem.backend.shots == 5000
        assert isinstance(qaoa_problem.optimizer, ScipyOptimizer)
        assert qaoa_problem.optimizer.method == ScipyMethod.NELDER_MEAD
        assert qaoa_problem.max_iterations == 10
        assert isinstance(qaoa_problem.problem, BinaryOptimizationProblem)
        assert qaoa_problem.n_layers == 1

        assert len(qaoa_problem.cost_hamiltonian) == 3
        assert all(
            isinstance(op, qp.Z) for op in qaoa_problem.cost_hamiltonian.terms()[1]
        )
        assert len(qaoa_problem.problem.mixer_hamiltonian) == 3
        assert all(
            isinstance(op, qp.X)
            for op in qaoa_problem.problem.mixer_hamiltonian.terms()[1]
        )

        verify_metacircuit_dict(qaoa_problem, ["cost_circuit", "meas_circuit"])

    @pytest.mark.e2e
    def test_binary_quadratic_model_minimize_correct(
        self, bqm_minimize, default_test_simulator
    ):
        default_test_simulator.set_seed(1997)

        qaoa_problem = QAOA(
            BinaryOptimizationProblem(bqm_minimize),
            n_layers=2,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=15,
            backend=default_test_simulator,
            seed=1997,
        )

        qaoa_problem.run()

        expected_solution = {"w": 0, "x": 1, "y": 0, "z": 1}
        assert qaoa_problem.solution == expected_solution

    @pytest.mark.e2e
    def test_binary_quadratic_model_maximize_correct(
        self, bqm_maximize, default_test_simulator
    ):
        default_test_simulator.set_seed(1997)

        qaoa_problem = QAOA(
            BinaryOptimizationProblem(bqm_maximize),
            n_layers=2,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=15,
            backend=default_test_simulator,
            seed=1997,
        )

        qaoa_problem.run()

        expected_solution = {"w": 1, "x": 0, "y": 1, "z": 0}
        assert qaoa_problem.solution == expected_solution

    @pytest.mark.e2e
    def test_binary_quadratic_model_e2e_checkpointing_resume(
        self, bqm_minimize, default_test_simulator, tmp_path
    ):
        """Test QAOA BinaryQuadraticModel e2e with checkpointing and resume."""
        optimizer = MonteCarloOptimizer(population_size=10, n_best_sets=3)

        checkpoint_dir = tmp_path / "checkpoint_test"
        default_test_simulator.set_seed(1997)

        bqm_problem = BinaryOptimizationProblem(bqm_minimize)

        qaoa_problem1 = QAOA(
            bqm_problem,
            n_layers=2,
            optimizer=optimizer,
            max_iterations=7,
            backend=default_test_simulator,
            seed=1997,
        )

        qaoa_problem1.run(
            checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir)
        )

        assert checkpoint_dir.exists()
        checkpoint_path = checkpoint_dir / "checkpoint_007"
        assert checkpoint_path.exists()
        assert (checkpoint_path / "program_state.json").exists()

        first_run_iteration = qaoa_problem1.current_iteration
        first_run_losses_count = len(qaoa_problem1.losses_history)

        qaoa_problem2 = QAOA.load_state(
            checkpoint_dir,
            backend=default_test_simulator,
            problem=bqm_problem,
            n_layers=2,
        )

        assert qaoa_problem2.current_iteration == first_run_iteration
        assert len(qaoa_problem2.losses_history) == first_run_losses_count

        qaoa_problem2.max_iterations = 15
        qaoa_problem2.run()

        expected_solution = {"w": 0, "x": 1, "y": 0, "z": 1}
        assert qaoa_problem2.solution == expected_solution
        assert qaoa_problem2.current_iteration == 15


# ---------------------------------------------------------------------------
# QUBO PartitioningProgramEnsemble integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_ansatz() -> GenericLayerAnsatz:
    return GenericLayerAnsatz([qp.RY, qp.RZ])


@pytest.fixture
def qubo_ensemble_qaoa(sample_qubo_matrix, default_test_simulator):
    """Provides a PartitioningProgramEnsemble with BinaryOptimizationProblem for testing."""
    decomposer = hybrid.EnergyImpactDecomposer(size=2)
    problem = BinaryOptimizationProblem(sample_qubo_matrix, decomposer=decomposer)
    return PartitioningProgramEnsemble(
        problem=problem,
        n_layers=1,
        optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        max_iterations=10,
        backend=default_test_simulator,
    )


@pytest.fixture
def qubo_ensemble_pce(sample_qubo_matrix, basic_ansatz, default_test_simulator):
    """Provides a PartitioningProgramEnsemble configured to use PCE partitions."""
    decomposer = hybrid.EnergyImpactDecomposer(size=2)
    problem = BinaryOptimizationProblem(sample_qubo_matrix, decomposer=decomposer)
    return PartitioningProgramEnsemble(
        problem=problem,
        n_layers=1,
        quantum_routine="pce",
        ansatz=basic_ansatz,
        optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        max_iterations=10,
        backend=default_test_simulator,
    )


class TestQUBOPartitioningEnsemble:
    def test_correct_initialization(self, qubo_ensemble_qaoa, sample_qubo_matrix):
        assert qubo_ensemble_qaoa.max_iterations == 10
        assert qubo_ensemble_qaoa.quantum_routine == "qaoa"

    def test_create_programs(self, mocker, qubo_ensemble_qaoa):
        mock_constructor = mocker.MagicMock()
        qubo_ensemble_qaoa._constructor = mock_constructor

        qubo_ensemble_qaoa.create_programs()

        assert len(qubo_ensemble_qaoa.programs) == 2
        assert mock_constructor.call_count == 2

    def test_correct_initialization_pce(self, qubo_ensemble_pce, sample_qubo_matrix):
        assert qubo_ensemble_pce.quantum_routine == "pce"

    def test_create_programs_pce_creates_pce_programs(self, qubo_ensemble_pce):
        qubo_ensemble_pce.create_programs()
        assert len(qubo_ensemble_pce.programs) == 2
        assert all(
            isinstance(program, PCE) for program in qubo_ensemble_pce.programs.values()
        )

    def test_invalid_engine_raises(self, sample_qubo_matrix, dummy_simulator):
        decomposer = hybrid.EnergyImpactDecomposer(size=2)
        problem = BinaryOptimizationProblem(sample_qubo_matrix, decomposer=decomposer)
        with pytest.raises(ValueError, match="Unsupported quantum_routine"):
            PartitioningProgramEnsemble(
                problem=problem,
                n_layers=1,
                optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
                quantum_routine="invalid",
                backend=dummy_simulator,
            )

    def test_trivial_subproblem_is_identified_and_skipped(self, dummy_simulator):
        trivial_qubo = np.array(
            [
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, -1],
            ]
        )

        decomposer = hybrid.EnergyImpactDecomposer(size=2)
        problem = BinaryOptimizationProblem(trivial_qubo, decomposer=decomposer)

        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            backend=dummy_simulator,
        )

        ensemble.create_programs()

        assert len(problem._trivial_program_ids) == 2
        assert len(ensemble.programs) == 0

    def test_aggregate_results_error_handling(self, mocker, qubo_ensemble_qaoa):
        """Test comprehensive error handling in aggregate_results method."""
        qubo_ensemble_qaoa.create_programs()

        mock_program_empty_losses = mocker.MagicMock(spec=QAOA)
        mock_program_empty_losses.best_probs = {}
        mock_program_empty_losses.has_results.return_value = False
        qubo_ensemble_qaoa.programs = {("A", 2): mock_program_empty_losses}

        with pytest.raises(RuntimeError, match="Some/All programs have no results"):
            qubo_ensemble_qaoa.aggregate_results()

        mock_program_empty_final_probs = mocker.MagicMock(spec=QAOA)
        mock_program_empty_final_probs.best_probs = {}
        mock_program_empty_final_probs.has_results.return_value = True
        qubo_ensemble_qaoa.programs = {("A", 2): mock_program_empty_final_probs}

        with pytest.raises(
            RuntimeError, match="Not all final probabilities computed yet"
        ):
            qubo_ensemble_qaoa.aggregate_results()

    def test_get_top_solutions_numerical_correctness(self, dummy_simulator):
        """Verify exact solutions and energies for a known QUBO problem."""
        qubo = {
            (0, 0): -0.5,
            (1, 1): 1,
            (0, 1): -2,
            (2, 2): 1,
            (3, 3): 1,
            (2, 3): 2,
        }
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
        decomposer = hybrid.EnergyImpactDecomposer(size=2)
        problem = BinaryOptimizationProblem(bqm, decomposer=decomposer)

        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=dummy_simulator,
        )
        ensemble.create_programs()

        for program in ensemble.programs.values():
            program._best_probs = {"tag": {"00": 0.3, "11": 0.7}}
            program._losses_history = [{"dummy_loss": 0.0}]

        results = ensemble.get_top_solutions(
            n=4, beam_width=4, n_partition_candidates=4
        )

        assert len(results) == 4

        solutions = [tuple(sol.tolist()) for sol, _ in results]
        energies = [energy for _, energy in results]

        assert solutions[0] == (1, 1, 0, 0)
        assert energies[0] == pytest.approx(-1.5)

        assert solutions[1] == (0, 0, 0, 0)
        assert energies[1] == pytest.approx(0.0)

        assert solutions[2] == (1, 1, 1, 1)
        assert energies[2] == pytest.approx(2.5)

        assert solutions[3] == (0, 0, 1, 1)
        assert energies[3] == pytest.approx(4.0)

    def test_get_top_solutions_best_matches_aggregate(self, qubo_ensemble_qaoa):
        """The best solution from get_top_solutions matches aggregate_results."""
        qubo_ensemble_qaoa.create_programs()

        for program in qubo_ensemble_qaoa.programs.values():
            n_qubits = program.n_qubits
            program._best_probs = {"tag": {"0" * n_qubits: 0.6, "1" * n_qubits: 0.4}}
            program._losses_history = [{"dummy_loss": 0.0}]

        agg_solution, agg_energy = qubo_ensemble_qaoa.aggregate_results()

        qubo_ensemble_qaoa.reset()
        qubo_ensemble_qaoa.create_programs()
        for program in qubo_ensemble_qaoa.programs.values():
            n_qubits = program.n_qubits
            program._best_probs = {"tag": {"0" * n_qubits: 0.6, "1" * n_qubits: 0.4}}
            program._losses_history = [{"dummy_loss": 0.0}]

        results = qubo_ensemble_qaoa.get_top_solutions(n=1, beam_width=1)
        assert len(results) == 1
        top_solution, top_energy = results[0]

        np.testing.assert_array_equal(top_solution, agg_solution)
        assert top_energy == pytest.approx(agg_energy)

    def test_get_top_solutions_does_not_mutate_state(self, qubo_ensemble_qaoa):
        """get_top_solutions does not mutate _bqm_subproblem_states."""
        qubo_ensemble_qaoa.create_programs()

        for program in qubo_ensemble_qaoa.programs.values():
            n_qubits = program.n_qubits
            program._best_probs = {"tag": {"0" * n_qubits: 1.0}}
            program._losses_history = [{"dummy_loss": 0.0}]

        problem = qubo_ensemble_qaoa._problem
        state_ids_before = {k: id(v) for k, v in problem._bqm_subproblem_states.items()}

        qubo_ensemble_qaoa.get_top_solutions(n=2, beam_width=2)

        for k, v in problem._bqm_subproblem_states.items():
            assert id(v) == state_ids_before[k]

    @pytest.mark.e2e
    def test_qubo_partitioning_e2e(self, default_test_simulator):
        """An end-to-end test solving a small QUBO."""
        qubo = {
            (0, 0): -0.5,
            (1, 1): 1,
            (0, 1): -2,
            (2, 2): 1,
            (3, 3): 1,
            (2, 3): 2,
        }
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)

        decomposer = hybrid.EnergyImpactDecomposer(size=2)
        problem = BinaryOptimizationProblem(bqm, decomposer=decomposer)

        default_test_simulator.set_seed(1997)

        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=2,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=15,
            backend=default_test_simulator,
            seed=1997,
        )

        ensemble.create_programs()
        ensemble.run(blocking=True, batch_config=BatchConfig(_sort_programs=True))
        solution, energy = ensemble.aggregate_results()

        expected_solution = np.array([1, 1, 0, 0])
        np.testing.assert_array_equal(solution, expected_solution)

        assert isinstance(energy, float)
        assert energy == pytest.approx(-1.5)

    @pytest.mark.e2e
    def test_qubo_partitioning_pce_e2e(self, default_test_simulator, basic_ansatz):
        """An end-to-end test solving a small QUBO with PCE engine."""
        qubo = {
            (0, 0): -0.5,
            (1, 1): 1,
            (0, 1): -2,
            (2, 2): 1,
            (3, 3): 1,
            (2, 3): 2,
        }
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
        decomposer = hybrid.EnergyImpactDecomposer(size=2)
        problem = BinaryOptimizationProblem(bqm, decomposer=decomposer)

        default_test_simulator.set_seed(1997)

        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=2,
            quantum_routine="pce",
            ansatz=basic_ansatz,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=15,
            backend=default_test_simulator,
            seed=1997,
        )

        ensemble.create_programs()
        ensemble.run(blocking=True, batch_config=BatchConfig(_sort_programs=True))
        solution, energy = ensemble.aggregate_results()

        expected_solution = np.array([1, 1, 0, 0])
        np.testing.assert_array_equal(solution, expected_solution)

        assert isinstance(energy, float)
        assert energy == pytest.approx(-1.5)
