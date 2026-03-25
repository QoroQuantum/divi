# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import dimod
import hybrid
import numpy as np
import pennylane as qml
import pytest
import scipy.sparse as sps

from divi.qprog import PCE, QAOA, BatchConfig, ScipyMethod, ScipyOptimizer
from divi.qprog.algorithms import GenericLayerAnsatz
from divi.qprog.problems import BinaryOptimizationProblem
from divi.qprog.problems._binary import (
    _merge_substates,
    _sanitize_problem_input,
)
from divi.qprog.workflows import PartitioningProgramEnsemble
from tests.qprog.qprog_contracts import verify_basic_program_ensemble_behaviour

# --- Fixtures and Test Data ---


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


@pytest.fixture
def basic_ansatz() -> GenericLayerAnsatz:
    return GenericLayerAnsatz([qml.RY, qml.RZ])


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


# --- Test Helper Functions ---


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

    def test_with_sparse_matrix(self, sample_qubo_matrix):
        sparse_matrix = sps.coo_matrix(sample_qubo_matrix)
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


# --- Test Evaluate Solution ---


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


# --- Test PartitioningProgramEnsemble with BinaryOptimizationProblem ---


class TestQUBOPartitioningEnsemble:
    def test_correct_initialization(self, qubo_ensemble_qaoa, sample_qubo_matrix):
        assert qubo_ensemble_qaoa.max_iterations == 10
        assert qubo_ensemble_qaoa.quantum_routine == "qaoa"

    def test_create_programs(self, mocker, qubo_ensemble_qaoa):
        # Mock the QAOA constructor to verify it's called correctly
        mock_constructor = mocker.MagicMock()

        # Replace the _constructor attribute on the instance with our mock
        qubo_ensemble_qaoa._constructor = mock_constructor

        qubo_ensemble_qaoa.create_programs()

        # The decomposer should split the 4 variables into two programs of size 2
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

    def test_verify_basic_behaviour(self, mocker, qubo_ensemble_qaoa):
        """Verify the class adheres to the ProgramEnsemble contract."""
        verify_basic_program_ensemble_behaviour(mocker, qubo_ensemble_qaoa)

    def test_trivial_subproblem_is_identified_and_skipped(self, dummy_simulator):
        """
        Tests that a subproblem with no interactions is correctly identified
        as trivial and that a QAOA program is NOT created for it.
        """
        # 1. SETUP: Create a QUBO with no interaction terms. Any partition
        # of this problem will be "trivial" (have no quadratic terms).
        trivial_qubo = np.array(
            [
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, -1],
            ]
        )

        # Decomposer will split this into two trivial problems of size 2.
        decomposer = hybrid.EnergyImpactDecomposer(size=2)
        problem = BinaryOptimizationProblem(trivial_qubo, decomposer=decomposer)

        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            backend=dummy_simulator,
        )

        # 2. ACT: Run the program creation process.
        ensemble.create_programs()

        # 3. ASSERT: Verify the behavioral outcome.
        # The problem should have identified two trivial programs.
        assert len(problem._trivial_program_ids) == 2

        # Most importantly, no complex QAOA programs should have been created.
        assert len(ensemble.programs) == 0

    def test_aggregate_results_error_handling(self, mocker, qubo_ensemble_qaoa):
        """Test comprehensive error handling in aggregate_results method."""
        qubo_ensemble_qaoa.create_programs()

        # Test 1: Empty losses error
        mock_program_empty_losses = mocker.MagicMock(spec=QAOA)
        mock_program_empty_losses.best_probs = {}
        mock_program_empty_losses._losses_history = []  # Empty losses
        qubo_ensemble_qaoa.programs = {("A", 2): mock_program_empty_losses}

        with pytest.raises(RuntimeError, match="Some/All programs have empty losses"):
            qubo_ensemble_qaoa.aggregate_results()

        # Test 2: Empty final_probs error
        mock_program_empty_final_probs = mocker.MagicMock(spec=QAOA)
        mock_program_empty_final_probs.best_probs = {}  # Empty final_probs
        mock_program_empty_final_probs.losses_history = [
            {"dummy_loss": 0.0}
        ]  # Non-empty losses
        qubo_ensemble_qaoa.programs = {("A", 2): mock_program_empty_final_probs}

        with pytest.raises(
            RuntimeError, match="Not all final probabilities computed yet"
        ):
            qubo_ensemble_qaoa.aggregate_results()

    def test_get_top_solutions_numerical_correctness(self, dummy_simulator):
        """Verify exact solutions and energies for a known QUBO problem.

        The QUBO has known optimal [1,1,0,0] with energy -1.5.
        The decomposer splits into partition A (vars 2,3) and B (vars 0,1).
        Mocking two candidates per partition ("00" and "11") produces 4 global
        solutions whose energies can be computed analytically.
        """
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

        # Mock two candidates per partition: "00" and "11"
        for program in ensemble.programs.values():
            program._best_probs = {"tag": {"00": 0.3, "11": 0.7}}
            program._losses_history = [{"dummy_loss": 0.0}]

        results = ensemble.get_top_solutions(
            n=4, beam_width=4, n_partition_candidates=4
        )

        # Expected solutions sorted by energy:
        # [1,1,0,0] -> -1.5, [0,0,0,0] -> 0.0, [1,1,1,1] -> 2.5, [0,0,1,1] -> 4.0
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

        # Re-create to avoid the mutation from aggregate_results
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
        # Snapshot subproblem state ids before the call
        state_ids_before = {k: id(v) for k, v in problem._bqm_subproblem_states.items()}

        qubo_ensemble_qaoa.get_top_solutions(n=2, beam_width=2)

        # The original state objects should not have been replaced
        for k, v in problem._bqm_subproblem_states.items():
            assert id(v) == state_ids_before[k]

    def test_get_top_solutions_raises_if_not_run(self, qubo_ensemble_qaoa):
        qubo_ensemble_qaoa.create_programs()
        with pytest.raises(RuntimeError, match="Some/All programs have empty losses"):
            qubo_ensemble_qaoa.get_top_solutions()

    def test_get_top_solutions_raises_on_invalid_n(self, qubo_ensemble_qaoa):
        with pytest.raises(ValueError, match="n must be >= 1"):
            qubo_ensemble_qaoa.get_top_solutions(n=0)

    @pytest.mark.e2e
    def test_qubo_partitioning_e2e(self, default_test_simulator):
        """An end-to-end test solving a small QUBO."""
        qubo = {
            (0, 0): -0.5,
            (1, 1): 1,
            (0, 1): -2,  # Partition 1: min at x0=1, x1=1
            (2, 2): 1,
            (3, 3): 1,
            (2, 3): 2,  # Partition 2: min at x2=0, x3=0
        }
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)

        # Decomposer to split into two problems of size 2
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

        # Run the full flow -- _sort_programs=True ensures the merged circuit
        # order is deterministic across runs when a fixed seed is set.
        ensemble.create_programs()
        ensemble.run(blocking=True, batch_config=BatchConfig(_sort_programs=True))
        solution, energy = ensemble.aggregate_results()

        # The known optimal solution for this QUBO is [1, 1, 0, 0]
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


class TestExtendSolutionQUBO:
    """Tests for BinaryOptimizationProblem.extend_solution."""

    @staticmethod
    def _make_problem(qubo):
        decomposer = hybrid.EnergyImpactDecomposer(size=2)
        return BinaryOptimizationProblem(qubo, decomposer=decomposer)

    def test_maps_local_bits_to_global_positions(self, dummy_simulator):
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
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=dummy_simulator,
        )
        ensemble.create_programs()

        # Pick a program and build a candidate with all-ones decoded
        prog_id = list(ensemble.programs.keys())[0]
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

    def test_does_not_mutate_input(self, dummy_simulator):
        """extend_solution returns a new list, not a mutation of the input."""
        qubo = np.diag([-1.0, -1.0, -1.0])
        problem = self._make_problem(qubo)
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=dummy_simulator,
        )
        ensemble.create_programs()

        prog_id = list(problem._variable_maps.keys())[0]
        global_indices = problem._variable_maps[prog_id]
        n_local = len(global_indices)

        original = [0, 0, 0]
        decoded = np.ones(n_local, dtype=np.int32)
        result = problem.extend_solution(original, prog_id, decoded)

        assert result is not original
        assert original == [0, 0, 0]

    def test_overwrites_previous_partition_values(self, dummy_simulator):
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
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=dummy_simulator,
        )
        ensemble.create_programs()

        prog_id = list(ensemble.programs.keys())[0]
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


class TestComposeSolutionQUBO:
    """Tests for BinaryOptimizationProblem._compose_solution."""

    @staticmethod
    def _make_problem(qubo):
        decomposer = hybrid.EnergyImpactDecomposer(size=2)
        return BinaryOptimizationProblem(qubo, decomposer=decomposer)

    def test_optimal_solution_has_correct_energy(self, dummy_simulator):
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
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=dummy_simulator,
        )
        ensemble.create_programs()

        # [1,1,0,0] is the optimal solution with energy -1.5
        solution_array, energy = problem._compose_solution([1, 1, 0, 0])

        assert isinstance(solution_array, np.ndarray)
        assert energy == pytest.approx(-1.5)
        np.testing.assert_array_equal(solution_array, [1, 1, 0, 0])

    def test_all_zeros_returns_zero_energy(self, dummy_simulator):
        """All-zero solution has zero energy for a QUBO with no constant offset."""
        qubo = {
            (0, 0): -0.5,
            (1, 1): 1,
            (0, 1): -2,
        }
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)
        problem = self._make_problem(bqm)
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=dummy_simulator,
        )
        ensemble.create_programs()

        solution_array, energy = problem._compose_solution([0, 0])

        assert energy == pytest.approx(0.0)

    def test_does_not_mutate_subproblem_states(self, dummy_simulator):
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
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=dummy_simulator,
        )
        ensemble.create_programs()

        # Snapshot subproblem state identities before call
        state_ids_before = {k: id(v) for k, v in problem._bqm_subproblem_states.items()}

        problem._compose_solution([1, 1, 0, 0])

        # Original state objects should not have been replaced
        for k, v in problem._bqm_subproblem_states.items():
            assert id(v) == state_ids_before[k]


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


class TestInitialSolutionSizeQUBO:
    def test_returns_number_of_variables(self, sample_qubo_matrix):
        decomposer = hybrid.EnergyImpactDecomposer(size=2)
        problem = BinaryOptimizationProblem(sample_qubo_matrix, decomposer=decomposer)

        assert problem.initial_solution_size() == 4


class TestFinalizeSolutionQUBO:
    def test_returns_array_and_energy(self, sample_qubo_matrix, dummy_simulator):
        decomposer = hybrid.EnergyImpactDecomposer(size=2)
        problem = BinaryOptimizationProblem(sample_qubo_matrix, decomposer=decomposer)
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=dummy_simulator,
        )
        ensemble.create_programs()

        result = problem.finalize_solution(-1.0, [1, 1, 0, 0])

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], float)


class TestFormatTopSolutionsQUBO:
    def test_formats_and_sorts_by_energy(self, sample_qubo_matrix, dummy_simulator):
        decomposer = hybrid.EnergyImpactDecomposer(size=2)
        problem = BinaryOptimizationProblem(sample_qubo_matrix, decomposer=decomposer)
        ensemble = PartitioningProgramEnsemble(
            problem=problem,
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=dummy_simulator,
        )
        ensemble.create_programs()

        results = [(-1.0, [1, 1, 0, 0]), (-0.5, [0, 0, 1, 1])]
        formatted = problem.format_top_solutions(results)

        assert len(formatted) == 2
        # Should be sorted by energy (second element of each tuple)
        energies = [entry[1] for entry in formatted]
        assert energies == sorted(energies)
