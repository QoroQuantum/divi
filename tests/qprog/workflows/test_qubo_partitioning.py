# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import dimod
import hybrid
import numpy as np
import pytest
import scipy.sparse as sps

from divi.backends import ParallelSimulator
from divi.qprog import QAOA, ScipyMethod, ScipyOptimizer
from divi.qprog.workflows._qubo_partitioning import (
    QUBOPartitioningQAOA,
    _sanitize_problem_input,
)
from tests.qprog.qprog_contracts import verify_basic_program_batch_behaviour

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
def qubo_partitioning_qaoa(sample_qubo_matrix):
    """Provides a default QUBOPartitioningQAOA instance for testing."""
    # A simple decomposer that splits variables into two groups
    decomposer = hybrid.EnergyImpactDecomposer(size=2)
    return QUBOPartitioningQAOA(
        qubo=sample_qubo_matrix,
        decomposer=decomposer,
        n_layers=1,
        optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        max_iterations=10,
        backend=ParallelSimulator(shots=1000),
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
        with pytest.raises(ValueError, match="Only matrices supported."):
            _sanitize_problem_input(non_square)

    def test_raises_on_unsupported_type(self):
        unsupported = {"a": 1, "b": 2}
        with pytest.raises(ValueError, match="Got an unsupported QUBO input format"):
            _sanitize_problem_input(unsupported)


# --- Test QUBOPartitioningQAOA Class ---


class TestQUBOPartitioningQAOA:
    def test_correct_initialization(self, qubo_partitioning_qaoa, sample_qubo_matrix):
        assert np.array_equal(qubo_partitioning_qaoa.main_qubo, sample_qubo_matrix)
        assert isinstance(qubo_partitioning_qaoa._bqm, dimod.BinaryQuadraticModel)
        assert isinstance(qubo_partitioning_qaoa._partitioning, hybrid.Unwind)
        assert isinstance(qubo_partitioning_qaoa._aggregating, hybrid.Runnable)
        assert qubo_partitioning_qaoa.max_iterations == 10

    def test_create_programs(self, mocker, qubo_partitioning_qaoa):
        # Mock the QAOA constructor to verify it's called correctly
        mock_constructor = mocker.MagicMock()

        # Replace the _constructor attribute on the instance with our mock
        qubo_partitioning_qaoa._constructor = mock_constructor

        qubo_partitioning_qaoa.create_programs()

        # The decomposer should split the 4 variables into two programs of size 2
        assert len(qubo_partitioning_qaoa.programs) == 2
        assert mock_constructor.call_count == 2

        # Check program IDs and state tracking
        prog_ids = list(qubo_partitioning_qaoa.programs.keys())
        assert prog_ids[0] == ("A", 2)
        assert prog_ids[1] == ("B", 2)
        assert set(prog_ids) == set(
            qubo_partitioning_qaoa.prog_id_to_bqm_subproblem_states.keys()
        )

        # Verify that the QAOA programs were created with the correct subproblems
        for call in mock_constructor.call_args_list:
            kwargs = call.kwargs
            # Check the kwargs that were passed to the partial object
            assert isinstance(kwargs["problem"], sps.coo_matrix)
            assert kwargs["problem"].shape == (2, 2)
            assert "job_id" in kwargs

    def test_verify_basic_behaviour(self, mocker, qubo_partitioning_qaoa):
        """Verify the class adheres to the ProgramBatch contract."""
        verify_basic_program_batch_behaviour(mocker, qubo_partitioning_qaoa)

    def test_trivial_subproblem_is_identified_and_skipped(self):
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

        batch = QUBOPartitioningQAOA(
            qubo=trivial_qubo,
            decomposer=decomposer,
            n_layers=1,
            backend=ParallelSimulator(shots=100),
        )

        # 2. ACT: Run the program creation process.
        batch.create_programs()

        # 3. ASSERT: Verify the behavioral outcome.
        # The class should have identified two trivial programs.
        assert len(batch.trivial_program_ids) == 2
        assert ("A", 2) in batch.trivial_program_ids
        assert ("B", 2) in batch.trivial_program_ids

        # Most importantly, no complex QAOA programs should have been created.
        assert len(batch.programs) == 0

    def test_aggregate_results_error_handling(self, mocker, qubo_partitioning_qaoa):
        """Test comprehensive error handling in aggregate_results method."""
        qubo_partitioning_qaoa.create_programs()

        # Test 1: Empty losses error
        mock_program_empty_losses = mocker.MagicMock(spec=QAOA)
        mock_program_empty_losses._final_probs = {}
        mock_program_empty_losses._losses_history = []  # Empty losses
        qubo_partitioning_qaoa.programs = {("A", 2): mock_program_empty_losses}

        with pytest.raises(RuntimeError, match="Some/All programs have empty losses"):
            qubo_partitioning_qaoa.aggregate_results()

        # Test 2: Empty final_probs error
        mock_program_empty_final_probs = mocker.MagicMock(spec=QAOA)
        mock_program_empty_final_probs.final_probs = {}  # Empty final_probs
        mock_program_empty_final_probs.losses_history = [
            {"dummy_loss": 0.0}
        ]  # Non-empty losses
        qubo_partitioning_qaoa.programs = {("A", 2): mock_program_empty_final_probs}

        with pytest.raises(
            RuntimeError, match="Not all final probabilities computed yet"
        ):
            qubo_partitioning_qaoa.aggregate_results()

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

        default_test_simulator.set_seed(1997)

        batch = QUBOPartitioningQAOA(
            qubo=bqm,
            decomposer=decomposer,
            n_layers=2,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=15,
            backend=default_test_simulator,
            seed=1997,
        )

        # Run the full flow
        batch.create_programs()
        batch.run(blocking=True)
        solution, energy = batch.aggregate_results()

        # The known optimal solution for this QUBO is [1, 1, 0, 0]
        expected_solution = np.array([1, 1, 0, 0])
        np.testing.assert_array_equal(solution, expected_solution)

        assert isinstance(energy, float)
        assert energy == pytest.approx(-1.5)
