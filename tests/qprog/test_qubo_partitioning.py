# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import dimod
import hybrid
import numpy as np
import pytest
import scipy.sparse as sps
from qprog_contracts import verify_basic_program_batch_behaviour

from divi.parallel_simulator import ParallelSimulator
from divi.qprog import QAOA, Optimizer
from divi.qprog._qubo_partitoning import (
    QUBOPartitioningQAOA,
    _sanitize_problem_input,
)

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
        optimizer=Optimizer.NELDER_MEAD,
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

    def test_aggregate_results_raises_before_run(self, mocker, qubo_partitioning_qaoa):
        qubo_partitioning_qaoa.create_programs()

        # Manually add a mock program that hasn't been "run" (empty probs)
        mock_program = mocker.MagicMock(spec=QAOA)
        mock_program.probs = {}
        # Add a non-empty losses list to pass the superclass check
        mock_program.losses = [{"dummy_loss": 0.0}]
        qubo_partitioning_qaoa.programs = {("A", 2): mock_program}

        # This test now correctly targets the error from the QUBOPartitioningQAOA class
        with pytest.raises(RuntimeError, match="Not all final probabilities computed"):
            qubo_partitioning_qaoa.aggregate_results()

    def test_results_aggregated_correctly(self, mocker, qubo_partitioning_qaoa):
        qubo_partitioning_qaoa.create_programs()

        # Get handles to the sub-programs
        prog_keys = list(qubo_partitioning_qaoa.programs.keys())
        prog_a = qubo_partitioning_qaoa.programs[prog_keys[0]]
        prog_b = qubo_partitioning_qaoa.programs[prog_keys[1]]

        # Mock the solutions from the sub-programs
        prog_a._solution_bitstring = np.array([1, 0])
        prog_b._solution_bitstring = np.array([1, 1])

        # Mark programs as "run" by populating both probs AND losses
        prog_a.probs = {"dummy": 1}
        prog_b.probs = {"dummy": 1}
        prog_a.losses = [{"dummy_loss": 0.0}]
        prog_b.losses = [{"dummy_loss": 0.0}]

        # Mock the final aggregation step to return a predictable result
        final_samples = dimod.SampleSet.from_samples(
            {0: 1, 1: 0, 2: 1, 3: 1}, "BINARY", -3.0
        )
        final_state = hybrid.State(samples=final_samples)
        mock_aggregator = mocker.patch.object(
            qubo_partitioning_qaoa._aggregating,
            "run",
            return_value=mocker.MagicMock(result=lambda: final_state),
        )

        solution, energy = qubo_partitioning_qaoa.aggregate_results()

        # The variable names in the final sampleset may not be ordered
        # so we create the expected array from the sample dictionary.
        expected_solution = np.array([final_samples.first.sample[i] for i in range(4)])

        assert np.array_equal(solution, expected_solution)
        assert energy == -3.0
        mock_aggregator.assert_called_once()

    def test_verify_basic_behaviour(self, mocker, qubo_partitioning_qaoa):
        """Verify the class adheres to the ProgramBatch contract."""
        verify_basic_program_batch_behaviour(mocker, qubo_partitioning_qaoa)

    @pytest.mark.e2e
    def test_qubo_partitioning_e2e(self):
        """An end-to-end test solving a small QUBO."""
        qubo = {
            (0, 0): 1,
            (1, 1): 1,
            (0, 1): -2,  # Partition 1: min at x0=1, x1=1
            (2, 2): 1,
            (3, 3): 1,
            (2, 3): 2,  # Partition 2: min at x2=0, x3=0
        }
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo)

        # Decomposer to split into two problems of size 2
        decomposer = hybrid.EnergyImpactDecomposer(size=2)

        batch = QUBOPartitioningQAOA(
            qubo=bqm,
            decomposer=decomposer,
            n_layers=2,
            optimizer=Optimizer.COBYLA,
            max_iterations=15,
            backend=ParallelSimulator(shots=2000),
        )

        # Run the full flow
        batch.create_programs()
        batch.run(blocking=True)
        solution, energy = batch.aggregate_results()

        # The known optimal solution for this QUBO is [1, 1, 0, 0]
        expected_solution = np.array([1, 1, 0, 0])

        assert np.array_equal(solution, expected_solution)
        assert isinstance(energy, float)
