# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0


import networkx as nx
import numpy as np
import pytest

from divi.circuits.qem import ZNE, LinearExtrapolator
from divi.hamiltonians import (
    ExactTrotterization,
    QDrift,
)
from divi.pipeline.stages import TrotterSpecStage
from divi.qprog import (
    QAOA,
    ScipyMethod,
    ScipyOptimizer,
)
from divi.qprog.problems import (
    BinaryOptimizationProblem,
    MaxCliqueProblem,
    MaxCutProblem,
)
from tests.qprog.problems._helpers import QUBO_MATRIX, make_bull_graph
from tests.qprog.qprog_contracts import (
    OPTIMIZERS_TO_TEST,
    verify_correct_circuit_count,
)


class TestGeneralQAOA:
    @pytest.mark.parametrize("optimizer", **OPTIMIZERS_TO_TEST)
    def test_qaoa_optimization_runs_cost_pipeline(
        self, mocker, optimizer, default_test_simulator
    ):
        """The cost pipeline is invoked during the optimization loop.

        Note: this is an implementation-coupling test that spies on an internal
        pipeline. The behavioral outcomes (losses, circuit counts) are verified
        by dedicated end-to-end tests.
        """
        optimizer = optimizer()  # Create fresh instance
        qaoa_problem = QAOA(
            MaxCliqueProblem(nx.bull_graph(), is_constrained=True),
            n_layers=1,
            optimizer=optimizer,
            max_iterations=1,
            backend=default_test_simulator,
        )

        # Spy on the cost pipeline's run method
        spy = mocker.spy(qaoa_problem._cost_pipeline, "run")

        # Mock final computation to isolate optimization phase
        mocker.patch.object(qaoa_problem, "_perform_final_computation")

        qaoa_problem.run()

        # Cost pipeline should be called once per iteration
        assert spy.call_count >= 1

    def test_qaoa_cost_pipeline_uses_cost_hamiltonian_initial_spec(
        self, mocker, default_test_simulator
    ):
        """QAOA should customize cost evaluation via the initial-spec hook only."""
        qaoa_problem = QAOA(
            MaxCliqueProblem(nx.bull_graph(), is_constrained=True),
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=default_test_simulator,
        )

        spy = mocker.spy(qaoa_problem._cost_pipeline, "run")
        mocker.patch.object(qaoa_problem, "_perform_final_computation")

        qaoa_problem.run()

        assert spy.call_args.kwargs["initial_spec"] is qaoa_problem.cost_hamiltonian

    @pytest.mark.parametrize("optimizer", **OPTIMIZERS_TO_TEST)
    def test_qaoa_final_computation_runs_measurement_pipeline(
        self, mocker, optimizer, default_test_simulator
    ):
        """The measurement pipeline is invoked during final computation.

        Note: this is an implementation-coupling test that spies on an internal
        pipeline. The behavioral outcomes (solution extraction) are verified
        by dedicated end-to-end tests.
        """
        optimizer = optimizer()  # Create fresh instance
        qaoa_problem = QAOA(
            MaxCliqueProblem(nx.bull_graph(), is_constrained=True),
            n_layers=1,
            optimizer=optimizer,
            max_iterations=1,
            backend=default_test_simulator,
        )
        # Set preconditions
        qaoa_problem._final_params = np.array([[0.1, 0.2]])
        qaoa_problem._best_params = np.array([[0.1, 0.2]])

        # Spy on measurement pipeline
        spy = mocker.spy(qaoa_problem._measurement_pipeline, "run")

        qaoa_problem._perform_final_computation()

        # Measurement pipeline should be called once
        assert spy.call_count == 1

    @pytest.mark.parametrize("optimizer", **OPTIMIZERS_TO_TEST)
    def test_graph_correct_circuits_count_and_energies(
        self, optimizer, dummy_simulator
    ):
        optimizer = optimizer()  # Create fresh instance
        qaoa_problem = QAOA(
            MaxCliqueProblem(nx.bull_graph(), is_constrained=True),
            n_layers=1,
            optimizer=optimizer,
            max_iterations=1,
            backend=dummy_simulator,
        )

        qaoa_problem.run()

        verify_correct_circuit_count(qaoa_problem)


class TestQAOAQDriftMultiSample:
    """Tests for QAOA with multi-sample QDrift (n_hamiltonians_per_iteration > 1).

    Several tests locate the ``TrotterSpecStage`` inside the cost pipeline via
    ``_cost_pipeline._stages``. This couples to the pipeline's internal structure,
    but there is no public API to observe the number of Hamiltonian samples
    produced — the stage's ``expand`` output is the only observable.
    """

    @staticmethod
    def _find_trotter_stage(qaoa):
        """Walk ``_cost_pipeline._stages`` to locate the TrotterSpecStage."""
        for stage in qaoa._cost_pipeline._stages:
            if isinstance(stage, TrotterSpecStage):
                return stage
        raise AssertionError("TrotterSpecStage not found in cost pipeline")

    def test_exact_trotterization_uses_single_hamiltonian_sample(
        self, mocker, default_test_simulator
    ):
        """With ExactTrotterization, TrotterSpecStage.expand produces a single ham sample."""
        strategy = ExactTrotterization(keep_top_n=3)
        qaoa = QAOA(
            MaxCutProblem(nx.bull_graph()),
            n_layers=1,
            trotterization_strategy=strategy,
            max_iterations=1,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        )

        trotter_stage = self._find_trotter_stage(qaoa)

        # Spy on the TrotterSpecStage.expand to inspect how many ham samples are produced
        spy = mocker.spy(trotter_stage, "expand")
        mocker.patch.object(qaoa, "_perform_final_computation")
        qaoa.run()

        # Check that expand produced only 1 ham sample (ham_id=0)
        batch, _token = spy.spy_return
        ham_ids = {
            key[0][1] for key in batch
        }  # Extract ham_id from (("ham", id),) keys
        assert ham_ids == {0}

    def test_multi_sample_generates_circuits_with_hamiltonian_id(
        self, mocker, default_test_simulator
    ):
        """TrotterSpecStage.expand with multi-sample QDrift produces multiple ham IDs."""
        strategy = QDrift(
            keep_fraction=0.3,
            sampling_budget=5,
            n_hamiltonians_per_iteration=3,
            seed=42,
        )
        qaoa = QAOA(
            MaxCutProblem(nx.bull_graph()),
            n_layers=1,
            trotterization_strategy=strategy,
            max_iterations=1,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        )

        trotter_stage = self._find_trotter_stage(qaoa)

        spy = mocker.spy(trotter_stage, "expand")
        mocker.patch.object(qaoa, "_perform_final_computation")
        qaoa.run()

        # Check that expand produced 3 ham samples
        batch, _token = spy.spy_return
        ham_ids = {key[0][1] for key in batch}
        assert ham_ids == {0, 1, 2}

    @pytest.mark.e2e
    def test_multi_sample_qaoa_e2e_solution(self, default_test_simulator):
        """QAOA with multi-sample QDrift runs to completion and finds correct MAXCUT."""
        G = make_bull_graph()
        default_test_simulator.set_seed(1997)

        strategy = QDrift(
            keep_fraction=0.5,
            sampling_budget=6,
            n_hamiltonians_per_iteration=2,
            seed=123,
        )
        qaoa = QAOA(
            MaxCutProblem(G),
            n_layers=1,
            trotterization_strategy=strategy,
            max_iterations=20,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            seed=1997,
        )
        qaoa.run()

        assert qaoa.total_circuit_count > 0
        assert qaoa.total_run_time >= 0
        assert len(qaoa.losses_history) == 20
        assert qaoa.best_loss < float("inf")

        # At least one of the top solutions achieves the optimal cut (more stable than single best)
        max_cut_val, _ = nx.algorithms.approximation.maxcut.one_exchange(G)

        def cut_value(partition1):
            partition0 = set(G.nodes()) - set(partition1)
            return sum(
                1 for u, v in G.edges() if (u in partition0) != (v in partition0)
            )

        top_solutions = qaoa.get_top_solutions(n=5, include_decoded=True)
        optimal_solutions = [
            sol
            for sol in top_solutions
            if sol.decoded is not None and cut_value(sol.decoded) == max_cut_val
        ]
        assert len(optimal_solutions) >= 1

        # Verify nodes in the optimal cut are valid graph nodes
        for sol in optimal_solutions:
            assert all(node in G.nodes() for node in sol.decoded)

    def test_multi_sample_trotter_stage_is_stateful_for_qdrift(
        self, default_test_simulator
    ):
        """TrotterSpecStage is correctly marked stateful for QDrift (ensures cache invalidation)."""
        strategy = QDrift(
            keep_fraction=0.5,
            sampling_budget=4,
            n_hamiltonians_per_iteration=3,
            seed=42,
        )
        qaoa = QAOA(
            MaxCutProblem(nx.bull_graph()),
            n_layers=1,
            trotterization_strategy=strategy,
            max_iterations=1,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        )

        trotter_stage = self._find_trotter_stage(qaoa)
        assert trotter_stage.stateful is True

    def test_multi_sample_final_computation_merges_histograms(
        self, default_test_simulator
    ):
        """Final computation with multi-sample QDrift samples Hamiltonians and merges histograms."""
        strategy = QDrift(
            keep_fraction=0.5,
            sampling_budget=4,
            n_hamiltonians_per_iteration=3,
            seed=456,
        )
        qaoa = QAOA(
            MaxCutProblem(nx.bull_graph()),
            n_layers=1,
            trotterization_strategy=strategy,
            max_iterations=2,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        )
        qaoa.run()
        # best_probs should contain merged distribution (one entry per param set)
        assert len(qaoa.best_probs) >= 1
        probs = next(iter(qaoa.best_probs.values()))
        assert isinstance(probs, dict)
        assert all(
            isinstance(k, str) and isinstance(v, (int, float)) for k, v in probs.items()
        )
        assert np.isclose(sum(probs.values()), 1.0)

    @pytest.mark.e2e
    def test_qdrift_zne_with_shot_based_backend(self, default_test_simulator):
        """ZNE works with shot-based backends via per-observable postprocessing."""
        G = make_bull_graph()
        default_test_simulator.set_seed(1997)

        scale_factors = [1.0, 3.0]  # odd integers for GlobalFoldPass
        zne_protocol = ZNE(
            scale_factors=scale_factors,
            extrapolator=LinearExtrapolator(),
        )
        strategy = QDrift(
            keep_fraction=0.5,
            sampling_budget=4,
            n_hamiltonians_per_iteration=2,
            seed=123,
        )
        qaoa = QAOA(
            MaxCutProblem(G),
            n_layers=1,
            trotterization_strategy=strategy,
            qem_protocol=zne_protocol,
            max_iterations=5,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            seed=1997,
        )

        qaoa.run()

        assert qaoa.total_circuit_count > 0
        assert qaoa.total_run_time >= 0
        assert len(qaoa.losses_history) == 5
        assert qaoa.best_loss < float("inf")


# ---------------------------------------------------------------------------
# Final computation with custom decode functions
# ---------------------------------------------------------------------------


class TestFinalComputationDecode:
    """Test that _perform_final_computation handles arbitrary decode returns."""

    def test_decode_returns_none(self, default_test_simulator):
        """When decode_fn returns None, solution is None."""
        problem = BinaryOptimizationProblem(QUBO_MATRIX)
        qaoa = QAOA(problem, backend=default_test_simulator, max_iterations=1)
        # Override the decode fn on the QAOA instance after construction
        qaoa._decode_solution_fn = lambda bs: None
        qaoa.run()
        assert qaoa.solution is None

    def test_decode_returns_custom_type(self, default_test_simulator):
        """When decode_fn returns a custom type, solution passes it through."""
        problem = BinaryOptimizationProblem(QUBO_MATRIX)
        qaoa = QAOA(problem, backend=default_test_simulator, max_iterations=1)
        qaoa._decode_solution_fn = lambda bs: [0, 2, 1, 0]
        qaoa.run()
        assert qaoa.solution == [0, 2, 1, 0]

    def test_default_decode(self, default_test_simulator):
        """Default QUBO decode returns a binary int array."""
        qaoa = QAOA(
            BinaryOptimizationProblem(QUBO_MATRIX),
            backend=default_test_simulator,
            max_iterations=1,
        )
        qaoa.run()
        sol = qaoa.solution
        assert all(b in (0, 1) for b in sol)
