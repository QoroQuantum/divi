# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0


import warnings

import networkx as nx
import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from divi.circuits.zne import ZNE, LinearExtrapolator
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
from divi.qprog.algorithms import IterativeQAOA, SuperpositionState
from divi.qprog.problems import (
    BinaryOptimizationProblem,
    MaxCliqueProblem,
    MaxCutProblem,
    QAOAProblem,
)
from tests.qprog._program_contracts import (
    ObservableMeasuringContractsBase,
    verify_correct_circuit_count,
)
from tests.qprog.problems._helpers import QUBO_MATRIX, make_bull_graph


class TestGeneralQAOA:
    def test_qaoa_optimization_runs_cost_pipeline(
        self, mocker, optimizer, default_test_simulator
    ):
        """The cost pipeline is invoked during the optimization loop.

        Note: this is an implementation-coupling test that spies on an internal
        pipeline. The behavioral outcomes (losses, circuit counts) are verified
        by dedicated end-to-end tests.
        """
        qaoa_problem = QAOA(
            MaxCliqueProblem(nx.bull_graph(), is_constrained=True),
            n_layers=1,
            optimizer=optimizer,
            max_iterations=1,
            backend=default_test_simulator,
        )

        # Spy on the program's one entry point; isolate the optimization phase.
        spy = mocker.spy(qaoa_problem, "evaluate")
        mocker.patch.object(qaoa_problem, "sample_solution")

        qaoa_problem.run()

        # The cost protocol is evaluated at least once per iteration.
        assert spy.call_count >= 1
        assert any(call.args[1].name == "cost" for call in spy.call_args_list)

    def test_qaoa_cost_seed_is_the_cost_hamiltonian(
        self, mocker, default_test_simulator
    ):
        """QAOA prepares its state by trotterizing the cost Hamiltonian."""
        qaoa_problem = QAOA(
            MaxCliqueProblem(nx.bull_graph(), is_constrained=True),
            n_layers=1,
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=1,
            backend=default_test_simulator,
        )

        assert qaoa_problem._initial_spec() is qaoa_problem.cost_hamiltonian

    def test_qaoa_final_computation_runs_sample_preprocessor(
        self, mocker, optimizer, default_test_simulator
    ):
        """Final computation evaluates the sample protocol exactly once.

        Note: implementation-coupling test; the behavioral outcomes (solution
        extraction) are verified by dedicated end-to-end tests.
        """
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

        spy = mocker.spy(qaoa_problem, "evaluate")

        qaoa_problem.sample_solution()

        assert spy.call_count == 1
        assert spy.call_args.args[1].name == "sample"

    def test_graph_correct_circuits_count_and_energies(
        self, optimizer, dummy_simulator
    ):
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
        """Locate the TrotterSpecStage in the (memoized) cost-protocol pipeline."""
        pipeline = qaoa._build_preprocessor_pipeline(qaoa.cost_preprocessor())
        for stage in pipeline.stages:
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
        mocker.patch.object(qaoa, "sample_solution")
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
        mocker.patch.object(qaoa, "sample_solution")
        qaoa.run()

        # Check that expand produced 3 ham samples
        batch, _token = spy.spy_return
        ham_ids = {key[0][1] for key in batch}
        assert ham_ids == {0, 1, 2}

    @pytest.mark.e2e
    def test_multi_sample_qaoa_e2e_solution(self, default_test_simulator, mocker):
        """QAOA with multi-sample QDrift runs to completion and finds correct MAXCUT."""
        G = make_bull_graph()
        default_test_simulator.set_seed(1997)

        # p=2 + COBYLA so the optimal cut tops the distribution despite unseeded shots.
        strategy = QDrift(
            keep_fraction=0.5,
            sampling_budget=6,
            n_hamiltonians_per_iteration=5,
            seed=123,
        )
        qaoa = QAOA(
            MaxCutProblem(G),
            n_layers=2,
            trotterization_strategy=strategy,
            max_iterations=20,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            seed=1997,
        )

        # COBYLA may converge before max_iterations; pin the history length to the
        # per-iteration callback count rather than the nominal cap.
        iteration_spy = mocker.spy(qaoa.reporter, "update")
        qaoa.run()

        assert qaoa.total_circuit_count > 0
        assert qaoa.total_run_time >= 0
        assert len(qaoa.losses_history) == iteration_spy.call_count
        assert qaoa.best_loss < float("inf")

        # At least one of the top solutions achieves the optimal cut (more stable
        # than single best). Seed one_exchange: unseeded it falls back to the
        # global RNG, making max_cut_val depend on test ordering under -n auto.
        max_cut_val, _ = nx.algorithms.approximation.maxcut.one_exchange(G, seed=1997)

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

    def test_multi_sample_trotter_stage_is_evaluation_scoped_for_qdrift(
        self, default_test_simulator
    ):
        """QDrift samples are reused within, but not across, evaluations."""
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
        env = qaoa._build_pipeline_env()
        assert trotter_stage.cache_key_extras(env) == (env.evaluation_counter,)

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

        scale_factors = [1.0, 3.0]
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


class TestFinalComputationDecode:
    """Test that sample_solution handles arbitrary decode returns."""

    def test_decode_returns_none(self, default_test_simulator, default_optimizer):
        """When decode_fn returns None, solution is None."""
        problem = BinaryOptimizationProblem(QUBO_MATRIX)
        qaoa = QAOA(
            problem,
            backend=default_test_simulator,
            max_iterations=1,
            optimizer=default_optimizer,
        )
        # Override the decode fn on the QAOA instance after construction
        qaoa._decode_solution_fn = lambda bs: None
        qaoa.run()
        assert qaoa.solution is None

    def test_decode_returns_custom_type(
        self, default_test_simulator, default_optimizer
    ):
        """When decode_fn returns a custom type, solution passes it through."""
        problem = BinaryOptimizationProblem(QUBO_MATRIX)
        qaoa = QAOA(
            problem,
            backend=default_test_simulator,
            max_iterations=1,
            optimizer=default_optimizer,
        )
        qaoa._decode_solution_fn = lambda bs: [0, 2, 1, 0]
        qaoa.run()
        assert qaoa.solution == [0, 2, 1, 0]

    def test_default_decode(self, default_test_simulator, default_optimizer):
        """Default QUBO decode returns a binary int array."""
        qaoa = QAOA(
            BinaryOptimizationProblem(QUBO_MATRIX),
            backend=default_test_simulator,
            max_iterations=1,
            optimizer=default_optimizer,
        )
        qaoa.run()
        sol = qaoa.solution
        assert all(b in (0, 1) for b in sol)

    def test_solution_bitstring_after_run(
        self, default_test_simulator, default_optimizer
    ):
        """``solution_bitstring`` exposes the raw measured bitstring as a string."""
        qaoa = QAOA(
            BinaryOptimizationProblem(QUBO_MATRIX),
            backend=default_test_simulator,
            max_iterations=1,
            optimizer=default_optimizer,
        )
        qaoa.run()
        bs = qaoa.solution_bitstring
        assert isinstance(bs, str)
        assert len(bs) == qaoa.n_qubits
        assert set(bs) <= {"0", "1"}
        # Bitstring corresponds to the same state ``solution`` was decoded from.
        assert [int(c) for c in bs] == list(qaoa.solution)

    def test_solution_bitstring_before_run_raises(
        self, default_test_simulator, default_optimizer
    ):
        """Accessing ``solution_bitstring`` before ``.run()`` is an error."""
        qaoa = QAOA(
            BinaryOptimizationProblem(QUBO_MATRIX),
            backend=default_test_simulator,
            max_iterations=1,
            optimizer=default_optimizer,
        )
        with pytest.raises(RuntimeError, match="Call .run\\(\\) first"):
            _ = qaoa.solution_bitstring


class TestSampleSolution:
    """Tests for ``sample_solution(params)`` — sampling without training."""

    def _make_qaoa(self, backend, optimizer, n_layers=1):
        return QAOA(
            BinaryOptimizationProblem(QUBO_MATRIX),
            n_layers=n_layers,
            backend=backend,
            max_iterations=1,
            optimizer=optimizer,
        )

    def test_populates_solution_and_bitstring(
        self, default_test_simulator, default_optimizer
    ):
        """``sample_solution`` produces the same kind of outputs as ``run()``'s final step."""
        qaoa = self._make_qaoa(default_test_simulator, default_optimizer)
        params = np.array([0.1, 0.2])

        qaoa.sample_solution(params)

        assert qaoa.solution_bitstring is not None
        assert isinstance(qaoa.solution_bitstring, str)
        assert len(qaoa.solution_bitstring) == qaoa.n_qubits
        assert qaoa.best_probs  # measurement probs were populated
        assert qaoa.total_circuit_count > 0

    def test_skips_cost_pipeline(
        self, mocker, default_test_simulator, default_optimizer
    ):
        """``sample_solution`` must not dispatch any EXPECTATION job."""
        qaoa = self._make_qaoa(default_test_simulator, default_optimizer)
        spy = mocker.spy(qaoa, "evaluate")

        qaoa.sample_solution(np.array([0.1, 0.2]))

        # Only the sample protocol runs — the cost protocol is never evaluated.
        assert [call.args[1].name for call in spy.call_args_list] == ["sample"]

    def test_does_not_mutate_best_params(
        self, default_test_simulator, default_optimizer
    ):
        """After ``run()`` then ``sample_solution(other_params)``, ``best_params`` is unchanged."""
        qaoa = self._make_qaoa(default_test_simulator, default_optimizer)
        qaoa.run()
        trained = qaoa.best_params.copy()

        other = trained + 1.0
        qaoa.sample_solution(other)

        np.testing.assert_array_equal(qaoa.best_params, trained)

    def test_wrong_shape_raises(self, default_test_simulator, default_optimizer):
        """``params`` with mismatched per-set size raises ``ValueError``."""
        qaoa = self._make_qaoa(
            default_test_simulator, default_optimizer, n_layers=2
        )  # expects 4 params

        with pytest.raises(ValueError, match="does not match"):
            qaoa.sample_solution(np.array([0.1, 0.2, 0.3]))  # 3 != 4

    def test_no_args_before_run_raises(self, default_test_simulator, default_optimizer):
        """``sample_solution()`` before ``run()`` raises a clear ``RuntimeError``."""
        qaoa = self._make_qaoa(default_test_simulator, default_optimizer)

        with pytest.raises(RuntimeError, match="call run\\(\\) first"):
            qaoa.sample_solution()

    def test_returns_self(self, default_test_simulator, default_optimizer):
        """``sample_solution`` returns the program for method chaining."""
        qaoa = self._make_qaoa(default_test_simulator, default_optimizer)
        assert qaoa.sample_solution(np.array([0.1, 0.2])) is qaoa

    def test_does_not_mutate_optimizer_state(
        self, default_test_simulator, default_optimizer
    ):
        """``sample_solution`` leaves optimizer-side state (losses, iterations) untouched."""
        qaoa = self._make_qaoa(default_test_simulator, default_optimizer)

        qaoa.sample_solution(np.array([0.1, 0.2]))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            assert qaoa.losses_history == []
        assert qaoa.current_iteration == 0
        assert qaoa.optimize_result is None

    def test_followed_by_run_works(self, default_test_simulator, default_optimizer):
        """Calling ``run()`` after ``sample_solution()`` produces a normal training run."""
        qaoa = self._make_qaoa(default_test_simulator, default_optimizer)
        qaoa.sample_solution(np.array([0.1, 0.2]))

        qaoa.run()

        assert qaoa.current_iteration > 0
        assert qaoa.optimize_result is not None
        assert qaoa.solution is not None


def _make_problem(cost: SparsePauliOp, mixer: SparsePauliOp, wire_labels=None):
    _labels = wire_labels

    class _Problem(QAOAProblem):
        @property
        def cost_hamiltonian(self):
            return cost

        @property
        def mixer_hamiltonian(self):
            return mixer

        @property
        def loss_constant(self):
            return 0.0

        @property
        def decode_fn(self):
            return lambda bs: bs

        @property
        def recommended_initial_state(self):
            return SuperpositionState()

        @property
        def wire_labels(self):
            return _labels if _labels is not None else super().wire_labels

    return _Problem()


class TestWireSpaceInvariant:
    def test_mixer_wider_than_cost_raises(self, dummy_simulator, default_optimizer):
        prob = _make_problem(
            cost=SparsePauliOp.from_list([("IZZ", 1.0), ("ZZI", 1.0)]),
            mixer=SparsePauliOp.from_list(
                [("IIIX", 1.0), ("IIXI", 1.0), ("IXII", 1.0), ("XIII", 1.0)]
            ),
        )
        with pytest.raises(
            ValueError,
            match=r"wire_labels has 3 entries.*mixer_hamiltonian\.num_qubits is 4",
        ):
            QAOA(prob, backend=dummy_simulator, optimizer=default_optimizer)

    def test_cost_wider_than_mixer_raises(self, dummy_simulator, default_optimizer):
        prob = _make_problem(
            cost=SparsePauliOp.from_list([("IIZZ", 1.0), ("ZZII", 1.0)]),
            mixer=SparsePauliOp.from_list([("IIX", 1.0), ("IXI", 1.0), ("XII", 1.0)]),
        )
        with pytest.raises(
            ValueError,
            match=r"wire_labels has 4 entries.*mixer_hamiltonian\.num_qubits is 3",
        ):
            QAOA(prob, backend=dummy_simulator, optimizer=default_optimizer)

    def test_wire_labels_misaligned_with_hamiltonians_raises(
        self, dummy_simulator, default_optimizer
    ):
        # cost & mixer both 3-qubit but wire_labels claims 4 — qubit `i` of
        # the SPOs no longer maps to wire_labels[i].
        prob = _make_problem(
            cost=SparsePauliOp.from_list([("IZZ", 1.0), ("ZZI", 1.0)]),
            mixer=SparsePauliOp.from_list([("IIX", 1.0), ("IXI", 1.0), ("XII", 1.0)]),
            wire_labels=("a", "b", "c", "d"),
        )
        with pytest.raises(
            ValueError,
            match=r"wire_labels has 4 entries.*cost_hamiltonian\.num_qubits is 3",
        ):
            QAOA(prob, backend=dummy_simulator, optimizer=default_optimizer)

    def test_isolated_node_graph_with_wire_labels_succeeds(
        self, dummy_simulator, default_optimizer
    ):
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (0, 2)])
        g.add_node(3)
        qaoa = QAOA(
            MaxCutProblem(g), backend=dummy_simulator, optimizer=default_optimizer
        )
        assert qaoa.n_qubits == 4

    def test_isolated_node_maxcut_runs_end_to_end(
        self, default_test_simulator, default_optimizer
    ):
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (0, 2)])
        g.add_node(3)
        qaoa = QAOA(
            MaxCutProblem(g),
            backend=default_test_simulator,
            max_iterations=1,
            n_layers=1,
            optimizer=default_optimizer,
        )
        qaoa.run()
        assert len(qaoa.solution_bitstring) == 4
        assert set(qaoa.solution_bitstring) <= {"0", "1"}
        assert qaoa.solution is not None


class TestCostPipelineCache:
    """QAOA cost-circuit construction reuses each pipeline's forward cache."""

    @staticmethod
    def _make_qaoa(strategy, backend, optimizer):
        return QAOA(
            MaxCutProblem(make_bull_graph()),
            n_layers=1,
            trotterization_strategy=strategy,
            backend=backend,
            max_iterations=1,
            optimizer=optimizer,
        )

    @staticmethod
    def _cost_pipeline(qaoa):
        return qaoa._build_preprocessor_pipeline(qaoa.cost_preprocessor())

    @staticmethod
    def _forward(qaoa):
        return TestCostPipelineCache._cost_pipeline(qaoa).run_forward_pass(
            qaoa.cost_hamiltonian,
            qaoa._build_pipeline_env(),
        )

    def test_deterministic_strategy_reuses_stage_output(
        self, dummy_simulator, mocker, default_optimizer
    ):
        qaoa = self._make_qaoa(
            ExactTrotterization(), dummy_simulator, default_optimizer
        )
        spy = mocker.spy(qaoa, "_build_qaoa_qiskit_circuit")

        first = self._forward(qaoa)
        second = self._forward(qaoa)

        assert second.initial_batch is first.initial_batch
        assert spy.call_count == 1

    def test_qdrift_reuses_only_within_evaluation(
        self, dummy_simulator, default_optimizer
    ):
        strategy = QDrift(
            sampling_budget=2,
            n_hamiltonians_per_iteration=1,
            seed=42,
        )
        qaoa = self._make_qaoa(strategy, dummy_simulator, default_optimizer)
        first = self._forward(qaoa)
        same_evaluation = self._forward(qaoa)
        qaoa._evaluation_counter += 1
        next_evaluation = self._forward(qaoa)

        assert same_evaluation.initial_batch is first.initial_batch
        assert next_evaluation.initial_batch is not first.initial_batch

    def test_qdrift_resamples_for_non_qng_optimizer(self, dummy_simulator, mocker):
        qaoa = QAOA(
            MaxCutProblem(make_bull_graph()),
            n_layers=1,
            trotterization_strategy=QDrift(
                sampling_budget=2,
                n_hamiltonians_per_iteration=1,
                seed=42,
            ),
            optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
            max_iterations=2,
            backend=dummy_simulator,
        )
        trotter_stage = next(
            stage
            for stage in self._cost_pipeline(qaoa).stages
            if isinstance(stage, TrotterSpecStage)
        )
        expand_spy = mocker.spy(trotter_stage, "expand")

        qaoa.run(perform_final_computation=False)

        assert expand_spy.call_count >= 2

    def test_construction_does_not_eager_build(
        self, dummy_simulator, default_optimizer
    ):
        """Construction leaves the compatibility factory unpopulated."""
        qaoa = self._make_qaoa(
            ExactTrotterization(), dummy_simulator, default_optimizer
        )
        assert qaoa._cost_circuit is None

    def test_cache_independent_per_instance(self, dummy_simulator, default_optimizer):
        qaoa1 = self._make_qaoa(
            ExactTrotterization(), dummy_simulator, default_optimizer
        )
        qaoa2 = self._make_qaoa(
            ExactTrotterization(), dummy_simulator, default_optimizer
        )

        assert (
            self._cost_pipeline(qaoa1)._forward_cache
            is not self._cost_pipeline(qaoa2)._forward_cache
        )

    def test_depth_rebuild_invalidates_persistent_entries(
        self, dummy_simulator, default_optimizer
    ):
        qaoa = IterativeQAOA(
            MaxCutProblem(make_bull_graph()),
            max_depth=2,
            backend=dummy_simulator,
            max_iterations_per_depth=1,
            optimizer=default_optimizer,
        )
        first = self._forward(qaoa)

        qaoa._rebuild_for_depth(2)
        second = self._forward(qaoa)

        assert second.initial_batch is not first.initial_batch


class TestObservableMeasuringContracts(ObservableMeasuringContractsBase):
    @pytest.fixture
    def make_program(self, dummy_simulator, default_optimizer):
        def _make(**kwargs):
            return QAOA(
                MaxCliqueProblem(nx.bull_graph(), is_constrained=True),
                backend=dummy_simulator,
                optimizer=default_optimizer,
                **kwargs,
            )

        return _make
