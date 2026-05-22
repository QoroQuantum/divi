# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0


import warnings

import networkx as nx
import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

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
from divi.qprog.algorithms import IterativeQAOA, SuperpositionState
from divi.qprog.problems import (
    BinaryOptimizationProblem,
    MaxCliqueProblem,
    MaxCutProblem,
    QAOAProblem,
)
from tests.qprog._program_contracts import verify_correct_circuit_count
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

        # Spy on the cost pipeline's run method
        spy = mocker.spy(qaoa_problem._cost_pipeline, "run")

        # Mock final computation to isolate optimization phase
        mocker.patch.object(qaoa_problem, "sample_solution")

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
        mocker.patch.object(qaoa_problem, "sample_solution")

        qaoa_problem.run()

        assert spy.call_args.kwargs["initial_spec"] is qaoa_problem.cost_hamiltonian

    def test_qaoa_final_computation_runs_measurement_pipeline(
        self, mocker, optimizer, default_test_simulator
    ):
        """The measurement pipeline is invoked during final computation.

        Note: this is an implementation-coupling test that spies on an internal
        pipeline. The behavioral outcomes (solution extraction) are verified
        by dedicated end-to-end tests.
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

        # Spy on measurement pipeline
        spy = mocker.spy(qaoa_problem._measurement_pipeline, "run")

        qaoa_problem.sample_solution()

        # Measurement pipeline should be called once
        assert spy.call_count == 1

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

    def test_solution_bitstring_after_run(self, default_test_simulator):
        """``solution_bitstring`` exposes the raw measured bitstring as a string."""
        qaoa = QAOA(
            BinaryOptimizationProblem(QUBO_MATRIX),
            backend=default_test_simulator,
            max_iterations=1,
        )
        qaoa.run()
        bs = qaoa.solution_bitstring
        assert isinstance(bs, str)
        assert len(bs) == qaoa.n_qubits
        assert set(bs) <= {"0", "1"}
        # Bitstring corresponds to the same state ``solution`` was decoded from.
        assert [int(c) for c in bs] == list(qaoa.solution)

    def test_solution_bitstring_before_run_raises(self, default_test_simulator):
        """Accessing ``solution_bitstring`` before ``.run()`` is an error."""
        qaoa = QAOA(
            BinaryOptimizationProblem(QUBO_MATRIX),
            backend=default_test_simulator,
            max_iterations=1,
        )
        with pytest.raises(RuntimeError, match="Call .run\\(\\) first"):
            _ = qaoa.solution_bitstring


class TestSampleSolution:
    """Tests for ``sample_solution(params)`` — sampling without training."""

    def _make_qaoa(self, backend, n_layers=1):
        return QAOA(
            BinaryOptimizationProblem(QUBO_MATRIX),
            n_layers=n_layers,
            backend=backend,
            max_iterations=1,
        )

    def test_populates_solution_and_bitstring(self, default_test_simulator):
        """``sample_solution`` produces the same kind of outputs as ``run()``'s final step."""
        qaoa = self._make_qaoa(default_test_simulator)
        params = np.array([0.1, 0.2])

        qaoa.sample_solution(params)

        assert qaoa.solution_bitstring is not None
        assert isinstance(qaoa.solution_bitstring, str)
        assert len(qaoa.solution_bitstring) == qaoa.n_qubits
        assert qaoa.best_probs  # measurement probs were populated
        assert qaoa.total_circuit_count > 0

    def test_skips_cost_pipeline(self, mocker, default_test_simulator):
        """``sample_solution`` must not dispatch any EXPECTATION job."""
        qaoa = self._make_qaoa(default_test_simulator)
        cost_spy = mocker.spy(qaoa._cost_pipeline, "run")
        meas_spy = mocker.spy(qaoa._measurement_pipeline, "run")

        qaoa.sample_solution(np.array([0.1, 0.2]))

        assert cost_spy.call_count == 0
        assert meas_spy.call_count == 1

    def test_does_not_mutate_best_params(self, default_test_simulator):
        """After ``run()`` then ``sample_solution(other_params)``, ``best_params`` is unchanged."""
        qaoa = self._make_qaoa(default_test_simulator)
        qaoa.run()
        trained = qaoa.best_params.copy()

        other = trained + 1.0
        qaoa.sample_solution(other)

        np.testing.assert_array_equal(qaoa.best_params, trained)

    def test_wrong_shape_raises(self, default_test_simulator):
        """``params`` with mismatched per-set size raises ``ValueError``."""
        qaoa = self._make_qaoa(default_test_simulator, n_layers=2)  # expects 4 params

        with pytest.raises(ValueError, match="does not match"):
            qaoa.sample_solution(np.array([0.1, 0.2, 0.3]))  # 3 != 4

    def test_no_args_before_run_raises(self, default_test_simulator):
        """``sample_solution()`` before ``run()`` raises a clear ``RuntimeError``."""
        qaoa = self._make_qaoa(default_test_simulator)

        with pytest.raises(RuntimeError, match="call run\\(\\) first"):
            qaoa.sample_solution()

    def test_returns_self(self, default_test_simulator):
        """``sample_solution`` returns the program for method chaining."""
        qaoa = self._make_qaoa(default_test_simulator)
        assert qaoa.sample_solution(np.array([0.1, 0.2])) is qaoa

    def test_does_not_mutate_optimizer_state(self, default_test_simulator):
        """``sample_solution`` leaves optimizer-side state (losses, iterations) untouched."""
        qaoa = self._make_qaoa(default_test_simulator)

        qaoa.sample_solution(np.array([0.1, 0.2]))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            assert qaoa.losses_history == []
        assert qaoa.current_iteration == 0
        assert qaoa.optimize_result is None

    def test_followed_by_run_works(self, default_test_simulator):
        """Calling ``run()`` after ``sample_solution()`` produces a normal training run."""
        qaoa = self._make_qaoa(default_test_simulator)
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
    def test_mixer_wider_than_cost_raises(self, dummy_simulator):
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
            QAOA(prob, backend=dummy_simulator)

    def test_cost_wider_than_mixer_raises(self, dummy_simulator):
        prob = _make_problem(
            cost=SparsePauliOp.from_list([("IIZZ", 1.0), ("ZZII", 1.0)]),
            mixer=SparsePauliOp.from_list([("IIX", 1.0), ("IXI", 1.0), ("XII", 1.0)]),
        )
        with pytest.raises(
            ValueError,
            match=r"wire_labels has 4 entries.*mixer_hamiltonian\.num_qubits is 3",
        ):
            QAOA(prob, backend=dummy_simulator)

    def test_wire_labels_misaligned_with_hamiltonians_raises(self, dummy_simulator):
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
            QAOA(prob, backend=dummy_simulator)

    def test_isolated_node_graph_with_wire_labels_succeeds(self, dummy_simulator):
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (0, 2)])
        g.add_node(3)
        qaoa = QAOA(MaxCutProblem(g), backend=dummy_simulator)
        assert qaoa.n_qubits == 4

    def test_isolated_node_maxcut_runs_end_to_end(self, default_test_simulator):
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (0, 2)])
        g.add_node(3)
        qaoa = QAOA(
            MaxCutProblem(g),
            backend=default_test_simulator,
            max_iterations=1,
            n_layers=1,
        )
        qaoa.run()
        assert len(qaoa.solution_bitstring) == 4
        assert set(qaoa.solution_bitstring) <= {"0", "1"}
        assert qaoa.solution is not None


class TestCostMetaCircuitCache:
    """``_cost_meta_circuit_factory`` memoizes built MetaCircuits per
    ``ham_id`` for stateless trotterization strategies, where the operator
    is deterministic across calls.  Stateful strategies (e.g. QDrift) must
    bypass the cache because their operator is resampled each call."""

    @staticmethod
    def _make_qaoa(strategy, backend):
        return QAOA(
            MaxCutProblem(make_bull_graph()),
            n_layers=1,
            trotterization_strategy=strategy,
            backend=backend,
            max_iterations=1,
        )

    def test_stateless_strategy_caches_by_ham_id(self, dummy_simulator):
        qaoa = self._make_qaoa(ExactTrotterization(), dummy_simulator)
        m1 = qaoa._cost_meta_circuit_factory(qaoa.cost_hamiltonian, ham_id=0)
        m2 = qaoa._cost_meta_circuit_factory(qaoa.cost_hamiltonian, ham_id=0)
        assert m1 is m2

    def test_stateless_distinct_ham_ids_cached_separately(self, dummy_simulator):
        qaoa = self._make_qaoa(ExactTrotterization(), dummy_simulator)
        m1 = qaoa._cost_meta_circuit_factory(qaoa.cost_hamiltonian, ham_id=0)
        m2 = qaoa._cost_meta_circuit_factory(qaoa.cost_hamiltonian, ham_id=1)
        assert m1 is not m2
        size = qaoa._params.size
        assert qaoa._cost_meta_cache[(0, size)] is m1
        assert qaoa._cost_meta_cache[(1, size)] is m2

    def test_stateless_skips_circuit_build_on_cache_hit(self, dummy_simulator, mocker):
        qaoa = self._make_qaoa(ExactTrotterization(), dummy_simulator)

        # Prime the cache, then start counting.
        qaoa._cost_meta_circuit_factory(qaoa.cost_hamiltonian, ham_id=0)
        spy = mocker.spy(qaoa, "_build_qaoa_qiskit_circuit")

        for _ in range(3):
            qaoa._cost_meta_circuit_factory(qaoa.cost_hamiltonian, ham_id=0)
        spy.assert_not_called()

    def test_stateful_strategy_skips_cache(self, dummy_simulator):
        strategy = QDrift(
            sampling_budget=2,
            n_hamiltonians_per_iteration=1,
            seed=42,
        )
        qaoa = self._make_qaoa(strategy, dummy_simulator)
        m1 = qaoa._cost_meta_circuit_factory(qaoa.cost_hamiltonian, ham_id=0)
        m2 = qaoa._cost_meta_circuit_factory(qaoa.cost_hamiltonian, ham_id=0)
        assert m1 is not m2
        assert qaoa._cost_meta_cache == {}

    def test_construction_does_not_eager_build(self, dummy_simulator):
        """Construction leaves ``_meta_circuit_factories`` and ``_cost_meta_cache``
        unpopulated; the heavy build runs on first access via the
        ``meta_circuit_factories`` property."""
        qaoa = self._make_qaoa(ExactTrotterization(), dummy_simulator)
        assert qaoa._meta_circuit_factories is None
        assert qaoa._cost_meta_cache == {}

    def test_cache_independent_per_instance(self, dummy_simulator):
        qaoa1 = self._make_qaoa(ExactTrotterization(), dummy_simulator)
        qaoa2 = self._make_qaoa(ExactTrotterization(), dummy_simulator)

        # Construction is now lazy (no eager build), so caches start empty.
        # Seed each independently and verify they are isolated.
        qaoa1._cost_meta_circuit_factory(qaoa1.cost_hamiltonian, ham_id=0)
        qaoa2._cost_meta_circuit_factory(qaoa2.cost_hamiltonian, ham_id=0)
        assert qaoa1._cost_meta_cache is not qaoa2._cost_meta_cache
        qaoa1._cost_meta_cache.clear()
        assert qaoa1._cost_meta_cache == {}
        assert qaoa2._cost_meta_cache  # untouched

    def test_cache_self_invalidates_on_depth_rebuild(self, dummy_simulator):
        """IterativeQAOA mutates ``self._params`` per depth.  The cache key
        embeds the parameter size, so the depth-2 lookup misses, builds a
        new MetaCircuit, and stores it under a distinct key — without the
        subclass needing to clear the cache.
        """
        qaoa = IterativeQAOA(
            MaxCutProblem(make_bull_graph()),
            max_depth=2,
            backend=dummy_simulator,
            max_iterations_per_depth=1,
        )
        m_d1 = qaoa._cost_meta_circuit_factory(qaoa.cost_hamiltonian, ham_id=0)
        d1_size = qaoa._params.size

        qaoa._rebuild_for_depth(2)
        d2_size = qaoa._params.size
        assert d2_size != d1_size

        m_d2 = qaoa._cost_meta_circuit_factory(qaoa.cost_hamiltonian, ham_id=0)
        assert m_d2 is not m_d1
        # Both entries coexist under distinct keys.
        assert qaoa._cost_meta_cache[(0, d1_size)] is m_d1
        assert qaoa._cost_meta_cache[(0, d2_size)] is m_d2
