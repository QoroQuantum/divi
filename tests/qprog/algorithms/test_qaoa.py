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
from divi.pipeline.stages import TrotterSpecStage, _trotter_spec_stage
from divi.qprog import (
    QAOA,
    ScipyMethod,
    ScipyOptimizer,
)
from divi.qprog.algorithms import IterativeQAOA
from divi.qprog.algorithms._qaoa import _hamiltonian_parameter_frequency
from divi.qprog.checkpointing import CheckpointConfig
from divi.qprog.initial_states import SuperpositionState
from divi.qprog.problems import (
    BinaryOptimizationProblem,
    MaxCliqueProblem,
    MaxCutProblem,
    MinVertexCoverProblem,
    QAOAProblem,
)
from divi.reporting._events import EventKind, TerminalStatus
from tests._helpers import exact_match
from tests.qprog._program_contracts import (
    ObservableMeasuringContractsBase,
    edit_checkpointed_subclass_state,
    verify_correct_circuit_count,
    verify_load_state_rejects_missing_state_key,
)
from tests.qprog.algorithms._helpers import seed_best_probs
from tests.qprog.problems._helpers import QUBO_MATRIX, make_bull_graph

_INCOMMENSURATE_MESSAGE = exact_match(
    "QAOA parameter-shift gradients require commensurate Hamiltonian "
    "coefficients. Use a gradient-free optimizer or SPSA for this problem."
)


class TestGeneralQAOA:
    def test_qaoa_optimization_runs_cost_pipeline(
        self, mocker, gradient_free_optimizer, default_test_simulator
    ):
        """The cost pipeline is invoked during the optimization loop.

        Note: this is an implementation-coupling test that spies on an internal
        pipeline. The behavioral outcomes (losses, circuit counts) are verified
        by dedicated end-to-end tests.
        """
        qaoa_problem = QAOA(
            MaxCliqueProblem(nx.bull_graph(), use_constrained_mixer=True),
            n_layers=1,
            optimizer=gradient_free_optimizer,
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
            MaxCliqueProblem(nx.bull_graph(), use_constrained_mixer=True),
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
            MaxCliqueProblem(nx.bull_graph(), use_constrained_mixer=True),
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
        self, gradient_free_optimizer, dummy_simulator
    ):
        qaoa_problem = QAOA(
            MaxCliqueProblem(nx.bull_graph(), use_constrained_mixer=True),
            n_layers=1,
            optimizer=gradient_free_optimizer,
            max_iterations=1,
            backend=dummy_simulator,
        )

        qaoa_problem.run()

        verify_correct_circuit_count(qaoa_problem)

    def test_gradient_rule_uses_the_hamiltonian_frequency_content(
        self, dummy_simulator, default_optimizer
    ):
        qaoa = QAOA(
            MaxCutProblem(nx.bull_graph()),
            n_layers=2,
            optimizer=default_optimizer,
            max_iterations=1,
            backend=dummy_simulator,
        )

        assert qaoa._parameter_frequencies() == [
            (1.0, 5),
            (2.0, 5),
            (1.0, 5),
            (2.0, 5),
        ]
        shifts, weights = qaoa._grad_shift_rule
        assert shifts.shape == (40, 4)
        assert weights.shape == (4, 40)

    @pytest.mark.parametrize(
        "hamiltonian",
        [
            SparsePauliOp(["ZI", "IZ"], coeffs=[0.5, np.sqrt(2)]),
            SparsePauliOp("Z", 0.5 / 10001),
        ],
        ids=["irrational-ratio", "denominator-past-the-limit"],
    )
    def test_hamiltonian_frequency_family_rejects_incommensurate_weights(
        self, hamiltonian
    ):
        with pytest.raises(NotImplementedError, match=_INCOMMENSURATE_MESSAGE):
            _hamiltonian_parameter_frequency(hamiltonian)

    @pytest.mark.parametrize(
        "hamiltonian, expected",
        [
            (SparsePauliOp(["Z", "I"], coeffs=[1.0, 0.3]), (2.0, 1)),
            (SparsePauliOp("Z"), (2.0, 1)),
            (SparsePauliOp("I"), (1.0, 1)),
            (SparsePauliOp("Z", 5e-13), (1.0, 1)),
            (SparsePauliOp(["ZI", "IZ"], coeffs=[0.25, 0.5]), (0.5, 3)),
            (
                SparsePauliOp(["ZI", "IZ", "ZZ"], coeffs=[0.37, 0.42, 0.58]),
                (0.02, 137),
            ),
        ],
        ids=[
            "identity-ignored",
            "single-term",
            "identity-only",
            "below-threshold",
            "commensurate-weights",
            "large-shift-rule",
        ],
    )
    def test_hamiltonian_frequency_family(self, hamiltonian, expected):
        assert _hamiltonian_parameter_frequency(hamiltonian) == expected

    def test_qdrift_has_no_parameter_shift_rule(
        self, gradient_free_optimizer, dummy_simulator
    ):
        qaoa = QAOA(
            MaxCutProblem(nx.path_graph(3)),
            trotterization_strategy=QDrift(sampling_budget=2, seed=42),
            optimizer=gradient_free_optimizer,
            backend=dummy_simulator,
        )

        with pytest.raises(
            NotImplementedError,
            match=exact_match(
                "QAOA has no parameter-shift gradient for stochastic or approximate "
                "trotterization. Use a gradient-free optimizer or SPSA."
            ),
        ):
            qaoa._parameter_frequencies()

    def test_shift_limit_admits_its_own_value(
        self, gradient_free_optimizer, dummy_simulator
    ):
        """The path graph needs 4 cost and 6 mixer evaluations per parameter."""
        qaoa = QAOA(
            MaxCutProblem(nx.path_graph(3)),
            optimizer=gradient_free_optimizer,
            backend=dummy_simulator,
            max_shift_evaluations_per_parameter=6,
        )

        assert [2 * order for _, order in qaoa._parameter_frequencies()] == [4, 6]

    def test_qaoa_accepts_a_shift_limit_of_one(
        self, gradient_free_optimizer, dummy_simulator
    ):
        qaoa = QAOA(
            MaxCutProblem(nx.path_graph(2)),
            optimizer=gradient_free_optimizer,
            backend=dummy_simulator,
            max_shift_evaluations_per_parameter=1,
        )

        assert qaoa.max_shift_evaluations_per_parameter == 1

    @pytest.mark.parametrize("limit", [0, -1, True, 1.5])
    def test_qaoa_rejects_invalid_shift_evaluation_limit(
        self,
        limit,
        gradient_free_optimizer,
        default_test_simulator,
    ):
        with pytest.raises(ValueError, match="positive integer or None"):
            QAOA(
                MaxCutProblem(nx.path_graph(2)),
                optimizer=gradient_free_optimizer,
                backend=default_test_simulator,
                max_shift_evaluations_per_parameter=limit,
            )

    def test_qaoa_shift_limit_reports_full_gradient_cost(
        self,
        gradient_free_optimizer,
        default_test_simulator,
    ):
        qaoa = QAOA(
            MaxCutProblem(nx.complete_graph(3)),
            optimizer=gradient_free_optimizer,
            backend=default_test_simulator,
        )
        qaoa.cost_hamiltonian = SparsePauliOp(
            ["ZII", "IZI", "IIZ"],
            coeffs=[0.37, 0.42, 0.58],
        )

        with pytest.raises(
            NotImplementedError,
            match=exact_match(
                "QAOA parameter-shift gradients require 274 cost and 6 mixer "
                "circuit evaluations per parameter; the full 2-parameter gradient "
                "requires 280 evaluations, and the per-parameter limit is 256. "
                "Increase max_shift_evaluations_per_parameter, set it to None to "
                "opt out, or use a gradient-free optimizer or SPSA for this problem."
            ),
        ):
            qaoa._parameter_frequencies()

        qaoa.max_shift_evaluations_per_parameter = None
        assert qaoa._parameter_frequencies()[0] == (0.02, 137)

    def test_a_gradient_free_run_never_builds_a_shift_rule(
        self, gradient_free_optimizer, dummy_simulator
    ):
        """The refusal above must not break QAOA's supported optimizers."""
        qaoa = QAOA(
            MaxCutProblem(nx.bull_graph()),
            n_layers=1,
            optimizer=gradient_free_optimizer,
            max_iterations=1,
            backend=dummy_simulator,
        )

        qaoa.run()

        assert qaoa.current_iteration == 1


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
    def test_multi_sample_qaoa_e2e_solution(self, default_test_simulator):
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

        emitted = []
        with qaoa._bind_progress_emitter(emitted.append):
            qaoa.run()

        assert qaoa.total_circuit_count > 0
        assert qaoa.total_run_time >= 0
        iteration_events = [
            event for event in emitted if event.kind is EventKind.ADVANCE
        ]
        assert len(qaoa.losses_history) == len(iteration_events)
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
        self, mocker, default_test_simulator
    ):
        """Every sampled Hamiltonian's histogram reaches the merge, not just one."""
        n_hamiltonians = 3
        strategy = QDrift(
            keep_fraction=0.5,
            sampling_budget=4,
            n_hamiltonians_per_iteration=n_hamiltonians,
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
        spy = mocker.spy(_trotter_spec_stage, "reduce_merge_histograms")

        qaoa.run()

        assert spy.call_count > 0
        for call in spy.call_args_list:
            grouped = call.args[0]
            assert all(len(values) == n_hamiltonians for values in grouped.values())

        probs = next(iter(qaoa.best_probs.values()))
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


def test_loaded_state_keeps_the_constructed_trotterization_strategy(
    dummy_simulator, default_optimizer
):
    def make_qaoa(strategy):
        return QAOA(
            MaxCutProblem(nx.bull_graph()),
            trotterization_strategy=strategy,
            backend=dummy_simulator,
            optimizer=default_optimizer,
        )

    source = make_qaoa(ExactTrotterization(keep_top_n=3))
    strategy = ExactTrotterization(keep_top_n=2)
    target = make_qaoa(strategy)

    target._load_subclass_state(source._save_subclass_state())

    assert target.trotterization_strategy is strategy


def test_loaded_solution_is_decoded_by_the_constructed_problem(
    dummy_simulator, default_optimizer
):
    """A checkpoint restored onto a relabelled graph reports the new labels."""
    graph = nx.bull_graph()

    def make_qaoa(g):
        return QAOA(
            MaxCutProblem(g), backend=dummy_simulator, optimizer=default_optimizer
        )

    source = make_qaoa(graph)
    source._results["solution_bitstring"] = "10100"
    target = make_qaoa(nx.relabel_nodes(graph, {node: f"n{node}" for node in graph}))

    target._load_subclass_state(source._save_subclass_state())

    assert target.solution == target._decode_solution_fn("10100")
    assert target.solution != source._decode_solution_fn("10100")
    assert target.solution_bitstring == "10100"


def _path_qaoa(dummy_simulator, default_optimizer, **kwargs):
    return QAOA(
        MaxCutProblem(nx.path_graph(3)),
        backend=dummy_simulator,
        optimizer=default_optimizer,
        **kwargs,
    )


def _checkpointed_path_qaoa(backend, optimizer, checkpoint_dir, **kwargs):
    """Run a path-graph QAOA for one iteration, checkpointing into ``checkpoint_dir``."""
    qaoa = _path_qaoa(backend, optimizer, **kwargs)
    qaoa.run(
        max_iterations=1,
        perform_final_computation=False,
        checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir),
    )
    return qaoa


@pytest.mark.parametrize(
    "drop_limit, expected_limit",
    [(False, 7), (True, 9)],
    ids=["saved-limit", "legacy-checkpoint-keeps-constructor-limit"],
)
def test_loaded_state_restores_metadata_and_shift_limit(
    dummy_simulator, default_optimizer, tmp_path, drop_limit, expected_limit
):
    _checkpointed_path_qaoa(
        dummy_simulator,
        default_optimizer,
        tmp_path,
        max_shift_evaluations_per_parameter=7,
    )
    if drop_limit:
        edit_checkpointed_subclass_state(
            tmp_path, lambda data: data.pop("max_shift_evaluations_per_parameter")
        )

    target = QAOA.load_state(
        tmp_path,
        backend=dummy_simulator,
        problem=MaxCutProblem(nx.path_graph(3)),
        max_shift_evaluations_per_parameter=9,
    )

    assert target.problem_metadata == {}
    assert target.max_shift_evaluations_per_parameter == expected_limit


def test_checkpoint_restores_numpy_problem_metadata_as_plain_python(
    dummy_simulator, default_optimizer, tmp_path
):
    source = _path_qaoa(dummy_simulator, default_optimizer)
    source.problem_metadata = {
        "weights": np.array([1.5, 2.0]),
        "nested": {"size": np.int64(3)},
        "pairs": [(0, np.float64(0.5))],
    }
    source.run(
        max_iterations=1,
        perform_final_computation=False,
        checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path),
    )

    target = QAOA.load_state(
        tmp_path, backend=dummy_simulator, problem=MaxCutProblem(nx.path_graph(3))
    )

    assert target.problem_metadata == {
        "weights": [1.5, 2.0],
        "nested": {"size": 3},
        "pairs": [[0, 0.5]],
    }


def test_load_rejects_a_state_missing_required_keys(
    dummy_simulator, default_optimizer, tmp_path
):
    _checkpointed_path_qaoa(dummy_simulator, default_optimizer, tmp_path)

    verify_load_state_rejects_missing_state_key(
        tmp_path,
        "loss_constant",
        lambda: QAOA.load_state(
            tmp_path, backend=dummy_simulator, problem=MaxCutProblem(nx.path_graph(3))
        ),
    )


def test_qaoa_defaults(dummy_simulator, default_optimizer):
    qaoa = QAOA(
        _make_problem(
            cost=SparsePauliOp(["ZI", "IZ"]), mixer=SparsePauliOp(["XI", "IX"])
        ),
        backend=dummy_simulator,
        optimizer=default_optimizer,
    )

    assert qaoa.max_iterations == 10
    assert qaoa.problem_metadata == {}


def test_qaoa_rejects_a_non_initial_state(dummy_simulator, default_optimizer):
    with pytest.raises(
        TypeError,
        match=exact_match(
            "initial_state must be an InitialState instance or None, got str"
        ),
    ):
        _path_qaoa(dummy_simulator, default_optimizer, initial_state="zeros")


def test_cost_pipeline_circuits_render_at_the_program_precision(
    dummy_simulator, default_optimizer
):
    qaoa = _path_qaoa(dummy_simulator, default_optimizer, precision=4)

    forward = qaoa._build_preprocessor_pipeline(
        qaoa.cost_preprocessor()
    ).run_forward_pass(qaoa.cost_hamiltonian, qaoa._build_pipeline_env())

    assert {meta.precision for meta in forward.initial_batch.values()} == {4}


class _RepairToFullCover(MinVertexCoverProblem):
    """Repairs every infeasible cover to the all-selected one."""

    def repair_infeasible_bitstring(self, bitstring):
        return "1" * len(bitstring), "repaired", float(len(bitstring))


@pytest.fixture
def ramp_qaoa(dummy_simulator, default_optimizer):
    """Min vertex cover on a 4-node path, sampled with ``P(i) ∝ i + 1``.

    The covers of the path ``0-1-2-3`` are 1010, 0110, 0101, 1110, 1101, 1011,
    0111 and 1111; every other bitstring leaves an edge uncovered.
    """
    qaoa = QAOA(
        _RepairToFullCover(nx.path_graph(4)),
        backend=dummy_simulator,
        optimizer=default_optimizer,
    )
    seed_best_probs(qaoa, {format(i, "04b"): (i + 1) / 136 for i in range(16)}, "0")
    return qaoa


def test_get_top_solutions_defaults_to_ten_undecoded(ramp_qaoa):
    solutions = ramp_qaoa.get_top_solutions()

    assert len(solutions) == 10
    assert all(solution.decoded is None for solution in solutions)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        (
            {"min_prob": 10 / 136},
            ["1111", "1110", "1101", "1100", "1011", "1010", "1001"],
        ),
        (
            {"n": 0, "feasibility": "filter"},
            ["1010", "0110", "0101", "1110", "1101", "1011", "0111", "1111"],
        ),
        (
            {"n": 0, "feasibility": "filter", "min_prob": 10 / 136},
            ["1010", "1110", "1101", "1011", "1111"],
        ),
    ],
    ids=["ignore-min-prob", "filter-all", "filter-min-prob"],
)
def test_get_top_solutions_selection(ramp_qaoa, kwargs, expected):
    solutions = ramp_qaoa.get_top_solutions(**kwargs)

    assert [solution.bitstring for solution in solutions] == expected
    assert all(solution.decoded is None for solution in solutions)


def test_get_top_solutions_repair_merges_into_the_repaired_cover(ramp_qaoa):
    """The eight infeasible samples (total weight 47/136) fold into 1111."""
    solutions = ramp_qaoa.get_top_solutions(n=0, feasibility="repair")

    full_cover = next(s for s in solutions if s.bitstring == "1111")
    assert full_cover.prob == pytest.approx((16 + 47) / 136)
    assert len(solutions) == 8


@pytest.mark.parametrize(
    "kwargs, match",
    [({"feasibility": "bogus"}, "feasibility"), ({"n": -1}, "non-negative")],
)
def test_get_top_solutions_rejects_bad_arguments(
    kwargs, match, dummy_simulator, default_optimizer
):
    qaoa = QAOA(
        BinaryOptimizationProblem(QUBO_MATRIX),
        n_layers=1,
        optimizer=default_optimizer,
        backend=dummy_simulator,
    )
    with pytest.raises(ValueError, match=match):
        qaoa.get_top_solutions(**kwargs)


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
        with pytest.raises(
            RuntimeError, match=r"not available yet.*sample_solution\(\)"
        ):
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

    def test_standalone_sampling_uses_one_direct_progress_session(
        self, default_test_simulator, default_optimizer, recording_direct_sessions
    ):
        qaoa = self._make_qaoa(default_test_simulator, default_optimizer)

        qaoa.sample_solution(np.array([0.1, 0.2]))

        assert len(recording_direct_sessions) == 1
        session = recording_direct_sessions[0]
        terminal_events = [
            event for event in session.emitted if event.kind is EventKind.FINISH
        ]
        assert len(terminal_events) == 1
        assert terminal_events[0].terminal_status is TerminalStatus.SUCCESS
        assert session.state.get(qaoa._progress_key).terminal_status is (
            TerminalStatus.SUCCESS
        )

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

        assert qaoa.current_iteration == 1
        assert len(qaoa.losses_history) == 1
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
    @pytest.mark.parametrize(
        "cost_terms, mixer_terms, match",
        [
            pytest.param(
                ["IZZ", "ZZI"],
                ["IIIX", "IIXI", "IXII", "XIII"],
                r"wire_labels has 3 entries.*mixer_hamiltonian\.num_qubits is 4",
                id="mixer_wider_than_cost",
            ),
            pytest.param(
                ["IIZZ", "ZZII"],
                ["IIX", "IXI", "XII"],
                r"wire_labels has 4 entries.*mixer_hamiltonian\.num_qubits is 3",
                id="cost_wider_than_mixer",
            ),
        ],
    )
    def test_width_mismatch_raises(
        self, dummy_simulator, default_optimizer, cost_terms, mixer_terms, match
    ):
        prob = _make_problem(
            cost=SparsePauliOp.from_list([(t, 1.0) for t in cost_terms]),
            mixer=SparsePauliOp.from_list([(t, 1.0) for t in mixer_terms]),
        )
        with pytest.raises(ValueError, match=match):
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
                MaxCliqueProblem(nx.bull_graph(), use_constrained_mixer=True),
                backend=dummy_simulator,
                optimizer=default_optimizer,
                **kwargs,
            )

        return _make
