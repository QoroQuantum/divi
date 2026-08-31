# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import shutil

import networkx as nx
import numpy as np
import pytest

from divi.qprog import (
    QAOA,
    InterpolationStrategy,
    IterativeQAOA,
    MonteCarloOptimizer,
    ScipyMethod,
    ScipyOptimizer,
)
from divi.qprog.algorithms._iterative_qaoa import (
    _chebyshev,
    _fourier,
    _interp,
    interpolate_qaoa_params,
)
from divi.qprog.checkpointing import CheckpointConfig, list_checkpoints
from divi.qprog.problems import (
    BinaryOptimizationProblem,
    MaxCliqueProblem,
    MaxCutProblem,
)
from divi.reporting._events import EventKind, ProgressEvent, TerminalStatus
from tests.qprog.problems._helpers import QUBO_MATRIX, QUBO_SOLUTION, make_bull_graph


class TestInterp:
    def test_output_length(self):
        u = np.array([1.0, 2.0, 3.0])
        result = _interp(u)
        assert len(result) == 4

    def test_p1_to_p2(self):
        """Depth 1 → 2: u = [a] → [a, a] (boundary blending)."""
        u = np.array([0.5])
        result = _interp(u)
        assert len(result) == 2
        # j=0: (0/1)*0 + (1/1)*u[0] = 0.5
        # j=1: (1/1)*u[0] + (0/1)*0 = 0.5
        np.testing.assert_allclose(result, [0.5, 0.5])

    def test_known_values(self):
        """Verify INTERP formula with hand-computed values."""
        u = np.array([1.0, 2.0])  # p=2
        result = _interp(u)
        # j=0: (0/2)*0 + (2/2)*1.0 = 1.0
        # j=1: (1/2)*1.0 + (1/2)*2.0 = 1.5
        # j=2: (2/2)*2.0 + (0/2)*0 = 2.0
        np.testing.assert_allclose(result, [1.0, 1.5, 2.0])

    def test_zero_params_stay_zero(self):
        u = np.zeros(5)
        result = _interp(u)
        np.testing.assert_allclose(result, np.zeros(6))


class TestFourier:
    def test_output_length(self):
        u = np.array([1.0, 2.0, 3.0])
        result = _fourier(u)
        assert len(result) == 4

    def test_round_trip_identity(self):
        """With k=p DCT-II basis terms, fitting and reconstructing is exact."""
        rng = np.random.default_rng(42)
        for p in [2, 3, 5]:
            u = rng.uniform(-1, 1, p)
            j_grid = np.arange(p, dtype=np.float64)
            l_terms = np.arange(p, dtype=np.float64)
            basis = np.cos(np.outer(np.pi * (2 * j_grid + 1) / (2 * p), l_terms))
            coeffs, *_ = np.linalg.lstsq(basis, u, rcond=None)
            reconstructed = basis @ coeffs
            np.testing.assert_allclose(reconstructed, u, atol=1e-10)

    def test_p1_to_p2(self):
        u = np.array([1.0])
        result = _fourier(u, n_basis_terms=1)
        assert len(result) == 2


class TestChebyshev:
    def test_output_length(self):
        u = np.array([1.0, 2.0, 3.0])
        result = _chebyshev(u)
        assert len(result) == 4

    def test_round_trip_identity(self):
        """With k=p basis terms, fitting and reconstructing is exact."""
        rng = np.random.default_rng(42)
        for p in [2, 3, 5]:
            u = rng.uniform(-1, 1, p)
            # Build basis at depth p with k=p terms (exact fit)
            j_grid = np.arange(p, dtype=np.float64)
            x_p = np.cos(np.pi * (j_grid + 0.5) / p)
            basis = np.empty((p, p), dtype=np.float64)
            for l in range(p):
                basis[:, l] = np.cos(l * np.arccos(x_p))
            coeffs, *_ = np.linalg.lstsq(basis, u, rcond=None)
            reconstructed = basis @ coeffs
            np.testing.assert_allclose(reconstructed, u, atol=1e-10)

    def test_p1_to_p2(self):
        u = np.array([1.0])
        result = _chebyshev(u, n_basis_terms=1)
        assert len(result) == 2


class TestInterpolateQaoaParams:
    def test_output_length(self):
        params = np.array([0.1, 0.2, 0.3, 0.4])  # depth=2
        result = interpolate_qaoa_params(params, 2, InterpolationStrategy.INTERP)
        assert len(result) == 6  # depth=3 → 2*3=6

    def test_deinterleave_reinterleave(self):
        """Verify beta/gamma are handled independently."""
        betas = np.array([1.0, 2.0])
        gammas = np.array([10.0, 20.0])
        params = np.empty(4)
        params[0::2] = betas
        params[1::2] = gammas

        result = interpolate_qaoa_params(params, 2, InterpolationStrategy.INTERP)

        # Check that betas and gammas were interpolated independently
        result_betas = result[0::2]
        result_gammas = result[1::2]
        np.testing.assert_allclose(result_betas, _interp(betas))
        np.testing.assert_allclose(result_gammas, _interp(gammas))

    @pytest.mark.parametrize("strategy", list(InterpolationStrategy))
    def test_all_strategies_produce_correct_length(self, strategy):
        params = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # depth=3
        result = interpolate_qaoa_params(params, 3, strategy)
        assert len(result) == 8  # depth=4


class TestIterativeQAOA:
    def test_run_uses_one_direct_session_for_the_full_depth_schedule(
        self, default_test_simulator, mocker
    ):
        iterative = IterativeQAOA(
            MaxCutProblem(make_bull_graph()),
            max_depth=2,
            max_iterations_per_depth=1,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        )
        session = mocker.MagicMock()
        session.__enter__.return_value = session
        direct = mocker.patch(
            "divi.qprog.quantum_program.ProgressSession.direct",
            return_value=session,
        )

        iterative.run(perform_final_computation=False)

        direct.assert_called_once()
        session.__enter__.assert_called_once_with()
        session.__exit__.assert_called_once_with(None, None, None)

    def test_runs_through_depths(self, default_test_simulator):
        graph = make_bull_graph()
        iterative = IterativeQAOA(
            MaxCutProblem(graph),
            max_depth=3,
            strategy=InterpolationStrategy.INTERP,
            max_iterations_per_depth=3,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        )
        iterative.run()

        assert len(iterative.depth_history) == 3
        assert iterative.best_depth in [1, 2, 3]
        assert iterative.solution is not None

    def test_depth_history_structure(self, default_test_simulator):
        graph = make_bull_graph()
        iterative = IterativeQAOA(
            MaxCutProblem(graph),
            max_depth=2,
            strategy=InterpolationStrategy.INTERP,
            max_iterations_per_depth=2,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        )
        iterative.run()

        history = iterative.depth_history
        assert len(history) == 2
        for entry in history:
            assert "depth" in entry
            assert "best_loss" in entry
            assert "best_params" in entry
            assert "n_iterations" in entry

        assert history[0]["depth"] == 1
        assert history[1]["depth"] == 2
        assert len(history[0]["best_params"]) == 2  # depth 1: 2 params
        assert len(history[1]["best_params"]) == 4  # depth 2: 4 params

    def test_total_circuit_count_matches_circuits_submitted(
        self, default_test_simulator, mocker
    ):
        graph = make_bull_graph()
        iterative = IterativeQAOA(
            MaxCutProblem(graph),
            max_depth=3,
            max_iterations_per_depth=2,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        )
        spy = mocker.spy(default_test_simulator, "submit_circuits")

        iterative.run()

        submitted = sum(len(call.args[0]) for call in spy.call_args_list)
        assert iterative.total_circuit_count == submitted

    def test_total_circuit_count_accumulates_across_runs(
        self, default_test_simulator, mocker
    ):
        graph = make_bull_graph()
        iterative = IterativeQAOA(
            MaxCutProblem(graph),
            max_depth=2,
            max_iterations_per_depth=2,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        )
        spy = mocker.spy(default_test_simulator, "submit_circuits")

        iterative.run()
        iterative.run()

        submitted = sum(len(call.args[0]) for call in spy.call_args_list)
        assert iterative.total_circuit_count == submitted

    def test_best_depth_matches_lowest_loss(self, default_test_simulator):
        graph = make_bull_graph()
        iterative = IterativeQAOA(
            MaxCutProblem(graph),
            max_depth=3,
            strategy=InterpolationStrategy.INTERP,
            max_iterations_per_depth=3,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        )
        iterative.run()

        history = iterative.depth_history
        best_entry = min(history, key=lambda d: d["best_loss"])
        assert iterative.best_depth == best_entry["depth"]

    def test_convergence_threshold_early_exit(self, default_test_simulator):
        graph = make_bull_graph()
        iterative = IterativeQAOA(
            MaxCutProblem(graph),
            max_depth=10,
            strategy=InterpolationStrategy.INTERP,
            max_iterations_per_depth=3,
            convergence_threshold=1e10,  # very large → always converges
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        )
        iterative.run()

        # Should stop at depth 2 (first time convergence can be checked)
        assert len(iterative.depth_history) == 2

    def test_max_iterations_per_depth_callable(self, default_test_simulator):
        graph = make_bull_graph()
        iterative = IterativeQAOA(
            MaxCutProblem(graph),
            max_depth=3,
            strategy=InterpolationStrategy.INTERP,
            max_iterations_per_depth=lambda depth: depth + 1,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        )
        iterative.run()

        history = iterative.depth_history
        # Depth 1 → 2 iters, depth 2 → 3 iters, depth 3 → 4 iters
        for entry in history:
            expected_max = entry["depth"] + 1
            assert entry["n_iterations"] <= expected_max

    @pytest.mark.parametrize(
        "strategy",
        [
            InterpolationStrategy.INTERP,
            InterpolationStrategy.FOURIER,
            InterpolationStrategy.CHEBYSHEV,
        ],
    )
    def test_all_strategies_run(self, strategy, default_test_simulator):
        graph = make_bull_graph()
        iterative = IterativeQAOA(
            MaxCutProblem(graph),
            max_depth=3,
            strategy=strategy,
            max_iterations_per_depth=2,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        )
        iterative.run()
        assert len(iterative.depth_history) == 3

    def test_with_monte_carlo_optimizer(self, default_test_simulator):
        graph = make_bull_graph()
        iterative = IterativeQAOA(
            MaxCutProblem(graph),
            max_depth=2,
            strategy=InterpolationStrategy.INTERP,
            max_iterations_per_depth=2,
            backend=default_test_simulator,
            optimizer=MonteCarloOptimizer(population_size=5),
        )
        iterative.run()

        assert len(iterative.depth_history) == 2
        assert iterative.solution is not None

    def test_expected_total_iterations_constant(self, default_test_simulator):
        """_expected_total_iterations equals max_depth * max_iterations_per_depth."""
        graph = make_bull_graph()
        iterative = IterativeQAOA(
            MaxCutProblem(graph),
            max_depth=4,
            strategy=InterpolationStrategy.INTERP,
            max_iterations_per_depth=5,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        )
        assert iterative._expected_total_iterations == 20

    def test_expected_total_iterations_callable(self, default_test_simulator):
        """_expected_total_iterations sums per-depth budgets from callable."""
        graph = make_bull_graph()
        iterative = IterativeQAOA(
            MaxCutProblem(graph),
            max_depth=3,
            strategy=InterpolationStrategy.INTERP,
            max_iterations_per_depth=lambda depth: depth + 1,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        )
        # depth 1 → 2, depth 2 → 3, depth 3 → 4 = 9
        assert iterative._expected_total_iterations == 9

    def test_depth_info_is_emitted_as_typed_progress(self, default_test_simulator):
        """Each depth is exposed through the program's bound event emitter."""
        graph = make_bull_graph()
        iterative = IterativeQAOA(
            MaxCutProblem(graph),
            max_depth=3,
            strategy=InterpolationStrategy.INTERP,
            max_iterations_per_depth=2,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        )
        emitted = []
        with iterative._bind_progress_emitter(emitted.append):
            iterative.run()

        assert [
            event
            for event in emitted
            if event.message is not None and event.message.startswith("Depth")
        ] == [
            ProgressEvent.show(iterative._progress_key, "Depth 1/3"),
            ProgressEvent.show(iterative._progress_key, "Depth 2/3"),
            ProgressEvent.show(iterative._progress_key, "Depth 3/3"),
        ]

    def test_depth_run_forwards_success_finish_for_another_target(
        self, default_test_simulator, mocker
    ):
        iterative = IterativeQAOA(
            MaxCutProblem(make_bull_graph()),
            max_depth=1,
            strategy=InterpolationStrategy.INTERP,
            max_iterations_per_depth=1,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        )
        other_finish = ProgressEvent.finish("other-target", TerminalStatus.SUCCESS)

        def run_one_depth(program, **kwargs):
            program._progress_emitter(other_finish)
            program._best_loss = 0.0
            program._best_params = np.zeros(program.n_params)
            program.current_iteration = 1
            return program

        mocker.patch(
            "divi.qprog.variational_quantum_algorithm."
            "VariationalQuantumAlgorithm.run",
            autospec=True,
            side_effect=run_one_depth,
        )
        emitted = []

        with iterative._bind_progress_emitter(emitted.append):
            iterative.run(perform_final_computation=False)

        assert [
            event
            for event in emitted
            if event.kind is EventKind.FINISH and event.progress_key == "other-target"
        ] == [other_finish]

    def test_n_layers_matches_best_depth(self, default_test_simulator):
        """After run, instance n_layers should match best_depth."""
        graph = make_bull_graph()
        iterative = IterativeQAOA(
            MaxCutProblem(graph),
            max_depth=3,
            strategy=InterpolationStrategy.INTERP,
            max_iterations_per_depth=3,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        )
        iterative.run()
        assert iterative.n_layers == iterative.best_depth


class TestIterativeQAOACheckpointing:
    MAX_DEPTH = 3
    ITERS_PER_DEPTH = 2

    def _run_with_checkpoints(self, backend, checkpoint_dir):
        program = IterativeQAOA(
            MaxCutProblem(make_bull_graph()),
            max_depth=self.MAX_DEPTH,
            max_iterations_per_depth=self.ITERS_PER_DEPTH,
            backend=backend,
            optimizer=MonteCarloOptimizer(population_size=4, n_best_sets=2),
            seed=1997,
        )
        program.run(
            checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir),
            perform_final_computation=False,
        )
        return program

    def _load(self, backend, checkpoint_dir):
        return IterativeQAOA.load_state(
            checkpoint_dir,
            backend=backend,
            problem=MaxCutProblem(make_bull_graph()),
            max_depth=self.MAX_DEPTH,
            max_iterations_per_depth=self.ITERS_PER_DEPTH,
        )

    def test_each_depth_keeps_its_own_checkpoints(
        self, default_test_simulator, tmp_path
    ):
        """Depths write to separate subdirectories instead of overwriting."""
        self._run_with_checkpoints(default_test_simulator, tmp_path)

        for depth in range(1, self.MAX_DEPTH + 1):
            depth_dir = tmp_path / f"depth_{depth:02d}"
            iterations = [info.iteration for info in list_checkpoints(depth_dir)]
            assert iterations == list(range(1, self.ITERS_PER_DEPTH + 1))

    def test_load_resolves_deepest_checkpoint(self, default_test_simulator, tmp_path):
        """load_state picks the deepest depth and rebuilds its ansatz."""
        self._run_with_checkpoints(default_test_simulator, tmp_path)

        loaded = self._load(default_test_simulator, tmp_path)

        assert loaded.n_layers == self.MAX_DEPTH
        assert loaded.best_params.size == loaded.n_params
        # The deepest checkpoint is written mid-depth, so the completed depths
        # are the ones before it.
        assert [entry["depth"] for entry in loaded.depth_history] == [1, 2]

    def test_resume_continues_the_depth_schedule(
        self, default_test_simulator, tmp_path
    ):
        """A resumed run finishes the remaining depths without restarting at 1."""
        self._run_with_checkpoints(default_test_simulator, tmp_path)
        shutil.rmtree(tmp_path / f"depth_{self.MAX_DEPTH:02d}")

        loaded = self._load(default_test_simulator, tmp_path)
        assert loaded.n_layers == self.MAX_DEPTH - 1
        assert len(loaded.depth_history) == self.MAX_DEPTH - 2

        loaded.run(perform_final_computation=False)

        assert [entry["depth"] for entry in loaded.depth_history] == list(
            range(1, self.MAX_DEPTH + 1)
        )

    def test_resume_finishes_a_partially_optimised_depth(
        self, default_test_simulator, tmp_path
    ):
        """A depth interrupted with budget left is continued, not restarted."""
        self._run_with_checkpoints(default_test_simulator, tmp_path)
        deepest = tmp_path / f"depth_{self.MAX_DEPTH:02d}"
        for info in list_checkpoints(deepest):
            if info.iteration > 1:
                shutil.rmtree(info.path)

        loaded = self._load(default_test_simulator, tmp_path)
        assert loaded.n_layers == self.MAX_DEPTH
        assert loaded.current_iteration == 1
        assert loaded.best_params.size == loaded.n_params

        loaded.run(perform_final_computation=False)

        deepest_entry = loaded.depth_history[-1]
        assert deepest_entry["depth"] == self.MAX_DEPTH
        assert deepest_entry["n_iterations"] == self.ITERS_PER_DEPTH
        assert deepest_entry["best_params"].size == 2 * self.MAX_DEPTH

    def test_second_run_does_not_accumulate_depth_history(
        self, default_test_simulator, tmp_path
    ):
        """A repeated run replaces the depth history instead of appending to it."""
        program = self._run_with_checkpoints(default_test_simulator, tmp_path)

        program.run(perform_final_computation=False)

        assert [entry["depth"] for entry in program.depth_history] == list(
            range(1, self.MAX_DEPTH + 1)
        )


@pytest.mark.e2e
class TestIterativeQAOAE2E:
    def test_graph_max_clique_e2e(self, default_test_simulator):
        """Iterative QAOA finds the known max clique for a bull graph."""
        default_test_simulator.set_seed(1997)
        graph = make_bull_graph()

        iterative = IterativeQAOA(
            MaxCliqueProblem(graph),
            max_depth=3,
            strategy=InterpolationStrategy.INTERP,
            max_iterations_per_depth=15,
            backend=default_test_simulator,
            seed=1997,
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        )
        iterative.run()

        assert set(iterative.solution) == nx.algorithms.approximation.max_clique(graph)

    def test_qubo_e2e(self, default_test_simulator):
        """Iterative QAOA recovers the known QUBO optimum."""
        default_test_simulator.set_seed(1997)

        iterative = IterativeQAOA(
            BinaryOptimizationProblem(QUBO_MATRIX),
            max_depth=3,
            strategy=InterpolationStrategy.INTERP,
            max_iterations_per_depth=15,
            backend=default_test_simulator,
            seed=1997,
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        )
        iterative.run()

        np.testing.assert_equal(iterative.solution, QUBO_SOLUTION)

    def test_iterative_beats_shallow_random_init(self, default_test_simulator):
        """Iterative QAOA at depth 3 outperforms random-init QAOA at depth 1.

        With the same per-depth budget, warm-starting should find a better
        loss at depth 3 than a single-depth random-init run at depth 1.
        """
        default_test_simulator.set_seed(1997)
        graph = nx.random_regular_graph(3, 10, seed=1997)
        budget = 10

        # Standard QAOA at depth 1
        standard = QAOA(
            MaxCutProblem(graph),
            n_layers=1,
            max_iterations=budget,
            backend=default_test_simulator,
            seed=1997,
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        )
        standard.run()

        # Iterative QAOA up to depth 3
        default_test_simulator.set_seed(1997)
        iterative = IterativeQAOA(
            MaxCutProblem(graph),
            max_depth=3,
            strategy=InterpolationStrategy.INTERP,
            max_iterations_per_depth=budget,
            backend=default_test_simulator,
            seed=1997,
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        )
        iterative.run()

        assert iterative.best_loss < standard.best_loss
        assert iterative.solution is not None
        assert len(iterative.solution) > 0
