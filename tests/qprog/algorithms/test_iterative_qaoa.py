# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

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
from divi.qprog.problems import (
    BinaryOptimizationProblem,
    MaxCliqueProblem,
    MaxCutProblem,
)
from tests.qprog.algorithms.problems import QUBO_MATRIX, QUBO_SOLUTION, make_bull_graph

pytestmark = pytest.mark.algo


# ---------------------------------------------------------------------------
# Interpolation function tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# IterativeQAOA tests
# ---------------------------------------------------------------------------


class TestIterativeQAOA:
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

    def test_depth_info_reported(self, default_test_simulator, mocker):
        """Reporter receives depth info messages during run."""
        graph = make_bull_graph()
        iterative = IterativeQAOA(
            MaxCutProblem(graph),
            max_depth=3,
            strategy=InterpolationStrategy.INTERP,
            max_iterations_per_depth=2,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        )
        spy = mocker.patch.object(
            iterative.reporter, "info", wraps=iterative.reporter.info
        )
        iterative.run()

        depth_messages = [call for call in spy.call_args_list if "Depth" in str(call)]
        assert len(depth_messages) == 3
        assert depth_messages[0] == mocker.call(message="Depth 1/3")
        assert depth_messages[1] == mocker.call(message="Depth 2/3")
        assert depth_messages[2] == mocker.call(message="Depth 3/3")

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


# ---------------------------------------------------------------------------
# End-to-end tests
# ---------------------------------------------------------------------------


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
