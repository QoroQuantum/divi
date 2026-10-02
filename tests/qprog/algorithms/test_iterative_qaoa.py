# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import math
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
from divi.qprog.algorithms import _iterative_qaoa
from divi.qprog.algorithms._iterative_qaoa import (
    _chebyshev,
    _extract_depth_from_subdir,
    _fourier,
    _interp,
    interpolate_qaoa_params,
)
from divi.qprog.checkpointing import (
    PROGRAM_COMPLETION_FILE,
    CheckpointConfig,
    list_checkpoints,
)
from divi.qprog.problems import (
    BinaryOptimizationProblem,
    MaxCliqueProblem,
    MaxCutProblem,
)
from divi.reporting._events import EventKind, ProgressEvent, TerminalStatus
from tests._helpers import exact_match
from tests.qprog._program_contracts import verify_load_state_rejects_missing_state_key
from tests.qprog.problems._helpers import QUBO_MATRIX, QUBO_SOLUTION, make_bull_graph


def _bull_iterative(backend, **kwargs):
    return IterativeQAOA(
        MaxCutProblem(make_bull_graph()),
        backend=backend,
        optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        **kwargs,
    )


@pytest.mark.parametrize(
    "make_entry",
    [
        lambda root: (root / "depth_09").touch(),
        lambda root: (root / "depth_final").mkdir(),
        lambda root: (root / "abcdef12").mkdir(),
    ],
    ids=["file", "non-numeric-suffix", "no-prefix"],
)
def test_extract_depth_ignores_entries_that_are_not_depth_dirs(tmp_path, make_entry):
    make_entry(tmp_path)
    (entry,) = tmp_path.iterdir()

    assert _extract_depth_from_subdir(entry) is None


class TestInterp:
    def test_output_length(self):
        u = np.array([1.0, 2.0, 3.0])
        result = _interp(u)
        assert len(result) == 4

    @pytest.mark.parametrize(
        "u, expected",
        [
            pytest.param([0.5], [0.5, 0.5], id="p1_to_p2_boundary_blending"),
            pytest.param([1.0, 2.0], [1.0, 1.5, 2.0], id="p2_hand_computed"),
            pytest.param([0.0] * 5, [0.0] * 6, id="zero_params_stay_zero"),
        ],
    )
    def test_known_values(self, u, expected):
        """INTERP: result[j] = (j/p)·u[j-1] + ((p-j)/p)·u[j], with u[-1] = u[p] = 0."""
        result = _interp(np.array(u))
        assert len(result) == len(expected)
        np.testing.assert_allclose(result, expected)


def _zhou_angles(coeffs, p, trig):
    """Zhou et al. Eq. (8): sum_k c_k trig[(k - 1/2)(i - 1/2) pi / p], i = 1..p."""
    return np.array(
        [
            sum(
                c * trig((k - 0.5) * (i - 0.5) * math.pi / p)
                for k, c in enumerate(coeffs, start=1)
            )
            for i in range(1, p + 1)
        ]
    )


_CHEBYSHEV_T = [
    lambda x: 1.0,
    lambda x: x,
    lambda x: 2 * x**2 - 1,
    lambda x: 4 * x**3 - 3 * x,
    lambda x: 8 * x**4 - 8 * x**2 + 1,
]


def _shifted_chebyshev_angles(coeffs, p):
    """arXiv:2504.01694 Eq. (7): sum_j c_j T_{j-1}(2 i/p - 1), i = 1..p."""
    return np.array(
        [
            sum(c * _CHEBYSHEV_T[j](2 * i / p - 1) for j, c in enumerate(coeffs))
            for i in range(1, p + 1)
        ]
    )


def _least_squares_refit(u, q, angles):
    """Least-squares fit ``u`` in the q-term ``angles`` basis, evaluated at p+1."""
    p = len(u)
    basis = np.column_stack([angles(unit, p) for unit in np.eye(q)])
    coeffs, *_ = np.linalg.lstsq(basis, u, rcond=None)
    return angles(coeffs, p + 1)


def _interleave(betas, gammas):
    params = np.empty(2 * len(betas))
    params[0::2], params[1::2] = betas, gammas
    return params


_FOURIER_BASES = pytest.mark.parametrize("trig", [np.sin, np.cos], ids=["sin", "cos"])


class TestFourier:
    @_FOURIER_BASES
    def test_p1_rescales_the_single_mode(self, trig):
        result = _fourier(np.array([0.7]), trig)

        scale = 0.7 / trig(math.pi / 4)
        np.testing.assert_allclose(
            result, [scale * trig(math.pi / 8), scale * trig(3 * math.pi / 8)]
        )

    @_FOURIER_BASES
    @pytest.mark.parametrize(
        "coeffs", [[0.4, -0.1], [0.3, 0.05, -0.02, 0.01]], ids=["p2", "p4"]
    )
    def test_full_basis_keeps_coefficients_and_appends_zero(self, trig, coeffs):
        p = len(coeffs)

        result = _fourier(_zhou_angles(coeffs, p, trig), trig)

        np.testing.assert_allclose(result, _zhou_angles(coeffs + [0.0], p + 1, trig))

    @_FOURIER_BASES
    def test_p6_caps_at_five_terms(self, trig):
        u = np.arange(1.0, 7.0) ** 2

        result = _fourier(u, trig)

        expected = _least_squares_refit(
            u, 5, lambda coeffs, p: _zhou_angles(coeffs, p, trig)
        )
        np.testing.assert_allclose(result, expected)


class TestChebyshev:
    def test_p1_keeps_the_constant(self):
        np.testing.assert_allclose(_chebyshev(np.array([0.7])), [0.7, 0.7])

    @pytest.mark.parametrize(
        "coeffs", [[0.4, -0.1], [0.3, 0.05, -0.02, 0.01]], ids=["p2", "p4"]
    )
    def test_full_basis_evaluates_the_same_polynomial_on_the_finer_grid(self, coeffs):
        p = len(coeffs)

        result = _chebyshev(_shifted_chebyshev_angles(coeffs, p))

        np.testing.assert_allclose(result, _shifted_chebyshev_angles(coeffs, p + 1))

    def test_p6_caps_at_five_terms(self):
        u = np.sqrt(np.arange(1.0, 7.0))

        result = _chebyshev(u)

        np.testing.assert_allclose(
            result, _least_squares_refit(u, 5, _shifted_chebyshev_angles)
        )


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

    def test_fourier_uses_cosine_for_betas_and_sine_for_gammas(self):
        beta_coeffs, gamma_coeffs = [0.3, -0.05], [0.4, 0.1]

        result = interpolate_qaoa_params(
            _interleave(
                _zhou_angles(beta_coeffs, 2, np.cos),
                _zhou_angles(gamma_coeffs, 2, np.sin),
            ),
            2,
            InterpolationStrategy.FOURIER,
        )

        np.testing.assert_allclose(result[0::2], _zhou_angles(beta_coeffs, 3, np.cos))
        np.testing.assert_allclose(result[1::2], _zhou_angles(gamma_coeffs, 3, np.sin))

    def test_one_basis_term_fourier_projects_onto_the_lowest_mode(self):
        betas, gammas = np.array([1.0, 2.0, 4.0]), np.array([10.0, 20.0, 40.0])

        result = interpolate_qaoa_params(
            _interleave(betas, gammas),
            3,
            InterpolationStrategy.FOURIER,
            n_basis_terms=1,
        )

        np.testing.assert_allclose(
            result[0::2],
            _least_squares_refit(betas, 1, lambda c, p: _zhou_angles(c, p, np.cos)),
        )
        np.testing.assert_allclose(
            result[1::2],
            _least_squares_refit(gammas, 1, lambda c, p: _zhou_angles(c, p, np.sin)),
        )

    def test_one_basis_term_chebyshev_flattens_both_sequences_to_their_mean(self):
        params = _interleave(np.array([1.0, 2.0, 4.0]), np.array([10.0, 20.0, 40.0]))

        result = interpolate_qaoa_params(
            params, 3, InterpolationStrategy.CHEBYSHEV, n_basis_terms=1
        )

        np.testing.assert_allclose(result, [7 / 3, 70 / 3] * 4)

    def test_fourier_and_chebyshev_differ(self):
        params = np.array([0.1, 0.2, 0.3, 0.5, 0.2, 0.9])

        fourier = interpolate_qaoa_params(params, 3, InterpolationStrategy.FOURIER)
        chebyshev = interpolate_qaoa_params(params, 3, InterpolationStrategy.CHEBYSHEV)

        assert np.max(np.abs(fourier - chebyshev)) > 1e-2

    @pytest.mark.parametrize("strategy", list(InterpolationStrategy))
    def test_all_strategies_produce_correct_length(self, strategy):
        params = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])  # depth=3
        result = interpolate_qaoa_params(params, 3, strategy)
        assert len(result) == 8  # depth=4


class TestIterativeQAOA:
    def test_run_uses_one_direct_session_for_all_depths(
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

    @pytest.mark.parametrize(
        "max_depth, convergence_threshold",
        [(2, None), (10, 1e10)],
        ids=["last-depth", "converged"],
    )
    def test_run_after_the_last_depth_does_nothing(
        self, default_test_simulator, max_depth, convergence_threshold
    ):
        """A second run() once every depth ran, or the run converged, warns and
        spends no circuits."""
        iterative = IterativeQAOA(
            MaxCutProblem(make_bull_graph()),
            max_depth=max_depth,
            max_iterations_per_depth=2,
            convergence_threshold=convergence_threshold,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        )
        iterative.run()
        circuits, history = iterative.total_circuit_count, iterative.depth_history
        ended = (
            f"converged at depth {history[-1]['depth']}"
            if convergence_threshold is not None
            else f"already run all its depths (1 to max_depth={max_depth})"
        )

        with pytest.warns(
            UserWarning,
            match=exact_match(
                f"This IterativeQAOA has {ended}, so run() neither trains nor "
                "finalises. To sample the best depth's parameters, call "
                "sample_solution()."
            ),
        ):
            iterative.run()

        assert iterative.total_circuit_count == circuits
        assert iterative.depth_history == history
        assert iterative._completed_iterations == sum(
            entry["n_iterations"] for entry in history
        )

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
            convergence_threshold=1e10,  # any improvement converges
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        )
        assert iterative.run() is iterative

        # Stops at the first depth that improves on the one before it.
        losses = [entry["best_loss"] for entry in iterative.depth_history]
        first_improvement = next(
            depth
            for depth in range(1, len(losses))
            if losses[depth] <= losses[depth - 1]
        )
        assert len(losses) == first_improvement + 1 < 10
        # A converged run still restores and samples its best depth.
        assert iterative.n_layers == iterative.best_depth
        assert iterative.solution is not None
        np.testing.assert_array_equal(iterative.final_params, iterative.best_params)
        assert len(iterative.cost_circuit.parameters) == 2 * iterative.best_depth

    @pytest.mark.parametrize(
        "losses, converged",
        [((-1.0, -1.1), True), ((-1.0, -0.9), False), ((-1.0, -3.0), False)],
        ids=["small-gain", "worse", "large-gain"],
    )
    def test_converges_only_on_a_small_improvement(
        self, dummy_simulator, losses, converged
    ):
        iterative = IterativeQAOA(
            MaxCutProblem(make_bull_graph()),
            convergence_threshold=0.5,
            backend=dummy_simulator,
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        )
        iterative._depth_history = [{"best_loss": loss} for loss in losses]

        assert iterative._converged is converged

    @pytest.mark.parametrize(
        "argument, replacement",
        [("n_layers", "max_depth"), ("max_iterations", "max_iterations_per_depth")],
    )
    def test_rejects_the_fixed_depth_arguments(
        self, dummy_simulator, argument, replacement
    ):
        with pytest.raises(
            TypeError,
            match=exact_match(
                f"IterativeQAOA sets {argument} for each depth; "
                f"pass {replacement} instead."
            ),
        ):
            IterativeQAOA(
                MaxCutProblem(make_bull_graph()),
                backend=dummy_simulator,
                **{argument: 2},
            )

    def test_defaults(self, dummy_simulator):
        iterative = _bull_iterative(dummy_simulator)

        assert iterative.max_iterations == 10
        assert iterative.n_layers == 1
        assert iterative.best_depth == 1

    def test_default_budget_trains_five_depths_of_ten_iterations(self, dummy_simulator):
        iterative = IterativeQAOA(
            MaxCutProblem(make_bull_graph()),
            backend=dummy_simulator,
            optimizer=MonteCarloOptimizer(population_size=2, n_best_sets=1),
        )

        iterative.run(perform_final_computation=False)

        assert [
            (entry["depth"], entry["n_iterations"]) for entry in iterative.depth_history
        ] == [(depth, 10) for depth in range(1, 6)]

    @pytest.mark.parametrize(
        "budget, expected", [(3, 3), (lambda depth: 10 * depth, 10)]
    )
    def test_construction_uses_the_first_depth_budget(
        self, dummy_simulator, budget, expected
    ):
        iterative = _bull_iterative(dummy_simulator, max_iterations_per_depth=budget)

        assert iterative.max_iterations == expected

    def test_initial_params_are_ignored_with_a_warning(self, default_test_simulator):
        iterative = _bull_iterative(
            default_test_simulator, max_depth=1, max_iterations_per_depth=1
        )

        with pytest.warns(
            UserWarning,
            match=exact_match(
                "IterativeQAOA ignores `initial_params` — each depth computes its "
                "own warm-start via interpolation of the previous depth's best "
                "parameters. Use QAOA directly if you want to seed the first run."
            ),
        ):
            iterative.run(initial_params=np.zeros(2), perform_final_computation=False)

    def test_skipping_the_final_computation_leaves_no_solution(
        self, default_test_simulator
    ):
        iterative = _bull_iterative(
            default_test_simulator, max_depth=1, max_iterations_per_depth=1
        )

        iterative.run(perform_final_computation=False)

        with pytest.raises(RuntimeError, match="not available yet"):
            _ = iterative.solution

    def test_n_basis_terms_reach_the_interpolation(
        self, default_test_simulator, mocker
    ):
        interpolate = mocker.spy(_iterative_qaoa, "interpolate_qaoa_params")
        iterative = _bull_iterative(
            default_test_simulator,
            max_depth=3,
            max_iterations_per_depth=1,
            strategy=InterpolationStrategy.FOURIER,
            n_basis_terms=1,
        )

        iterative.run(perform_final_computation=False)

        assert [call.args[1:] for call in interpolate.call_args_list] == [
            (1, InterpolationStrategy.FOURIER, 1),
            (2, InterpolationStrategy.FOURIER, 1),
        ]

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

    def test_run_max_iterations_overrides_every_depth(self, default_test_simulator):
        iterative = IterativeQAOA(
            MaxCutProblem(make_bull_graph()),
            max_depth=2,
            max_iterations_per_depth=3,
            backend=default_test_simulator,
            optimizer=MonteCarloOptimizer(population_size=4, n_best_sets=2),
        )

        iterative.run(max_iterations=1, perform_final_computation=False)

        assert [entry["n_iterations"] for entry in iterative.depth_history] == [1, 1]

    def test_run_rejects_unknown_keywords(self, default_test_simulator):
        iterative = IterativeQAOA(
            MaxCutProblem(make_bull_graph()),
            max_depth=1,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        )

        with pytest.raises(
            TypeError,
            match=exact_match(
                "IterativeQAOA.run() got unexpected keyword argument(s): dry_run. "
                "To preview without executing anything, call the separate method "
                "instead: program.dry_run()."
            ),
        ):
            iterative.run(dry_run=True)

    def test_progress_row_finishes_once_after_all_depths(self, default_test_simulator):
        """Depths do not finish the row; the whole run does, once."""
        iterative = IterativeQAOA(
            MaxCutProblem(make_bull_graph()),
            max_depth=2,
            strategy=InterpolationStrategy.INTERP,
            max_iterations_per_depth=1,
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        )
        emitted = []

        with iterative._bind_progress_emitter(emitted.append):
            iterative.run(perform_final_computation=False)

        assert [
            event.terminal_status
            for event in emitted
            if event.kind is EventKind.FINISH
            and event.progress_key == iterative._progress_key
        ] == [TerminalStatus.SUCCESS]

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

    def _run_without_completion(self, backend, checkpoint_dir):
        """A finished run's checkpoints without its completion file: what a
        crash between the last depth and the completion write leaves."""
        program = self._run_unsampled(backend, checkpoint_dir)
        (checkpoint_dir / PROGRAM_COMPLETION_FILE).unlink()
        return program

    def _run_unsampled(self, backend, checkpoint_dir):
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

    def _load(self, backend, checkpoint_dir, subdirectory=None, max_depth=MAX_DEPTH):
        return IterativeQAOA.load_state(
            checkpoint_dir,
            backend=backend,
            subdirectory=subdirectory,
            problem=MaxCutProblem(make_bull_graph()),
            max_depth=max_depth,
            max_iterations_per_depth=self.ITERS_PER_DEPTH,
        )

    def test_load_skips_a_deeper_depth_without_checkpoints(
        self, default_test_simulator, tmp_path
    ):
        self._run_without_completion(default_test_simulator, tmp_path)
        (tmp_path / "depth_09").mkdir()

        loaded = self._load(default_test_simulator, tmp_path)

        assert loaded.n_layers == self.MAX_DEPTH

    def test_a_finished_run_extends_to_a_larger_max_depth(
        self, default_test_simulator, tmp_path, mocker
    ):
        """The new depth warm-starts from the deepest completed one."""
        self._run_unsampled(default_test_simulator, tmp_path)
        loaded = self._load(
            default_test_simulator, tmp_path, max_depth=self.MAX_DEPTH + 1
        )
        deepest_params = loaded.depth_history[-1]["best_params"]
        interpolate = mocker.spy(_iterative_qaoa, "interpolate_qaoa_params")

        loaded.run(perform_final_computation=False)

        history = loaded.depth_history
        assert [entry["depth"] for entry in history] == [1, 2, 3, 4]
        assert history[-1]["best_params"].size == 2 * (self.MAX_DEPTH + 1)
        (call,) = interpolate.call_args_list
        np.testing.assert_array_equal(call.args[0], deepest_params)
        assert call.args[1] == self.MAX_DEPTH

    def test_load_rejects_a_state_missing_depth_keys(
        self, default_test_simulator, tmp_path
    ):
        self._run_without_completion(default_test_simulator, tmp_path)

        verify_load_state_rejects_missing_state_key(
            tmp_path, "depth", lambda: self._load(default_test_simulator, tmp_path)
        )

    def test_each_depth_keeps_its_own_checkpoints(
        self, default_test_simulator, tmp_path
    ):
        """Depths write to separate subdirectories instead of overwriting."""
        self._run_without_completion(default_test_simulator, tmp_path)

        for depth in range(1, self.MAX_DEPTH + 1):
            depth_dir = tmp_path / f"depth_{depth:02d}"
            iterations = [info.iteration for info in list_checkpoints(depth_dir)]
            assert iterations == list(range(1, self.ITERS_PER_DEPTH + 1))

    def test_load_resolves_deepest_checkpoint(self, default_test_simulator, tmp_path):
        """load_state picks the deepest depth and rebuilds its ansatz."""
        self._run_without_completion(default_test_simulator, tmp_path)

        loaded = self._load(default_test_simulator, tmp_path)

        assert loaded.n_layers == self.MAX_DEPTH
        assert loaded.best_params.size == loaded.n_params
        # The deepest checkpoint is written mid-depth, so the completed depths
        # are the ones before it.
        assert [entry["depth"] for entry in loaded.depth_history] == [1, 2]

    def test_load_a_given_iteration_names_its_depth(
        self, default_test_simulator, tmp_path
    ):
        self._run_without_completion(default_test_simulator, tmp_path)

        loaded = self._load(
            default_test_simulator, tmp_path, subdirectory="depth_02/checkpoint_001"
        )

        assert loaded.n_layers == 2
        assert loaded.current_iteration == 1

    def test_restore_existing_instance_resolves_deepest_checkpoint(
        self, default_test_simulator, tmp_path
    ):
        self._run_without_completion(default_test_simulator, tmp_path)
        target = IterativeQAOA(
            MaxCutProblem(make_bull_graph()),
            max_depth=self.MAX_DEPTH,
            max_iterations_per_depth=self.ITERS_PER_DEPTH,
            backend=default_test_simulator,
            optimizer=MonteCarloOptimizer(population_size=4, n_best_sets=2),
        )

        target._restore_state(tmp_path)

        assert target.n_layers == self.MAX_DEPTH
        assert target.current_iteration == self.ITERS_PER_DEPTH

    def test_restored_finished_run_does_nothing(
        self, default_test_simulator, tmp_path, mocker
    ):
        self._run_unsampled(default_test_simulator, tmp_path)
        loaded = self._load(default_test_simulator, tmp_path)
        sample = mocker.spy(loaded, "sample_solution")

        with pytest.warns(UserWarning, match="call sample_solution"):
            loaded.run()

        sample.assert_not_called()

    def test_unsampled_finished_run_restores_best_depth(
        self, default_test_simulator, tmp_path
    ):
        program = self._run_unsampled(default_test_simulator, tmp_path)
        # Otherwise the deepest-depth fallback would pass on its own.
        assert program.best_depth != self.MAX_DEPTH

        loaded = self._load(default_test_simulator, tmp_path)

        assert loaded.n_layers == program.best_depth
        assert loaded.best_depth == program.best_depth
        np.testing.assert_allclose(loaded.best_params, program.best_params)

    def test_sampled_finished_run_restores_best_depth(
        self, default_test_simulator, tmp_path
    ):
        program = IterativeQAOA(
            MaxCutProblem(make_bull_graph()),
            max_depth=self.MAX_DEPTH,
            max_iterations_per_depth=self.ITERS_PER_DEPTH,
            backend=default_test_simulator,
            optimizer=MonteCarloOptimizer(population_size=4, n_best_sets=2),
            seed=1997,
        )
        program.run(
            checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path),
            perform_final_computation=True,
        )
        loaded = self._load(default_test_simulator, tmp_path)

        assert loaded.n_layers == program.best_depth
        assert loaded.best_probs == program.best_probs
        assert [entry["depth"] for entry in loaded.depth_history] == [
            entry["depth"] for entry in program.depth_history
        ]
        for loaded_entry, original_entry in zip(
            loaded.depth_history, program.depth_history
        ):
            np.testing.assert_allclose(
                loaded_entry["best_params"], original_entry["best_params"]
            )

    def test_resume_continues_after_the_completed_depths(
        self, default_test_simulator, tmp_path
    ):
        """A resumed run finishes the remaining depths without restarting at 1."""
        self._run_without_completion(default_test_simulator, tmp_path)
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
        self._run_without_completion(default_test_simulator, tmp_path)
        deepest = tmp_path / f"depth_{self.MAX_DEPTH:02d}"
        for info in list_checkpoints(deepest):
            if info.iteration > 1:
                shutil.rmtree(info.path)

        loaded = self._load(default_test_simulator, tmp_path)
        assert loaded.n_layers == self.MAX_DEPTH
        assert loaded.current_iteration == 1
        assert loaded.best_params.size == loaded.n_params
        assert (
            loaded._expected_total_iterations == self.MAX_DEPTH * self.ITERS_PER_DEPTH
        )
        assert loaded._completed_iterations == (
            (self.MAX_DEPTH - 1) * self.ITERS_PER_DEPTH + 1
        )

        loaded.run(perform_final_computation=False)

        deepest_entry = loaded.depth_history[-1]
        assert deepest_entry["depth"] == self.MAX_DEPTH
        assert deepest_entry["n_iterations"] == self.ITERS_PER_DEPTH
        assert deepest_entry["best_params"].size == 2 * self.MAX_DEPTH

    def test_load_rejects_a_different_problem(self, default_test_simulator, tmp_path):
        self._run_unsampled(default_test_simulator, tmp_path)

        with pytest.raises(ValueError, match="different cost Hamiltonian"):
            IterativeQAOA.load_state(
                tmp_path,
                backend=default_test_simulator,
                problem=MaxCutProblem(nx.cycle_graph(5)),
                max_depth=self.MAX_DEPTH,
                max_iterations_per_depth=self.ITERS_PER_DEPTH,
            )

    def test_fresh_run_rejects_a_used_directory(self, default_test_simulator, tmp_path):
        """The refused run leaves the finished run's completion file in place."""
        self._run_unsampled(default_test_simulator, tmp_path)

        with pytest.raises(ValueError, match="already holds checkpoints"):
            self._run_unsampled(default_test_simulator, tmp_path)
        assert (tmp_path / PROGRAM_COMPLETION_FILE).is_file()


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
