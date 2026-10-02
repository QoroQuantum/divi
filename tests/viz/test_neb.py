# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import numpy as np
import pytest

from divi.viz import GradientMethod, run_neb
from divi.viz._gradients import (
    _compute_gradients,
    _finite_difference_gradients,
    _parameter_shift_gradients,
)
from divi.viz._neb import (
    _cumulative_distances,
    _neb_perpendicular_gradients,
    _redistribute_uniform,
)

_SPHERE_PIVOTS = np.array([[1.0, 2.0], [-0.5, 0.3]])
_TRIG_PIVOTS = np.array([[0.3, 0.7], [1.0, -0.5]])


def _sphere(param_sets):
    return np.sum(param_sets**2, axis=1)


def _trig(param_sets):
    return np.sin(param_sets[:, 0]) + np.cos(param_sets[:, 1])


def _trig_gradient(pivots):
    return np.column_stack([np.cos(pivots[:, 0]), -np.sin(pivots[:, 1])])


class TestRunNEB:
    def test_uses_one_direct_session_and_closes_it_on_failure(
        self, vqe_program, mocker
    ):
        session = mocker.MagicMock()
        session.__enter__.return_value = session
        direct = mocker.patch(
            "divi.qprog.quantum_program.ProgressSession.direct",
            return_value=session,
        )

        with pytest.raises(ValueError, match="n_pivots must be >= 3"):
            run_neb(vqe_program, np.zeros(2), np.ones(2), n_pivots=2)

        direct.assert_called_once()
        session.__enter__.assert_called_once_with()
        assert session.__exit__.call_count == 1
        assert session.__exit__.call_args.args[0] is ValueError

    def test_shapes(self, vqe_program):
        t1 = np.array([0.0, 0.0])
        t2 = np.array([1.0, 1.0])
        result = run_neb(vqe_program, t1, t2, n_pivots=5, n_steps=3, learning_rate=0.01)

        assert result.path.shape == (5, 2)
        assert result.energies.shape == (5,)
        assert result.path_distances.shape == (5,)
        assert result.program_type == "VQE"

    def test_endpoints_fixed(self, vqe_program):
        t1 = np.array([0.0, 0.5])
        t2 = np.array([2.0, 1.5])
        result = run_neb(vqe_program, t1, t2, n_pivots=5, n_steps=5, learning_rate=0.01)

        np.testing.assert_allclose(result.path[0], t1, atol=1e-12)
        np.testing.assert_allclose(result.path[-1], t2, atol=1e-12)

    def test_path_distances_normalised(self, vqe_program):
        t1 = np.zeros(2)
        t2 = np.ones(2)
        result = run_neb(vqe_program, t1, t2, n_pivots=4, n_steps=2, learning_rate=0.01)

        np.testing.assert_allclose(result.path_distances[0], 0.0)
        np.testing.assert_allclose(result.path_distances[-1], 1.0)
        assert np.all(np.diff(result.path_distances) >= -1e-12)

    def test_all_paths_records_history(self, vqe_program):
        t1 = np.zeros(2)
        t2 = np.ones(2)
        n_steps = 4
        result = run_neb(
            vqe_program, t1, t2, n_pivots=4, n_steps=n_steps, learning_rate=0.01
        )

        # Initial chain + one per step.
        assert len(result.all_paths) == n_steps + 1

    def test_rejects_too_few_pivots(self, vqe_program):
        with pytest.raises(ValueError, match="n_pivots must be >= 3"):
            run_neb(vqe_program, np.zeros(2), np.ones(2), n_pivots=2)

    def test_rejects_wrong_shape(self, vqe_program):
        with pytest.raises(ValueError, match="theta_1 must have shape"):
            run_neb(vqe_program, np.zeros(5), np.ones(2), n_pivots=4, n_steps=1)

    def test_plot_returns_figure_and_axes(self, vqe_program):
        t1 = np.zeros(2)
        t2 = np.ones(2)
        result = run_neb(vqe_program, t1, t2, n_pivots=4, n_steps=2, learning_rate=0.01)
        fig, ax = result.plot(show=False)

        try:
            assert fig is ax.figure
            assert len(ax.lines) >= 1
        finally:
            plt.close(fig)

    def test_barrier_decreases_on_known_landscape(self, vqe_program, mock_landscape):
        """On a 1D double-well, NEB should find a path with lower barrier than the straight line."""
        # f(x0, x1) = (x0^2 - 1)^2  (two minima at x0=±1, barrier at x0=0)
        mock_landscape(vqe_program, lambda p: (p[0] ** 2 - 1) ** 2)
        t1 = np.array([-1.0, 0.0])  # minimum
        t2 = np.array([1.0, 0.0])  # minimum

        # Evaluate initial straight-line barrier.
        init_chain = np.linspace(t1, t2, 8)
        init_losses = np.array([(p[0] ** 2 - 1) ** 2 for p in init_chain])
        init_barrier = float(np.max(init_losses))

        result = run_neb(
            vqe_program, t1, t2, n_pivots=8, n_steps=30, learning_rate=0.05
        )
        final_barrier = float(np.max(result.energies))

        # The straight-line barrier for (x0^2-1)^2 at x0=0 is 1.0.
        # NEB should find a path with barrier strictly below the initial one.
        assert final_barrier <= init_barrier + 1e-6
        assert final_barrier < 1.0

    def test_finite_difference_gradients_correct_for_known_function(self):
        """Verify finite-difference gradients of f(x) = x0^2 + x1^2."""
        grads = _finite_difference_gradients(_sphere, _SPHERE_PIVOTS, eps=1e-5)

        np.testing.assert_allclose(grads, 2.0 * _SPHERE_PIVOTS, atol=1e-8)

    def test_parameter_shift_gradients_correct_for_trig(self):
        """Verify parameter-shift gradients of f(x) = sin(x0) + cos(x1)."""
        grads = _parameter_shift_gradients(_trig, _TRIG_PIVOTS)

        np.testing.assert_allclose(grads, _trig_gradient(_TRIG_PIVOTS), atol=1e-10)

    def test_neb_with_finite_difference_method(self, vqe_program):
        t1 = np.zeros(2)
        t2 = np.ones(2)
        result = run_neb(
            vqe_program,
            t1,
            t2,
            n_pivots=4,
            n_steps=2,
            learning_rate=0.01,
            gradient_method=GradientMethod.FINITE_DIFFERENCE,
            eps=1e-3,
        )

        assert result.path.shape == (4, 2)

    def test_identical_endpoints(self, vqe_program):
        t = np.array([0.5, 0.5])
        result = run_neb(vqe_program, t, t, n_pivots=4, n_steps=2, learning_rate=0.01)

        np.testing.assert_allclose(result.path[0], t, atol=1e-12)
        np.testing.assert_allclose(result.path[-1], t, atol=1e-12)
        np.testing.assert_allclose(result.path_distances[0], 0.0)
        np.testing.assert_allclose(result.path_distances[-1], 1.0)

    def test_early_stopping(self, vqe_program):
        t1 = np.zeros(2)
        t2 = np.ones(2)
        result = run_neb(
            vqe_program,
            t1,
            t2,
            n_pivots=4,
            n_steps=100,
            learning_rate=0.001,
            convergence_tol=1e10,  # Very loose — should converge immediately.
        )

        # Should have stopped well before 100 steps (initial + at most a few).
        assert len(result.all_paths) < 100

    def test_fluent_api(self, vqe_program):
        t1 = np.zeros(2)
        t2 = np.ones(2)
        result = vqe_program.viz.run_neb(
            t1, t2, n_pivots=4, n_steps=2, learning_rate=0.01
        )

        assert result.path.shape == (4, 2)


@pytest.mark.parametrize("scale", [1.0, 0.01])
def test_cumulative_distances_normalised_by_total_length(scale):
    chain = scale * np.array([[1.0], [3.0], [4.0], [8.0], [11.0]])

    np.testing.assert_allclose(
        _cumulative_distances(chain), [0.0, 0.2, 0.3, 0.7, 1.0], rtol=0, atol=1e-12
    )


@pytest.mark.filterwarnings("error")
def test_cumulative_distances_of_collapsed_chain():
    np.testing.assert_array_equal(_cumulative_distances(np.zeros((3, 2))), [0, 0, 1])


def test_redistribute_uniform_spaces_pivots_by_arc_length():
    chain = np.array([[0.0], [0.1], [1.0]])

    np.testing.assert_allclose(
        _redistribute_uniform(chain, 3), [[0.0], [0.5], [1.0]], atol=1e-12
    )


@pytest.mark.parametrize(
    "chain, grads, expected",
    [
        pytest.param(
            [[0.0, 1.0], [0.5, 1.0], [1.0, 1.0], [1.5, 1.0]],
            [[1.0, 1.0], [2.0, -1.0]],
            [[0.0, 1.0], [0.0, -1.0]],
            id="tangent_removed",
        ),
        pytest.param(
            [[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]],
            [[1.0, 2.0]],
            [[1.0, 2.0]],
            id="repeated_pivot",
        ),
    ],
)
def test_perpendicular_gradients(chain, grads, expected):
    perp = _neb_perpendicular_gradients(np.array(chain), np.array(grads))

    np.testing.assert_allclose(perp, expected, atol=1e-12)


@pytest.mark.parametrize(
    "method, fn, pivots, expected, atol",
    [
        pytest.param(
            GradientMethod.FINITE_DIFFERENCE,
            _sphere,
            _SPHERE_PIVOTS,
            2.0 * _SPHERE_PIVOTS,
            1e-8,
            id="finite_difference",
        ),
        pytest.param(
            GradientMethod.PARAMETER_SHIFT,
            _trig,
            _TRIG_PIVOTS,
            _trig_gradient(_TRIG_PIVOTS),
            1e-10,
            id="parameter_shift",
        ),
    ],
)
def test_compute_gradients_dispatches_on_method(method, fn, pivots, expected, atol):
    grads = _compute_gradients(fn, pivots, method, eps=1e-5)

    np.testing.assert_allclose(grads, expected, atol=atol)
