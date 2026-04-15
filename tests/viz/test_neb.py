# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import pytest

from divi.qprog import VQE, GenericLayerAnsatz
from divi.viz import GradientMethod, run_neb
from divi.viz._gradients import _finite_difference_gradients, _parameter_shift_gradients


@pytest.fixture
def basic_ansatz():
    return GenericLayerAnsatz([qml.RY, qml.RZ])


@pytest.fixture
def vqe_program(dummy_simulator, basic_ansatz):
    return VQE(
        hamiltonian=qml.Z(0),
        n_electrons=1,
        ansatz=basic_ansatz,
        n_layers=1,
        backend=dummy_simulator,
    )


class TestRunNEB:
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

    def test_barrier_decreases_on_known_landscape(self, vqe_program, mocker):
        """On a 1D double-well, NEB should find a path with lower barrier than the straight line."""

        def _double_well(param_sets, **kwargs):
            # f(x0, x1) = (x0^2 - 1)^2  (two minima at x0=±1, barrier at x0=0)
            ps = np.atleast_2d(param_sets)
            return {i: float((p[0] ** 2 - 1) ** 2) for i, p in enumerate(ps)}

        mocker.patch.object(
            vqe_program, "_evaluate_cost_param_sets", side_effect=_double_well
        )
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

        def _sphere(param_sets):
            return np.array([float(np.sum(p**2)) for p in param_sets])

        pivots = np.array([[1.0, 2.0], [-0.5, 0.3]], dtype=np.float64)
        grads = _finite_difference_gradients(_sphere, pivots, eps=1e-5)

        # Analytical gradient of x0^2 + x1^2 is [2*x0, 2*x1].
        expected = 2.0 * pivots
        np.testing.assert_allclose(grads, expected, atol=1e-8)

    def test_parameter_shift_gradients_correct_for_trig(self):
        """Verify parameter-shift gradients of f(x) = sin(x0) + cos(x1)."""

        def _trig(param_sets):
            return np.array([float(np.sin(p[0]) + np.cos(p[1])) for p in param_sets])

        pivots = np.array([[0.3, 0.7], [1.0, -0.5]], dtype=np.float64)
        grads = _parameter_shift_gradients(_trig, pivots)

        # Analytical: df/dx0 = cos(x0), df/dx1 = -sin(x1).
        expected = np.column_stack([np.cos(pivots[:, 0]), -np.sin(pivots[:, 1])])
        np.testing.assert_allclose(grads, expected, atol=1e-10)

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
