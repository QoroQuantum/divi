# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pennylane as qp
import pytest
from sklearn.decomposition import PCA

from divi.qprog import (
    PCE,
    QAOA,
    VQE,
    CustomVQA,
    GenericLayerAnsatz,
    IterativeQAOA,
    ScipyMethod,
    ScipyOptimizer,
)
from divi.qprog.algorithms import InterpolationStrategy
from divi.qprog.problems import MaxCutProblem
from divi.viz import scan_1d, scan_2d, scan_interp_1d, scan_interp_2d, scan_pca


@pytest.fixture
def basic_ansatz():
    return GenericLayerAnsatz([qp.RY, qp.RZ])


@pytest.fixture
def vqe_program(dummy_simulator, basic_ansatz):
    return VQE(
        hamiltonian=qp.Z(0),
        n_electrons=1,
        ansatz=basic_ansatz,
        n_layers=1,
        backend=dummy_simulator,
    )


@pytest.fixture
def qaoa_program(dummy_simulator):
    return QAOA(
        MaxCutProblem(nx.path_graph(2)),
        n_layers=1,
        optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
        backend=dummy_simulator,
    )


@pytest.fixture
def pce_program_soft(dummy_simulator, basic_ansatz):
    return PCE(
        problem=np.array([[1.0, 0.2], [0.2, 2.0]]),
        ansatz=basic_ansatz,
        n_layers=1,
        backend=dummy_simulator,
        alpha=2.0,
    )


@pytest.fixture
def custom_vqa_program(dummy_simulator):
    qscript = qp.tape.QuantumScript(
        ops=[qp.RX(0.0, wires=0), qp.RZ(0.0, wires=0)],
        measurements=[qp.expval(qp.Z(0))],
    )
    return CustomVQA(qscript=qscript, backend=dummy_simulator)


class TestDirectAPI:
    def test_scan_1d_vqe_returns_expected_shapes(self, vqe_program):
        center = np.zeros(vqe_program.get_expected_param_shape()[1])
        result = scan_1d(vqe_program, center=center, n_points=5, span=(-0.5, 0.5))

        assert result.offsets.shape == (5,)
        assert result.values.shape == (5,)
        assert result.parameter_sets.shape == (5, center.size)

    def test_scan_1d_random_direction_reproducible_with_rng(self, vqe_program):
        center = np.zeros(vqe_program.get_expected_param_shape()[1])
        a = scan_1d(vqe_program, center=center, n_points=4, rng=123, direction=None)
        b = scan_1d(vqe_program, center=center, n_points=4, rng=123, direction=None)
        np.testing.assert_allclose(a.parameter_sets, b.parameter_sets)
        np.testing.assert_allclose(a.direction, b.direction)

    def test_scan_1d_normalize_directions_false_scales_step(self, vqe_program):
        center = np.zeros(vqe_program.get_expected_param_shape()[1])
        direction = np.array([2.0, 0.0])
        off = np.linspace(-1.0, 1.0, 3, dtype=np.float64)
        unnorm = scan_1d(
            vqe_program,
            center=center,
            direction=direction,
            n_points=3,
            span=(-1.0, 1.0),
            normalize_directions=False,
        )
        norm = scan_1d(
            vqe_program,
            center=center,
            direction=direction,
            n_points=3,
            span=(-1.0, 1.0),
            normalize_directions=True,
        )
        np.testing.assert_allclose(unnorm.offsets, off)
        np.testing.assert_allclose(norm.offsets, off)
        np.testing.assert_allclose(
            unnorm.parameter_sets, center + off[:, None] * direction[None, :]
        )
        np.testing.assert_allclose(
            norm.parameter_sets, center + off[:, None] * (direction / 2.0)[None, :]
        )

    def test_scan_1d_pce_soft_mode_runs(self, pce_program_soft):
        center = np.zeros(pce_program_soft.get_expected_param_shape()[1])
        result = scan_1d(pce_program_soft, center=center, n_points=4)

        assert result.values.shape == (4,)
        assert result.program_type == "PCE"

    def test_scan_1d_pce_preserves_hard_mode_backend_guard(
        self, dummy_expval_backend, basic_ansatz
    ):
        pce = PCE(
            problem=np.array([[1.0, 0.2], [0.2, 2.0]]),
            ansatz=basic_ansatz,
            n_layers=1,
            backend=dummy_expval_backend,
            alpha=6.0,
        )
        center = np.zeros(pce.get_expected_param_shape()[1])

        with pytest.raises(
            ValueError,
            match="hard CVaR mode.*cannot use expectation-value backends",
        ):
            scan_1d(pce, center=center, n_points=3)

    def test_scan_pca_returns_expected_shapes(self, vqe_program):
        center = np.zeros(vqe_program.get_expected_param_shape()[1])
        samples = np.array(
            [
                [-0.4, -0.2],
                [-0.2, 0.3],
                [0.2, -0.1],
                [0.4, 0.5],
            ]
        )
        result = scan_pca(
            vqe_program, center=center, samples=samples, grid_shape=(4, 3)
        )

        assert result.x_offsets.shape == (4,)
        assert result.y_offsets.shape == (3,)
        assert result.values.shape == (3, 4)
        assert result.parameter_sets.shape == (12, center.size)
        assert result.explained_variance_ratio.shape == (2,)
        assert result.projected_samples.shape == (4, 2)
        assert result.scan_component_ids == (0, 1)

    def test_scan_pca_center_none_matches_sample_mean(self, vqe_program):
        """orqviz-style default: anchor at sample mean when center is omitted."""
        samples = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=np.float64,
        )
        result = scan_pca(
            vqe_program,
            samples=samples,
            center=None,
            grid_shape=(3, 3),
            offset=(0.0, 0.0),
            span_x=(-0.5, 0.5),
            span_y=(-0.5, 0.5),
        )
        np.testing.assert_allclose(
            result.center, np.mean(samples, axis=0), rtol=0, atol=1e-10
        )

    def test_scan_pca_projected_samples_relative_to_center(self, vqe_program):
        """When center != mean, projected_samples must be relative to center."""
        samples = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=np.float64,
        )
        mean = samples.mean(axis=0)
        center = np.array([2.0, 3.0], dtype=np.float64)

        result = scan_pca(
            vqe_program,
            samples=samples,
            center=center,
            grid_shape=(3, 3),
            span_x=(-2.0, 2.0),
            span_y=(-2.0, 2.0),
        )

        # The projected samples should be (sample - center) projected onto PCs,
        # NOT (sample - mean) projected onto PCs.
        pca = PCA(n_components=2)
        pca.fit(samples)
        expected = pca.transform(samples)
        shift = center - mean
        expected[:, 0] -= np.dot(shift, pca.components_[0])
        expected[:, 1] -= np.dot(shift, pca.components_[1])
        np.testing.assert_allclose(
            result.projected_samples, expected, rtol=0, atol=1e-10
        )

        # Roundtrip is exact because n_params == n_components (no PCA truncation).
        # With higher-dimensional parameters this would only be approximate.
        for k in range(len(samples)):
            sx, sy = result.projected_samples[k]
            reconstructed = (
                sx * result.principal_component_x
                + sy * result.principal_component_y
                + center
            )
            np.testing.assert_allclose(reconstructed, samples[k], atol=1e-10)

    def test_scan_pca_rejects_rank_deficient_samples(self, vqe_program):
        center = np.zeros(vqe_program.get_expected_param_shape()[1])
        samples = np.array(
            [
                [-1.0, -1.0],
                [0.0, 0.0],
                [1.0, 1.0],
            ]
        )

        with pytest.raises(
            ValueError,
            match="samples must span at least two independent directions",
        ):
            scan_pca(vqe_program, center=center, samples=samples)

    def test_scan_2d_plot_returns_figure_and_axes(self, custom_vqa_program):
        center = np.zeros(custom_vqa_program.get_expected_param_shape()[1])
        result = scan_2d(custom_vqa_program, center=center, grid_shape=(3, 3), rng=0)
        fig, ax = result.plot(show=False)

        try:
            assert fig is ax.figure
            assert len(ax.collections) > 0
        finally:
            plt.close(fig)

    def test_scan_1d_plot_returns_figure_and_axes(self, vqe_program):
        center = np.zeros(vqe_program.get_expected_param_shape()[1])
        result = scan_1d(vqe_program, center=center, n_points=3, rng=0)
        fig, ax = result.plot(show=False)

        try:
            assert fig is ax.figure
            assert len(ax.lines) == 1
        finally:
            plt.close(fig)

    def test_iterative_qaoa_scan_is_rejected(self, dummy_simulator):
        program = IterativeQAOA(
            problem=MaxCutProblem(nx.path_graph(2)),
            max_depth=2,
            strategy=InterpolationStrategy.INTERP,
            backend=dummy_simulator,
        )
        center = np.zeros(2)

        with pytest.raises(
            NotImplementedError, match="IterativeQAOA varies its parameter space"
        ):
            scan_1d(program, center=center, n_points=3)


class TestProgramVizWrapper:
    def test_scan_2d_qaoa_returns_grid(self, qaoa_program):
        center = np.zeros(qaoa_program.get_expected_param_shape()[1])
        result = qaoa_program.viz.scan_2d(center=center, grid_shape=(3, 4), rng=0)

        assert result.x_offsets.shape == (3,)
        assert result.y_offsets.shape == (4,)
        assert result.values.shape == (4, 3)

    def test_scan_1d_custom_vqa_runs(self, custom_vqa_program):
        center = np.zeros(custom_vqa_program.get_expected_param_shape()[1])
        result = custom_vqa_program.viz.scan_1d(center=center, n_points=3, rng=0)

        assert result.values.shape == (3,)
        assert result.program_type == "CustomVQA"

    def test_scan_pca_qaoa_returns_projected_samples(self, qaoa_program):
        center = np.zeros(qaoa_program.get_expected_param_shape()[1])
        samples = np.array(
            [
                [-0.3, 0.1],
                [0.0, -0.2],
                [0.2, 0.4],
                [0.5, -0.1],
            ]
        )
        result = qaoa_program.viz.scan_pca(
            center=center,
            samples=samples,
            grid_shape=(3, 3),
        )

        assert result.values.shape == (3, 3)
        assert result.projected_samples.shape == (4, 2)
        assert result.program_type == "QAOA"


class TestScanInterp1D:
    def test_endpoints_match_theta_1_and_theta_2(self, vqe_program):
        t1 = np.array([0.0, 1.0])
        t2 = np.array([2.0, 3.0])
        result = scan_interp_1d(vqe_program, t1, t2, n_points=5)

        np.testing.assert_allclose(result.parameter_sets[0], t1, atol=1e-12)
        np.testing.assert_allclose(result.parameter_sets[-1], t2, atol=1e-12)

    def test_offsets_span_zero_to_one(self, vqe_program):
        t1 = np.array([0.0, 0.0])
        t2 = np.array([1.0, 1.0])
        result = scan_interp_1d(vqe_program, t1, t2, n_points=3)

        np.testing.assert_allclose(result.offsets, [0.0, 0.5, 1.0])

    def test_center_is_theta_1_and_direction_is_delta(self, vqe_program):
        t1 = np.array([0.5, 0.5])
        t2 = np.array([1.5, 2.5])
        result = scan_interp_1d(vqe_program, t1, t2, n_points=3)

        np.testing.assert_allclose(result.center, t1)
        np.testing.assert_allclose(result.direction, t2 - t1)

    def test_shapes(self, vqe_program):
        t1 = np.zeros(2)
        t2 = np.ones(2)
        result = scan_interp_1d(vqe_program, t1, t2, n_points=7)

        assert result.offsets.shape == (7,)
        assert result.values.shape == (7,)
        assert result.parameter_sets.shape == (7, 2)

    def test_rejects_mismatched_shapes(self, vqe_program):
        with pytest.raises(ValueError, match="theta_1 must have shape"):
            scan_interp_1d(vqe_program, np.zeros(3), np.ones(2), n_points=3)

    def test_fluent_api(self, vqe_program):
        t1 = np.zeros(2)
        t2 = np.ones(2)
        result = vqe_program.viz.scan_interp_1d(t1, t2, n_points=3)

        assert result.values.shape == (3,)


class TestPCAScanTrajectoryOverlay:
    def test_trajectory_overlay_adds_line(self, vqe_program):
        samples = np.array([[-0.4, -0.2], [-0.2, 0.3], [0.2, -0.1], [0.4, 0.5]])
        center = np.zeros(2)
        result = scan_pca(
            vqe_program, samples=samples, center=center, grid_shape=(3, 3)
        )
        fig, ax = result.plot(show=False, show_trajectory=True)

        try:
            lines = ax.get_lines()
            # At least 3 lines: trajectory line + start marker + end marker
            assert len(lines) >= 3
        finally:
            plt.close(fig)

    def test_trajectory_kwargs_override(self, vqe_program):
        samples = np.array([[-0.4, -0.2], [-0.2, 0.3], [0.2, -0.1], [0.4, 0.5]])
        center = np.zeros(2)
        result = scan_pca(
            vqe_program, samples=samples, center=center, grid_shape=(3, 3)
        )
        fig, ax = result.plot(
            show=False,
            show_trajectory=True,
            trajectory_kwargs={"color": "red", "linewidth": 3.0},
        )

        try:
            line = ax.get_lines()[0]
            assert line.get_color() == "red"
            assert line.get_linewidth() == 3.0
        finally:
            plt.close(fig)

    def test_no_trajectory_by_default(self, vqe_program):
        samples = np.array([[-0.4, -0.2], [-0.2, 0.3], [0.2, -0.1], [0.4, 0.5]])
        center = np.zeros(2)
        result = scan_pca(
            vqe_program, samples=samples, center=center, grid_shape=(3, 3)
        )
        fig, ax = result.plot(show=False)

        try:
            assert len(ax.get_lines()) == 0
        finally:
            plt.close(fig)


class TestScanInterp2D:
    def test_endpoints_on_interpolation_axis(self, vqe_program):
        t1 = np.array([0.0, 1.0])
        t2 = np.array([2.0, 3.0])
        result = scan_interp_2d(
            vqe_program, t1, t2, grid_shape=(5, 3), span_x=(0.0, 1.0)
        )

        # At y=0 (middle row when span_y is symmetric), x=0 -> theta_1, x=1 -> theta_2
        mid_y = 1  # middle of 3 y-points with default symmetric span
        row = result.parameter_sets.reshape(3, 5, 2)[mid_y]
        np.testing.assert_allclose(row[0], t1, atol=1e-12)
        np.testing.assert_allclose(row[-1], t2, atol=1e-12)

    def test_direction_x_is_interpolation_vector(self, vqe_program):
        t1 = np.array([0.5, 0.5])
        t2 = np.array([1.5, 2.5])
        result = scan_interp_2d(vqe_program, t1, t2, grid_shape=(3, 3), rng=0)

        np.testing.assert_allclose(result.center, t1)
        np.testing.assert_allclose(result.direction_x, t2 - t1)

    def test_y_direction_orthogonal_to_interpolation(self, vqe_program):
        t1 = np.zeros(2)
        t2 = np.array([1.0, 0.0])
        result = scan_interp_2d(vqe_program, t1, t2, grid_shape=(3, 3), rng=42)

        dot = float(np.dot(result.direction_x, result.direction_y))
        np.testing.assert_allclose(dot, 0.0, atol=1e-12)

    def test_random_y_has_same_norm_as_interp_dir(self, vqe_program):
        t1 = np.zeros(2)
        t2 = np.array([3.0, 4.0])
        result = scan_interp_2d(vqe_program, t1, t2, grid_shape=(3, 3), rng=0)

        np.testing.assert_allclose(
            np.linalg.norm(result.direction_y),
            np.linalg.norm(result.direction_x),
            rtol=1e-10,
        )

    def test_shapes(self, vqe_program):
        t1 = np.zeros(2)
        t2 = np.ones(2)
        result = scan_interp_2d(vqe_program, t1, t2, grid_shape=(5, 4), rng=0)

        assert result.x_offsets.shape == (5,)
        assert result.y_offsets.shape == (4,)
        assert result.values.shape == (4, 5)
        assert result.parameter_sets.shape == (20, 2)

    def test_rejects_identical_points(self, vqe_program):
        t = np.zeros(2)
        with pytest.raises(ValueError, match="must be distinct"):
            scan_interp_2d(vqe_program, t, t, grid_shape=(3, 3))

    def test_fluent_api(self, vqe_program):
        t1 = np.zeros(2)
        t2 = np.ones(2)
        result = vqe_program.viz.scan_interp_2d(t1, t2, grid_shape=(3, 3), rng=0)

        assert result.values.shape == (3, 3)


class TestPlot3D:
    def test_scan_2d_plot_3d_returns_3d_axes(self, custom_vqa_program):
        center = np.zeros(custom_vqa_program.get_expected_param_shape()[1])
        result = scan_2d(custom_vqa_program, center=center, grid_shape=(3, 3), rng=0)
        fig, ax = result.plot_3d(show=False)

        try:
            assert fig is ax.figure
            assert hasattr(ax, "plot_surface")
        finally:
            plt.close(fig)

    def test_pca_scan_plot_3d_returns_3d_axes(self, vqe_program):
        samples = np.array([[-0.4, -0.2], [-0.2, 0.3], [0.2, -0.1], [0.4, 0.5]])
        center = np.zeros(2)
        result = scan_pca(
            vqe_program, samples=samples, center=center, grid_shape=(3, 3)
        )
        fig, ax = result.plot_3d(show=False)

        try:
            assert fig is ax.figure
            assert hasattr(ax, "plot_surface")
        finally:
            plt.close(fig)

    def test_plot_3d_kwargs_forwarded(self, custom_vqa_program):
        center = np.zeros(custom_vqa_program.get_expected_param_shape()[1])
        result = scan_2d(custom_vqa_program, center=center, grid_shape=(3, 3), rng=0)
        fig, ax = result.plot_3d(show=False, cmap="plasma", alpha=0.5)

        try:
            assert fig is ax.figure
        finally:
            plt.close(fig)


class TestGradientOverlay:
    def test_scan_2d_gradient_overlay_adds_quiver(self, custom_vqa_program):
        center = np.zeros(custom_vqa_program.get_expected_param_shape()[1])
        result = scan_2d(custom_vqa_program, center=center, grid_shape=(4, 4), rng=0)
        fig, ax = result.plot(show=False, show_gradients=True)

        try:
            quivers = [c for c in ax.get_children() if type(c).__name__ == "Quiver"]
            assert len(quivers) == 1
        finally:
            plt.close(fig)

    def test_pca_scan_gradient_overlay_adds_quiver(self, vqe_program):
        samples = np.array([[-0.4, -0.2], [-0.2, 0.3], [0.2, -0.1], [0.4, 0.5]])
        center = np.zeros(2)
        result = scan_pca(
            vqe_program, samples=samples, center=center, grid_shape=(4, 4)
        )
        fig, ax = result.plot(show=False, show_gradients=True)

        try:
            quivers = [c for c in ax.get_children() if type(c).__name__ == "Quiver"]
            assert len(quivers) == 1
        finally:
            plt.close(fig)

    def test_no_gradient_by_default(self, custom_vqa_program):
        center = np.zeros(custom_vqa_program.get_expected_param_shape()[1])
        result = scan_2d(custom_vqa_program, center=center, grid_shape=(3, 3), rng=0)
        fig, ax = result.plot(show=False)

        try:
            quivers = [c for c in ax.get_children() if type(c).__name__ == "Quiver"]
            assert len(quivers) == 0
        finally:
            plt.close(fig)
