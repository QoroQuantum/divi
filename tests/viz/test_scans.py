# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pennylane as qml
import pytest

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
from divi.viz import scan_1d, scan_2d, scan_pca


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
    qscript = qml.tape.QuantumScript(
        ops=[qml.RX(0.0, wires=0), qml.RZ(0.0, wires=0)],
        measurements=[qml.expval(qml.Z(0))],
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
            NotImplementedError, match="IterativeQAOA varies circuit depth"
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
