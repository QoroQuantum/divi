# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest

from divi.qprog import QAOA
from divi.qprog.problems import MaxCutProblem
from divi.viz import Scan2DResult, fourier_analysis_2d, scan_2d


@pytest.fixture
def qaoa_program(dummy_simulator, default_optimizer):
    return QAOA(
        MaxCutProblem(nx.path_graph(2)),
        n_layers=1,
        backend=dummy_simulator,
        optimizer=default_optimizer,
    )


class TestFourierAnalysis2D:
    def test_shapes_match_input_grid(self, qaoa_program):
        center = np.zeros(qaoa_program.get_expected_param_shape()[1])
        scan = scan_2d(qaoa_program, center=center, grid_shape=(8, 6), rng=0)
        result = fourier_analysis_2d(scan)

        assert result.frequencies_x.shape == (8,)
        assert result.frequencies_y.shape == (6,)
        assert result.power_spectrum.shape == (6, 8)
        assert result.program_type == "QAOA"

    def test_power_spectrum_non_negative(self, qaoa_program):
        center = np.zeros(qaoa_program.get_expected_param_shape()[1])
        scan = scan_2d(qaoa_program, center=center, grid_shape=(5, 5), rng=0)
        result = fourier_analysis_2d(scan)

        assert np.all(result.power_spectrum >= 0.0)

    def test_dc_component_is_dominant_for_constant(self, qaoa_program):
        """A constant-value grid should have all power in the DC component."""
        center = np.zeros(qaoa_program.get_expected_param_shape()[1])
        scan = scan_2d(qaoa_program, center=center, grid_shape=(8, 8), rng=0)
        # Override values with a constant to test DC dominance.
        scan.values[:] = 5.0
        result = fourier_analysis_2d(scan)

        # DC component is at the center of the shifted spectrum.
        cy, cx = (
            result.power_spectrum.shape[0] // 2,
            result.power_spectrum.shape[1] // 2,
        )
        dc_power = result.power_spectrum[cy, cx]
        total_power = float(np.sum(result.power_spectrum))
        # A constant signal has all power in DC — ratio should be exactly 1.0.
        np.testing.assert_allclose(dc_power, total_power, rtol=1e-10)

    def test_plot_returns_figure_and_axes(self, qaoa_program):
        center = np.zeros(qaoa_program.get_expected_param_shape()[1])
        scan = scan_2d(qaoa_program, center=center, grid_shape=(5, 5), rng=0)
        result = fourier_analysis_2d(scan)
        fig, ax = result.plot(show=False)

        try:
            assert fig is ax.figure
            assert len(ax.images) > 0
        finally:
            plt.close(fig)


def _grid_scan(x_offsets, y_offsets) -> Scan2DResult:
    values = np.random.default_rng(0).normal(size=(len(y_offsets), len(x_offsets)))
    return Scan2DResult(
        x_offsets=np.asarray(x_offsets, dtype=np.float64),
        y_offsets=np.asarray(y_offsets, dtype=np.float64),
        values=values,
        parameter_sets=np.zeros((values.size, 2)),
        center=np.zeros(2),
        direction_x=np.array([1.0, 0.0]),
        direction_y=np.array([0.0, 1.0]),
        program_type="Synthetic",
    )


def test_frequencies_follow_grid_spacing():
    scan = _grid_scan(np.linspace(0.0, 1.5, 4), [0.0, 0.25])

    result = fourier_analysis_2d(scan)

    np.testing.assert_allclose(
        result.frequencies_x, np.fft.fftshift(np.fft.fftfreq(4, d=0.5))
    )
    np.testing.assert_allclose(
        result.frequencies_y, np.fft.fftshift(np.fft.fftfreq(2, d=0.25))
    )


def test_single_point_axis_has_zero_frequency():
    result = fourier_analysis_2d(_grid_scan([0.0, 0.5, 1.0], [0.0]))

    np.testing.assert_array_equal(result.frequencies_y, [0.0])
    assert result.power_spectrum.shape == (1, 3)


def test_power_spectrum_satisfies_parseval():
    scan = _grid_scan(np.linspace(0.0, 1.0, 5), np.linspace(0.0, 2.0, 3))

    result = fourier_analysis_2d(scan)

    np.testing.assert_allclose(
        np.sum(result.power_spectrum), 5 * 3 * np.sum(scan.values**2), rtol=1e-12
    )
