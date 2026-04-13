# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest

from divi.qprog import QAOA
from divi.qprog.problems import MaxCutProblem
from divi.viz import fourier_analysis_2d, scan_2d


@pytest.fixture
def qaoa_program(dummy_simulator):
    return QAOA(
        MaxCutProblem(nx.path_graph(2)),
        n_layers=1,
        backend=dummy_simulator,
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
