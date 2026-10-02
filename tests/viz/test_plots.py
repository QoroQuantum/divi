# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Plot helpers on result containers built from synthetic arrays."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.colors import BoundaryNorm, LogNorm, Normalize
from matplotlib.contour import ContourSet
from matplotlib.quiver import Quiver

from divi.viz import (
    Fourier2DResult,
    NEBResult,
    PCAScanResult,
    Scan1DResult,
    Scan2DResult,
)
from divi.viz._results import _cell_edges_from_centers

# Grid spacing differs per axis (dx = 0.5, dy = 2) so a swapped spacing shows up.
_X = np.array([0.0, 0.5, 1.0, 1.5])
_Y = np.array([0.0, 2.0, 4.0])
_XX, _YY = np.meshgrid(_X, _Y)
_PLANE = 2.0 * _XX + 3.0 * _YY
_SAMPLES = np.array([[0.1, 0.2], [0.5, 1.0], [1.2, 3.5]])


def _scan_1d() -> Scan1DResult:
    return Scan1DResult(
        offsets=np.linspace(-1.0, 1.0, 5),
        values=np.array([3.0, 1.0, 0.0, 1.5, 4.0]),
        parameter_sets=np.zeros((5, 2)),
        center=np.zeros(2),
        direction=np.array([1.0, 0.0]),
        program_type="Synthetic",
    )


def _scan_2d(values=_PLANE) -> Scan2DResult:
    return Scan2DResult(
        x_offsets=_X,
        y_offsets=_Y,
        values=values,
        parameter_sets=np.zeros((values.size, 2)),
        center=np.zeros(2),
        direction_x=np.array([1.0, 0.0]),
        direction_y=np.array([0.0, 1.0]),
        program_type="Synthetic",
    )


def _pca_scan(values=_PLANE, samples=_SAMPLES) -> PCAScanResult:
    return PCAScanResult(
        x_offsets=_X,
        y_offsets=_Y,
        values=values,
        parameter_sets=np.zeros((values.size, 2)),
        center=np.zeros(2),
        principal_component_x=np.array([1.0, 0.0]),
        principal_component_y=np.array([0.0, 1.0]),
        explained_variance_ratio=np.array([0.6, 0.4]),
        projected_samples=samples,
        scan_component_ids=(0, 1),
        program_type="Synthetic",
    )


def _fourier(power=None) -> Fourier2DResult:
    if power is None:
        power = np.arange(1.0, 13.0).reshape(3, 4)
    return Fourier2DResult(
        frequencies_x=np.array([-2.0, -1.0, 0.0, 1.0]),
        frequencies_y=np.array([-0.25, 0.0, 0.25]),
        power_spectrum=power,
        program_type="Synthetic",
    )


def _neb() -> NEBResult:
    return NEBResult(
        path=np.zeros((3, 2)),
        energies=np.array([0.0, 1.0, 0.5]),
        path_distances=np.array([0.0, 0.4, 1.0]),
        all_paths=[],
        program_type="Synthetic",
    )


_PLOTS_2D = [
    pytest.param(lambda **kw: _scan_1d().plot(**kw), id="scan_1d"),
    pytest.param(lambda **kw: _scan_2d().plot(**kw), id="scan_2d"),
    pytest.param(lambda **kw: _pca_scan().plot(**kw), id="pca"),
    pytest.param(lambda **kw: _fourier().plot(**kw), id="fourier"),
    pytest.param(lambda **kw: _neb().plot(**kw), id="neb"),
]
_PLOTS_3D = [
    pytest.param(lambda **kw: _scan_2d().plot_3d(**kw), id="scan_2d_3d"),
    pytest.param(lambda **kw: _pca_scan().plot_3d(**kw), id="pca_3d"),
]
_COLORBAR_PLOTS = [
    pytest.param(lambda **kw: _scan_2d().plot(**kw), id="scan_2d"),
    pytest.param(lambda **kw: _pca_scan().plot(**kw), id="pca"),
]


def _only(ax, artist_type):
    found = [c for c in ax.get_children() if isinstance(c, artist_type)]
    assert len(found) == 1
    return found[0]


def _mesh(ax) -> QuadMesh:
    return _only(ax, QuadMesh)


@pytest.mark.parametrize("plot", _PLOTS_2D)
def test_plot_draws_into_given_axes(plot):
    fig, ax = plt.subplots()

    out_fig, out_ax = plot(ax=ax)

    assert out_ax is ax
    assert out_fig is fig
    assert plt.get_fignums() == [fig.number]


@pytest.mark.parametrize("plot", _PLOTS_3D)
def test_plot_3d_draws_into_given_axes(plot):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    out_fig, out_ax = plot(ax=ax)

    assert out_ax is ax
    assert out_fig is fig


@pytest.mark.parametrize("plot", _PLOTS_2D + _PLOTS_3D)
@pytest.mark.parametrize(
    "kwargs, n_shows",
    [({}, 0), ({"show": False}, 0), ({"show": True}, 1)],
    ids=["default", "show_false", "show_true"],
)
def test_plot_calls_show_only_when_asked(plot, kwargs, n_shows, mocker):
    show_spy = mocker.patch("matplotlib.pyplot.show")

    plot(**kwargs)

    assert show_spy.call_count == n_shows


@pytest.mark.parametrize("plot", _COLORBAR_PLOTS)
def test_colorbar_is_optional(plot):
    fig, _ = plot()
    assert len(fig.axes) == 2

    fig, _ = plot(add_colorbar=False)
    assert len(fig.axes) == 1


def test_fourier_plot_adds_colorbar():
    fig, _ = _fourier().plot()

    assert len(fig.axes) == 2


@pytest.mark.parametrize(
    "result, xdata, ydata",
    [
        pytest.param(_scan_1d(), "offsets", "values", id="scan_1d"),
        pytest.param(_neb(), "path_distances", "energies", id="neb"),
    ],
)
def test_line_plot_draws_result_arrays(result, xdata, ydata):
    _, ax = result.plot()

    (line,) = ax.lines
    np.testing.assert_array_equal(line.get_xdata(), getattr(result, xdata))
    np.testing.assert_array_equal(line.get_ydata(), getattr(result, ydata))


@pytest.mark.parametrize("result", [_scan_1d(), _neb()], ids=["scan_1d", "neb"])
def test_line_plot_forwards_kwargs(result):
    _, ax = result.plot(alpha=0.3)

    assert ax.lines[0].get_alpha() == 0.3


def test_scan_2d_plot_contours_the_grid():
    _, default_ax = _scan_2d().plot()
    _, ax = _scan_2d().plot(levels=3)

    contour = _only(ax, ContourSet)
    assert contour.zmin == _PLANE.min()
    assert contour.zmax == _PLANE.max()
    assert len(contour.levels) < len(_only(default_ax, ContourSet).levels)
    assert contour.levels[0] <= _PLANE.min()
    assert contour.levels[-1] >= _PLANE.max()


def test_scan_2d_plot_forwards_contour_kwargs():
    _, ax = _scan_2d().plot(cmap="plasma", alpha=0.4)

    contour = _only(ax, ContourSet)
    assert contour.get_cmap().name == "plasma"
    assert contour.get_alpha() == 0.4


@pytest.mark.parametrize("make", [_scan_2d, _pca_scan], ids=["scan_2d", "pca"])
def test_gradient_overlay_uses_per_axis_spacing(make):
    _, ax = make().plot(show_gradients=True, gradient_kwargs={"alpha": 0.25})

    quiver = _only(ax, Quiver)
    np.testing.assert_allclose(quiver.X, _XX.ravel())
    np.testing.assert_allclose(quiver.Y, _YY.ravel())
    np.testing.assert_allclose(quiver.U, 2.0)
    np.testing.assert_allclose(quiver.V, 3.0)
    assert quiver.get_alpha() == 0.25


def test_pca_plot_mesh_matches_grid():
    _, ax = _pca_scan().plot()

    mesh = _mesh(ax)
    np.testing.assert_array_equal(np.ravel(mesh.get_array()), _PLANE.ravel())
    coords = mesh.get_coordinates()
    np.testing.assert_allclose(coords[0, :, 0], _cell_edges_from_centers(_X))
    np.testing.assert_allclose(coords[:, 0, 1], _cell_edges_from_centers(_Y))


def test_pca_plot_overlays_projected_samples():
    _, ax = _pca_scan().plot(sample_kwargs={"s": 99})

    scatter = _only(ax, PathCollection)
    np.testing.assert_array_equal(scatter.get_offsets(), _SAMPLES)
    np.testing.assert_array_equal(scatter.get_sizes(), [99])


def test_pca_plot_samples_can_be_hidden():
    _, ax = _pca_scan().plot(show_samples=False)

    assert not [c for c in ax.get_children() if isinstance(c, PathCollection)]


def test_pca_plot_trajectory_marks_first_and_last_sample():
    _, ax = _pca_scan().plot(show_trajectory=True)

    path, start, end = ax.lines
    np.testing.assert_array_equal(path.get_xdata(), _SAMPLES[:, 0])
    np.testing.assert_array_equal(path.get_ydata(), _SAMPLES[:, 1])
    np.testing.assert_array_equal(start.get_xydata(), [_SAMPLES[0]])
    np.testing.assert_array_equal(end.get_xydata(), [_SAMPLES[-1]])


def test_pca_plot_trajectory_needs_two_samples():
    _, ax = _pca_scan(samples=_SAMPLES[:1]).plot(show_trajectory=True)

    assert not ax.lines


def test_pca_plot_forwards_mesh_kwargs():
    _, ax = _pca_scan().plot(cmap="plasma", alpha=0.4, corner_mask=True)

    mesh = _mesh(ax)
    assert mesh.get_cmap().name == "plasma"
    assert mesh.get_alpha() == 0.4


@pytest.mark.parametrize("levels", [1, 5])
def test_pca_plot_bands_span_finite_range(levels):
    values = _PLANE.copy()
    values[0, 0] = np.nan

    _, ax = _pca_scan(values=values).plot(levels=levels)

    norm = _mesh(ax).norm
    assert isinstance(norm, BoundaryNorm)
    assert len(norm.boundaries) == levels + 1
    assert norm.boundaries[0] == np.nanmin(values)
    assert norm.boundaries[-1] == np.nanmax(values)


def test_pca_plot_constant_grid_is_bracketed():
    _, ax = _pca_scan(values=np.full_like(_PLANE, 2.5)).plot()

    boundaries = _mesh(ax).norm.boundaries
    assert boundaries[0] < 2.5 < boundaries[-1]


def test_pca_plot_rejects_zero_levels():
    with pytest.raises(ValueError, match="levels must be at least 1"):
        _pca_scan().plot(levels=0)


def test_pca_plot_rejects_all_nan_grid():
    with pytest.raises(ValueError, match="No finite values"):
        _pca_scan(values=np.full_like(_PLANE, np.nan)).plot()


@pytest.mark.parametrize(
    "centers, edges",
    [
        pytest.param([1, 3, 4, 8, 11], [0, 2, 3.5, 6, 9.5, 12.5], id="uneven"),
        pytest.param([0.0], [-1.0, 1.0], id="single_at_zero"),
        pytest.param([4.0], [2.0, 6.0], id="single"),
        pytest.param([], [], id="empty"),
    ],
)
def test_cell_edges_from_centers(centers, edges):
    np.testing.assert_allclose(_cell_edges_from_centers(np.array(centers)), edges)


def test_fourier_plot_extent_spans_outer_bin_edges():
    _, ax = _fourier().plot()

    assert ax.images[0].get_extent() == pytest.approx([-2.5, 1.5, -0.375, 0.375])


def test_fourier_plot_log_norm_floors_at_relative_threshold():
    power = np.array([[1e-20, 0.5], [1.0, 0.0]])
    result = Fourier2DResult(
        frequencies_x=np.array([-1.0, 0.0]),
        frequencies_y=np.array([-1.0, 0.0]),
        power_spectrum=power,
        program_type="Synthetic",
    )

    _, ax = result.plot()

    image = ax.images[0]
    assert isinstance(image.norm, LogNorm)
    assert image.norm.vmin == pytest.approx(1e-10)
    assert image.norm.vmax == 1.0
    assert image.origin == "lower"


@pytest.mark.parametrize(
    "power, log_scale",
    [
        pytest.param(np.arange(1.0, 13.0).reshape(3, 4), False, id="linear"),
        pytest.param(np.zeros((3, 4)), True, id="all_zero"),
    ],
)
def test_fourier_plot_linear_norm(power, log_scale):
    _, ax = _fourier(power).plot(log_scale=log_scale)

    assert not isinstance(ax.images[0].norm, LogNorm)


def test_fourier_plot_respects_user_norm():
    norm = Normalize(vmin=0.0, vmax=5.0)

    _, ax = _fourier().plot(norm=norm)

    assert ax.images[0].norm is norm


def test_fourier_plot_forwards_imshow_kwargs():
    _, ax = _fourier().plot(cmap="plasma")

    assert ax.images[0].get_cmap().name == "plasma"
