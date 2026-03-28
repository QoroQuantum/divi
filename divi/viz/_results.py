# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Result containers and plotting helpers for :mod:`divi.viz` scans."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.colors import BoundaryNorm


def _cell_edges_from_centers(
    centers: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Bin edges for :func:`matplotlib.axes.Axes.pcolormesh` from 1D cell centers."""
    c = np.asarray(centers, dtype=np.float64)
    n = int(c.size)
    if n == 0:
        return c
    if n == 1:
        span = 1.0 if c[0] == 0.0 else 0.5 * abs(float(c[0]))
        span = max(span, 1e-12)
        return np.array([c[0] - span, c[0] + span], dtype=np.float64)
    half_lo = 0.5 * (c[1] - c[0])
    half_hi = 0.5 * (c[-1] - c[-2])
    inner = 0.5 * (c[:-1] + c[1:])
    return np.concatenate([[c[0] - half_lo], inner, [c[-1] + half_hi]])


def _plot_2d_contourf(
    *,
    x_offsets: npt.NDArray[np.float64],
    y_offsets: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    title: str,
    xlabel: str,
    ylabel: str,
    ax=None,
    show: bool = False,
    levels: int = 20,
    add_colorbar: bool = True,
    **contour_kwargs,
):
    """Smooth filled contours for direction scans (``scan_2d``)."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    xx, yy = np.meshgrid(x_offsets, y_offsets, indexing="xy")
    # corner_mask=True (matplotlib default) leaves triangular holes on rectilinear
    # grids; they read as white patches and often sit near overlay markers.
    cf_kwargs = {"corner_mask": False}
    cf_kwargs.update(contour_kwargs)
    contour = ax.contourf(xx, yy, values, levels=levels, **cf_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if add_colorbar:
        fig.colorbar(contour, ax=ax)

    if show:
        plt.show()

    return fig, ax


def _plot_pca_scan_cells(
    *,
    x_offsets: npt.NDArray[np.float64],
    y_offsets: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    title: str,
    xlabel: str,
    ylabel: str,
    ax=None,
    show: bool = False,
    levels: int = 20,
    add_colorbar: bool = True,
    **plot_kwargs,
):
    """Cell heatmap for PCA scans (avoids large unfilled regions from ``contourf``)."""
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    z = np.asarray(values, dtype=np.float64)
    if not np.any(np.isfinite(z)):
        raise ValueError("No finite values to plot in PCA scan landscape.")

    xe = _cell_edges_from_centers(x_offsets)
    ye = _cell_edges_from_centers(y_offsets)

    pm_kwargs = dict(plot_kwargs)
    pm_kwargs.pop("corner_mask", None)

    cmap_in = pm_kwargs.pop("cmap", None)
    if cmap_in is None:
        cmap_in = plt.rcParams.get("image.cmap", "viridis")
    cmap = plt.get_cmap(cmap_in).copy()
    cmap.set_bad(color=(0.88, 0.88, 0.88, 1.0))

    z_plot = np.ma.masked_invalid(z)

    vmin = float(np.nanmin(z))
    vmax = float(np.nanmax(z))
    if vmin == vmax:
        eps = 1e-12 + 1e-9 * (abs(vmin) + 1.0)
        vmin -= eps
        vmax += eps

    if levels < 1:
        raise ValueError("levels must be at least 1")

    bounds = np.linspace(vmin, vmax, levels + 1)
    norm = BoundaryNorm(bounds, cmap.N, clip=True)

    pm_kwargs.pop("vmin", None)
    pm_kwargs.pop("vmax", None)
    mesh = ax.pcolormesh(
        xe,
        ye,
        z_plot,
        shading="flat",
        cmap=cmap,
        norm=norm,
        **pm_kwargs,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if add_colorbar:
        fig.colorbar(mesh, ax=ax)

    if show:
        plt.show()

    return fig, ax


@dataclass(slots=True)
class Scan1DResult:
    offsets: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]
    parameter_sets: npt.NDArray[np.float64]
    center: npt.NDArray[np.float64]
    direction: npt.NDArray[np.float64]
    program_type: str

    def plot(self, *, ax=None, show: bool = False, **plot_kwargs):
        """Plot the sampled 1D landscape and return ``(fig, ax)``.

        Args:
            ax: Optional matplotlib axes to draw into. When omitted, a new
                figure and axes are created.
            show: Whether to call ``matplotlib.pyplot.show()`` after drawing.
                Defaults to ``False`` so the method works cleanly in scripts and tests.
            **plot_kwargs: Additional keyword arguments forwarded to ``ax.plot``.

        Returns:
            tuple: ``(fig, ax)`` for the rendered plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        ax.plot(self.offsets, self.values, **plot_kwargs)
        ax.set_xlabel("Offset")
        ax.set_ylabel("Objective")
        ax.set_title(f"{self.program_type} 1D Scan")

        if show:
            plt.show()

        return fig, ax


@dataclass(slots=True)
class Scan2DResult:
    x_offsets: npt.NDArray[np.float64]
    y_offsets: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]
    parameter_sets: npt.NDArray[np.float64]
    center: npt.NDArray[np.float64]
    direction_x: npt.NDArray[np.float64]
    direction_y: npt.NDArray[np.float64]
    program_type: str

    def plot(
        self,
        *,
        ax=None,
        show: bool = False,
        levels: int = 20,
        add_colorbar: bool = True,
        **contour_kwargs,
    ):
        """Plot the sampled 2D landscape and return ``(fig, ax)``.

        Args:
            ax: Optional matplotlib axes to draw into. When omitted, a new
                figure and axes are created.
            show: Whether to call ``matplotlib.pyplot.show()`` after drawing.
            levels: Number of contour levels passed to ``ax.contourf``.
            add_colorbar: Whether to attach a colorbar to the figure.
            **contour_kwargs: Additional keyword arguments forwarded to
                ``ax.contourf``.

        Returns:
            tuple: ``(fig, ax)`` for the rendered plot.
        """
        return _plot_2d_contourf(
            x_offsets=self.x_offsets,
            y_offsets=self.y_offsets,
            values=self.values,
            title=f"{self.program_type} 2D Scan",
            xlabel="Direction X Offset",
            ylabel="Direction Y Offset",
            ax=ax,
            show=show,
            levels=levels,
            add_colorbar=add_colorbar,
            **contour_kwargs,
        )


@dataclass(slots=True)
class PCAScanResult:
    x_offsets: npt.NDArray[np.float64]
    y_offsets: npt.NDArray[np.float64]
    values: npt.NDArray[np.float64]
    parameter_sets: npt.NDArray[np.float64]
    center: npt.NDArray[np.float64]
    principal_component_x: npt.NDArray[np.float64]
    principal_component_y: npt.NDArray[np.float64]
    explained_variance_ratio: npt.NDArray[np.float64]
    projected_samples: npt.NDArray[np.float64]
    scan_component_ids: tuple[int, int]
    program_type: str

    def plot(
        self,
        *,
        ax=None,
        show: bool = False,
        levels: int = 20,
        add_colorbar: bool = True,
        show_samples: bool = True,
        sample_kwargs: dict | None = None,
        **contour_kwargs,
    ):
        """Plot the sampled PCA landscape and return ``(fig, ax)``.

        Args:
            ax: Optional matplotlib axes to draw into. When omitted, a new
                figure and axes are created.
            show: Whether to call ``matplotlib.pyplot.show()`` after drawing.
            levels: Number of discrete color bands (``BoundaryNorm``) for
                ``pcolormesh``.
            add_colorbar: Whether to attach a colorbar to the figure.
            show_samples: Whether to overlay the projected PCA samples on top of
                the heatmap.
            sample_kwargs: Optional keyword arguments forwarded to ``ax.scatter``
                for the PCA sample overlay.
            **contour_kwargs: Additional keyword arguments forwarded to
                ``ax.pcolormesh`` (``corner_mask`` is ignored). PCA scans use a
                cell heatmap instead of ``contourf`` so noisy objectives still
                fill the axes.

        Returns:
            tuple: ``(fig, ax)`` for the rendered plot.
        """
        c0, c1 = self.scan_component_ids
        fig, ax = _plot_pca_scan_cells(
            x_offsets=self.x_offsets,
            y_offsets=self.y_offsets,
            values=self.values,
            title=f"{self.program_type} PCA Scan",
            xlabel=f"PC{c0} score",
            ylabel=f"PC{c1} score",
            ax=ax,
            show=False,
            levels=levels,
            add_colorbar=add_colorbar,
            **contour_kwargs,
        )

        if show_samples:
            scatter_kwargs = {
                "s": 28,
                "facecolors": "none",
                "edgecolors": "0.15",
                "linewidths": 1.2,
            }
            if sample_kwargs is not None:
                scatter_kwargs.update(sample_kwargs)
            ax.scatter(
                self.projected_samples[:, 0],
                self.projected_samples[:, 1],
                **scatter_kwargs,
            )

        if show:
            plt.show()

        return fig, ax
