# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Result containers and plotting helpers for :mod:`divi.viz`."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.colors import BoundaryNorm, LogNorm


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


def _overlay_gradients(
    ax,
    x_offsets: npt.NDArray[np.float64],
    y_offsets: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    gradient_kwargs: dict | None,
) -> None:
    """Overlay a quiver plot of :func:`numpy.gradient` on existing axes."""
    dy = float(y_offsets[1] - y_offsets[0]) if y_offsets.size > 1 else 1.0
    dx = float(x_offsets[1] - x_offsets[0]) if x_offsets.size > 1 else 1.0
    grad_y, grad_x = np.gradient(values, dy, dx)
    xx, yy = np.meshgrid(x_offsets, y_offsets, indexing="xy")
    defaults: dict = {
        "color": "white",
        "alpha": 0.7,
        "scale": None,
        "zorder": 3,
    }
    if gradient_kwargs is not None:
        defaults.update(gradient_kwargs)
    ax.quiver(xx, yy, grad_x, grad_y, **defaults)


def _plot_3d_surface(
    *,
    x_offsets: npt.NDArray[np.float64],
    y_offsets: npt.NDArray[np.float64],
    values: npt.NDArray[np.float64],
    title: str,
    xlabel: str,
    ylabel: str,
    ax=None,
    show: bool = False,
    **surface_kwargs,
):
    """3D surface rendering for 2D scan grids."""
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    xx, yy = np.meshgrid(x_offsets, y_offsets, indexing="xy")
    defaults = {"cmap": "viridis", "edgecolor": "none"}
    defaults.update(surface_kwargs)
    ax.plot_surface(xx, yy, values, **defaults)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel("Objective")
    ax.set_title(title)

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
        show_gradients: bool = False,
        gradient_kwargs: dict | None = None,
        **contour_kwargs,
    ):
        """Plot the sampled 2D landscape and return ``(fig, ax)``.

        Args:
            ax: Optional matplotlib axes to draw into. When omitted, a new
                figure and axes are created.
            show: Whether to call ``matplotlib.pyplot.show()`` after drawing.
            levels: Number of contour levels passed to ``ax.contourf``.
            add_colorbar: Whether to attach a colorbar to the figure.
            show_gradients: Whether to overlay a quiver plot of the numerical
                gradient (computed via :func:`numpy.gradient` on the grid).
            gradient_kwargs: Optional keyword arguments forwarded to
                ``ax.quiver`` for the gradient overlay.
            **contour_kwargs: Additional keyword arguments forwarded to
                ``ax.contourf``.

        Returns:
            tuple: ``(fig, ax)`` for the rendered plot.
        """
        fig, ax = _plot_2d_contourf(
            x_offsets=self.x_offsets,
            y_offsets=self.y_offsets,
            values=self.values,
            title=f"{self.program_type} 2D Scan",
            xlabel="Direction X Offset",
            ylabel="Direction Y Offset",
            ax=ax,
            show=False,
            levels=levels,
            add_colorbar=add_colorbar,
            **contour_kwargs,
        )

        if show_gradients:
            _overlay_gradients(
                ax,
                self.x_offsets,
                self.y_offsets,
                self.values,
                gradient_kwargs,
            )

        if show:
            plt.show()

        return fig, ax

    def plot_3d(self, *, ax=None, show: bool = False, **surface_kwargs):
        """Render the 2D landscape as a 3D surface and return ``(fig, ax)``.

        Args:
            ax: Optional :class:`~mpl_toolkits.mplot3d.axes3d.Axes3D`.
                When omitted, a new figure with a 3D projection is created.
            show: Whether to call ``matplotlib.pyplot.show()`` after drawing.
            **surface_kwargs: Additional keyword arguments forwarded to
                ``ax.plot_surface``.

        Returns:
            tuple: ``(fig, ax)`` for the rendered 3D plot.
        """
        return _plot_3d_surface(
            x_offsets=self.x_offsets,
            y_offsets=self.y_offsets,
            values=self.values,
            title=f"{self.program_type} 2D Scan",
            xlabel="Direction X Offset",
            ylabel="Direction Y Offset",
            ax=ax,
            show=show,
            **surface_kwargs,
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
        show_trajectory: bool = False,
        trajectory_kwargs: dict | None = None,
        show_gradients: bool = False,
        gradient_kwargs: dict | None = None,
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
            show_trajectory: Whether to draw a connected line through
                ``projected_samples`` in order. The samples must have been
                supplied in temporal order (e.g. from
                :meth:`~divi.qprog.VariationalQuantumAlgorithm.param_history`).
            trajectory_kwargs: Optional keyword arguments forwarded to
                ``ax.plot`` for the trajectory line. Defaults to a thin white
                line with start/end markers.
            show_gradients: Whether to overlay a quiver plot of the numerical
                gradient (computed via :func:`numpy.gradient` on the grid).
            gradient_kwargs: Optional keyword arguments forwarded to
                ``ax.quiver`` for the gradient overlay.
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

        if show_trajectory and self.projected_samples.shape[0] >= 2:
            traj_defaults = {
                "color": "white",
                "linewidth": 1.5,
                "alpha": 0.8,
                "zorder": 3,
            }
            if trajectory_kwargs is not None:
                traj_defaults.update(trajectory_kwargs)
            xs = self.projected_samples[:, 0]
            ys = self.projected_samples[:, 1]
            ax.plot(xs, ys, **traj_defaults)
            ax.plot(xs[0], ys[0], "o", color="white", markersize=6, zorder=4)
            ax.plot(xs[-1], ys[-1], "*", color="white", markersize=9, zorder=4)

        if show_gradients:
            _overlay_gradients(
                ax,
                self.x_offsets,
                self.y_offsets,
                self.values,
                gradient_kwargs,
            )

        if show:
            plt.show()

        return fig, ax

    def plot_3d(self, *, ax=None, show: bool = False, **surface_kwargs):
        """Render the PCA landscape as a 3D surface and return ``(fig, ax)``.

        Args:
            ax: Optional :class:`~mpl_toolkits.mplot3d.axes3d.Axes3D`.
                When omitted, a new figure with a 3D projection is created.
            show: Whether to call ``matplotlib.pyplot.show()`` after drawing.
            **surface_kwargs: Additional keyword arguments forwarded to
                ``ax.plot_surface``.

        Returns:
            tuple: ``(fig, ax)`` for the rendered 3D plot.
        """
        c0, c1 = self.scan_component_ids
        return _plot_3d_surface(
            x_offsets=self.x_offsets,
            y_offsets=self.y_offsets,
            values=self.values,
            title=f"{self.program_type} PCA Scan",
            xlabel=f"PC{c0} score",
            ylabel=f"PC{c1} score",
            ax=ax,
            show=show,
            **surface_kwargs,
        )


@dataclass(slots=True)
class HessianResult:
    """Result of a Hessian computation at a parameter point.

    Contains the symmetric Hessian matrix (``hessian``), eigenvalues sorted in
    ascending order (``eigenvalues``), corresponding eigenvectors as columns
    (``eigenvectors``), the evaluation point (``center``), and the program
    class name (``program_type``).
    """

    hessian: npt.NDArray[np.float64]
    eigenvalues: npt.NDArray[np.float64]
    eigenvectors: npt.NDArray[np.float64]
    center: npt.NDArray[np.float64]
    program_type: str

    def top_eigenvectors(self, k: int = 2) -> list[npt.NDArray[np.float64]]:
        """Return the *k* eigenvectors with the largest eigenvalues.

        These correspond to the steepest curvature directions and are natural
        choices for :func:`divi.viz.scan_2d` ``direction_x`` / ``direction_y``.
        """
        return [self.eigenvectors[:, -(i + 1)].copy() for i in range(k)]

    def bottom_eigenvectors(self, k: int = 2) -> list[npt.NDArray[np.float64]]:
        """Return the *k* eigenvectors with the smallest eigenvalues.

        These correspond to the flattest directions.
        """
        return [self.eigenvectors[:, i].copy() for i in range(k)]


@dataclass(slots=True)
class Fourier2DResult:
    """Result of a 2D Fourier analysis on a scan grid.

    Contains the shifted frequency axes (``frequencies_x``, ``frequencies_y``),
    the magnitude-squared of the shifted 2D FFT (``power_spectrum``), and the
    source program class name (``program_type``).
    """

    frequencies_x: npt.NDArray[np.float64]
    frequencies_y: npt.NDArray[np.float64]
    power_spectrum: npt.NDArray[np.float64]
    program_type: str

    def plot(
        self, *, ax=None, show: bool = False, log_scale: bool = True, **imshow_kwargs
    ):
        """Plot the power spectrum and return ``(fig, ax)``.

        Args:
            ax: Optional matplotlib axes. When omitted, a new figure is created.
            show: Whether to call ``matplotlib.pyplot.show()`` after drawing.
            log_scale: Use logarithmic color scale (default ``True``). This
                makes non-DC frequency components visible when the DC component
                dominates. Pass ``False`` for a linear scale.
            **imshow_kwargs: Additional keyword arguments forwarded to
                ``ax.imshow``.

        Returns:
            tuple: ``(fig, ax)`` for the rendered plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        extent = [
            float(self.frequencies_x[0]),
            float(self.frequencies_x[-1]),
            float(self.frequencies_y[0]),
            float(self.frequencies_y[-1]),
        ]
        defaults: dict = {"cmap": "inferno", "aspect": "auto", "origin": "lower"}
        if log_scale and "norm" not in imshow_kwargs:
            floor = float(np.max(self.power_spectrum)) * 1e-10
            positive_vals = self.power_spectrum[self.power_spectrum > 0]
            if positive_vals.size == 0:
                # Degenerate (all-zero) spectrum — fall back to linear scale.
                pass
            else:
                vmin = max(float(np.min(positive_vals)), floor)
                defaults["norm"] = LogNorm(
                    vmin=vmin, vmax=float(np.max(self.power_spectrum))
                )
        defaults.update(imshow_kwargs)
        data = self.power_spectrum.copy()
        if log_scale and "norm" not in imshow_kwargs:
            data = np.where(data > 0, data, float("nan"))
        im = ax.imshow(data, extent=extent, **defaults)
        ax.set_xlabel("Frequency X")
        ax.set_ylabel("Frequency Y")
        ax.set_title(f"{self.program_type} Power Spectrum")
        fig.colorbar(im, ax=ax)

        if show:
            plt.show()

        return fig, ax


@dataclass(slots=True)
class NEBResult:
    """Result of a NEB relaxation.

    Contains the final relaxed chain (``path``, shape ``(n_pivots, n_params)``
    including fixed endpoints), objective values at each pivot (``energies``),
    normalized cumulative distances (``path_distances``, ``[0, 1]``), a
    history of all chains across iterations (``all_paths``), and the program
    class name (``program_type``).
    """

    path: npt.NDArray[np.float64]
    energies: npt.NDArray[np.float64]
    path_distances: npt.NDArray[np.float64]
    all_paths: list[npt.NDArray[np.float64]]
    program_type: str

    def plot(self, *, ax=None, show: bool = False, **plot_kwargs):
        """Plot the energy profile along the relaxed path.

        Args:
            ax: Optional matplotlib axes.
            show: Whether to call ``matplotlib.pyplot.show()``.
            **plot_kwargs: Forwarded to ``ax.plot``.

        Returns:
            tuple: ``(fig, ax)`` for the rendered plot.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        defaults = {"marker": "o", "markersize": 4}
        defaults.update(plot_kwargs)
        ax.plot(self.path_distances, self.energies, **defaults)
        ax.set_xlabel("Normalised path distance")
        ax.set_ylabel("Objective")
        ax.set_title(f"{self.program_type} NEB Energy Profile")

        if show:
            plt.show()

        return fig, ax
