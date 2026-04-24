# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0
#
# Attribution: PCA scan geometry follows patterns described in orqviz (Zapata
# Engineering, Apache-2.0). See LICENSES/ORQViz-Apache-2.0-acknowledgement.txt.

"""Loss-landscape scan API for variational programs.

Line, plane, and PCA scans follow the conventions of ``orqviz`` (Zapata
Engineering, Apache-2.0) where applicable, but all cost evaluations run
through Divi's variational-program batching and cost pipeline.

The :func:`scan_pca` workflow aligns with ``orqviz.pca`` (``PCAobject`` /
``perform_2D_pca_scan``): ``sklearn.decomposition.PCA`` fit on ``samples``, a 2D
grid in PCA *score* space with axis limits from the projected sample cloud plus
``offset``, and reconstruction via ``inverse_transform`` into full parameter
space. An optional ``center`` argument applies an extra translation
``(center - sample_mean)`` after reconstruction so scans can be anchored away
from the sample mean (e.g. at ``best_params``).

:func:`scan_1d` and :func:`scan_2d` use orqviz-like defaults: scalar spans
default to :math:`(-\\pi, \\pi)`, omitted directions are drawn at random
(orthogonal in 2D), and ``normalize_directions=False`` recovers the raw
``center + t \\cdot \\mathrm{direction}`` scaling used in ``orqviz`` line and
plane scans.
"""

from dataclasses import dataclass
from typing import Protocol, TypeAlias

import numpy as np
import numpy.typing as npt
from sklearn.decomposition import PCA

from divi.reporting import LoggingProgressReporter, ProgressReporter

from ._gradients import GradientMethod, _compute_gradients
from ._neb import (
    _cumulative_distances,
    _neb_perpendicular_gradients,
    _redistribute_uniform,
)
from ._results import (
    Fourier2DResult,
    HessianResult,
    NEBResult,
    PCAScanResult,
    Scan1DResult,
    Scan2DResult,
)

# orqviz-style 1D/2D scans use scalar endpoints in (-pi, pi) along the direction vector.
_DEFAULT_SCAN_SPAN: tuple[float, float] = (-np.pi, np.pi)

VizRng: TypeAlias = np.random.Generator | int | None
OptionalArray: TypeAlias = npt.ArrayLike | None


class _SupportsVizScan(Protocol):
    n_layers: int
    n_params_per_layer: int
    _best_params: npt.NDArray[np.float64]

    def _has_run_optimization(self) -> bool: ...

    def _evaluate_cost_param_sets(
        self, param_sets: npt.NDArray[np.float64], **kwargs
    ) -> dict[int, float]: ...


def _resolve_viz_reporter(program: _SupportsVizScan) -> ProgressReporter:
    """Return the program's reporter, falling back to a logging reporter."""
    r = getattr(program, "reporter", None)
    return r if isinstance(r, ProgressReporter) else LoggingProgressReporter()


def _evaluate_param_sets_reported(
    program: _SupportsVizScan,
    param_sets: npt.NDArray[np.float64],
    *,
    reporter: ProgressReporter,
    scan_label: str,
) -> npt.NDArray[np.float64]:
    """Single batched evaluation with start/finish messaging."""
    n_points = int(param_sets.shape[0])
    reporter.info(
        message=(f"💸 divi.viz {scan_label}: evaluating {n_points} parameter set(s)"),
        iteration=0,
    )
    values = _evaluate_param_sets(program, param_sets)
    reporter.info(
        f"divi.viz {scan_label}: finished ({n_points} evaluations).",
    )
    return values


def _n_program_params(program: _SupportsVizScan) -> int:
    return int(program.n_layers * program.n_params_per_layer)


def _require_supported_program(program: _SupportsVizScan) -> None:
    """Raise if *program* lacks the viz protocol or has a variable parameter space."""
    required_attrs = ("n_layers", "n_params_per_layer", "_best_params")
    required_methods = ("_has_run_optimization", "_evaluate_cost_param_sets")
    if not all(hasattr(program, attr) for attr in required_attrs) or not all(
        callable(getattr(program, name, None)) for name in required_methods
    ):
        raise TypeError("divi.viz currently supports VariationalQuantumAlgorithm only.")

    if not getattr(program, "_supports_fixed_param_scans", True):
        raise NotImplementedError(
            f"{type(program).__name__} varies its parameter space during optimization "
            "and has no fixed parameter space for these scans; use a fixed-depth variant."
        )


def _resolve_center(
    program: _SupportsVizScan,
    center: OptionalArray,
) -> npt.NDArray[np.float64]:
    """Return *center* as a flat float64 array, defaulting to ``program._best_params``."""
    n_params = _n_program_params(program)

    if center is None:
        if not program._has_run_optimization() or len(program._best_params) == 0:
            raise ValueError(
                "center must be provided unless the program already has best_params "
                "from a previous optimization run."
            )
        center_arr = np.asarray(program._best_params, dtype=np.float64).reshape(-1)
    else:
        center_arr = np.asarray(center, dtype=np.float64).reshape(-1)

    if center_arr.shape != (n_params,):
        raise ValueError(
            f"center must have shape ({n_params},), got {center_arr.shape}"
        )

    return center_arr


def _resolve_viz_rng(rng: VizRng) -> np.random.Generator:
    """Normalise *rng* to a :class:`numpy.random.Generator`."""
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, int):
        return np.random.default_rng(rng)
    return rng


def _random_gaussian_direction(
    n_params: int, rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    """Sample a random direction from a standard Gaussian."""
    v = rng.standard_normal(n_params).astype(np.float64, copy=False)
    if float(np.linalg.norm(v)) < 1e-15:  # practically impossible
        raise RuntimeError("Sampled a near-zero direction; try a different rng seed.")
    return v


def _resolve_1d_direction(
    direction: OptionalArray,
    *,
    n_params: int,
    rng: np.random.Generator,
    normalize_directions: bool,
) -> npt.NDArray[np.float64]:
    """Validate or generate a 1D scan direction, optionally unit-normalizing."""
    if direction is None:
        direction_arr = _random_gaussian_direction(n_params, rng)
    else:
        direction_arr = np.asarray(direction, dtype=np.float64).reshape(-1)

    if direction_arr.shape != (n_params,):
        raise ValueError(
            f"direction must have shape ({n_params},), got {direction_arr.shape}"
        )

    norm = float(np.linalg.norm(direction_arr))
    if norm == 0.0:
        raise ValueError("direction must be non-zero.")

    if normalize_directions:
        return direction_arr / norm

    return direction_arr


def _gram_schmidt_remove_component(
    v: npt.NDArray[np.float64], ref: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    ref_sq = float(np.dot(ref, ref))
    if ref_sq < 1e-30:
        return v
    return v - (float(np.dot(v, ref)) / ref_sq) * ref


def _resolve_2d_directions(
    direction_x: OptionalArray,
    direction_y: OptionalArray,
    *,
    n_params: int,
    rng: np.random.Generator,
    normalize_directions: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Validate or generate orthogonal 2D scan directions."""
    has_x = direction_x is not None
    has_y = direction_y is not None

    if has_x and has_y:
        dx = np.asarray(direction_x, dtype=np.float64).reshape(-1)
        dy = np.asarray(direction_y, dtype=np.float64).reshape(-1)
        if dx.shape != (n_params,) or dy.shape != (n_params,):
            raise ValueError(
                f"direction_x and direction_y must have shape ({n_params},), "
                f"got {dx.shape} and {dy.shape}"
            )
        nx = float(np.linalg.norm(dx))
        ny = float(np.linalg.norm(dy))
        if nx == 0.0 or ny == 0.0:
            raise ValueError("direction_x and direction_y must be non-zero.")
        if normalize_directions:
            dx = dx / nx
            dy = dy / ny
    elif has_x:
        dx = np.asarray(direction_x, dtype=np.float64).reshape(-1)
        if dx.shape != (n_params,):
            raise ValueError(
                f"direction_x must have shape ({n_params},), got {dx.shape}"
            )
        nx = float(np.linalg.norm(dx))
        if nx == 0.0:
            raise ValueError("direction_x must be non-zero.")
        dy_raw = _random_gaussian_direction(n_params, rng)
        dy = _gram_schmidt_remove_component(dy_raw, dx)
        ny = float(np.linalg.norm(dy))
        if ny < 1e-15:
            return _resolve_2d_directions(
                direction_x,
                direction_y,
                n_params=n_params,
                rng=rng,
                normalize_directions=normalize_directions,
            )
        if normalize_directions:
            dx = dx / nx
            dy = dy / ny
        else:
            dy = (dy / ny) * nx
    elif has_y:
        dy = np.asarray(direction_y, dtype=np.float64).reshape(-1)
        if dy.shape != (n_params,):
            raise ValueError(
                f"direction_y must have shape ({n_params},), got {dy.shape}"
            )
        ny = float(np.linalg.norm(dy))
        if ny == 0.0:
            raise ValueError("direction_y must be non-zero.")
        dx_raw = _random_gaussian_direction(n_params, rng)
        dx = _gram_schmidt_remove_component(dx_raw, dy)
        nx = float(np.linalg.norm(dx))
        if nx < 1e-15:
            return _resolve_2d_directions(
                direction_x,
                direction_y,
                n_params=n_params,
                rng=rng,
                normalize_directions=normalize_directions,
            )
        if normalize_directions:
            dx = dx / nx
            dy = dy / ny
        else:
            dx = (dx / nx) * ny
    else:
        dx = _random_gaussian_direction(n_params, rng)
        nx = float(np.linalg.norm(dx))
        dy_raw = _random_gaussian_direction(n_params, rng)
        dy = _gram_schmidt_remove_component(dy_raw, dx)
        ny = float(np.linalg.norm(dy))
        if ny < 1e-15:
            return _resolve_2d_directions(
                direction_x,
                direction_y,
                n_params=n_params,
                rng=rng,
                normalize_directions=normalize_directions,
            )
        if normalize_directions:
            dx = dx / nx
            dy = dy / ny
        else:
            dy = (dy / ny) * nx

    nxu = float(np.linalg.norm(dx))
    nyu = float(np.linalg.norm(dy))
    overlap = abs(float(np.dot(dx, dy)) / (nxu * nyu))
    if overlap > 1.0 - 1e-8:
        raise ValueError("direction_x and direction_y must be linearly independent.")

    return dx, dy


def _evaluate_param_sets(
    program: _SupportsVizScan,
    param_sets: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Evaluate *param_sets* and return losses as a sorted 1D array."""
    losses = program._evaluate_cost_param_sets(param_sets)
    return np.asarray([value for _, value in sorted(losses.items())], dtype=np.float64)


def _resolve_samples(
    samples: npt.ArrayLike,
    *,
    n_params: int,
) -> npt.NDArray[np.float64]:
    """Validate and convert *samples* to a ``(n_samples, n_params)`` float64 array."""
    sample_arr = np.asarray(samples, dtype=np.float64)
    if sample_arr.ndim != 2:
        raise ValueError("samples must be a 2D array of shape (n_samples, n_params).")
    if sample_arr.shape[1] != n_params:
        raise ValueError(
            f"samples must have shape (n_samples, {n_params}), got {sample_arr.shape}"
        )
    if sample_arr.shape[0] < 2:
        raise ValueError("samples must contain at least two parameter vectors.")
    return sample_arr


def _normalize_pca_offset(
    offset: float | tuple[float, float],
) -> tuple[float, float]:
    """Match orqviz: scalar offset expands to (-|a|, |a|) padding on min/max."""
    if isinstance(offset, tuple):
        return (float(offset[0]), float(offset[1]))
    a = float(abs(offset))
    return (-a, a)


def _pca_scan_endpoints(
    scores: npt.NDArray[np.float64],
    i0: int,
    i1: int,
    offset: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Axis limits in PCA score space (orqviz ``PCAobject._get_endpoints_from_pca``)."""
    c0 = scores[:, i0].astype(np.float64, copy=False)
    c1 = scores[:, i1].astype(np.float64, copy=False)
    lo0, hi0 = float(np.min(c0)), float(np.max(c0))
    lo1, hi1 = float(np.min(c1)), float(np.max(c1))
    eps = 1e-12
    if hi0 - lo0 < eps:
        lo0, hi0 = lo0 - 1.0, hi0 + 1.0
    if hi1 - lo1 < eps:
        lo1, hi1 = lo1 - 1.0, hi1 + 1.0
    end_x = (lo0 + offset[0], hi0 + offset[1])
    end_y = (lo1 + offset[0], hi1 + offset[1])
    return end_x, end_y


def _param_sets_from_pca_grid(
    pca: PCA,
    *,
    x_offsets: npt.NDArray[np.float64],
    y_offsets: npt.NDArray[np.float64],
    components_ids: tuple[int, int],
    shift: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Map PCA (score_x, score_y) grid to parameter vectors (orqviz ``perform_2D_pca_scan``)."""
    i0, i1 = components_ids
    n_comp = int(pca.n_components_)

    # Build a (n_y * n_x, n_comp) coefficient matrix and call inverse_transform once.
    xx, yy = np.meshgrid(x_offsets, y_offsets, indexing="xy")
    n_points = xx.size
    coefs = np.zeros((n_points, n_comp), dtype=np.float64)
    coefs[:, i0] = xx.ravel()
    coefs[:, i1] = yy.ravel()
    return (pca.inverse_transform(coefs) + shift).astype(np.float64)


def scan_1d(
    program: _SupportsVizScan,
    *,
    center: OptionalArray = None,
    direction: OptionalArray = None,
    n_points: int = 51,
    span: tuple[float, float] = _DEFAULT_SCAN_SPAN,
    normalize_directions: bool = True,
    rng: VizRng = None,
    reporter: ProgressReporter | None = None,
) -> Scan1DResult:
    """One-dimensional loss-landscape scan for a variational program.

    The scan is constructed by taking a center point in parameter space and a
    direction vector, then evaluating the program's cost function at evenly
    spaced offsets along that line. Internally, the scan uses
    :meth:`~divi.qprog.VariationalQuantumAlgorithm._evaluate_cost_param_sets`
    directly, so it reuses the program's normal batched evaluation path instead
    of going through ``run()``.

    Geometry matches the usual ``orqviz`` 1D pattern when ``span`` uses the
    default :math:`(-\\pi, \\pi)` and ``normalize_directions`` is ``False``: each
    offset :math:`t` maps to ``center + t * direction`` with the direction norm
    affecting Euclidean step size. The default ``normalize_directions=True``
    keeps a unit direction so ``span`` is measured along a normalized axis.

    Args:
        program: Variational program to scan.
        center: Flat parameter vector around which the scan is performed. If
            omitted, the scan centers on ``program.best_params`` from a previous
            optimization run.
        direction: Flat direction vector in parameter space. If omitted, a
            random direction is drawn (reproducible via ``rng``).
        n_points: Number of sample points along the scan line. Must be at least 2.
        span: Inclusive scalar offset range applied along ``direction`` (orqviz
            ``end_points`` default is :math:`(-\\pi, \\pi)`).
        normalize_directions: If ``True`` (default), non-zero ``direction`` is
            unit-normalized before building ``center + t * direction``.
        rng: Optional :class:`numpy.random.Generator` or integer seed used when
            ``direction`` is ``None``.
        reporter: Optional :class:`~divi.reporting.ProgressReporter`. When omitted,
            uses ``program.reporter`` if present, otherwise
            :class:`~divi.reporting.LoggingProgressReporter`. Start/finish messages
            are emitted around the single batched evaluation (same pattern as
            :class:`~divi.qprog.VariationalQuantumAlgorithm.run`).
    Returns:
        Scan1DResult: Object containing offsets, sampled loss values, the
        concrete parameter sets that were evaluated, and plotting helpers.

    Raises:
        TypeError: If ``program`` is not a supported
            :class:`~divi.qprog.variational_quantum_algorithm.VariationalQuantumAlgorithm`.
        NotImplementedError: If ``program`` is an ``IterativeQAOA`` instance.
        ValueError: If ``center`` or ``direction`` has the wrong shape, if
            ``direction`` is zero, or if ``n_points`` is invalid.
    """
    _require_supported_program(program)
    rep = reporter if reporter is not None else _resolve_viz_reporter(program)

    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")

    center_arr = _resolve_center(program, center)
    rng_gen = _resolve_viz_rng(rng)
    direction_arr = _resolve_1d_direction(
        direction,
        n_params=center_arr.size,
        rng=rng_gen,
        normalize_directions=normalize_directions,
    )

    offsets = np.linspace(span[0], span[1], n_points, dtype=np.float64)
    # (n_points, n_params): row i = center + offsets[i] * direction
    param_sets = center_arr + offsets[:, None] * direction_arr[None, :]
    values = _evaluate_param_sets_reported(
        program,
        param_sets,
        reporter=rep,
        scan_label="1D scan",
    )

    return Scan1DResult(
        offsets=offsets,
        values=values,
        parameter_sets=param_sets,
        center=center_arr,
        direction=direction_arr,
        program_type=type(program).__name__,
    )


def scan_2d(
    program: _SupportsVizScan,
    *,
    center: OptionalArray = None,
    direction_x: OptionalArray = None,
    direction_y: OptionalArray = None,
    grid_shape: tuple[int, int] = (41, 41),
    span_x: tuple[float, float] = _DEFAULT_SCAN_SPAN,
    span_y: tuple[float, float] = _DEFAULT_SCAN_SPAN,
    normalize_directions: bool = True,
    rng: VizRng = None,
    reporter: ProgressReporter | None = None,
) -> Scan2DResult:
    """Two-dimensional loss-landscape scan for a variational program.

    The scan is constructed from a center point plus two linearly independent
    directions in parameter space. A rectangular grid of offsets is generated,
    converted into concrete parameter sets, and evaluated using the program's
    batched cost pipeline.

    When either direction is omitted, the missing axis is filled with a random
    vector orthogonal to the other (same norm as the reference axis when
    ``normalize_directions`` is ``False``, matching the usual ``orqviz`` 2D scan
    construction). Use ``rng`` for reproducibility.

    Args:
        program: Variational program to scan.
        center: Flat parameter vector around which the scan plane is defined. If
            omitted, the scan centers on ``program.best_params`` from a previous
            optimization run.
        direction_x: First direction spanning the scan plane. If omitted, a
            random direction is drawn unless ``direction_y`` alone is given, in
            which case ``direction_x`` is a random vector orthogonal to
            ``direction_y``.
        direction_y: Second direction spanning the scan plane. If omitted, a
            random direction orthogonal to ``direction_x`` is used (orqviz-style).
        grid_shape: Number of sample points along the x and y scan directions.
            Both entries must be at least 2.
        span_x: Inclusive offset range along ``direction_x`` (default
            :math:`(-\\pi, \\pi)`).
        span_y: Inclusive offset range along ``direction_y``.
        normalize_directions: If ``True`` (default), user-supplied directions are
            unit-normalized; random supplementary directions are unit vectors. If
            ``False``, coefficients multiply the raw direction vectors; random
            pairs keep matched norms like ``orqviz``.
        rng: Optional :class:`numpy.random.Generator` or integer seed for any
            randomly generated directions.
        reporter: See :func:`scan_1d`.
    Returns:
        Scan2DResult: Object containing the sampled grid, loss values,
        evaluated parameter sets, and plotting helpers.

    Raises:
        TypeError: If ``program`` is not a supported variational program.
        NotImplementedError: If ``program`` is an ``IterativeQAOA`` instance.
        ValueError: If the directions have the wrong shape, are zero, are not
            linearly independent, or if ``grid_shape`` is invalid.
    """
    _require_supported_program(program)
    rep = reporter if reporter is not None else _resolve_viz_reporter(program)

    n_x, n_y = grid_shape
    if n_x < 2 or n_y < 2:
        raise ValueError(f"grid_shape entries must be >= 2, got {grid_shape}")

    center_arr = _resolve_center(program, center)
    rng_gen = _resolve_viz_rng(rng)
    direction_x_arr, direction_y_arr = _resolve_2d_directions(
        direction_x,
        direction_y,
        n_params=center_arr.size,
        rng=rng_gen,
        normalize_directions=normalize_directions,
    )

    x_offsets = np.linspace(span_x[0], span_x[1], n_x, dtype=np.float64)
    y_offsets = np.linspace(span_y[0], span_y[1], n_y, dtype=np.float64)

    # (n_y*n_x, n_params): meshgrid broadcast of center + x*dx + y*dy
    xx, yy = np.meshgrid(x_offsets, y_offsets, indexing="xy")
    param_sets = (
        center_arr
        + xx.ravel()[:, None] * direction_x_arr
        + yy.ravel()[:, None] * direction_y_arr
    )
    values = _evaluate_param_sets_reported(
        program,
        param_sets,
        reporter=rep,
        scan_label="2D scan",
    ).reshape(n_y, n_x)

    return Scan2DResult(
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        values=values,
        parameter_sets=param_sets,
        center=center_arr,
        direction_x=direction_x_arr,
        direction_y=direction_y_arr,
        program_type=type(program).__name__,
    )


def scan_pca(
    program: _SupportsVizScan,
    *,
    samples: npt.ArrayLike,
    center: OptionalArray = None,
    grid_shape: tuple[int, int] = (41, 41),
    components_ids: tuple[int, int] = (0, 1),
    offset: float | tuple[float, float] = (-1.0, 1.0),
    span_x: tuple[float, float] | None = None,
    span_y: tuple[float, float] | None = None,
    reporter: ProgressReporter | None = None,
) -> PCAScanResult:
    """Evaluate a 2D scan in PCA score space (orqviz-compatible layout).

    Fits :class:`sklearn.decomposition.PCA` on ``samples``, builds a rectangular
    grid in the selected principal-component *scores* (default PC0 vs PC1), maps
    each grid point through ``inverse_transform`` into full parameter space, and
    evaluates the program's cost function in one batch.

    A typical choice for ``samples`` is the optimization trajectory::

        samples = np.vstack(program.param_history(mode="best_per_iteration"))

    When using ``program.viz.scan_pca()``, samples are collected automatically
    if omitted.

    When ``span_x`` and ``span_y`` are both omitted, axis limits match orqviz
    ``PCAobject._get_endpoints_from_pca``: min/max of the projected ``samples``
    on each selected component, plus ``offset`` applied to the low and high ends.

    Args:
        program: Variational program to scan.
        samples: Rows are parameter vectors used to fit PCA; shape
            ``(n_samples, n_params)``.
        center: Optional anchor in parameter space. If ``None``, the scan
            plane is the standard PCA affine subspace through the sample mean
            (orqviz default). If provided (e.g. ``program.best_params``), each
            reconstructed point is shifted by ``(center - sample_mean)`` so the
            same PC directions pass through ``center``.
        grid_shape: ``(n_x, n_y)`` grid resolution in score space; both ≥ 2.
        components_ids: Which PCA components form the scan axes (default
            ``(0, 1)``). ``n_components`` is set to ``max(components_ids) + 1``.
        offset: Extra padding on score-axis limits when using automatic spans.
            A scalar ``a`` becomes ``(-|a|, |a|)`` on each end (orqviz rule).
        span_x: If given together with ``span_y``, fixed PC0 score range
            (bypasses automatic endpoints).
        span_y: Fixed PC1 score range; must be set if ``span_x`` is set.
        reporter: See :func:`scan_1d`.

    Returns:
        PCAScanResult: Grid in PCA score coordinates, loss values, full
        parameter sets, sklearn PC directions for the chosen indices, explained
        variance ratios, and projected ``samples`` scores for plotting.

    Raises:
        TypeError: If ``program`` is not a supported variational program.
        NotImplementedError: If ``program`` is an ``IterativeQAOA`` instance.
        ValueError: If shapes, ranks, ``grid_shape``, or span arguments are invalid.
    """
    _require_supported_program(program)
    rep = reporter if reporter is not None else _resolve_viz_reporter(program)

    n_x, n_y = grid_shape
    if n_x < 2 or n_y < 2:
        raise ValueError(f"grid_shape entries must be >= 2, got {grid_shape}")

    n_params = _n_program_params(program)
    sample_arr = _resolve_samples(samples, n_params=n_params)
    mean = np.mean(sample_arr, axis=0)
    if np.linalg.matrix_rank(sample_arr - mean, tol=1e-10) < 2:
        raise ValueError(
            "samples must span at least two independent directions for PCA scans."
        )

    i0, i1 = components_ids
    if i0 == i1:
        raise ValueError("components_ids must be two distinct integers.")
    if i0 < 0 or i1 < 0:
        raise ValueError("components_ids must be non-negative.")
    n_comp = max(i0, i1) + 1
    n_comp_cap = min(sample_arr.shape[0], n_params)
    if n_comp > n_comp_cap:
        raise ValueError(
            "components_ids require more PCA components than allowed by the "
            f"sample count and parameter dimension (need {n_comp}, cap {n_comp_cap})."
        )

    pca = PCA(n_components=n_comp)
    pca.fit(sample_arr)
    scores = pca.transform(sample_arr)

    pc_x = np.asarray(pca.components_[i0], dtype=np.float64)
    pc_y = np.asarray(pca.components_[i1], dtype=np.float64)

    if center is None:
        anchor = mean.astype(np.float64, copy=True)
    else:
        anchor = np.asarray(center, dtype=np.float64).reshape(-1)
        if anchor.shape != (n_params,):
            raise ValueError(
                f"center must have shape ({n_params},), got {anchor.shape}"
            )
    shift = anchor - mean

    # Project the shift onto the selected PCs so that projected_samples and
    # auto-computed endpoints are expressed relative to *center*, not the
    # sample mean.  When center is None the shift is zero and this is a no-op.
    shift_on_i0 = float(np.dot(shift, pc_x))
    shift_on_i1 = float(np.dot(shift, pc_y))
    centered_scores_i0 = scores[:, i0] - shift_on_i0
    centered_scores_i1 = scores[:, i1] - shift_on_i1

    if span_x is not None or span_y is not None:
        if span_x is None or span_y is None:
            raise ValueError("span_x and span_y must both be set or both omitted.")
        end_x = (float(span_x[0]), float(span_x[1]))
        end_y = (float(span_y[0]), float(span_y[1]))
    else:
        centered_scores = np.column_stack((centered_scores_i0, centered_scores_i1))
        end_x, end_y = _pca_scan_endpoints(
            centered_scores, 0, 1, _normalize_pca_offset(offset)
        )

    x_offsets = np.linspace(end_x[0], end_x[1], n_x, dtype=np.float64)
    y_offsets = np.linspace(end_y[0], end_y[1], n_y, dtype=np.float64)

    param_sets = _param_sets_from_pca_grid(
        pca,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        components_ids=(i0, i1),
        shift=shift,
    )

    flat_values = _evaluate_param_sets_reported(
        program,
        param_sets,
        reporter=rep,
        scan_label="PCA scan",
    )
    values = flat_values.reshape(n_y, n_x)

    explained_ratio = np.asarray(
        pca.explained_variance_ratio_[[i0, i1]], dtype=np.float64
    )
    projected_samples = np.column_stack((centered_scores_i0, centered_scores_i1))

    return PCAScanResult(
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        values=values,
        parameter_sets=param_sets,
        center=anchor,
        principal_component_x=pc_x,
        principal_component_y=pc_y,
        explained_variance_ratio=explained_ratio,
        projected_samples=projected_samples,
        scan_component_ids=(i0, i1),
        program_type=type(program).__name__,
    )


def scan_interp_1d(
    program: _SupportsVizScan,
    theta_1: npt.ArrayLike,
    theta_2: npt.ArrayLike,
    *,
    n_points: int = 51,
    reporter: ProgressReporter | None = None,
) -> Scan1DResult:
    """One-dimensional interpolation scan between two parameter vectors.

    Scans the line ``(1 - t) * theta_1 + t * theta_2`` for *t* in ``[0, 1]``,
    which is the orqviz ``perform_1D_interpolation`` pattern.  At ``t = 0`` the
    parameter vector equals *theta_1*, at ``t = 1`` it equals *theta_2*.

    Args:
        program: Variational program to scan.
        theta_1: Starting parameter vector.
        theta_2: Ending parameter vector.
        n_points: Number of sample points along the interpolation line.
            Must be at least 2.
        reporter: Optional :class:`~divi.reporting.ProgressReporter`.

    Returns:
        Scan1DResult: ``offsets`` are the *t* values; ``center`` is *theta_1*;
        ``direction`` is ``theta_2 - theta_1``.

    Raises:
        TypeError: If ``program`` is not a supported variational program.
        NotImplementedError: If ``program`` is an ``IterativeQAOA`` instance.
        ValueError: If shapes do not match or ``n_points`` is invalid.
    """
    _require_supported_program(program)
    rep = reporter if reporter is not None else _resolve_viz_reporter(program)

    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")

    n_params = _n_program_params(program)
    t1 = np.asarray(theta_1, dtype=np.float64).reshape(-1)
    t2 = np.asarray(theta_2, dtype=np.float64).reshape(-1)

    if t1.shape != (n_params,):
        raise ValueError(f"theta_1 must have shape ({n_params},), got {t1.shape}")
    if t2.shape != (n_params,):
        raise ValueError(f"theta_2 must have shape ({n_params},), got {t2.shape}")

    direction = t2 - t1
    offsets = np.linspace(0.0, 1.0, n_points, dtype=np.float64)
    param_sets = t1 + offsets[:, None] * direction[None, :]
    values = _evaluate_param_sets_reported(
        program,
        param_sets,
        reporter=rep,
        scan_label="1D interpolation",
    )

    return Scan1DResult(
        offsets=offsets,
        values=values,
        parameter_sets=param_sets,
        center=t1,
        direction=direction,
        program_type=type(program).__name__,
    )


def scan_interp_2d(
    program: _SupportsVizScan,
    theta_1: npt.ArrayLike,
    theta_2: npt.ArrayLike,
    *,
    direction_y: OptionalArray = None,
    grid_shape: tuple[int, int] = (41, 41),
    span_x: tuple[float, float] = (-0.5, 1.5),
    span_y: tuple[float, float] = (-0.5, 0.5),
    rng: VizRng = None,
    reporter: ProgressReporter | None = None,
) -> Scan2DResult:
    """Two-dimensional interpolation scan between two parameter vectors.

    The x-direction is the unnormalized vector ``theta_2 - theta_1``, so
    offset ``t = 0`` corresponds to *theta_1* and ``t = 1`` to *theta_2*.  The
    y-direction is orthogonal to the interpolation vector and may be supplied or
    generated randomly.  This is the orqviz ``perform_2D_interpolation`` pattern.

    Default x-span ``(-0.5, 1.5)`` extends the interpolation line by half its
    length in each direction; default y-span ``(-0.5, 0.5)`` shows a symmetric
    strip around the interpolation axis.

    Args:
        program: Variational program to scan.
        theta_1: Starting parameter vector.
        theta_2: Ending parameter vector.
        direction_y: Optional y-direction. If omitted, a random vector
            orthogonal to the interpolation direction is used (same norm,
            matching orqviz).
        grid_shape: ``(n_x, n_y)`` grid resolution. Both entries must be >= 2.
        span_x: Offset range along the interpolation direction in units of
            ``theta_2 - theta_1``.
        span_y: Offset range along ``direction_y`` in units of the y-direction
            norm.
        rng: Optional seed or :class:`numpy.random.Generator` for the random
            y-direction.
        reporter: Optional :class:`~divi.reporting.ProgressReporter`.

    Returns:
        Scan2DResult: ``center`` is *theta_1*; ``direction_x`` is
        ``theta_2 - theta_1``; ``x_offsets`` are the *t* values.

    Raises:
        TypeError: If ``program`` is not a supported variational program.
        NotImplementedError: If ``program`` is an ``IterativeQAOA`` instance.
        ValueError: If shapes do not match, directions are zero/parallel, or
            ``grid_shape`` is invalid.
    """
    _require_supported_program(program)
    rep = reporter if reporter is not None else _resolve_viz_reporter(program)

    n_x, n_y = grid_shape
    if n_x < 2 or n_y < 2:
        raise ValueError(f"grid_shape entries must be >= 2, got {grid_shape}")

    n_params = _n_program_params(program)
    t1 = np.asarray(theta_1, dtype=np.float64).reshape(-1)
    t2 = np.asarray(theta_2, dtype=np.float64).reshape(-1)
    if t1.shape != (n_params,):
        raise ValueError(f"theta_1 must have shape ({n_params},), got {t1.shape}")
    if t2.shape != (n_params,):
        raise ValueError(f"theta_2 must have shape ({n_params},), got {t2.shape}")

    interp_dir = t2 - t1
    interp_norm = float(np.linalg.norm(interp_dir))
    if interp_norm < 1e-15:
        raise ValueError("theta_1 and theta_2 must be distinct.")

    rng_gen = _resolve_viz_rng(rng)

    if direction_y is not None:
        dy = np.asarray(direction_y, dtype=np.float64).reshape(-1)
        if dy.shape != (n_params,):
            raise ValueError(
                f"direction_y must have shape ({n_params},), got {dy.shape}"
            )
        dy_norm = float(np.linalg.norm(dy))
        if dy_norm < 1e-15:
            raise ValueError("direction_y must be non-zero.")
    else:
        dy_raw = _random_gaussian_direction(n_params, rng_gen)
        dy = _gram_schmidt_remove_component(dy_raw, interp_dir)
        dy_norm = float(np.linalg.norm(dy))
        if dy_norm < 1e-15:
            # Extremely unlikely with random vectors; retry once.
            dy_raw = _random_gaussian_direction(n_params, rng_gen)
            dy = _gram_schmidt_remove_component(dy_raw, interp_dir)
            dy_norm = float(np.linalg.norm(dy))
        # Match orqviz: random y-direction has same norm as the interpolation vector.
        dy = (dy / dy_norm) * interp_norm

    overlap = abs(
        float(np.dot(interp_dir, dy)) / (interp_norm * float(np.linalg.norm(dy)))
    )
    if overlap > 1.0 - 1e-8:
        raise ValueError("direction_y must not be parallel to theta_2 - theta_1.")

    x_offsets = np.linspace(span_x[0], span_x[1], n_x, dtype=np.float64)
    y_offsets = np.linspace(span_y[0], span_y[1], n_y, dtype=np.float64)

    xx, yy = np.meshgrid(x_offsets, y_offsets, indexing="xy")
    param_sets = t1 + xx.ravel()[:, None] * interp_dir + yy.ravel()[:, None] * dy

    values = _evaluate_param_sets_reported(
        program,
        param_sets,
        reporter=rep,
        scan_label="2D interpolation",
    ).reshape(n_y, n_x)

    return Scan2DResult(
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        values=values,
        parameter_sets=param_sets,
        center=t1,
        direction_x=interp_dir,
        direction_y=dy,
        program_type=type(program).__name__,
    )


def compute_hessian(
    program: _SupportsVizScan,
    center: npt.ArrayLike,
    *,
    gradient_method: GradientMethod = GradientMethod.PARAMETER_SHIFT,
    eps: float = 1e-3,
    reporter: ProgressReporter | None = None,
) -> HessianResult:
    """Compute the Hessian matrix at *center*.

    Eigenvalues reveal local curvature; eigenvectors can be used as scan
    directions via :meth:`HessianResult.top_eigenvectors`.

    All evaluations are batched into a single call to
    ``_evaluate_cost_param_sets``.  The total number of evaluation points is
    :math:`2n^2 + 1` where *n* is the number of parameters.

    Args:
        program: Variational program whose Hessian is computed.
        center: Flat parameter vector at which to evaluate the Hessian.
            A typical choice is ``program.best_params``. When using
            ``program.viz.compute_hessian()``, this defaults to
            ``best_params`` if omitted.
        gradient_method: Strategy for computing second derivatives.
            :attr:`GradientMethod.PARAMETER_SHIFT` (default) uses the
            double parameter-shift rule (exact for standard quantum gates).
            :attr:`GradientMethod.FINITE_DIFFERENCE` uses centered finite
            differences with step size *eps*.
        eps: Finite-difference step size.  Only used when *gradient_method*
            is :attr:`GradientMethod.FINITE_DIFFERENCE`.  Defaults to ``1e-3``.
        reporter: Optional :class:`~divi.reporting.ProgressReporter`.

    Returns:
        HessianResult: Hessian matrix, eigenvalues (ascending), eigenvectors,
        and the center point.

    Raises:
        TypeError: If ``program`` is not a supported variational program.
        NotImplementedError: If ``program`` is an ``IterativeQAOA`` instance.
        ValueError: If ``center`` has the wrong shape or ``eps`` is non-positive.
    """
    _require_supported_program(program)
    rep = reporter if reporter is not None else _resolve_viz_reporter(program)
    gradient_method = GradientMethod(gradient_method)

    if gradient_method is GradientMethod.FINITE_DIFFERENCE and eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    n_params = _n_program_params(program)
    c = np.asarray(center, dtype=np.float64).reshape(-1)
    if c.shape != (n_params,):
        raise ValueError(f"center must have shape ({n_params},), got {c.shape}")

    # Shifts and coefficients depend on the gradient method.
    if gradient_method is GradientMethod.PARAMETER_SHIFT:
        diag_shift = np.pi
        cross_shift = 0.5 * np.pi
        diag_coeff = 0.25
        cross_coeff = 0.25
    else:
        diag_shift = eps
        cross_shift = eps
        diag_coeff = 1.0 / (eps * eps)
        cross_coeff = 1.0 / (4.0 * eps * eps)

    n = n_params
    eye_diag = diag_shift * np.eye(n, dtype=np.float64)
    eye_cross = cross_shift * np.eye(n, dtype=np.float64)

    # Layout: [center, +e0, -e0, ..., +ei+ej, +ei-ej, -ei+ej, -ei-ej, ...]
    diag_probes = np.empty((2 * n, n), dtype=np.float64)
    for i in range(n):
        diag_probes[2 * i] = c + eye_diag[i]
        diag_probes[2 * i + 1] = c - eye_diag[i]

    ii, jj = np.triu_indices(n, k=1)
    n_pairs = len(ii)
    cross_probes = np.empty((4 * n_pairs, n), dtype=np.float64)
    for k, (i, j) in enumerate(zip(ii, jj)):
        cross_probes[4 * k] = c + eye_cross[i] + eye_cross[j]
        cross_probes[4 * k + 1] = c + eye_cross[i] - eye_cross[j]
        cross_probes[4 * k + 2] = c - eye_cross[i] + eye_cross[j]
        cross_probes[4 * k + 3] = c - eye_cross[i] - eye_cross[j]

    param_sets = np.concatenate([[c], diag_probes, cross_probes])
    vals = _evaluate_param_sets_reported(
        program, param_sets, reporter=rep, scan_label="Hessian"
    )
    if not np.all(np.isfinite(vals)):
        raise ValueError(
            "Cost evaluations returned NaN or Inf; Hessian cannot be computed."
        )

    f0 = vals[0]
    H = np.zeros((n, n), dtype=np.float64)

    # Diagonal: H_ii = coeff * (f+ - 2*f0 + f-)
    diag_vals = vals[1 : 1 + 2 * n].reshape(n, 2)
    np.fill_diagonal(H, diag_coeff * (diag_vals[:, 0] - 2 * f0 + diag_vals[:, 1]))

    # Off-diagonal: H_ij = coeff * (f++ - f+- - f-+ + f--)
    if n_pairs > 0:
        cross_vals = vals[1 + 2 * n :].reshape(n_pairs, 4)
        h_off = cross_coeff * (
            cross_vals[:, 0] - cross_vals[:, 1] - cross_vals[:, 2] + cross_vals[:, 3]
        )
        H[ii, jj] = h_off
        H[jj, ii] = h_off

    eigenvalues, eigenvectors = np.linalg.eigh(H)

    return HessianResult(
        hessian=H,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        center=c,
        program_type=type(program).__name__,
    )


def run_neb(
    program: _SupportsVizScan,
    theta_1: npt.ArrayLike,
    theta_2: npt.ArrayLike,
    *,
    n_pivots: int = 12,
    n_steps: int = 50,
    learning_rate: float = 0.1,
    gradient_method: GradientMethod = GradientMethod.PARAMETER_SHIFT,
    eps: float = 1e-3,
    convergence_tol: float | None = None,
    reporter: ProgressReporter | None = None,
) -> NEBResult:
    """Find a minimum-energy path between two parameter vectors via NEB.

    Places *n_pivots* images (including the two fixed endpoints) along a
    straight line from *theta_1* to *theta_2* and iteratively relaxes the
    interior images.  At each step the gradient perpendicular to the chain
    tangent is computed and the images are updated via gradient descent, then
    redistributed uniformly along the path.

    .. warning::
       This function is **experimental**.  Convergence is sensitive to
       ``learning_rate``, ``eps``, and ``n_pivots``.

    Args:
        program: Variational program whose loss landscape is explored.
        theta_1: First endpoint (fixed).
        theta_2: Second endpoint (fixed).
        n_pivots: Total number of images including the two endpoints.
            Must be at least 3.
        n_steps: Maximum number of NEB relaxation iterations.
        learning_rate: Step size for gradient descent on interior images.
        gradient_method: Strategy for computing gradients.
            :attr:`GradientMethod.PARAMETER_SHIFT` (default) uses the
            parameter-shift rule.
            :attr:`GradientMethod.FINITE_DIFFERENCE` uses centered finite
            differences with step size *eps*.
        eps: Finite-difference step size.  Only used when *gradient_method*
            is :attr:`GradientMethod.FINITE_DIFFERENCE`.
        convergence_tol: If set, stop early when the maximum pivot
            displacement in a step falls below this threshold.
        reporter: Optional :class:`~divi.reporting.ProgressReporter`.

    Returns:
        NEBResult: Relaxed path, energies, cumulative distances, and the
        full iteration history.

    Raises:
        TypeError: If ``program`` is not a supported variational program.
        NotImplementedError: If ``program`` is an ``IterativeQAOA`` instance.
        ValueError: If shapes mismatch or ``n_pivots < 3``.
    """
    _require_supported_program(program)
    rep = reporter if reporter is not None else _resolve_viz_reporter(program)
    gradient_method = GradientMethod(gradient_method)

    if n_pivots < 3:
        raise ValueError(f"n_pivots must be >= 3, got {n_pivots}")

    n_params = _n_program_params(program)
    t1 = np.asarray(theta_1, dtype=np.float64).reshape(-1)
    t2 = np.asarray(theta_2, dtype=np.float64).reshape(-1)
    if t1.shape != (n_params,):
        raise ValueError(f"theta_1 must have shape ({n_params},), got {t1.shape}")
    if t2.shape != (n_params,):
        raise ValueError(f"theta_2 must have shape ({n_params},), got {t2.shape}")

    # Initialise: straight-line interpolation.
    chain = np.linspace(t1, t2, n_pivots).astype(np.float64)
    all_paths: list[npt.NDArray[np.float64]] = [chain.copy()]

    for step in range(n_steps):
        interior = chain[1:-1]  # (n_pivots - 2, n_params)
        grads = _compute_gradients(
            lambda ps: _evaluate_param_sets(program, ps),
            interior,
            gradient_method,
            eps,
        )
        perp_grads = _neb_perpendicular_gradients(chain, grads)

        rep.info(f"divi.viz NEB: step {step + 1}/{n_steps}")

        displacement = learning_rate * perp_grads
        chain[1:-1] = interior - displacement
        chain = _redistribute_uniform(chain, n_pivots)
        # Pin endpoints (redistribution may shift them by float rounding).
        chain[0] = t1
        chain[-1] = t2
        all_paths.append(chain.copy())

        if convergence_tol is not None:
            max_disp = float(np.max(np.linalg.norm(displacement, axis=1)))
            if max_disp < convergence_tol:
                rep.info(
                    f"divi.viz NEB: converged at step {step + 1} "
                    f"(max displacement {max_disp:.2e} < {convergence_tol:.2e})"
                )
                break

    # Final energy evaluation.
    energies = _evaluate_param_sets_reported(
        program, chain, reporter=rep, scan_label="NEB final"
    )
    distances = _cumulative_distances(chain)

    return NEBResult(
        path=chain,
        energies=energies,
        path_distances=distances,
        all_paths=all_paths,
        program_type=type(program).__name__,
    )


def fourier_analysis_2d(
    scan_result: Scan2DResult | PCAScanResult,
) -> Fourier2DResult:
    """Compute the 2D Fourier power spectrum of a scan grid.

    Applies :func:`numpy.fft.fft2` to the ``values`` grid and returns the
    shifted power spectrum.

    .. note::
       Assumes uniform grid spacing (as produced by :func:`scan_2d`,
       :func:`scan_pca`, and all interpolation scans which use
       :func:`numpy.linspace`).  Non-uniform grids will produce incorrect
       frequency axes.

    Args:
        scan_result: A :class:`~divi.viz.Scan2DResult` or
            :class:`~divi.viz.PCAScanResult` whose ``values`` grid will be
            transformed.

    Returns:
        Fourier2DResult: Shifted frequencies and power spectrum.
    """
    values = np.asarray(scan_result.values, dtype=np.float64)
    n_y, n_x = values.shape

    dx = float(scan_result.x_offsets[1] - scan_result.x_offsets[0]) if n_x > 1 else 1.0
    dy = float(scan_result.y_offsets[1] - scan_result.y_offsets[0]) if n_y > 1 else 1.0

    fft = np.fft.fft2(values)
    fft_shifted = np.fft.fftshift(fft)
    power = np.abs(fft_shifted) ** 2

    freq_x = np.fft.fftshift(np.fft.fftfreq(n_x, d=dx))
    freq_y = np.fft.fftshift(np.fft.fftfreq(n_y, d=dy))

    return Fourier2DResult(
        frequencies_x=freq_x.astype(np.float64),
        frequencies_y=freq_y.astype(np.float64),
        power_spectrum=power.astype(np.float64),
        program_type=scan_result.program_type,
    )


@dataclass(slots=True)
class ProgramViz:
    """Thin convenience wrapper for ``program.viz`` access.

    This wrapper mirrors the standalone :mod:`divi.viz` scan API so users can
    write fluent calls such as ``program.viz.scan_1d(...)`` without changing the
    underlying execution behavior.
    """

    program: _SupportsVizScan

    def scan_1d(self, **kwargs) -> Scan1DResult:
        """Call :func:`divi.viz.scan_1d` for the wrapped program."""
        return scan_1d(self.program, **kwargs)

    def scan_2d(self, **kwargs) -> Scan2DResult:
        """Call :func:`divi.viz.scan_2d` for the wrapped program."""
        return scan_2d(self.program, **kwargs)

    def scan_pca(self, *, wrap_periodic: bool = False, **kwargs) -> PCAScanResult:
        """Call :func:`divi.viz.scan_pca` for the wrapped program.

        When ``samples`` is omitted, automatically collects parameter vectors
        from ``program.param_history(mode="best_per_iteration")``.  Set
        ``wrap_periodic=True`` to apply
        :func:`~divi.viz.periodic_trajectory_wrap` to the auto-collected
        samples before fitting PCA.
        """
        if "samples" not in kwargs:
            from ._periodic import periodic_trajectory_wrap

            blocks = self.program.param_history(mode="best_per_iteration")
            if not blocks:
                raise ValueError(
                    "No param_history available — pass samples explicitly or "
                    "run optimization first."
                )
            samples = np.vstack(blocks)
            if wrap_periodic:
                samples = periodic_trajectory_wrap(samples)
            kwargs["samples"] = samples
        return scan_pca(self.program, **kwargs)

    def scan_interp_1d(self, theta_1, theta_2, **kwargs) -> Scan1DResult:
        """Call :func:`divi.viz.scan_interp_1d` for the wrapped program."""
        return scan_interp_1d(self.program, theta_1, theta_2, **kwargs)

    def scan_interp_2d(self, theta_1, theta_2, **kwargs) -> Scan2DResult:
        """Call :func:`divi.viz.scan_interp_2d` for the wrapped program."""
        return scan_interp_2d(self.program, theta_1, theta_2, **kwargs)

    def compute_hessian(self, center=None, **kwargs) -> HessianResult:
        """Call :func:`divi.viz.compute_hessian` for the wrapped program.

        When ``center`` is omitted, defaults to ``program.best_params``.
        """
        if center is None:
            center = _resolve_center(self.program, None)
        return compute_hessian(self.program, center, **kwargs)

    def run_neb(self, theta_1, theta_2, **kwargs) -> NEBResult:
        """Call :func:`divi.viz.run_neb` for the wrapped program."""
        return run_neb(self.program, theta_1, theta_2, **kwargs)
