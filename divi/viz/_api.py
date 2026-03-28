# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0
#
# Attribution: PCA scan geometry follows patterns described in orqviz (Zapata
# Engineering, Apache-2.0). See LICENSES/ORQViz-Apache-2.0-acknowledgement.txt.

"""Loss-landscape scan API for variational programs.

Line, plane, and PCA scans follow the conventions of ``orqviz`` (Zapata
Engineering, Apache-2.0) where applicable, but all objective evaluations run
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

from ._results import PCAScanResult, Scan1DResult, Scan2DResult

# orqviz-style 1D/2D scans use scalar endpoints in (-pi, pi) along the direction vector.
_DEFAULT_SCAN_SPAN: tuple[float, float] = (-np.pi, np.pi)

VizRng: TypeAlias = np.random.Generator | int | None


class _SupportsVizScan(Protocol):
    n_layers: int
    n_params_per_layer: int
    _best_params: npt.NDArray[np.float64] | list[float]

    def _has_run_optimization(self) -> bool: ...

    def _evaluate_cost_param_sets(
        self, param_sets: npt.NDArray[np.float64], **kwargs
    ) -> dict[int, float]: ...


def _resolve_viz_reporter(program: _SupportsVizScan) -> ProgressReporter:
    r = getattr(program, "reporter", None)
    return r if isinstance(r, ProgressReporter) else LoggingProgressReporter()


def _evaluate_param_sets_reported(
    program: _SupportsVizScan,
    param_sets: npt.NDArray[np.float64],
    *,
    reporter: ProgressReporter,
    scan_label: str,
) -> npt.NDArray[np.float64]:
    """Single batched evaluation with Rich/status start/finish messaging (no chunking)."""
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


def _require_supported_program(
    program: _SupportsVizScan,
) -> _SupportsVizScan:
    required_attrs = ("n_layers", "n_params_per_layer", "_best_params")
    required_methods = ("_has_run_optimization", "_evaluate_cost_param_sets")
    if not all(hasattr(program, attr) for attr in required_attrs) or not all(
        callable(getattr(program, name, None)) for name in required_methods
    ):
        raise TypeError("divi.viz currently supports VariationalQuantumAlgorithm only.")

    if type(program).__name__ == "IterativeQAOA":
        raise NotImplementedError(
            "IterativeQAOA varies circuit depth during optimization and has no "
            "fixed parameter space for these scans; use a fixed-depth QAOA instance."
        )

    return program


def _resolve_center(
    program: _SupportsVizScan,
    center: npt.ArrayLike | None,
) -> npt.NDArray[np.float64]:
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
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, int):
        return np.random.default_rng(rng)
    return rng


def _random_nonzero_gaussian(
    n_params: int, rng: np.random.Generator
) -> npt.NDArray[np.float64]:
    for _ in range(64):
        v = rng.standard_normal(n_params)
        if float(np.linalg.norm(v)) > 1e-15:
            return v.astype(np.float64, copy=False)
    raise RuntimeError(
        "Could not sample a non-zero direction; try a different rng seed."
    )


def _resolve_1d_direction(
    direction: npt.ArrayLike | None,
    *,
    n_params: int,
    rng: np.random.Generator,
    normalize_directions: bool,
) -> npt.NDArray[np.float64]:
    if direction is None:
        direction_arr = _random_nonzero_gaussian(n_params, rng)
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
    direction_x: npt.ArrayLike | None,
    direction_y: npt.ArrayLike | None,
    *,
    n_params: int,
    rng: np.random.Generator,
    normalize_directions: bool,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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
        dy_raw = _random_nonzero_gaussian(n_params, rng)
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
        dx_raw = _random_nonzero_gaussian(n_params, rng)
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
        dx = _random_nonzero_gaussian(n_params, rng)
        nx = float(np.linalg.norm(dx))
        dy_raw = _random_nonzero_gaussian(n_params, rng)
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
    losses = program._evaluate_cost_param_sets(param_sets)
    return np.asarray([value for _, value in sorted(losses.items())], dtype=np.float64)


def _resolve_samples(
    samples: npt.ArrayLike,
    *,
    n_params: int,
) -> npt.NDArray[np.float64]:
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
    rows: list[npt.NDArray[np.float64]] = []
    for y in y_offsets:
        for x in x_offsets:
            coef = np.zeros((1, n_comp), dtype=np.float64)
            coef[0, i0] = x
            coef[0, i1] = y
            theta = pca.inverse_transform(coef).reshape(-1) + shift
            rows.append(theta)
    return np.asarray(rows, dtype=np.float64)


def scan_1d(
    program: _SupportsVizScan,
    *,
    center: npt.ArrayLike | None = None,
    direction: npt.ArrayLike | None = None,
    n_points: int = 51,
    span: tuple[float, float] = _DEFAULT_SCAN_SPAN,
    normalize_directions: bool = True,
    rng: VizRng = None,
    reporter: ProgressReporter | None = None,
) -> Scan1DResult:
    """Evaluate a one-dimensional objective scan for a variational program.

    The scan is constructed by taking a center point in parameter space and a
    direction vector, then evaluating the program objective at evenly spaced
    offsets along that line. Internally, the scan uses
    :meth:`~divi.qprog.variational_quantum_algorithm.VariationalQuantumAlgorithm._evaluate_cost_param_sets`
    directly, so it reuses the program's normal batched objective-evaluation
    path instead of going through ``run()``.

    Geometry matches the usual ``orqviz`` 1D pattern when ``span`` uses the
    default :math:`(-\\pi, \\pi)` and ``normalize_directions`` is ``False``: each
    offset :math:`t` maps to ``center + t * direction`` with the direction norm
    affecting Euclidean step size. The default ``normalize_directions=True``
    keeps a unit direction so ``span`` is measured along a normalized axis.

    Args:
        program: Variational program whose objective should be scanned.
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
            :class:`~divi.qprog.variational_quantum_algorithm.VariationalQuantumAlgorithm.run`).
    Returns:
        Scan1DResult: Object containing offsets, sampled objective values, the
        concrete parameter sets that were evaluated, and plotting helpers.

    Raises:
        TypeError: If ``program`` is not a supported
            :class:`~divi.qprog.variational_quantum_algorithm.VariationalQuantumAlgorithm`.
        NotImplementedError: If ``program`` is an ``IterativeQAOA`` instance.
        ValueError: If ``center`` or ``direction`` has the wrong shape, if
            ``direction`` is zero, or if ``n_points`` is invalid.
    """
    program = _require_supported_program(program)

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
    param_sets = center_arr + offsets[:, None] * direction_arr[None, :]
    rep = reporter if reporter is not None else _resolve_viz_reporter(program)
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
    center: npt.ArrayLike | None = None,
    direction_x: npt.ArrayLike | None = None,
    direction_y: npt.ArrayLike | None = None,
    grid_shape: tuple[int, int] = (41, 41),
    span_x: tuple[float, float] = _DEFAULT_SCAN_SPAN,
    span_y: tuple[float, float] = _DEFAULT_SCAN_SPAN,
    normalize_directions: bool = True,
    rng: VizRng = None,
    reporter: ProgressReporter | None = None,
) -> Scan2DResult:
    """Evaluate a two-dimensional objective scan for a variational program.

    The scan is constructed from a center point plus two linearly independent
    directions in parameter space. A rectangular grid of offsets is generated,
    converted into concrete parameter sets, and evaluated using the program's
    standard batched objective path.

    When either direction is omitted, the missing axis is filled with a random
    vector orthogonal to the other (same norm as the reference axis when
    ``normalize_directions`` is ``False``, matching the usual ``orqviz`` 2D scan
    construction). Use ``rng`` for reproducibility.

    Args:
        program: Variational program whose objective should be scanned.
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
        Scan2DResult: Object containing the sampled grid, objective values,
        evaluated parameter sets, and plotting helpers.

    Raises:
        TypeError: If ``program`` is not a supported variational program.
        NotImplementedError: If ``program`` is an ``IterativeQAOA`` instance.
        ValueError: If the directions have the wrong shape, are zero, are not
            linearly independent, or if ``grid_shape`` is invalid.
    """
    program = _require_supported_program(program)

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

    param_sets = np.asarray(
        [
            center_arr + x * direction_x_arr + y * direction_y_arr
            for y in y_offsets
            for x in x_offsets
        ],
        dtype=np.float64,
    )
    rep = reporter if reporter is not None else _resolve_viz_reporter(program)
    flat_values = _evaluate_param_sets_reported(
        program,
        param_sets,
        reporter=rep,
        scan_label="2D scan",
    )
    values = flat_values.reshape(n_y, n_x)

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
    center: npt.ArrayLike | None = None,
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
    evaluates the program objective in one batch.

    When ``span_x`` and ``span_y`` are both omitted, axis limits match orqviz
    ``PCAobject._get_endpoints_from_pca``: min/max of the projected ``samples``
    on each selected component, plus ``offset`` applied to the low and high ends.

    Args:
        program: Variational program whose objective should be scanned.
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
        PCAScanResult: Grid in PCA score coordinates, objective values, full
        parameter sets, sklearn PC directions for the chosen indices, explained
        variance ratios, and projected ``samples`` scores for plotting.

    Raises:
        TypeError: If ``program`` is not a supported variational program.
        NotImplementedError: If ``program`` is an ``IterativeQAOA`` instance.
        ValueError: If shapes, ranks, ``grid_shape``, or span arguments are invalid.
    """
    program = _require_supported_program(program)

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

    if center is None:
        anchor = mean.astype(np.float64, copy=True)
    else:
        anchor = np.asarray(center, dtype=np.float64).reshape(-1)
        if anchor.shape != (n_params,):
            raise ValueError(
                f"center must have shape ({n_params},), got {anchor.shape}"
            )
    shift = anchor - mean

    if span_x is not None or span_y is not None:
        if span_x is None or span_y is None:
            raise ValueError("span_x and span_y must both be set or both omitted.")
        end_x = (float(span_x[0]), float(span_x[1]))
        end_y = (float(span_y[0]), float(span_y[1]))
    else:
        end_x, end_y = _pca_scan_endpoints(
            scores, i0, i1, _normalize_pca_offset(offset)
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

    rep = reporter if reporter is not None else _resolve_viz_reporter(program)
    flat_values = _evaluate_param_sets_reported(
        program,
        param_sets,
        reporter=rep,
        scan_label="PCA scan",
    )
    values = flat_values.reshape(n_y, n_x)

    pc_x = np.asarray(pca.components_[i0], dtype=np.float64)
    pc_y = np.asarray(pca.components_[i1], dtype=np.float64)
    explained_ratio = np.asarray(
        pca.explained_variance_ratio_[[i0, i1]], dtype=np.float64
    )
    projected_samples = np.column_stack((scores[:, i0], scores[:, i1]))

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

    def scan_pca(self, **kwargs) -> PCAScanResult:
        """Call :func:`divi.viz.scan_pca` for the wrapped program."""
        return scan_pca(self.program, **kwargs)
