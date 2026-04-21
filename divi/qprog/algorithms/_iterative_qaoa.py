# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Iterative QAOA with parameter interpolation across increasing circuit depths.

This module implements the iterative interpolation strategy for QAOA described in
`arXiv:2504.01694 <https://arxiv.org/abs/2504.01694>`_. Instead of optimizing at a
fixed depth with random initialization, the algorithm starts at depth p=1, optimizes,
then interpolates the optimal parameters to warm-start at depth p+1, repeating until
a target depth or convergence criterion is met.

Three interpolation strategies are provided:

- **INTERP**: Linear interpolation (Zhou et al.)
- **FOURIER**: Fourier basis representation
- **CHEBYSHEV**: Chebyshev polynomial basis representation
"""

from collections.abc import Callable
from enum import Enum

import numpy as np
import numpy.typing as npt
from qiskit.circuit import ParameterVector

from ._qaoa import QAOA

# ---------------------------------------------------------------------------
# Interpolation strategies
# ---------------------------------------------------------------------------


class InterpolationStrategy(Enum):
    """Strategy for interpolating QAOA parameters from depth p to p+1."""

    INTERP = "interp"
    """Linear interpolation (Zhou et al.)."""

    FOURIER = "fourier"
    """Fourier basis representation."""

    CHEBYSHEV = "chebyshev"
    """Chebyshev polynomial basis representation."""


def _interp(u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Linear interpolation from depth p to p+1 (Zhou et al.).

    Given a sequence u of length p, produce a sequence of length p+1 via:

        u'[j] = (j/p) * u[j-1] + (p - j)/p * u[j]

    with boundary conditions u[-1] = 0 and u[p] = 0.
    """
    p = len(u)
    result = np.empty(p + 1, dtype=np.float64)
    for j in range(p + 1):
        left = u[j - 1] if j > 0 else 0.0
        right = u[j] if j < p else 0.0
        result[j] = (j / p) * left + (p - j) / p * right
    return result


def _fourier(
    u: npt.NDArray[np.float64], n_basis_terms: int | None = None
) -> npt.NDArray[np.float64]:
    """Fourier (DCT-II) basis interpolation from depth p to p+1.

    Represents the p angles as k cosine coefficients using the DCT-II basis,
    then evaluates at p+1 grid points:

        u_j = sum_{l=0}^{k-1} a_l * cos(pi * l * (2j + 1) / (2p))

    The DCT-II basis is orthogonal and well-conditioned for all p >= k.

    Args:
        u: Parameter sequence of length p.
        n_basis_terms: Number of basis terms. Defaults to min(p, 5).
    """
    p = len(u)
    k = min(p, n_basis_terms) if n_basis_terms is not None else min(p, 5)

    # Build the DCT-II basis matrix at depth p: shape (p, k)
    j_grid = np.arange(p, dtype=np.float64)
    l_terms = np.arange(k, dtype=np.float64)
    basis_p = np.cos(np.outer(np.pi * (2 * j_grid + 1) / (2 * p), l_terms))

    # Fit coefficients via least squares
    coeffs, *_ = np.linalg.lstsq(basis_p, u, rcond=None)

    # Evaluate at p+1 grid points
    p_new = p + 1
    j_grid_new = np.arange(p_new, dtype=np.float64)
    basis_new = np.cos(np.outer(np.pi * (2 * j_grid_new + 1) / (2 * p_new), l_terms))

    return basis_new @ coeffs


def _chebyshev(
    u: npt.NDArray[np.float64], n_basis_terms: int | None = None
) -> npt.NDArray[np.float64]:
    """Chebyshev polynomial basis interpolation from depth p to p+1.

    Represents the p angles via k Chebyshev coefficients at Chebyshev nodes,
    then evaluates at p+1 nodes:

        u_j = sum_{l=0}^{k-1} c_l * T_l(x_j)
        x_j = cos(pi * (j + 0.5) / p)

    Args:
        u: Parameter sequence of length p.
        n_basis_terms: Number of Chebyshev terms. Defaults to min(p, 5).
    """
    p = len(u)
    k = min(p, n_basis_terms) if n_basis_terms is not None else min(p, 5)

    # Chebyshev nodes at depth p
    j_grid = np.arange(p, dtype=np.float64)
    x_p = np.cos(np.pi * (j_grid + 0.5) / p)

    # Build Chebyshev basis matrix at depth p: shape (p, k)
    basis_p = np.empty((p, k), dtype=np.float64)
    for l in range(k):
        basis_p[:, l] = np.cos(l * np.arccos(x_p))

    # Fit coefficients via least squares
    coeffs, *_ = np.linalg.lstsq(basis_p, u, rcond=None)

    # Chebyshev nodes at depth p+1
    p_new = p + 1
    j_grid_new = np.arange(p_new, dtype=np.float64)
    x_new = np.cos(np.pi * (j_grid_new + 0.5) / p_new)

    basis_new = np.empty((p_new, k), dtype=np.float64)
    for l in range(k):
        basis_new[:, l] = np.cos(l * np.arccos(x_new))

    return basis_new @ coeffs


def interpolate_qaoa_params(
    params: npt.NDArray[np.float64],
    current_depth: int,
    strategy: InterpolationStrategy,
    n_basis_terms: int | None = None,
) -> npt.NDArray[np.float64]:
    """Interpolate QAOA parameters from depth p to depth p+1.

    Deinterleaves the flat parameter array into beta and gamma sequences,
    applies the chosen interpolation strategy independently to each, then
    reinterleaves into the flat layout expected by QAOA.

    Args:
        params: Flat 1D parameter array of length ``2 * current_depth``
            with layout ``[beta_0, gamma_0, beta_1, gamma_1, ...]``.
        current_depth: Current circuit depth p.
        strategy: Interpolation strategy to use.
        n_basis_terms: Number of basis terms for FOURIER/CHEBYSHEV strategies.
            Ignored for INTERP. Defaults to ``min(p, 5)`` when ``None``.

    Returns:
        Flat 1D parameter array of length ``2 * (current_depth + 1)``.
    """
    betas = params[0::2]
    gammas = params[1::2]

    interp_fn: Callable[..., npt.NDArray[np.float64]]
    if strategy == InterpolationStrategy.INTERP:
        new_betas = _interp(betas)
        new_gammas = _interp(gammas)
    elif strategy == InterpolationStrategy.FOURIER:
        new_betas = _fourier(betas, n_basis_terms)
        new_gammas = _fourier(gammas, n_basis_terms)
    elif strategy == InterpolationStrategy.CHEBYSHEV:
        new_betas = _chebyshev(betas, n_basis_terms)
        new_gammas = _chebyshev(gammas, n_basis_terms)
    else:
        raise ValueError(f"Unknown interpolation strategy: {strategy}")

    new_params = np.empty(2 * (current_depth + 1), dtype=np.float64)
    new_params[0::2] = new_betas
    new_params[1::2] = new_gammas
    return new_params


# ---------------------------------------------------------------------------
# IterativeQAOA
# ---------------------------------------------------------------------------


class IterativeQAOA(QAOA):
    """Iterative QAOA with parameter interpolation across increasing depths.

    Instead of optimizing at a single fixed depth, this class iteratively
    increases the circuit depth from 1 to ``max_depth``, using the optimal
    parameters from depth p as a warm-start for depth p+1 via an
    interpolation strategy.

    After :meth:`run` completes, the instance represents the depth that
    achieved the best loss. All standard QAOA properties (``solution``,
    ``best_params``, ``best_loss``, ``get_top_solutions``) work as usual.

    Example::

        iterative = IterativeQAOA(
            problem=MaxCutProblem(graph),
            max_depth=5,
            strategy=InterpolationStrategy.INTERP,
            max_iterations_per_depth=20,
            backend=backend,
            optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        )
        iterative.run()
        print(iterative.best_depth)
        print(iterative.solution)

    Args:
        problem: A :class:`~divi.qprog.problems.QAOAProblem` instance providing the QAOA ingredients.
        max_depth: Maximum circuit depth to iterate up to. Defaults to 5.
        strategy: Interpolation strategy for warm-starting. Defaults to INTERP.
        n_basis_terms: Number of basis terms for FOURIER/CHEBYSHEV strategies.
            Ignored for INTERP. Defaults to ``min(p, 5)`` when ``None``.
        max_iterations_per_depth: Maximum optimization iterations per depth.
            Can be an integer (same for all depths) or a callable
            ``(depth) -> int`` for adaptive budgets. Defaults to 10.
        convergence_threshold: If set, stop iterating when the absolute
            improvement in loss between consecutive depths is below this value.
        **kwargs: All remaining QAOA keyword arguments (``backend``,
            ``optimizer``, ``initial_state``, etc.).
    """

    _supports_fixed_param_scans = False

    def __init__(
        self,
        problem,
        *,
        max_depth: int = 5,
        strategy: InterpolationStrategy = InterpolationStrategy.INTERP,
        n_basis_terms: int | None = None,
        max_iterations_per_depth: int | Callable[[int], int] = 10,
        convergence_threshold: float | None = None,
        **kwargs,
    ):
        self._max_depth = max_depth
        self._strategy = strategy
        self._n_basis_terms = n_basis_terms
        self._max_iterations_per_depth = max_iterations_per_depth
        self._convergence_threshold = convergence_threshold

        self._depth_history: list[dict] = []
        self._best_depth: int = 1

        super().__init__(
            problem,
            n_layers=1,
            max_iterations=self._get_max_iters(1),
            **kwargs,
        )

    @property
    def _expected_total_iterations(self) -> int:
        """Total expected iterations across all depths (for progress display)."""
        return sum(self._get_max_iters(d) for d in range(1, self._max_depth + 1))

    def _get_max_iters(self, depth: int) -> int:
        if callable(self._max_iterations_per_depth):
            return self._max_iterations_per_depth(depth)
        return self._max_iterations_per_depth

    def _rebuild_for_depth(self, depth: int) -> None:
        """Rebuild parameters and pipelines for a new circuit depth."""
        self.n_layers = depth
        self._n_params_per_layer = 2
        betas = ParameterVector("β", depth)
        gammas = ParameterVector("γ", depth)
        # Replaces the parent QAOA._params so _build_qaoa_ops and the
        # meta-circuit factories pick up the new layer count.
        self._params = np.array([[b, g] for b, g in zip(betas, gammas)], dtype=object)
        self._build_pipelines()

    def _reset_optimization_state(self) -> None:
        """Reset VQA optimization tracking state for a fresh run."""
        self._losses_history = []
        self._param_history = []
        self._best_params = []
        self._best_loss = float("inf")
        self._best_probs = {}
        self.current_iteration = 0
        self.optimize_result = None
        self._stop_reason = None
        self.optimizer.reset()

    def run(self, perform_final_computation=True, **kwargs):
        """Run the iterative QAOA procedure across increasing depths.

        At each depth from 1 to ``max_depth``, the algorithm optimizes the
        QAOA parameters, then interpolates the best parameters to warm-start
        the next depth. After all depths are explored, the instance is
        restored to the depth that achieved the best overall loss.

        Args:
            perform_final_computation: Whether to run the final measurement
                at the best depth to extract the solution. Defaults to True.
            **kwargs: Additional keyword arguments passed to the parent ``run()``.

        Returns:
            IterativeQAOA: Returns ``self`` for method chaining.
        """
        depth_history: list[dict] = []
        prev_best_params: npt.NDArray[np.float64] | None = None
        total_circuits = 0
        total_time = 0.0

        for depth in range(1, self._max_depth + 1):
            self.reporter.info(message=f"Depth {depth}/{self._max_depth}")
            self._rebuild_for_depth(depth)
            self._reset_optimization_state()
            self.max_iterations = self._get_max_iters(depth)
            initial_params = None

            if depth > 1 and prev_best_params is not None:
                interpolated = interpolate_qaoa_params(
                    prev_best_params,
                    depth - 1,
                    self._strategy,
                    self._n_basis_terms,
                )
                initial_params = np.tile(interpolated, (self.optimizer.n_param_sets, 1))

            super().run(
                initial_params=initial_params,
                perform_final_computation=False,
                **kwargs,
            )
            total_circuits += self._total_circuit_count
            total_time += self._total_run_time

            depth_history.append(
                {
                    "depth": depth,
                    "best_loss": self._best_loss,
                    "best_params": self._best_params.copy(),
                    "n_iterations": self.current_iteration,
                }
            )
            prev_best_params = self._best_params.copy()

            if (
                self._convergence_threshold is not None
                and depth > 1
                and abs(depth_history[-2]["best_loss"] - self._best_loss)
                < self._convergence_threshold
            ):
                break

        # Store history and find best depth
        self._depth_history = depth_history
        best_entry = min(depth_history, key=lambda d: d["best_loss"])
        self._best_depth = best_entry["depth"]

        # Restore the instance to the best depth
        self._rebuild_for_depth(self._best_depth)
        self._best_params = best_entry["best_params"]
        self._best_loss = best_entry["best_loss"]

        if perform_final_computation:
            self._perform_final_computation(**kwargs)

        self._total_circuit_count = total_circuits
        self._total_run_time = total_time
        return self

    @property
    def best_depth(self) -> int:
        """The circuit depth that achieved the best (lowest) loss."""
        return self._best_depth

    @property
    def depth_history(self) -> list[dict]:
        """Per-depth optimization results.

        Each entry is a dict with keys:
        ``depth``, ``best_loss``, ``best_params``, ``n_iterations``.
        """
        return self._depth_history.copy()
