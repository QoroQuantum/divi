# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Iterative QAOA with parameter interpolation across increasing circuit depths.

This module implements the iterative interpolation strategy for QAOA described in
`arXiv:2504.01694 <https://arxiv.org/abs/2504.01694>`_. Instead of optimising at a
fixed depth with random initialisation, the algorithm starts at depth p=1, optimises,
then interpolates the optimal parameters to warm-start at depth p+1, repeating until
a target depth or convergence criterion is met.

Three interpolation strategies are provided:

- **INTERP**: Linear interpolation (Zhou et al.)
- **FOURIER**: Fourier basis representation
- **CHEBYSHEV**: Chebyshev polynomial basis representation
"""

from collections.abc import Callable
from dataclasses import replace
from enum import Enum
from pathlib import Path
from typing import Any
from warnings import warn

import numpy as np
import numpy.typing as npt
from qiskit.circuit import ParameterVector

from divi.qprog.checkpointing import (
    PROGRAM_COMPLETION_FILE,
    CheckpointConfig,
    CheckpointNotFoundError,
    _find_latest_checkpoint_subdir,
    _write_program_completion,
)
from divi.qprog.quantum_program import reject_unclaimed_run_kwargs
from divi.reporting._events import ProgressEvent, TerminalStatus

from ._qaoa import QAOA

DEPTH_SUBDIR_PREFIX = "depth_"


def _extract_depth_from_subdir(path: Path) -> int | None:
    """Depth encoded in a ``depth_NN`` subdirectory name, or None if not one."""
    if not path.is_dir() or not path.name.startswith(DEPTH_SUBDIR_PREFIX):
        return None
    suffix = path.name[len(DEPTH_SUBDIR_PREFIX) :]
    return int(suffix) if suffix.isdigit() else None


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

    return (basis_new @ coeffs).astype(np.float64, copy=False)


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

    Instead of optimising at a single fixed depth, this class iteratively
    increases the circuit depth from 1 to ``max_depth``, using the optimal
    parameters from depth p as a warm-start for depth p+1 via an
    interpolation strategy.

    After :meth:`run` completes, the instance represents the depth that
    achieved the best loss. All standard QAOA properties (``solution``,
    ``best_params``, ``best_loss``, ``get_top_solutions``) work as usual.

    Checkpoints are written per depth, so the ``subdirectory`` passed to
    ``load_state()`` names the depth as well, e.g.
    ``"depth_02/checkpoint_003"``.

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
        max_iterations_per_depth: Maximum optimisation iterations per depth.
            Can be an integer (same for all depths) or a callable
            ``(depth) -> int`` for adaptive budgets. Defaults to 10.
        convergence_threshold: If set, stop iterating when a depth improves
            the loss over the previous depth by less than this value. A depth
            that makes the loss worse does not stop the run.
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
        for fixed, replacement in (
            ("n_layers", "max_depth"),
            ("max_iterations", "max_iterations_per_depth"),
        ):
            if fixed in kwargs:
                raise TypeError(
                    f"IterativeQAOA sets {fixed} for each depth; "
                    f"pass {replacement} instead."
                )
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

    @property
    def _completed_iterations(self) -> int:
        """Iterations already run across all depths (for progress display)."""
        done = sum(entry["n_iterations"] for entry in self._depth_history)
        # A depth checkpointed mid-way has run part of its budget.
        if self.n_layers == len(self._depth_history) + 1:
            done += self.current_iteration
        return done

    def _get_max_iters(self, depth: int) -> int:
        if callable(self._max_iterations_per_depth):
            return self._max_iterations_per_depth(depth)
        return self._max_iterations_per_depth

    def _rebuild_for_depth(self, depth: int) -> None:
        """Rebuild parameters for a new circuit depth and drop stale pipelines."""
        self.n_layers = depth
        betas = ParameterVector("β", depth)
        gammas = ParameterVector("γ", depth)
        # Replaces the parent QAOA._params so the meta-circuit factories
        # pick up the new layer count.
        self._params = np.array([[b, g] for b, g in zip(betas, gammas)], dtype=object)
        # Drop the memoized protocol pipelines (and their forward caches): the
        # ansatz changed shape, so the cost/sample circuits must be rebuilt.
        self._preprocessor_pipeline_cache.clear()
        self._cost_circuit = None

    def _save_subclass_state(self) -> dict[str, Any]:
        """Save QAOA state plus the depths run so far."""
        state = super()._save_subclass_state()
        state.update(
            {
                "depth": self.n_layers,
                "best_depth": self._best_depth,
                "depth_history": [
                    {**entry, "best_params": np.asarray(entry["best_params"]).tolist()}
                    for entry in self._depth_history
                ],
            }
        )
        return state

    def _load_subclass_state(self, state: dict[str, Any]) -> None:
        """Restore the depths run so far and rebuild the ansatz at the saved depth."""
        super()._load_subclass_state(state)

        required_keys = ("depth", "best_depth", "depth_history")
        missing_keys = [key for key in required_keys if key not in state]
        if missing_keys:
            raise KeyError(
                f"Corrupted checkpoint: missing required state keys: {missing_keys}"
            )

        self._best_depth = state["best_depth"]
        self._depth_history = [
            {
                **entry,
                "best_params": np.asarray(entry["best_params"], dtype=np.float64),
            }
            for entry in state["depth_history"]
        ]
        self._rebuild_for_depth(state["depth"])

    @classmethod
    def _resolve_checkpoint_path(
        cls,
        checkpoint_dir: Path | str,
        subdirectory: str | None = None,
    ) -> Path:
        """Resolve the deepest checkpointed depth before its iteration.

        ``run()`` writes each depth under its own ``depth_NN`` subdirectory, so
        a directory holding those is resolved to the deepest one carrying a
        complete checkpoint before the usual per-iteration resolution runs.
        Loading a finished run then applies its top-level completion file,
        which restores the best depth.
        """
        main_dir = Path(checkpoint_dir)
        if subdirectory is None and main_dir.is_dir():
            depth_dirs = sorted(
                (
                    d
                    for d in main_dir.iterdir()
                    if _extract_depth_from_subdir(d) is not None
                ),
                key=lambda d: _extract_depth_from_subdir(d) or -1,
                reverse=True,
            )
            for depth_dir in depth_dirs:
                try:
                    _find_latest_checkpoint_subdir(depth_dir)
                except CheckpointNotFoundError:
                    continue
                main_dir = depth_dir
                break

        return super()._resolve_checkpoint_path(main_dir, subdirectory)

    @staticmethod
    def _depth_checkpoint_config(
        checkpoint_config: CheckpointConfig, depth: int
    ) -> CheckpointConfig:
        """Point ``checkpoint_config`` at this depth's own subdirectory."""
        if checkpoint_config.checkpoint_dir is None:
            return checkpoint_config
        return replace(
            checkpoint_config,
            checkpoint_dir=Path(checkpoint_config.checkpoint_dir)
            / f"{DEPTH_SUBDIR_PREFIX}{depth:02d}",
        )

    def run(
        self,
        initial_params=None,
        perform_final_computation=True,
        checkpoint_config=None,
        **kwargs,
    ):
        """Run the iterative QAOA procedure across increasing depths.

        At each depth from 1 to ``max_depth``, the algorithm optimises the
        QAOA parameters, then interpolates the best parameters to warm-start
        the next depth. After all depths are explored, the instance is
        restored to the depth that achieved the best overall loss. A program
        that has run all its depths, or converged, warns and returns without
        training or finalising; build a new program to run them again.

        Args:
            initial_params: Ignored — each depth computes its own warm-start
                via interpolation of the previous depth's best parameters.
                Passing a non-None value emits a ``UserWarning``.
            perform_final_computation: Whether to sample the best depth's
                parameters once the last depth finishes. Defaults to True.
            checkpoint_config: Each depth is checkpointed under its own
                ``depth_NN`` subdirectory of ``checkpoint_dir``, so depths do
                not overwrite one another and
                :meth:`~divi.qprog.variational_quantum_algorithm.VariationalQuantumAlgorithm.load_state`
                can resume at the depth where the run stopped.
            **kwargs: ``max_iterations`` overrides every depth's iteration
                budget (the ``max_iterations`` attribute is reset per depth);
                any other keyword raises ``TypeError``.

        Returns:
            IterativeQAOA: Returns ``self`` for method chaining.
        """
        with self._ensure_progress_session(
            label="Iterative QAOA",
            total=self._expected_total_iterations,
        ):
            result = self._run_depths(
                initial_params=initial_params,
                perform_final_computation=perform_final_computation,
                checkpoint_config=checkpoint_config,
                **kwargs,
            )
            self._progress_emitter(
                ProgressEvent.finish(
                    self._progress_key,
                    TerminalStatus.SUCCESS,
                    detail="Finished successfully!",
                )
            )
            return result

    def _run_depths(
        self,
        initial_params=None,
        perform_final_computation=True,
        checkpoint_config=None,
        **kwargs,
    ):
        """The body of :meth:`run`, inside its progress session."""
        if initial_params is not None:
            warn(
                "IterativeQAOA ignores `initial_params` — each depth computes its "
                "own warm-start via interpolation of the previous depth's best "
                "parameters. Use QAOA directly if you want to seed the first run.",
                UserWarning,
                stacklevel=2,
            )

        if checkpoint_config is not None:
            self._checkpoint_config = checkpoint_config
        checkpoint_config = self._checkpoint_config or CheckpointConfig()
        max_iterations = kwargs.pop("max_iterations", None)
        reject_unclaimed_run_kwargs(self, kwargs)

        if self._training_finished:
            ended = (
                f"converged at depth {self._depth_history[-1]['depth']}"
                if self._converged
                else f"already run all its depths (1 to max_depth={self._max_depth})"
            )
            warn(
                f"This IterativeQAOA has {ended}, so run() neither trains nor "
                "finalises. To sample the best depth's parameters, call "
                "sample_solution().",
                UserWarning,
                stacklevel=3,
            )
            return self

        # Depths already in the history are done; a restored or interrupted
        # run continues after them. Mid-depth checkpoints read the history.
        start_depth = len(self._depth_history) + 1
        resumes_mid_depth = self.n_layers == start_depth and self.current_iteration > 0
        if checkpoint_config.checkpoint_dir is not None:
            # Refused before this run supersedes the top-level completion file.
            self._reject_used_checkpoint_dir(
                self._depth_checkpoint_config(
                    checkpoint_config, start_depth
                ).checkpoint_dir,
                self.current_iteration if resumes_mid_depth else 0,
            )
            completion = (
                Path(checkpoint_config.checkpoint_dir) / PROGRAM_COMPLETION_FILE
            )
            completion.unlink(missing_ok=True)
        prev_best_params = (
            self._depth_history[-1]["best_params"].copy()
            if self._depth_history
            else None
        )

        for depth in range(start_depth, self._max_depth + 1):
            self._progress_emitter(
                ProgressEvent.show(
                    self._progress_key, f"Depth {depth}/{self._max_depth}"
                )
            )
            depth_initial_params = None

            # A checkpoint taken mid-depth already carries that depth's ansatz,
            # parameters and optimizer state; continue it rather than restart it.
            if not (depth == start_depth and resumes_mid_depth):
                self._rebuild_for_depth(depth)
                self._reset_optimization_state()

                if depth > 1 and prev_best_params is not None:
                    interpolated = interpolate_qaoa_params(
                        prev_best_params,
                        depth - 1,
                        self._strategy,
                        self._n_basis_terms,
                    )
                    depth_initial_params = np.tile(
                        interpolated, (self.optimizer.n_param_sets, 1)
                    )

            self.max_iterations = (
                self._get_max_iters(depth) if max_iterations is None else max_iterations
            )

            if self.current_iteration < self.max_iterations:
                self._optimize(
                    depth_initial_params,
                    self._depth_checkpoint_config(checkpoint_config, depth),
                )

            self._depth_history.append(
                {
                    "depth": depth,
                    "best_loss": self._best_loss,
                    "best_params": self._best_params.copy(),
                    "n_iterations": self.current_iteration,
                }
            )
            prev_best_params = self._best_params.copy()

            if self._converged:
                break

        best_entry = min(self._depth_history, key=lambda d: d["best_loss"])
        self._best_depth = best_entry["depth"]

        # Restore the instance to the best depth
        self._rebuild_for_depth(self._best_depth)
        self._best_params = best_entry["best_params"]
        self._final_params = self._best_params.copy()
        self._best_loss = best_entry["best_loss"]

        self._finalize(perform_final_computation)
        _write_program_completion(self, checkpoint_config.checkpoint_dir)

        return self

    @property
    def _converged(self) -> bool:
        """Whether the last depth improved the loss, by less than the
        convergence threshold."""
        history = self._depth_history
        return (
            self._convergence_threshold is not None
            and len(history) > 1
            and 0
            <= history[-2]["best_loss"] - history[-1]["best_loss"]
            < self._convergence_threshold
        )

    @property
    def _training_finished(self) -> bool:
        # Per-depth iteration limits do not end the run; only the last depth
        # or convergence does.
        return bool(self._depth_history) and (
            self._depth_history[-1]["depth"] == self._max_depth or self._converged
        )

    @property
    def best_depth(self) -> int:
        """The circuit depth that achieved the best (lowest) loss."""
        return self._best_depth

    @property
    def depth_history(self) -> list[dict]:
        """Per-depth optimisation results.

        Each entry is a dict with keys:
        ``depth``, ``best_loss``, ``best_params``, ``n_iterations``.
        """
        return self._depth_history.copy()
