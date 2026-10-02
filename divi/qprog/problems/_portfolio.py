# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Mean-variance portfolio optimisation problems.

Both problems describe a portfolio as integer *units* per asset: ``total``
units make a fully invested portfolio and each asset holds at most ``cap`` of
them, so the weights are ``units / total``. Selection holds ``K`` assets with
one unit each; allocation spreads ``L`` units on a grid of step ``1/L``.
"""

import math
import operator
import warnings
from collections.abc import Callable, Hashable, Mapping, Sequence
from functools import cached_property
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import scipy.sparse as sps
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import LinearConstraint as ScipyLinearConstraint
from scipy.optimize import minimize

from divi.hamiltonians import xy_mixer
from divi.hamiltonians._mixers import single_pauli_label
from divi.qprog.algorithms import DickeState, InitialState, SuperpositionState
from divi.qprog.algorithms._initial_state import build_block_xy_mixer_graph
from divi.qprog.problems._base import QAOAProblem
from divi.qprog.problems._binary import BinaryOptimizationProblem
from divi.qprog.problems._constraints import LinearConstraint, _UnreachableBoundError
from divi.qprog.problems._partitioning_config import QUBOPartitioningConfig
from divi.qprog.problems._qubo_partitioning_utils import partition_by_method

_MAX_DOUBLE_TRANSFERS = 2_000_000
_EPS = 1e-12


class _PortfolioBase(BinaryOptimizationProblem):
    """Market data, formulation, repair and metrics shared by the portfolio problems.

    A portfolio holds ``total`` units, at most ``cap`` per asset; qubit ``j``
    of an asset carries ``place[j]`` units in the given ``encoding``.
    """

    def __init__(
        self,
        expected_returns: npt.ArrayLike,
        covariance: npt.ArrayLike,
        *,
        total: int,
        cap: int,
        place: np.ndarray,
        encoding: Literal["log", "domain_wall"],
        risk_tolerance: float,
        constraints: Sequence[LinearConstraint],
        penalty_weight: float,
        config: QUBOPartitioningConfig | None,
    ):
        if config is not None and not isinstance(config, QUBOPartitioningConfig):
            raise TypeError(
                f"config must be a QUBOPartitioningConfig, got {type(config).__name__}."
            )
        self._config = config
        self._clusters: dict[Hashable, tuple[np.ndarray, int]] = {}
        self._fixed_units: dict[int, int] = {}
        mu = np.asarray(expected_returns, dtype=float)
        sigma = np.asarray(covariance, dtype=float)
        if mu.ndim != 1 or mu.size == 0:
            raise ValueError("expected_returns must be a non-empty 1-D array.")
        n = mu.size
        if sigma.shape != (n, n):
            raise ValueError(
                f"covariance must have shape ({n}, {n}), got {sigma.shape}."
            )
        if not (np.isfinite(mu).all() and np.isfinite(sigma).all()):
            raise ValueError("expected_returns and covariance must be finite.")
        if not np.allclose(sigma, sigma.T, rtol=1e-8, atol=1e-12):
            raise ValueError("covariance must be symmetric.")
        sigma = 0.5 * (sigma + sigma.T)
        min_eig = float(np.linalg.eigvalsh(sigma).min())
        if min_eig < -1e-10 * max(1.0, float(np.abs(sigma).max())):
            warnings.warn(
                f"covariance is not positive semi-definite (smallest eigenvalue "
                f"{min_eig:.3g}); some portfolios will have negative variance.",
                UserWarning,
                stacklevel=3,
            )
        risk_tolerance = float(risk_tolerance)
        if not math.isfinite(risk_tolerance) or risk_tolerance < 0:
            raise ValueError("risk_tolerance must be finite and non-negative.")

        self._mu, self._sigma, self._risk_tolerance = mu, sigma, risk_tolerance
        self._total, self._cap, self._place = total, cap, place
        self._encoding: Literal["log", "domain_wall"] = encoding
        self._weight_constraints = tuple(constraints)
        self._weight_matrix = np.zeros((len(self._weight_constraints), n))
        for row, constraint in zip(self._weight_matrix, self._weight_constraints):
            if constraint._n_positions not in (None, n):
                raise ValueError(
                    f"{constraint!r} has {constraint._n_positions} coefficients "
                    f"for {n} assets."
                )
            for i, c in constraint.coefficients.items():
                if not isinstance(i, (int, np.integer)) or not 0 <= i < n:
                    raise ValueError(
                        f"Constraint coefficients must be keyed by asset index "
                        f"0..{n - 1}; got {i!r}."
                    )
                row[i] = c
        spread = np.ptp(self._weight_matrix, axis=1)
        self._violation_scale = np.where(spread > _EPS, spread, 1.0)

        cost, budget, on_qubits, walls = self._formulate()
        try:
            super().__init__(
                cost,
                constraints=[budget, *on_qubits],
                penalty=walls,
                penalty_weight=penalty_weight,
            )
        except _UnreachableBoundError as err:
            index = next(
                (i for i, c in enumerate(on_qubits) if c is err.constraint), None
            )
            if index is None:
                raise
            raise ValueError(
                f"{self._weight_constraints[index]!r} is infeasible: over the "
                f"fully invested portfolios this problem allows, its weighted sum "
                f"ranges over [{err.lo / total + 0.0:.6g}, {err.hi / total + 0.0:.6g}]."
            ) from err

    def _formulate(
        self,
    ) -> tuple[np.ndarray, LinearConstraint, list[LinearConstraint], dict | None]:
        """The QUBO objective, budget and weight constraints on the qubits, and walls."""
        n, total, width = self.n_assets, self._total, len(self._place)
        # Bit (i, j) carries place[j] units of asset i.
        encoder = np.kron(np.eye(n), self._place[None, :])
        cost = encoder.T @ self._sigma @ encoder / total**2 - np.diag(
            self._risk_tolerance * (self._mu @ encoder) / total
        )
        budget = LinearConstraint(encoder.T @ np.ones(n), "==", total)
        on_qubits = [
            LinearConstraint(encoder.T @ row, c.sense, c.bound * total)
            for row, c in zip(self._weight_matrix, self._weight_constraints)
        ]
        walls: dict[tuple, float] | None = None
        if self._encoding == "domain_wall" and width > 1:
            # x_{j+1} (1 - x_j) for each adjacent pair: 1 per broken wall.
            walls = {}
            for s in range(0, n * width, width):
                for j in range(width - 1):
                    walls[(s + j + 1,)] = 1.0
                    walls[(s + j, s + j + 1)] = -1.0
        return cost, budget, on_qubits, walls

    def _asset_bits(self, bitstring: str) -> np.ndarray:
        """Decision bits of ``bitstring``, one row per asset."""
        return np.array(self._solution_key(bitstring)).reshape(self.n_assets, -1)

    def _units(self, bitstring: str) -> np.ndarray:
        """Units per asset encoded in ``bitstring``."""
        return self._asset_bits(bitstring) @ self._place

    def _unit_bits(self, units: np.ndarray) -> list[int]:
        """Decision bits that encode ``units``."""
        width = len(self._place)
        if self._encoding == "domain_wall":
            return [int(j < k) for k in units for j in range(width)]
        return [(k >> j) & 1 for k in units for j in range(width)]

    def _activity_bounds(
        self, coefficients: Mapping[Hashable, float]
    ) -> tuple[float, float]:
        """Minimum and maximum of ``Σ aᵢxᵢ`` over fully invested portfolios.

        Only the decision bits of portfolios holding ``total`` units, at most
        ``cap`` per asset, count, so constraint slack covers no more than those.
        """
        n, total, cap = self.n_assets, self._total, self._cap
        levels = np.array([self._unit_bits(np.array([k])) for k in range(cap + 1)])
        a = np.array([coefficients.get(q, 0.0) for q in range(n * levels.shape[1])])
        # values[i, k]: the activity contributed by asset i holding k units.
        values = a.reshape(n, -1) @ levels.T

        def extreme(sign: float) -> float:
            # Knapsack over assets: best[t] is the best activity using t units so far.
            best = np.full(total + 1, -np.inf)
            best[0] = 0.0
            for row in sign * values:
                shifted = np.full((cap + 1, total + 1), -np.inf)
                for k in range(cap + 1):
                    shifted[k, k:] = best[: total + 1 - k] + row[k]
                best = shifted.max(axis=0)
            return sign * float(best[total])

        return extreme(-1.0), extreme(1.0)

    @property
    def constraints(self) -> tuple[LinearConstraint, ...]:
        """The constraints on portfolio weights passed at construction."""
        return self._weight_constraints

    @property
    def expected_returns(self) -> np.ndarray:
        """Expected return of each asset."""
        return self._mu.copy()

    @property
    def covariance(self) -> np.ndarray:
        """Covariance matrix of asset returns."""
        return self._sigma.copy()

    @property
    def risk_tolerance(self) -> float:
        r"""Weight :math:`\tau` on expected return in the objective."""
        return self._risk_tolerance

    @property
    def n_assets(self) -> int:
        """Number of assets in the universe."""
        return self._mu.size

    def is_feasible(self, bitstring: str) -> bool:
        """Budget, extra constraints and (domain-wall only) canonical walls all hold."""
        if self._encoding == "domain_wall":
            bits = self._asset_bits(bitstring)
            if np.any(bits[:, :-1] < bits[:, 1:]):
                return False
        return super().is_feasible(bitstring)

    def _objective(self, units: np.ndarray) -> np.ndarray:
        """Mean-variance objective of one or more unit vectors (rows)."""
        w = np.asarray(units, dtype=float) / self._total
        return np.einsum("...i,ij,...j->...", w, self._sigma, w) - (
            self._risk_tolerance * w @ self._mu
        )

    def metrics(self, bitstring: str, *, risk_free: float = 0.0) -> dict[str, float]:
        """Expected return, volatility and Sharpe ratio of the portfolio in ``bitstring``.

        All values are in the period of ``expected_returns`` and
        ``covariance``; ``risk_free`` must be in the same period. The weights
        sum to one only for bitstrings that meet the budget, so pass a
        feasible (e.g. repaired) bitstring.

        Args:
            bitstring: Measured or repaired bitstring over all qubits.
            risk_free: Risk-free rate for the Sharpe ratio. Defaults to ``0.0``.

        Returns:
            ``{"expected_return", "volatility", "sharpe"}``. ``sharpe`` is
            ``nan`` when the volatility is zero.
        """
        w = self._units(bitstring) / self._total
        ret = float(self._mu @ w)
        vol = math.sqrt(max(float(w @ self._sigma @ w), 0.0))
        return {
            "expected_return": ret,
            "volatility": vol,
            "sharpe": (ret - risk_free) / vol if vol > 0 else float("nan"),
        }

    def repair_infeasible_bitstring(
        self, bitstring: str
    ) -> tuple[str, Any, float | None]:
        """Repair a bitstring to a fully invested portfolio that meets every constraint.

        First adds or removes single units greedily on the objective until the
        portfolio is fully invested. Then, while any constraint is violated,
        moves one unit from one asset to another, picking the move that most
        reduces the total violation (each constraint's violation divided by the
        spread of its coefficients) and breaking ties on the objective; when no
        single move helps, it tries two moves at once. Slack bits are
        recomputed to match.

        Returns:
            ``(bitstring, decoded, energy)``, with ``decoded`` as from
            :attr:`decode_fn`. ``energy`` is ``None`` when no sequence of such
            moves reaches feasibility.
        """
        units = self._transfer_to_feasible(self._fill_budget(self._units(bitstring)))
        repaired = self._complete_bitstring(dict(enumerate(self._unit_bits(units))))
        energy = self.compute_energy(repaired) if self.is_feasible(repaired) else None
        return repaired, self.decode_fn(repaired), energy

    def _fill_budget(self, units: np.ndarray) -> np.ndarray:
        while (gap := self._total - units.sum()) != 0:
            step = 1 if gap > 0 else -1
            movable = np.flatnonzero((units + step >= 0) & (units + step <= self._cap))
            candidates = units + step * np.eye(len(units), dtype=int)[movable]
            units = candidates[np.argmin(self._objective(candidates))]
        return units

    def _violation(self, lhs: np.ndarray) -> np.ndarray:
        """Total weight-constraint violation, each scaled by its coefficients' spread."""
        total = np.zeros(lhs.shape[1:])
        for gap, c, s in zip(lhs, self._weight_constraints, self._violation_scale):
            gap = gap - c.bound
            if c.sense == "==":
                total += np.abs(gap) / s
            else:
                total += np.maximum(gap if c.sense == "<=" else -gap, 0.0) / s
        return total

    def _transfer_to_feasible(self, units: np.ndarray) -> np.ndarray:
        if not self._weight_constraints:
            return units
        for _ in range(4 * self.n_assets * self._cap + 16):
            current = float(self._violation(self._weight_matrix @ units / self._total))
            if current <= _EPS:
                break
            moved = next(
                (
                    best[0]
                    for n_moves in (1, 2)
                    if (best := self._best_transfers(units, n_moves))
                    and best[1] < current - _EPS
                ),
                None,
            )
            if moved is None:
                break
            units = moved
        return units

    def _improve_by_transfers(self, units: np.ndarray) -> np.ndarray:
        """Best feasible one-unit transfer, repeated while the objective falls."""
        current = float(self._objective(units))
        while (best := self._best_transfers(units, 1)) is not None:
            moved, violation, objective = best
            if violation > _EPS or objective >= current - _EPS * max(1.0, abs(current)):
                break
            units, current = moved, objective
        return units

    def _best_transfers(
        self, units: np.ndarray, n_moves: int
    ) -> tuple[np.ndarray, float, float] | None:
        """``(units, violation, objective)`` after the best ``n_moves`` one-unit transfers.

        Best means least total violation, then lowest objective. ``None`` when
        there is no such transfer.
        """
        eye = np.eye(len(units), dtype=int)
        src, dst = np.nonzero(
            (units > 0)[:, None] & (units < self._cap)[None, :] & (eye == 0)
        )
        if n_moves == 1:
            combos = np.arange(len(src))[:, None]
        else:
            if len(src) * (len(src) - 1) // 2 > _MAX_DOUBLE_TRANSFERS:
                return None
            i, j = np.triu_indices(len(src), 1)
            distinct = (src[i] != src[j]) & (dst[i] != dst[j])
            distinct &= (src[i] != dst[j]) & (dst[i] != src[j])
            combos = np.stack([i[distinct], j[distinct]], axis=1)
        if not len(combos):
            return None

        delta = (
            self._weight_matrix[:, dst] - self._weight_matrix[:, src]
        ) / self._total
        lhs = self._weight_matrix @ units / self._total
        violations = self._violation(lhs[:, None] + delta[:, combos].sum(axis=2))
        least = float(violations.min())
        ties = combos[violations <= least + _EPS]
        candidates = units + (eye[dst[ties]] - eye[src[ties]]).sum(axis=1)
        objectives = self._objective(candidates)
        best = int(np.argmin(objectives))
        return candidates[best], least, float(objectives[best])

    def _cluster_problem(self, assets: np.ndarray, share: int) -> "_PortfolioBase":
        """Sub-problem over ``assets`` whose objective is their term of the global one."""
        raise NotImplementedError

    def decompose(self) -> dict[Hashable, QAOAProblem]:
        """One sub-problem per asset cluster, each holding a share of the budget.

        Shares follow the continuous relaxation, or cluster size if it fails or
        leaves nothing to solve. Extra constraints are enforced after combining.

        Raises:
            ValueError: If no partitioning config was provided at construction,
                or no cluster has more than one possible holding.
        """
        config = self._config
        if config is None:
            raise ValueError(
                "Cannot decompose: no partitioning config was provided at construction."
            )
        clusters = partition_by_method(sps.csr_matrix(self._sigma), config)
        # Group uncorrelated assets, within the size limit and above the floor.
        singles = [c for c in clusters if len(c) == 1]
        if len(singles) > 1:
            multi = [c for c in clusters if len(c) > 1]
            flat = np.concatenate(singles)
            n_groups = math.ceil(len(flat) / (config.max_cluster_size or len(flat)))
            n_groups = max(n_groups, (config.minimum_n_clusters or 0) - len(multi))
            clusters = multi + np.array_split(flat, min(n_groups, len(flat)))
        for relaxed in (True, False):
            shares = self._cluster_shares(clusters, relaxed=relaxed)
            self._clusters, self._fixed_units = {}, {}
            sub_problems: dict[Hashable, QAOAProblem] = {}
            for i, (assets, share) in enumerate(zip(clusters, shares.tolist())):
                if len(assets) == 1 or share in (0, len(assets) * self._cap):
                    self._fixed_units.update(
                        dict.fromkeys(assets.tolist(), share // len(assets))
                    )
                    continue
                prog_id = (f"P{i}", len(assets))
                self._clusters[prog_id] = (assets, share)
                sub_problems[prog_id] = self._cluster_problem(assets, share)
            if sub_problems:
                return sub_problems
        raise ValueError(
            "Every asset cluster has a single possible holding, leaving nothing "
            "to solve; solve the problem without partitioning."
        )

    def _cluster_shares(
        self, clusters: list[np.ndarray], *, relaxed: bool
    ) -> np.ndarray:
        """Budget per cluster by largest remainder, from the relaxation or by size."""
        n, total, cap = self.n_assets, self._total, self._cap
        sigma, mu, tau = self._sigma, self._mu, self._risk_tolerance
        weights = caps = np.array([len(c) * cap for c in clusters])
        if relaxed:
            cons = self._weight_constraints
            a = np.vstack([np.ones(n), self._weight_matrix])
            lb = np.r_[1.0, [-np.inf if c.sense == "<=" else c.bound for c in cons]]
            ub = np.r_[1.0, [np.inf if c.sense == ">=" else c.bound for c in cons]]
            # SLSQP's tolerance is absolute: scale the objective to order one.
            scale = max(np.abs(sigma).max(), tau * np.abs(mu).max(), _EPS)
            relaxation = minimize(
                lambda w: (w @ sigma @ w - tau * mu @ w) / scale,
                np.full(n, 1 / n),
                jac=lambda w: (2 * sigma @ w - tau * mu) / scale,
                bounds=[(0.0, cap / total)] * n,
                constraints=[
                    ScipyLinearConstraint(a[m], lb[m], ub[m])
                    for m in (lb == ub, lb != ub)
                    if m.any()
                ],
                method="SLSQP",
            )
            x = np.clip(relaxation.x, 0.0, None)
            if relaxation.success and np.isfinite(x).all() and x.sum() > 0:
                weights = np.array([x[c].sum() for c in clusters])
        quota = np.minimum(total * weights / weights.sum(), caps)
        shares = np.floor(quota).astype(int)
        while (short := total - shares.sum()) > 0:
            room = np.flatnonzero(shares < caps)
            shares[room[np.argsort((shares - quota)[room], kind="stable")[:short]]] += 1
        return shares

    def _has_reproducible_decomposition(self) -> bool:
        return True

    def initial_solution_size(self) -> int:
        return self.n_assets

    def _global_bitstring(self, solution: Sequence[int]) -> str:
        """Bitstring of a units-per-asset solution, with fixed assets and slack set."""
        units = np.array(solution)
        units[list(self._fixed_units)] = list(self._fixed_units.values())
        return self._complete_bitstring(dict(enumerate(self._unit_bits(units))))

    @cached_property
    def _penalised_qubo(self) -> tuple[np.ndarray, float]:
        """Upper-triangular matrix and constant of the (quadratic) penalised QUBO."""
        idx = self._canonical_problem.variable_to_idx
        q, offset = np.zeros((len(idx), len(idx))), 0.0
        for term, coeff in self._canonical_problem.terms.items():
            if term:
                q[idx[term[0]], idx[term[-1]]] += coeff
            else:
                offset += coeff
        return q, offset

    def evaluate_global_solution(self, solution: list[int]) -> float:
        """Penalised QUBO energy, with the slack that minimises it."""
        q, offset = self._penalised_qubo
        x = np.fromiter(map(int, self._global_bitstring(solution)), dtype=float)
        return float(x @ q @ x + offset)

    def postprocess_candidates(
        self, candidates: list[tuple[float, list[int]]], *, strict: bool = False
    ) -> list[tuple[Any, float]]:
        """Distinct feasible ``(decoded, objective)`` pairs, best first.

        Infeasible candidates are repaired, or dropped when ``strict`` is set or
        the repair fails, then improved by feasible one-unit transfers.
        """
        results: dict[str, tuple[Any, float]] = {}
        for _, solution in candidates:
            bitstring = self._global_bitstring(solution)
            if not self.is_feasible(bitstring):
                if strict:
                    continue
                bitstring, _, energy = self.repair_infeasible_bitstring(bitstring)
                if energy is None:
                    continue
            units = self._improve_by_transfers(self._units(bitstring))
            bitstring = self._complete_bitstring(
                dict(enumerate(self._unit_bits(units)))
            )
            results[bitstring] = (
                self.decode_fn(bitstring),
                self.compute_energy(bitstring),
            )
        return sorted(results.values(), key=lambda entry: entry[1])


class PortfolioSelectionProblem(_PortfolioBase):
    r"""Cardinality-constrained, equal-weight mean-variance portfolio.

    Chooses exactly :math:`K` of :math:`N` assets and holds each at weight
    :math:`w_i = x_i / K`, so the portfolio is long-only and fully invested.
    Each asset is one qubit :math:`x_i`. The objective is

    .. math::

        \min_x \; \frac{x^\top \Sigma x}{K^2} - \tau \frac{\mu^\top x}{K}
        \quad \text{s.t.} \quad \sum_i x_i = K,

    plus any extra ``constraints``, which are stated on the weights
    :math:`w`. For example, ``LinearConstraint(esg, ">=", 65)`` asks for a
    mean ESG score of at least 65 across the holdings, and
    ``LinearConstraint({i: 1 for i in tech}, "<=", 0.4)`` caps the tech
    sector at 40% of the portfolio (at most :math:`0.4K` names). A limit on
    the number of names is written the same way: at most :math:`m` names from
    a group is ``LinearConstraint({i: 1 for i in group}, "<=", m / K)``.

    The cardinality constraint is always encoded as a penalty, so
    :class:`~divi.qprog.algorithms.PCE` can take the problem. With
    ``use_constrained_mixer=True``, QAOA uses a ring XY mixer on the asset
    qubits, which conserves their Hamming weight, and starts from a
    :class:`~divi.qprog.algorithms.DickeState` of weight :math:`K` on them.
    Slack qubits from inequality constraints keep an X mixer.

    Args:
        expected_returns: Expected return :math:`\mu_i` of each asset.
        covariance: Symmetric covariance matrix :math:`\Sigma` of asset
            returns, in the same period as ``expected_returns`` (typically
            annualised).
        n_holdings: Number of assets :math:`K` to hold.
        risk_tolerance: Weight :math:`\tau \ge 0` on expected return; larger
            values accept more variance for more return. ``0`` minimises
            variance alone over the equal-weight :math:`K`-asset portfolios.
            Equivalent to Qiskit Finance ``risk_factor = 1/(Kτ)`` and
            PyPortfolioOpt ``risk_aversion = 2/τ``. Defaults to ``0.0``.
        constraints: Extra :class:`~divi.qprog.problems.LinearConstraint`\ s on
            the weights, keyed by asset index.
        penalty_weight: Multiplier on every constraint penalty. Each penalty is
            the squared violation of the constraint on the selection
            :math:`x`, i.e. :math:`K` times its weight form (one missing name
            costs 1; an ESG shortfall of 1 point in the mean costs about
            :math:`K^2`, measured from the bound rounded inwards to the ESG
            scores' common step). Choose it so that one unit of violation
            outweighs the spread of the objective across portfolios.
        use_constrained_mixer: Use the ring XY mixer and Dicke initial state.
        config: Enables partitioned solving with
            :class:`~divi.qprog.workflows.PartitioningProgramEnsemble`; each
            asset cluster holds a share of the :math:`K` names. Its variables
            are assets, so ``max_n_variables_per_cluster`` limits the assets
            per cluster.

    Examples:
        >>> import numpy as np
        >>> from divi.qprog.problems import LinearConstraint, PortfolioSelectionProblem
        >>> mu = np.array([0.10, 0.12, 0.08, 0.15])
        >>> sigma = np.diag([0.04, 0.05, 0.03, 0.09])
        >>> esg = np.array([70.0, 55.0, 80.0, 60.0])
        >>> problem = PortfolioSelectionProblem(
        ...     mu, sigma, n_holdings=2, risk_tolerance=0.5,
        ...     constraints=[LinearConstraint(esg, ">=", 65)],
        ... )
    """

    def __init__(
        self,
        expected_returns: npt.ArrayLike,
        covariance: npt.ArrayLike,
        n_holdings: int,
        *,
        risk_tolerance: float = 0.0,
        constraints: Sequence[LinearConstraint] = (),
        penalty_weight: float = 1.0,
        use_constrained_mixer: bool = False,
        config: QUBOPartitioningConfig | None = None,
    ):
        k = operator.index(n_holdings)
        n = np.size(expected_returns)
        if not 1 <= k <= n:
            raise ValueError(f"n_holdings must be in 1..{n}, got {k}.")
        self._n_holdings = k
        self._use_constrained_mixer = bool(use_constrained_mixer)
        super().__init__(
            expected_returns,
            covariance,
            total=k,
            cap=1,
            place=np.ones(1, dtype=int),
            encoding="log",
            risk_tolerance=risk_tolerance,
            constraints=constraints,
            penalty_weight=penalty_weight,
            config=config,
        )

    def _cluster_problem(
        self, assets: np.ndarray, share: int
    ) -> "PortfolioSelectionProblem":
        ratio = share / self._total
        return PortfolioSelectionProblem(
            self._mu[assets],
            self._sigma[np.ix_(assets, assets)] * ratio**2,
            share,
            risk_tolerance=self._risk_tolerance * ratio,
            penalty_weight=self._penalty_weight,
            use_constrained_mixer=self._use_constrained_mixer,
        )

    def extend_solution(
        self, current_solution: list[int], prog_id: Hashable, candidate_decoded: Any
    ) -> list[int]:
        """Hold the cluster's chosen assets, one unit each."""
        assets, _ = self._clusters[prog_id]
        extended = np.array(current_solution)
        extended[assets] = 0
        extended[assets[list(candidate_decoded)]] = 1
        return extended.tolist()

    @property
    def n_holdings(self) -> int:
        """Number of assets :math:`K` the portfolio holds."""
        return self._n_holdings

    @property
    def use_constrained_mixer(self) -> bool:
        """Whether QAOA uses the ring XY mixer and Dicke initial state."""
        return self._use_constrained_mixer

    @property
    def mixer_hamiltonian(self) -> SparsePauliOp:
        """Ring XY mixer on the assets plus X on slack qubits, or the X mixer."""
        if not self._use_constrained_mixer:
            return super().mixer_hamiltonian
        if self._mixer_cache is None:
            n_qubits = self._ising.n_qubits
            idx = self._canonical_problem.variable_to_idx
            assets = [idx[i] for i in range(self.n_assets)]
            ring = build_block_xy_mixer_graph(
                self.n_assets, 1, assets, connectivity="ring"
            )
            slack_x = [
                (single_pauli_label(n_qubits, idx[s], "X"), 1.0)
                for s in self._slack_variables
            ]
            mixer = xy_mixer(ring, n_qubits=n_qubits)
            if slack_x:
                mixer = (mixer + SparsePauliOp.from_list(slack_x)).simplify()
            self._mixer_cache = mixer
        return self._mixer_cache

    @property
    def recommended_initial_state(self) -> InitialState:
        """Dicke state of weight :math:`K` on the assets with the constrained mixer, else superposition."""
        if self._use_constrained_mixer:
            return DickeState(self._n_holdings, n_qubits=self.n_assets)
        return SuperpositionState()

    @property
    def decode_fn(self) -> Callable[[str], Any]:
        """Map a bitstring to the sorted list of held asset indices."""
        return lambda bitstring: np.flatnonzero(self._units(bitstring)).tolist()


class PortfolioAllocationProblem(_PortfolioBase):
    r"""Long-only, fully invested mean-variance portfolio on a discrete weight grid.

    Every weight is a multiple of :math:`1/L`, :math:`w_i = k_i / L` with
    integer :math:`k_i \in \{0, \dots, L\}` and :math:`L` = ``n_steps``, and
    the budget constraint :math:`\sum_i w_i = 1` (:math:`\sum_i k_i = L`)
    keeps the portfolio fully invested. The objective is

    .. math::

        \min_w \; w^\top \Sigma w - \tau \mu^\top w,

    plus any extra ``constraints`` on the weights, e.g.
    :math:`\sum_i \mathrm{esg}_i w_i \ge 60` for a weighted-mean ESG floor.

    Two encodings of :math:`k_i` are available:

    - ``"log"``: binary (logarithmic) encoding with :math:`\log_2(L+1)`
      qubits per asset. ``n_steps`` must be :math:`2^b - 1` (1, 3, 7, 15, …);
      round grids such as 5% steps (:math:`L = 20`) need ``"domain_wall"``.
    - ``"domain_wall"``: :math:`L` qubits per asset, valid states are
      :math:`1^{k_i} 0^{L-k_i}`. An extra penalty (scaled by
      ``penalty_weight``) charges each broken wall.

    Args:
        expected_returns: Expected return :math:`\mu_i` of each asset.
        covariance: Symmetric covariance matrix :math:`\Sigma` of asset
            returns, in the same period as ``expected_returns`` (typically
            annualised).
        n_steps: Grid resolution :math:`L`; weights move in steps of
            :math:`1/L`. Defaults to ``15``.
        encoding: ``"log"`` (default) or ``"domain_wall"``.
        risk_tolerance: Weight :math:`\tau \ge 0` on expected return; larger
            values accept more variance for more return. ``0`` minimises
            variance alone over the portfolios on the grid. Equivalent to
            Qiskit Finance ``risk_factor = 1/(Lτ)`` over :math:`L` units and
            PyPortfolioOpt ``risk_aversion = 2/τ``. Defaults to ``0.0``.
        constraints: Extra :class:`~divi.qprog.problems.LinearConstraint`\ s on
            the weights, keyed by asset index.
        penalty_weight: Multiplier on every constraint penalty. Each penalty is
            the squared violation of the constraint on the levels :math:`k`,
            i.e. :math:`L` times its weight form (one missing step of budget
            costs 1). Choose it so that one unit of violation outweighs the
            spread of the objective across portfolios.
        config: Enables partitioned solving with
            :class:`~divi.qprog.workflows.PartitioningProgramEnsemble`; each
            asset cluster holds a share of the :math:`L` units. Its variables
            are assets, so ``max_n_variables_per_cluster`` limits the assets
            (not qubits) per cluster. Requires ``encoding="domain_wall"``.

    Examples:
        >>> import numpy as np
        >>> from divi.qprog.problems import PortfolioAllocationProblem
        >>> mu = np.array([0.10, 0.12, 0.08])
        >>> sigma = np.diag([0.04, 0.05, 0.03])
        >>> problem = PortfolioAllocationProblem(mu, sigma, n_steps=7, risk_tolerance=0.5)
    """

    def __init__(
        self,
        expected_returns: npt.ArrayLike,
        covariance: npt.ArrayLike,
        *,
        n_steps: int = 15,
        encoding: Literal["log", "domain_wall"] = "log",
        risk_tolerance: float = 0.0,
        constraints: Sequence[LinearConstraint] = (),
        penalty_weight: float = 1.0,
        config: QUBOPartitioningConfig | None = None,
    ):
        steps = operator.index(n_steps)
        if steps < 1:
            raise ValueError(f"n_steps must be ≥ 1, got {steps}.")
        if config is not None and encoding == "log":
            raise ValueError("Partitioning requires encoding='domain_wall'.")
        if encoding == "log":
            if steps != 2 ** steps.bit_length() - 1:
                raise ValueError(
                    f"The log encoding needs n_steps = 2**b - 1 (1, 3, 7, 15, 31, "
                    f"…), got {steps}. Use encoding='domain_wall' for other "
                    "grids, at n_steps qubits per asset."
                )
            place = 2 ** np.arange(steps.bit_length())
        elif encoding == "domain_wall":
            place = np.ones(steps, dtype=int)
        else:
            raise ValueError(
                f"encoding must be 'log' or 'domain_wall', got {encoding!r}."
            )
        self._n_steps = steps
        super().__init__(
            expected_returns,
            covariance,
            total=steps,
            cap=steps,
            place=place,
            encoding=encoding,
            risk_tolerance=risk_tolerance,
            constraints=constraints,
            penalty_weight=penalty_weight,
            config=config,
        )

    def _cluster_problem(
        self, assets: np.ndarray, share: int
    ) -> "PortfolioAllocationProblem":
        ratio = share / self._total
        return PortfolioAllocationProblem(
            self._mu[assets],
            self._sigma[np.ix_(assets, assets)] * ratio**2,
            n_steps=share,
            encoding="domain_wall",
            risk_tolerance=self._risk_tolerance * ratio,
            penalty_weight=self._penalty_weight,
        )

    def extend_solution(
        self, current_solution: list[int], prog_id: Hashable, candidate_decoded: Any
    ) -> list[int]:
        """Write the cluster's weights as units of the global solution."""
        assets, share = self._clusters[prog_id]
        extended = np.array(current_solution)
        extended[assets] = np.rint(np.asarray(candidate_decoded) * share)
        return extended.tolist()

    @property
    def n_steps(self) -> int:
        """Grid resolution :math:`L`."""
        return self._n_steps

    @property
    def encoding(self) -> Literal["log", "domain_wall"]:
        """``"log"`` or ``"domain_wall"``."""
        return self._encoding

    @property
    def decode_fn(self) -> Callable[[str], Any]:
        """Map a bitstring to the weight vector :math:`k / L`."""
        return lambda bitstring: self._units(bitstring) / self._n_steps
