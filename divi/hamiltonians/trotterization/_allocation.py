# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""The StateDRIFT sampling distribution for a given depth (arXiv:2610.31515,
Appendices F and J)."""

from dataclasses import dataclass
from warnings import warn

import numpy as np
import numpy.typing as npt

from divi.hamiltonians.trotterization._state_drift import StateDriftExpansion

# Least share τ of the vanilla (|c_j| / λ) distribution mixed into every term.
_LEAST_VANILLA_SHARE = 0.2
# Largest rotation angle |c_j|·dt/q_j any term may get (θ_max, Eq. F10).
_MAX_ROTATION_ANGLE = 0.5
# Bias and variance may exceed those of vanilla sampling by this fraction (η, Eq. 49).
_MARGIN = 0.05
# The fourth-order expansion is trusted only above this validity ratio (Eq. I13).
_MIN_VALIDITY = 2.0
# Appendix J: Adam on the four shared scalars from several seeds, then per term.
_SEEDS = 4
_SHARED_STEPS, _SHARED_RATE = 400, 0.1
_TERM_STEPS, _TERM_RATE = 600, 0.05
_TAUS = np.linspace(_LEAST_VANILLA_SHARE, 1.0, 81)


def state_drift_distribution(
    expansion: StateDriftExpansion,
    *,
    time: float,
    n_samples: int,
    n_trajectories: int,
) -> dict[str, float]:
    """StateDRIFT probabilities for every term of ``expansion.densities.terms``
    when ``n_trajectories`` runs of ``n_samples`` rotations over ``time`` are
    averaged.

    The family ``q_j ∝ |c_j| sqrt(ρ_j|γ_j| + α_j β_j² + μ_j Φ_j + ν_j φ_{H,j}²
    + ν'_j φ_{j,H}²)`` (Eq. F7), mixed with vanilla sampling (Eq. 23), is fitted
    by Adam to the fourth-order mean-squared error, or to ``|B_N|`` when every
    ``β_j`` is zero (Eq. F9). Each distribution keeps every rotation angle
    ``|c_j| dt / q_j`` within 0.5 (Eq. F10) and is kept only if its bias and
    variance stay within 5% of vanilla sampling's (Eq. 49); the vanilla share
    is then searched over ``[0.2, 1]`` for the smallest ``|B_N|`` (Eq. J1).
    Sampling falls back to weighted, with a warning, when no term affects the
    observable, when no distribution keeps the angles within 0.5, or when the
    expansion is not reliable (``R_val ≤ 2``, Eq. I13).
    """
    problem = _Problem.of(expansion, time, n_samples, n_trajectories)
    reason = problem.unsuitable()
    if reason is not None:
        warn(
            f"StateDRIFT falls back to weighted sampling: {reason}.",
            UserWarning,
            stacklevel=2,
        )
        probs = problem.vanilla
    else:
        probs = problem.solve()
    labels = expansion.densities.terms.paulis.to_labels()
    return dict(zip(labels, (probs / probs.sum()).tolist()))


@dataclass(frozen=True)
class _Problem:
    expansion: StateDriftExpansion
    coeffs: npt.NDArray[np.floating]
    vanilla: npt.NDArray[np.floating]
    floors: npt.NDArray[np.floating]
    """The least probability of each term that keeps its angle within θ_max."""
    basis: npt.NDArray[np.floating]
    """``|γ_j|``, ``β_j²``, ``Φ_j``, ``φ_{H,j}²`` and ``φ_{j,H}²`` per term (rows)."""
    dt: float
    n_samples: int
    n_trajectories: int

    @classmethod
    def of(
        cls,
        expansion: StateDriftExpansion,
        time: float,
        n_samples: int,
        n_trajectories: int,
    ) -> "_Problem":
        d = expansion.densities
        coeffs = d.terms.coeffs.real
        dt = time / n_samples
        a, b, phi = expansion.nested_curvatures
        squares = phi**2
        # Φ_j = ½ Σ_b c_b² (φ_{j,b}² + φ_{b,j}²) (Eq. F6).
        pair_curvatures = 0.5 * (
            np.bincount(a, coeffs[b] ** 2 * squares, len(coeffs))
            + np.bincount(b, coeffs[a] ** 2 * squares, len(coeffs))
        )
        return cls(
            expansion,
            coeffs,
            np.abs(coeffs) / np.abs(coeffs).sum(),
            np.abs(coeffs) * abs(dt) / _MAX_ROTATION_ANGLE,
            np.stack(
                [
                    np.abs(d.conjugation_shifts),
                    d.rotation_slopes**2,
                    pair_curvatures,
                    expansion.inner_curvatures**2,
                    expansion.outer_curvatures**2,
                ]
            ),
            dt,
            n_samples,
            n_trajectories,
        )

    def unsuitable(self) -> str | None:
        """Why StateDRIFT cannot be used here, or ``None``."""
        if not np.any(self.basis):
            return "no Hamiltonian term changes the observable in the initial state"
        if np.any(self.vanilla < self.floors):
            return (
                f"no distribution keeps every rotation angle |c_j|·dt/q_j within "
                f"{_MAX_ROTATION_ANGLE} (dt = time / (n_steps · sampling_budget)); "
                "increase n_steps or sampling_budget"
            )
        second, third, fourth = self.expansion.truncated_bias(
            self.vanilla, dt=self.dt, n_samples=self.n_samples
        )
        remainder = abs(third - second) + abs(fourth - third)
        if remainder > 0 and abs(fourth) / remainder <= _MIN_VALIDITY:
            return (
                "its fourth-order expansion is not reliable for this evolution "
                f"(R_val = {abs(fourth) / remainder:.3g} ≤ {_MIN_VALIDITY:g})"
            )
        return None

    @property
    def real_amplitudes(self) -> bool:
        return not np.any(self.basis[1])

    def bias(self, probs: npt.NDArray[np.floating]) -> float:
        return self.expansion.truncated_bias(
            probs, dt=self.dt, n_samples=self.n_samples
        )[-1]

    def variance(self, probs: npt.NDArray[np.floating]) -> float:
        return self.expansion.truncated_variance(
            probs, dt=self.dt, n_samples=self.n_samples
        )[-1]

    def solve(self) -> npt.NDArray[np.floating]:
        caps = (
            (1 + _MARGIN) * abs(self.bias(self.vanilla)),
            (1 + _MARGIN) * self.variance(self.vanilla),
        )
        best = _Best(caps)
        shared = self._fit_shared(best)
        weights = np.repeat(np.exp(shared)[:, None], len(self.coeffs), axis=1)
        self._adam(np.log(weights), best, _TERM_STEPS, _TERM_RATE, shared=False)
        spread = self._spread(best.weights if best.weights is not None else weights)
        return self._best_mixture(spread, caps)

    def _fit_shared(self, best: "_Best") -> npt.NDArray[np.floating]:
        """Log-weights of the shared-scalar family (Eq. F14, ``ρ = 1``) that
        fit best over ``_SEEDS`` random starts."""
        weighted = np.abs(self.coeffs) @ self.basis.T
        scales = np.ones(len(self.basis))
        active = weighted > 0
        if active[0]:
            scales[active] = weighted[0] / weighted[active]
        starts = []
        for seed in range(_SEEDS):
            spread = np.random.default_rng(seed).uniform(-2, 2, len(self.basis))
            log_weights = np.log(scales) + spread * np.log(10)
            log_weights[0] = 0.0
            fitted, value = self._adam(
                log_weights, best, _SHARED_STEPS, _SHARED_RATE, shared=True
            )
            starts.append((value, fitted))
        return min(starts, key=lambda start: start[0])[1]

    def _adam(
        self,
        log_weights: npt.NDArray[np.floating],
        best: "_Best",
        steps: int,
        rate: float,
        *,
        shared: bool,
    ) -> tuple[npt.NDArray[np.floating], float]:
        """Adam on the log-weights; returns them and the last objective value."""
        moment = np.zeros_like(log_weights)
        second_moment = np.zeros_like(log_weights)
        value = np.inf
        for step in range(1, steps + 1):
            weights = np.exp(log_weights)
            if shared:
                weights = np.repeat(weights[:, None], len(self.coeffs), axis=1)
            value, gradient = self._objective_in_weights(weights, best)
            if shared:
                gradient = gradient.sum(axis=1)
                gradient[0] = 0.0
            moment = 0.9 * moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * gradient**2
            log_weights = log_weights - rate * (moment / (1 - 0.9**step)) / (
                np.sqrt(second_moment / (1 - 0.999**step)) + 1e-12
            )
        return log_weights, value

    def _spread(self, weights: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """``q^SD`` of Eq. (F15) for per-term weights ``(5, L)``."""
        root = np.abs(self.coeffs) * np.sqrt((weights * self.basis).sum(axis=0))
        return root / root.sum()

    def _objective_in_weights(
        self, weights: npt.NDArray[np.floating], best: "_Best"
    ) -> tuple[float, npt.NDArray[np.floating]]:
        """The objective at the distribution of ``weights`` and its gradient
        with respect to their logarithms (Eqs. F15-F18), the vanilla share
        held at its least angle-keeping value."""
        f = (weights * self.basis).sum(axis=0)
        spread = self._spread(weights)
        probs, share = _mix_with_vanilla(spread, self.vanilla, self.floors)
        value, gradient, bias, variance = self._objective(probs)
        best.offer(value, weights, bias, variance)
        centred = gradient - np.dot(spread, gradient)
        scale = np.divide(
            (1 - share) * spread * centred,
            2 * f,
            out=np.zeros_like(f),
            where=f > 0,
        )
        return value, weights * self.basis * scale

    def _objective(
        self, probs: npt.NDArray[np.floating]
    ) -> tuple[float, npt.NDArray[np.floating], float, float]:
        """``|B_N|`` for real amplitudes, else ``MSE_N = N² b₂² dt⁴ + V_N / M``
        (Eqs. F9, F21), with its gradient, ``B_N`` and ``V_N``."""
        bias, bias_gradient = self.expansion.bias_with_gradient(
            probs, dt=self.dt, n_samples=self.n_samples
        )
        variance, variance_gradient = self.expansion.variance_with_gradient(
            probs, dt=self.dt, n_samples=self.n_samples
        )
        if self.real_amplitudes:
            return abs(bias), np.sign(bias) * bias_gradient, bias, variance
        d = self.expansion.densities
        squares = self.coeffs**2
        leading = np.dot(squares / probs, d.conjugation_shifts) - d.curvature / 2
        scale = self.n_samples**2 * self.dt**4
        value = scale * leading**2 + variance / self.n_trajectories
        gradient = (
            -2 * scale * leading * squares * d.conjugation_shifts / probs**2
            + variance_gradient / self.n_trajectories
        )
        return value, gradient, bias, variance

    def _best_mixture(
        self,
        spread: npt.NDArray[np.floating],
        caps: tuple[float, float],
    ) -> npt.NDArray[np.floating]:
        """The mixture with vanilla sampling over the vanilla shares of
        ``_TAUS`` that keeps every angle and both caps with the smallest
        ``|B_N|`` (Eq. J1); pure vanilla sampling always qualifies."""
        best_probs, best_bias = self.vanilla, abs(self.bias(self.vanilla))
        for share in _TAUS:
            probs = share * self.vanilla + (1 - share) * spread
            if np.any(probs < self.floors):
                continue
            bias = abs(self.bias(probs))
            if bias < best_bias and _within(caps, bias, self.variance(probs)):
                best_probs, best_bias = probs, bias
        return best_probs


@dataclass
class _Best:
    """The weights with the lowest objective among those within both caps."""

    caps: tuple[float, float]
    value: float = np.inf
    weights: npt.NDArray[np.floating] | None = None

    def offer(
        self,
        value: float,
        weights: npt.NDArray[np.floating],
        bias: float,
        variance: float,
    ) -> None:
        if value < self.value and _within(self.caps, abs(bias), variance):
            self.value, self.weights = value, weights


def _within(caps: tuple[float, float], bias: float, variance: float) -> bool:
    bias_cap, variance_cap = caps
    return bias <= bias_cap and 0 <= variance <= variance_cap


def _mix_with_vanilla(
    spread: npt.NDArray[np.floating],
    vanilla: npt.NDArray[np.floating],
    floors: npt.NDArray[np.floating],
) -> tuple[npt.NDArray[np.floating], float]:
    """``τ·vanilla + (1 − τ)·spread`` and ``τ``, at the smallest ``τ ≥ 0.2``
    that lifts every probability to its floor; vanilla meets every floor."""
    gain = vanilla - spread
    shortfall = floors - spread
    short = (shortfall > 0) & (gain > 0)
    share = max(
        _LEAST_VANILLA_SHARE,
        float(np.max(shortfall[short] / gain[short], initial=0.0)),
    )
    share = min(share, 1.0)
    return share * vanilla + (1 - share) * spread, share
