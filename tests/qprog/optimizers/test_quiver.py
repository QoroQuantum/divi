# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the QUIVER optimizer (forward gradients, arXiv 2606.09734) and the
per-evaluation shot-budget + measurement-variance channel that powers its
adaptive ``M`` allocation."""

import networkx as nx
import numpy as np
import pytest

from divi.qprog import QAOA, EarlyStopping, QUIVEROptimizer
from divi.qprog.early_stopping import StopReason
from divi.qprog.optimizers._spsa import _cost_fn_supports_variance
from divi.qprog.problems import MaxCutProblem
from tests._helpers import exact_match
from tests.qprog.optimizers._helpers import (
    FIXED_SINGLE_DIRECTION,
    callback_trace,
    counting_cost_fn,
    small_vqe,
)
from tests.qprog.optimizers._helpers import sphere_cost_fn_batch_aware as _sphere


def _linear(slope):
    def cost(params):
        rows = np.atleast_2d(params)
        values = slope * rows.sum(axis=1)
        return values if rows.shape[0] > 1 else float(values[0])

    return cost


def _negative_cosine(params):
    return -np.cos(np.atleast_2d(params)).sum(axis=1)


def test_optimizer_contract(gradient_optimizer_contract):
    gradient_optimizer_contract(QUIVEROptimizer(learning_rate=0.1), {})


@pytest.fixture(
    params=[
        lambda: QUIVEROptimizer(learning_rate=0.1),
        lambda: QUIVEROptimizer(learning_rate=0.1, adapt_V=False, adapt_M=False),
    ],
    ids=["adaptive", "fixed"],
)
def noisy_quiver_optimizer_factory(request):
    return request.param


def test_noisy_optimizer_contract(
    noisy_quiver_optimizer_factory, noisy_optimizer_contract
):
    noisy_optimizer_contract(noisy_quiver_optimizer_factory, {}, ceiling=0.20)


# --------------------------------------------------------------------------- #
# Forward-gradient optimizer
# --------------------------------------------------------------------------- #


def test_quiver_parameter_shift_mode_converges():
    """The π/2 directional shift drives the parameters to the minimum. The
    recorded ``fun`` is the perturbation-average proxy, biased by O(shift²) — so
    convergence is asserted on the recovered parameters, not the proxy value."""
    opt = QUIVEROptimizer(
        learning_rate=0.2, V_init=2, V_max=8, derivative_mode="parameter_shift"
    )
    result = opt.optimize(
        _sphere,
        initial_params=np.array([1.0, -0.8]),
        max_iterations=200,
        rng=np.random.default_rng(1),
    )
    assert np.linalg.norm(result.x) < 0.1


_V_BOUNDS = "Require 1 <= V_min <= V_init <= V_max when V_max is set, got "
_M_INIT_BOUNDS = "Require 1 <= M_min and, when set, M_min <= M_init, got "
_M_MAX_BOUNDS = "When M_max is set, require M_min <= M_max and M_init <= M_max, got "
_RATE_BOUNDS = "Require 0 < rate_down <= 1 <= rate_up, got "
_ADAM_BETAS = "adam_beta1 and adam_beta2 must both be in (0, 1)."


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"learning_rate": 0.0}, "learning_rate must be positive, got 0.0."),
        ({"epsilon": -0.1}, "epsilon must be positive, got -0.1."),
        ({"epsilon": 0.0}, "epsilon must be positive, got 0.0."),
        ({"V_init": 0}, _V_BOUNDS + "V_min=1, V_init=0, V_max=None."),
        ({"V_min": 2, "V_init": 1}, _V_BOUNDS + "V_min=2, V_init=1, V_max=None."),
        ({"V_max": 1, "V_init": 2}, _V_BOUNDS + "V_min=1, V_init=2, V_max=1."),
        ({"M_min": 5, "M_init": 1}, _M_INIT_BOUNDS + "M_min=5, M_init=1, M_max=None."),
        ({"M_init": 10, "M_max": 5}, _M_MAX_BOUNDS + "M_min=10, M_init=10, M_max=5."),
        ({"M_min": 10, "M_max": 5}, _M_MAX_BOUNDS + "M_min=10, M_init=None, M_max=5."),
        (
            {"M_min": 1, "M_init": 10, "M_max": 5},
            _M_MAX_BOUNDS + "M_min=1, M_init=10, M_max=5.",
        ),
        ({"mu": 1.0}, "mu must be in (0, 1), got 1.0."),
        ({"mu": 0.0}, "mu must be in (0, 1), got 0.0."),
        ({"allocation_alpha": 0.0}, "allocation_alpha must be positive, got 0.0."),
        (
            {"target_gradient_variance": 0.0},
            "target_gradient_variance must be positive, got 0.0.",
        ),
        ({"warmup": -1}, "warmup must be non-negative, got -1."),
        ({"rate_down": 0.0}, _RATE_BOUNDS + "rate_down=0.0, rate_up=1.5."),
        ({"rate_up": 0.5}, _RATE_BOUNDS + "rate_down=0.7, rate_up=0.5."),
        ({"adam_beta1": 0.0}, _ADAM_BETAS),
        ({"adam_beta1": 1.0}, _ADAM_BETAS),
        ({"adam_beta2": 0.0}, _ADAM_BETAS),
        ({"adam_beta2": 1.0}, _ADAM_BETAS),
        ({"adam_epsilon": 0.0}, "adam_epsilon must be positive, got 0.0."),
        (
            {"derivative_mode": "nope"},
            "derivative_mode must be 'finite_diff' or 'parameter_shift', got 'nope'.",
        ),
    ],
)
def test_quiver_constructor_validation(kwargs, message):
    with pytest.raises(ValueError, match=exact_match(message)):
        QUIVEROptimizer(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"V_init": 2, "V_max": 2},
        {"M_min": 1},
        {"M_min": 10, "M_max": 10},
        {"M_init": 20, "M_max": 20},
        {"warmup": 0},
        {"rate_down": 1.0},
        {"rate_up": 1.0},
    ],
)
def test_quiver_constructor_accepts_boundary_values(kwargs):
    optimizer = QUIVEROptimizer(**kwargs)
    assert {name: getattr(optimizer, name) for name in kwargs} == kwargs


@pytest.mark.parametrize(
    "measurement_variance, expected_M_star",
    [(4.0, 10.0 / 3.0), (0.5, 5.0 / 12.0)],
)
def test_quiver_joint_allocation_matches_published_rule(
    measurement_variance, expected_M_star
):
    optimizer = QUIVEROptimizer(
        allocation_alpha=2.0,
        target_gradient_variance=0.5,
    )

    V_star, M_star = optimizer._allocation_targets(
        n_params=5,
        gradient_norm_squared=3.0,
        measurement_variance=measurement_variance,
    )

    assert V_star == pytest.approx(36.0)
    assert M_star == pytest.approx(expected_M_star)


@pytest.mark.parametrize(
    ("target", "current", "lower", "expected"),
    [
        (100.0, 4, 2, 6),  # rise capped at rate_up * current
        (0.0, 4, 2, 3),  # fall capped at rate_down * current
        (0.0, 2, 1, 1),  # floor wins over the rate limit
        (100.0, 1, 2, 2),  # hard bound wins when the intervals do not overlap
        (100.0, 30, 1, 20),  # an allocation above the hard cap is clamped to it
    ],
)
def test_quiver_rate_limits_allocation_changes(target, current, lower, expected):
    optimizer = QUIVEROptimizer(V_init=4, V_min=2, V_max=20)

    assert optimizer._rate_limited_integer(target, current, lower, 20) == expected


def test_quiver_default_learning_rate_scales_adams_update_by_dimension():
    opt = QUIVEROptimizer(calibration_steps=10)
    gain = opt._calibrated_learning_rate(
        _sphere,
        np.ones(100),
        np.random.default_rng(0),
    )

    assert gain == pytest.approx(2.0 * (2.0 * np.pi / 10.0) / 100.0)


def test_quiver_default_first_step_descends_on_a_sparse_high_dimensional_sphere():
    start = np.zeros(100)
    start[0] = 1.0
    seen = []

    QUIVEROptimizer(adapt_V=False, adapt_M=False).optimize(
        _sphere,
        initial_params=start,
        max_iterations=2,
        callback_fn=lambda result: seen.append(result.x.squeeze().copy()),
        rng=np.random.default_rng(0),
    )

    assert _sphere(seen[1]) < _sphere(start)


def test_quiver_default_learning_rate_spends_no_calibration_evaluations():
    """Unlike SPSA, QUIVER derives its default gain from the dimension alone."""
    calls, counting = counting_cost_fn()
    QUIVEROptimizer(V_init=1, V_min=1).optimize(
        counting,
        initial_params=np.array([1.0, 1.0]),
        max_iterations=3,
        rng=np.random.default_rng(0),
    )
    assert calls["n"] == 3


# --------------------------------------------------------------------------- #
# Integration with a shot-based VQE — the realistic QUIVER setting
# --------------------------------------------------------------------------- #


def test_quiver_runs_under_vqe(sampling_test_simulator, default_optimizer):
    vqe = small_vqe(sampling_test_simulator, default_optimizer)
    vqe.backend.set_seed(1997)
    vqe.optimizer = QUIVEROptimizer(learning_rate=0.2, epsilon=0.1, V_init=2)
    vqe.max_iterations = 8
    vqe.run(perform_final_computation=False)
    assert len(vqe.losses_history) == 8
    assert np.isfinite(vqe.best_loss)


# --------------------------------------------------------------------------- #
# Per-evaluation shot budget + measurement-variance channel
# --------------------------------------------------------------------------- #


def test_shots_override_threads_to_backend_without_mutation(
    sampling_test_simulator, default_optimizer
):
    """A per-evaluation ``shots`` budget reaches the backend as ``shot_groups``
    and never mutates the immutable backend's configured ``shots``."""
    vqe = small_vqe(sampling_test_simulator, default_optimizer)
    configured_shots = vqe.backend.shots
    vqe.backend.set_seed(7)
    theta = np.linspace(0.1, 1.0, vqe.n_layers * vqe.n_params_per_layer)

    captured = {}
    original = vqe.backend.submit_circuits

    def spy(payloads, **kwargs):
        captured["shot_groups"] = kwargs.get("shot_groups")
        return original(payloads, **kwargs)

    vqe.backend.submit_circuits = spy

    vqe._evaluate_cost_param_sets(theta[None, :], shots=512)
    # Every emitted [start, end, shots] triple carries the override budget.
    assert captured["shot_groups"] is not None
    assert all(triple[2] == 512 for triple in captured["shot_groups"])
    assert vqe.backend.shots == configured_shots


def test_cost_variance_is_positive_and_scales_inversely_with_shots(
    sampling_test_simulator, default_optimizer
):
    """The returned shot-noise variance is finite and positive, and shrinks ~1/M
    as the per-evaluation shot budget grows."""
    vqe = small_vqe(sampling_test_simulator, default_optimizer)
    vqe.backend.set_seed(7)
    theta = np.linspace(0.1, 1.0, vqe.n_layers * vqe.n_params_per_layer)

    var_low = vqe._cost_shot_variances(
        vqe._evaluate_cost_param_sets(theta[None, :], shots=500, collect_variance=True)
    )
    var_high = vqe._cost_shot_variances(
        vqe._evaluate_cost_param_sets(theta[None, :], shots=8000, collect_variance=True)
    )
    assert np.isfinite(var_low[0]) and var_low[0] > 0
    assert np.isfinite(var_high[0]) and var_high[0] > 0
    # 16× shots: 1/M scaling gives 16, 1/√M would give 4.
    assert 8.0 < var_low[0] / var_high[0] < 32.0


def test_cost_variance_is_nan_on_native_expval_backend(
    default_test_simulator, default_optimizer
):
    """On a native-expval backend no counts are produced, so the variance is nan
    and QUIVER falls back to fixed-M (V-from-spread only)."""
    vqe = small_vqe(default_test_simulator, default_optimizer)
    theta = np.linspace(0.1, 1.0, vqe.n_layers * vqe.n_params_per_layer)
    variances = vqe._cost_shot_variances(
        vqe._evaluate_cost_param_sets(theta[None, :], collect_variance=True)
    )
    assert np.isnan(variances[0])


def test_cost_fn_supports_variance_detection():
    """Variance capability is declared by the producer via a ``supports_variance``
    attribute, not inferred from the signature — so a flagged callable is
    variance-capable while any unflagged callable, whatever its signature shape,
    is not (no fragile signature sniffing)."""

    def declared(params, *, shots=None, return_variance=False):
        return params

    setattr(declared, "supports_variance", True)
    assert _cost_fn_supports_variance(declared)

    # No flag → not variance-capable, regardless of signature.
    assert not _cost_fn_supports_variance(
        lambda x, shots=None, return_variance=False: x
    )
    assert not _cost_fn_supports_variance(lambda x, **kwargs: x)
    assert not _cost_fn_supports_variance(lambda x: x)


def test_quiver_falls_back_for_partial_variance_signatures():
    """A loss-only callable that merely happens to accept ``**kwargs`` or
    ``return_variance`` must run via the plain (variance-free) path rather than
    crashing on the ``shots`` keyword or a bad unpack."""
    partial_callables = [
        lambda p: _sphere(p),
        lambda p, **kw: _sphere(p),
        lambda p, return_variance=False: _sphere(p),
    ]
    for fn in partial_callables:
        result = QUIVEROptimizer(V_init=2).optimize(
            fn,
            initial_params=np.array([1.0, 1.0]),
            max_iterations=5,
            rng=np.random.default_rng(0),
        )
        assert np.isfinite(result.fun[0])


@pytest.mark.parametrize(
    "config, cost_fn, expected",
    [
        ({"epsilon": 0.1}, _sphere, 2.0),
        ({"derivative_mode": "parameter_shift"}, _negative_cosine, np.sin(1.0)),
    ],
    ids=["finite-difference", "parameter-shift"],
)
def test_quiver_callback_jac_recovers_the_gradient_scale(config, cost_fn, expected):
    """At θ = 1 the central difference is exact on the sphere (∇f = 2), and the
    ±π/2 rule with its ½ prefactor recovers ∇(−cos θ) = sin θ exactly — not the
    2/π-scaled finite-difference value. Adam's first step is scale-free, so the
    scale is observable only through ``jac``."""
    _, trace = callback_trace(
        QUIVEROptimizer(learning_rate=0.1, **FIXED_SINGLE_DIRECTION, **config),
        cost_fn,
        [1.0],
        1,
    )
    np.testing.assert_allclose(trace[0].jac, [[expected]], rtol=1e-12)


def test_quiver_rejects_multistart_initial_params():
    """Single-point optimizer: a multi-row (m, n) array is rejected with a clear
    error rather than silently flattened or crashing on a broadcast mismatch."""
    message = (
        "QUIVEROptimizer is a single-point optimizer; initial_params must be 1-D "
        "or shape (1, n_params), got shape (3, 2)."
    )
    with pytest.raises(ValueError, match=exact_match(message)):
        QUIVEROptimizer(V_init=2).optimize(
            _sphere,
            initial_params=np.zeros((3, 2)),
            max_iterations=2,
            rng=np.random.default_rng(0),
        )


def test_quiver_accepts_single_row_2d_initial_params():
    """A (1, n) array is accepted and returns a 1-D parameter vector."""
    result = QUIVEROptimizer(V_init=2).optimize(
        _sphere,
        initial_params=np.array([[1.0, -0.5]]),
        max_iterations=3,
        rng=np.random.default_rng(0),
    )
    assert result.x.shape == (2,)


def test_quiver_blocking_converges_on_sphere():
    """Blocking routes the candidate evaluation through the variance-stashing
    ``cost_only`` adapter; the run must still converge."""
    result = QUIVEROptimizer(learning_rate=0.3, V_init=2, blocking=True).optimize(
        _sphere,
        initial_params=np.array([1.5, -1.2, 0.8]),
        max_iterations=200,
        rng=np.random.default_rng(0),
    )
    assert result.fun[0] < 0.1


def test_quiver_exact_loss_spends_one_extra_evaluation_per_step():
    """``exact_loss`` adds one unperturbed cost call per step on top of the ``V``
    perturbation calls, plus a single final-iterate evaluation after the loop
    (so best-tracking covers the last step). V fixed at 2, adaptation off."""
    calls_base, fn_base = counting_cost_fn()
    QUIVEROptimizer(
        learning_rate=0.1, V_init=2, adapt_V=False, adapt_M=False, exact_loss=False
    ).optimize(
        fn_base,
        initial_params=np.array([1.0, 1.0]),
        max_iterations=5,
        rng=np.random.default_rng(0),
    )
    calls_exact, fn_exact = counting_cost_fn()
    QUIVEROptimizer(
        learning_rate=0.1, V_init=2, adapt_V=False, adapt_M=False, exact_loss=True
    ).optimize(
        fn_exact,
        initial_params=np.array([1.0, 1.0]),
        max_iterations=5,
        rng=np.random.default_rng(0),
    )
    assert calls_base["n"] == 2 * 5
    # 5 per-step unperturbed evals + 1 final-iterate eval.
    assert calls_exact["n"] == calls_base["n"] + 5 + 1


@pytest.mark.parametrize(
    "config, cost_fn, start, max_iterations, expected_calls",
    [
        (
            {"V_init": 2, "V_max": 10, "allocation_alpha": 2.0},
            _linear(2.0),
            [1.0],
            3,
            6,
        ),
        (
            {"V_init": 2, "target_gradient_variance": 1e-12},
            _sphere,
            np.ones(12),
            6,
            2 + 3 + 5 + 8 + 8 + 8,
        ),
        (
            {"V_init": 2, "V_max": 3, "target_gradient_variance": 1e-12},
            _sphere,
            np.ones(12),
            4,
            2 + 3 + 3 + 3,
        ),
        ({"V_init": 4}, _sphere, np.ones(2), 2, 4 + 4),
    ],
    ids=[
        "anchored-target-holds-V",
        "automatic-cap-is-eight",
        "explicit-cap",
        "automatic-cap-never-below-V_init",
    ],
)
def test_quiver_adapts_the_direction_count(
    config, cost_fn, start, max_iterations, expected_calls
):
    """With no ``target_gradient_variance`` the target is anchored so that ``V*``
    equals ``V_init`` while the gradient norm holds (a linear cost). A tiny
    explicit target drives ``V`` up by ``rate_up`` per step until the cap,
    ``min(n_params, 8)`` unless ``V_max`` is given."""
    calls, counting = counting_cost_fn(cost_fn)
    QUIVEROptimizer(learning_rate=0.01, warmup=0, adapt_M=False, **config).optimize(
        counting,
        initial_params=np.asarray(start, dtype=np.float64),
        max_iterations=max_iterations,
        rng=np.random.default_rng(0),
    )
    assert calls["n"] == expected_calls


def test_quiver_adam_epsilon_damps_the_first_step():
    """The first Adam step is ``a·ĝ/(|ĝ| + ε)``; with ĝ = 2 and ε = 1 it is 2/3."""
    _, trace = callback_trace(
        QUIVEROptimizer(
            learning_rate=1.0,
            epsilon=0.5,
            adam_epsilon=1.0,
            **FIXED_SINGLE_DIRECTION,
        ),
        _sphere,
        [1.0],
        2,
    )
    np.testing.assert_allclose(trace[1].x, [[1.0 / 3.0]], rtol=1e-12)


def _forwarded_shots(max_iterations, cost_kwargs, config):
    """Shot budget QUIVER forwards per step to a variance-declaring linear cost
    (one direction per step, adaptation from step 0)."""
    slope = cost_kwargs.get("slope", 2.0)
    variance = cost_kwargs.get("variance", 0.15)
    forwarded = []

    def cost_fn(params, *, shots=None, return_variance=False):
        rows = np.atleast_2d(params)
        forwarded.append(shots)
        return slope * rows.sum(axis=1), np.full(rows.shape[0], variance)

    setattr(cost_fn, "supports_variance", True)
    if "default_shots" in cost_kwargs:
        setattr(cost_fn, "default_shots", cost_kwargs["default_shots"])
    settings = {
        "learning_rate": 0.01,
        "epsilon": 0.125,
        "V_init": 1,
        "V_min": 1,
        "adapt_V": False,
        "warmup": 0,
        "M_min": 10,
        "M_max": 1000,
        **config,
    }
    QUIVEROptimizer(**settings).optimize(
        cost_fn,
        initial_params=np.array([1.0]),
        max_iterations=max_iterations,
        rng=np.random.default_rng(0),
    )
    return forwarded


@pytest.mark.parametrize(
    "cost_kwargs, config, max_iterations, expected",
    [
        ({"default_shots": 1000}, {"M_init": 80}, 1, [80]),
        ({"default_shots": 5}, {"M_max": None}, 1, [10]),
        ({"default_shots": 1000}, {"M_max": 100}, 1, [100]),
        ({"variance": 1e6}, {"M_init": 10}, 4, [10, 15, 23, 35]),
        ({}, {"M_init": 100}, 3, [100, 120, 122]),
        (
            {"variance": 120.0 * np.pi**2 / 50.0},
            {"M_init": 100, "derivative_mode": "parameter_shift"},
            2,
            [100, 120],
        ),
        ({"slope": 0.01, "variance": 0.0}, {"M_init": 100}, 2, [100, 70]),
        ({}, {"M_init": 100, "adapt_M": False}, 3, [100, 100, 100]),
        ({"variance": np.nan}, {"M_init": 100}, 3, [100, 100, 100]),
        ({}, {"M_init": 100, "warmup": 1}, 3, [100, 100, 120]),
    ],
    ids=[
        "explicit-M_init",
        "inherited-below-M_min",
        "inherited-above-M_max",
        "rate-limited-growth",
        "joint-rule",
        "parameter-shift-scale",
        "noise-free-shrinks",
        "adapt_M-off",
        "variance-unavailable",
        "warmup",
    ],
)
def test_quiver_forwards_the_adapted_shot_budget(
    cost_kwargs, config, max_iterations, expected
):
    """``M* = n·σ̂²/(α·ĝ²)`` with ``σ̂² = M·s²·(v₊ + v₋)``, rate-limited to
    ``[0.7·M, 1.5·M]``. Slope 2 gives ``ĝ² = 4`` exactly; with ``ε = 0.125``
    (``s = 1/2ε = 4``) and ``v = 0.15`` the first target is 120, and the second
    step's EMA gives 122.4. Under the parameter-shift rule ``s = ½`` and
    ``ĝ = π``."""
    assert _forwarded_shots(max_iterations, cost_kwargs, config) == expected


def _record_forwarded_shots(vqe, optimizer, max_iterations):
    """Run ``vqe`` and return every per-evaluation shot count it forwarded."""
    vqe.optimizer = optimizer
    vqe.max_iterations = max_iterations
    seen_shots: list[int] = []
    original = vqe.backend.submit_circuits

    def spy(payloads, **kwargs):
        shot_groups = kwargs.get("shot_groups")
        if shot_groups:
            seen_shots.extend(triple[2] for triple in shot_groups)
        return original(payloads, **kwargs)

    vqe.backend.submit_circuits = spy
    vqe.run(perform_final_computation=False)
    return seen_shots


def test_quiver_default_m_inherits_and_caps_at_backend_shots(
    sampling_test_simulator, default_optimizer
):
    """With no ``M_init`` the budget starts at the backend's own shot count and
    never exceeds it, so adaptation cannot silently outspend the user."""
    vqe = small_vqe(sampling_test_simulator, default_optimizer)
    seen_shots = _record_forwarded_shots(
        vqe,
        QUIVEROptimizer(learning_rate=0.2, adapt_V=False, adapt_M=True),
        max_iterations=8,
    )

    assert seen_shots
    assert seen_shots[0] == vqe.backend.shots
    assert max(seen_shots) <= vqe.backend.shots


def test_quiver_gradient_drives_vqa_early_stopping(
    sampling_test_simulator, default_optimizer
):
    vqe = small_vqe(
        sampling_test_simulator,
        default_optimizer,
        early_stopping=EarlyStopping(
            patience=10,
            grad_norm_threshold=np.inf,
        ),
    )
    vqe.optimizer = QUIVEROptimizer(
        learning_rate=0.2,
        adapt_V=False,
        adapt_M=False,
    )
    vqe.max_iterations = 5

    vqe.run(perform_final_computation=False)

    assert vqe.current_iteration == 1
    assert vqe.stop_reason is StopReason.GRADIENT_BELOW_THRESHOLD


def test_quiver_adapt_m_warns_with_shot_distribution(
    sampling_test_simulator, default_optimizer
):
    """``adapt_M`` assumes uniform per-group shots; combining it with a shot
    distribution is flagged at program-validation time."""
    vqe = small_vqe(
        sampling_test_simulator, default_optimizer, shot_distribution="weighted"
    )
    message = (
        "QUIVEROptimizer: adapt_M=True with a configured shot_distribution — the "
        "per-direction shot-budget adaptation assumes uniform per-group shots and "
        "may be miscalibrated. Disable adapt_M or remove the shot distribution."
    )
    with pytest.warns(UserWarning, match=exact_match(message)):
        QUIVEROptimizer(adapt_M=True).validate_program(vqe)


def test_quiver_no_shot_distribution_warning_when_safe(
    recwarn, sampling_test_simulator, default_optimizer
):
    """No shot-distribution warning when ``adapt_M`` is off or no distribution
    is configured."""
    QUIVEROptimizer(adapt_M=True).validate_program(
        small_vqe(sampling_test_simulator, default_optimizer)
    )
    QUIVEROptimizer(adapt_M=False).validate_program(
        small_vqe(
            sampling_test_simulator, default_optimizer, shot_distribution="weighted"
        )
    )
    assert not [w for w in recwarn if "shot_distribution" in str(w.message)]


def test_quiver_runs_under_qaoa_with_variance(sampling_test_simulator):
    """QAOA's cost is an expectation measurement, so the variance channel
    populates and M-adaptation works there as it does for VQE."""
    qaoa = QAOA(
        MaxCutProblem(nx.bull_graph()),
        n_layers=1,
        optimizer=QUIVEROptimizer(learning_rate=0.2, epsilon=0.1, V_init=2),
        max_iterations=5,
        backend=sampling_test_simulator,
    )
    qaoa.run(perform_final_computation=False)
    assert len(qaoa.losses_history) == 5
    assert np.isfinite(qaoa.best_loss)
    assert qaoa._last_cost_variance is not None


def test_variance_nan_when_keys_collapse_across_extra_axes(
    sampling_test_simulator, default_optimizer
):
    """A pipeline with reduce axes beyond ``param_set`` (e.g. ZNE scales) yields
    several variance entries per parameter set; they cannot be collapsed to one
    scalar, so the variance is reported as nan and the optimizer falls back."""
    vqe = small_vqe(sampling_test_simulator, default_optimizer)
    vqe._last_cost_variance = {
        (("circuit", 0), ("zne_scale", 1), ("param_set", 0)): 0.01,
        (("circuit", 0), ("zne_scale", 3), ("param_set", 0)): 0.04,
    }
    variances = vqe._cost_shot_variances({0: 0.5})
    assert np.isnan(variances[0])


def test_shots_override_is_ignored_on_analytic_expval_backend(
    default_test_simulator, default_optimizer
):
    """A per-evaluation shots override on the backend-native expval path is a
    no-op (analytic expval ignores shots), not a ham_ops/shot_groups crash."""
    vqe = small_vqe(default_test_simulator, default_optimizer)
    theta = np.linspace(0.1, 1.0, vqe.n_layers * vqe.n_params_per_layer)
    losses = vqe._evaluate_cost_param_sets(
        theta[None, :], shots=128, collect_variance=True
    )
    variances = vqe._cost_shot_variances(losses)
    assert np.isfinite(losses[0])
    assert np.isnan(variances[0])


def test_quiver_runs_on_analytic_backend(default_test_simulator, default_optimizer):
    """QUIVER always sends a shots override; on an analytic backend that override
    must be silently ignored rather than crashing, with variance falling back to
    nan (fixed-M, V-from-spread only)."""
    vqe = small_vqe(default_test_simulator, default_optimizer)
    vqe.optimizer = QUIVEROptimizer(
        learning_rate=0.2, epsilon=0.1, V_init=2, adapt_M=True
    )
    vqe.max_iterations = 6
    vqe.run(perform_final_computation=False)
    assert len(vqe.losses_history) == 6
    assert np.isfinite(vqe.best_loss)
