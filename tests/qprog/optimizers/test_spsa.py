# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SPSA and QN-SPSA optimizers, the SPSA-family behaviour QUIVER
shares with them, and the state-overlap primitive."""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import Statevector

from divi.circuits import MetaCircuit, build_overlap_meta
from divi.hamiltonians import QDrift
from divi.pipeline import ResultFormat
from divi.pipeline.abc import ContractViolation
from divi.qprog import (
    QAOA,
    CustomVQA,
    FubiniStudyMetricEstimator,
    QNSPSAOptimizer,
    QUIVEROptimizer,
    SPSAOptimizer,
    _metrics,
)
from divi.qprog._metrics import (
    METRIC_ROUTINE,
    StochasticFidelityMetricEstimator,
    _zeros_probability,
)
from divi.qprog.optimizers._linalg import _matrix_abs_psd
from divi.qprog.optimizers._spsa import (
    _fidelity_metric_sample,
    _spsa_gain_a,
    _spsa_gain_c,
    _spsa_gradient,
)
from divi.qprog.problems import MaxCutProblem
from tests._helpers import exact_match
from tests.qprog.optimizers._helpers import (
    FIXED_SINGLE_DIRECTION,
    bowl_jac,
    bowl_metric,
    callback_trace,
    count_divergence_warnings,
    counting_cost_fn,
    data_bound_custom_vqa,
)
from tests.qprog.optimizers._helpers import sphere_cost_fn_batch_aware as _sphere

_EYE_METRIC = {"metric_fn": lambda x: np.eye(np.asarray(x).reshape(-1).shape[0])}
_UNDAMPED_QNSPSA = {"c": 0.5, "regularization": 0.0, "max_step_norm": None}
_CONSTANT_GAINS = {"alpha": 0.0, "gamma": 0.0}
_DIVERGING = "ignore:.*appears to be diverging.*:UserWarning"

#: ``(optimizer_cls, config, optimize_kwargs)`` per SPSA-family optimizer, with
#: perturbation 0.5 and no damping.
_FAMILY = [
    (SPSAOptimizer, {"c": 0.5}, {}),
    (QNSPSAOptimizer, _UNDAMPED_QNSPSA, _EYE_METRIC),
    (
        QUIVEROptimizer,
        {"epsilon": 0.5, "adam_epsilon": 1e-300, **FIXED_SINGLE_DIRECTION},
        {},
    ),
]
_FAMILY_IDS = ["spsa", "qnspsa", "quiver"]
_over_family = pytest.mark.parametrize(
    "optimizer_cls, config, optimize_kwargs", _FAMILY, ids=_FAMILY_IDS
)


@pytest.fixture(
    params=[
        (lambda: SPSAOptimizer(learning_rate=0.2, c=0.1), {}),
        (lambda: QNSPSAOptimizer(learning_rate=0.2, c=0.1), _EYE_METRIC),
    ],
    ids=["default", "quantum-natural"],
)
def spsa_contract_case(request):
    factory, optimize_kwargs = request.param
    return factory(), optimize_kwargs


def test_optimizer_contract(spsa_contract_case, gradient_optimizer_contract):
    optimizer, optimize_kwargs = spsa_contract_case
    gradient_optimizer_contract(optimizer, optimize_kwargs)


_BOWL_EVALUATORS = {"jac": bowl_jac, "metric_fn": bowl_metric}


@pytest.fixture(
    params=[
        (lambda: SPSAOptimizer(learning_rate=0.2, c=0.2), {}, 0.10),
        (
            lambda: SPSAOptimizer(learning_rate=0.2, c=0.2, blocking=True),
            {},
            0.10,
        ),
        (
            lambda: SPSAOptimizer(learning_rate=0.2, c=0.2, resamplings=4),
            {},
            0.10,
        ),
        (
            lambda: QNSPSAOptimizer(learning_rate=0.1, c=0.2),
            _BOWL_EVALUATORS,
            0.60,
        ),
        (
            lambda: QNSPSAOptimizer(learning_rate=0.1, c=0.2, blocking=True),
            _BOWL_EVALUATORS,
            0.60,
        ),
    ],
    ids=[
        "default",
        "blocking",
        "resampled",
        "quantum-natural",
        "quantum-natural-blocking",
    ],
)
def noisy_spsa_contract_case(request):
    return request.param


def test_noisy_optimizer_contract(noisy_spsa_contract_case, noisy_optimizer_contract):
    noisy_optimizer_contract(*noisy_spsa_contract_case)


# --------------------------------------------------------------------------- #
# Batch-aware test costs (the real cost_fn handles 2D batches; SPSA evaluates
# its ± perturbations as a single two-row batch)
# --------------------------------------------------------------------------- #


def _quadratic(matrix: np.ndarray):
    def cost(params: np.ndarray) -> float | np.ndarray:
        params = np.atleast_2d(params)
        values = 0.5 * np.einsum("ij,jk,ik->i", params, matrix, params)
        return values if params.shape[0] > 1 else float(values[0])

    return cost


def _quadratic_fidelity(metric: np.ndarray):
    """Mock overlap ``F(a, b) = 1 - ½(a-b)ᵀM(a-b)`` whose FS metric is ``M/2``."""

    def fidelity_fn(theta, perturbations):
        return np.array([1.0 - 0.5 * (p @ metric @ p) for p in perturbations])

    return fidelity_fn


def _nan_cost(params):
    return np.full(np.atleast_2d(params).shape[0], np.nan)


def _first_coordinate(params):
    return np.atleast_2d(params)[:, 0].copy()


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #


def test_spsa_gradient_matches_directional_finite_difference():
    """The SPSA estimate equals the analytic projection ``2 (θ·h) h`` for the
    sphere, regardless of the perturbation size."""
    theta = np.array([0.5, -1.0, 2.0])
    h = np.array([1.0, -1.0, 1.0])
    ghat, returned_h, f_plus, f_minus = _spsa_gradient(
        _sphere, theta, c_k=0.1, rng=np.random.default_rng(0), direction=h
    )
    np.testing.assert_allclose(returned_h, h)
    np.testing.assert_allclose(ghat, 2.0 * (theta @ h) * h)
    assert f_plus == pytest.approx(_sphere(theta + 0.1 * h))
    assert f_minus == pytest.approx(_sphere(theta - 0.1 * h))


def test_spsa_gradient_draws_rademacher_directions():
    _, h, _, _ = _spsa_gradient(
        _sphere, np.zeros(8), c_k=0.1, rng=np.random.default_rng(0)
    )
    assert set(h.tolist()) == {-1.0, 1.0}


def test_spsa_gradient_accepts_a_column_shaped_batch_result():
    ghat, _, _, _ = _spsa_gradient(
        lambda batch: np.sum(batch**2, axis=1)[:, None],
        np.ones(2),
        c_k=0.1,
        rng=np.random.default_rng(0),
        direction=np.ones(2),
    )
    np.testing.assert_allclose(ghat, [4.0, 4.0])


def test_matrix_abs_psd_takes_eigenvalue_absolute_value():
    """``_matrix_abs_psd`` returns ``V |Λ| Vᵀ`` for a symmetric indefinite matrix."""
    eigvecs, _ = np.linalg.qr(np.random.default_rng(1).standard_normal((3, 3)))
    eigvals = np.array([-2.0, 0.5, 3.0])
    g = eigvecs @ np.diag(eigvals) @ eigvecs.T

    result = _matrix_abs_psd(g)
    expected = eigvecs @ np.diag(np.abs(eigvals)) @ eigvecs.T

    np.testing.assert_allclose(result, expected, atol=1e-10)
    assert np.linalg.eigvalsh(result).min() >= -1e-10


def test_matrix_abs_psd_is_identity_on_psd():
    """For a PSD matrix ``|G| == G``, so ``|G| + βI`` is plain Tikhonov damping."""
    g = np.array([[2.0, 0.3], [0.3, 1.0]])
    np.testing.assert_allclose(_matrix_abs_psd(g), g, atol=1e-10)


def test_fidelity_metric_sample_assembles_outer_products():
    """One stochastic FS sample equals ``-(δF/8c²)(h1 h2ᵀ + h2 h1ᵀ)`` for the
    four overlaps the optimizer requests, in the exact order expected."""
    theta = np.array([0.3, -0.7])
    h1 = np.array([1.0, -1.0])
    c_k = 0.2
    captured = {}

    def fidelity_fn(theta_in, perturbations):
        captured["perts"] = perturbations
        return np.array([0.9, 0.7, 0.6, 0.5])

    raw = _fidelity_metric_sample(fidelity_fn, theta, h1, c_k, np.random.default_rng(0))

    # The perturbation list must be [c(h1+h2), c·h1, c(-h1+h2), -c·h1].
    h2 = captured["perts"][0] / c_k - h1  # perts[0] = c(h1+h2)
    np.testing.assert_allclose(captured["perts"][1], c_k * h1)
    np.testing.assert_allclose(captured["perts"][0], c_k * h1 + c_k * h2)
    np.testing.assert_allclose(captured["perts"][2], -c_k * h1 + c_k * h2)
    np.testing.assert_allclose(captured["perts"][3], -c_k * h1)

    delta_f = 0.9 - 0.7 - 0.6 + 0.5
    expected = -(delta_f / (8.0 * c_k * c_k)) * (np.outer(h1, h2) + np.outer(h2, h1))
    np.testing.assert_allclose(raw, expected)


def test_spsa_gain_schedules_decay():
    """Gain helpers follow Spall's a/(A+k+1)^α and c/(k+1)^γ."""
    assert _spsa_gain_a(0, a=0.2, A=1.0, alpha=0.602) == pytest.approx(0.2 / 2.0**0.602)
    assert _spsa_gain_c(0, c=0.2, gamma=0.101) == pytest.approx(0.2)
    # Larger A damps the early learning rate.
    assert _spsa_gain_a(0, a=0.2, A=1.0, alpha=0.602) > _spsa_gain_a(
        0, a=0.2, A=100.0, alpha=0.602
    )


def test_zeros_probability_dict_and_list_branches():
    assert _zeros_probability({"00": 0.6, "01": 0.4}, "00") == pytest.approx(0.6)
    assert _zeros_probability({"01": 1.0}, "00") == pytest.approx(0.0)
    dists = [{"00": 0.6, "01": 0.4}, {"00": 0.8, "10": 0.2}]
    assert _zeros_probability(dists, "00") == pytest.approx(0.7)
    assert _zeros_probability([], "00") == pytest.approx(0.0)
    partial = [{"00": 0.5, "11": 0.5}, {"11": 1.0}]
    assert _zeros_probability(partial, "00") == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# SPSA optimizer
# --------------------------------------------------------------------------- #


def test_spsa_converges_on_sphere(recwarn):
    opt = SPSAOptimizer(learning_rate=0.3, c=0.1)
    result = opt.optimize(
        _sphere,
        initial_params=np.array([1.5, -1.2, 0.8]),
        max_iterations=200,
        rng=np.random.default_rng(0),
    )
    assert result.fun[0] < 0.05
    assert result.x.shape == (3,)
    assert result.success
    assert count_divergence_warnings(recwarn) == 0


@_over_family
def test_seeded_runs_are_reproducible(optimizer_cls, config, optimize_kwargs):
    def seeded_x():
        result, _ = callback_trace(
            optimizer_cls(learning_rate=0.2, **config),
            _sphere,
            [1.0, -0.5, 0.3],
            5,
            rng=np.random.default_rng(3),
            **optimize_kwargs,
        )
        return result.x

    np.testing.assert_array_equal(seeded_x(), seeded_x())


@_over_family
def test_requires_initial_params(optimizer_cls, config, optimize_kwargs):
    message = f"{optimizer_cls.__name__} requires initial_params."
    with pytest.raises(ValueError, match=exact_match(message)):
        optimizer_cls(**config).optimize(
            _sphere, initial_params=None, max_iterations=3, **optimize_kwargs
        )


@_over_family
def test_zero_iterations_raises(optimizer_cls, config, optimize_kwargs):
    message = (
        "max_iterations must be >= 1, got 0; the optimisation loop performs no "
        "evaluation with zero steps."
    )
    with pytest.raises(ValueError, match=exact_match(message)):
        optimizer_cls(**config).optimize(
            _sphere, initial_params=np.ones(2), max_iterations=0, **optimize_kwargs
        )


@_over_family
def test_a_cost_that_collapses_the_batch_is_rejected(
    optimizer_cls, config, optimize_kwargs
):
    message = (
        "cost_fn must return one value per batch row; the two-row perturbation "
        "batch produced 1 value(s). Ensure the cost function is batch-aware "
        "(returns a 1-D array for a 2-D input)."
    )
    with pytest.raises(ValueError, match=exact_match(message)):
        optimizer_cls(learning_rate=0.1, **config).optimize(
            lambda batch: float(np.sum(np.asarray(batch) ** 2)),
            initial_params=np.ones(2),
            max_iterations=2,
            rng=np.random.default_rng(0),
            **optimize_kwargs,
        )


@_over_family
def test_explicit_learning_rate_spends_one_cost_batch_per_step(
    optimizer_cls, config, optimize_kwargs
):
    calls, counting = counting_cost_fn()
    optimizer_cls(learning_rate=0.1, **config).optimize(
        counting,
        initial_params=np.ones(2),
        max_iterations=3,
        rng=np.random.default_rng(0),
        **optimize_kwargs,
    )
    assert calls["n"] == 3


@_over_family
def test_blocking_counts_calibration_seed_and_candidate_evals(
    optimizer_cls, config, optimize_kwargs
):
    """One band-calibration batch and one seed evaluation before the loop, then
    a gradient batch and a candidate evaluation per step."""
    calls, counting = counting_cost_fn()
    optimizer_cls(learning_rate=0.1, blocking=True, **config).optimize(
        counting,
        initial_params=np.array([1.0, 0.5]),
        max_iterations=3,
        rng=np.random.default_rng(0),
        **optimize_kwargs,
    )
    assert calls["n"] == 1 + 1 + 3 * (1 + 1)


@_over_family
def test_a_diverging_run_warns_once(optimizer_cls, config, optimize_kwargs):
    with pytest.warns(UserWarning, match="diverging") as record:
        optimizer_cls(learning_rate=1e4, **config).optimize(
            _sphere,
            initial_params=np.ones(2),
            max_iterations=5,
            rng=np.random.default_rng(0),
            **optimize_kwargs,
        )
    assert count_divergence_warnings(record) == 1


_GRADIENT_FREE = [
    (SPSAOptimizer, {"c": 0.1}),
    (QUIVEROptimizer, {"V_init": 2}),
]


@pytest.mark.parametrize(
    "optimizer_cls, config", _GRADIENT_FREE, ids=["spsa", "quiver"]
)
def test_ignores_jac_and_metric_fn(optimizer_cls, config):
    result = optimizer_cls(learning_rate=0.3, **config).optimize(
        _sphere,
        initial_params=np.array([1.0, 1.0]),
        max_iterations=20,
        rng=np.random.default_rng(0),
        jac=lambda x: 1 / 0,
        metric_fn=lambda x: 1 / 0,
    )
    assert np.isfinite(result.fun[0])


_SINGLE_STEP_CASES = [
    (SPSAOptimizer, {"c": 0.1}, {}),
    (QNSPSAOptimizer, {}, {"metric_fn": lambda x: np.eye(2)}),
    (QUIVEROptimizer, {"epsilon": 0.1, "V_init": 2, "adapt_V": False}, {}),
]


@pytest.mark.parametrize(
    "optimizer_cls, config, optimize_kwargs",
    _SINGLE_STEP_CASES,
    ids=_FAMILY_IDS,
)
@pytest.mark.parametrize(
    "mode", ["exact_loss", "blocking"], ids=["exact-loss", "blocking"]
)
def test_a_final_improving_step_is_returned_as_best(
    optimizer_cls, config, optimize_kwargs, mode
):
    """Best-tracking otherwise runs only before each step, so the single step of
    a one-iteration run must still be reflected in the result."""
    start = np.array([1.0, 0.3])
    result = optimizer_cls(learning_rate=0.5, **{mode: True}, **config).optimize(
        _sphere,
        initial_params=start,
        max_iterations=1,
        rng=np.random.default_rng(0),
        **optimize_kwargs,
    )
    assert result.fun[0] < _sphere(start)
    assert not np.allclose(result.x, start)


@pytest.mark.parametrize(
    "optimizer_cls, kwargs",
    [
        (
            SPSAOptimizer,
            {
                "learning_rate": 0.05,
                "c": 0.3,
                "alpha": 0.7,
                "gamma": 0.2,
                "A": 5.0,
                "blocking": True,
                "allowed_increase": 3.0,
                "resamplings": 2,
                "calibration_steps": 7,
                "exact_loss": True,
            },
        ),
        (
            QNSPSAOptimizer,
            {
                "learning_rate": 0.05,
                "A": 5.0,
                "regularization": 1e-2,
                "allowed_increase": 1.5,
                "exact_loss": True,
            },
        ),
        (
            QUIVEROptimizer,
            {
                "learning_rate": 0.05,
                "epsilon": 0.2,
                "V_init": 3,
                "V_max": 10,
                "M_init": 250,
                "M_max": 500,
                "A": 5.0,
                "allowed_increase": 1.5,
                "derivative_mode": "parameter_shift",
            },
        ),
    ],
    ids=_FAMILY_IDS,
)
def test_copy_preserves_config(optimizer_cls, kwargs):
    clone = optimizer_cls(**kwargs).copy()
    assert type(clone) is optimizer_cls
    assert {name: getattr(clone, name) for name in kwargs} == kwargs


def test_qnspsa_copy_owns_its_metric_estimator():
    estimator = FubiniStudyMetricEstimator()
    clone = QNSPSAOptimizer(metric_estimator=estimator).copy()
    assert isinstance(clone.metric_estimator, FubiniStudyMetricEstimator)
    assert clone.metric_estimator is not estimator


@_over_family
def test_optimize_draws_its_own_rng_when_none_is_given(
    optimizer_cls, config, optimize_kwargs
):
    result = optimizer_cls(learning_rate=0.2, **config).optimize(
        _sphere,
        initial_params=np.array([1.0, -0.5, 0.3]),
        max_iterations=2,
        **optimize_kwargs,
    )
    assert np.isfinite(result.fun[0])


@_over_family
def test_callback_reports_the_proxy_loss_and_iteration_count(
    optimizer_cls, config, optimize_kwargs
):
    """At θ = 1 with perturbation 0.5 the proxy ½(f₊ + f₋) is 1.25 exactly."""
    result, trace = callback_trace(
        optimizer_cls(learning_rate=0.1, **config),
        _sphere,
        [1.0],
        3,
        **optimize_kwargs,
    )
    np.testing.assert_allclose(trace[0].fun, [1.25])
    assert [step.nit for step in trace] == [1, 2, 3]
    assert all(step.success is True for step in trace)
    assert all(step.message == "Optimisation in progress." for step in trace)
    assert result.nit == 3
    assert result.message == "Optimisation terminated: reached max_iterations."


@pytest.mark.parametrize(
    "optimizer_cls, config, optimize_kwargs, A, max_iterations, expected_x1",
    [
        (*_FAMILY[0], 3.0, 2, 0.5),
        (*_FAMILY[1], 3.0, 2, 0.5),
        (*_FAMILY[2], 3.0, 2, 0.75),
        (*_FAMILY[0], None, 10, 0.0),
        (*_FAMILY[1], None, 10, 0.0),
        (*_FAMILY[2], None, 10, 0.5),
    ],
    ids=[f"{name}-{kind}" for kind in ("A", "default-A") for name in _FAMILY_IDS],
)
def test_first_step_follows_the_learning_rate_schedule(
    optimizer_cls, config, optimize_kwargs, A, max_iterations, expected_x1
):
    """``a_0 = a/(A+1)`` with ``A`` defaulting to ``0.1·max_iterations``. On the
    one-parameter sphere at θ = 1 the gradient estimate is exactly 2, and
    QUIVER's first Adam step has magnitude ``a_0``."""
    _, trace = callback_trace(
        optimizer_cls(learning_rate=1.0, alpha=1.0, gamma=0.0, A=A, **config),
        _sphere,
        [1.0],
        max_iterations,
        **optimize_kwargs,
    )
    np.testing.assert_allclose(trace[1].x, [[expected_x1]], atol=1e-12)


@_over_family
def test_blocking_records_the_exact_loss(optimizer_cls, config, optimize_kwargs):
    _, trace = callback_trace(
        optimizer_cls(learning_rate=0.1, blocking=True, **config),
        _sphere,
        [1.0, 0.5],
        3,
        **optimize_kwargs,
    )
    for step in trace:
        assert step.fun[0] == pytest.approx(_sphere(step.x))


@pytest.mark.parametrize(
    "optimizer_cls, config, optimize_kwargs, learning_rate, max_iterations",
    [
        (*_FAMILY[0], 1.0, 2),
        (SPSAOptimizer, {"c": 0.5, "exact_loss": True}, {}, 1.0, 1),
        (*_FAMILY[1], 1.0, 2),
        (*_FAMILY[2], 2.0, 2),
    ],
    ids=["spsa", "spsa-exact-loss", "qnspsa", "quiver"],
)
def test_an_equal_later_loss_keeps_the_earliest_best(
    optimizer_cls, config, optimize_kwargs, learning_rate, max_iterations
):
    """Each step maps θ = 1 to θ = -1, whose loss is identical."""
    result, _ = callback_trace(
        optimizer_cls(learning_rate=learning_rate, **_CONSTANT_GAINS, **config),
        _sphere,
        [1.0],
        max_iterations,
        **optimize_kwargs,
    )
    np.testing.assert_array_equal(result.x, [1.0])


@pytest.mark.parametrize(
    "optimizer_cls, expected",
    [
        (
            SPSAOptimizer,
            {
                "learning_rate": None,
                "c": 0.2,
                "alpha": 0.602,
                "gamma": 0.101,
                "A": None,
                "resamplings": 1,
                "blocking": False,
                "allowed_increase": None,
                "exact_loss": False,
                "calibration_steps": 25,
            },
        ),
        (
            QNSPSAOptimizer,
            {
                "learning_rate": None,
                "c": 0.2,
                "alpha": 0.602,
                "gamma": 0.101,
                "A": None,
                "regularization": 1e-3,
                "resamplings": 1,
                "blocking": False,
                "allowed_increase": None,
                "exact_loss": False,
                "calibration_steps": 25,
                "max_step_norm": pytest.approx(2.0 * np.pi / 10.0),
            },
        ),
        (
            QUIVEROptimizer,
            {
                "learning_rate": None,
                "epsilon": 0.1,
                "c": 0.1,
                "V_init": 2,
                "V_min": 1,
                "V_max": None,
                "M_init": None,
                "M_min": 10,
                "M_max": None,
                "adapt_V": True,
                "adapt_M": True,
                "derivative_mode": "finite_diff",
                "allocation_alpha": 1.0,
                "target_gradient_variance": None,
                "warmup": 5,
                "rate_down": 0.7,
                "rate_up": 1.5,
                "mu": 0.9,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_epsilon": 1e-8,
                "b": 1e-6,
                "alpha": 0.0,
                "gamma": 0.0,
                "A": None,
                "blocking": False,
                "allowed_increase": None,
                "exact_loss": False,
                "calibration_steps": 25,
            },
        ),
    ],
    ids=_FAMILY_IDS,
)
def test_documented_defaults(optimizer_cls, expected):
    optimizer = optimizer_cls()
    assert {name: getattr(optimizer, name) for name in expected} == expected


def test_block_or_step_rejects_worsening_candidate():
    """Look-ahead blocking accepts an improving candidate and rejects one that
    worsens the loss by more than ``allowed_increase``, holding the iterate."""
    opt = SPSAOptimizer(blocking=True)
    theta, proposed = np.array([0.0, 0.0]), np.array([1.0, 1.0])

    # candidate worsens the loss (sphere: 0 -> 2) beyond the band -> rejected, holds.
    nxt, loss = opt._block_or_step(
        _sphere, theta, proposed, current_loss=0.0, allowed_increase=0.5
    )
    np.testing.assert_array_equal(nxt, theta)
    assert loss == 0.0

    # candidate improves the loss (2 -> 0) -> accepted, moves, loss updates.
    nxt, loss = opt._block_or_step(
        _sphere, proposed, theta, current_loss=2.0, allowed_increase=0.5
    )
    np.testing.assert_array_equal(nxt, theta)
    assert loss == pytest.approx(0.0)


@pytest.mark.parametrize(
    "allowed_increase", [2.0, 1.0], ids=["inside-the-band", "on-the-band-edge"]
)
def test_block_or_step_accepts_a_regression_within_the_band(allowed_increase):
    """The band is what absorbs shot noise: a candidate worse by at most
    ``allowed_increase`` is still accepted, so a noisy run keeps moving."""
    nxt, loss = SPSAOptimizer(blocking=True)._block_or_step(
        _sphere,
        np.array([0.0]),
        np.array([1.0]),
        current_loss=0.0,
        allowed_increase=allowed_increase,
    )
    np.testing.assert_array_equal(nxt, [1.0])
    assert loss == 1.0


def test_block_or_step_rejects_nonfinite_candidate():
    """A NaN candidate loss is held (not accepted) — without this guard NaN would
    slip through, since ``nan > x`` is False."""
    opt = SPSAOptimizer(blocking=True)
    nxt, loss = opt._block_or_step(
        lambda x: float("nan"),
        np.array([0.0]),
        np.array([0.5]),
        current_loss=1.0,
        allowed_increase=1.0,
    )
    np.testing.assert_array_equal(nxt, np.array([0.0]))  # held at theta
    assert loss == 1.0


@pytest.mark.filterwarnings(_DIVERGING)
@_over_family
def test_an_all_nan_cost_reports_failure_rather_than_an_infinite_loss(
    optimizer_cls, config, optimize_kwargs
):
    """When nothing finite is ever measured there is no result to report, so the
    untouched starting point must not come back as an optimum."""
    start = np.array([1, 2])

    result, _ = callback_trace(
        optimizer_cls(learning_rate=0.2, **config),
        _nan_cost,
        start,
        10,
        rng=np.random.default_rng(1997),
        **optimize_kwargs,
    )

    assert result.success is False
    assert result.message == (
        "Optimisation failed: no finite cost value was observed in 10 iterations."
    )
    assert result.nit == 10
    np.testing.assert_array_equal(result.fun, [np.inf])
    np.testing.assert_array_equal(result.x, start)
    assert result.x.dtype == np.float64


@pytest.mark.parametrize(
    "optimizer, cost_fn, theta, expected",
    [
        (SPSAOptimizer(c=0.2), _first_coordinate, np.zeros(2), 2.0 * np.pi / 10.0),
        (
            SPSAOptimizer(calibration_steps=2),
            _nan_cost,
            np.zeros(2),
            2.0 * np.pi / 10.0,
        ),
        (
            SPSAOptimizer(c=0.5, calibration_steps=1),
            lambda batch: np.array([1e-10, 0.0]),
            np.zeros(1),
            2.0 * np.pi / 10.0 / 1e-10,
        ),
    ],
    ids=["unit-slope", "all-non-finite", "at-the-noise-floor"],
)
def test_calibrated_learning_rate_sizes_the_first_step(
    optimizer, cost_fn, theta, expected
):
    """The gain makes the first update ``2π/10`` divided by the mean directional
    derivative; a derivative below ``1e-10`` (or none finite) leaves ``2π/10``."""
    gain = optimizer._calibrated_learning_rate(cost_fn, theta, np.random.default_rng(0))
    assert gain == pytest.approx(expected)


def test_blocking_band_is_calibrated_from_the_loss_noise():
    """``allowed_increase`` defaults to twice the loss's standard deviation at the
    starting point, so it admits shot noise and scales with the problem."""
    sigma = 0.3
    rng = np.random.default_rng(1997)

    def noisy(params):
        rows = np.atleast_2d(params)
        values = np.sum(rows**2, axis=1) + rng.normal(0.0, sigma, size=rows.shape[0])
        return values if rows.shape[0] > 1 else float(values[0])

    optimizer = SPSAOptimizer(blocking=True, calibration_steps=400)

    band = optimizer._calibrated_allowed_increase(noisy, np.zeros(3))

    assert band == pytest.approx(2.0 * sigma, rel=0.15)


def test_explicit_allowed_increase_is_used_verbatim():
    """A supplied band skips calibration entirely."""
    optimizer = SPSAOptimizer(blocking=True, allowed_increase=1.5)

    assert optimizer._calibrated_allowed_increase(
        _sphere, np.zeros(3)
    ) == pytest.approx(1.5)


def test_blocking_band_is_zero_when_calibration_losses_are_all_non_finite():
    optimizer = SPSAOptimizer(blocking=True, calibration_steps=4)

    assert optimizer._calibrated_allowed_increase(_nan_cost, np.zeros(3)) == 0.0


@pytest.mark.filterwarnings("ignore:.*blocking has rejected.*:UserWarning")
def test_blocking_prevents_divergence():
    """Look-ahead blocking keeps a would-be-divergent run bounded.

    A noisy/indefinite metric (mock fidelity) drives huge preconditioned steps;
    without blocking the iterate explodes (and warns), with blocking it stays bounded.
    """
    d = 30
    a_matrix = np.diag(np.linspace(1.0, 30.0, d))
    metric = np.diag(np.linspace(0.5, 3.0, d))
    quad = _quadratic(a_matrix)
    start = quad(np.ones(d))

    def peak(blocking):
        traj = []
        QNSPSAOptimizer(
            learning_rate=0.5,
            c=0.1,
            regularization=1e-3,
            blocking=blocking,
            max_step_norm=None,
        ).optimize(
            quad,
            initial_params=np.ones(d),
            max_iterations=150,
            fidelity_fn=_quadratic_fidelity(metric),
            callback_fn=lambda r: traj.append(r.fun[0]),
            rng=np.random.default_rng(0),
        )
        return max(traj)

    with pytest.warns(UserWarning, match="diverging"):
        diverged_peak = peak(blocking=False)  # diverges (and warns) without blocking
    assert diverged_peak > 100 * start
    assert peak(blocking=True) < 10 * start  # bounded with blocking


@pytest.mark.parametrize(
    "fun, reference, already_warned, expected",
    [
        (1000.0, 1.0, False, False),
        (1000.5, 1.0, False, True),
        (200.0, 10.0, False, False),
        (1500.0, 0.5, False, True),
        (np.nan, 1.0, False, True),
        (1e9, 1.0, True, True),
    ],
    ids=[
        "at-threshold",
        "past-threshold",
        "threshold-scales-with-reference",
        "reference-floored-at-one",
        "non-finite",
        "already-warned",
    ],
)
def test_divergence_warning_fires_past_a_thousandfold_rise(
    recwarn, fun, reference, already_warned, expected
):
    returned = SPSAOptimizer()._warn_if_diverging(fun, reference, already_warned)
    assert returned is expected
    assert count_divergence_warnings(recwarn) == int(expected and not already_warned)


def test_divergence_warning_message():
    message = (
        "SPSAOptimizer appears to be diverging (loss 1.000e+00 -> 1.000e+09); "
        "best_loss/best_params may reflect an early iterate. Enable blocking, "
        "raise regularization, or lower learning_rate."
    )
    with pytest.warns(UserWarning, match=exact_match(message)):
        SPSAOptimizer()._warn_if_diverging(1e9, 1.0, False)


def test_spsa_resamplings_averages_over_extra_samples():
    """resamplings=N issues N gradient batches per step (one cost_fn call each)."""
    calls, counting = counting_cost_fn()

    # An explicit learning_rate skips calibration, which would add its own evals.
    SPSAOptimizer(learning_rate=0.2, resamplings=2).optimize(
        counting,
        initial_params=np.array([1.0, 1.0]),
        max_iterations=3,
        rng=np.random.default_rng(0),
    )
    assert calls["n"] == 2 * 3  # resamplings × steps (exact_loss off → no extra)


def test_spsa_exact_loss_records_unperturbed_value():
    """exact_loss spends one extra eval/step and records the true f(theta)."""
    calls, counting = counting_cost_fn()

    captured = []
    SPSAOptimizer(learning_rate=0.2, exact_loss=True).optimize(
        counting,
        initial_params=np.array([1.0, 1.0]),
        callback_fn=lambda res: captured.append((res.x.squeeze().copy(), res.fun[0])),
        max_iterations=3,
        rng=np.random.default_rng(0),
    )
    # (gradient batch + exact eval) per step, plus one for the final iterate.
    assert calls["n"] == 3 * 2 + 1
    for theta, fun in captured:
        assert fun == pytest.approx(_sphere(theta))  # exact, not the c²-biased proxy


def test_spsa_exact_loss_is_noop_under_blocking():
    """exact_loss adds no extra eval when blocking is on — blocking already
    carries the true f(theta) as the next step's loss."""

    def count_calls(exact_loss):
        calls, counting = counting_cost_fn()
        SPSAOptimizer(blocking=True, exact_loss=exact_loss).optimize(
            counting,
            initial_params=np.array([1.0, 1.0]),
            max_iterations=3,
            rng=np.random.default_rng(0),
        )
        return calls["n"]

    assert count_calls(exact_loss=True) == count_calls(exact_loss=False)


@pytest.mark.parametrize(
    "optimizer_cls, kwargs, message",
    [
        (
            SPSAOptimizer,
            {"learning_rate": 0.0},
            "learning_rate must be positive, got 0.0.",
        ),
        (SPSAOptimizer, {"c": -1.0}, "c must be positive, got -1.0."),
        (SPSAOptimizer, {"c": 0.0}, "c must be positive, got 0.0."),
        (SPSAOptimizer, {"resamplings": 0}, "resamplings must be >= 1, got 0."),
        (
            SPSAOptimizer,
            {"allowed_increase": -1.0},
            "allowed_increase must be non-negative, got -1.0.",
        ),
        (
            SPSAOptimizer,
            {"calibration_steps": 0},
            "calibration_steps must be >= 1, got 0.",
        ),
        (
            QNSPSAOptimizer,
            {"regularization": -1.0},
            "regularization must be non-negative, got -1.0.",
        ),
        (
            QNSPSAOptimizer,
            {"max_step_norm": 0.0},
            "max_step_norm must be positive or None, got 0.0.",
        ),
    ],
)
def test_constructor_validation(optimizer_cls, kwargs, message):
    with pytest.raises(ValueError, match=exact_match(message)):
        optimizer_cls(**kwargs)


@pytest.mark.parametrize(
    "optimizer_cls, kwargs",
    [
        (SPSAOptimizer, {"allowed_increase": 0.0}),
        (SPSAOptimizer, {"calibration_steps": 1}),
        (QNSPSAOptimizer, {"regularization": 0.0}),
    ],
)
def test_constructor_accepts_boundary_values(optimizer_cls, kwargs):
    optimizer = optimizer_cls(**kwargs)
    assert {name: getattr(optimizer, name) for name in kwargs} == kwargs


# --------------------------------------------------------------------------- #
# QN-SPSA optimizer
# --------------------------------------------------------------------------- #


def test_qnspsa_default_metric_is_stochastic_fidelity():
    assert isinstance(
        QNSPSAOptimizer().metric_estimator, StochasticFidelityMetricEstimator
    )


def test_qnspsa_requires_a_metric_evaluator():
    message = (
        "QNSPSAOptimizer requires a metric evaluator (`fidelity_fn` or "
        "`metric_fn`). It is driven by VariationalQuantumAlgorithm.run(), which "
        "supplies one via the metric estimator."
    )
    with pytest.raises(ValueError, match=exact_match(message)):
        QNSPSAOptimizer().optimize(
            _sphere, initial_params=np.zeros(2), max_iterations=3
        )


def test_qnspsa_fidelity_path_converges_with_quadratic_overlap_model():
    """A mock fidelity ``F(a,b)=1-½(a-b)ᵀM(a-b)`` makes the stochastic FS estimate
    track ``M/2``; QN-SPSA then converges on a quadratic cost."""
    a_matrix = np.array([[3.0, 0.0], [0.0, 1.0]])
    metric = np.array([[2.0, 0.3], [0.3, 1.0]])

    opt = QNSPSAOptimizer(learning_rate=0.3, c=0.15, regularization=1e-3)
    result = opt.optimize(
        _quadratic(a_matrix),
        initial_params=np.array([1.0, 1.0]),
        max_iterations=400,
        fidelity_fn=_quadratic_fidelity(metric),
        rng=np.random.default_rng(2),
    )
    assert result.fun[0] < 0.05


@pytest.mark.parametrize(
    "start, metric, seed, expected",
    [
        ([1.0], [[6.0]], 0, [[1.0], [0.5], [2.0 / 7.0]]),
        (
            [1.0, 0.5],
            np.diag([6.0, 2.0]),
            4,
            [[1.0, 0.5], [2.0 / 3.0, 1.0 / 6.0], [7.0 / 6.0, -4.0 / 3.0]],
        ),
    ],
    ids=["one-parameter", "two-parameter"],
)
def test_qnspsa_running_metric_average_keeps_the_identity_seed(
    start, metric, seed, expected
):
    """``ḡ_k = (k·ḡ_{k-1} + ĝ)/(k+1)`` from ``ḡ_0 = I``. With one parameter the
    quadratic overlap gives ``ĝ = M/2 = 3`` exactly, so ``ḡ`` runs 1 → 2 → 7/3
    against a sphere gradient of ``2θ``."""
    _, trace = callback_trace(
        QNSPSAOptimizer(learning_rate=0.5, **_CONSTANT_GAINS, **_UNDAMPED_QNSPSA),
        _sphere,
        start,
        3,
        fidelity_fn=_quadratic_fidelity(np.asarray(metric)),
        rng=np.random.default_rng(seed),
    )
    np.testing.assert_allclose([step.x[0] for step in trace], expected, atol=1e-12)


def test_qnspsa_exact_metric_preconditions_the_step():
    """``ĝ = 2`` at θ = 1 and ``g = 2``, so a unit gain steps by exactly 1."""
    _, trace = callback_trace(
        QNSPSAOptimizer(learning_rate=1.0, **_CONSTANT_GAINS, **_UNDAMPED_QNSPSA),
        _sphere,
        [1.0],
        2,
        metric_fn=lambda _theta: 2.0 * np.eye(1),
    )
    np.testing.assert_allclose(trace[1].x, [[0.0]], atol=1e-12)


def test_qnspsa_blocking_spends_one_fidelity_batch_per_step():
    """The fidelity path stays off ``cost_fn``: one overlap batch per step and
    resampling, blocking included."""
    cost_calls, counting_cost = counting_cost_fn()
    fid_calls = {"n": 0}

    def counting_fid(theta, perts):
        fid_calls["n"] += 1
        return np.ones(len(perts))

    QNSPSAOptimizer(learning_rate=0.01, blocking=True).optimize(
        counting_cost,
        initial_params=np.array([1.0, 1.0]),
        max_iterations=3,
        fidelity_fn=counting_fid,
        rng=np.random.default_rng(0),
    )
    assert cost_calls["n"] == 1 + 1 + 3 * (1 + 1)
    assert fid_calls["n"] == 3


def test_qnspsa_default_trust_region_bounds_a_near_singular_metric_step():
    """The near-singular metric inflates the natural-gradient step far past the
    trust region, so the update is rescaled to exactly ``max_step_norm``."""
    start = np.array([1.0, 0.3, -0.2, 0.7])
    optimizer = QNSPSAOptimizer(exact_loss=True, calibration_steps=8)

    _, trace = callback_trace(
        optimizer,
        _sphere,
        start,
        2,
        metric_fn=lambda _theta: 1e-6 * np.eye(start.size),
    )

    assert optimizer.max_step_norm == pytest.approx(2.0 * np.pi / 10.0)
    step = trace[1].x - trace[0].x
    assert np.linalg.norm(step) == pytest.approx(optimizer.max_step_norm)


# --------------------------------------------------------------------------- #
# State-overlap primitive
# --------------------------------------------------------------------------- #


def _ry_rz_overlap_meta():
    params = [Parameter(f"t{i}") for i in range(4)]
    qc = QuantumCircuit(2)
    qc.ry(params[0], 0)
    qc.rz(params[1], 0)
    qc.ry(params[2], 1)
    qc.rz(params[3], 1)
    qc.cx(0, 1)
    cost = MetaCircuit(
        circuit_bodies=(((), circuit_to_dag(qc)),), parameters=tuple(params)
    )
    return qc, params, build_overlap_meta(cost)


def test_build_overlap_meta_shape():
    _, params, overlap = _ry_rz_overlap_meta()
    assert len(overlap.parameters) == 2 * len(params)
    assert overlap.measured_wires == (0, 1)
    assert overlap.observable is None


@pytest.mark.parametrize(
    "make_params, expected",
    [
        (
            lambda: tuple(ParameterVector("theta_uncompute", 2)),
            [
                "theta_uncompute[0]",
                "theta_uncompute[1]",
                "theta_uncompute_[0]",
                "theta_uncompute_[1]",
            ],
        ),
        (
            lambda: (
                ParameterVector("theta_uncompute", 1)[0],
                ParameterVector("_", 1)[0],
            ),
            [
                "theta_uncompute[0]",
                "_[0]",
                "theta_uncompute_[0]",
                "theta_uncompute_[1]",
            ],
        ),
    ],
    ids=["one-bump", "extends-until-free"],
)
def test_build_overlap_meta_moves_the_backward_prefix_off_ansatz_names(
    make_params, expected
):
    """``compose()`` rejects two distinct parameters sharing a name, so the
    backward namespace is extended until no ansatz parameter uses it."""
    params = make_params()
    qc = QuantumCircuit(1)
    qc.ry(params[0], 0)
    qc.rz(params[1], 0)
    cost = MetaCircuit(circuit_bodies=(((), circuit_to_dag(qc)),), parameters=params)

    overlap = build_overlap_meta(cost)

    assert [p.name for p in overlap.parameters] == expected


def _overlap_p_zero(qc, overlap, theta_fwd, theta_bwd):
    bound = dag_to_circuit(overlap.circuit_bodies[0][1]).assign_parameters(
        dict(zip(overlap.parameters, [*theta_fwd, *theta_bwd]))
    )
    return abs(Statevector.from_instruction(bound).data[0]) ** 2


def test_overlap_identical_params_is_one():
    qc, _, overlap = _ry_rz_overlap_meta()
    theta = np.random.default_rng(0).random(4)
    assert _overlap_p_zero(qc, overlap, theta, theta) == pytest.approx(1.0)


def test_overlap_orthogonal_states_is_zero():
    qc, _, overlap = _ry_rz_overlap_meta()
    # RY(π) vs RY(0) on qubit 0 produces orthogonal states.
    p0 = _overlap_p_zero(qc, overlap, [np.pi, 0, 0, 0], [0, 0, 0, 0])
    assert p0 == pytest.approx(0.0, abs=1e-9)


def test_overlap_matches_statevector_inner_product():
    qc, params, overlap = _ry_rz_overlap_meta()
    rng = np.random.default_rng(3)
    a, b = rng.random(4), rng.random(4)

    def state(theta):
        return Statevector.from_instruction(
            qc.assign_parameters(dict(zip(params, theta)))
        )

    expected = abs(state(a).inner(state(b))) ** 2
    assert _overlap_p_zero(qc, overlap, a, b) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Compatibility gate + end-to-end
# --------------------------------------------------------------------------- #


def test_qnspsa_accepts_composite_angle_ansatz(dummy_simulator, default_optimizer):
    """The fidelity metric only needs an invertible ansatz, so a composite angle
    (which the Fubini–Study metric rejects) is accepted here."""
    x = Parameter("x")
    qc = QuantumCircuit(1, 1)
    qc.rx(2 * x, 0)
    qc.measure(0, 0)
    program = CustomVQA(
        qscript=qc, backend=dummy_simulator, optimizer=default_optimizer
    )
    QNSPSAOptimizer().validate_program(program)  # must not raise


def test_qnspsa_rejects_data_bound_program(dummy_simulator, default_optimizer):
    program = data_bound_custom_vqa(
        dummy_simulator, default_optimizer, labels=[1.0, -1.0]
    )
    with pytest.raises(ContractViolation, match="data-bound"):
        QNSPSAOptimizer().validate_program(program)


def _one_qubit_meta(gates, order=None) -> MetaCircuit:
    """``gates`` is a sequence of ``(rotation, parameter name)``; ``order`` names
    the parameter order (defaults to first use)."""
    params = {name: Parameter(name) for _, name in gates}
    qc = QuantumCircuit(1)
    for rotation, name in gates:
        getattr(qc, rotation)(params[name], 0)
    names = order or list(params)
    return MetaCircuit(
        circuit_bodies=(((), circuit_to_dag(qc)),),
        parameters=tuple(params[name] for name in names),
    )


_RX_A_RY_B = (("rx", "a"), ("ry", "b"))


def _overlap_routine():
    program = SimpleNamespace(cost_circuit=_one_qubit_meta(_RX_A_RY_B))
    (routine,) = StochasticFidelityMetricEstimator().preprocessors(program)
    return routine


def test_overlap_routine_reads_probabilities_from_dag_bodies():
    routine = _overlap_routine()
    assert routine.name == METRIC_ROUTINE
    assert routine.result_format is ResultFormat.PROBS
    assert routine.consumes_dag_bodies is True
    assert routine.cache_key == METRIC_ROUTINE


def test_overlap_cache_is_keyed_on_ansatz_structure():
    """Rebuilt parameter objects with the same names reuse the cached overlap
    circuit; any change of gate, gate-parameter assignment or parameter order
    builds a new one."""
    overlap_for = _overlap_routine().preprocess
    reference = overlap_for(_one_qubit_meta(_RX_A_RY_B))

    assert overlap_for(_one_qubit_meta(_RX_A_RY_B)) is reference
    variants = [
        _one_qubit_meta((("rx", "a"), ("rz", "b"))),
        _one_qubit_meta((("rx", "b"), ("ry", "a")), order=["a", "b"]),
        _one_qubit_meta(_RX_A_RY_B, order=["b", "a"]),
    ]
    for variant in variants:
        overlap = overlap_for(variant)
        assert overlap is not reference
        assert overlap.parameters[: len(variant.parameters)] == variant.parameters


def test_overlap_cache_clears_once_full(monkeypatch):
    monkeypatch.setattr("divi.qprog._metrics._OVERLAP_CACHE_CAP", 2)
    overlap_for = _overlap_routine().preprocess
    metas = [_one_qubit_meta(((rotation, "a"),)) for rotation in ("rx", "ry", "rz")]

    first, _, last = (overlap_for(meta) for meta in metas)

    assert overlap_for(metas[0]) is not first
    assert overlap_for(metas[2]) is last


def test_fidelity_fn_pairs_theta_with_each_perturbed_point(monkeypatch):
    captured = {}

    def fake_run_overlap(_program, _routine, rows, zeros):
        captured["rows"], captured["zeros"] = rows, zeros
        return {0: 0.9, 1: 0.4}

    monkeypatch.setattr("divi.qprog._metrics._run_overlap", fake_run_overlap)
    program = SimpleNamespace(cost_circuit=_one_qubit_meta(_RX_A_RY_B))
    fidelity_fn = StochasticFidelityMetricEstimator().bind(program)["fidelity_fn"]

    overlaps = fidelity_fn(
        np.array([0.1, 0.2]), [np.array([0.5, 0.0]), np.array([0.0, -0.3])]
    )

    np.testing.assert_allclose(
        captured["rows"], [[0.1, 0.2, 0.6, 0.2], [0.1, 0.2, 0.1, -0.1]]
    )
    assert captured["zeros"] == "0"
    np.testing.assert_allclose(overlaps, [0.9, 0.4])


def test_stochastic_fidelity_rejects_a_non_invertible_ansatz():
    qc = QuantumCircuit(1)
    qc.rx(Parameter("a"), 0)
    qc.reset(0)
    program = SimpleNamespace(
        cost_circuit=MetaCircuit(
            circuit_bodies=(((), circuit_to_dag(qc)),), parameters=tuple(qc.parameters)
        )
    )
    message = (
        "The stochastic-fidelity metric requires an invertible ansatz (qiskit "
        "QuantumCircuit.inverse()); this program's cost circuit could not be "
        "inverted."
    )
    with pytest.raises(ContractViolation, match=exact_match(message)) as excinfo:
        StochasticFidelityMetricEstimator().check_compatible(program)
    assert excinfo.value.__cause__ is not None


@pytest.mark.e2e
def test_vqe_runs_under_spsa(toy_vqe):
    toy_vqe.backend.set_seed(1997)
    toy_vqe.optimizer = SPSAOptimizer(learning_rate=0.2, c=0.1)
    toy_vqe.max_iterations = 10
    toy_vqe.run(perform_final_computation=False)
    assert len(toy_vqe.losses_history) == 10
    assert np.isfinite(toy_vqe.best_loss)


@pytest.mark.e2e
def test_qnspsa_fidelity_fn_returns_valid_overlaps(toy_vqe):
    """The bound fidelity_fn runs the overlap pipeline on a real backend:
    identical params give overlap 1, perturbed params stay in [0, 1]."""
    toy_vqe.backend.set_seed(1997)
    fidelity_fn = StochasticFidelityMetricEstimator().bind(toy_vqe)["fidelity_fn"]
    n_params = toy_vqe.n_layers * toy_vqe.n_params_per_layer
    theta = np.linspace(0.1, 1.0, n_params)

    overlaps = fidelity_fn(theta, [np.zeros(n_params), 0.4 * np.ones(n_params)])
    assert overlaps[0] == pytest.approx(1.0)  # U·U† = identity → P(0ⁿ) = 1 exactly
    assert 0.0 <= overlaps[1] <= 1.0


def test_stochastic_fidelity_bind_does_not_advance_rng(toy_vqe):
    """Binding probes a structural flag via the pipeline env; it must not draw
    from the program RNG, which would shift QN-SPSA's seeded perturbations."""
    before = toy_vqe._rng.bit_generator.state
    StochasticFidelityMetricEstimator().bind(toy_vqe)
    assert toy_vqe._rng.bit_generator.state == before


@pytest.mark.e2e
def test_stochastic_fidelity_caches_overlap_circuit(toy_vqe, mocker):
    """A deterministic ansatz builds the overlap circuit once and reuses it
    across fidelity evaluations, and the cached circuit stays correct."""
    spy = mocker.spy(_metrics, "build_overlap_meta")
    fidelity_fn = StochasticFidelityMetricEstimator().bind(toy_vqe)["fidelity_fn"]
    n_params = toy_vqe.n_layers * toy_vqe.n_params_per_layer
    theta = np.linspace(0.1, 1.0, n_params)

    results = [
        fidelity_fn(theta, [np.zeros(n_params), 0.2 * np.ones(n_params)])
        for _ in range(3)
    ]

    assert spy.call_count == 1  # built once, then served from the cache
    # Zero perturbation is U·U† = identity → P(0ⁿ) = 1 exactly, every call: the
    # cached circuit is reused without being corrupted between evaluations.
    for overlaps in results:
        assert overlaps[0] == pytest.approx(1.0)


@pytest.mark.e2e
def test_stochastic_fidelity_uses_bounded_structural_cache_for_qdrift(
    dummy_simulator, mocker
):
    """A stochastic (QDrift) ansatz may resample, but overlap construction is
    still cached by the post-spec ansatz fingerprint rather than by VQA internals."""
    qaoa = QAOA(
        MaxCutProblem(nx.bull_graph()),
        n_layers=1,
        trotterization_strategy=QDrift(
            sampling_budget=2, n_hamiltonians_per_iteration=3, seed=42
        ),
        optimizer=QNSPSAOptimizer(learning_rate=0.1, c=0.15),
        max_iterations=2,
        backend=dummy_simulator,
    )
    fidelity_fn = StochasticFidelityMetricEstimator().bind(qaoa)["fidelity_fn"]
    n_params = len(qaoa.cost_circuit.parameters)
    theta = np.linspace(0.1, 1.0, n_params)

    spy = mocker.spy(_metrics, "build_overlap_meta")
    fidelity_fn(theta, [np.zeros(n_params)])
    after_first = spy.call_count
    fidelity_fn(theta, [np.zeros(n_params)])

    assert after_first >= 1
    assert spy.call_count >= after_first
    assert spy.call_count <= 2 * after_first


@pytest.mark.e2e
def test_vqe_runs_under_qnspsa_stochastic_fidelity(toy_vqe):
    toy_vqe.backend.set_seed(1997)
    toy_vqe.optimizer = QNSPSAOptimizer(learning_rate=0.1, c=0.15)
    toy_vqe.max_iterations = 5
    toy_vqe.run(perform_final_computation=False)
    assert len(toy_vqe.losses_history) == 5


@pytest.mark.e2e
def test_vqe_runs_under_qnspsa_exact_fubini_study(toy_vqe):
    toy_vqe.backend.set_seed(1997)
    toy_vqe.optimizer = QNSPSAOptimizer(
        learning_rate=0.1, c=0.15, metric_estimator=FubiniStudyMetricEstimator()
    )
    toy_vqe.max_iterations = 5
    toy_vqe.run(perform_final_computation=False)
    assert len(toy_vqe.losses_history) == 5


@pytest.mark.e2e
def test_qaoa_qdrift_qnspsa_runs_end_to_end(dummy_simulator):
    """QN-SPSA + QDrift completes: the fidelity metric is Hamiltonian-independent,
    so the sampled cohort does not affect the overlap measurement."""
    qaoa = QAOA(
        MaxCutProblem(nx.bull_graph()),
        n_layers=1,
        trotterization_strategy=QDrift(
            sampling_budget=2, n_hamiltonians_per_iteration=3, seed=42
        ),
        optimizer=QNSPSAOptimizer(learning_rate=0.1, c=0.15),
        max_iterations=2,
        backend=dummy_simulator,
    )
    qaoa.run()
    assert len(qaoa.losses_history) == 2
    assert qaoa.best_probs


@pytest.mark.e2e
def test_qaoa_qdrift_qnspsa_blocking_runs_end_to_end(dummy_simulator):
    """QN-SPSA + QDrift + blocking completes: each step's seed/candidate cost evals
    draw their own cohort (different from the gradient batch), so this exercises
    blocking's cross-cohort loss comparison end-to-end without error."""
    qaoa = QAOA(
        MaxCutProblem(nx.bull_graph()),
        n_layers=1,
        trotterization_strategy=QDrift(
            sampling_budget=2, n_hamiltonians_per_iteration=3, seed=42
        ),
        optimizer=QNSPSAOptimizer(learning_rate=0.1, c=0.15, blocking=True),
        max_iterations=3,
        backend=dummy_simulator,
    )
    qaoa.run()
    assert len(qaoa.losses_history) == 3
    assert qaoa.best_probs
