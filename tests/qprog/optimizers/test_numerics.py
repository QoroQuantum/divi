# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Numerical-accuracy tests for the noise-robust optimizers.

Every test here has a closed-form oracle and needs no quantum backend beyond the
metric estimators' own measurement seam, so a failure points at the optimizer's
arithmetic rather than at sampling noise. The three seams under test are the
linear-algebra helpers (:mod:`divi.qprog.optimizers._linalg`), the stochastic
Fubini-Study sampler that drives QN-SPSA, and the metric assembly in
:mod:`divi.qprog._metrics`.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.converters import circuit_to_dag
from scipy.linalg import hilbert

from divi.circuits import MetaCircuit
from divi.qprog import CustomVQA, FubiniStudyMetricEstimator
from divi.qprog._metrics import PullbackMetricEstimator, _fs_blocks
from divi.qprog.optimizers import QNGOptimizer, QNSPSAOptimizer
from divi.qprog.optimizers._linalg import _matrix_abs_psd, _regularized_solve
from divi.qprog.optimizers._spsa import _fidelity_metric_sample, _spsa_gain_c
from tests._helpers import exact_match
from tests.qprog.optimizers._helpers import (
    inject_term_expectations,
    set_unit_shift_rule,
)


def _solve(metric, grad, *, solver="tikhonov", regularization=0.0, scaled=False):
    return _regularized_solve(
        grad,
        metric,
        solver=solver,
        regularization=regularization,
        scale_regularization=scaled,
        rcond=1e-6,
    )


#: Rank-1, so the damped solve's null direction is what ``regularization`` sets.
_SINGULAR_METRIC = np.outer([1.0, 0.0], [1.0, 0.0])


def _run_qng_with(optimizer, metric, grad, max_iterations=3):
    """Drive ``optimize`` with a fixed metric and gradient, so the overshoot
    guard sees exactly the pair under test rather than a moving iterate."""
    return optimizer.optimize(
        cost_fn=lambda x: float(np.sum(np.asarray(x) ** 2)),
        initial_params=np.ones(metric.shape[0]),
        max_iterations=max_iterations,
        jac=lambda _theta: grad,
        metric_fn=lambda _theta: metric,
    )


def _overshoot_warnings(records):
    return [w for w in records if "more than pi/4" in str(w.message)]


def _psd_margin(metric: np.ndarray) -> float:
    """Most-negative eigenvalue relative to the matrix scale, since an absolute
    floor is meaningless for a metric whose trace spans many decades."""
    return float(np.linalg.eigvalsh(metric).min() / np.trace(metric))


def _bind_pullback(vqe, jacobian, coeffs, monkeypatch):
    """Bind the pullback estimator to ``jacobian``'s parameter-shift pairs, so
    only the assembly arithmetic is exercised."""
    shifted = np.empty((2 * jacobian.shape[0], jacobian.shape[1]))
    shifted[0::2] = jacobian
    shifted[1::2] = -jacobian
    set_unit_shift_rule(vqe)
    inject_term_expectations(monkeypatch, lambda _param_sets: (shifted, coeffs))
    return PullbackMetricEstimator().bind(vqe)


def _product_ry_fidelity(_theta, perturbations):
    """Exact fidelity of a product of single-qubit RY rotations.

    ``F(theta, theta + d) = prod_i cos^2(d_i / 2)``, whose Fubini-Study metric is
    exactly ``I / 4`` — the oracle the stochastic sampler must reproduce.
    """
    return np.array(
        [
            np.prod(np.cos(np.asarray(p, dtype=np.float64) / 2.0) ** 2)
            for p in perturbations
        ]
    )


def _noisy_product_ry_fidelity(sigma, rng):
    def fidelity_fn(theta, perturbations):
        values = _product_ry_fidelity(theta, perturbations)
        return np.clip(values + rng.normal(0.0, sigma, size=values.shape), 0.0, 1.0)

    return fidelity_fn


def _mean_fidelity_metric(fidelity_fn, n_params, c_k, rng, n_samples):
    accumulator = np.zeros((n_params, n_params))
    theta = np.zeros(n_params)
    for _ in range(n_samples):
        h1 = rng.choice([-1.0, 1.0], size=n_params)
        accumulator += _fidelity_metric_sample(fidelity_fn, theta, h1, c_k, rng)
    return accumulator / n_samples


# --- _matrix_abs_psd under ill conditioning --------------------------------- #


@pytest.mark.parametrize("n", [4, 8, 12])
def test_matrix_abs_psd_reconstructs_ill_conditioned_psd_matrix(n):
    """On an already-PSD input the matrix absolute value is the identity map, and
    stays so across twelve decades of conditioning (Hilbert matrices)."""
    matrix = hilbert(n)
    result = _matrix_abs_psd(matrix)
    np.testing.assert_allclose(result, matrix, atol=1e-14)
    np.testing.assert_allclose(result, result.T, atol=1e-15)


def test_matrix_abs_psd_output_is_psd_only_to_round_off():
    """The reconstruction ``V |L| V^T`` reintroduces round-off, so the result is
    PSD only relative to the matrix scale, not to an absolute floor."""
    matrix = hilbert(16)
    result = _matrix_abs_psd(matrix)
    scale = float(np.linalg.norm(matrix, 2))
    assert _psd_margin(result) >= -1e-12
    assert np.linalg.eigvalsh(result).min() >= -1e-14 * scale


@pytest.mark.parametrize("eigenvalue", [1e-14, -1e-14, -1e-8, -1e-2, 0.0])
def test_matrix_abs_psd_lifts_eigenvalues_of_every_magnitude(eigenvalue):
    """Sign flipping is magnitude-independent: a metric estimate whose noise has
    driven an eigenvalue negative comes back at the same magnitude, positive."""
    result = _matrix_abs_psd(np.diag([1.0, eigenvalue]))
    np.testing.assert_allclose(
        np.sort(np.linalg.eigvalsh(result)),
        np.sort([1.0, abs(eigenvalue)]),
        atol=1e-16,
    )


# --- _regularized_solve on indefinite and rank-deficient metrics ------------ #


@pytest.mark.parametrize("regularization", [0.0, 1e-6, 1e-4])
def test_regularized_solve_rejects_underdamped_indefinite_metric(regularization):
    """A shot-noisy Fubini-Study estimate is indefinite, not merely singular. With
    damping below the negative eigenvalue the Cholesky solve must surface an
    actionable error rather than return a direction that ascends the loss."""
    indefinite = np.diag([1.0, -1e-3])
    with pytest.raises(np.linalg.LinAlgError, match="not positive-definite"):
        _solve(indefinite, np.array([1.0, 1.0]), regularization=regularization)


def test_regularized_solve_absorbs_indefinite_metric_once_damping_dominates():
    """Damping above the negative eigenvalue restores definiteness, and the
    returned step solves the damped system."""
    indefinite = np.diag([1.0, -1e-3])
    grad = np.array([1.0, 1.0])
    delta = _solve(indefinite, grad, regularization=1e-2)

    damped = indefinite + 1e-2 * np.eye(2)
    np.testing.assert_allclose(damped @ delta, grad, atol=1e-10)


def test_regularized_solve_pinv_survives_an_indefinite_metric():
    """``pinv`` has no definiteness precondition, so it is the escape hatch the
    tikhonov error message points at."""
    delta = _solve(
        np.diag([1.0, -1e-3]), np.array([1.0, 1.0]), solver="pinv", regularization=0.0
    )
    assert np.all(np.isfinite(delta))


@pytest.mark.parametrize("regularization", [1e-12, 1e-8, 1e-4])
def test_regularized_solve_step_on_singular_metric_scales_as_inverse_damping(
    regularization,
):
    """On a rank-deficient metric the damped solve is backward stable but the step
    along the null direction grows as ``1 / regularization``. The solve therefore
    cannot be relied on to signal under-damping: it succeeds and returns an
    enormous, exactly-computed direction."""
    grad = np.array([1.0, 1.0])
    delta = _solve(_SINGULAR_METRIC, grad, regularization=regularization)

    damped = _SINGULAR_METRIC + regularization * np.eye(2)
    residual = np.linalg.norm(damped @ delta - grad) / np.linalg.norm(grad)
    assert residual < 1e-12
    assert delta[1] == pytest.approx(1.0 / regularization, rel=1e-9)


def test_qng_step_on_singular_metric_is_unbounded_without_max_step_norm():
    """The finiteness check cannot catch the blow-up above, so a small
    ``regularization`` on a singular metric yields a finite but enormous update.
    Only ``max_step_norm`` bounds it."""
    grad = np.array([1.0, 1.0])

    unclipped = QNGOptimizer(
        step_size=1.0, regularization=1e-12, scale_regularization=False
    )
    delta = unclipped._natural_gradient(grad, _SINGULAR_METRIC)
    assert np.all(np.isfinite(delta))
    assert np.linalg.norm(delta) > 1e11

    clipped = QNGOptimizer(
        step_size=1.0,
        regularization=1e-12,
        scale_regularization=False,
        max_step_norm=1.0,
    )
    clipped_delta = clipped._natural_gradient(grad, _SINGULAR_METRIC)
    assert np.linalg.norm(clipped.step_size * clipped_delta) == pytest.approx(1.0)


@pytest.mark.filterwarnings("ignore:.*more than pi/4:UserWarning")
def test_qng_reports_a_non_finite_gradient_actionably():
    """Divergence must surface as the documented :class:`FloatingPointError`,
    naming the optimizer and the remedies, not as a bare error from scipy's solve
    about an array containing infs or NaNs."""
    optimizer = QNGOptimizer(step_size=0.1, regularization=1e-2)

    # Rosenbrock at step_size 0.1 diverges to overflow within a few steps.
    def rosenbrock(params):
        x = np.atleast_2d(np.asarray(params, dtype=np.float64))
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.sum(
                100.0 * (x[:, 1:] - x[:, :-1] ** 2) ** 2 + (1.0 - x[:, :-1]) ** 2,
                axis=1,
            )
        return values if x.shape[0] > 1 else float(values[0])

    def gradient(theta):
        x = np.asarray(theta, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            grad = np.zeros_like(x)
            diff = x[1:] - x[:-1] ** 2
            grad[:-1] += -400.0 * x[:-1] * diff - 2.0 * (1.0 - x[:-1])
            grad[1:] += 200.0 * diff
        return grad

    message = (
        "QNGOptimizer received a non-finite gradient or metric. The iterate has "
        "diverged, or the cost function returned a non-finite value. Lower "
        "`step_size`, raise `regularization`, or set `max_step_norm`."
    )
    with pytest.raises(FloatingPointError, match=exact_match(message)):
        optimizer.optimize(
            rosenbrock,
            np.full(4, 0.5),
            max_iterations=40,
            jac=gradient,
            metric_fn=lambda _theta: np.eye(4),
        )


@pytest.mark.parametrize(
    "metric",
    [_SINGULAR_METRIC, 1e-2 * np.eye(2)],
    ids=["under-damped-singular", "small-against-the-gradient"],
)
def test_qng_warns_once_on_an_oversized_step(metric):
    """A tiny damping on a singular metric, or a well-conditioned metric small
    against the gradient (the H2 overshoot), passes the finiteness check, so the
    run reports the oversized step itself — once, not once per step."""
    optimizer = QNGOptimizer(
        step_size=1.0, regularization=1e-12, scale_regularization=False
    )
    with pytest.warns(UserWarning, match="more than pi/4") as rec:
        _run_qng_with(optimizer, metric, np.ones(2))

    assert len(rec) == 1


@pytest.mark.parametrize(
    "settings, metric",
    [
        (
            {"step_size": 1.0, "regularization": 1e-12, "max_step_norm": 0.5},
            _SINGULAR_METRIC,
        ),
        ({"step_size": 0.5, "regularization": 1e-9}, np.diag([3.0, 1.0])),
        ({"step_size": 0.5, "regularization": 1e-9}, 1e8 * np.eye(2)),
    ],
    ids=["clipped-by-max_step_norm", "well-conditioned", "large-metric"],
)
def test_qng_bounded_steps_do_not_warn(recwarn, settings, metric):
    optimizer = QNGOptimizer(scale_regularization=False, **settings)
    _run_qng_with(optimizer, metric, np.ones(2))

    assert _overshoot_warnings(recwarn) == []


# --- Pullback metric: coefficient dynamic range ----------------------------- #


@pytest.mark.parametrize("half_decades", [0, 2, 4])
def test_pullback_metric_is_psd_relative_to_its_own_scale(
    injectable_vqe, monkeypatch, half_decades
):
    """Across eight decades of Hamiltonian coefficients the assembled metric stays
    symmetric and PSD *relative to its trace*. An absolute eigenvalue floor is not
    a meaningful check here — the trace itself moves by sixteen decades."""
    n_params = injectable_vqe.n_layers * injectable_vqe.n_params_per_layer
    n_terms = 3
    jacobian = np.random.default_rng(0).standard_normal((n_params, n_terms))
    coeffs = np.logspace(-half_decades, half_decades, n_terms)

    metric = _bind_pullback(injectable_vqe, jacobian, coeffs, monkeypatch)["metric_fn"](
        np.zeros(n_params)
    )

    np.testing.assert_allclose(metric, metric.T, rtol=0, atol=1e-12 * np.trace(metric))
    assert _psd_margin(metric) >= -1e-12


def test_pullback_metric_condition_number_squares_the_coefficient_range(
    injectable_vqe, monkeypatch
):
    """``G = (J * c^2) J^T / ||c||^2`` squares the coefficient dynamic range:
    widening the coefficients by two decades costs four decades of condition
    number."""
    n_params = injectable_vqe.n_layers * injectable_vqe.n_params_per_layer
    n_terms = 4
    jacobian = np.random.default_rng(1).standard_normal((n_params, n_terms))

    conditions = {}
    for half_decades in (1, 2):
        coeffs = np.logspace(-half_decades, half_decades, n_terms)
        metric = _bind_pullback(injectable_vqe, jacobian, coeffs, monkeypatch)[
            "metric_fn"
        ](np.zeros(n_params))
        conditions[half_decades] = np.linalg.cond(metric)

    growth = conditions[2] / conditions[1]
    assert 1e3 < growth < 1e5


@pytest.mark.parametrize("scale", [1e-3, 1e3])
def test_pullback_metric_is_unchanged_by_rescaling_the_hamiltonian(
    injectable_vqe, monkeypatch, scale
):
    n_params = injectable_vqe.n_layers * injectable_vqe.n_params_per_layer
    jacobian = np.random.default_rng(2).standard_normal((n_params, 3))
    coeffs = np.array([0.5, -0.3, 0.2])

    def metric_for(c):
        return _bind_pullback(injectable_vqe, jacobian, c, monkeypatch)["metric_fn"](
            np.zeros(n_params)
        )

    np.testing.assert_allclose(metric_for(scale * coeffs), metric_for(coeffs))


# --- Fubini-Study metric on a degenerate ansatz ----------------------------- #


def test_fs_blocks_emits_one_entry_per_gate_for_a_shared_parameter():
    """Two gates driven by the *same* parameter land in one block as two entries
    carrying the same parameter index."""
    theta = Parameter("t")
    circuit = QuantumCircuit(2)
    circuit.ry(theta, 0)
    circuit.ry(theta, 1)
    meta = MetaCircuit(
        circuit_bodies=(((), circuit_to_dag(circuit)),), parameters=(theta,)
    )

    blocks, full_params, _ = _fs_blocks(meta)

    assert len(full_params) == 1
    assert len(blocks) == 1
    indices = [index for index, _ in blocks[0][1]]
    assert indices == [0, 0]


def test_fubini_study_metric_for_a_shared_parameter_sums_both_generators(
    default_test_simulator, default_optimizer
):
    """A parameter driving two gates has generator ``K = K_0 + K_1``, so its metric
    entry is ``Var(K_0 + K_1)``. For ``RY(t)`` on two fresh qubits that is
    ``1/4 + 1/4 = 1/2``, not the ``1/4`` of a single generator."""
    theta = Parameter("t")
    circuit = QuantumCircuit(2, 2)
    circuit.ry(theta, 0)
    circuit.ry(theta, 1)
    circuit.measure(range(2), range(2))

    program = CustomVQA(
        qscript=circuit, backend=default_test_simulator, optimizer=default_optimizer
    )
    program.backend.set_seed(1997)

    metric = FubiniStudyMetricEstimator().bind(program)["metric_fn"](np.array([0.3]))

    assert metric.shape == (1, 1)
    assert metric[0, 0] == pytest.approx(0.5, abs=1e-10)


def test_fubini_study_metric_accumulates_a_parameter_across_blocks(
    default_test_simulator, default_optimizer
):
    """The same holds when the shared parameter's gates fall in *different* blocks:
    the block-diagonal metric drops the inter-block correlation but keeps each
    block's own contribution. Both pre-states here have real amplitudes, so
    ``<Y> = 0`` and each block contributes ``Var(Y/2) = 1/4``."""
    theta = Parameter("t")
    circuit = QuantumCircuit(1, 1)
    circuit.ry(theta, 0)
    circuit.x(0)  # non-parametric gate closes the block
    circuit.ry(theta, 0)
    circuit.measure(0, 0)

    program = CustomVQA(
        qscript=circuit, backend=default_test_simulator, optimizer=default_optimizer
    )
    program.backend.set_seed(1997)

    metric = FubiniStudyMetricEstimator().bind(program)["metric_fn"](np.array([0.7]))

    assert metric.shape == (1, 1)
    assert metric[0, 0] == pytest.approx(0.5, abs=1e-10)


# --- Stochastic Fubini-Study sampler (QN-SPSA) ------------------------------ #


def test_fidelity_metric_sample_recovers_the_exact_fubini_study_metric():
    """Averaged over Rademacher directions the sampler is unbiased: for a product
    of RY rotations it converges to the exact metric ``I / 4``."""
    n_params = 4
    rng = np.random.default_rng(0)
    estimate = _mean_fidelity_metric(
        _product_ry_fidelity, n_params, c_k=0.1, rng=rng, n_samples=2000
    )
    target = np.eye(n_params) / 4.0

    relative_error = np.linalg.norm(estimate - target) / np.linalg.norm(target)
    assert relative_error < 0.15


def test_fidelity_metric_sample_bias_grows_with_a_large_perturbation():
    """The estimator is a second difference, so an over-large ``c_k`` pays an
    ``O(c_k^2)`` bias even with an exact fidelity."""
    n_params = 4
    target = np.eye(n_params) / 4.0

    def error_at(c_k):
        rng = np.random.default_rng(7)
        estimate = _mean_fidelity_metric(
            _product_ry_fidelity, n_params, c_k=c_k, rng=rng, n_samples=2000
        )
        return np.linalg.norm(estimate - target) / np.linalg.norm(target)

    assert error_at(0.5) > 3 * error_at(0.1)


@pytest.mark.parametrize("sigma", [1e-4, 1e-3])
def test_fidelity_metric_sample_noise_is_amplified_as_the_perturbation_shrinks(sigma):
    """The ``1 / (8 c_k^2)`` prefactor turns fidelity shot noise into metric noise
    quadratically, so shrinking ``c_k`` past the noise scale destroys the estimate
    rather than refining it: once noise dominates, each decade of ``c_k`` costs two
    decades of error. This is the bound a user lowering ``c`` has to respect."""
    n_params = 4
    target = np.eye(n_params) / 4.0

    def error_at(c_k):
        rng = np.random.default_rng(11)
        estimate = _mean_fidelity_metric(
            _noisy_product_ry_fidelity(sigma, rng),
            n_params,
            c_k=c_k,
            rng=rng,
            n_samples=2000,
        )
        return np.linalg.norm(estimate - target) / np.linalg.norm(target)

    assert error_at(1e-1) < 0.25
    assert error_at(1e-3) > 10 * error_at(1e-2)
    # Deep in the noise-dominated regime the law is the bare 1/c_k^2 amplification.
    assert 30.0 < error_at(1e-4) / error_at(1e-3) < 300.0


def test_qnspsa_fidelity_path_survives_a_failed_overlap_evaluation():
    """The stochastic-fidelity branch end to end: a nan overlap must not poison
    the metric, and the run must still descend."""
    n_params = 4
    rng = np.random.default_rng(3)

    def flaky_fidelity(_theta, perturbations):
        values = _product_ry_fidelity(_theta, perturbations)
        if rng.random() < 0.1:
            values = np.full_like(values, np.nan)
        return values

    # The fidelity model's metric is I/4, so the preconditioner scales the step by ~4.
    optimizer = QNSPSAOptimizer(learning_rate=0.05, c=0.2)
    start = np.full(n_params, 1.0)
    result = optimizer.optimize(
        lambda params: (
            np.sum(np.atleast_2d(params) ** 2, axis=1)
            if np.ndim(params) > 1
            else float(np.sum(np.asarray(params) ** 2))
        ),
        start,
        max_iterations=40,
        fidelity_fn=flaky_fidelity,
        rng=rng,
    )

    assert result.success
    assert np.all(np.isfinite(np.ravel(result.x)))
    assert float(np.sum(np.ravel(result.x) ** 2)) < 0.5 * float(np.sum(start**2))


def test_default_qnspsa_perturbation_schedule_stays_out_of_the_noise_regime():
    """Spall's ``gamma = 0.101`` decays ``c_k`` so slowly that a thousand
    iterations from the default ``c = 0.2`` still leave it above 0.09, clear of
    the noise-amplified regime above."""
    assert _spsa_gain_c(0, 0.2, 0.101) == pytest.approx(0.2)
    assert _spsa_gain_c(999, 0.2, 0.101) > 0.09
