# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SPSA and QN-SPSA optimizers and the state-overlap primitive."""

from collections import deque

import networkx as nx
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RYGate, RZGate
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import SparsePauliOp, Statevector

from divi.circuits import MetaCircuit, build_overlap_meta
from divi.hamiltonians import QDrift
from divi.pipeline.abc import ContractViolation
from divi.qprog import (
    QAOA,
    VQE,
    CustomVQA,
    FubiniStudyMetricEstimator,
    QNSPSAOptimizer,
    SPSAOptimizer,
)
from divi.qprog._metrics import StochasticFidelityMetricEstimator, _zeros_probability
from divi.qprog.algorithms import GenericLayerAnsatz
from divi.qprog.optimizers._linalg import _matrix_abs_psd
from divi.qprog.optimizers._spsa import (
    _fidelity_metric_sample,
    _spsa_gain_a,
    _spsa_gain_c,
    _spsa_gradient,
)
from divi.qprog.problems import MaxCutProblem
from tests.qprog.optimizers._contracts import sphere_cost_fn_batch_aware as _sphere

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


# --------------------------------------------------------------------------- #
# SPSA optimizer
# --------------------------------------------------------------------------- #


def test_spsa_converges_on_sphere():
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


def test_spsa_ignores_jac_and_metric_fn():
    """SPSA is gradient-free: a supplied jac/metric_fn must not break it."""
    opt = SPSAOptimizer(learning_rate=0.3, c=0.1)
    result = opt.optimize(
        _sphere,
        initial_params=np.array([1.0, 1.0]),
        max_iterations=20,
        rng=np.random.default_rng(0),
        jac=lambda x: 1 / 0,  # never called
        metric_fn=lambda x: 1 / 0,
    )
    assert np.isfinite(result.fun[0])


def test_spsa_callback_receives_2d_x_and_1d_fun():
    captured = []
    SPSAOptimizer().optimize(
        _sphere,
        initial_params=np.array([1.0, 2.0]),
        callback_fn=lambda res: captured.append((res.x, res.fun)),
        max_iterations=4,
        rng=np.random.default_rng(0),
    )
    assert len(captured) == 4
    for x, fun in captured:
        assert x.shape == (1, 2)
        assert fun.shape == (1,)


def test_spsa_callback_stop_iteration_propagates():
    def callback(res):
        if res.nit == 2:
            raise StopIteration

    with pytest.raises(StopIteration):
        SPSAOptimizer().optimize(
            _sphere,
            initial_params=np.array([1.0, 2.0]),
            callback_fn=callback,
            max_iterations=10,
            rng=np.random.default_rng(0),
        )


def test_block_or_step_rejects_worsening_candidate():
    """Look-ahead blocking accepts an improving candidate and rejects a worsening
    one (beyond the std band), holding the iterate."""
    opt = SPSAOptimizer(blocking=True, blocking_history=3, blocking_tol=2.0)
    recent = deque([1.0, 1.0, 1.0], maxlen=3)  # mean 1, std 0 -> band 0
    theta, proposed = np.array([0.0, 0.0]), np.array([1.0, 1.0])

    # candidate worsens the loss (sphere: 0 -> 2) beyond the band -> rejected, holds.
    nxt, loss = opt._block_or_step(
        _sphere, theta, proposed, current_loss=0.0, recent=recent
    )
    np.testing.assert_array_equal(nxt, theta)
    assert loss == 0.0

    # candidate improves the loss (2 -> 0) -> accepted, moves, loss updates.
    nxt, loss = opt._block_or_step(
        _sphere, proposed, theta, current_loss=2.0, recent=recent
    )
    np.testing.assert_array_equal(nxt, theta)
    assert loss == pytest.approx(0.0)


def test_block_or_step_startup_accepts_with_too_little_history():
    """With fewer than two prior losses the band is infinite, so any candidate is
    accepted (Spall start-up: need >=2 samples to estimate dispersion)."""
    opt = SPSAOptimizer(blocking=True, blocking_history=5, blocking_tol=2.0)
    theta, proposed = np.array([0.0]), np.array([10.0])  # catastrophically worse
    nxt, loss = opt._block_or_step(
        _sphere, theta, proposed, current_loss=0.0, recent=deque([9999.0], maxlen=5)
    )
    np.testing.assert_array_equal(nxt, proposed)  # accepted despite worsening
    assert loss == pytest.approx(_sphere(proposed))


def test_block_or_step_rejects_nonfinite_candidate():
    """A NaN candidate loss is held (not accepted) — without this guard NaN would
    slip through (`nan > x` is False) and poison the recent window."""
    opt = SPSAOptimizer(blocking=True, blocking_history=3, blocking_tol=2.0)
    recent = deque([1.0, 1.0, 1.0], maxlen=3)
    nxt, loss = opt._block_or_step(
        lambda x: float("nan"),
        np.array([0.0]),
        np.array([0.5]),
        current_loss=1.0,
        recent=recent,
    )
    np.testing.assert_array_equal(nxt, np.array([0.0]))  # held at theta
    assert loss == 1.0  # current_loss unchanged (window not poisoned)


def test_spsa_blocking_still_converges():
    """With blocking enabled the acceptance path stays active and SPSA converges."""
    opt = SPSAOptimizer(learning_rate=0.3, c=0.1, blocking=True)
    result = opt.optimize(
        _sphere,
        initial_params=np.array([1.5, -1.0]),
        max_iterations=200,
        rng=np.random.default_rng(0),
    )
    assert result.fun[0] < 0.1


def test_blocking_prevents_divergence():
    """Look-ahead blocking keeps a would-be-divergent run bounded.

    A noisy/indefinite metric (mock fidelity) drives huge preconditioned steps;
    without blocking the iterate explodes (and warns), with blocking it stays bounded.
    """
    d = 30
    a_matrix = np.diag(np.linspace(1.0, 30.0, d))
    metric = np.diag(np.linspace(0.5, 3.0, d))

    def quad(x):
        x = np.atleast_2d(x)
        v = 0.5 * np.einsum("ij,jk,ik->i", x, a_matrix, x)
        return v if x.shape[0] > 1 else float(v[0])

    def fid(theta, perts):
        return np.array([1.0 - 0.5 * (p @ metric @ p) for p in perts])

    start = quad(np.ones(d))

    def peak(blocking):
        traj = []
        QNSPSAOptimizer(
            learning_rate=0.5, c=0.1, regularization=1e-3, blocking=blocking
        ).optimize(
            quad,
            initial_params=np.ones(d),
            max_iterations=150,
            fidelity_fn=fid,
            callback_fn=lambda r: traj.append(r.fun[0]),
            rng=np.random.default_rng(0),
        )
        return max(traj)

    with pytest.warns(UserWarning, match="diverging"):
        diverged_peak = peak(blocking=False)  # diverges (and warns) without blocking
    assert diverged_peak > 100 * start
    assert peak(blocking=True) < 10 * start  # bounded with blocking


def test_diverging_run_emits_warning():
    """Without blocking, a divergent run warns once (best-tracking would otherwise
    silently return an early finite iterate)."""
    d = 20
    a_matrix = np.diag(np.linspace(1.0, 40.0, d))
    metric = np.diag(np.linspace(0.5, 4.0, d))

    def quad(x):
        x = np.atleast_2d(x)
        v = 0.5 * np.einsum("ij,jk,ik->i", x, a_matrix, x)
        return v if x.shape[0] > 1 else float(v[0])

    def fid(theta, perts):
        return np.array([1.0 - 0.5 * (p @ metric @ p) for p in perts])

    with pytest.warns(UserWarning, match="diverging"):
        QNSPSAOptimizer(learning_rate=0.5, c=0.1, regularization=1e-3).optimize(
            quad,
            initial_params=np.ones(d),
            max_iterations=120,
            fidelity_fn=fid,
            rng=np.random.default_rng(0),
        )


def test_spsa_diverging_run_emits_warning():
    """SPSA (no metric) warns once when a too-large step size makes a run diverge."""
    d = 20
    a_matrix = np.diag(np.linspace(1.0, 40.0, d))

    def quad(x):
        x = np.atleast_2d(x)
        v = 0.5 * np.einsum("ij,jk,ik->i", x, a_matrix, x)
        return v if x.shape[0] > 1 else float(v[0])

    with pytest.warns(UserWarning, match="diverging"):
        SPSAOptimizer(learning_rate=0.5, c=0.1).optimize(
            quad,
            initial_params=np.ones(d),
            max_iterations=120,
            rng=np.random.default_rng(0),
        )


def test_spsa_resamplings_averages_over_extra_samples():
    """resamplings=N issues N gradient batches per step (one cost_fn call each)."""
    calls = {"n": 0}

    def counting(params):
        calls["n"] += 1
        return _sphere(params)

    SPSAOptimizer(resamplings=2).optimize(
        counting,
        initial_params=np.array([1.0, 1.0]),
        max_iterations=3,
        rng=np.random.default_rng(0),
    )
    assert calls["n"] == 2 * 3  # resamplings × steps (exact_loss off → no extra)


def test_spsa_blocking_counts_initial_and_per_step_evals():
    """blocking issues one seed eval before the loop and one candidate eval/step."""
    calls = {"n": 0}

    def counting(params):
        calls["n"] += 1
        return _sphere(params)

    SPSAOptimizer(blocking=True).optimize(
        counting,
        initial_params=np.array([1.0, 1.0]),
        max_iterations=3,
        rng=np.random.default_rng(0),
    )
    # 1 seed + 3 steps × (1 gradient batch + 1 candidate eval) = 7
    assert calls["n"] == 1 + 3 * 2


def test_spsa_exact_loss_records_unperturbed_value():
    """exact_loss spends one extra eval/step and records the true f(theta)."""
    calls = {"n": 0}

    def counting(params):
        calls["n"] += 1
        return _sphere(params)

    captured = []
    SPSAOptimizer(exact_loss=True).optimize(
        counting,
        initial_params=np.array([1.0, 1.0]),
        callback_fn=lambda res: captured.append((res.x.squeeze().copy(), res.fun[0])),
        max_iterations=3,
        rng=np.random.default_rng(0),
    )
    assert calls["n"] == 3 * 2  # (gradient batch + exact eval) per step
    for theta, fun in captured:
        assert fun == pytest.approx(_sphere(theta))  # exact, not the c²-biased proxy


def test_spsa_exact_loss_is_noop_under_blocking():
    """exact_loss adds no extra eval when blocking is on — blocking already
    carries the true f(theta) as the next step's loss."""

    def count_calls(exact_loss):
        calls = {"n": 0}

        def counting(params):
            calls["n"] += 1
            return _sphere(params)

        SPSAOptimizer(blocking=True, exact_loss=exact_loss).optimize(
            counting,
            initial_params=np.array([1.0, 1.0]),
            max_iterations=3,
            rng=np.random.default_rng(0),
        )
        return calls["n"]

    assert count_calls(exact_loss=True) == count_calls(exact_loss=False)


def test_spsa_requires_initial_params():
    with pytest.raises(ValueError, match="requires initial_params"):
        SPSAOptimizer().optimize(_sphere, initial_params=None, max_iterations=3)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"learning_rate": 0.0}, "learning_rate must be positive"),
        ({"c": -1.0}, "c must be positive"),
        ({"resamplings": 0}, "resamplings must be >= 1"),
        ({"blocking_history": 0}, "blocking_history must be >= 1"),
    ],
)
def test_spsa_constructor_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SPSAOptimizer(**kwargs)


def test_spsa_does_not_support_checkpointing(tmp_path):
    opt = SPSAOptimizer()
    assert opt.supports_checkpointing is False
    with pytest.raises(NotImplementedError):
        opt.get_config()
    with pytest.raises(NotImplementedError):
        opt.save_state(tmp_path)
    with pytest.raises(NotImplementedError):
        SPSAOptimizer.load_state(tmp_path)
    opt.reset()  # no-op


def test_spsa_reset_allows_reuse():
    opt = SPSAOptimizer(learning_rate=0.3, c=0.1)
    r1 = opt.optimize(
        _sphere,
        initial_params=np.array([1.0, 1.0]),
        max_iterations=10,
        rng=np.random.default_rng(0),
    )
    opt.reset()
    r2 = opt.optimize(
        _sphere,
        initial_params=np.array([-1.0, 0.5]),
        max_iterations=10,
        rng=np.random.default_rng(1),
    )
    assert np.isfinite(r1.fun[0]) and np.isfinite(r2.fun[0])


def test_copy_preserves_spsa_config():
    opt = SPSAOptimizer(
        learning_rate=0.05,
        c=0.3,
        alpha=0.7,
        gamma=0.2,
        A=5.0,
        blocking=True,
        blocking_tol=3.0,
        resamplings=2,
    )
    clone = opt.copy()
    assert isinstance(clone, SPSAOptimizer)
    assert clone.learning_rate == 0.05
    assert clone.c == 0.3
    assert clone.alpha == 0.7
    assert clone.gamma == 0.2
    assert clone.A == 5.0
    assert clone.blocking is True
    assert clone.blocking_tol == 3.0
    assert clone.resamplings == 2


# --------------------------------------------------------------------------- #
# QN-SPSA optimizer
# --------------------------------------------------------------------------- #


def test_qnspsa_default_metric_is_stochastic_fidelity():
    assert isinstance(
        QNSPSAOptimizer().metric_estimator, StochasticFidelityMetricEstimator
    )


def test_qnspsa_requires_a_metric_evaluator():
    with pytest.raises(ValueError, match="requires a metric evaluator"):
        QNSPSAOptimizer().optimize(
            _sphere, initial_params=np.zeros(2), max_iterations=3
        )


def test_qnspsa_requires_initial_params():
    with pytest.raises(ValueError, match="requires initial_params"):
        QNSPSAOptimizer().optimize(
            _sphere,
            initial_params=None,
            max_iterations=3,
            metric_fn=lambda x: np.eye(2),
        )


def test_qnspsa_exact_metric_path_converges_on_quadratic():
    """With ``metric_fn`` supplying the exact Hessian, QN-SPSA preconditions the
    SPSA gradient and converges (SPSA gradient + exact metric hybrid)."""
    a_matrix = np.array([[3.0, 0.0], [0.0, 1.0]])
    opt = QNSPSAOptimizer(learning_rate=0.3, c=0.1, regularization=1e-6)
    result = opt.optimize(
        _quadratic(a_matrix),
        initial_params=np.array([1.0, 1.0]),
        max_iterations=300,
        metric_fn=lambda x: a_matrix,
        rng=np.random.default_rng(1),
    )
    assert result.fun[0] < 0.05
    assert result.x.shape == (2,)


def test_qnspsa_fidelity_path_converges_with_quadratic_overlap_model():
    """A mock fidelity ``F(a,b)=1-½(a-b)ᵀM(a-b)`` makes the stochastic FS estimate
    track ``M``; QN-SPSA then converges on a quadratic cost."""
    a_matrix = np.array([[3.0, 0.0], [0.0, 1.0]])
    metric = np.array([[2.0, 0.3], [0.3, 1.0]])

    def fidelity_fn(theta, perturbations):
        return np.array([1.0 - 0.5 * (p @ metric @ p) for p in perturbations])

    opt = QNSPSAOptimizer(learning_rate=0.3, c=0.15, regularization=1e-3)
    result = opt.optimize(
        _quadratic(a_matrix),
        initial_params=np.array([1.0, 1.0]),
        max_iterations=400,
        fidelity_fn=fidelity_fn,
        rng=np.random.default_rng(2),
    )
    assert result.fun[0] < 0.05


def test_qnspsa_callback_receives_2d_x_and_1d_fun():
    captured = []
    QNSPSAOptimizer().optimize(
        _sphere,
        initial_params=np.array([1.0, 2.0]),
        callback_fn=lambda res: captured.append((res.x, res.fun)),
        max_iterations=3,
        metric_fn=lambda x: np.eye(2),
        rng=np.random.default_rng(0),
    )
    assert len(captured) == 3
    for x, fun in captured:
        assert x.shape == (1, 2)
        assert fun.shape == (1,)


def test_qnspsa_constructor_rejects_negative_regularization():
    with pytest.raises(ValueError, match="regularization must be non-negative"):
        QNSPSAOptimizer(regularization=-1.0)


def test_qnspsa_does_not_support_checkpointing(tmp_path):
    opt = QNSPSAOptimizer()
    assert opt.supports_checkpointing is False
    with pytest.raises(NotImplementedError):
        opt.get_config()
    with pytest.raises(NotImplementedError):
        opt.save_state(tmp_path)
    with pytest.raises(NotImplementedError):
        QNSPSAOptimizer.load_state(tmp_path)


def test_qnspsa_blocking_counts_cost_and_fidelity_evals():
    """QN-SPSA + blocking: cost_fn = 1 seed + steps×(resamplings + 1 candidate);
    fidelity_fn = steps×resamplings (the metric path, separate from cost_fn)."""
    cost_calls = {"n": 0}
    fid_calls = {"n": 0}

    def counting_cost(params):
        cost_calls["n"] += 1
        return _sphere(params)

    def counting_fid(theta, perts):
        fid_calls["n"] += 1
        return np.ones(len(perts))

    QNSPSAOptimizer(blocking=True).optimize(
        counting_cost,
        initial_params=np.array([1.0, 1.0]),
        max_iterations=3,
        fidelity_fn=counting_fid,
        rng=np.random.default_rng(0),
    )
    assert cost_calls["n"] == 1 + 3 * (1 + 1)  # seed + steps×(grad batch + candidate)
    assert fid_calls["n"] == 3 * 1  # steps × resamplings


def test_copy_preserves_qnspsa_config():
    estimator = FubiniStudyMetricEstimator()
    opt = QNSPSAOptimizer(
        learning_rate=0.05, regularization=1e-2, metric_estimator=estimator
    )
    clone = opt.copy()
    assert isinstance(clone, QNSPSAOptimizer)
    assert clone.learning_rate == 0.05
    assert clone.regularization == 1e-2
    # copy() deep-copies, so the clone gets its own (stateless) estimator instance.
    assert isinstance(clone.metric_estimator, FubiniStudyMetricEstimator)
    assert clone is not opt
    assert clone.metric_estimator is not estimator  # independent copy, not shared


def test_copy_preserves_exact_loss_flag():
    assert SPSAOptimizer(exact_loss=True).copy().exact_loss is True
    assert QNSPSAOptimizer(exact_loss=True).copy().exact_loss is True


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


@pytest.fixture
def toy_vqe(default_test_simulator):
    return VQE(
        hamiltonian=SparsePauliOp.from_list([("ZI", 0.5), ("IZ", -0.3), ("XX", 0.2)]),
        ansatz=GenericLayerAnsatz([RYGate, RZGate]),
        n_layers=1,
        backend=default_test_simulator,
        seed=1997,
    )


def test_qnspsa_accepts_composite_angle_ansatz(dummy_simulator):
    """The fidelity metric only needs an invertible ansatz, so a composite angle
    (which the Fubini–Study metric rejects) is accepted here."""
    x = Parameter("x")
    qc = QuantumCircuit(1, 1)
    qc.rx(2 * x, 0)
    qc.measure(0, 0)
    program = CustomVQA(qscript=qc, backend=dummy_simulator)
    QNSPSAOptimizer().validate_program(program)  # must not raise


def test_qnspsa_rejects_data_bound_program(dummy_simulator):
    x = Parameter("x")
    w = Parameter("w")
    qc = QuantumCircuit(1, 1)
    qc.ry(x, 0)
    qc.rz(w, 0)
    qc.measure(0, 0)
    program = CustomVQA(
        qscript=qc,
        data_param_indices=[list(qc.parameters).index(x)],
        feature_batch=np.array([[0.1], [0.3]]),
        labels=[1.0, -1.0],
        backend=dummy_simulator,
    )
    with pytest.raises(ContractViolation, match="data-bound"):
        QNSPSAOptimizer().validate_program(program)


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
    assert qaoa.best_probs
