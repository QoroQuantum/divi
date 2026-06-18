# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the QUIVER optimizer (forward gradients, arXiv 2606.09734) and the
per-evaluation shot-budget + measurement-variance channel that powers its
adaptive ``M`` allocation."""

import networkx as nx
import numpy as np
import pytest
from qiskit.circuit.library import RYGate, RZGate
from qiskit.quantum_info import SparsePauliOp

from divi.backends import QiskitSimulator
from divi.qprog import QAOA, VQE, QUIVEROptimizer
from divi.qprog.algorithms import GenericLayerAnsatz
from divi.qprog.optimizers._spsa import _cost_fn_supports_variance, _spsa_gradient
from divi.qprog.problems import MaxCutProblem
from tests.qprog.optimizers._contracts import sphere_cost_fn_batch_aware as _sphere

# --------------------------------------------------------------------------- #
# Forward-gradient optimizer
# --------------------------------------------------------------------------- #


def test_quiver_forward_gradient_recovers_gradient():
    """Averaging V Rademacher directional derivatives converges to ∇f. On the
    sphere ∇f = 2θ; the V→large forward-gradient estimate approaches it."""
    theta = np.array([0.5, -1.0, 2.0])
    rng = np.random.default_rng(0)
    ghats = [_spsa_gradient(_sphere, theta, 0.1, rng)[0] for _ in range(2000)]
    estimate = np.mean(ghats, axis=0)
    np.testing.assert_allclose(estimate, 2.0 * theta, atol=0.1)


def test_quiver_converges_on_sphere():
    opt = QUIVEROptimizer(learning_rate=0.3, epsilon=0.1, V_init=2, V_max=8)
    result = opt.optimize(
        _sphere,
        initial_params=np.array([1.5, -1.2, 0.8]),
        max_iterations=200,
        rng=np.random.default_rng(0),
    )
    assert result.fun[0] < 0.05
    assert result.x.shape == (3,)
    assert result.success


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


def test_quiver_v_adaptivity_without_variance_handle():
    """V-adaptivity is driven by the spread of the V samples, so it works (and
    converges) even when cost_fn exposes no measurement-variance channel."""
    calls = {"n": 0}

    def counting(params):
        calls["n"] += 1
        return _sphere(params)

    result = QUIVEROptimizer(
        learning_rate=0.3, epsilon=0.1, V_init=2, V_min=1, V_max=10, adapt_V=True
    ).optimize(
        counting,
        initial_params=np.array([1.5, -1.0, 0.7]),
        max_iterations=60,
        rng=np.random.default_rng(0),
    )
    assert result.fun[0] < 0.1
    # Each step costs V_k cost calls (one two-row batch per direction); with
    # V_k >= 1 over 60 steps the run makes at least 60 calls. Whether V grows or
    # shrinks is landscape-dependent; that adaptivity *moves* V is asserted
    # separately in test_quiver_adapt_v_changes_evaluation_count.
    assert calls["n"] >= 60


def test_quiver_ignores_jac_and_metric_fn():
    result = QUIVEROptimizer(learning_rate=0.3, V_init=2).optimize(
        _sphere,
        initial_params=np.array([1.0, 1.0]),
        max_iterations=20,
        rng=np.random.default_rng(0),
        jac=lambda x: 1 / 0,  # never called
        metric_fn=lambda x: 1 / 0,
    )
    assert np.isfinite(result.fun[0])


def test_quiver_callback_receives_2d_x_and_1d_fun():
    captured = []
    QUIVEROptimizer(V_init=2).optimize(
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


def test_quiver_callback_stop_iteration_propagates():
    def callback(res):
        if res.nit == 2:
            raise StopIteration

    with pytest.raises(StopIteration):
        QUIVEROptimizer().optimize(
            _sphere,
            initial_params=np.array([1.0, 2.0]),
            callback_fn=callback,
            max_iterations=10,
            rng=np.random.default_rng(0),
        )


def test_quiver_requires_initial_params():
    with pytest.raises(ValueError, match="requires initial_params"):
        QUIVEROptimizer().optimize(_sphere, initial_params=None, max_iterations=3)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"learning_rate": 0.0}, "learning_rate must be positive"),
        ({"epsilon": -0.1}, "c must be positive"),
        ({"V_init": 0}, "resamplings must be >= 1"),
        ({"V_min": 2, "V_init": 1}, "V_min <= V_init <= V_max"),
        ({"V_max": 1, "V_init": 2}, "V_min <= V_init <= V_max"),
        ({"M_min": 5, "M_init": 1}, "M_min <= M_init <= M_max"),
        ({"mu": 1.0}, "mu must be in"),
        ({"lipschitz": 0.0}, "lipschitz must be positive"),
        ({"derivative_mode": "nope"}, "derivative_mode must be"),
    ],
)
def test_quiver_constructor_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        QUIVEROptimizer(**kwargs)


def test_quiver_does_not_support_checkpointing(tmp_path):
    opt = QUIVEROptimizer()
    assert opt.supports_checkpointing is False
    with pytest.raises(NotImplementedError):
        opt.get_config()
    with pytest.raises(NotImplementedError):
        opt.save_state(tmp_path)
    with pytest.raises(NotImplementedError):
        QUIVEROptimizer.load_state(tmp_path)
    opt.reset()  # no-op


def test_copy_preserves_quiver_config():
    opt = QUIVEROptimizer(
        learning_rate=0.05,
        epsilon=0.2,
        V_init=3,
        M_init=250,
        derivative_mode="parameter_shift",
    )
    clone = opt.copy()
    assert isinstance(clone, QUIVEROptimizer)
    assert clone.learning_rate == 0.05
    assert clone.epsilon == 0.2
    assert clone.V_init == 3
    assert clone.M_init == 250
    assert clone.derivative_mode == "parameter_shift"
    assert clone.adapt_V is True
    assert clone.adapt_M is True


def test_quiver_warns_once_when_step_leaves_stability_regime():
    """L*a_k >= 2 (here L=2 via lipschitz, a_k=1) leaves the gCANS regime and
    warns exactly once over the whole run, not once per step."""
    with pytest.warns(UserWarning, match="gCANS stability") as record:
        QUIVEROptimizer(learning_rate=1.0, lipschitz=2.0, V_init=2).optimize(
            _sphere,
            initial_params=np.array([1.0, 1.0]),
            max_iterations=3,
            rng=np.random.default_rng(0),
        )
    assert sum("gCANS stability" in str(w.message) for w in record) == 1


# --------------------------------------------------------------------------- #
# Integration with a shot-based VQE — the realistic QUIVER setting
# --------------------------------------------------------------------------- #


def _shot_based_vqe():
    """A VQE on a shot-based simulator, where the cost closure can expose a
    measurement-variance estimate and honour a per-evaluation shot budget."""
    return VQE(
        hamiltonian=SparsePauliOp.from_list([("ZI", 0.5), ("IZ", -0.3), ("XX", 0.2)]),
        ansatz=GenericLayerAnsatz([RYGate, RZGate]),
        n_layers=1,
        backend=QiskitSimulator(shots=4000, force_sampling=True),
        seed=1997,
    )


def test_quiver_runs_under_vqe():
    vqe = _shot_based_vqe()
    vqe.backend.set_seed(1997)
    vqe.optimizer = QUIVEROptimizer(learning_rate=0.2, epsilon=0.1, V_init=2)
    vqe.max_iterations = 8
    vqe.run(perform_final_computation=False)
    assert len(vqe.losses_history) == 8
    assert np.isfinite(vqe.best_loss)


# --------------------------------------------------------------------------- #
# Per-evaluation shot budget + measurement-variance channel
# --------------------------------------------------------------------------- #


def test_shots_override_threads_to_backend_without_mutation():
    """A per-evaluation ``shots`` budget reaches the backend as ``shot_groups``
    and never mutates the immutable backend's configured ``shots``."""
    vqe = _shot_based_vqe()
    vqe.backend.set_seed(7)
    theta = np.linspace(0.1, 1.0, vqe.n_layers * vqe.n_params_per_layer)

    captured = {}
    original = vqe.backend.submit_circuits

    def spy(circuits, **kwargs):
        captured["shot_groups"] = kwargs.get("shot_groups")
        return original(circuits, **kwargs)

    vqe.backend.submit_circuits = spy

    vqe._evaluate_cost_param_sets(theta[None, :], shots=512)
    # Every emitted [start, end, shots] triple carries the override budget.
    assert captured["shot_groups"] is not None
    assert all(triple[2] == 512 for triple in captured["shot_groups"])
    assert vqe.backend.shots == 4000  # unchanged


def test_cost_variance_is_positive_and_scales_inversely_with_shots():
    """The returned shot-noise variance is finite and positive, and shrinks ~1/M
    as the per-evaluation shot budget grows."""
    vqe = _shot_based_vqe()
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
    # 16× shots → variance drops ~16×; wide band because the estimate is itself
    # shot-noisy. The exact 1/M coefficient is pinned in
    # test_counts_to_cost_variance_matches_analytic_formula instead.
    assert 3.0 < var_low[0] / var_high[0] < 80.0


def test_cost_variance_is_nan_on_native_expval_backend():
    """On a native-expval backend no counts are produced, so the variance is nan
    and QUIVER falls back to fixed-M (V-from-spread only)."""
    vqe = VQE(
        hamiltonian=SparsePauliOp.from_list([("ZI", 0.5), ("IZ", -0.3), ("XX", 0.2)]),
        ansatz=GenericLayerAnsatz([RYGate, RZGate]),
        n_layers=1,
        backend=QiskitSimulator(shots=4000),  # analytic expval, no force_sampling
        seed=1997,
    )
    theta = np.linspace(0.1, 1.0, vqe.n_layers * vqe.n_params_per_layer)
    variances = vqe._cost_shot_variances(
        vqe._evaluate_cost_param_sets(theta[None, :], collect_variance=True)
    )
    assert np.isnan(variances[0])


def _counting_sphere():
    """A ``_sphere`` wrapper that counts how many times it is called."""
    calls = {"n": 0}

    def fn(params):
        calls["n"] += 1
        return _sphere(params)

    return calls, fn


def test_cost_fn_supports_variance_detection():
    """Capability sniffing: ``**kwargs`` or an explicit ``return_variance``
    parameter counts as supporting the variance channel; a plain callable
    does not."""
    assert _cost_fn_supports_variance(lambda x, **kwargs: x)
    assert _cost_fn_supports_variance(lambda x, return_variance=False: x)
    assert not _cost_fn_supports_variance(lambda x: x)


def test_quiver_adapt_v_changes_evaluation_count():
    """Adaptivity moves ``V``: an adaptive run spends a different number of cost
    evaluations than the fixed-``V`` run it would otherwise match."""
    calls_fixed, fn_fixed = _counting_sphere()
    QUIVEROptimizer(
        learning_rate=0.3, V_init=2, V_max=10, adapt_V=False, adapt_M=False
    ).optimize(
        fn_fixed,
        initial_params=np.array([1.5, -1.0, 0.7]),
        max_iterations=60,
        rng=np.random.default_rng(0),
    )
    calls_adaptive, fn_adaptive = _counting_sphere()
    QUIVEROptimizer(
        learning_rate=0.3, V_init=2, V_min=1, V_max=10, adapt_V=True, adapt_M=False
    ).optimize(
        fn_adaptive,
        initial_params=np.array([1.5, -1.0, 0.7]),
        max_iterations=60,
        rng=np.random.default_rng(0),
    )
    assert calls_fixed["n"] == 2 * 60  # fixed V_init=2 over 60 steps
    assert calls_adaptive["n"] != calls_fixed["n"]


def test_quiver_fixed_budget_converges_on_sphere():
    """With both adaptations off QUIVER is a fixed-V forward-gradient method and
    still converges."""
    result = QUIVEROptimizer(
        learning_rate=0.3, V_init=3, adapt_V=False, adapt_M=False
    ).optimize(
        _sphere,
        initial_params=np.array([1.5, -1.2, 0.8]),
        max_iterations=200,
        rng=np.random.default_rng(0),
    )
    assert result.fun[0] < 0.05


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
    """``exact_loss`` adds one unperturbed cost call per step on top of the
    ``V`` perturbation calls (V fixed at 2, adaptation off)."""
    calls_base, fn_base = _counting_sphere()
    QUIVEROptimizer(V_init=2, adapt_V=False, adapt_M=False, exact_loss=False).optimize(
        fn_base,
        initial_params=np.array([1.0, 1.0]),
        max_iterations=5,
        rng=np.random.default_rng(0),
    )
    calls_exact, fn_exact = _counting_sphere()
    QUIVEROptimizer(V_init=2, adapt_V=False, adapt_M=False, exact_loss=True).optimize(
        fn_exact,
        initial_params=np.array([1.0, 1.0]),
        max_iterations=5,
        rng=np.random.default_rng(0),
    )
    assert calls_base["n"] == 2 * 5
    assert calls_exact["n"] == calls_base["n"] + 5


def test_quiver_adapt_m_updates_shot_budget_to_backend():
    """The headline feature: with ``adapt_M`` the per-evaluation shot budget
    forwarded to the backend changes across iterations (the closed loop)."""
    vqe = _shot_based_vqe()
    vqe.backend.set_seed(11)
    vqe.optimizer = QUIVEROptimizer(
        learning_rate=0.2, epsilon=0.1, V_init=2, M_init=80, M_min=10, M_max=5000
    )
    vqe.max_iterations = 10

    seen_shots: list[int] = []
    original = vqe.backend.submit_circuits

    def spy(circuits, **kwargs):
        shot_groups = kwargs.get("shot_groups")
        if shot_groups:
            seen_shots.extend(triple[2] for triple in shot_groups)
        return original(circuits, **kwargs)

    vqe.backend.submit_circuits = spy
    vqe.run(perform_final_computation=False)
    # M starts at M_init and the adaptation moves it at least once.
    assert len(set(seen_shots)) >= 2


def test_quiver_adapt_m_warns_with_shot_distribution():
    """``adapt_M`` assumes uniform per-group shots; combining it with a shot
    distribution is flagged at program-validation time."""
    vqe = _shot_based_vqe()
    vqe._shot_distribution = "weighted"
    with pytest.warns(UserWarning, match="shot_distribution"):
        QUIVEROptimizer(adapt_M=True).validate_program(vqe)


def test_quiver_no_shot_distribution_warning_when_safe(recwarn):
    """No shot-distribution warning when ``adapt_M`` is off or no distribution
    is configured."""
    vqe = _shot_based_vqe()
    QUIVEROptimizer(adapt_M=True).validate_program(vqe)  # no distribution
    vqe._shot_distribution = "weighted"
    QUIVEROptimizer(adapt_M=False).validate_program(vqe)  # adaptation off
    assert not [w for w in recwarn if "shot_distribution" in str(w.message)]


def test_quiver_runs_under_qaoa_with_variance():
    """QAOA's cost is an expectation measurement, so the variance channel
    populates and M-adaptation works there as it does for VQE."""
    qaoa = QAOA(
        MaxCutProblem(nx.bull_graph()),
        n_layers=1,
        optimizer=QUIVEROptimizer(learning_rate=0.2, epsilon=0.1, V_init=2),
        max_iterations=5,
        backend=QiskitSimulator(shots=2000, force_sampling=True),
    )
    qaoa.run(perform_final_computation=False)
    assert len(qaoa.losses_history) == 5
    assert np.isfinite(qaoa.best_loss)
    assert qaoa._last_cost_variance is not None


def test_variance_nan_when_keys_collapse_across_extra_axes(mocker):
    """A pipeline with reduce axes beyond ``param_set`` (e.g. ZNE scales) yields
    several variance entries per parameter set; they cannot be collapsed to one
    scalar, so the variance is reported as nan and the optimizer falls back."""
    vqe = _shot_based_vqe()
    fake_result = {(("circuit", 0), ("param_set", 0)): np.array([0.5])}
    mocker.patch.object(vqe, "_run_pipeline", return_value=fake_result)
    vqe._last_cost_variance = {
        (("circuit", 0), ("zne_scale", 1), ("param_set", 0)): 0.01,
        (("circuit", 0), ("zne_scale", 3), ("param_set", 0)): 0.04,
    }
    theta = np.zeros((1, vqe.n_layers * vqe.n_params_per_layer))
    variances = vqe._cost_shot_variances(
        vqe._evaluate_cost_param_sets(theta, collect_variance=True)
    )
    assert np.isnan(variances[0])


def _analytic_vqe():
    """A VQE on an analytic backend — its cost pipeline promotes to the
    backend-native expval path (uses ham_ops, ignores shots)."""
    return VQE(
        hamiltonian=SparsePauliOp.from_list([("ZI", 0.5), ("IZ", -0.3), ("XX", 0.2)]),
        ansatz=GenericLayerAnsatz([RYGate, RZGate]),
        n_layers=1,
        backend=QiskitSimulator(shots=4000),
        seed=1997,
    )


def test_shots_override_is_ignored_on_analytic_expval_backend():
    """A per-evaluation shots override on the backend-native expval path is a
    no-op (analytic expval ignores shots), not a ham_ops/shot_groups crash."""
    vqe = _analytic_vqe()
    theta = np.linspace(0.1, 1.0, vqe.n_layers * vqe.n_params_per_layer)
    losses = vqe._evaluate_cost_param_sets(
        theta[None, :], shots=128, collect_variance=True
    )
    variances = vqe._cost_shot_variances(losses)
    assert np.isfinite(losses[0])
    assert np.isnan(variances[0])


def test_quiver_runs_on_analytic_backend():
    """QUIVER always sends a shots override; on an analytic backend that override
    must be silently ignored rather than crashing, with variance falling back to
    nan (fixed-M, V-from-spread only)."""
    vqe = _analytic_vqe()
    vqe.optimizer = QUIVEROptimizer(
        learning_rate=0.2, epsilon=0.1, V_init=2, adapt_M=True
    )
    vqe.max_iterations = 6
    vqe.run(perform_final_computation=False)
    assert len(vqe.losses_history) == 6
    assert np.isfinite(vqe.best_loss)
