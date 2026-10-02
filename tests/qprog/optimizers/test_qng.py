# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Quantum Natural Gradient optimizer and its pullback metric."""

from dataclasses import replace
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RYGate
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import MetaCircuit
from divi.hamiltonians import QDrift
from divi.pipeline import CircuitPreprocessor
from divi.pipeline._result_keys_operations import average_by_param_set
from divi.pipeline.abc import ContractViolation
from divi.qprog import PCE, QAOA, VQE, CustomVQA, FubiniStudyMetricEstimator
from divi.qprog._metrics import (
    METRIC_ROUTINE,
    PullbackMetricEstimator,
    StochasticFidelityMetricEstimator,
    _all_terms_preprocessor,
    _fs_block_prefix,
    _fs_blocks,
    _fs_prefix_labels_preprocessor,
    _measure_prefix_paulis,
    _PrefixOpsView,
    _split_into_terms,
    _split_observable_into_terms,
    _term_expectations,
)
from divi.qprog.algorithms import GenericLayerAnsatz, HartreeFockAnsatz
from divi.qprog.checkpointing import CheckpointConfig
from divi.qprog.optimizers import QNGOptimizer
from divi.qprog.problems import (
    BinaryOptimizationProblem,
    HamiltonianProblem,
    MaxCutProblem,
    MolecularProblem,
)
from tests._helpers import exact_match
from tests.qprog.optimizers._helpers import (
    bowl_jac,
    bowl_metric,
    data_bound_custom_vqa,
    inject_term_expectations,
    set_unit_shift_rule,
)


def test_optimizer_contract(gradient_optimizer_contract):
    optimizer = QNGOptimizer(step_size=0.2)
    optimize_kwargs = {
        "jac": lambda x: 2 * x,
        "metric_fn": lambda x: np.eye(len(x)),
    }
    gradient_optimizer_contract(optimizer, optimize_kwargs)


def test_noisy_optimizer_contract(noisy_optimizer_contract):
    noisy_optimizer_contract(
        lambda: QNGOptimizer(step_size=0.1, regularization=1e-2, max_step_norm=1.0),
        {"jac": bowl_jac, "metric_fn": bowl_metric},
        ceiling=0.05,
    )


# --------------------------------------------------------------------------- #
# Optimizer numerics (no quantum backend)
# --------------------------------------------------------------------------- #


def test_natural_gradient_identity_metric_is_gradient_descent():
    opt = QNGOptimizer(regularization=0.0, scale_regularization=False)
    grad = np.array([1.0, -2.0, 0.5])
    delta = opt._natural_gradient(grad, np.eye(3))
    np.testing.assert_allclose(delta, grad)


def test_natural_gradient_diagonal_metric_scales_gradient():
    opt = QNGOptimizer(regularization=0.0, scale_regularization=False)
    grad = np.array([2.0, 3.0])
    delta = opt._natural_gradient(grad, np.diag([4.0, 9.0]))
    np.testing.assert_allclose(delta, [0.5, 1.0 / 3.0])


def test_natural_gradient_tikhonov_singular_metric_raises_actionably():
    opt = QNGOptimizer(regularization=0.0, scale_regularization=False)
    singular = np.outer([1.0, 0.0], [1.0, 0.0])
    message = (
        "Regularised natural-gradient solve failed: the damped metric (G + λI) is "
        "not positive-definite. λ must exceed the most negative eigenvalue of G, "
        "which a rank-deficient metric leaves at zero and a shot-noisy one can push "
        "below it. Raise `regularization` or use solver='pinv'."
    )
    with pytest.raises(np.linalg.LinAlgError, match=exact_match(message)):
        opt._natural_gradient(np.array([1.0, 1.0]), singular)


def test_max_step_norm_clips_update():
    opt = QNGOptimizer(
        step_size=1.0, regularization=0.0, scale_regularization=False, max_step_norm=1.0
    )
    delta = opt._natural_gradient(np.array([3.0, 4.0]), np.eye(2))
    update = opt.step_size * delta
    assert np.linalg.norm(update) == pytest.approx(1.0)
    # Direction is preserved.
    np.testing.assert_allclose(delta / np.linalg.norm(delta), [0.6, 0.8])


_REQUIRES_EVALUATORS = (
    "QNGOptimizer requires both a gradient function (`jac`) and a metric "
    "function (`metric_fn`). It is driven by VariationalQuantumAlgorithm.run(), "
    "which supplies both."
)


def _zero_jac(p):
    return np.zeros(len(p))


def _identity_metric(p):
    return np.eye(len(p))


@pytest.mark.parametrize(
    "evaluators",
    [
        {"jac": None, "metric_fn": _identity_metric},
        {"jac": _zero_jac, "metric_fn": None},
        {"metric_fn": _identity_metric},
        {"jac": _zero_jac},
    ],
    ids=["jac-none", "metric-none", "jac-missing", "metric-missing"],
)
def test_optimize_requires_jac_and_metric(evaluators):
    with pytest.raises(ValueError, match=exact_match(_REQUIRES_EVALUATORS)):
        QNGOptimizer().optimize(
            cost_fn=lambda x: 0.0, initial_params=np.zeros(2), **evaluators
        )


def test_optimize_requires_initial_params():
    with pytest.raises(
        ValueError, match=exact_match("QNGOptimizer requires initial_params.")
    ):
        QNGOptimizer().optimize(
            cost_fn=lambda x: 0.0, jac=_zero_jac, metric_fn=_identity_metric
        )


def test_optimize_reports_iteration_progress_and_termination():
    """Each step reaches the callback numbered from 1 as a successful in-progress
    result, the default step size of 0.1 drives the update, and the final result
    reports the iteration budget."""
    records = []
    result = QNGOptimizer(regularization=0.0).optimize(
        cost_fn=lambda x: float(x @ x),
        initial_params=np.array([1.0, -2.0]),
        callback_fn=records.append,
        max_iterations=3,
        jac=lambda x: 2 * x,
        metric_fn=_identity_metric,
    )

    assert [r.nit for r in records] == [1, 2, 3]
    assert all(r.success is True for r in records)
    assert [r.message for r in records] == ["Optimisation in progress."] * 3
    np.testing.assert_allclose(
        np.vstack([r.x for r in records]),
        [[1.0, -2.0], [0.8, -1.6], [0.64, -1.28]],
    )
    assert result.nit == 3
    assert result.success is True
    assert result.message == "Optimisation terminated: reached max_iterations."


def test_optimize_keeps_the_first_of_tied_best_iterates():
    initial = np.array([0.3, 0.7])
    result = QNGOptimizer(regularization=0.0).optimize(
        cost_fn=lambda x: 1.0,
        initial_params=initial,
        max_iterations=3,
        jac=lambda x: np.ones(2),
        metric_fn=_identity_metric,
    )
    np.testing.assert_array_equal(result.x, initial)


def test_overshoot_guard_is_silent_up_to_pi_over_4_and_once_warned(recwarn):
    opt = QNGOptimizer()
    assert opt._warn_if_overshooting(np.array([np.pi / 4]), False) is False
    assert opt._warn_if_overshooting(np.array([0.1]), False) is False
    assert opt._warn_if_overshooting(np.array([10.0]), True) is True
    assert len(recwarn) == 0


_OVERSHOOT_WARNING = (
    "QNGOptimizer: a step moved a parameter by 1 rad (more than pi/4), so the "
    "update is likely overshooting. Lower `step_size`, raise `regularization`, "
    "or set `max_step_norm`."
)


def _small_then_large_step_kwargs():
    grads = iter([np.array([0.1]), np.array([1.0])])
    return {
        "cost_fn": lambda x: 0.0,
        "initial_params": np.zeros(1),
        "max_iterations": 2,
        "jac": lambda x: next(grads),
        "metric_fn": _identity_metric,
    }


def _unit_step_qng():
    return QNGOptimizer(step_size=1.0, regularization=0.0, scale_regularization=False)


def test_overshooting_step_warns_once_at_the_optimize_caller():
    with pytest.warns(UserWarning, match=exact_match(_OVERSHOOT_WARNING)) as record:
        _unit_step_qng().optimize(**_small_then_large_step_kwargs())
    assert len(record) == 1
    assert record[0].filename == __file__


@pytest.mark.parametrize(
    "kwargs, metric, expected",
    [
        ({}, np.diag([1.0, 1e-4]), [1.0, 1e4]),
        ({"rcond": 1e-3}, np.diag([1.0, 1e-4]), [1.0, 0.0]),
        ({"rcond": 1e-9}, np.diag([1.0, 0.0]), [1.0, 0.0]),
    ],
    ids=["default-rcond", "custom-rcond", "singular-metric"],
)
def test_pinv_solver_applies_rcond_cutoff(kwargs, metric, expected):
    opt = QNGOptimizer(solver="pinv", **kwargs)
    delta = opt._natural_gradient(np.array([1.0, 1.0]), metric)
    np.testing.assert_allclose(delta, expected)


@pytest.mark.parametrize(
    "kwargs, scale, expected",
    [
        ({}, 4.0, 0.125),
        ({"scale_regularization": False}, 4.0, 0.2),
        ({}, 0.25, 0.8),
    ],
    ids=["scaled-by-default", "unscaled", "scale-floored-at-one"],
)
def test_tikhonov_damping_scales_with_metric_diagonal(kwargs, scale, expected):
    opt = QNGOptimizer(regularization=1.0, **kwargs)
    delta = opt._natural_gradient(np.ones(2), scale * np.eye(2))
    np.testing.assert_allclose(delta, [expected, expected])


@pytest.mark.parametrize(
    "grad, step_size, message",
    [
        (
            np.array([np.inf, 0.0]),
            0.1,
            "QNGOptimizer received a non-finite gradient or metric. The iterate "
            "has diverged, or the cost function returned a non-finite value. "
            "Lower `step_size`, raise `regularization`, or set `max_step_norm`.",
        ),
        (
            np.array([1e308, 0.0]),
            10.0,
            "QNGOptimizer produced a non-finite parameter update; the pullback "
            "metric is severely ill-conditioned. Raise `regularization` or lower "
            "`step_size`.",
        ),
    ],
    ids=["non-finite-input", "non-finite-update"],
)
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_non_finite_natural_gradient_raises(grad, step_size, message):
    opt = QNGOptimizer(
        step_size=step_size, regularization=0.0, scale_regularization=False
    )
    with pytest.raises(FloatingPointError, match=exact_match(message)):
        opt._natural_gradient(grad, np.eye(2))


_NO_CHECKPOINTING = (
    "QNGOptimizer does not support {}. Its only state is the current parameter "
    "vector, which is already persisted by the variational algorithm's program "
    "state."
)


@pytest.mark.parametrize(
    "call, feature",
    [
        (lambda tmp_path: QNGOptimizer().get_config(), "checkpointing"),
        (lambda tmp_path: QNGOptimizer().save_state(tmp_path), "state saving"),
        (lambda tmp_path: QNGOptimizer.load_state(tmp_path), "state loading"),
    ],
    ids=["get_config", "save_state", "load_state"],
)
def test_checkpointing_entry_points_raise(call, feature, tmp_path):
    with pytest.raises(
        NotImplementedError, match=exact_match(_NO_CHECKPOINTING.format(feature))
    ):
        call(tmp_path)


def test_rejects_a_fidelity_sampling_estimator():
    message = (
        "StochasticFidelityMetricEstimator estimates the metric from sampled "
        "fidelity overlaps and supplies no closed-form metric function, which "
        "QNGOptimizer requires. Use QNSPSAOptimizer for the stochastic-fidelity "
        "metric, or pass a closed-form estimator (PullbackMetricEstimator, "
        "FubiniStudyMetricEstimator)."
    )
    with pytest.raises(ValueError, match=exact_match(message)):
        QNGOptimizer(metric_estimator=StochasticFidelityMetricEstimator())


def test_optimize_converges_on_quadratic():
    # f(x) = 0.5 x^T A x; grad = A x. With metric == A, one full natural-gradient
    # step lands exactly on the minimum (Newton step).
    a_matrix = np.array([[3.0, 0.0], [0.0, 1.0]])

    opt = QNGOptimizer(step_size=1.0, regularization=1e-9, scale_regularization=False)
    result = opt.optimize(
        cost_fn=lambda x: 0.5 * x @ a_matrix @ x,
        initial_params=np.array([0.5, 0.5]),
        max_iterations=20,
        jac=lambda x: a_matrix @ x,
        metric_fn=lambda x: a_matrix,
    )

    np.testing.assert_allclose(result.x.squeeze(), [0.0, 0.0], atol=1e-8)
    assert result.fun[0] == pytest.approx(0.0, abs=1e-12)
    assert result.success


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"step_size": 0.0}, "step_size must be positive, got 0.0."),
        ({"regularization": -1.0}, "regularization must be non-negative, got -1.0."),
        ({"solver": "bogus"}, "solver must be 'tikhonov' or 'pinv', got 'bogus'."),
        ({"max_step_norm": 0}, "max_step_norm must be positive or None, got 0."),
    ],
)
def test_invalid_constructor_args(kwargs, match):
    with pytest.raises(ValueError, match=exact_match(match)):
        QNGOptimizer(**kwargs)


def test_qng_zero_iterations_raises():
    """Zero steps would return success with an infinite loss — reject it."""
    message = (
        "max_iterations must be >= 1, got 0; the optimisation loop performs no "
        "evaluation with zero steps."
    )
    with pytest.raises(ValueError, match=exact_match(message)):
        QNGOptimizer().optimize(
            lambda x: 0.0,
            initial_params=np.zeros(2),
            max_iterations=0,
            jac=lambda x: np.zeros(2),
            metric_fn=lambda x: np.eye(2),
        )


def test_qng_runs_fifty_iterations_by_default():
    result = QNGOptimizer().optimize(
        lambda x: 0.0,
        initial_params=np.zeros(2),
        jac=_zero_jac,
        metric_fn=_identity_metric,
    )
    assert result.nit == 50


def test_qng_returns_float_parameters_for_integer_initial_params():
    result = QNGOptimizer().optimize(
        lambda x: 1.0,
        initial_params=np.array([1, 2]),
        max_iterations=1,
        jac=_zero_jac,
        metric_fn=_identity_metric,
    )
    assert result.x.dtype == np.float64


def test_qng_run_without_a_finite_cost_reports_failure():
    initial = np.array([0.3, 0.7])
    result = QNGOptimizer().optimize(
        lambda x: np.nan,
        initial_params=initial,
        max_iterations=2,
        jac=_zero_jac,
        metric_fn=_identity_metric,
    )

    np.testing.assert_array_equal(result.x, initial)
    assert not result.success
    assert result.message == (
        "Optimisation failed: no finite cost value was observed in 2 iterations."
    )


# --------------------------------------------------------------------------- #
# Pullback metric: assembly + integration
# --------------------------------------------------------------------------- #


_COEFFS = np.array([0.5, -0.3, 0.2])


def _cosine_terms(coeffs):
    return lambda param_sets: (np.cos(param_sets[:, : len(coeffs)]), coeffs)


def test_pullback_metric_assembly(injectable_vqe, monkeypatch):
    """grad and G are assembled correctly from the per-term Jacobian; the
    measurement seam is injected, so only the assembly maths is under test."""
    n_params = set_unit_shift_rule(injectable_vqe)
    fake = np.random.default_rng(0).standard_normal((2 * n_params, len(_COEFFS)))
    inject_term_expectations(monkeypatch, lambda _param_sets: (fake, _COEFFS))

    evaluators = PullbackMetricEstimator().bind(injectable_vqe)
    grad = evaluators["jac"](np.zeros(n_params))
    metric = evaluators["metric_fn"](np.zeros(n_params))

    jac = 0.5 * (fake[0::2] - fake[1::2])
    np.testing.assert_allclose(grad, jac @ _COEFFS)
    np.testing.assert_allclose(metric, (jac * _COEFFS**2) @ jac.T / np.sum(_COEFFS**2))
    np.testing.assert_allclose(metric, metric.T, atol=1e-12)
    assert np.linalg.eigvalsh(metric).min() >= -1e-9


def test_pullback_shares_one_measurement_per_parameter_point(
    injectable_vqe, monkeypatch
):
    n_params = set_unit_shift_rule(injectable_vqe)
    calls = inject_term_expectations(monkeypatch, _cosine_terms(_COEFFS))
    evaluators = PullbackMetricEstimator().bind(injectable_vqe)
    first, second = np.full(n_params, 0.3), np.full(n_params, 1.1)

    grad_first = evaluators["jac"](first)
    evaluators["metric_fn"](first)
    assert len(calls) == 1

    grad_second = evaluators["jac"](second)
    assert len(calls) == 2
    assert not np.allclose(grad_first, grad_second)


def test_pullback_metric_with_all_zero_coefficients_is_zero(
    injectable_vqe, monkeypatch
):
    n_params = set_unit_shift_rule(injectable_vqe)
    inject_term_expectations(monkeypatch, _cosine_terms(np.zeros(3)))
    metric = PullbackMetricEstimator().bind(injectable_vqe)["metric_fn"](
        np.full(n_params, 0.3)
    )
    np.testing.assert_array_equal(metric, np.zeros((n_params, n_params)))


def test_metric_pipeline_terms_sum_to_energy(toy_vqe):
    """sum_r a_r <P_r> from the metric pipeline reproduces the summed energy."""
    toy_vqe.backend.set_seed(1997)
    n_params = toy_vqe.n_layers * toy_vqe.n_params_per_layer
    params = np.random.default_rng(1).uniform(0, 2 * np.pi, size=(1, n_params))

    terms, coeffs = _split_into_terms(toy_vqe.cost_hamiltonian)
    preprocessor = CircuitPreprocessor(
        "metric",
        preprocess=lambda meta: replace(
            meta, observable=tuple(terms), _was_multi_obs=True
        ),
    )
    result = toy_vqe.evaluate(params, preprocessor, preserve_keys=True)
    indexed = average_by_param_set(
        result,
        lambda value: np.asarray(value, dtype=np.float64).reshape(-1),
    )
    reconstructed = float(indexed[0] @ coeffs) + toy_vqe.loss_constant

    energy = toy_vqe._evaluate_cost_param_sets(params)[0]
    assert reconstructed == pytest.approx(energy, abs=1e-10)


def test_fubini_study_metric_smoke(default_test_simulator, default_optimizer):
    """The FS metric runs end-to-end and is a symmetric PSD matrix."""
    vqe = VQE(
        HamiltonianProblem(SparsePauliOp.from_list([("ZI", 0.5), ("IZ", 0.5)])),
        ansatz=GenericLayerAnsatz([RYGate]),
        n_layers=2,
        backend=default_test_simulator,
        optimizer=default_optimizer,
        seed=1997,
    )
    vqe.backend.set_seed(1997)
    n_params = vqe.n_layers * vqe.n_params_per_layer
    metric_fn = FubiniStudyMetricEstimator().bind(vqe)["metric_fn"]
    g = metric_fn(np.linspace(0.1, 1.0, n_params))

    assert g.shape == (n_params, n_params)
    np.testing.assert_allclose(g, g.T, atol=1e-8)
    assert np.linalg.eigvalsh(g).min() >= -1e-6


#: ``(gate, wires, weight index)`` tapes; ``None`` marks a fixed gate.
_RY_CNOT_LAYERS = tuple(
    op
    for layer in range(2)
    for op in (
        *(("ry", (q,), layer * 3 + q) for q in range(3)),
        ("cx", (0, 1), None),
        ("cx", (1, 2), None),
    )
)
#: Later blocks see non-zero generator expectations.
_MIXED_ROTATIONS = (
    ("rx", (0,), 0),
    ("ry", (1,), 1),
    ("cx", (0, 1), None),
    ("rz", (0,), 2),
    ("ry", (1,), 3),
    ("rx", (0,), 4),
    ("cx", (1, 0), None),
    ("rz", (1,), 5),
    ("ry", (0,), 6),
)


def _apply_qiskit_tape(qc, tape, weights):
    for gate, wires, index in tape:
        args = () if index is None else (weights[index],)
        getattr(qc, gate)(*args, *wires)


def _apply_pennylane_tape(qp, tape, weights):
    gates = {
        "rx": qp.RX,
        "ry": qp.RY,
        "rz": qp.RZ,
        "cx": qp.CNOT,
    }
    for gate, wires, index in tape:
        args = () if index is None else (weights[index],)
        gates[gate](*args, wires=list(wires))


@pytest.mark.parametrize(
    "tape, n_qubits",
    [(_RY_CNOT_LAYERS, 3), (_MIXED_ROTATIONS, 2)],
    ids=["ry-cnot-layers", "mixed-rotations"],
)
def test_fubini_study_matches_pennylane_block_diag(
    qp, default_test_simulator, default_optimizer, tape, n_qubits
):
    """The block-diagonal FS metric agrees with PennyLane's reference, including
    off-diagonal blocks from an entangled prefix and the covariance term of
    pre-states with non-zero generator expectations."""
    m = 1 + max(index for _, _, index in tape if index is not None)
    weights = [Parameter(f"w{i}") for i in range(m)]
    qc = QuantumCircuit(n_qubits, n_qubits)
    _apply_qiskit_tape(qc, tape, weights)
    qc.measure(range(n_qubits), range(n_qubits))

    program = CustomVQA(
        qscript=qc, backend=default_test_simulator, optimizer=default_optimizer
    )
    program.backend.set_seed(1997)

    dev = qp.device("default.qubit", wires=n_qubits)

    @qp.qnode(dev)
    def circuit(w):
        _apply_pennylane_tape(qp, tape, w)
        return qp.expval(qp.Z(0))

    values = np.linspace(0.2, 1.5, m)
    pl_metric = np.array(
        qp.metric_tensor(circuit, approx="block-diag")(
            qp.numpy.array(values, requires_grad=True)
        )
    )

    # Align divi's parameter order to the logical weight index before comparing.
    full_params = program.cost_circuit.parameters
    order = [int(p.name[1:]) for p in full_params]
    divi_metric = FubiniStudyMetricEstimator().bind(program)["metric_fn"](values[order])

    np.testing.assert_allclose(divi_metric, pl_metric[np.ix_(order, order)], atol=1e-10)


def test_fubini_study_prefix_preprocessor_uses_incoming_branch_meta(monkeypatch):
    """FS prefix measurement derives the prefix + per-branch block structure
    from the sampled branch meta (recomputed from the cost cohort) and measures
    every label in one multi-observable pass."""
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.rx(theta, 0)
    sampled_meta = MetaCircuit(
        circuit_bodies=(((), circuit_to_dag(qc)),),
        parameters=(theta,),
    )
    seen = []

    def fake_fs_blocks(meta):
        seen.append(meta)
        return [([], [(0, SparsePauliOp("X"))])], [theta], 1

    prefix_metas = []

    def fake_run_metric_by_branch(_program, preprocessor, _param_sets):
        prefix_metas.append(preprocessor.preprocess(sampled_meta))
        return {(("ham", 0),): {0: np.array([0.25])}}

    monkeypatch.setattr("divi.qprog._metrics._fs_blocks", fake_fs_blocks)
    monkeypatch.setattr(
        "divi.qprog._metrics._run_metric_by_branch", fake_run_metric_by_branch
    )

    program = SimpleNamespace(
        cost_circuit=sampled_meta,
        _post_spec_batch=lambda: {(("ham", 0),): sampled_meta},
    )
    exp_by_branch, branch_data = _measure_prefix_paulis(
        program, [theta], np.array([0.1]), block_id=0
    )

    assert sampled_meta in seen
    (prefix_meta,) = prefix_metas
    assert prefix_meta._was_multi_obs is True
    assert [str(o.paulis[0]) for o in prefix_meta.observable] == ["X"]
    assert branch_data[(("ham", 0),)][0] == (0,)
    assert exp_by_branch[(("ham", 0),)]["X"] == pytest.approx(0.25)


# --------------------------------------------------------------------------- #
# Optimizer ↔ program compatibility gate
# --------------------------------------------------------------------------- #


def _small_pce(backend, optimizer):
    return PCE(
        problem=BinaryOptimizationProblem(np.array([[1.0, 0.2], [0.2, 2.0]])),
        ansatz=GenericLayerAnsatz([RYGate]),
        n_layers=1,
        backend=backend,
        optimizer=optimizer,
    )


def test_pce_rejects_pullback_metric(dummy_simulator, default_optimizer):
    """PCE's loss is a classical objective, not <cost_hamiltonian>, so the
    pullback metric is rejected up front rather than run on the placeholder."""
    pce = _small_pce(dummy_simulator, default_optimizer)
    message = (
        "The pullback metric requires the loss to be the expectation value of the "
        "cost Hamiltonian, but this program's cost computes a classical objective "
        "(e.g. PCE). Use the Fubini–Study estimator (FubiniStudyMetricEstimator) "
        "instead."
    )
    with pytest.raises(ContractViolation, match=exact_match(message)):
        QNGOptimizer().validate_program(pce)


def test_pce_accepts_fubini_study_metric(dummy_simulator, default_optimizer):
    """FS is observable-agnostic, so it is valid for PCE."""
    pce = _small_pce(dummy_simulator, default_optimizer)
    QNGOptimizer(metric_estimator=FubiniStudyMetricEstimator()).validate_program(pce)


def test_supervised_custom_vqa_rejects_pullback_metric(
    dummy_simulator, default_optimizer
):
    """A supervised data-binding loss is non-linear in the expectations, so the
    pullback metric is rejected."""
    program = data_bound_custom_vqa(
        dummy_simulator, default_optimizer, labels=[1.0, -1.0]
    )
    message = (
        "The pullback metric is invalid for a supervised data-binding loss: the "
        "per-sample loss is a non-linear function of the expectation values. No "
        "metric estimator supports this case — the Fubini–Study estimators reject "
        "data-bound programs outright — so use a non-metric optimizer such as "
        "SPSAOptimizer."
    )
    with pytest.raises(ContractViolation, match=exact_match(message)):
        QNGOptimizer().validate_program(program)


def test_fubini_study_rejects_composite_angle(dummy_simulator, default_optimizer):
    """FS needs a bare-parameter Pauli rotation; a composite angle is rejected."""
    x = Parameter("x")
    qc = QuantumCircuit(1, 1)
    qc.rx(2 * x, 0)
    qc.measure(0, 0)
    program = CustomVQA(
        qscript=qc, backend=dummy_simulator, optimizer=default_optimizer
    )
    with pytest.raises(ContractViolation, match="Fubini"):
        QNGOptimizer(metric_estimator=FubiniStudyMetricEstimator()).validate_program(
            program
        )


def test_metric_pipelines_are_cacheable(dummy_simulator, default_optimizer):
    """The consolidated metric transforms are pure, so their preprocessors carry
    stable cache keys and ``_build_preprocessor_pipeline`` reuses one pipeline
    object across iterations (its forward cache survives)."""
    vqe = VQE(
        HamiltonianProblem(
            SparsePauliOp.from_list([("ZI", 0.5), ("IZ", -0.3), ("XX", 0.2)])
        ),
        ansatz=GenericLayerAnsatz([RYGate]),
        n_layers=1,
        backend=dummy_simulator,
        optimizer=default_optimizer,
    )
    assert _all_terms_preprocessor().cache_key == METRIC_ROUTINE
    assert _fs_prefix_labels_preprocessor(0, ()).cache_key == (METRIC_ROUTINE, 0)
    assert _fs_prefix_labels_preprocessor(0, ()).consumes_dag_bodies is True

    # Fresh-but-equal preprocessors hit the cache -> same pipeline object.
    p1 = vqe._build_preprocessor_pipeline(_all_terms_preprocessor())
    p2 = vqe._build_preprocessor_pipeline(_all_terms_preprocessor())
    assert p1 is p2


def _meta_from(qc: QuantumCircuit, observable=None) -> MetaCircuit:
    return MetaCircuit(
        circuit_bodies=(((), circuit_to_dag(qc)),),
        parameters=tuple(qc.parameters),
        observable=observable,
    )


def _single_ry_meta() -> MetaCircuit:
    qc = QuantumCircuit(1)
    qc.ry(Parameter("a"), 0)
    return _meta_from(qc)


@pytest.mark.parametrize("block_id", [1, 5])
def test_fs_block_prefix_rejects_out_of_range_block(block_id):
    """A branch with fewer FS blocks than the requested index fails loudly."""
    message = (
        "A sampled metric branch has fewer Fubini-Study blocks than the "
        "reference ansatz."
    )
    with pytest.raises(ContractViolation, match=exact_match(message)):
        _fs_block_prefix(
            _single_ry_meta(), block_id=block_id, reference_prefix_param_names=()
        )


def test_fs_block_prefix_rejects_prefix_param_layout_mismatch():
    """A branch whose prefix-parameter layout differs from the reference ansatz
    fails loudly rather than mis-aligning the metric."""
    message = (
        "A sampled metric branch has a different Fubini-Study prefix parameter "
        "layout than the reference ansatz."
    )
    with pytest.raises(ContractViolation, match=exact_match(message)):
        _fs_block_prefix(
            _single_ry_meta(), block_id=0, reference_prefix_param_names=("z",)
        )


_FS_GATE_HINT = (
    ". Use the pullback metric (PullbackMetricEstimator) or a non-metric optimizer."
)


@pytest.mark.parametrize(
    "build, used",
    [
        (lambda qc, x: qc.p(x, 0), "'p'"),
        (lambda qc, x: qc.p(2 * x, 0), "'p'"),
        (
            lambda qc, x: qc.rx(2 * x, 0),
            "'rx' with the angle expression '2*x', which is not a bare trainable "
            "parameter",
        ),
    ],
    ids=["unsupported-gate", "unsupported-gate-composite-angle", "composite-angle"],
)
def test_fs_blocks_rejects_unsupported_rotations(build, used):
    qc = QuantumCircuit(1)
    build(qc, Parameter("x"))
    message = (
        "The Fubini–Study metric supports only single-parameter Pauli-rotation "
        "gates (rx/ry/rz) with a bare parameter as the angle; this "
        f"ansatz uses {used}{_FS_GATE_HINT}"
    )
    with pytest.raises(ContractViolation, match=exact_match(message)):
        _fs_blocks(_meta_from(qc))


def test_prefix_ops_view_indexes_only_its_prefix():
    view = _PrefixOpsView(("a", "b", "c", "d"), stop=3)

    assert len(view) == 3
    assert list(view) == ["a", "b", "c"]
    assert [view[i] for i in (0, 2, -1, -3)] == ["a", "c", "c", "a"]
    assert view[1:] == ("b", "c")
    assert view[::-1] == ("c", "b", "a")
    with pytest.raises(IndexError, match="^3$"):
        view[3]
    with pytest.raises(IndexError, match="^-4$"):
        view[-4]


def test_split_into_terms_drops_identity_terms():
    hamiltonian = SparsePauliOp.from_list([("ZI", 0.5), ("II", 1.0), ("IZ", -0.3)])
    terms, coeffs = _split_into_terms(hamiltonian)
    assert [term.paulis.to_labels() for term in terms] == [["ZI"], ["IZ"]]
    np.testing.assert_allclose(coeffs, [0.5, -0.3])


def test_split_into_terms_rejects_an_identity_only_observable():
    message = (
        "Pullback metric requires a loss observable with at least one "
        "non-identity Pauli term."
    )
    with pytest.raises(ValueError, match=exact_match(message)):
        _split_into_terms(SparsePauliOp.from_list([("II", 1.0)]))


def test_fubini_study_metric_sums_shared_parameters_and_averages_branches(
    dummy_simulator, default_optimizer, monkeypatch
):
    """Block entries of a parameter driving several gates add; per-branch metrics
    are averaged."""
    qc = QuantumCircuit(1, 1)
    qc.rx(Parameter("a"), 0)
    qc.measure(0, 0)
    program = CustomVQA(
        qscript=qc, backend=dummy_simulator, optimizer=default_optimizer
    )
    monkeypatch.setattr(
        "divi.qprog._metrics._fs_block_covariance",
        lambda *_: {
            (("ham", 0),): ((0, 0), np.array([[1.0, 2.0], [2.0, 4.0]])),
            (("ham", 1),): ((0, 0), np.array([[0.0, 0.0], [0.0, 1.0]])),
        },
    )
    metric = FubiniStudyMetricEstimator().bind(program)["metric_fn"](np.array([0.3]))
    np.testing.assert_allclose(metric, [[5.0]])


def _single_rx_program(source_batch):
    """Stand-in program whose cost ansatz is ``rx(theta)`` and whose cost cohort
    is ``source_batch``."""
    theta = Parameter("theta")
    qc = QuantumCircuit(1)
    qc.rx(theta, 0)
    meta = _meta_from(qc)
    return (
        SimpleNamespace(cost_circuit=meta, _post_spec_batch=lambda: source_batch(meta)),
        theta,
    )


@pytest.mark.parametrize(
    "block_id, source_batch, measured, message",
    [
        (
            1,
            lambda meta: {(("ham", 0),): meta},
            [0.25],
            "The Fubini-Study block index is outside the reference ansatz.",
        ),
        (
            0,
            lambda meta: {},
            [0.25],
            "A measured Fubini-Study branch is absent from the cost cohort.",
        ),
        (
            0,
            lambda meta: {(("ham", 0),): meta},
            [0.25, 0.5],
            "Fubini-Study per-label measurement count does not match the "
            "branch's label count.",
        ),
    ],
    ids=["block-out-of-range", "branch-absent", "label-count-mismatch"],
)
def test_measure_prefix_paulis_rejects_inconsistent_measurements(
    monkeypatch, block_id, source_batch, measured, message
):
    program, theta = _single_rx_program(source_batch)
    monkeypatch.setattr(
        "divi.qprog._metrics._run_metric_by_branch",
        lambda *_: {(("ham", 0),): {0: np.array(measured)}},
    )
    with pytest.raises(ContractViolation, match=exact_match(message)):
        _measure_prefix_paulis(program, [theta], np.array([0.1]), block_id=block_id)


def _expval_program(observable):
    return SimpleNamespace(
        cost_preprocessor=lambda: CircuitPreprocessor("cost"),
        cost_circuit=SimpleNamespace(observable=observable),
    )


@pytest.mark.parametrize(
    "observable",
    [None, (SparsePauliOp("Z"), SparsePauliOp("X"))],
    ids=["no-observable", "two-observables"],
)
def test_pullback_requires_exactly_one_loss_observable(observable):
    message = (
        "The pullback metric requires the cost circuit to carry exactly one loss "
        "observable."
    )
    with pytest.raises(ContractViolation, match=exact_match(message)):
        PullbackMetricEstimator().check_compatible(_expval_program(observable))


def test_pullback_accepts_a_single_loss_observable():
    PullbackMetricEstimator().check_compatible(_expval_program((SparsePauliOp("Z"),)))


_DATA_BOUND_SUPERVISED = (
    "The {} metric does not support data-bound programs, and no metric estimator "
    "supports a supervised (labelled) data-binding loss: the per-sample loss is a "
    "non-linear function of the expectation values. Use a non-metric optimizer "
    "such as SPSAOptimizer or a gradient-free ScipyOptimizer method."
)
_DATA_BOUND_UNSUPERVISED = (
    "The {} metric does not support data-bound programs: the ansatz state depends "
    "on the data input, so the metric is data-dependent. Use QNGOptimizer with its "
    "default pullback metric, which supports an unlabelled data-bound loss, or a "
    "non-metric optimizer such as SPSAOptimizer."
)


@pytest.mark.parametrize(
    "estimator, metric",
    [
        (FubiniStudyMetricEstimator, "Fubini–Study"),
        (StochasticFidelityMetricEstimator, "stochastic-fidelity"),
    ],
    ids=["fubini-study", "stochastic-fidelity"],
)
@pytest.mark.parametrize(
    "labels, template",
    [(None, _DATA_BOUND_UNSUPERVISED), ([1.0, -1.0], _DATA_BOUND_SUPERVISED)],
    ids=["unsupervised", "supervised"],
)
def test_state_metrics_reject_data_bound_programs(
    dummy_simulator, default_optimizer, estimator, metric, labels, template
):
    program = data_bound_custom_vqa(dummy_simulator, default_optimizer, labels)
    with pytest.raises(ContractViolation, match=exact_match(template.format(metric))):
        estimator().check_compatible(program)


def test_term_expectations_rejects_term_count_mismatch(
    dummy_simulator, default_optimizer, monkeypatch
):
    """A branch whose measured term count disagrees with its coefficient count
    fails loudly rather than mis-assembling the metric."""
    vqe = VQE(
        HamiltonianProblem(SparsePauliOp.from_list([("ZI", 0.5)])),
        ansatz=GenericLayerAnsatz([RYGate]),
        n_layers=1,
        backend=dummy_simulator,
        optimizer=default_optimizer,
    )
    branch_key = next(iter(vqe._post_spec_batch()))
    monkeypatch.setattr(
        "divi.qprog._metrics._run_metric_by_branch",
        lambda _p, _prep, _ps: {branch_key: {0: np.array([0.5, 0.3])}},
    )
    message = (
        "Per-term measurement count does not match the branch's coefficient count."
    )
    with pytest.raises(ContractViolation, match=exact_match(message)):
        _term_expectations(vqe, np.zeros((1, vqe.n_layers * vqe.n_params_per_layer)))


@pytest.mark.parametrize(
    "source_batch",
    [
        lambda meta: {},
        lambda meta: {(("ham", 0),): replace(meta, observable=None)},
    ],
    ids=["branch-absent", "branch-without-observable"],
)
def test_term_expectations_rejects_an_unusable_source_branch(monkeypatch, source_batch):
    program, theta = _single_rx_program(source_batch)
    monkeypatch.setattr(
        "divi.qprog._metrics._run_metric_by_branch",
        lambda *_: {(("ham", 0),): {0: np.array([0.5])}},
    )
    message = (
        "A measured metric branch is absent from the cost cohort or carries no "
        "observable."
    )
    with pytest.raises(ContractViolation, match=exact_match(message)):
        _term_expectations(program, np.zeros((1, 1)))


def test_all_terms_preprocessor_rejects_branch_without_observable():
    """The pullback term-expansion transform raises on a branch that carries no
    single loss observable, rather than silently producing a wrong metric."""
    qc = QuantumCircuit(1)
    qc.rx(Parameter("a"), 0)
    meta = MetaCircuit(circuit_bodies=(((), circuit_to_dag(qc)),), observable=None)

    message = (
        "The pullback metric requires each cost branch to carry exactly one loss "
        "observable."
    )
    with pytest.raises(ContractViolation, match=exact_match(message)):
        _split_observable_into_terms(meta)


def test_all_terms_preprocessor_expands_into_a_multi_observable_meta():
    qc = QuantumCircuit(2)
    qc.rx(Parameter("a"), 0)
    meta = _meta_from(qc, SparsePauliOp.from_list([("ZI", 0.5), ("XX", -0.2)]))

    expanded = _split_observable_into_terms(meta)

    assert expanded._was_multi_obs is True
    assert [o.paulis.to_labels() for o in expanded.observable] == [["ZI"], ["XX"]]


def test_vqe_runs_under_fubini_study_qng(toy_vqe):
    """VQE optimizes end-to-end under QNG with the FS metric (gradient from the
    program's parameter-shift, metric from FS)."""
    toy_vqe.backend.set_seed(1997)
    toy_vqe.optimizer = QNGOptimizer(
        metric_estimator=FubiniStudyMetricEstimator(), step_size=0.2
    )
    toy_vqe.max_iterations = 5
    toy_vqe.run(perform_final_computation=False)
    assert len(toy_vqe.losses_history) == 5
    assert np.isfinite(toy_vqe.best_loss)


def test_qaoa_qdrift_qng_reuses_cost_cohort(dummy_simulator):
    """The cost pipeline exposes the same cached QDrift cohort within one
    evaluation, instead of resampling when the source batch is inspected."""
    qaoa = QAOA(
        MaxCutProblem(nx.bull_graph()),
        n_layers=1,
        trotterization_strategy=QDrift(
            sampling_budget=2,
            n_hamiltonians_per_iteration=3,
            seed=42,
        ),
        optimizer=QNGOptimizer(),
        max_iterations=1,
        backend=dummy_simulator,
    )
    pipeline = qaoa._build_preprocessor_pipeline(qaoa.cost_preprocessor())
    env = qaoa._build_pipeline_env()
    cost_trace = pipeline.run_forward_pass(qaoa.cost_hamiltonian, env)

    sourced = qaoa._post_spec_batch()

    assert sourced is cost_trace.initial_batch


def test_qaoa_qdrift_pullback_uses_sampled_branch_observables(dummy_simulator):
    """Pullback term coefficients come from each sampled QDrift branch, not the
    static reference cost circuit."""
    qaoa = QAOA(
        MaxCutProblem(nx.bull_graph()),
        n_layers=1,
        trotterization_strategy=QDrift(
            sampling_budget=2,
            n_hamiltonians_per_iteration=3,
            seed=42,
        ),
        optimizer=QNGOptimizer(),
        max_iterations=1,
        backend=dummy_simulator,
    )
    assert qaoa.cost_circuit.observable is not None
    param_sets = np.zeros((1, len(qaoa.cost_circuit.parameters)))

    branch_payloads = _term_expectations(qaoa, param_sets)
    sourced = qaoa._post_spec_batch()

    assert set(branch_payloads) == set(sourced)
    for branch_key, meta in sourced.items():
        assert meta.observable is not None
        _, sampled_coeffs = _split_into_terms(meta.observable[0])
        np.testing.assert_allclose(branch_payloads[branch_key][1], sampled_coeffs)


def test_qaoa_qdrift_qng_refuses_upfront(dummy_simulator):
    """QNG needs a gradient, and QAOA has no exact parameter-shift rule.

    This ran to completion while the two-term rule returned a near-zero gradient
    for both layer angles, so the natural-gradient step was a null step.
    """
    qaoa = QAOA(
        MaxCutProblem(nx.bull_graph()),
        n_layers=1,
        trotterization_strategy=QDrift(
            sampling_budget=2, n_hamiltonians_per_iteration=3, seed=42
        ),
        optimizer=QNGOptimizer(),
        max_iterations=1,
        backend=dummy_simulator,
    )
    with pytest.raises(NotImplementedError, match="no parameter-shift gradient"):
        qaoa.run()


def test_qng_run_with_checkpointing_raises_upfront(toy_vqe, tmp_path):
    # A non-checkpointable optimizer + a checkpoint_dir must fail before any
    # optimization, not mid-run at the first checkpoint attempt.
    toy_vqe.optimizer = QNGOptimizer(metric_estimator=FubiniStudyMetricEstimator())
    with pytest.raises(ValueError, match="does not support checkpointing"):
        toy_vqe.run(checkpoint_config=CheckpointConfig(checkpoint_dir=tmp_path))


def test_pce_runs_under_fubini_study_qng(default_test_simulator):
    """PCE — whose loss is a classical QUBO, not <cost_hamiltonian> — optimizes
    end-to-end under QNG with the FS metric."""
    default_test_simulator.set_seed(1997)
    pce = PCE(
        problem=BinaryOptimizationProblem(np.array([[1.0, 0.2], [0.2, 2.0]])),
        ansatz=GenericLayerAnsatz([RYGate]),
        n_layers=1,
        backend=default_test_simulator,
        optimizer=QNGOptimizer(
            metric_estimator=FubiniStudyMetricEstimator(), step_size=0.1
        ),
        max_iterations=3,
        seed=1997,
    )
    pce.run(perform_final_computation=False)
    assert len(pce.losses_history) == 3
    assert np.isfinite(pce.best_loss)


def test_pullback_metric_is_symmetric_psd_low_rank(toy_vqe):
    """The metric computed on a real backend is symmetric, PSD, and rank <= v."""
    toy_vqe.backend.set_seed(7)
    n_params = set_unit_shift_rule(toy_vqe)
    params = np.linspace(0.1, 1.0, n_params)

    evaluators = PullbackMetricEstimator().bind(toy_vqe)
    grad = evaluators["jac"](params)
    metric = evaluators["metric_fn"](params)

    assert metric.shape == (n_params, n_params)
    np.testing.assert_allclose(metric, metric.T, atol=1e-10)
    assert np.linalg.eigvalsh(metric).min() >= -1e-8
    assert np.linalg.matrix_rank(metric, tol=1e-6) <= len(toy_vqe.cost_hamiltonian)

    # Fused energy gradient agrees with the standard parameter-shift gradient
    # (both analytic on an expval backend).
    grad_ref = toy_vqe._evaluate_gradient_at(params)
    np.testing.assert_allclose(grad, grad_ref, atol=1e-3)


# --------------------------------------------------------------------------- #
# End-to-end convergence
# --------------------------------------------------------------------------- #


@pytest.mark.e2e
def test_qng_vqe_h2_converges(qp, default_test_simulator):
    """QNG drives a HartreeFock-ansatz VQE to the H2 ground-state energy."""
    seed = 1997
    default_test_simulator.set_seed(seed)
    molecule = qp.qchem.Molecule(
        ["H", "H"], np.array([[0.0, 0.0, -0.6614], [0.0, 0.0, 0.6614]])
    )

    vqe = VQE(
        MolecularProblem.from_molecule(molecule),
        ansatz=HartreeFockAnsatz(),
        n_layers=1,
        optimizer=QNGOptimizer(step_size=0.02),
        max_iterations=15,
        backend=default_test_simulator,
        seed=seed,
    )
    vqe.run()

    hartree_fock_energy = vqe.losses_history[0]["0"]
    exact_energy = (
        np.linalg.eigvalsh(vqe.cost_hamiltonian.to_matrix()).min() + vqe.loss_constant
    )
    assert len(vqe.losses_history) == 15
    assert vqe.best_loss < hartree_fock_energy - 0.01
    assert vqe.best_loss == pytest.approx(exact_energy, abs=1e-6)
