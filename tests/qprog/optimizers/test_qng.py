# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Quantum Natural Gradient optimizer and its pullback metric."""

from dataclasses import replace
from typing import Any, cast

import networkx as nx
import numpy as np
import pennylane as qp
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RYGate, RZGate
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import MetaCircuit
from divi.hamiltonians import QDrift
from divi.pipeline import CircuitPreprocessor
from divi.pipeline._result_keys_operations import average_by_param_set
from divi.pipeline.abc import ContractViolation
from divi.qprog import PCE, QAOA, VQE, CustomVQA, FubiniStudyMetricEstimator
from divi.qprog._metrics import (
    PullbackMetricEstimator,
    _all_terms_preprocessor,
    _fs_block_prefix,
    _fs_prefix_labels_preprocessor,
    _measure_prefix_paulis,
    _split_into_terms,
    _split_observable_into_terms,
    _term_expectations,
)
from divi.qprog.algorithms import GenericLayerAnsatz, HartreeFockAnsatz
from divi.qprog.checkpointing import CheckpointConfig
from divi.qprog.optimizers import QNGOptimizer
from divi.qprog.problems import BinaryOptimizationProblem, MaxCutProblem
from divi.qprog.variational_quantum_algorithm import _compute_parameter_shift_mask

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


def test_natural_gradient_pinv_solver_handles_singular_metric():
    opt = QNGOptimizer(solver="pinv", rcond=1e-9)
    grad = np.array([1.0, 1.0])
    # Rank-1 (singular) metric: pinv must still return a finite direction.
    metric = np.outer([1.0, 0.0], [1.0, 0.0])
    delta = opt._natural_gradient(grad, metric)
    np.testing.assert_allclose(delta, [1.0, 0.0])


def test_natural_gradient_tikhonov_singular_metric_raises_actionably():
    # tikhonov + λ=0 on a rank-deficient metric must surface an actionable error,
    # not an opaque scipy LinAlgError. (Full-rank λ=0 still works — see above.)
    opt = QNGOptimizer(regularization=0.0, scale_regularization=False)
    singular = np.outer([1.0, 0.0], [1.0, 0.0])
    with pytest.raises(np.linalg.LinAlgError, match="rank-deficient"):
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


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_non_finite_update_raises():
    opt = QNGOptimizer(solver="pinv")
    with pytest.raises(FloatingPointError):
        opt._natural_gradient(np.array([np.inf, 0.0]), np.eye(2))


def test_optimize_requires_jac_and_metric():
    opt = QNGOptimizer()
    with pytest.raises(ValueError, match="requires both"):
        opt.optimize(
            cost_fn=lambda x: 0.0,
            initial_params=np.zeros(2),
            jac=None,
            metric_fn=lambda p: np.eye(2),
        )
    with pytest.raises(ValueError, match="requires both"):
        opt.optimize(
            cost_fn=lambda x: 0.0,
            initial_params=np.zeros(2),
            jac=lambda p: np.zeros(2),
            metric_fn=None,
        )


def test_optimize_converges_on_quadratic():
    # f(x) = 0.5 x^T A x; grad = A x. With metric == A, one full natural-gradient
    # step lands exactly on the minimum (Newton step).
    a_matrix = np.array([[3.0, 0.0], [0.0, 1.0]])

    opt = QNGOptimizer(step_size=1.0, regularization=1e-9, scale_regularization=False)
    result = opt.optimize(
        cost_fn=lambda x: 0.5 * x @ a_matrix @ x,
        initial_params=np.array([1.0, 1.0]),
        max_iterations=20,
        jac=lambda x: a_matrix @ x,
        metric_fn=lambda x: a_matrix,
    )

    np.testing.assert_allclose(result.x.squeeze(), [0.0, 0.0], atol=1e-8)
    assert result.fun[0] == pytest.approx(0.0, abs=1e-12)
    assert result.success


def test_callback_receives_2d_x_and_1d_fun():
    captured = []
    opt = QNGOptimizer(step_size=0.5)
    opt.optimize(
        cost_fn=lambda x: float(x @ x),
        initial_params=np.array([1.0, 2.0]),
        callback_fn=lambda res: captured.append((res.x, res.fun)),
        max_iterations=3,
        jac=lambda x: 2 * x,
        metric_fn=lambda x: np.eye(2),
    )
    assert len(captured) == 3
    for x, fun in captured:
        assert x.shape == (1, 2)
        assert fun.shape == (1,)


def test_callback_stop_iteration_propagates():
    def raising_callback(_res):
        raise StopIteration

    opt = QNGOptimizer()
    with pytest.raises(StopIteration):
        opt.optimize(
            cost_fn=lambda x: 0.0,
            initial_params=np.zeros(2),
            callback_fn=raising_callback,
            max_iterations=5,
            jac=lambda x: np.zeros(2),
            metric_fn=lambda x: np.eye(2),
        )


def test_checkpointing_is_not_supported(tmp_path):
    opt = QNGOptimizer()
    with pytest.raises(NotImplementedError):
        opt.get_config()
    with pytest.raises(NotImplementedError):
        opt.save_state(tmp_path)
    with pytest.raises(NotImplementedError):
        QNGOptimizer.load_state(tmp_path)
    opt.reset()  # no-op; must not raise


@pytest.mark.parametrize(
    "kwargs",
    [
        {"step_size": 0.0},
        {"regularization": -1.0},
        {"solver": "bogus"},
    ],
)
def test_invalid_constructor_args(kwargs):
    with pytest.raises(ValueError):
        QNGOptimizer(**kwargs)


def test_qng_reports_no_checkpointing_support():
    # QNG's only state is the parameter vector, already persisted by the program.
    assert QNGOptimizer().supports_checkpointing is False


def test_qng_zero_iterations_raises():
    """Zero steps would return success with an infinite loss — reject it."""
    with pytest.raises(ValueError, match="max_iterations must be >= 1"):
        QNGOptimizer().optimize(
            lambda x: 0.0,
            initial_params=np.zeros(2),
            max_iterations=0,
            jac=lambda x: np.zeros(2),
            metric_fn=lambda x: np.eye(2),
        )


# --------------------------------------------------------------------------- #
# Pullback metric: assembly + integration
# --------------------------------------------------------------------------- #


@pytest.fixture
def toy_vqe(default_test_simulator, default_optimizer):
    """A small 2-qubit VQE with a generic RY-RZ ansatz (no chemistry inputs)."""
    hamiltonian = SparsePauliOp.from_list([("ZI", 0.5), ("IZ", -0.3), ("XX", 0.2)])
    return VQE(
        hamiltonian=hamiltonian,
        ansatz=GenericLayerAnsatz([RYGate, RZGate]),
        n_layers=1,
        backend=default_test_simulator,
        optimizer=default_optimizer,
        seed=1997,
    )


def test_pullback_metric_assembly(dummy_simulator, default_optimizer, monkeypatch):
    """grad and G are assembled correctly from the per-term Jacobian. Uses a
    sampling backend (one qwc run) and injects per-term expectations by patching
    the measurement seam, so only the assembly math is under test."""
    vqe = VQE(
        hamiltonian=SparsePauliOp.from_list([("ZI", 0.5), ("IZ", -0.3), ("XX", 0.2)]),
        ansatz=GenericLayerAnsatz([RYGate, RZGate]),
        n_layers=1,
        backend=dummy_simulator,  # supports_expval=False -> single qwc run
        optimizer=default_optimizer,
        seed=1997,
    )
    coeffs = np.array([0.5, -0.3, 0.2])
    n_params = vqe.n_layers * vqe.n_params_per_layer
    rng = np.random.default_rng(0)
    fake = rng.standard_normal((2 * n_params, len(coeffs)))

    vqe._grad_shift_mask = np.zeros((2 * n_params, n_params))
    monkeypatch.setattr(
        "divi.qprog._metrics._term_expectations",
        lambda _program, _param_sets: {(("circuit", 0),): (fake, coeffs)},
    )

    evaluators = PullbackMetricEstimator().bind(vqe)
    grad = evaluators["jac"](np.zeros(n_params))
    metric = evaluators["metric_fn"](np.zeros(n_params))

    jac = 0.5 * (fake[0::2] - fake[1::2])
    np.testing.assert_allclose(grad, jac @ coeffs)
    np.testing.assert_allclose(metric, (jac * coeffs**2) @ jac.T)
    np.testing.assert_allclose(metric, metric.T, atol=1e-12)
    assert np.linalg.eigvalsh(metric).min() >= -1e-9


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
    result = cast(
        dict[tuple, Any],
        toy_vqe.evaluate(params, preprocessor, preserve_keys=True),
    )
    indexed = average_by_param_set(
        result,
        lambda value: np.asarray(value, dtype=np.float64).reshape(-1),
    )
    reconstructed = float(indexed[0] @ coeffs) + toy_vqe.loss_constant

    energy = toy_vqe._evaluate_cost_param_sets(params)[0]
    assert reconstructed == pytest.approx(energy, abs=0.1)


def test_fubini_study_metric_smoke(default_test_simulator, default_optimizer):
    """The FS metric runs end-to-end and is a symmetric PSD matrix."""
    vqe = VQE(
        hamiltonian=SparsePauliOp.from_list([("ZI", 0.5), ("IZ", 0.5)]),
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


def test_fubini_study_matches_pennylane_block_diag(
    default_test_simulator, default_optimizer
):
    """The block-diagonal FS metric agrees with PennyLane's reference, including
    off-diagonal blocks from an entangled prefix."""
    n, layers = 3, 2
    m = n * layers
    weights = [Parameter(f"w{i}") for i in range(m)]
    qc = QuantumCircuit(n, n)
    for layer in range(layers):
        for q in range(n):
            qc.ry(weights[layer * n + q], q)
        for q in range(n - 1):
            qc.cx(q, q + 1)
    qc.measure(range(n), range(n))

    program = CustomVQA(
        qscript=qc, backend=default_test_simulator, optimizer=default_optimizer
    )
    program.backend.set_seed(1997)

    dev = qp.device("default.qubit", wires=n)

    @qp.qnode(dev)
    def circuit(w):
        for layer in range(layers):
            for q in range(n):
                qp.RY(w[layer * n + q], wires=q)
            for q in range(n - 1):
                qp.CNOT([q, q + 1])
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

    np.testing.assert_allclose(divi_metric, pl_metric[np.ix_(order, order)], atol=1e-2)


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

    captured_observables = []

    def fake_run_metric_by_branch(_program, preprocessor, _param_sets):
        # The pure single-arg transform builds the multi-observable prefix meta.
        out_meta = preprocessor.preprocess(sampled_meta)
        captured_observables.append([str(o.paulis[0]) for o in out_meta.observable])
        return {(("ham", 0),): {0: np.array([0.25])}}

    monkeypatch.setattr("divi.qprog._metrics._fs_blocks", fake_fs_blocks)
    monkeypatch.setattr(
        "divi.qprog._metrics._run_metric_by_branch", fake_run_metric_by_branch
    )

    program = type(
        "Program",
        (),
        {
            "cost_circuit": sampled_meta,
            "_post_spec_batch": lambda self: {(("ham", 0),): sampled_meta},
        },
    )()
    exp_by_branch, branch_data = _measure_prefix_paulis(
        program, [theta], np.array([0.1]), block_id=0
    )

    assert sampled_meta in seen
    assert captured_observables == [["X"]]
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
    with pytest.raises(ContractViolation, match="cost Hamiltonian"):
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
        optimizer=default_optimizer,
    )
    with pytest.raises(ContractViolation, match="supervised"):
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


def test_metric_preprocessors_are_cacheable(dummy_simulator, default_optimizer):
    """The consolidated metric transforms are pure, so their preprocessors carry
    stable cache keys and ``_build_preprocessor_pipeline`` reuses one pipeline
    object across iterations (its forward cache survives)."""
    vqe = VQE(
        hamiltonian=SparsePauliOp.from_list([("ZI", 0.5), ("IZ", -0.3), ("XX", 0.2)]),
        ansatz=GenericLayerAnsatz([RYGate]),
        n_layers=1,
        backend=dummy_simulator,
        optimizer=default_optimizer,
    )
    assert _all_terms_preprocessor().cache_key == "metric-terms"
    assert _fs_prefix_labels_preprocessor(0, ()).cache_key == ("metric-prefix", 0)

    # Fresh-but-equal preprocessors hit the cache -> same pipeline object.
    p1 = vqe._build_preprocessor_pipeline(_all_terms_preprocessor())
    p2 = vqe._build_preprocessor_pipeline(_all_terms_preprocessor())
    assert p1 is p2


def test_fs_block_prefix_rejects_out_of_range_block():
    """A branch with fewer FS blocks than the requested index fails loudly."""
    a = Parameter("a")
    qc = QuantumCircuit(1)
    qc.ry(a, 0)
    meta = MetaCircuit(circuit_bodies=(((), circuit_to_dag(qc)),), parameters=(a,))
    with pytest.raises(ContractViolation, match="fewer Fubini-Study blocks"):
        _fs_block_prefix(meta, block_id=5, reference_prefix_param_names=())


def test_fs_block_prefix_rejects_prefix_param_layout_mismatch():
    """A branch whose prefix-parameter layout differs from the reference ansatz
    fails loudly rather than mis-aligning the metric."""
    a = Parameter("a")
    qc = QuantumCircuit(1)
    qc.ry(a, 0)
    meta = MetaCircuit(circuit_bodies=(((), circuit_to_dag(qc)),), parameters=(a,))
    with pytest.raises(ContractViolation, match="different Fubini-Study prefix"):
        _fs_block_prefix(meta, block_id=0, reference_prefix_param_names=("z",))


def test_term_expectations_rejects_term_count_mismatch(
    dummy_simulator, default_optimizer, monkeypatch
):
    """A branch whose measured term count disagrees with its coefficient count
    fails loudly rather than mis-assembling the metric."""
    vqe = VQE(
        hamiltonian=SparsePauliOp.from_list([("ZI", 0.5)]),
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
    with pytest.raises(ContractViolation, match="Per-term measurement count"):
        _term_expectations(vqe, np.zeros((1, vqe.n_layers * vqe.n_params_per_layer)))


def test_all_terms_preprocessor_rejects_branch_without_observable():
    """The pullback term-expansion transform raises on a branch that carries no
    single loss observable, rather than silently producing a wrong metric."""
    qc = QuantumCircuit(1)
    qc.rx(Parameter("a"), 0)
    meta = MetaCircuit(circuit_bodies=(((), circuit_to_dag(qc)),), observable=None)

    with pytest.raises(ContractViolation, match="one loss observable"):
        _split_observable_into_terms(meta)


def test_vqe_runs_under_fubini_study_qng(toy_vqe):
    """VQE optimizes end-to-end under QNG with the FS metric (gradient from the
    program's parameter-shift, metric from FS)."""
    toy_vqe.backend.set_seed(1997)
    toy_vqe.optimizer = QNGOptimizer(
        metric_estimator=FubiniStudyMetricEstimator(), step_size=0.2
    )
    toy_vqe.max_iterations = 5
    toy_vqe.run(perform_final_computation=False)
    assert len(toy_vqe.losses_history) >= 1


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


def test_qaoa_qdrift_qng_runs_end_to_end(dummy_simulator):
    """QNG + QDrift completes and produces a solution."""
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
    qaoa.run()
    assert qaoa.best_probs


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
    assert len(pce.losses_history) >= 1


def test_pullback_metric_is_symmetric_psd_low_rank(toy_vqe):
    """The metric computed on a real backend is symmetric, PSD, and rank <= v."""
    toy_vqe.backend.set_seed(7)
    n_params = toy_vqe.n_layers * toy_vqe.n_params_per_layer
    toy_vqe._grad_shift_mask = _compute_parameter_shift_mask(n_params)
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
def test_qng_vqe_h2_converges(default_test_simulator):
    """QNG drives a HartreeFock-ansatz VQE to the H2 ground-state energy."""
    seed = 1997
    default_test_simulator.set_seed(seed)
    molecule = qp.qchem.Molecule(
        ["H", "H"], np.array([[0.0, 0.0, -0.6614], [0.0, 0.0, 0.6614]])
    )

    vqe = VQE(
        molecule=molecule,
        ansatz=HartreeFockAnsatz(),
        n_layers=1,
        optimizer=QNGOptimizer(step_size=0.2),
        max_iterations=15,
        backend=default_test_simulator,
        seed=seed,
    )
    vqe.run()

    assert len(vqe.losses_history) == 15
    assert vqe.best_loss == pytest.approx(-1.1398024781381293, abs=0.3)
