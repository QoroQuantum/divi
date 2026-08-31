# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pytest
from qiskit.circuit.library import CXGate, RYGate, RZGate
from qiskit.quantum_info import SparsePauliOp

from divi.pipeline import CircuitPreprocessor, ContractViolation
from divi.qprog import (
    QNN,
    AngleEmbedding,
    GenericLayerAnsatz,
    ZZFeatureMap,
)
from divi.qprog.algorithms._data_binding import DataBindingMixin
from divi.qprog.optimizers import QNSPSAOptimizer, ScipyMethod, ScipyOptimizer
from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm
from tests.qprog._program_contracts import (
    ObservableMeasuringContractsBase,
    verify_cost_circuit,
)


@pytest.fixture
def simple_ansatz():
    return GenericLayerAnsatz(
        gate_sequence=[RYGate, RZGate],
        entangler=CXGate,
        entangling_layout="linear",
    )


@pytest.fixture
def simple_feature_map():
    return AngleEmbedding(rotation="Y")


@pytest.fixture
def two_qubit_observable():
    return SparsePauliOp.from_list([("ZI", 1.0)])


@pytest.fixture
def feature_batch_2x2():
    return np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])


@pytest.fixture
def make_qnn(
    simple_feature_map,
    simple_ansatz,
    two_qubit_observable,
    feature_batch_2x2,
    dummy_simulator,
    default_optimizer,
):
    """Build a 2-qubit QNN from the standard test building blocks.

    Each building block is a default that any test can override by keyword
    (e.g. ``make_qnn(observable=None)`` for the parity default,
    ``make_qnn(backend=default_test_simulator, n_layers=2)`` for e2e runs).
    """
    defaults = {
        "n_qubits": 2,
        "feature_map": simple_feature_map,
        "ansatz": simple_ansatz,
        "observable": two_qubit_observable,
        "feature_batch": feature_batch_2x2,
        "backend": dummy_simulator,
        "optimizer": default_optimizer,
    }

    def _make(**overrides):
        return QNN(**{**defaults, **overrides})

    return _make


class TestInitialization:
    def test_basic_initialization(self, make_qnn):
        program = make_qnn(n_layers=2)

        assert program.n_qubits == 2
        assert program.n_layers == 2
        # GenericLayerAnsatz with two single-qubit gates × 2 qubits = 4 params/layer
        assert program.n_params_per_layer == 4
        assert program._n_data_params == 2
        assert program._n_weight_params == 8
        assert program.feature_batch.shape == (4, 2)
        verify_cost_circuit(program)

    def test_default_observable_is_all_z_parity(self, make_qnn):
        """Omitting ``observable`` falls back to ``Z ⊗ Z ⊗ … ⊗ Z``."""
        program = make_qnn(observable=None)
        labels = [str(p) for p in program.cost_hamiltonian.paulis]
        assert labels == ["ZZ"]
        assert program.loss_constant == 0.0

    def test_loss_constant_extracted_from_observable(self, make_qnn):
        """Identity terms in the observable land on ``loss_constant``."""
        program = make_qnn(
            observable=SparsePauliOp.from_list([("ZI", 1.0), ("II", 3.0)])
        )
        assert program.loss_constant == pytest.approx(3.0)
        assert program.cost_hamiltonian.size == 1

    def test_meta_circuit_parameter_ordering(self, make_qnn):
        program = make_qnn()
        params = program.cost_circuit.parameters
        # Data params come first, then weight params
        assert len(params) == program._n_data_params + program._n_weight_params
        data_names = {str(p) for p in params[: program._n_data_params]}
        weight_names = {str(p) for p in params[program._n_data_params :]}
        assert all(n.startswith("x") for n in data_names)
        assert all(n.startswith("w") for n in weight_names)


class TestConstructionValidation:
    @pytest.mark.parametrize(
        "bad_kwarg,bad_value,match",
        [
            ("n_layers", 0, "n_layers must be positive"),
        ],
    )
    def test_layer_counts_must_be_positive(self, bad_kwarg, bad_value, match, make_qnn):
        with pytest.raises(ValueError, match=match):
            make_qnn(**{bad_kwarg: bad_value})

    @pytest.mark.parametrize("bad_n_qubits", [0, -1])
    def test_n_qubits_must_be_positive(self, bad_n_qubits, make_qnn):
        with pytest.raises(ValueError, match="n_qubits must be positive"):
            make_qnn(
                n_qubits=bad_n_qubits,
                feature_batch=np.zeros((1, max(bad_n_qubits, 1))),
            )

    def test_wrong_observable_qubits(self, make_qnn):
        with pytest.raises(ValueError, match="acts on 4 qubits"):
            make_qnn(observable=SparsePauliOp.from_list([("ZIII", 1.0)]))

    def test_non_sparsepauli_observable_rejected(self, make_qnn):
        with pytest.raises(TypeError, match="observable must be a SparsePauliOp"):
            make_qnn(observable="ZI")  # a string, not a SparsePauliOp

    def test_callable_loss_reduction_accepted(self, make_qnn):
        def reduction(arr):
            return float(np.max(arr))

        program = make_qnn(loss_reduction=reduction)
        assert program.loss_reduction is reduction
        assert callable(program._loss_reduction_fn)

    def test_loss_fn_without_labels_warns_at_caller(self, make_qnn):
        # loss_fn is ignored without labels; the warning must be attributed to
        # the user's constructor call, not to a frame inside divi.
        with pytest.warns(UserWarning, match="loss_fn is ignored") as record:
            make_qnn(loss_fn=lambda pred, label: (pred - label) ** 2)
        ignored = [w for w in record if "loss_fn is ignored" in str(w.message)]
        assert ignored and ignored[0].filename == __file__

    def test_feature_batch_wrong_columns(self, make_qnn):
        bad_batch = np.array([[0.1, 0.2, 0.3]])  # 3 columns but only 2 data params
        with pytest.raises(ValueError, match="binds 2 data parameters"):
            make_qnn(feature_batch=bad_batch)

    def test_feature_batch_1d_rejected(self, make_qnn):
        with pytest.raises(ValueError, match="feature_batch must be 2D"):
            make_qnn(feature_batch=np.array([0.1, 0.2]))

    def test_feature_batch_empty_rejected(self, make_qnn):
        with pytest.raises(ValueError, match="at least one sample"):
            make_qnn(feature_batch=np.empty((0, 2)))

    def test_non_feature_map_rejected(self, make_qnn):
        with pytest.raises(TypeError, match="feature_map must be"):
            make_qnn(feature_map="not a feature map")  # type: ignore[arg-type]

    def test_non_ansatz_rejected(self, make_qnn):
        with pytest.raises(TypeError, match="ansatz must be"):
            make_qnn(ansatz="not an ansatz")  # type: ignore[arg-type]

    def test_invalid_loss_reduction(self, make_qnn):
        with pytest.raises(ValueError, match="loss_reduction must be"):
            make_qnn(loss_reduction="median")  # type: ignore[arg-type]

    def test_constant_only_observable_rejected(self, make_qnn):
        with pytest.raises(ValueError, match="only constant terms"):
            make_qnn(observable=SparsePauliOp.from_list([("II", 5.0)]))

    def test_labels_default_unsupervised(self, make_qnn):
        program = make_qnn()
        assert program.labels is None
        assert program._sample_loss_fn is None

    def test_labels_stored_when_supervised(self, make_qnn):
        program = make_qnn(labels=[0.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_equal(program.labels, [0.0, 1.0, 0.0, 1.0])
        assert callable(program._sample_loss_fn)

    def test_labels_wrong_length_rejected(self, make_qnn):
        with pytest.raises(ValueError, match="labels has 2 entries but feature_batch"):
            make_qnn(labels=[0.0, 1.0])

    def test_dry_run_rejects_a_batch_that_no_longer_matches_the_labels(self, make_qnn):
        """``feature_batch`` is public, so swapping in a split after construction
        outruns the constructor's check. A preview that passes here is followed by a
        run that fails once every circuit has been submitted."""
        program = make_qnn(labels=[0.0, 1.0, 0.0, 1.0])
        program.dry_run()  # consistent to begin with

        program.feature_batch = np.asarray(program.feature_batch)[:2]
        with pytest.raises(ValueError, match="labels has 4 entries"):
            program.dry_run()

    @pytest.mark.parametrize(
        "bad_batch, match",
        [
            (None, "feature_batch is None"),
            (np.zeros((0, 2)), "at least one sample"),
            (np.zeros((4, 99)), "columns but the circuit binds"),
            (np.zeros(4), "must be 2D"),
        ],
        ids=["none", "empty", "wrong-width", "one-dimensional"],
    )
    def test_dry_run_rejects_a_batch_the_constructor_would_have(
        self, make_qnn, bad_batch, match
    ):
        """``feature_batch`` is public, so any state the constructor rejects can be
        assigned afterwards. ``None`` is the dangerous one: it drops the data axis
        and the report under-counts by the whole batch instead of failing."""
        program = make_qnn()
        program.feature_batch = bad_batch
        with pytest.raises(ValueError, match=match):
            program.dry_run()

    def test_dry_run_rejects_labels_the_program_cannot_consume(self, make_qnn):
        """The per-sample loss is resolved at construction from the labels given
        then, so labels assigned afterwards have nothing to consume them. The
        preview would otherwise report ``supervised: False`` for a program holding
        labels, and the run fails once it reduces them."""
        program = make_qnn()  # unsupervised: no sample loss resolved
        program.labels = np.array([0.0, 1.0, 0.0, 1.0])
        with pytest.raises(ValueError, match="no per-sample loss"):
            program.dry_run()

    def test_dry_run_rejects_labels_cleared_after_construction(self, make_qnn):
        """Clearing labels leaves the per-sample loss in place, so the run trains
        the unsupervised objective while the report still calls it supervised —
        the one corruption here that yields a wrong answer instead of a failure."""
        program = make_qnn(labels=[0.0, 1.0, 0.0, 1.0])
        program.dry_run()  # supervised and consistent to begin with

        program.labels = None
        with pytest.raises(ValueError, match="labels is now None"):
            program.dry_run()

    def test_supervision_is_visible_in_the_report(self, make_qnn):
        """Supervision decides which optimizers are legal at all, so two reports
        that differ in it must not render identically."""
        supervised = make_qnn(labels=[0.0, 1.0, 0.0, 1.0]).dry_run()["cost"]
        unsupervised = make_qnn().dry_run()["cost"]

        def flag(report):
            data_stage = next(s for s in report.stages if s.name == "DataBindingStage")
            return data_stage.metadata["supervised"]

        assert flag(supervised) is True
        assert flag(unsupervised) is False

    def test_no_metric_estimator_is_recommended_for_a_supervised_batch(self, make_qnn):
        """Each metric used to redirect to another that rejects the same program,
        so following the advice went in a circle. For a labelled batch there is no
        metric alternative and the message must say so."""
        program = make_qnn(labels=[0.0, 1.0, 0.0, 1.0], optimizer=QNSPSAOptimizer())
        # Previewing refuses for the same reason run() does, and says the same thing.
        for call in (program.dry_run, lambda: program.run(max_iterations=1)):
            with pytest.raises(ContractViolation) as exc:
                call()
            assert "no metric estimator" in str(exc.value)
            assert "SPSAOptimizer" in str(exc.value)

    def test_invalid_loss_fn_rejected(self, make_qnn):
        with pytest.raises(ValueError, match="loss_fn must be"):
            make_qnn(
                labels=[0.0, 1.0, 0.0, 1.0],
                loss_fn="huber",  # type: ignore[arg-type]
            )

    def test_out_of_range_labels_warn_for_default_observable(self, make_qnn):
        # Default parity observable reads out in [-1, 1]; {0, 2} labels can't be
        # matched, so squared error floors above zero — warn the user.
        with pytest.warns(UserWarning, match=r"reads out in \[-1, 1\]"):
            make_qnn(observable=None, labels=[0.0, 2.0, 0.0, 2.0])

    def test_loss_fn_without_labels_warns(self, make_qnn):
        with pytest.warns(UserWarning, match="loss_fn"):
            make_qnn(loss_fn=lambda pred, label: abs(pred - label))


class TestDryRun:
    def test_does_not_emit_progress(self, make_qnn):
        program = make_qnn()
        emitted = []

        with program._bind_progress_emitter(emitted.append):
            program.dry_run()

        assert emitted == []

    def test_data_axis_appears_with_correct_factor(self, make_qnn, feature_batch_2x2):
        """dry_run must surface the data axis with ``n_samples`` fan-out."""
        program = make_qnn(n_layers=2)
        reports = program.dry_run()
        data_stage = next(s for s in reports["cost"].stages if s.axis == "data_sample")
        assert data_stage.factor == feature_batch_2x2.shape[0]
        assert data_stage.metadata["n_samples"] == feature_batch_2x2.shape[0]

    def test_total_circuits_includes_data_and_param_set_axes(
        self, make_qnn, feature_batch_2x2
    ):
        """Total dry-run count = n_samples × n_param_sets for a single-group obs."""
        program = make_qnn()
        reports = program.dry_run()
        expected = feature_batch_2x2.shape[0] * program.optimizer.n_param_sets
        assert reports["cost"].total_circuits == expected


class TestQNNPipelines:
    def test_no_sample_pipeline(self, make_qnn):
        program = make_qnn()
        names = [protocol.name for protocol in program._preprocessors()]
        # Metric is built on demand, never enumerated.
        assert "sample" not in names
        assert "cost" in names

    def test_data_binding_injected_into_cost_and_metric_pipeline(self, make_qnn):
        program = make_qnn()
        # The data axis fans out in both the cost pipeline and the on-demand
        # metric pipeline.
        pipelines = (
            program._build_preprocessor_pipeline(program.cost_preprocessor()),
            program._build_preprocessor_pipeline(CircuitPreprocessor("metric")),
        )
        for pipeline in pipelines:
            types = [type(s).__name__ for s in pipeline.stages]
            assert "DataBindingStage" in types


@pytest.mark.e2e
def test_batch_loss_matches_per_sample_mean(
    make_qnn, feature_batch_2x2, default_test_simulator
):
    """End-to-end integration check: the batched cost equals the mean of
    per-sample costs, modulo shot noise from independent simulator runs.

    The DataBindingStage reduce invariant is asserted directly in
    ``tests/pipeline/stages/test_data_binding_stage.py``; this is the
    QNN-end-to-end sanity that the stage is wired into the cost pipeline.
    """
    weights = np.array([[0.5, 1.0, 1.5, 2.0]])

    default_test_simulator.set_seed(1997)
    batched = make_qnn(backend=default_test_simulator, n_layers=1, seed=1997)
    batched_loss = batched._evaluate_cost_param_sets(weights)[0]

    per_sample_losses = []
    for row in feature_batch_2x2:
        default_test_simulator.set_seed(1997)
        single = make_qnn(
            feature_batch=row[None, :],
            backend=default_test_simulator,
            n_layers=1,
            seed=1997,
        )
        per_sample_losses.append(single._evaluate_cost_param_sets(weights)[0])

    # Tolerance reflects shot noise from independent simulator submissions:
    # each per-sample call samples ``shots`` fresh outcomes, so the per-sample
    # mean drifts from the batched mean by ~1/sqrt(shots) per sample.
    np.testing.assert_allclose(
        batched_loss, float(np.mean(per_sample_losses)), atol=0.05
    )


def test_identity_constant_scales_with_sample_count_under_sum_reduction(
    make_qnn, feature_batch_2x2, default_test_simulator
):
    """Regression for the post-hoc ``loss_constant`` add at the VQA layer.

    Under ``loss_reduction="sum"`` an Identity term in the observable must
    contribute ``n_samples * c`` to the final loss (one per sample), not
    just ``c`` (one global post-reduction add). The pre-fix code added the
    constant once at the VQA layer after the sample axis had already been
    collapsed; with N=4 samples and c=2.5 the bug would manifest as a
    7.5-unit underestimate.
    """
    weights = np.array([[0.5, 1.0, 1.5, 2.0]])
    n_samples = feature_batch_2x2.shape[0]
    constant = 2.5

    default_test_simulator.set_seed(1997)
    no_const = make_qnn(
        backend=default_test_simulator,
        n_layers=1,
        seed=1997,
        loss_reduction="sum",
    )
    base_loss = no_const._evaluate_cost_param_sets(weights)[0]

    default_test_simulator.set_seed(1997)
    with_const = make_qnn(
        observable=SparsePauliOp.from_list([("ZI", 1.0), ("II", constant)]),
        backend=default_test_simulator,
        n_layers=1,
        seed=1997,
        loss_reduction="sum",
    )
    shifted_loss = with_const._evaluate_cost_param_sets(weights)[0]

    # With per-sample constant folding the gap is exactly ``n_samples * c``;
    # the pre-fix bug would yield ``c`` (off by ``(n_samples - 1) * c``).
    assert with_const.loss_constant == pytest.approx(constant)
    np.testing.assert_allclose(
        shifted_loss - base_loss, n_samples * constant, atol=1e-9
    )


@pytest.mark.e2e
def test_supervised_loss_matches_manual_mse(
    make_qnn, feature_batch_2x2, default_test_simulator
):
    """A supervised QNN's batched loss equals the MSE of per-sample predictions
    against the labels, modulo shot noise.

    Mirrors ``test_batch_loss_matches_per_sample_mean`` but with labels: the
    per-sample unsupervised prediction is the readout, and the supervised loss
    must be ``mean((prediction_i - label_i) ** 2)``.
    """
    weights = np.array([[0.5, 1.0, 1.5, 2.0]])
    labels = np.array([0.3, -0.2, 0.5, -0.6])

    default_test_simulator.set_seed(1997)
    supervised = make_qnn(
        backend=default_test_simulator,
        n_layers=1,
        seed=1997,
        labels=labels,
        loss_fn="squared_error",
    )
    supervised_loss = supervised._evaluate_cost_param_sets(weights)[0]

    predictions = []
    for row in feature_batch_2x2:
        default_test_simulator.set_seed(1997)
        single = make_qnn(
            feature_batch=row[None, :],
            backend=default_test_simulator,
            n_layers=1,
            seed=1997,
        )
        predictions.append(single._evaluate_cost_param_sets(weights)[0])

    expected = float(np.mean((np.array(predictions) - labels) ** 2))
    np.testing.assert_allclose(supervised_loss, expected, atol=0.05)


@pytest.mark.e2e
class TestE2E:
    def test_optimization_loop_runs(self, make_qnn, default_test_simulator):
        default_test_simulator.set_seed(1997)
        program = make_qnn(
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=3,
            n_layers=2,
            seed=1997,
        )
        program.run(perform_final_computation=False)
        assert len(program.losses_history) == 3
        assert math.isfinite(program.best_loss)
        assert program.best_params.shape == (
            program.n_params_per_layer * program.n_layers,
        )

    def test_zz_feature_map_runs(self, make_qnn, default_test_simulator):
        default_test_simulator.set_seed(1997)
        program = make_qnn(
            feature_map=ZZFeatureMap(entangling_layout="linear"),
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=2,
            seed=1997,
        )
        program.run(perform_final_computation=False)
        assert len(program.losses_history) == 2

    def test_supervised_optimization_runs(self, make_qnn, default_test_simulator):
        default_test_simulator.set_seed(1997)
        program = make_qnn(
            backend=default_test_simulator,
            optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
            max_iterations=3,
            n_layers=2,
            seed=1997,
            labels=[1.0, -1.0, 1.0, -1.0],
            loss_fn="squared_error",
        )
        program.run(perform_final_computation=False)
        assert len(program.losses_history) == 3
        # Squared error is non-negative, so the aggregated mean loss is too.
        assert math.isfinite(program.best_loss)
        assert program.best_loss >= 0.0


class TestPredictValidation:
    """predict() input checks that raise before any measurement runs."""

    def test_predict_before_training_raises(self, make_qnn, feature_batch_2x2):
        program = make_qnn()
        with pytest.raises(RuntimeError, match="trained weights"):
            program.predict(feature_batch_2x2)

    def test_predict_wrong_feature_columns_raises(self, make_qnn):
        program = make_qnn()
        with pytest.raises(ValueError, match="binds 2 data parameters"):
            program.predict(np.zeros((3, 5)), params=np.zeros(4))

    def test_predict_wrong_params_length_raises(self, make_qnn, feature_batch_2x2):
        program = make_qnn()
        with pytest.raises(ValueError, match="weight parameters"):
            program.predict(feature_batch_2x2, params=np.zeros(3))


@pytest.mark.e2e
class TestPredict:
    def test_predict_returns_class_labels(
        self, make_qnn, feature_batch_2x2, default_test_simulator
    ):
        default_test_simulator.set_seed(1997)
        program = make_qnn(backend=default_test_simulator, n_layers=1, seed=1997)
        weights = np.array([0.5, 1.0, 1.5, 2.0])  # n_params_per_layer * n_layers

        labels = program.predict(feature_batch_2x2, params=weights)
        readout = program.predict(feature_batch_2x2, params=weights, return_scores=True)

        assert labels.shape == (feature_batch_2x2.shape[0],)
        assert set(np.unique(labels)).issubset({-1.0, 1.0})
        # predict() is exactly the sign of the readout.
        np.testing.assert_array_equal(labels, np.where(readout >= 0.0, 1.0, -1.0))
        # Single-qubit Z readout sits in [-1, 1] modulo shot noise.
        assert np.all(readout >= -1.2) and np.all(readout <= 1.2)

        # A single 1D feature row (shape (n_data_params,)) is promoted to one
        # sample via atleast_2d.
        single = program.predict(feature_batch_2x2[0], params=weights)
        assert single.shape == (1,)
        assert set(np.unique(single)).issubset({-1.0, 1.0})

    def test_predict_does_not_emit_progress(
        self, make_qnn, feature_batch_2x2, default_test_simulator
    ):
        # predict() runs a pipeline outside the optimizer loop; it must stay
        # silent. A spinner opened here is never closed and hijacks stdout in
        # notebooks, recursing on the next print().
        program = make_qnn(backend=default_test_simulator)
        emitted = []

        with program._bind_progress_emitter(emitted.append):
            program.predict(feature_batch_2x2, params=np.array([0.5, 1.0, 1.5, 2.0]))

        assert emitted == []


class TestObservableMeasuringContracts(ObservableMeasuringContractsBase):
    @pytest.fixture
    def make_program(self, make_qnn):
        return make_qnn


def test_data_binding_mixin_wrong_mro_order_rejected():
    """DataBindingMixin must precede the VQA base; the reverse order would let
    the base shadow the mixin's _assemble_pipeline and silently skip data
    binding, so it is rejected at class-definition time."""
    with pytest.raises(TypeError, match="DataBindingMixin before"):

        class _BadOrder(VariationalQuantumAlgorithm, DataBindingMixin):
            pass


def test_build_pipeline_env_honors_silent_override(make_qnn):
    """A caller-supplied silent override must win over the program emitter."""
    program = make_qnn()
    assert program._build_pipeline_env()._progress_emitter is program._progress_emitter
    assert program._build_pipeline_env(progress_emitter=None)._progress_emitter is None
