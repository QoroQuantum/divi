# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import math
import warnings

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
from divi.qprog.checkpointing import CheckpointConfig
from divi.qprog.mixins import DataBindingMixin
from divi.qprog.mixins._data_binding import _LOSS_FN_IGNORED_MSG
from divi.qprog.optimizers import (
    MonteCarloOptimizer,
    QNSPSAOptimizer,
    ScipyMethod,
    ScipyOptimizer,
)
from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm
from tests._helpers import exact_match
from tests.qprog._program_contracts import (
    ObservableMeasuringContractsBase,
    verify_cost_circuit,
)


class _DoubleFrequencyAnsatz(GenericLayerAnsatz):
    """Declares frequency ``{1, 2}`` for each of its per-layer parameters."""

    def parameter_frequencies(self, n_qubits, **kwargs):
        return [(1.0, 2)] * self.n_params_per_layer(n_qubits)


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
        assert program.cost_hamiltonian.coeffs.real.tolist() == [1.0]
        assert program.loss_constant == 0.0

    def test_defaults(self, make_qnn, simple_feature_map, simple_ansatz):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            program = make_qnn()

        assert program.max_iterations == 10
        assert program.feature_map is simple_feature_map
        assert program.ansatz is simple_ansatz

    def test_single_qubit_circuit_is_accepted(self, make_qnn):
        program = make_qnn(
            n_qubits=1,
            ansatz=GenericLayerAnsatz(gate_sequence=[RYGate]),
            observable=None,
            feature_batch=np.array([[0.1], [0.2]]),
        )

        assert program.n_qubits == 1

    @pytest.mark.parametrize(
        "n_layers, expected",
        [(1, [(1.0, 2)] * 2), (2, [(1.0, 2)] * 4)],
    )
    def test_parameter_frequencies_repeat_the_ansatz_per_layer(
        self, make_qnn, n_layers, expected
    ):
        program = make_qnn(ansatz=_DoubleFrequencyAnsatz([RYGate]), n_layers=n_layers)

        assert program._parameter_frequencies() == expected

    def test_default_parameter_frequencies_are_undeclared(self, make_qnn):
        assert make_qnn(n_layers=2)._parameter_frequencies() is None

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
        """The ignored-``loss_fn`` warning points at the caller's constructor call."""
        with pytest.warns(
            UserWarning, match=exact_match(_LOSS_FN_IGNORED_MSG)
        ) as record:
            make_qnn(loss_fn=lambda pred, label: (pred - label) ** 2)
        ignored = [w for w in record if str(w.message) == _LOSS_FN_IGNORED_MSG]
        assert ignored and ignored[0].filename == __file__

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"labels": [1.0, -1.0, 1.0, -1.0], "loss_fn": lambda p, l: abs(p - l)},
            {"observable": None, "labels": [1.0, -1.0, 1.0, -1.0]},
            {"observable": None, "labels": [1.0, -1.0, 1.0, -1.0], "fit_bias": True},
            {
                "feature_batch": [[0.1, 0.2], [0.3, 0.4]],
                "labels": [0.5, -0.5],
                "fit_bias": True,
            },
        ],
        ids=[
            "custom-loss-with-labels",
            "labels-at-the-readout-edges",
            "fitted-labels-spanning-the-readout",
            "fit-bias-on-two-samples",
        ],
    )
    def test_construction_stays_silent(self, make_qnn, kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            make_qnn(**kwargs)

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
        assert program.sample_loss is None

    def test_labels_stored_when_supervised(self, make_qnn):
        program = make_qnn(labels=[0.0, 1.0, 0.0, 1.0])
        np.testing.assert_array_equal(program.labels, [0.0, 1.0, 0.0, 1.0])
        assert callable(program.sample_loss)

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

    def test_run_rejects_a_batch_that_no_longer_matches_the_labels(
        self, make_qnn, mocker
    ):
        program = make_qnn(labels=[0.0, 1.0, 0.0, 1.0])
        submit = mocker.spy(program.backend, "submit_circuits")
        program.feature_batch = np.asarray(program.feature_batch)[:2]

        with pytest.raises(ValueError, match="labels has 4 entries"):
            program.run()
        submit.assert_not_called()

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
        assert supervised.objective_fingerprint != unsupervised.objective_fingerprint

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

    @pytest.mark.parametrize(
        "labels",
        [[0.0, 2.0, 0.0, 2.0], [2.0, -2.0, 2.0, -2.0]],
        ids=["width-2", "width-4"],
    )
    def test_out_of_range_labels_warn_for_default_observable(self, make_qnn, labels):
        # Default parity observable reads out in [-1, 1]; labels outside it can't
        # be matched, so squared error floors above zero — warn the user.
        with pytest.warns(
            UserWarning,
            match=exact_match(
                "labels fall outside [-1, 1] but the default parity observable "
                "reads out in [-1, 1]; the supervised loss cannot reach zero. "
                "Encode labels in [-1, 1] (e.g. -1/+1) or pass an observable "
                "whose range matches your labels."
            ),
        ):
            make_qnn(observable=None, labels=labels)

    def test_fit_bias_requires_labels(self, make_qnn):
        with pytest.raises(ValueError, match="fit_bias requires labels"):
            make_qnn(fit_bias=True)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"loss_fn": lambda pred, label: abs(pred - label)},
            {"loss_reduction": lambda arr: float(np.max(arr))},
        ],
        ids=["custom-loss", "custom-reduction"],
    )
    def test_fit_bias_requires_squared_error_mean_or_sum(self, make_qnn, overrides):
        # The fitted bias mean(labels - prediction) is only the exact optimum for
        # squared error under a mean or sum reduction.
        with pytest.raises(ValueError, match="fit_bias requires loss_fn"):
            make_qnn(labels=[0.0, 1.0, 0.0, 1.0], fit_bias=True, **overrides)

    def test_fit_bias_silences_out_of_range_label_warning(self, make_qnn, recwarn):
        # A fitted bias shifts the readout range, so {0, 2} labels are reachable.
        make_qnn(observable=None, labels=[0.0, 2.0, 0.0, 2.0], fit_bias=True)
        assert not [w for w in recwarn if "reads out in" in str(w.message)]

    def test_fit_bias_warns_for_labels_wider_than_the_readout(self, make_qnn):
        # A bias shifts the [-1, 1] readout but cannot widen it.
        with pytest.warns(
            UserWarning,
            match=exact_match(
                "labels span more than the readout range of the default parity "
                "observable ([-1, 1], width 2); a fitted bias shifts that range "
                "but cannot widen it, so the supervised loss cannot reach zero. "
                "Rescale your labels or pass an observable whose range matches "
                "them."
            ),
        ):
            make_qnn(observable=None, labels=[0.0, 3.0, 0.0, 3.0], fit_bias=True)

    def test_fit_bias_requires_two_samples(self, make_qnn):
        # With one sample the bias absorbs the whole error, so the loss is 0.
        with pytest.raises(
            ValueError,
            match=exact_match(
                "fit_bias requires at least 2 samples; with one, the bias "
                "absorbs the whole error and the loss is always 0."
            ),
        ):
            make_qnn(feature_batch=[[0.1, 0.2]], labels=[1.0], fit_bias=True)

    def test_fitted_bias_without_fit_bias_raises(self, make_qnn):
        program = make_qnn(labels=[0.0, 1.0, 0.0, 1.0])
        with pytest.raises(RuntimeError, match="requires constructing with fit_bias"):
            program.fitted_bias

    def test_fitted_bias_before_run_raises(self, make_qnn):
        program = make_qnn(labels=[0.0, 1.0, 0.0, 1.0], fit_bias=True)
        with pytest.raises(RuntimeError, match=r"call run\(\) first"):
            program.fitted_bias
        with pytest.raises(RuntimeError, match=r"call run\(\) first"):
            program.predict(program.feature_batch, params=np.zeros(4))


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
    per-sample costs.

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

    np.testing.assert_allclose(
        batched_loss, float(np.mean(per_sample_losses)), atol=1e-9
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
    against the labels.

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
    np.testing.assert_allclose(supervised_loss, expected, atol=1e-9)


@pytest.mark.e2e
def test_reassigned_feature_batch_reaches_the_cost(make_qnn, default_test_simulator):
    """A feature batch assigned after the cost pipeline ran is re-expanded, not
    served from the pipeline's cache."""
    new_features = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.4], [0.7, 0.3]])
    labels = [1.0, 0.0, 1.0, 0.0]
    weights = np.array([[0.5, 1.0, 1.5, 2.0]])

    def cost(program):
        default_test_simulator.set_seed(1997)
        return program._evaluate_cost_param_sets(weights)[0]

    fresh = make_qnn(
        backend=default_test_simulator,
        n_layers=1,
        seed=1997,
        feature_batch=new_features,
        labels=labels,
    )
    program = make_qnn(
        backend=default_test_simulator, n_layers=1, seed=1997, labels=labels
    )
    cost(program)
    program.feature_batch = new_features

    np.testing.assert_allclose(cost(program), cost(fresh), rtol=1e-9)


@pytest.mark.e2e
class TestFitBias:
    WEIGHTS = np.array([0.5, 1.0, 1.5, 2.0])
    LABELS = np.array([1.0, 0.0, 1.0, 0.0])

    def _unbiased_scores(self, make_qnn, simulator, params):
        simulator.set_seed(1997)
        plain = make_qnn(backend=simulator, n_layers=1, seed=1997)
        return plain.predict(plain.feature_batch, params=params, return_scores=True)

    def _make_biased_qnn(self, make_qnn, simulator, **overrides):
        simulator.set_seed(1997)
        return make_qnn(
            **{
                "backend": simulator,
                "n_layers": 1,
                "optimizer": MonteCarloOptimizer(population_size=4, n_best_sets=2),
                "max_iterations": 2,
                "seed": 1997,
                "labels": self.LABELS,
                "fit_bias": True,
                **overrides,
            }
        )

    def _checkpointed_biased_qnn(self, make_qnn, simulator, checkpoint_dir):
        program = self._make_biased_qnn(make_qnn, simulator)
        program.run(
            perform_final_computation=False,
            checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir),
        )
        return program

    def _assert_bias_fits_the_readout(self, program, make_qnn, simulator):
        readout = self._unbiased_scores(make_qnn, simulator, program.best_params)
        np.testing.assert_allclose(
            program.fitted_bias, np.mean(self.LABELS - readout), atol=1e-9
        )

    def test_cost_is_mse_at_the_fitted_bias(self, make_qnn, default_test_simulator):
        """The loss equals the MSE after shifting by ``mean(labels - readout)``,
        i.e. the variance of the residuals."""
        readout = self._unbiased_scores(make_qnn, default_test_simulator, self.WEIGHTS)
        labels = np.array([0.3, -0.2, 0.5, -0.6])

        program = self._make_biased_qnn(make_qnn, default_test_simulator, labels=labels)
        loss = program._evaluate_cost_param_sets(self.WEIGHTS[None, :])[0]

        np.testing.assert_allclose(loss, float(np.var(readout - labels)), rtol=1e-9)

    def test_run_stores_the_bias_at_best_params(self, make_qnn, default_test_simulator):
        """The bias is fitted in closed form, not appended to the optimised weights."""
        program = self._make_biased_qnn(make_qnn, default_test_simulator)
        program.run(perform_final_computation=False)

        n_weights = program.n_layers * program.ansatz.n_params_per_layer(
            program.n_qubits
        )
        assert program.best_params.shape == (n_weights,)
        self._assert_bias_fits_the_readout(program, make_qnn, default_test_simulator)

    def test_labels_assigned_after_run_do_not_move_the_bias(
        self, make_qnn, default_test_simulator
    ):
        program = self._make_biased_qnn(make_qnn, default_test_simulator)
        program.run(perform_final_computation=False)
        bias = program.fitted_bias
        scores = program.predict(program.feature_batch, return_scores=True)

        program.labels = self.LABELS + 5.0

        assert program.fitted_bias == bias
        np.testing.assert_allclose(
            program.predict(program.feature_batch, return_scores=True),
            scores,
            atol=1e-9,
        )

    def test_fitted_bias_survives_a_checkpoint(
        self, make_qnn, default_test_simulator, tmp_path
    ):
        program = self._checkpointed_biased_qnn(
            make_qnn, default_test_simulator, tmp_path
        )

        restored = self._make_biased_qnn(make_qnn, default_test_simulator)
        restored._restore_state(tmp_path)

        assert restored.fitted_bias == program.fitted_bias

    def test_load_rejects_a_mismatched_fit_bias(
        self, make_qnn, default_test_simulator, tmp_path
    ):
        self._checkpointed_biased_qnn(make_qnn, default_test_simulator, tmp_path)
        unbiased = self._make_biased_qnn(
            make_qnn, default_test_simulator, fit_bias=False
        )

        with pytest.raises(ValueError, match="trained with fit_bias=True"):
            unbiased._restore_state(tmp_path)

    def test_bias_missing_after_restore_is_fitted_on_read(
        self, make_qnn, default_test_simulator, tmp_path
    ):
        """An iteration checkpoint holds no bias; reading it fits the bias at the
        restored best_params."""
        program = self._checkpointed_biased_qnn(
            make_qnn, default_test_simulator, tmp_path
        )
        restored = self._make_biased_qnn(make_qnn, default_test_simulator)
        restored._restore_state(tmp_path, subdirectory="checkpoint_002")

        np.testing.assert_allclose(restored.fitted_bias, program.fitted_bias, atol=1e-9)

    def test_sum_reduction_with_final_computation_fits_the_bias(
        self, make_qnn, default_test_simulator
    ):
        program = self._make_biased_qnn(
            make_qnn, default_test_simulator, loss_reduction="sum"
        )
        program.run()

        self._assert_bias_fits_the_readout(program, make_qnn, default_test_simulator)


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
