# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import functools

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import MetaCircuit
from divi.circuits._conversions import _format_bound_param
from divi.circuits.zne import ZNE
from divi.pipeline import CircuitPipeline, PipelineEnv
from divi.pipeline.stages import (
    CircuitSpecStage,
    DataBindingStage,
    MeasurementStage,
    ParameterBindingStage,
    PauliTwirlStage,
    QEMStage,
    resolve_loss_reduction,
    resolve_sample_loss,
)
from divi.pipeline.stages._data_binding_stage import DATA_AXIS
from tests._helpers import exact_match


def _make_circuit():
    data = ParameterVector("x", 2)
    weights = ParameterVector("w", 2)
    qc = QuantumCircuit(2)
    qc.ry(data[0], 0)
    qc.ry(data[1], 1)
    qc.rz(weights[0], 0)
    qc.rz(weights[1], 1)
    qc.cx(0, 1)
    return qc, tuple(data), tuple(weights)


def _make_meta(qc, data_params, weight_params, **kwargs):
    return MetaCircuit(
        circuit_bodies=(((), circuit_to_dag(qc)),),
        parameters=data_params + weight_params,
        observable=SparsePauliOp.from_list([("ZI", 1.0)]),
        **kwargs,
    )


def _mean(arr):
    return float(np.mean(arr))


def _env(*, feature_batch=None, labels=None):
    """Minimal stage-test env: no backend (expand/reduce don't execute) with the
    data axis passed through. End-to-end tests that actually run circuits build
    their own env with a real backend."""
    return PipelineEnv(backend=None, feature_batch=feature_batch, labels=labels)


def _reported_path(stage: DataBindingStage) -> str:
    return stage.introspect({}, _env(), token=None)["path"]


@pytest.fixture
def composed():
    return _make_circuit()


def test_expand_emits_one_body_per_sample(composed):
    qc, data_params, weight_params = composed
    feature_batch = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    stage = DataBindingStage(data_params=data_params, loss_reduction=_mean)
    meta = _make_meta(qc, data_params, weight_params)
    result = stage.expand({(): meta}, env=_env(feature_batch=feature_batch))
    out_meta = result.batch[()]
    # One body per sample, tagged with (DATA_AXIS, i).
    assert len(out_meta.circuit_bodies) == feature_batch.shape[0]
    for i, (body_tag, _) in enumerate(out_meta.circuit_bodies):
        assert body_tag[-1] == (DATA_AXIS, i)
    # parameters stripped of data params — weights are derived from the batch.
    assert out_meta.parameters == weight_params


def test_template_path_renders_partial_bodies_with_data_substituted(composed):
    """Template path: per-sample partial bodies are populated in
    ``qasm_bodies``, with data names absent and weight names
    preserved as placeholders."""
    qc, data_params, weight_params = composed
    feature_batch = np.array([[1.5, -0.5]])
    stage = DataBindingStage(data_params=data_params, loss_reduction=_mean)
    # Default path is template — validate() has not been called.
    assert stage._use_template_path is True
    meta = _make_meta(qc, data_params, weight_params)
    result = stage.expand({(): meta}, env=_env(feature_batch=feature_batch))
    out_meta = result.batch[()]
    assert len(out_meta.qasm_bodies) == feature_batch.shape[0]
    for _, partial_body in out_meta.qasm_bodies:
        for d in data_params:
            assert (
                d.name not in partial_body
            ), f"data placeholder {d.name!r} leaked into partial body"
        for w in weight_params:
            assert (
                w.name in partial_body
            ), f"weight placeholder {w.name!r} missing from partial body"


def test_template_tags_map_each_sample_to_its_row_index(composed):
    """expand binds feature_batch row ``i`` into the body tagged ``(DATA_AXIS, i)``.

    Supervised label pairing (in :meth:`reduce`) is positional, so it is only
    correct if the row→tag mapping here matches the row→label mapping. Unlike
    the mean/sum reductions, that correspondence is *not* permutation-invariant,
    so it gets an explicit guard.
    """
    qc, data_params, weight_params = composed
    feature_batch = np.array([[0.111, 0.0], [0.222, 0.0], [0.333, 0.0]])
    stage = DataBindingStage(data_params=data_params, loss_reduction=_mean)
    result = stage.expand(
        {(): _make_meta(qc, data_params, weight_params)},
        env=_env(feature_batch=feature_batch),
    )
    out = result.batch[()]
    for i, (tag, body) in enumerate(out.qasm_bodies):
        assert tag[-1] == (DATA_AXIS, i)
        expected = _format_bound_param(float(feature_batch[i][0]), out.precision)
        assert expected in body, f"row {i} data {expected!r} missing from its body"


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_expand_rejects_non_finite_feature_batch(composed, bad):
    """A NaN/Inf feature value is rejected at the data-ingestion boundary,
    before any render path bakes it into a circuit."""
    qc, data_params, weight_params = composed
    stage = DataBindingStage(data_params=data_params, loss_reduction=_mean)
    with pytest.raises(
        ValueError,
        match=exact_match(
            "Cannot bind non-finite gate parameters: env.feature_batch contains "
            "NaN or Inf. Check the feature batch / parameter values for missing "
            "data, divide-by-zero, or overflow in preprocessing."
        ),
    ):
        stage.expand(
            {(): _make_meta(qc, data_params, weight_params)},
            env=_env(feature_batch=np.array([[0.1, bad]])),
        )


def test_supervised_pairs_label_by_axis_index_not_arrival_order(composed):
    """reduce realigns predictions to the DATA_AXIS index before pairing labels.

    Results are fed out of axis order; the supervised loss must still pair
    ``label[i]`` with the row-``i`` prediction, not with whichever result
    arrived first. Catches a regression that scrambles sample↔label alignment —
    which the permutation-invariant mean/sum tests cannot.
    """
    qc, data_params, weight_params = composed
    stage = DataBindingStage(
        data_params=data_params,
        loss_reduction=lambda a: float(np.sum(a)),
        sample_loss=resolve_sample_loss("squared_error"),
    )
    # Inserted out of order: axis 2, then 0, then 1.
    results = {
        ((DATA_AXIS, 2), ("param_set", 0)): [2.0],
        ((DATA_AXIS, 0), ("param_set", 0)): [0.0],
        ((DATA_AXIS, 1), ("param_set", 0)): [1.0],
    }
    reduced = stage.reduce(results, env=_env(labels=[0.0, 10.0, 20.0]), token=None)
    # By axis index: preds=[0,1,2]; labels=[0,10,20];
    # errors=[0, 81, 324]; sum=405. Pairing by arrival order would give 465.
    assert reduced[(("param_set", 0),)] == [pytest.approx(405.0)]


def test_eager_path_substitutes_data_into_fresh_dags(composed):
    """Eager path: per-sample DAGs carry data values directly, leaving only weight
    parameters free, and each expand builds new DAGs."""
    qc, data_params, weight_params = composed
    stage = DataBindingStage(data_params=data_params, loss_reduction=_mean)
    stage.validate(
        before=(), after=(PauliTwirlStage(n_twirls=1), ParameterBindingStage())
    )
    env = _env(feature_batch=np.array([[1.5, -0.5]]))
    assert _reported_path(stage) == "eager"
    meta = _make_meta(qc, data_params, weight_params)

    result = stage.expand({(): meta}, env=env)
    second = stage.expand({(): meta}, env=env)

    _, dag = result.batch[()].circuit_bodies[0]
    remaining_params = {
        sym
        for node in dag.topological_op_nodes()
        for param in node.op.params
        for sym in getattr(param, "parameters", set())
    }
    assert remaining_params == set(weight_params)
    assert result.batch[()].parameters == tuple(weight_params)
    assert result.batch[()].qasm_bodies == ()
    assert second.batch[()].circuit_bodies[0][1] is not dag


def test_cache_key_extras_depend_on_feature_values_not_dtype(composed):
    _, data_params, _ = composed
    stage = DataBindingStage(data_params=data_params, loss_reduction=_mean)

    def key(features):
        return stage.cache_key_extras(_env(feature_batch=features))

    assert key(None) == ()
    assert key(np.array([[1, 2]])) == key(np.array([[1.0, 2.0]]))
    assert key(np.array([[1.0, 2.0]])) != key(np.array([[1.0, 3.0]]))


@pytest.mark.parametrize(
    "reduction,expected",
    [
        (lambda a: float(np.mean(a)), 2.0),
        (lambda a: float(np.sum(a)), 6.0),
        (lambda a: float(a.max()), 3.0),
    ],
)
def test_reduce_applies_reduction_to_each_observable(composed, reduction, expected):
    qc, data_params, weight_params = composed
    stage = DataBindingStage(data_params=data_params, loss_reduction=reduction)
    # Simulate three per-sample list[float] results (single observable).
    results = {
        ((DATA_AXIS, 0), ("param_set", 0)): [1.0],
        ((DATA_AXIS, 1), ("param_set", 0)): [2.0],
        ((DATA_AXIS, 2), ("param_set", 0)): [3.0],
    }
    reduced = stage.reduce(results, env=_env(), token=None)
    assert list(reduced.keys()) == [(("param_set", 0),)]
    assert reduced[(("param_set", 0),)] == [expected]


@pytest.mark.parametrize(
    "reduction,expected",
    [
        # mean([1, 2, 3] + 0.5) = 2.5
        (lambda a: float(np.mean(a)), 2.5),
        # sum([1, 2, 3] + 0.5) = 7.5   — would be 6.5 if constant added post-reduce
        (lambda a: float(np.sum(a)), 7.5),
        # rms = sqrt(mean((x+c)^2)) — a non-affine reduction with a clear signature
        (
            lambda a: float(np.sqrt(np.mean(a**2))),
            float(np.sqrt(np.mean(np.array([1.5, 2.5, 3.5]) ** 2))),
        ),
    ],
)
def test_reduce_folds_loss_constant_per_sample_before_reduction(
    composed, reduction, expected
):
    """Regression for identity-constant double-count under non-mean reductions.

    ``DataBindingStage`` owns ``loss_constant`` and must add it to each
    per-sample value *before* applying ``loss_reduction``. Adding it after
    would be correct only for mean (affine); ``sum`` would be off by
    ``(n_samples - 1) * const`` and arbitrary non-affine callables (rms,
    log-sum-exp, ...) would be silently wrong.
    """
    qc, data_params, weight_params = composed
    stage = DataBindingStage(
        data_params=data_params, loss_reduction=reduction, loss_constant=0.5
    )
    results = {
        ((DATA_AXIS, 0), ("param_set", 0)): [1.0],
        ((DATA_AXIS, 1), ("param_set", 0)): [2.0],
        ((DATA_AXIS, 2), ("param_set", 0)): [3.0],
    }
    reduced = stage.reduce(results, env=_env(), token=None)
    assert reduced[(("param_set", 0),)] == [pytest.approx(expected)]


def test_reduce_folds_loss_constant_for_scalar_inputs(composed):
    """Scalar-input branch of ``_reduce_one`` also folds the constant."""
    qc, data_params, weight_params = composed
    stage = DataBindingStage(
        data_params=data_params,
        loss_reduction=lambda a: float(np.sum(a)),
        loss_constant=0.5,
    )
    # Bare scalars (not lists) — exercise the non-list path in _reduce_one.
    results = {
        ((DATA_AXIS, 0), ("param_set", 0)): 1.0,
        ((DATA_AXIS, 1), ("param_set", 0)): 2.0,
        ((DATA_AXIS, 2), ("param_set", 0)): 3.0,
    }
    reduced = stage.reduce(results, env=_env(), token=None)
    # sum([1, 2, 3] + 0.5) = 7.5
    assert reduced[(("param_set", 0),)] == pytest.approx(7.5)


def test_reduce_handles_multiple_observables(composed):
    """``loss_reduction`` runs independently per observable in a multi-obs list."""
    qc, data_params, weight_params = composed
    stage = DataBindingStage(data_params=data_params, loss_reduction=_mean)
    # Three samples, each returning a per-observable list of length 2.
    results = {
        ((DATA_AXIS, 0), ("param_set", 0)): [1.0, 10.0],
        ((DATA_AXIS, 1), ("param_set", 0)): [2.0, 20.0],
        ((DATA_AXIS, 2), ("param_set", 0)): [3.0, 30.0],
    }
    reduced = stage.reduce(results, env=_env(), token=None)
    # mean of [1, 2, 3] = 2.0; mean of [10, 20, 30] = 20.0.
    assert reduced[(("param_set", 0),)] == [pytest.approx(2.0), pytest.approx(20.0)]


def test_reduce_applies_supervised_sample_loss_before_reduction(composed):
    """With env labels, predictions are mapped through the per-sample loss first.

    Squared error + mean reduction = mean-squared error against the labels,
    not the raw mean of the predictions.
    """
    qc, data_params, weight_params = composed
    stage = DataBindingStage(
        data_params=data_params,
        loss_reduction=_mean,
        sample_loss=resolve_sample_loss("squared_error"),
    )
    results = {
        ((DATA_AXIS, 0), ("param_set", 0)): [1.0],
        ((DATA_AXIS, 1), ("param_set", 0)): [2.0],
        ((DATA_AXIS, 2), ("param_set", 0)): [3.0],
    }
    reduced = stage.reduce(results, env=_env(labels=[1.0, 0.0, 0.0]), token=None)
    # errors = [(1-1)^2, (2-0)^2, (3-0)^2] = [0, 4, 9]; mean = 13/3.
    assert reduced[(("param_set", 0),)] == [pytest.approx(13.0 / 3.0)]


def test_supervised_loss_compares_after_loss_constant(composed):
    """``loss_constant`` shifts the prediction *before* the label comparison."""
    qc, data_params, weight_params = composed
    stage = DataBindingStage(
        data_params=data_params,
        loss_reduction=lambda a: float(np.sum(a)),
        loss_constant=0.5,
        sample_loss=resolve_sample_loss("squared_error"),
    )
    results = {
        ((DATA_AXIS, 0), ("param_set", 0)): [1.0],
        ((DATA_AXIS, 1), ("param_set", 0)): [2.0],
        ((DATA_AXIS, 2), ("param_set", 0)): [3.0],
    }
    reduced = stage.reduce(results, env=_env(labels=[1.5, 0.0, 0.0]), token=None)
    # preds = [1.5, 2.5, 3.5]; errors vs [1.5, 0, 0] = [0, 6.25, 12.25]; sum = 18.5.
    assert reduced[(("param_set", 0),)] == [pytest.approx(18.5)]


def test_supervised_scalar_input_branch(composed):
    """The scalar-result branch of ``_reduce_one`` also applies the label loss."""
    qc, data_params, weight_params = composed
    stage = DataBindingStage(
        data_params=data_params,
        loss_reduction=lambda a: float(np.sum(a)),
        sample_loss=resolve_sample_loss("squared_error"),
    )
    results = {
        ((DATA_AXIS, 0), ("param_set", 0)): 1.0,
        ((DATA_AXIS, 1), ("param_set", 0)): 2.0,
        ((DATA_AXIS, 2), ("param_set", 0)): 3.0,
    }
    reduced = stage.reduce(results, env=_env(labels=[1.0, 0.0, 0.0]), token=None)
    # sum of [0, 4, 9] = 13.
    assert reduced[(("param_set", 0),)] == pytest.approx(13.0)


def test_supervised_rejects_multiple_observables(composed):
    """A supervised loss needs one prediction per sample — multi-obs is ambiguous."""
    qc, data_params, weight_params = composed
    stage = DataBindingStage(
        data_params=data_params,
        loss_reduction=_mean,
        sample_loss=resolve_sample_loss("squared_error"),
    )
    results = {
        ((DATA_AXIS, 0), ("param_set", 0)): [1.0, 2.0],
        ((DATA_AXIS, 1), ("param_set", 0)): [3.0, 4.0],
    }
    with pytest.raises(
        ValueError,
        match=exact_match(
            "Supervised labels require a single cost observable so each sample "
            "has one prediction; got 2 observables."
        ),
    ):
        stage.reduce(results, env=_env(labels=[1.0, 0.0]), token=None)


def test_reduce_rejects_label_count_mismatch(composed):
    """reduce raises when the env's label count doesn't match the sample count."""
    qc, data_params, weight_params = composed
    stage = DataBindingStage(
        data_params=data_params,
        loss_reduction=_mean,
        sample_loss=resolve_sample_loss("squared_error"),
    )
    results = {
        ((DATA_AXIS, 0), ("param_set", 0)): [1.0],
        ((DATA_AXIS, 1), ("param_set", 0)): [2.0],
        ((DATA_AXIS, 2), ("param_set", 0)): [3.0],
    }
    with pytest.raises(
        ValueError, match=exact_match("got 3 per-sample predictions but 2 labels.")
    ):
        stage.reduce(results, env=_env(labels=[1.0, 0.0]), token=None)


def test_fit_bias_shifts_each_param_set_by_its_own_optimal_bias(composed):
    """``fit_bias`` shifts each param set's predictions by ``mean(labels - preds)``
    before the squared error, so the loss is the variance of the residuals and a
    constant offset between param sets is absorbed entirely."""
    qc, data_params, weight_params = composed
    stage = DataBindingStage(
        data_params=data_params,
        loss_reduction=_mean,
        sample_loss=resolve_sample_loss("squared_error"),
        fit_bias=True,
    )
    results = {
        ((DATA_AXIS, 0), ("param_set", 0)): [1.0],
        ((DATA_AXIS, 1), ("param_set", 0)): [2.0],
        ((DATA_AXIS, 2), ("param_set", 0)): [3.0],
        # param set 1 predicts param set 0 + 1 everywhere.
        ((DATA_AXIS, 0), ("param_set", 1)): [2.0],
        ((DATA_AXIS, 1), ("param_set", 1)): [3.0],
        ((DATA_AXIS, 2), ("param_set", 1)): [4.0],
    }
    reduced = stage.reduce(results, env=_env(labels=[1.0, 0.0, 0.0]), token=None)
    # residuals preds - labels = [0, 2, 3] (set 0) and [1, 3, 4] (set 1); both
    # have variance 14/9, which is the MSE at the fitted bias.
    assert reduced[(("param_set", 0),)] == [pytest.approx(14.0 / 9.0)]
    assert reduced[(("param_set", 1),)] == [pytest.approx(14.0 / 9.0)]


def test_reduce_rejects_fit_bias_without_labels(composed):
    qc, data_params, weight_params = composed
    stage = DataBindingStage(
        data_params=data_params, loss_reduction=_mean, fit_bias=True
    )
    results = {
        ((DATA_AXIS, 0), ("param_set", 0)): [1.0],
        ((DATA_AXIS, 1), ("param_set", 0)): [2.0],
    }
    with pytest.raises(
        ValueError, match=exact_match("fit_bias requires env.labels to be set.")
    ):
        stage.reduce(results, env=_env(), token=None)


def test_reduce_rejects_labels_without_sample_loss(composed):
    """env labels with no configured sample_loss is a misconfiguration."""
    qc, data_params, weight_params = composed
    stage = DataBindingStage(data_params=data_params, loss_reduction=_mean)
    results = {
        ((DATA_AXIS, 0), ("param_set", 0)): [1.0],
        ((DATA_AXIS, 1), ("param_set", 0)): [2.0],
    }
    with pytest.raises(
        ValueError,
        match=exact_match("env.labels were provided but the stage has no sample_loss."),
    ):
        stage.reduce(results, env=_env(labels=[1.0, 0.0]), token=None)


@pytest.mark.parametrize(
    "pred,label,expected",
    [(2.0, 0.5, 2.25), (1.0, 1.0, 0.0)],
)
def test_resolve_squared_error(pred, label, expected):
    fn = resolve_sample_loss("squared_error")
    assert fn(pred, label) == pytest.approx(expected)


def test_resolve_sample_loss_callable_passthrough():
    fn = resolve_sample_loss(lambda p, y: abs(p - y))
    assert fn(3.0, 1.0) == pytest.approx(2.0)


def test_resolve_sample_loss_rejects_unknown_literal():
    with pytest.raises(
        ValueError,
        match=exact_match(
            "loss_fn must be 'squared_error' or a callable; got 'huber'."
        ),
    ):
        resolve_sample_loss("huber")  # type: ignore[arg-type]


def test_expand_cross_multiplies_existing_body_count(composed):
    """``expand`` fans existing body variants over the sample axis."""
    qc, data_params, weight_params = composed
    feature_batch = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    stage = DataBindingStage(data_params=data_params, loss_reduction=_mean)
    # Build a MetaCircuit with two existing body variants — e.g. as if a prior
    # stage had already fanned out along its own axis.
    dag = circuit_to_dag(qc)
    # Each body_tag is a sequence of (axis_name, value) pairs, so a single
    # prior axis is one element wrapped in a 1-tuple.
    meta = MetaCircuit(
        circuit_bodies=(
            ((("prior", 0),), dag),
            ((("prior", 1),), dag),
        ),
        parameters=data_params + weight_params,
        observable=SparsePauliOp.from_list([("ZI", 1.0)]),
    )
    result = stage.expand({(): meta}, env=_env(feature_batch=feature_batch))
    bodies = result.batch[()].circuit_bodies
    assert len(bodies) == 2 * feature_batch.shape[0]
    sample_tags_per_prior = {0: set(), 1: set()}
    for body_tag, _ in bodies:
        prior_axis = next(t for t in body_tag if t[0] == "prior")
        data_axis = next(t for t in body_tag if t[0] == DATA_AXIS)
        sample_tags_per_prior[prior_axis[1]].add(data_axis[1])
    assert sample_tags_per_prior[0] == {0, 1, 2}
    assert sample_tags_per_prior[1] == {0, 1, 2}


def test_introspect_reports_sample_count_and_path(composed):
    qc, data_params, weight_params = composed
    feature_batch = np.array([[0.1, 0.2], [0.3, 0.4]])
    stage = DataBindingStage(data_params=data_params, loss_reduction=_mean)
    meta = _make_meta(qc, data_params, weight_params)
    env = _env(feature_batch=feature_batch)
    metadata = stage.introspect({(): meta}, env=env, token=None)
    assert metadata == {
        "n_samples": 2,
        "n_data_params": 2,
        "supervised": False,
        "loss_reduction": "mean",
        "fit_bias": False,
        "path": "template",
    }
    stage._use_template_path = False
    assert stage.introspect({(): meta}, env=env, token=None)["path"] == "eager"


def test_introspect_reports_supervision(composed):
    """Supervision decides which optimizers are legal at all, so two otherwise
    identical reports must not be indistinguishable."""
    qc, data_params, weight_params = composed
    env = _env(feature_batch=np.array([[0.1, 0.2]]))
    meta = _make_meta(qc, data_params, weight_params)
    supervised = DataBindingStage(
        data_params=data_params, loss_reduction=_mean, sample_loss=lambda p, l: p - l
    )
    assert supervised.introspect({(): meta}, env=env, token=None)["supervised"] is True


def test_introspect_names_the_loss_and_reduction(composed):
    """A loss ablation changes no circuit, so without these rows two arms of one
    render identically and a grouped ensemble report presents them as one config."""
    qc, data_params, weight_params = composed
    env = _env(feature_batch=np.array([[0.1, 0.2]]))
    meta = _make_meta(qc, data_params, weight_params)

    def hinge(prediction, label):
        return max(0.0, 1.0 - prediction * label)

    stage = DataBindingStage(
        data_params=data_params,
        loss_reduction=resolve_loss_reduction("sum"),
        sample_loss=resolve_sample_loss(hinge),
    )
    metadata = stage.introspect({(): meta}, env=env, token=None)
    assert metadata["loss_fn"] == "hinge"
    assert metadata["loss_reduction"] == "sum_reduction"

    unsupervised = DataBindingStage(
        data_params=data_params, loss_reduction=resolve_loss_reduction("mean")
    )
    # No loss row at all rather than a null one: there is no per-sample loss to name.
    assert "loss_fn" not in unsupervised.introspect({(): meta}, env=env, token=None)


def test_expand_rejects_mismatched_feature_columns(composed):
    """A feature batch whose columns don't match ``data_params`` is rejected at
    expand time (the data values now arrive via the env)."""
    qc, data_params, weight_params = composed
    stage = DataBindingStage(data_params=data_params, loss_reduction=_mean)
    env = _env(feature_batch=np.array([[0.1, 0.2, 0.3]]))
    with pytest.raises(
        ValueError,
        match=exact_match(
            "feature_batch has 3 columns but 2 data parameters were declared."
        ),
    ):
        stage.expand({(): _make_meta(qc, data_params, weight_params)}, env=env)


def test_expand_rejects_1d_feature_batch(composed):
    qc, data_params, weight_params = composed
    stage = DataBindingStage(data_params=data_params, loss_reduction=_mean)
    env = _env(feature_batch=np.array([0.1, 0.2]))
    with pytest.raises(
        ValueError, match=exact_match("feature_batch must be 2D; got shape (2,).")
    ):
        stage.expand({(): _make_meta(qc, data_params, weight_params)}, env=env)


def test_expand_requires_feature_batch_in_env(composed):
    qc, data_params, weight_params = composed
    stage = DataBindingStage(data_params=data_params, loss_reduction=_mean)
    with pytest.raises(
        ValueError,
        match=exact_match("DataBindingStage requires env.feature_batch to be set."),
    ):
        stage.expand(
            {(): _make_meta(qc, data_params, weight_params)},
            env=_env(),
        )


@pytest.mark.parametrize(
    "reduction,expected",
    [("mean", 2.5), ("sum", 10.0)],
)
def test_resolve_named_reductions(reduction, expected):
    fn = resolve_loss_reduction(reduction)
    assert fn(np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(expected)


def test_resolve_callable_reduction_passthrough():
    fn = resolve_loss_reduction(lambda a: float(a.max()))
    assert fn(np.array([1.0, 2.0, 3.0])) == 3.0


def test_resolve_loss_reduction_rejects_unknown_literal():
    with pytest.raises(
        ValueError,
        match=exact_match(
            "loss_reduction must be 'mean', 'sum', or a callable; got 'median'."
        ),
    ):
        resolve_loss_reduction("median")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "feature_batch",
    [
        pytest.param(np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]), id="finite"),
        pytest.param(np.array([[np.nan, 0.1], [0.2, np.inf]]), id="non-finite"),
    ],
)
def test_dry_expand_shares_one_dag_across_sample_variants(composed, feature_batch):
    """Dry-run skips per-sample data substitution; all N body variants
    point at the same incoming parametric DAG, giving O(1) DAG memory
    irrespective of ``feature_batch`` size. Values are never rendered, so
    non-finite features are accepted."""
    qc, data_params, weight_params = composed
    stage = DataBindingStage(data_params=data_params, loss_reduction=_mean)
    meta = _make_meta(qc, data_params, weight_params)
    result = stage.dry_expand({(): meta}, env=_env(feature_batch=feature_batch))
    bodies = result.batch[()].circuit_bodies
    assert len(bodies) == feature_batch.shape[0]
    # All variants share the same DAG instance — that's the "lazy" win.
    shared = bodies[0][1]
    assert all(dag is shared for _, dag in bodies)


def test_template_path_passes_through_incoming_dag(composed):
    """Template path hands the *incoming* parametric DAG straight to the output
    (no copy) — O(1) DAG memory. The same spec DAG flows each cost evaluation,
    so successive expands of one MetaCircuit share the same DAG reference."""
    qc, data_params, weight_params = composed
    feature_batch = np.array([[0.1, 0.2], [0.3, 0.4]])
    stage = DataBindingStage(data_params=data_params, loss_reduction=_mean)
    meta = _make_meta(qc, data_params, weight_params)
    incoming_dag = meta.circuit_bodies[0][1]
    env = _env(feature_batch=feature_batch)
    first = stage.expand({(): meta}, env=env)
    second = stage.expand({(): meta}, env=env)
    first_dag = first.batch[()].circuit_bodies[0][1]
    second_dag = second.batch[()].circuit_bodies[0][1]
    assert first_dag is incoming_dag
    assert first_dag is second_dag


def test_validate_picks_template_path_when_only_pb_downstream(composed):
    """With only ParameterBindingStage downstream, the template path is
    chosen — its fast-path lookup consumes ``qasm_bodies``."""
    qc, data_params, weight_params = composed
    stage = DataBindingStage(data_params=data_params, loss_reduction=_mean)
    stage.validate(before=(), after=(ParameterBindingStage(),))
    assert _reported_path(stage) == "template"


def test_validate_treats_no_mitigation_qem_as_transparent(composed):
    """``QEMStage(_NoMitigation())`` is structurally a pass-through, so
    DataBindingStage stays on the template path."""
    qc, data_params, weight_params = composed
    stage = DataBindingStage(data_params=data_params, loss_reduction=_mean)
    stage.validate(before=(), after=(QEMStage(), ParameterBindingStage()))
    assert _reported_path(stage) == "template"


def test_validate_falls_back_to_eager_when_active_qem_downstream(composed):
    """An active QEM protocol (ZNE) mutates the DAG, so we must materialise
    per-sample DAGs eagerly."""
    qc, data_params, weight_params = composed
    stage = DataBindingStage(data_params=data_params, loss_reduction=_mean)
    stage.validate(
        before=(),
        after=(
            QEMStage(protocol=ZNE(scale_factors=[1.0, 3.0])),
            ParameterBindingStage(),
        ),
    )
    assert _reported_path(stage) == "eager"


def test_validate_falls_back_to_eager_when_pauli_twirl_downstream(composed):
    qc, data_params, weight_params = composed
    stage = DataBindingStage(data_params=data_params, loss_reduction=_mean)
    stage.validate(
        before=(),
        after=(PauliTwirlStage(n_twirls=1), ParameterBindingStage()),
    )
    assert _reported_path(stage) == "eager"


@pytest.mark.e2e
def test_template_and_eager_paths_yield_equivalent_loss(
    composed, default_test_simulator
):
    """A QNN-style program produces the same scalar loss whether DataBindingStage
    takes its template fast path or its eager fallback."""
    qc, data_params, weight_params = composed
    feature_batch = np.array([[0.1, 0.2], [0.3, 0.4]])

    def _build_program(use_template: bool):
        stage = DataBindingStage(data_params=data_params, loss_reduction=_mean)
        backend = default_test_simulator
        backend.set_seed(1997)  # reset per build so both paths share one RNG draw
        pipeline = CircuitPipeline(
            stages=[
                CircuitSpecStage(),
                stage,
                MeasurementStage(),
                ParameterBindingStage(),
            ]
        )
        # Pipeline construction re-runs ``validate``, so the path is forced after it.
        stage._use_template_path = use_template
        meta = _make_meta(qc, data_params, weight_params)
        env = PipelineEnv(
            backend=backend,
            param_sets=np.array([[0.7, 1.3]]),
            feature_batch=feature_batch,
        )
        assert _reported_path(stage) == ("template" if use_template else "eager")
        return pipeline.run(initial_spec={"cost": meta}, env=env)

    template = _build_program(use_template=True)
    eager = _build_program(use_template=False)
    assert set(template.keys()) == set(eager.keys())
    for key in template:
        np.testing.assert_allclose(
            np.asarray(template[key], dtype=np.float64),
            np.asarray(eager[key], dtype=np.float64),
            rtol=0,
            atol=1e-12,
        )


def test_resolve_loss_reduction_wraps_numpy_callables_into_float():
    """``np.mean`` returns a 0-d ``ndarray``; the resolver must force-cast it
    to a plain Python ``float`` so downstream history serialization works."""
    fn = resolve_loss_reduction(np.mean)
    result = fn(np.array([1.0, 2.0, 3.0]))
    assert type(result) is float
    assert result == pytest.approx(2.0)


def _reported_names(stage: DataBindingStage) -> dict:
    metadata = stage.introspect({}, _env(), token=None)
    return {
        key: metadata[key] for key in ("loss_reduction", "loss_fn") if key in metadata
    }


@pytest.mark.parametrize(
    "loss_reduction, sample_loss, expected",
    [
        pytest.param(
            resolve_loss_reduction(np.median),
            None,
            {"loss_reduction": "median"},
            id="named-reduction",
        ),
        pytest.param(
            resolve_loss_reduction(functools.partial(np.quantile, q=0.5)),
            resolve_sample_loss(functools.partial(np.subtract)),
            {"loss_reduction": "custom", "loss_fn": "custom"},
            id="nameless-callables",
        ),
        pytest.param(
            functools.partial(np.quantile, q=0.5),
            None,
            {"loss_reduction": "partial"},
            id="unresolved-callable",
        ),
    ],
)
def test_introspect_names_user_callables(
    composed, loss_reduction, sample_loss, expected
):
    _, data_params, _ = composed
    stage = DataBindingStage(
        data_params=data_params,
        loss_reduction=loss_reduction,
        sample_loss=sample_loss,
    )
    assert _reported_names(stage) == expected


def test_template_path_renders_circuit_constants_at_the_circuit_precision(composed):
    qc, data_params, weight_params = composed
    qc.rz(0.123456789, 0)
    meta = _make_meta(qc, data_params, weight_params, precision=4)
    stage = DataBindingStage(data_params=data_params, loss_reduction=_mean)
    result = stage.expand(
        {(): meta}, env=_env(feature_batch=np.array([[0.1, 0.2], [0.3, 0.4]]))
    )
    bodies = [body for _, body in result.batch[()].qasm_bodies]
    assert len(bodies) == 2
    assert all("rz(0.1235) q[0];" in body for body in bodies)


def test_dry_expand_drops_data_parameters_like_expand(composed):
    qc, data_params, weight_params = composed
    stage = DataBindingStage(data_params=data_params, loss_reduction=_mean)
    batch = {(): _make_meta(qc, data_params, weight_params)}
    env = _env(feature_batch=np.array([[0.1, 0.2]]))
    dry = stage.dry_expand(batch, env=env).batch[()]
    real = stage.expand(batch, env=env).batch[()]
    assert dry.parameters == real.parameters == weight_params
