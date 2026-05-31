# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage that binds a classical data batch into a QNN circuit.

The composed QNN circuit is parametric in two disjoint groups of Qiskit
``Parameter`` objects: data parameters (fed from a classical feature batch
at every cost evaluation) and weight parameters (updated by the optimizer).
:class:`DataBindingStage` owns the ``data_sample`` axis: during *expand*
it replaces each MetaCircuit's bodies with one variant per sample — data
parameters baked into the DAG as floats, weight parameters left in place —
and during *reduce* it averages the per-sample results back into one
scalar per remaining label.

Idiomatically equivalent to :class:`PauliTwirlStage`: a stage that fans
out along its own axis on the forward pass and contracts it on the
backward pass. Downstream stages
(:class:`~divi.pipeline.stages.MeasurementStage`,
:class:`~divi.pipeline.stages.ParameterBindingStage`) see a clean
weight-only parametric DAG.
"""

import functools
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from qiskit.circuit import Parameter
from qiskit.dagcircuit import DAGCircuit

from divi.circuits import MetaCircuit, QASMTemplate, render_template
from divi.circuits._conversions import (
    _assert_finite,
    _format_bound_param,
    bind_parameters_in_dag,
)
from divi.circuits.qem import _NoMitigation
from divi.pipeline.abc import (
    BundleStage,
    ChildResults,
    ExpansionResult,
    MetaCircuitBatch,
    PipelineEnv,
    Stage,
    StageToken,
)
from divi.pipeline.stages import ParameterBindingStage, QEMStage
from divi.pipeline.stages._qasm_cache import _qasm_body_cached, _template_cached
from divi.pipeline.transformations import (
    group_by_base_key,
    reduce_postprocess_ordered,
)

DATA_AXIS = "data_sample"

#: Aggregation across the data-sample axis. ``"mean"``/``"sum"`` or a callable
#: ``(n_samples,) → float``.
LossReductionFn = Literal["mean", "sum"] | Callable[[npt.NDArray[np.float64]], float]


#: Per-sample supervised loss comparing a prediction to its label. The string
#: ``"squared_error"`` selects ``(prediction - label) ** 2``; with the default
#: ``"mean"`` reduction this yields mean-squared error.
SampleLossFn = Literal["squared_error"] | Callable[[float, float], float]


def _mean_reduction(arr: npt.NDArray[np.float64]) -> float:
    return float(np.mean(arr))


def _sum_reduction(arr: npt.NDArray[np.float64]) -> float:
    return float(np.sum(arr))


def _squared_error(prediction: float, label: float) -> float:
    return (prediction - label) ** 2


def resolve_sample_loss(
    loss: SampleLossFn,
) -> Callable[[float, float], float]:
    """Resolve a ``SampleLossFn`` literal/callable to a concrete callable.

    User-supplied callables are wrapped in ``float(...)`` so the per-sample loss
    is a plain Python float regardless of the numpy types it returns. A custom
    callable must return a finite value — a NaN/Inf result is not guarded and
    propagates into the reduction and on to the optimizer.
    """
    if loss == "squared_error":
        return _squared_error
    if callable(loss):

        def _user_loss(prediction: float, label: float) -> float:
            return float(loss(prediction, label))

        return _user_loss
    raise ValueError(f"loss_fn must be 'squared_error' or a callable; got {loss!r}.")


def resolve_loss_reduction(
    reduction: LossReductionFn,
) -> Callable[[npt.NDArray[np.float64]], float]:
    """Resolve a ``LossReductionFn`` literal/callable to a concrete callable.

    User-supplied callables are wrapped in ``float(...)`` so naked numpy
    reductions (e.g. ``loss_reduction=np.mean``) — which return a 0-d
    ``ndarray`` — produce a plain Python float, matching the contract that
    downstream stages and ``losses_history`` expect.
    """
    if reduction == "mean":
        return _mean_reduction
    if reduction == "sum":
        return _sum_reduction
    if callable(reduction):

        def _user_reduction(arr: npt.NDArray[np.float64]) -> float:
            return float(reduction(arr))

        return _user_reduction
    raise ValueError(
        f"loss_reduction must be 'mean', 'sum', or a callable; got {reduction!r}."
    )


class DataBindingStage(BundleStage):
    """Fan a parametric circuit out over a classical feature batch.

    Each row of ``env.feature_batch`` becomes a body variant whose
    ``data_params`` are pre-bound to that sample's values. Downstream
    stages see a weight-only parametric circuit; the per-sample results
    are aggregated on the backward pass via ``loss_reduction``.

    The stage owns only the **data axis** — which parameters are bound from
    data, how per-sample results are reduced, and the optional supervised
    loss. The circuit and the weight parameters come from the incoming
    ``MetaCircuit`` batch (weights are ``parameters`` minus ``data_params``,
    order preserved), and the ``feature_batch`` / ``labels`` come from the
    :class:`~divi.pipeline.abc.PipelineEnv` at run time — so one stage serves
    any batch (train, mini-batch, or inference).

    The stage owns two implementations of :meth:`expand`, selected at
    pipeline-construction time via :meth:`validate`:

    * **Template fast path** (default when no DAG-walking stage sits
      downstream): builds the parametric QASM body from the incoming DAG
      once (cached), renders per-sample partial bodies as strings (data
      substituted, weight placeholders preserved), and parks them on
      ``MetaCircuit.qasm_bodies`` keyed by ``body_tag``. All
      variants share the incoming parametric DAG ref in ``circuit_bodies``
      (~O(1) DAG memory regardless of batch size).
      :class:`~divi.pipeline.stages.ParameterBindingStage`'s fast path
      consults the pre-rendered bodies and skips its DAG → QASM step.

    * **Eager fallback path** (used when QEM, Pauli twirling, or any
      other stage with ``consumes_dag_bodies=True`` sits between this
      stage and ParameterBinding): substitutes each sample's data into
      its own bound DAG. Memory scales linearly with batch size, but
      per-sample DAGs are required because downstream stages walk them.

    Args:
        data_params: The Qiskit ``Parameter`` objects bound from data. Their
            order must match ``env.feature_batch``'s column order. Everything
            else in the incoming ``MetaCircuit.parameters`` is treated as a
            weight and handed downstream (order preserved).
        loss_reduction: Callable ``(n_samples,) → float`` applied during
            :meth:`reduce` to collapse one base-key's per-sample expectation
            values into a single scalar. Invoked once per observable when
            the upstream measurement stage returns a per-observable list.
        loss_constant: Constant added to each per-sample expectation value
            *before* ``loss_reduction``, so reductions apply to the full
            (unshifted) loss.
        sample_loss: Optional per-sample loss ``(prediction, label) → float``
            (e.g. squared error). When set and ``env.labels`` is provided, each
            per-sample prediction is mapped through it against its label before
            ``loss_reduction`` — turning the unsupervised aggregate into a
            supervised training loss.
    """

    @property
    def axis_name(self) -> str:
        return DATA_AXIS

    @property
    def handles_measurement(self) -> bool:
        return False

    @property
    def consumes_dag_bodies(self) -> bool:
        # Each variant carries a DAG whose data parameters have been
        # substituted away — downstream stages still walk the DAG.
        return True

    def __init__(
        self,
        data_params: tuple[Parameter, ...],
        loss_reduction: Callable[[npt.NDArray[np.float64]], float],
        loss_constant: float = 0.0,
        sample_loss: Callable[[float, float], float] | None = None,
    ) -> None:
        super().__init__(name=DATA_AXIS)
        self.data_params = tuple(data_params)
        self._data_param_names = tuple(p.name for p in self.data_params)
        self.loss_reduction = loss_reduction
        self.loss_constant = float(loss_constant)
        self.sample_loss = sample_loss

        # Defaults to template fast path; flipped by ``validate`` when a
        # DAG-walking stage (QEM/PauliTwirl) sits between us and PB.
        self._use_template_path: bool = True

    def _feature_batch(
        self, env: PipelineEnv, *, assert_finite: bool = True
    ) -> np.ndarray:
        """Read and validate the run's feature batch from the env.

        ``assert_finite`` is turned off by the dry/analytic path, which uses
        only the sample count and never renders the values into circuits.
        """
        if env.feature_batch is None:
            raise ValueError("DataBindingStage requires env.feature_batch to be set.")
        arr = np.asarray(env.feature_batch, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"feature_batch must be 2D; got shape {arr.shape}.")
        if arr.shape[1] != len(self.data_params):
            raise ValueError(
                f"feature_batch has {arr.shape[1]} columns but "
                f"{len(self.data_params)} data parameters were declared."
            )
        if assert_finite:
            _assert_finite(arr, source="env.feature_batch")
        return arr

    def _weight_params(self, mc: MetaCircuit) -> tuple[Parameter, ...]:
        """Weights = the incoming MetaCircuit's parameters minus ``data_params``.

        Order is preserved, so the surviving weights match the column order the
        optimizer's ``param_sets`` use.
        """
        data_set = set(self.data_params)
        return tuple(p for p in mc.parameters if p not in data_set)

    def validate(self, before: tuple[Stage, ...], after: tuple[Stage, ...]) -> None:
        """Pick template vs. eager path based on downstream DAG consumers.

        The template path emits per-sample QASM strings and a shared
        parametric DAG; it requires that no downstream stage walk
        ``circuit_bodies`` gate-by-gate. A stage is treated as a
        DAG-walker only when its forward pass is not structurally a
        no-op:

        * :class:`~divi.pipeline.stages.ParameterBindingStage` is
          transparent — its fast path reads the parked ``qasm_bodies``
          and never touches the shared DAG.
        * :class:`~divi.pipeline.stages.QEMStage` with the default
          :class:`~divi.circuits.qem._NoMitigation` protocol is
          transparent — its ``expand`` returns the input DAG unchanged.
        * Everything else with ``consumes_dag_bodies=True`` (active QEM
          protocols, :class:`~divi.pipeline.stages.PauliTwirlStage`,
          user-defined DAG-mutating stages) forces the eager fallback.
        """

        def _walks_dag(stage: Stage) -> bool:
            if not getattr(stage, "consumes_dag_bodies", False):
                return False
            if isinstance(stage, ParameterBindingStage):
                return False
            if isinstance(stage, QEMStage) and isinstance(
                stage.protocol, _NoMitigation
            ):
                return False
            return True

        self._use_template_path = not any(_walks_dag(s) for s in after)

    def _bind_sample_dag(self, sample: np.ndarray, body_dag: DAGCircuit) -> DAGCircuit:
        """Rebuild ``body_dag`` with this sample's data values bound in place.

        Used by the eager path. Binds the data parameters directly in the DAG
        (weights stay symbolic) via :func:`~divi.circuits.bind_parameters_in_dag`
        — no round-trip through a ``QuantumCircuit``. The returned DAG escapes
        the stage only via the ``expand`` batch dict, so no per-sample DAG state
        lives on ``self``.
        """
        substitution = {
            param: float(value) for param, value in zip(self.data_params, sample)
        }
        return bind_parameters_in_dag(body_dag, substitution)

    def _render_sample_body(
        self, sample: np.ndarray, template: QASMTemplate, precision: int
    ) -> str:
        """Render one sample's partial QASM body (data substituted, weights
        left as placeholders). Used by the template path."""
        formatted = tuple(_format_bound_param(float(v), precision) for v in sample)
        return render_template(template, formatted)

    def expand(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> tuple[ExpansionResult, StageToken]:
        feature_batch = self._feature_batch(env)
        if self._use_template_path:
            return self._expand_template(batch, feature_batch)
        return self._expand_eager(batch, feature_batch)

    def _expand_template(
        self, batch: MetaCircuitBatch, feature_batch: np.ndarray
    ) -> tuple[ExpansionResult, StageToken]:
        """Fast path: per-sample partial QASM bodies + shared parametric DAG.

        All variants share the incoming body DAG reference in
        ``circuit_bodies``; the per-sample data substitution lives in
        ``qasm_bodies`` keyed by ``body_tag``.
        :meth:`ParameterBindingStage._fast_prepare` consults that field
        before falling back to deriving a body from the DAG.
        """
        out: MetaCircuitBatch = {}
        for key, mc in batch.items():
            weight_params = self._weight_params(mc)
            fanned_bodies: list[tuple] = []
            partial_bodies: list[tuple] = []
            for body_tag, body_dag in mc.circuit_bodies:
                template = _template_cached(
                    _qasm_body_cached(body_dag, mc.precision),
                    self._data_param_names,
                )
                for sample_idx, sample in enumerate(feature_batch):
                    tag = (*body_tag, (DATA_AXIS, sample_idx))
                    fanned_bodies.append((tag, body_dag))
                    partial_bodies.append(
                        (tag, self._render_sample_body(sample, template, mc.precision))
                    )
            out[key] = replace(
                mc,
                circuit_bodies=tuple(fanned_bodies),
                parameters=weight_params,
                qasm_bodies=tuple(partial_bodies),
            )
        return ExpansionResult(batch=out), None

    def _expand_eager(
        self, batch: MetaCircuitBatch, feature_batch: np.ndarray
    ) -> tuple[ExpansionResult, StageToken]:
        """Fallback path: one bound DAG per sample.

        Used when a DAG-walking stage (e.g. QEM, Pauli twirl) sits
        between this stage and ParameterBinding — those stages need
        per-sample concrete DAGs to fold or twirl.
        """
        out: MetaCircuitBatch = {}
        for key, mc in batch.items():
            weight_params = self._weight_params(mc)
            fanned = tuple(
                (
                    (*body_tag, (DATA_AXIS, sample_idx)),
                    self._bind_sample_dag(sample, body_dag),
                )
                for body_tag, body_dag in mc.circuit_bodies
                for sample_idx, sample in enumerate(feature_batch)
            )
            out[key] = replace(mc, circuit_bodies=fanned, parameters=weight_params)
        return ExpansionResult(batch=out), None

    def dry_expand(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> tuple[ExpansionResult, StageToken]:
        """Dry path: share the incoming parametric DAG across all sample
        variants. Per-sample data substitution is skipped — dry-run only
        needs correct circuit counts and depth/width stats, both of which
        are invariant under data binding.
        """
        n_samples = self._feature_batch(env, assert_finite=False).shape[0]
        out: MetaCircuitBatch = {}
        for key, mc in batch.items():
            weight_params = self._weight_params(mc)
            fanned = tuple(
                ((*body_tag, (DATA_AXIS, sample_idx)), body_dag)
                for body_tag, body_dag in mc.circuit_bodies
                for sample_idx in range(n_samples)
            )
            out[key] = replace(mc, circuit_bodies=fanned, parameters=weight_params)
        return ExpansionResult(batch=out), None

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        labels = (
            None
            if env.labels is None
            else np.asarray(env.labels, dtype=np.float64).reshape(-1)
        )
        grouped = group_by_base_key(results, DATA_AXIS, indexed=True)
        # `reduce_postprocess_ordered` sorts each base key's per-sample values by
        # their data-axis index (robust to a sparse/dropped sample) before the
        # per-key reduction.
        return reduce_postprocess_ordered(
            grouped, functools.partial(self._reduce_one, labels=labels)
        )

    def _reduce_one(self, samples: list[Any], labels: np.ndarray | None) -> Any:
        """Apply ``loss_reduction`` to one base-key's per-sample values.

        :class:`~divi.pipeline.stages.MeasurementStage` returns a per-observable
        list of floats — usually length 1 for a QNN that targets a single
        cost observable. Reduce each observable's per-sample series
        independently so the multi-observable shape is preserved end-to-end.
        Scalar inputs (rare upstream path) are handled symmetrically.

        ``loss_constant`` is added to each per-sample value *before* the
        reduction so non-affine reductions (``"sum"``, custom callables)
        see the unshifted loss — adding the constant after the reduction
        would be correct only for ``"mean"``.

        When ``labels`` is given, each per-sample prediction (expectation value
        plus ``loss_constant``) is first mapped through ``sample_loss`` against
        its label, so the reduction aggregates a supervised training loss.
        """
        const = self.loss_constant
        # Per-observable values (a sequence of observables) have ndim >= 1;
        # scalars — Python float, numpy scalar, or 0-d array — have ndim 0
        # and take the scalar branch.
        first = samples[0]
        if np.ndim(first) > 0:
            n_obs = len(first)
            # Defensive stage-level invariant: QNN/CustomVQA always target a
            # single cost observable, so this never fires through their public
            # APIs — it only guards direct DataBindingStage use with a
            # multi-observable measurement, where "the prediction" is ambiguous.
            if labels is not None and n_obs != 1:
                raise ValueError(
                    f"Supervised labels require a single cost observable so each "
                    f"sample has one prediction; got {n_obs} observables."
                )
            return [
                float(
                    self.loss_reduction(
                        self._per_sample_loss(
                            np.asarray([float(s[i]) for s in samples], dtype=np.float64)
                            + const,
                            labels,
                        )
                    )
                )
                for i in range(n_obs)
            ]
        predictions = np.asarray([float(s) for s in samples], dtype=np.float64) + const
        return float(self.loss_reduction(self._per_sample_loss(predictions, labels)))

    def _per_sample_loss(
        self, predictions: npt.NDArray[np.float64], labels: np.ndarray | None
    ) -> npt.NDArray[np.float64]:
        """Map predictions to per-sample losses against ``labels``.

        Returns ``predictions`` unchanged in the unsupervised case
        (``labels is None``); otherwise applies ``sample_loss`` element-wise
        against the aligned labels.
        """
        if labels is None:
            return predictions
        if self.sample_loss is None:
            raise ValueError(
                "env.labels were provided but the stage has no sample_loss."
            )
        if predictions.shape[0] != labels.shape[0]:
            raise ValueError(
                f"got {predictions.shape[0]} per-sample predictions but "
                f"{labels.shape[0]} labels."
            )
        sample_loss = self.sample_loss
        return np.asarray(
            [
                sample_loss(float(pred), float(label))
                for pred, label in zip(predictions, labels)
            ],
            dtype=np.float64,
        )

    def introspect(
        self, batch: MetaCircuitBatch, env: PipelineEnv, token: StageToken
    ) -> dict[str, Any]:
        n_samples = 0 if env.feature_batch is None else len(env.feature_batch)
        return {
            "n_samples": n_samples,
            "n_data_params": len(self.data_params),
            "path": "template" if self._use_template_path else "eager",
        }
