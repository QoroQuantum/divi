# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared data-binding wiring for VQA subclasses that fan a data axis out.

Both :class:`~divi.qprog.algorithms.QNN` and
:class:`~divi.qprog.algorithms.CustomVQA` (when given a feature batch) fan a
classical feature batch across a parameterized circuit. The shared behavior —
inserting a :class:`~divi.pipeline.stages.DataBindingStage` into the cost
pipeline, resolving optional supervised labels, and sampled-class inference via
:meth:`DataBindingMixin.predict` — lives in :class:`DataBindingMixin`.
``build_data_binding_stage`` owns the single stage construction so the two
stay in lockstep as the stage's signature evolves.

Each subclass still builds its own data/weight parameter split and composed
circuit; the mixin only orchestrates the data axis on top of that state.
"""

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from qiskit.converters import circuit_to_dag

from divi.circuits import MetaCircuit
from divi.pipeline import CircuitPipeline, ResultFormat
from divi.pipeline._compilation import _extract_param_set_idx
from divi.pipeline.abc import Stage
from divi.pipeline.stages import (
    CircuitSpecStage,
    DataBindingStage,
    LossReductionFn,
    MeasurementStage,
    SampleLossFn,
    resolve_loss_reduction,
    resolve_sample_loss,
)

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm

    # Type-check the mixin as if it were mixed into the VQA host, so ``super()``
    # calls and the inherited attributes/methods it relies on resolve. At
    # runtime the base is ``object`` — it is a genuine mixin.
    _MixinBase = VariationalQuantumAlgorithm
else:
    _MixinBase = object


# Emitted by QNN/CustomVQA when a supervised loss_fn is set without labels.
_LOSS_FN_IGNORED_MSG = (
    "loss_fn was provided but labels is None, so loss_fn is ignored. Pass "
    "labels (with a feature_batch) to train a supervised loss."
)


def build_data_binding_stage(program) -> DataBindingStage:
    """Build a :class:`DataBindingStage` from ``program``'s data-binding config.

    ``program`` must have ``_data_symbols``, ``_loss_reduction_fn``, and
    ``loss_constant`` set. ``_sample_loss_fn`` is optional (``None`` keeps the
    unsupervised path). The per-run ``feature_batch`` and ``labels`` are *not*
    passed here — the mixin injects them into the
    :class:`~divi.pipeline.abc.PipelineEnv`, and the stage reads them at run
    time; the circuit, weight parameters, and rendering precision all come from
    the MetaCircuit batch.
    """
    return DataBindingStage(
        data_params=program._data_symbols,
        loss_reduction=program._loss_reduction_fn,
        loss_constant=program.loss_constant,
        sample_loss=getattr(program, "_sample_loss_fn", None),
    )


class DataBindingMixin(_MixinBase):
    """Shared data-axis behavior for VQA subclasses that fan a feature batch out.

    Mixed in *before* :class:`~divi.qprog.VariationalQuantumAlgorithm` so its
    :meth:`_assemble_pipeline` cooperatively wraps the base one (mirroring how
    :class:`~divi.qprog.ObservableMeasuringMixin` sits ahead of
    ``QuantumProgram``). It owns the orchestration common to
    :class:`~divi.qprog.algorithms.QNN` and
    :class:`~divi.qprog.algorithms.CustomVQA`; each subclass still constructs the
    ``_data_symbols`` / ``_weight_symbols`` split and the ``_composed_circuit``
    itself and sets the attributes ``build_data_binding_stage`` reads.

    The mixin declares no ``__init__``: the data-binding state is populated
    during each subclass's own construction, so there is no init ordering to
    coordinate.
    """

    # Data-binding state each subclass populates; the rest of the host interface
    # comes from VariationalQuantumAlgorithm (resolved via ``_MixinBase``).
    _data_symbols: tuple["Parameter", ...]
    _weight_symbols: tuple["Parameter", ...]
    _composed_circuit: "QuantumCircuit"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # The cooperative super() calls below only run if DataBindingMixin
        # precedes the host VQA base in the MRO. A base that defines
        # ``_assemble_pipeline`` placed *ahead* of the mixin would shadow it
        # and silently skip data binding (training on a data-free circuit), so
        # reject that ordering loudly at class-definition time.
        mro = cls.__mro__
        mixin_idx = mro.index(DataBindingMixin)
        shadowers = [c for c in mro[1:mixin_idx] if "_assemble_pipeline" in vars(c)]
        if shadowers:
            raise TypeError(
                f"{cls.__name__} must list DataBindingMixin before "
                f"{shadowers[0].__name__}, which otherwise shadows the mixin's "
                f"_assemble_pipeline and silently disables data binding."
            )

    @property
    def _loss_constant_consumed(self) -> bool:
        # DataBindingStage folds loss_constant into each per-sample value, so
        # the base cost path must not re-add it.
        return getattr(self, "feature_batch", None) is not None

    def _assemble_pipeline(
        self,
        spec_stage: Stage,
        terminal_stage: Stage,
        *,
        result_format: ResultFormat,
        extra_stages: tuple[Stage, ...] = (),
    ) -> CircuitPipeline:
        """Insert :class:`DataBindingStage` ahead of mitigation when a data axis is
        active, so the data axis fans out before any stage that walks
        ``circuit_bodies`` (QEM/twirling/measurement) sees the bodies. With no
        feature batch (a plain ``CustomVQA``) this is a no-op delegating to the
        base assembler. Applies to every pipeline assembled — cost and metric alike.
        """
        if getattr(self, "feature_batch", None) is not None:
            extra_stages = (build_data_binding_stage(self), *extra_stages)
        return super()._assemble_pipeline(
            spec_stage,
            terminal_stage,
            result_format=result_format,
            extra_stages=extra_stages,
        )

    def _build_pipeline_env(self, **overrides):
        """Inject the per-run data axis (``feature_batch`` / ``labels``) into the
        env so :class:`~divi.pipeline.stages.DataBindingStage` can read them,
        alongside the ``param_sets`` the base VQA adds. No-op when there is no
        data binding (a plain ``CustomVQA``)."""
        if getattr(self, "feature_batch", None) is not None:
            overrides.setdefault("feature_batch", self.feature_batch)
            overrides.setdefault("labels", getattr(self, "labels", None))
        return super()._build_pipeline_env(**overrides)

    def _resolve_supervision(
        self,
        labels: npt.ArrayLike | None,
        loss_fn: SampleLossFn,
        n_samples: int,
    ) -> tuple[np.ndarray | None, "object | None"]:
        """Validate optional supervised labels and resolve the per-sample loss.

        Returns ``(None, None)`` for the unsupervised case. Otherwise returns the
        ``(n_samples,)`` label array and the resolved per-sample loss callable,
        raising if the label count does not match the sample count.

        Does not warn about an ignored ``loss_fn``: each constructor emits that
        warning itself so its ``stacklevel`` points at the user's call.
        """
        if labels is None:
            return None, None
        arr = np.asarray(labels, dtype=np.float64).reshape(-1)
        if arr.shape[0] != n_samples:
            raise ValueError(
                f"labels has {arr.shape[0]} entries but feature_batch has "
                f"{n_samples} samples."
            )
        return arr, resolve_sample_loss(loss_fn)

    @staticmethod
    def _validate_feature_batch(
        feature_batch: npt.ArrayLike, n_data_params: int
    ) -> np.ndarray:
        """Coerce ``feature_batch`` to a 2D ``(n_samples, n_data_params)`` array."""
        arr = np.asarray(feature_batch, dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(
                f"feature_batch must be 2D (n_samples, n_data_params); "
                f"got shape {arr.shape}."
            )
        if arr.shape[1] != n_data_params:
            raise ValueError(
                f"feature_batch has {arr.shape[1]} columns but the circuit "
                f"binds {n_data_params} data parameters."
            )
        if arr.shape[0] == 0:
            raise ValueError("feature_batch must contain at least one sample.")
        return arr

    def _set_loss_reduction(self, loss_reduction: LossReductionFn) -> None:
        """Store the user-facing reduction and its resolved callable."""
        self.loss_reduction = loss_reduction
        self._loss_reduction_fn = resolve_loss_reduction(loss_reduction)

    def _cost_meta_circuit(self, parameters: Iterable["Parameter"]) -> MetaCircuit:
        """Cost MetaCircuit for the composed circuit in the given parameter order.

        The (composed circuit, observable, precision) construction lives here;
        the parameter *order* stays subclass-owned (QNN: data+weights; CustomVQA:
        the original circuit order). The composed DAG is converted once.
        """
        dag = getattr(self, "_composed_dag", None)
        if dag is None:
            dag = circuit_to_dag(self._composed_circuit)
            self._composed_dag = dag
        return MetaCircuit(
            circuit_bodies=(((), dag),),
            parameters=tuple(parameters),
            observable=self.cost_hamiltonian,
            precision=self._precision,
        )

    def predict(
        self,
        features: npt.ArrayLike,
        params: npt.NDArray[np.float64] | None = None,
        *,
        return_scores: bool = False,
    ) -> np.ndarray:
        """Predict for a feature batch with trained weights.

        Each row of ``features`` is bound into the composed circuit alongside the
        weights and the cost observable's expectation is estimated from shots —
        the same score the loss optimizes, including ``loss_constant`` so it
        matches the full observable. By default the sign of that score is the
        class label: ``+1`` for a non-negative score, ``-1`` otherwise. Pass
        ``return_scores=True`` to get the continuous scores instead (e.g. for a
        custom decision threshold or a regression-style output).

        This works for any observable (the expectation is measured directly,
        with no computational-basis decoding), and shares the measurement
        machinery the rest of the program uses.

        Args:
            features: Shape ``(n_samples, n_data_params)`` (or a single
                ``(n_data_params,)`` row) feature batch.
            params: Trained weights of shape ``(n_layers * n_params_per_layer,)``.
                Defaults to ``self.best_params``.
            return_scores: When ``True``, return the continuous per-sample score
                ``⟨H⟩ + loss_constant`` instead of the sign-thresholded label.

        Returns:
            numpy.ndarray: Shape ``(n_samples,)`` — class labels in
                ``{-1.0, +1.0}`` by default, or continuous scores when
                ``return_scores`` is ``True``.

        Raises:
            RuntimeError: If the program has no data axis, or if ``params`` is
                ``None`` and the program has not been trained yet.
            ValueError: On a feature-column or weight-length mismatch.
        """
        if getattr(self, "_data_symbols", None) is None:
            raise RuntimeError(
                "predict() requires a data axis, but this program was created "
                "without a feature_batch."
            )
        feature_arr = np.atleast_2d(np.asarray(features, dtype=np.float64))
        n_data = len(self._data_symbols)
        if feature_arr.shape[1] != n_data:
            raise ValueError(
                f"features has {feature_arr.shape[1]} columns but the circuit "
                f"binds {n_data} data parameters."
            )

        if params is None:
            # Read the private attribute directly: the public ``best_params``
            # property warns when untrained, but here an empty value is an
            # expected branch we turn into a clear error.
            if len(self._best_params) == 0:
                raise RuntimeError(
                    "predict() needs trained weights but none are available. "
                    "Pass params=... or call run() first."
                )
            weights = np.asarray(self._best_params, dtype=np.float64).reshape(-1)
        else:
            weights = np.asarray(params, dtype=np.float64).reshape(-1)
        n_weights = len(self._weight_symbols)
        if weights.shape[0] != n_weights:
            raise ValueError(
                f"params has {weights.shape[0]} weights but the circuit has "
                f"{n_weights} weight parameters."
            )

        # Each sample becomes one param-set row in the full (data + weights)
        # space — no DataBindingStage, no reduction. Columns follow the spec's
        # parameter order: data symbols first, then weights.
        joined = np.hstack([feature_arr, np.tile(weights, (feature_arr.shape[0], 1))])
        scores = self._measure_observable_for(joined) + self.loss_constant
        if return_scores:
            return scores
        return np.where(scores >= 0.0, 1.0, -1.0)

    def _measure_observable_for(
        self, param_sets: npt.NDArray[np.float64]
    ) -> np.ndarray:
        """Run the cost pipeline for the given rows and return per-row ``⟨H⟩``.

        Builds the same pipeline training uses (QEM, twirling, measurement,
        binding) minus the data-binding fan-out and sample-axis reduction, so
        the measurement model matches training exactly: each joined
        ``(data, weights)`` row is bound directly, and that row's result is the
        sample's prediction. Does not mutate optimizer/solution state.
        """
        spec = self._cost_meta_circuit(self._data_symbols + self._weight_symbols)
        # super() (not self) skips the mixin's data-binding injection: the predict
        # pipeline binds each joined (data, weights) row directly, with no data fan-out.
        # Data-bound programs measure expectation values; the predict pipeline
        # mirrors the cost terminal (a plain MeasurementStage) without the
        # data-binding fan-out.
        pipeline = super()._assemble_pipeline(
            CircuitSpecStage(),
            MeasurementStage(
                grouping_strategy=self._grouping_strategy,
                shot_distribution=self._shot_distribution,
            ),
            result_format=ResultFormat.EXPVALS,
        )
        # Base env (not the mixin override): the predict pipeline has no
        # DataBindingStage, so feature_batch/labels must not enter the env.
        # reporter=None keeps inference silent — no progress spinner.
        env = super()._build_pipeline_env(
            param_sets=np.atleast_2d(param_sets), reporter=None
        )
        result = pipeline.run(initial_spec=spec, env=env)
        self._total_circuit_count += env.artifacts.get("circuit_count", 0)
        self._total_run_time += env.artifacts.get("run_time", 0.0)

        indexed = {
            _extract_param_set_idx(key): float(value[0])
            for key, value in result.items()
        }
        return np.asarray([indexed[i] for i in range(len(indexed))], dtype=np.float64)
