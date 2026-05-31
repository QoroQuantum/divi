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

from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
import numpy.typing as npt
from qiskit.converters import circuit_to_dag

from divi.circuits import MetaCircuit
from divi.pipeline import CircuitPipeline
from divi.pipeline.abc import Stage
from divi.pipeline.stages import (
    CircuitSpecStage,
    DataBindingStage,
    MeasurementStage,
    ParameterBindingStage,
    SampleLossFn,
    resolve_sample_loss,
)
from divi.qprog.variational_quantum_algorithm import _extract_param_set_idx

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
    :meth:`_build_cost_pipeline` cooperatively wraps the base one (mirroring how
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

    # Data-binding state each subclass populates during its own construction
    # (the rest of the host interface comes from VariationalQuantumAlgorithm
    # via ``_MixinBase`` under TYPE_CHECKING).
    _data_symbols: tuple["Parameter", ...]
    _weight_symbols: tuple["Parameter", ...]
    _composed_circuit: "QuantumCircuit"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # The cooperative super() calls below only run if DataBindingMixin
        # precedes the host VQA base in the MRO. A base that defines
        # ``_build_cost_pipeline`` placed *ahead* of the mixin would shadow it
        # and silently skip data binding (training on a data-free circuit), so
        # reject that ordering loudly at class-definition time.
        mro = cls.__mro__
        mixin_idx = mro.index(DataBindingMixin)
        shadowers = [c for c in mro[1:mixin_idx] if "_build_cost_pipeline" in vars(c)]
        if shadowers:
            raise TypeError(
                f"{cls.__name__} must list DataBindingMixin before "
                f"{shadowers[0].__name__}, which otherwise shadows the mixin's "
                f"_build_cost_pipeline and silently disables data binding."
            )

    def _build_cost_pipeline(
        self,
        spec_stage: Stage,
        extra_stages: tuple[Stage, ...] = (),
    ) -> CircuitPipeline:
        """Insert :class:`DataBindingStage` ahead of the base pipeline when active.

        When ``feature_batch`` is set, the data axis is fanned out before any
        stage that walks ``circuit_bodies`` (QEM/twirling/measurement) sees the
        bodies. With no feature batch (a plain ``CustomVQA``) this is a no-op and
        delegates straight to the base pipeline.
        """
        if getattr(self, "feature_batch", None) is None:
            return super()._build_cost_pipeline(spec_stage, extra_stages=extra_stages)
        return super()._build_cost_pipeline(
            spec_stage,
            extra_stages=(build_data_binding_stage(self), *extra_stages),
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

        Returns ``(None, None)`` for the unsupervised case (and warns if a
        non-default ``loss_fn`` was supplied without labels, since it is
        ignored). Otherwise returns the ``(n_samples,)`` label array and the
        resolved per-sample loss callable, raising if the label count does not
        match the sample count.
        """
        if labels is None:
            if loss_fn != "squared_error":
                warn(
                    "loss_fn was provided but labels is None, so loss_fn is "
                    "ignored. Pass labels (with a feature_batch) to train a "
                    "supervised loss.",
                    UserWarning,
                    stacklevel=2,
                )
            return None, None
        arr = np.asarray(labels, dtype=np.float64).reshape(-1)
        if arr.shape[0] != n_samples:
            raise ValueError(
                f"labels has {arr.shape[0]} entries but feature_batch has "
                f"{n_samples} samples."
            )
        return arr, resolve_sample_loss(loss_fn)

    def predict(
        self,
        features: npt.ArrayLike,
        params: npt.NDArray[np.float64] | None = None,
    ) -> np.ndarray:
        """Predict class labels for a feature batch with trained weights.

        Each row of ``features`` is bound into the composed circuit alongside the
        weights, the cost observable's expectation is estimated from shots (the
        same readout the loss optimizes), and its sign is the class label:
        ``+1`` for a non-negative readout, ``-1`` otherwise. The readout includes
        ``loss_constant`` so it matches the full observable.

        This works for any observable (the expectation is measured directly,
        with no computational-basis decoding), and shares the measurement
        machinery the rest of the program uses.

        Args:
            features: Shape ``(n_samples, n_data_params)`` (or a single
                ``(n_data_params,)`` row) feature batch to classify.
            params: Trained weights of shape ``(n_layers * n_params_per_layer,)``.
                Defaults to ``self.best_params``.

        Returns:
            numpy.ndarray: Shape ``(n_samples,)`` class labels in ``{-1.0, +1.0}``.

        Raises:
            RuntimeError: If ``params`` is ``None`` and the program has not been
                trained yet.
            ValueError: On a feature-column or weight-length mismatch.
        """
        readouts = self.predict_readout(features, params=params)
        return np.where(readouts >= 0.0, 1.0, -1.0)

    def predict_readout(
        self,
        features: npt.ArrayLike,
        params: npt.NDArray[np.float64] | None = None,
    ) -> np.ndarray:
        """Per-sample cost-observable expectation (the raw, unthresholded readout).

        Same contract as :meth:`predict` but returns the continuous
        ``⟨H⟩ + loss_constant`` per sample instead of the sign-thresholded class,
        for callers that want a score (e.g. a custom decision threshold).
        """
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
        readouts = self._measure_observable_for(joined)
        return readouts + self.loss_constant

    def _measure_observable_for(
        self, param_sets: npt.NDArray[np.float64]
    ) -> np.ndarray:
        """Run a one-shot measurement pipeline and return per-row ``⟨H⟩``.

        Mirrors the cost pipeline's measurement (observable expectation via
        shots) but without the data-binding stage or sample-axis reduction: the
        joined ``(data, weights)`` rows are bound directly, so each row's result
        is that sample's prediction. Does not mutate optimizer/solution state.
        """
        spec = MetaCircuit(
            circuit_bodies=(((), circuit_to_dag(self._composed_circuit)),),
            parameters=self._data_symbols + self._weight_symbols,
            observable=self.cost_hamiltonian,
            precision=self._precision,
        )
        pipeline = CircuitPipeline(
            stages=[
                CircuitSpecStage(),
                MeasurementStage(
                    grouping_strategy=self._grouping_strategy,
                    shot_distribution=self._shot_distribution,
                ),
                ParameterBindingStage(),
            ]
        )
        env = self._build_pipeline_env(param_sets=np.atleast_2d(param_sets))
        result = pipeline.run(initial_spec=spec, env=env)
        self._total_circuit_count += env.artifacts.get("circuit_count", 0)
        self._total_run_time += env.artifacts.get("run_time", 0.0)

        indexed = {
            _extract_param_set_idx(key): float(value[0])
            for key, value in result.items()
        }
        return np.asarray([indexed[i] for i in range(len(indexed))], dtype=np.float64)
