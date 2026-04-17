# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from divi.circuits import MetaCircuit
from divi.circuits._conversions import (
    measurement_qasms_from_groups,
    sparse_pauli_op_to_ham_string,
)
from divi.pipeline._grouping import GroupingStrategy, compute_measurement_groups
from divi.pipeline.abc import (
    BundleStage,
    ChildResults,
    ExpansionResult,
    MetaCircuitBatch,
    PipelineEnv,
    ResultFormat,
    StageToken,
)
from divi.pipeline.transformations import group_by_base_key, reduce_postprocess_ordered

OBS_GROUP_AXIS = "obs_group"
PROBS_MEAS_AXIS = "meas"


@dataclass(frozen=True)
class MeasurementToken:
    """Token carrying measurement metadata from expand to reduce."""

    postprocess_fn_by_spec: dict[object, Callable] = field(default_factory=dict)
    """Per-spec postprocessing functions (combine groups with coefficients)."""

    is_probs: bool = False
    """True when this is a probabilities measurement (no observable grouping)."""

    effective_strategy: str | None = None
    """Strategy actually used (may differ from user-configured when auto-promoted
    to ``_backend_expval``). ``None`` for the probs path."""

    n_observable_terms: int | None = None
    """Number of single-Pauli terms in the source observable (only set for
    the ``_backend_expval`` path, where ``measurement_groups`` is a sentinel
    empty group)."""


class MeasurementStage(BundleStage):
    """Unified measurement stage for all circuit measurement types.

    Handles both ``probs()`` and ``expval(H)`` measurements, auto-detecting
    the measurement type from the MetaCircuit.  For ``expval(H)``, groups
    observables using the configured strategy (or delegates to expval-native
    backends via ``ham_ops``).

    During ``expand``, sets ``env.result_format`` to communicate the expected
    result format to ``pipeline.run()``:

    - ``ResultFormat.PROBS`` for ``probs()`` measurements
    - ``ResultFormat.EXPVALS`` for ``expval(H)`` measurements
    - ``ResultFormat.COUNTS`` is never set here (reserved for PCE)

    During ``reduce``, combines measurement groups using the postprocessing
    function from ``compute_measurement_groups``.
    """

    @property
    def axis_name(self) -> str | None:
        return OBS_GROUP_AXIS

    @property
    def handles_measurement(self) -> bool:
        return True

    @property
    def stateful(self) -> bool:
        return False

    def __init__(
        self,
        grouping_strategy: GroupingStrategy = "qwc",
        result_format_override: ResultFormat | None = None,
    ) -> None:
        """
        Args:
            grouping_strategy: Grouping strategy for expval observables.
                Ignored for ``probs()`` measurements.
                Defaults to "qwc" (qubit-wise commuting).
            result_format_override: If set, overrides the auto-detected result
                format. For example, pass ``ResultFormat.COUNTS`` to get raw
                shot counts even when an observable is present.
        """
        super().__init__(name=type(self).__name__)
        self._grouping_strategy = grouping_strategy
        self._result_format_override = result_format_override

    def expand(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> tuple[ExpansionResult, StageToken]:
        """Set up measurements and declare the result format on env."""
        sample_meta = next(iter(batch.values()))
        # Probs/counts circuits carry measured_wires; expval circuits carry
        # observable.  Exactly one is expected to be set.
        if sample_meta.observable is not None:
            result = self._expand_expval(batch, env)
        elif sample_meta.measured_wires is not None:
            result = self._expand_probs(batch, env)
        else:
            raise ValueError(
                "MeasurementStage: MetaCircuit has neither an observable "
                "nor measured_wires set. Did the spec stage run?"
            )

        if self._result_format_override is not None:
            env.result_format = self._result_format_override

        return result

    # ------------------------------------------------------------------ #
    # Probs path
    # ------------------------------------------------------------------ #

    def _expand_probs(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> tuple[ExpansionResult, StageToken]:
        """Generate measurement QASM for ``probs()`` / ``counts()`` circuits."""
        env.result_format = ResultFormat.PROBS

        out: dict[object, MetaCircuit] = {}

        for key, meta in batch.items():
            wires = meta.measured_wires
            if wires is None:
                # Defensive: expand() already dispatched based on observable
                # vs measured_wires, but the batch might be heterogeneous.
                raise ValueError(
                    f"MeasurementStage (probs path): key '{key}' has no "
                    "measured_wires set."
                )
            measure_qasm = "".join(f"measure q[{q}] -> c[{q}];\n" for q in wires)
            tagged_measurement_qasms = ((((PROBS_MEAS_AXIS, 0),), measure_qasm),)
            out[key] = meta.set_measurement_bodies(tagged_measurement_qasms)

        token = MeasurementToken(is_probs=True)
        return ExpansionResult(batch=out), token

    # ------------------------------------------------------------------ #
    # Expval path
    # ------------------------------------------------------------------ #

    def _expand_expval(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> tuple[ExpansionResult, StageToken]:
        """Group observables and generate measurement QASM (or ham_ops)."""
        env.result_format = ResultFormat.EXPVALS

        strategy = self._grouping_strategy
        if strategy in ("qwc", "_backend_expval") and env.backend.supports_expval:
            first_obs = next(iter(batch.values())).observable
            all_same_obs = all(meta.observable == first_obs for meta in batch.values())
            strategy = "_backend_expval" if all_same_obs else "qwc"

        result: dict[object, MetaCircuit] = {}
        postprocess_fn_by_spec: dict[object, Callable] = {}
        n_observable_terms: int | None = None

        for key, meta in batch.items():
            if meta.observable is None:
                raise ValueError(
                    f"MeasurementStage (expval path): key '{key}' has no "
                    "observable set."
                )

            measurement_groups, partition_indices, postprocessing_fn = (
                compute_measurement_groups(meta.observable, strategy, meta.n_qubits)
            )
            if strategy == "_backend_expval" and n_observable_terms is None:
                n_observable_terms = sum(len(p) for p in partition_indices)

            measurement_qasms = measurement_qasms_from_groups(
                measurement_groups, meta.n_qubits
            )
            tagged_measurement_qasms = tuple(
                (((OBS_GROUP_AXIS, idx),), meas_qasm)
                for idx, meas_qasm in enumerate(measurement_qasms)
            )

            result[key] = meta.set_measurement_bodies(
                tagged_measurement_qasms
            ).set_measurement_groups(measurement_groups)
            postprocess_fn_by_spec[key] = postprocessing_fn

        # For expval-native backends, compute ham_ops from the SparsePauliOp
        # and store it in env.artifacts so _default_execute_fn can use it.
        if strategy == "_backend_expval":
            sample_meta = next(iter(batch.values()))
            env.artifacts["ham_ops"] = sparse_pauli_op_to_ham_string(
                sample_meta.observable
            )
        else:
            env.artifacts.pop("ham_ops", None)

        token = MeasurementToken(
            postprocess_fn_by_spec=postprocess_fn_by_spec,
            effective_strategy=strategy,
            n_observable_terms=n_observable_terms,
        )
        return ExpansionResult(batch=result), token

    # ------------------------------------------------------------------ #
    def introspect(
        self,
        batch: MetaCircuitBatch,
        env: PipelineEnv,
        token: StageToken,
    ) -> dict[str, Any]:
        meta = next(iter(batch.values()), None)
        effective_strategy = getattr(token, "effective_strategy", None)
        info: dict[str, Any] = {
            "strategy": effective_strategy or self._grouping_strategy
        }
        if meta is None:
            return info

        # Backend-native expval: measurement_groups is a sentinel empty group
        # because the backend evaluates the full observable directly. Surface
        # the source observable instead of the empty placeholder.
        if effective_strategy == "_backend_expval":
            info["n_groups"] = 1
            if getattr(token, "n_observable_terms", None) is not None:
                info["n_terms"] = token.n_observable_terms
            obs_str = str(meta.observable) if meta.observable is not None else ""
            if len(obs_str) > 80:
                obs_str = obs_str[:77] + "..."
            info["observable"] = obs_str
            return info

        groups = meta.measurement_groups
        info["n_groups"] = len(groups)
        n_terms = sum(len(g) for g in groups)
        info["n_terms"] = n_terms
        # Largest group by term count
        if groups:
            largest = max(groups, key=len)
            info["largest_group"] = ", ".join(str(op) for op in largest[:5])
            if len(largest) > 5:
                info["largest_group"] += f" ... ({len(largest)} terms)"
        return info

    # Reduce
    # ------------------------------------------------------------------ #

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        """Combine measurement groups (expval) or strip meas axis (probs)."""
        if token.is_probs:
            return self._reduce_probs(results)
        if env.result_format in (ResultFormat.COUNTS, ResultFormat.PROBS):
            return self._reduce_raw(results)
        return self._reduce_expval(results, token)

    @staticmethod
    def _reduce_probs(results: ChildResults) -> ChildResults:
        """Strip the meas axis from probs results (identity reduce)."""
        out: ChildResults = {}
        for key, value in results.items():
            base_key = tuple(ax for ax in key if ax[0] != PROBS_MEAS_AXIS)
            out[base_key] = value
        return out

    @staticmethod
    def _reduce_raw(results: ChildResults) -> ChildResults:
        """Strip the obs_group axis without postprocessing.

        Used when the expand path grouped observables but the caller
        requested raw counts or probabilities instead of expectation values.
        """
        out: ChildResults = {}
        for key, value in results.items():
            base_key = tuple(ax for ax in key if ax[0] != OBS_GROUP_AXIS)
            out[base_key] = value
        return out

    def _reduce_expval(
        self, results: ChildResults, token: MeasurementToken
    ) -> ChildResults:
        """Combine expval results across measurement groups."""
        grouped = group_by_base_key(results, OBS_GROUP_AXIS, indexed=True)
        postprocess_fn_by_base = {
            base_key: token.postprocess_fn_by_spec[base_key] for base_key in grouped
        }
        return reduce_postprocess_ordered(grouped, postprocess_fn_by_base)
