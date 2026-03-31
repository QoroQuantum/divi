# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from divi.circuits import MetaCircuit
from divi.circuits._qasm_conversion import _measurements_to_qasm
from divi.hamiltonians import convert_hamiltonian_to_pauli_string
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
        # Auto-detect from the first MetaCircuit's measurement type.
        sample_meta = next(iter(batch.values()))
        measurement = sample_meta.source_circuit.measurements[0]
        is_probs = not hasattr(measurement, "obs") or measurement.obs is None

        if is_probs:
            result = self._expand_probs(batch, env)
        else:
            result = self._expand_expval(batch, env)

        if self._result_format_override is not None:
            env.result_format = self._result_format_override

        return result

    # ------------------------------------------------------------------ #
    # Probs path
    # ------------------------------------------------------------------ #

    def _expand_probs(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> tuple[ExpansionResult, StageToken]:
        """Generate 'measure all qubits' QASM for ``probs()`` circuits."""
        env.result_format = ResultFormat.PROBS

        out: dict[object, MetaCircuit] = {}

        for key, meta in batch.items():
            n_qubits = len(meta.source_circuit.wires)
            measure_qasm = "".join(
                f"measure q[{i}] -> c[{i}];\n" for i in range(n_qubits)
            )
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

        # Auto-select _backend_expval when the backend supports it and
        # all circuits share the same observable.  QDrift multi-sample
        # produces per-circuit observables that are incompatible with
        # the single ham_ops string, so we fall back to QWC grouping.
        strategy = self._grouping_strategy
        if strategy in ("qwc", "_backend_expval") and env.backend.supports_expval:
            first_obs = next(iter(batch.values())).source_circuit.measurements[0].obs
            all_same_obs = all(
                meta.source_circuit.measurements[0].obs is first_obs
                for meta in batch.values()
            )
            strategy = "_backend_expval" if all_same_obs else "qwc"

        result: dict[object, MetaCircuit] = {}
        postprocess_fn_by_spec: dict[object, Callable] = {}

        for key, meta in batch.items():
            if len(meta.source_circuit.measurements) != 1:
                raise ValueError(
                    f"MeasurementStage expects MetaCircuits with exactly one measurement, "
                    f"got {len(meta.source_circuit.measurements)}."
                )

            measurement = meta.source_circuit.measurements[0]
            measurement_groups, _, postprocessing_fn = compute_measurement_groups(
                measurement, strategy
            )
            measurement_qasms = _measurements_to_qasm(
                meta.source_circuit, measurement_groups, precision=meta.precision
            )
            tagged_measurement_qasms = tuple(
                (((OBS_GROUP_AXIS, idx),), meas_qasm)
                for idx, meas_qasm in enumerate(measurement_qasms)
            )

            result[key] = meta.set_measurement_bodies(
                tagged_measurement_qasms
            ).set_measurement_groups(measurement_groups)
            postprocess_fn_by_spec[key] = postprocessing_fn

        # For expval-native backends, compute ham_ops from the observable
        # and store it in env.artifacts so _default_execute_fn can use it.
        # When falling back to QWC, ensure ham_ops is cleared so a stale
        # value from a cached forward pass doesn't leak into the next run.
        if strategy == "_backend_expval":
            sample_meta = next(iter(batch.values()))
            observable = sample_meta.source_circuit.measurements[0].obs
            n_qubits = len(sample_meta.source_circuit.wires)
            ham_ops_str = convert_hamiltonian_to_pauli_string(
                observable, n_qubits, wires=sample_meta.source_circuit.wires
            )
            env.artifacts["ham_ops"] = ham_ops_str
        else:
            env.artifacts.pop("ham_ops", None)

        token = MeasurementToken(
            postprocess_fn_by_spec=postprocess_fn_by_spec,
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
        info: dict[str, Any] = {"strategy": self._grouping_strategy}
        if meta is None:
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
