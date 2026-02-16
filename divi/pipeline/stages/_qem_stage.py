# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from divi.circuits import MetaCircuit
from divi.circuits.qem import QEMProtocol, _NoMitigation, apply_protocol_to_qasm
from divi.pipeline.abc import (
    BundleStage,
    ChildResults,
    ExpansionResult,
    MetaCircuitBatch,
    PipelineEnv,
    StageToken,
)
from divi.pipeline.transformations import group_by_base_key, reduce_postprocess_ordered

QEM_AXIS = "qem"


class QEMStage(BundleStage):
    """BundleStage that computes QEM circuit-body variants per MetaCircuit.

    Operates on circuit_body_qasm from MetaCircuit (computed at creation).
    Stores QEM-transformed body variants via set_circuit_bodies while
    preserving key cardinality (no fan-out).
    """

    @property
    def axis_name(self) -> str | None:
        return f"{QEM_AXIS}_{self._protocol.name}"

    @property
    def stateful(self) -> bool:
        return False

    def __init__(self, protocol: QEMProtocol | None = None) -> None:
        super().__init__(name=type(self).__name__)
        self._protocol = protocol if protocol is not None else _NoMitigation()

    def expand(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> tuple[ExpansionResult, StageToken]:
        out: dict[object, MetaCircuit] = {}

        for parent_key, meta in batch.items():
            updated_bodies = apply_protocol_to_qasm(
                meta.circuit_body_qasms,
                self._protocol,
                axis_name=self.axis_name,
                symbols=meta.symbols,
            )
            out[parent_key] = meta.set_circuit_bodies(updated_bodies)

        return ExpansionResult(batch=out), None

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        grouped = group_by_base_key(results, self.axis_name, indexed=True)

        sample_val = next((v for g in grouped.values() for v in g.values()), None)

        if isinstance(sample_val, dict):
            # Multi-obs expval dicts ({int: float}) from _counts_to_expvals
            # are fine — apply ZNE per observable index.
            # Probability dicts ({str: float}) are not supported.
            sample_key = next(iter(sample_val))
            if isinstance(sample_key, str):
                if not isinstance(self._protocol, _NoMitigation):
                    raise TypeError(
                        "QEMStage.reduce expects scalar expectation values, "
                        f"but received dict results. {self._protocol.__class__.__name__} "
                        "is not supported for probability-based measurements."
                    )
            else:
                return self._reduce_per_obs(grouped)

        return reduce_postprocess_ordered(grouped, self._protocol.postprocess_results)

    def _reduce_per_obs(
        self, grouped: dict[tuple, dict[int, dict[int, float]]]
    ) -> ChildResults:
        """Apply QEM postprocessing to each observable index independently.

        When ``_counts_to_expvals`` returns ``{obs_idx: float}`` dicts (for
        multi-observable measurement groups), the ZNE extrapolation must be
        applied per observable, not on the whole dict.
        """
        reduced: ChildResults = {}
        for base_key, values_by_scale in grouped.items():
            ordered = [v for _, v in sorted(values_by_scale.items())]
            obs_keys = sorted(ordered[0].keys())
            reduced[base_key] = {
                obs_idx: self._protocol.postprocess_results(
                    [scale_dict[obs_idx] for scale_dict in ordered]
                )
                for obs_idx in obs_keys
            }
        return reduced
