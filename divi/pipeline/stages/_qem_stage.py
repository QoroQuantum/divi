# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import cirq
import numpy as np

from divi.circuits import MetaCircuit
from divi.circuits._qasm_conversion import (
    _cirq_circuit_from_qasm,
    normalize_qasm_after_cirq,
)
from divi.circuits.qem import QEMContext, QEMProtocol, _NoMitigation
from divi.circuits.quepp import QuEPP
from divi.pipeline.abc import (
    BundleStage,
    ChildResults,
    ContractViolation,
    ExpansionResult,
    MetaCircuitBatch,
    PipelineEnv,
    Stage,
    StageToken,
)
from divi.pipeline.stages._pauli_twirl_stage import PauliTwirlStage
from divi.pipeline.transformations import group_by_base_key

QEM_AXIS = "qem"


class QEMStage(BundleStage):
    """BundleStage that applies a QEM protocol to each circuit body.

    Threads :class:`QEMContext` objects through the ``StageToken`` so that
    ``reduce`` can pass them back to the protocol together with the quantum
    results.
    """

    @property
    def axis_name(self) -> str | None:
        return f"{QEM_AXIS}_{self.protocol.name}"

    @property
    def stateful(self) -> bool:
        return False

    def __init__(self, protocol: QEMProtocol | None = None) -> None:
        super().__init__(name=type(self).__name__)
        self.protocol = protocol if protocol is not None else _NoMitigation()

    def validate(self, before: tuple[Stage, ...], after: tuple[Stage, ...]) -> None:
        if isinstance(self.protocol, _NoMitigation):
            return

        # QuEPP's reduce uses classical simulation tied to the full
        # Hamiltonian, so observable groups must be recombined first.
        if isinstance(self.protocol, QuEPP):
            if not any(
                isinstance(s, BundleStage) and s.handles_measurement for s in after
            ):
                raise ContractViolation(
                    "QEMStage with QuEPP requires a measurement-handling "
                    "stage after it so that observable groups are "
                    "recombined before QEM reduction."
                )

        if isinstance(self.protocol, QuEPP) and self.protocol.n_twirls > 0:
            if not any(isinstance(s, PauliTwirlStage) for s in after):
                raise ContractViolation(
                    f"QEMStage with n_twirls={self.protocol.n_twirls} "
                    "requires a PauliTwirlStage after it in the pipeline."
                )

    def _expand_bodies(
        self, meta: MetaCircuit
    ) -> tuple[tuple[tuple, ...], list[QEMContext]]:
        """Apply the protocol to each body QASM and return tagged results + contexts."""
        mp = meta.source_circuit.measurements[0] if meta.source_circuit else None
        observable = getattr(mp, "obs", None) if mp else None
        ctxs: list[QEMContext] = []
        bodies: list[tuple] = []

        for tag, body in meta.circuit_body_qasms:
            circuits, ctx = self.protocol.expand(
                _cirq_circuit_from_qasm(body, meta.symbols), observable
            )
            ctxs.append(ctx)
            bodies.extend(
                ((*tag, (self.axis_name, i)), normalize_qasm_after_cirq(cirq.qasm(c)))
                for i, c in enumerate(circuits)
            )

        return tuple(bodies), ctxs

    def expand(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> tuple[ExpansionResult, StageToken]:
        out: dict[object, MetaCircuit] = {}
        contexts: dict[tuple, QEMContext] = {}

        for parent_key, meta in batch.items():
            bodies, ctxs = self._expand_bodies(meta)
            for (tag, _), ctx in zip(meta.circuit_body_qasms, ctxs):
                contexts[parent_key + tag] = ctx
            symbol_names = tuple(str(s) for s in meta.symbols)
            out[parent_key] = meta.set_circuit_bodies(bodies, symbol_names=symbol_names)

        return ExpansionResult(batch=out), contexts

    def _detect_per_obs(self, grouped: dict) -> bool:
        """Check whether results are per-observable dicts or scalars."""
        sample = next((v for g in grouped.values() for v in g.values()), None)
        if not isinstance(sample, dict):
            return False
        if isinstance(next(iter(sample)), str):
            if self.protocol.name != "NoMitigation":
                raise TypeError(
                    f"QEMStage expects scalar expectation values, "
                    f"but received probability dicts. "
                    f"{type(self.protocol).__name__} is not supported "
                    f"for probability-based measurements."
                )
            return False
        return True

    def _reduce_grouped(
        self,
        grouped: dict[tuple, dict[int, Any]],
        contexts: dict[tuple, QEMContext] | None,
        per_obs: bool,
    ) -> ChildResults:
        reduced: ChildResults = {}
        for base_key, values_by_idx in grouped.items():
            ordered = [v for _, v in sorted(values_by_idx.items())]
            ctx = {} if contexts is None else contexts[base_key]

            if per_obs:
                reduced[base_key] = {
                    obs_idx: self.protocol.reduce([d[obs_idx] for d in ordered], ctx)
                    for obs_idx in sorted(ordered[0].keys())
                }
            else:
                reduced[base_key] = self.protocol.reduce(ordered, ctx)
        return reduced

    def introspect(
        self, batch: MetaCircuitBatch, env: PipelineEnv, token: StageToken
    ) -> dict[str, Any]:
        info: dict[str, Any] = {"protocol": self.protocol.name}
        contexts: dict | None = token
        if not contexts:
            return info
        ctx = next(iter(contexts.values()), None)
        if ctx is None:
            return info
        data = ctx
        for key in ("n_rotations", "n_paths"):
            if key in data:
                info[key] = data[key]
        if "n_paths" in data:
            info["n_clifford_sims"] = data["n_paths"]
        weights = data.get("weights")
        if weights is not None and len(weights) > 0:
            info["weight_sum"] = round(float(np.sum(weights)), 4)
            info["weight_range"] = [
                round(float(np.min(weights)), 4),
                round(float(np.max(weights)), 4),
            ]
        classical = data.get("classical_values")
        if classical is not None and weights is not None and len(weights) > 0:
            info["classical_estimate"] = round(float(weights @ classical), 6)
        return info

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        contexts: dict[tuple, QEMContext] | None = token
        grouped = group_by_base_key(results, self.axis_name, indexed=True)
        per_obs = self._detect_per_obs(grouped)
        reduced = self._reduce_grouped(grouped, contexts, per_obs=per_obs)

        if contexts is not None:
            all_ctxs = list(contexts.values())
            self.protocol.post_reduce(all_ctxs)

        return reduced
