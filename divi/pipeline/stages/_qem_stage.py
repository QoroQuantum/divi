# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Any

import numpy as np

from divi.circuits import MetaCircuit
from divi.circuits.qem import QEMContext, QEMProtocol, _NoMitigation
from divi.circuits.quepp import QuEPP
from divi.pipeline.abc import (
    BundleStage,
    ChildResults,
    ContractViolation,
    DiviPerformanceWarning,
    MetaCircuitBatch,
    PipelineEnv,
    Stage,
    StageOutput,
    StageToken,
)
from divi.pipeline.stages import PauliTwirlStage
from divi.pipeline.transformations import FOREIGN_KEY_ATTR, group_by_base_key

QEM_AXIS = "qem"


class QEMStage(BundleStage):
    """BundleStage that applies a QEM protocol to each circuit body.

    Threads ``QEMContext`` objects through the ``StageToken`` so that
    ``reduce`` can pass them back to the protocol together with the quantum
    results.
    """

    @property
    def axis_name(self) -> str:
        return f"{QEM_AXIS}_{self.protocol.name}"

    @property
    def volatile(self) -> bool:
        return False

    @property
    def consumes_dag_bodies(self) -> bool:
        return self._consumes_dag_bodies

    def __init__(self, protocol: QEMProtocol | None = None) -> None:
        self.protocol = protocol if protocol is not None else _NoMitigation()
        self._consumes_dag_bodies = not isinstance(self.protocol, _NoMitigation)
        super().__init__(name=type(self).__name__)

    def validate(self, before: tuple[Stage, ...], after: tuple[Stage, ...]) -> None:
        if not self.consumes_dag_bodies:
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

        if self.protocol.n_twirls > 0:
            if not any(isinstance(s, PauliTwirlStage) for s in after):
                raise ContractViolation(
                    f"QEMStage with n_twirls={self.protocol.n_twirls} "
                    "requires a PauliTwirlStage after it in the pipeline."
                )

        if isinstance(self.protocol, QuEPP) and self.protocol.requires_bound_params:
            warnings.warn(
                "QuEPP with sampling='exhaustive' enumerates all Pauli "
                "paths and scales poorly with truncation_order and circuit "
                "depth. Consider sampling='montecarlo' unless you "
                "specifically need deterministic enumeration. "
                "To suppress this warning, pass "
                "suppress_performance_warnings=True to CircuitPipeline, or "
                "filter DiviPerformanceWarning via warnings.filterwarnings "
                "(import it from divi.pipeline).",
                DiviPerformanceWarning,
                stacklevel=3,
            )

    def _expand_bodies(
        self, meta: MetaCircuit, protocol_fn
    ) -> tuple[tuple[tuple, ...], list[QEMContext]]:
        """Apply ``protocol_fn`` to each body DAG and return tagged results + contexts.

        ``protocol_fn`` is ``self.protocol.expand`` for a real run and
        ``self.protocol.dry_expand`` for a dry run; the outer iteration,
        axis tagging, and context bookkeeping stay identical either way.
        """
        observable = meta.observable
        ctxs: list[QEMContext] = []
        bodies: list[tuple] = []

        for tag, dag in meta.circuit_bodies:
            expanded_dags, ctx = protocol_fn(dag, observable)
            ctxs.append(ctx)
            for i, expanded in enumerate(expanded_dags):
                bodies.append(((*tag, (self.axis_name, i)), expanded))

        return tuple(bodies), ctxs

    def _expand_with(
        self, batch: MetaCircuitBatch, protocol_fn
    ) -> StageOutput[MetaCircuitBatch]:
        """Shared outer pass: applies ``protocol_fn`` per parent key."""
        out: MetaCircuitBatch = {}
        contexts: dict[tuple, QEMContext] = {}

        for parent_key, meta in batch.items():
            bodies, ctxs = self._expand_bodies(meta, protocol_fn)
            for (tag, _), ctx in zip(meta.circuit_bodies, ctxs):
                contexts[parent_key + tag] = ctx
            out[parent_key] = meta.set_circuit_bodies(bodies)

        return StageOutput(batch=out, token=contexts)

    def expand(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> StageOutput[MetaCircuitBatch]:
        return self._expand_with(batch, self.protocol.expand)

    def dry_expand(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> StageOutput[MetaCircuitBatch]:
        """Analytic path: delegates each body to ``protocol.dry_expand``.

        The default :meth:`~divi.circuits.qem.QEMProtocol.dry_expand` falls
        back to ``expand`` so simple / cheap protocols (e.g.
        ``_NoMitigation``, :class:`~divi.circuits.zne.ZNE`) stay correct
        unchanged. Expensive protocols such as
        :class:`~divi.circuits.quepp.QuEPP` override ``dry_expand`` to skip
        Clifford simulation + per-path DAG cloning while preserving the
        emitted DAG count.
        """
        return self._expand_with(batch, self.protocol.dry_expand)

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

        for key in ("n_rotations", "n_paths"):
            if key in ctx:
                info[key] = ctx[key]
        if "n_paths" in ctx:
            info["n_clifford_sims"] = ctx["n_paths"]

        # Skip float-dependent stats for symbolic weights (not yet bound).
        if ctx.get("symbolic"):
            info["weights"] = "unbound (run after parameter binding)"
            return info

        per_obs = ctx.get("per_obs")
        if per_obs:
            info["n_observables"] = len(per_obs)
            data = per_obs[0]
        else:
            # Dry path skips per_obs but persists the count separately so
            # introspect() can still surface it.
            if "n_observables" in ctx:
                info["n_observables"] = ctx["n_observables"]
            data = ctx

        weights = data.get("weights")
        if weights is not None and len(weights) > 0:
            info["weight_sum"] = round(float(np.sum(weights)), 4)
            # L1 norm = ∑|w_i|. Coincides with weight_sum for non-negative
            # weight schemes but diverges when any path carries a negative
            # weight (e.g. QuEPP sin-branches with sign flips). It is the
            # variance-amplification factor: a noiseless mitigated estimate
            # has variance proportional to (l1_norm)^2 / shots, so a value
            # of 3.2 means the user's error bar grows ~3.2× compared to
            # an unmitigated estimate at the same shot budget.
            info["weight_l1_norm"] = round(float(np.sum(np.abs(weights))), 4)
            info["weight_range"] = [
                round(float(np.min(weights)), 4),
                round(float(np.max(weights)), 4),
            ]
        classical = data.get("classical_values")
        if classical is not None and weights is not None and len(weights) > 0:
            info["classical_estimate"] = round(float(weights @ classical), 6)
        return info

    def _bind_symbolic_weights(
        self,
        contexts: dict[tuple, QEMContext],
        foreign_key: tuple,
        env: PipelineEnv,
    ) -> None:
        """Evaluate symbolic QuEPP weights using bound parameter values.

        When QuEPP runs before ParameterBindingStage, weights are stored
        as sympy expressions.  This method substitutes the concrete
        parameter values for the current param_set group, iterating over
        each ``per_obs`` slot.
        """
        param_idx = next((v for k, v in foreign_key if k == "param_set"), None)
        if param_idx is None:
            return
        param_values = np.asarray(env.param_sets, dtype=float)[param_idx]

        for key, ctx in list(contexts.items()):
            if not isinstance(ctx, dict) or not ctx.get("symbolic"):
                continue
            # Shallow-copy so different param_set groups don't share state.
            new_ctx = dict(ctx)
            symbols = new_ctx.get("weight_symbols", [])
            per_obs = new_ctx.get("per_obs")
            if per_obs:
                new_per_obs = []
                for entry in per_obs:
                    new_entry = dict(entry)
                    QuEPP.evaluate_symbolic_weights(new_entry, symbols, param_values)
                    new_per_obs.append(new_entry)
                new_ctx["per_obs"] = new_per_obs
            new_ctx["symbolic"] = False
            contexts[key] = new_ctx

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        # ``token`` carries tuple keys → ``QEMContext`` plus an optional
        # ``FOREIGN_KEY_ATTR`` (str) entry whose value is the active foreign
        # key tuple injected by ``_scope_token``.
        contexts: dict[Any, Any] | None = token if isinstance(token, dict) else None
        if contexts is not None:
            foreign_key = contexts.pop(FOREIGN_KEY_ATTR, ())
            self._bind_symbolic_weights(contexts, foreign_key, env)

        grouped = group_by_base_key(results, self.axis_name, indexed=True)
        per_obs = self._detect_per_obs(grouped)
        reduced = self._reduce_grouped(grouped, contexts, per_obs=per_obs)

        if contexts is not None:
            self.protocol.post_reduce(list(contexts.values()))

        return reduced
