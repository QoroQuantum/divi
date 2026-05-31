# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import functools
import warnings
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit

from divi.backends import SupportsCircuitTemplates
from divi.circuits import MetaCircuit, render_template
from divi.circuits._conversions import _assert_finite, _format_bound_param
from divi.pipeline._compilation import PARAM_SET_AXIS
from divi.pipeline.abc import (
    BundleStage,
    ChildResults,
    DiviPerformanceWarning,
    ExpansionResult,
    MetaCircuitBatch,
    PipelineEnv,
    Stage,
    StageToken,
)
from divi.pipeline.stages import QEMStage
from divi.pipeline.stages._qasm_cache import _qasm_body_cached, _template_cached


def _validate_param_sets(env: PipelineEnv, *, assert_finite: bool = True) -> np.ndarray:
    """Coerce ``env.param_sets`` to a 2D float array, raising on invalid shape.

    ``assert_finite`` is turned off by the dry/analytic path, which uses only
    the param-set count and never renders the values into circuits.
    """
    param_sets = np.asarray(env.param_sets, dtype=float)
    if param_sets.ndim != 2:
        raise ValueError("ParameterBindingStage expects env.param_sets to be 2D.")
    if assert_finite:
        _assert_finite(param_sets, source="env.param_sets")
    return param_sets


def _iterate_bodies_over_param_sets(
    node: MetaCircuit,
    param_sets: np.ndarray,
    *,
    prepare: Callable[[MetaCircuit, tuple, DAGCircuit], Any],
    emit: Callable[[MetaCircuit, Any, np.ndarray], Any],
) -> list[tuple[tuple, Any]]:
    """Shared parametric loop for the real (non-dry) paths.

    Walks ``node.circuit_bodies`` once to precompute per-variant state via
    ``prepare``, then iterates ``param_sets`` producing one ``((*body_tag,
    (PARAM_SET_AXIS, param_set_idx)), emit(...))`` entry per (body, param set).
    Both the fast (QASM-template) and slow (DAG assign) paths plug into this
    helper via different ``prepare`` / ``emit`` callables.

    ``prepare`` receives ``(node, body_tag, body_dag)`` so it can consult
    per-tag state on the MetaCircuit (e.g. the ``qasm_bodies`` partials
    pre-rendered by :class:`~divi.pipeline.stages.DataBindingStage`'s
    template fast path).

    Dry mode skips this helper entirely: there is no per-variant state to
    cache, only shape-correct placeholders to emit — see
    :meth:`ParameterBindingStage._run_dry`.
    """
    n_params = len(node.parameters)
    prepared = tuple(
        (body_tag, prepare(node, body_tag, body_dag))
        for body_tag, body_dag in node.circuit_bodies
    )

    bound: list[tuple[tuple, Any]] = []
    for param_set_idx, param_values in enumerate(param_sets):
        if len(param_values) != n_params:
            raise ValueError(
                f"ParameterBindingStage expected {n_params} parameters, "
                f"got {len(param_values)} in param set {param_set_idx}."
            )
        param_set_tag = (PARAM_SET_AXIS, param_set_idx)
        for body_tag, prepared_item in prepared:
            bound.append(
                (
                    (*body_tag, param_set_tag),
                    emit(node, prepared_item, param_values),
                )
            )
    return bound


# ---------------------------------------------------------------------------
# Fast-path (QASM-template) prepare/emit pair.
# ---------------------------------------------------------------------------

# A prefix index over ``qasm_bodies``: a ``{stored_tag: body}`` map plus its
# distinct tag lengths (descending). Built once per node and reused across that
# node's bodies.
PrefixIndex = tuple[dict[tuple, str], tuple[int, ...]]


def _build_prefix_index(
    qasm_bodies: tuple[tuple[tuple, str], ...],
) -> PrefixIndex:
    index = {stored_tag: body for stored_tag, body in qasm_bodies}
    lengths = tuple(sorted({len(tag) for tag in index}, reverse=True))
    return index, lengths


def _prefix_index_for(node: MetaCircuit) -> PrefixIndex | None:
    """Build the prefix index over ``node.qasm_bodies`` (an upstream stage's
    parked partials), or ``None`` when none are present."""
    if node.qasm_bodies:
        return _build_prefix_index(node.qasm_bodies)
    return None


def _lookup_or_serialize(
    prefix_index: PrefixIndex | None,
    body_tag: tuple,
    dag: DAGCircuit,
    precision: int,
) -> str:
    """Find the pre-rendered parametric body for ``body_tag``, or serialize the DAG.

    Upstream stages park their partial body at the tag they had when they
    ran; later stages (QEM, MeasurementStage, …) may extend ``body_tag``
    with additional axes, so the lookup probes ``body_tag`` truncated to each
    stored tag length.

    Falls back to :func:`_qasm_body_cached` when no entry matches. This is
    safe: :class:`~divi.pipeline.stages.DataBindingStage` parks one entry per
    fanned body at that body's exact data tag, and every fanned body reaches
    here with that tag as a prefix — so a data-fanned body always matches. A
    non-match means a body that was never data-fanned, whose ``dag`` is fully
    parametric and serializes correctly.
    """
    if prefix_index is not None:
        index, lengths = prefix_index
        for n in lengths:
            body = index.get(body_tag[:n])
            if body is not None:
                return body
    return _qasm_body_cached(dag, precision)


def _fast_prepare(
    node: MetaCircuit,
    body_tag: tuple,
    dag: DAGCircuit,
    *,
    prefix_index: PrefixIndex | None = None,
):
    """Build a parametric QASM template for one body.

    Prefers the per-sample partial body parked in
    :attr:`~divi.circuits.MetaCircuit.qasm_bodies` (populated by upstream
    stages such as :class:`~divi.pipeline.stages.DataBindingStage`'s template
    fast path), falling back to deriving from the body DAG otherwise.
    """
    param_names = tuple(p.name for p in node.parameters)
    body = _lookup_or_serialize(prefix_index, body_tag, dag, node.precision)
    return _template_cached(body, param_names)


def _fast_emit(node: MetaCircuit, template, values: np.ndarray) -> str:
    """Render a concrete QASM string from a template + parameter values."""
    formatted = tuple(_format_bound_param(v, node.precision) for v in values)
    return render_template(template, formatted)


# ---------------------------------------------------------------------------
# Slow-path (DAG assign_parameters) prepare/emit pair.
# ---------------------------------------------------------------------------


def _slow_prepare(node: MetaCircuit, _body_tag: tuple, dag: DAGCircuit):
    """Convert a body DAG to a QuantumCircuit once, before the param-set loop.

    ``_body_tag`` is accepted for signature symmetry with :func:`_fast_prepare`
    but unused — the slow path always derives from the DAG.
    """
    return dag_to_circuit(dag)


def _slow_emit(node: MetaCircuit, qc, values: np.ndarray) -> DAGCircuit:
    """Bind a concrete parameter set into a prepared QuantumCircuit template."""
    binding = dict(zip(node.parameters, values))
    return circuit_to_dag(qc.assign_parameters(binding, inplace=False))


class ParameterBindingStage(BundleStage):
    """Bind ``env.param_sets`` into every circuit body.

    Two output shapes, selected at pipeline construction via
    :meth:`validate`:

    * **Fast path** — when no downstream stage reads
      ``meta.circuit_bodies`` (i.e. all subsequent stages declare
      ``consumes_dag_bodies=False``). Each body DAG is serialised once to
      a parametric QASM string, wrapped in a
      :class:`~divi.circuits.QASMTemplate`, and rendered per parameter
      set into ``meta.qasm_bodies`` with ``parameters`` cleared.  The
      pipeline's compilation pass then reads the bound strings directly.
    * **Slow path** — when any downstream stage consumes body DAGs (e.g.
      ``PauliTwirlStage`` or ``QEMStage``). For each body variant and
      parameter set, :meth:`qiskit.circuit.QuantumCircuit.assign_parameters`
      emits a bound :class:`~qiskit.dagcircuit.DAGCircuit` written back to
      ``meta.circuit_bodies`` (``parameters`` cleared). Slower, but preserves
      the DAG IR so downstream stages see concrete gate angles.

    A third **template** path (:meth:`_run_template`, when the backend
    implements :class:`~divi.backends.SupportsCircuitTemplates`) renders the
    parametric body into ``meta.qasm_bodies`` *without* binding, leaving
    ``parameters`` intact so the compilation pass defers substitution to the
    backend. Bound vs. template is therefore decided downstream by whether
    ``parameters`` survives, not by a dedicated field.

    :meth:`dry_expand` skips the prepare/emit helper entirely and emits
    shape-correct placeholders directly — the analytic path has no per-
    variant state to cache, so forcing it through the shared loop would
    only add dispatch noise.
    """

    @property
    def axis_name(self) -> str:
        return PARAM_SET_AXIS

    @property
    def stateful(self) -> bool:
        return True

    def __init__(self) -> None:
        super().__init__(name=type(self).__name__)

    def validate(self, before: tuple[Stage, ...], after: tuple[Stage, ...]) -> None:
        # ``consumes_dag_bodies`` is declared on BundleStage with default
        # True, so non-bundle stages (none exist after index 0 by pipeline
        # contract) or older third-party stages without the attribute
        # default to True — the safe assumption.
        self._fast_path = not any(
            getattr(s, "consumes_dag_bodies", True) for s in after
        )

        if any(
            isinstance(s, QEMStage) and s.protocol.name != "NoMitigation" for s in after
        ):
            warnings.warn(
                "ParameterBindingStage is placed before QEMStage. This "
                "forces QEM to re-expand on every bound parameter variant "
                "(one full QEM pass per param set). Consider placing "
                "ParameterBindingStage after QEMStage.",
                DiviPerformanceWarning,
                stacklevel=3,
            )

    def expand(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> tuple[ExpansionResult, StageToken]:
        param_sets = _validate_param_sets(env)
        if self._template_path_enabled(env):
            run = self._run_template
        else:
            run = self._run_fast if self._fast_path else self._run_slow
        return ExpansionResult(batch=run(batch, param_sets)), None

    def dry_expand(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> tuple[ExpansionResult, StageToken]:
        # Analytic path: only the param-set count matters, so skip the
        # finiteness check on values that are never rendered.
        param_sets = _validate_param_sets(env, assert_finite=False)
        return ExpansionResult(batch=self._run_dry(batch, param_sets)), None

    def _template_path_enabled(self, env: PipelineEnv) -> bool:
        """Whether to defer parameter binding to the backend for this run.

        Requires both the fast-path condition (no downstream stage consumes
        DAG bodies) and the active backend implementing the
        :class:`~divi.backends.SupportsCircuitTemplates` capability protocol.
        """
        if not self._fast_path:
            return False
        return isinstance(env.backend, SupportsCircuitTemplates)

    def _run_fast(
        self, batch: MetaCircuitBatch, param_sets: np.ndarray
    ) -> MetaCircuitBatch:
        out: MetaCircuitBatch = {}
        for key, node in batch.items():
            if len(node.parameters) == 0:
                # No weights to bind, but bodies may still carry per-sample data
                # baked in by DataBindingStage (parked in qasm_bodies), so
                # consult those before serialising the shared DAG. Parameters are
                # already empty here.
                prefix_index = _prefix_index_for(node)
                bodies = tuple(
                    (tag, _lookup_or_serialize(prefix_index, tag, dag, node.precision))
                    for tag, dag in node.circuit_bodies
                )
                out[key] = node.set_qasm_bodies(bodies)
                continue

            prefix_index = _prefix_index_for(node)
            bound = _iterate_bodies_over_param_sets(
                node,
                param_sets,
                prepare=functools.partial(_fast_prepare, prefix_index=prefix_index),
                emit=_fast_emit,
            )
            # Fully bound: clear parameters so compile routes this to the bound
            # path (empty parameters == bound).
            out[key] = replace(node, qasm_bodies=tuple(bound), parameters=())
        return out

    def _run_slow(
        self, batch: MetaCircuitBatch, param_sets: np.ndarray
    ) -> MetaCircuitBatch:
        out: MetaCircuitBatch = {}
        for key, node in batch.items():
            if len(node.parameters) == 0:
                # No parameters to bind; nothing to rewrite.
                out[key] = node
                continue

            bound = _iterate_bodies_over_param_sets(
                node, param_sets, prepare=_slow_prepare, emit=_slow_emit
            )
            # Fully bound into DAGs; clear parameters (empty == bound at compile).
            out[key] = replace(node, circuit_bodies=tuple(bound), parameters=())
        return out

    def _run_template(
        self, batch: MetaCircuitBatch, param_sets: np.ndarray
    ) -> MetaCircuitBatch:
        """Defer parameter substitution to a template-capable backend.

        Serialises each body DAG once into parametric QASM (named symbol
        placeholders preserved) and parks it in
        :attr:`~divi.circuits.MetaCircuit.qasm_bodies`, leaving ``parameters``
        intact. The compilation pass sees the surviving parameters and emits a
        payload of :class:`~divi.circuits.TemplateEntry` rows that the backend
        resolves per parameter set, replacing N near-identical bound circuits
        with one template plus N parameter vectors.
        """
        out: MetaCircuitBatch = {}
        for key, node in batch.items():
            if len(node.parameters) == 0:
                # No weights to defer. Emit bound bodies (consulting any
                # data-baked partials DataBindingStage parked) so compile takes
                # its normal bound route. Parameters already empty.
                prefix_index = _prefix_index_for(node)
                bodies = tuple(
                    (tag, _lookup_or_serialize(prefix_index, tag, dag, node.precision))
                    for tag, dag in node.circuit_bodies
                )
                out[key] = node.set_qasm_bodies(bodies)
                continue

            prefix_index = _prefix_index_for(node)
            template_bodies = tuple(
                (tag, _lookup_or_serialize(prefix_index, tag, dag, node.precision))
                for tag, dag in node.circuit_bodies
            )
            # Keep parameters: their presence is the "this is a template" signal
            # and supplies the payload's parameter_names.
            out[key] = node.set_qasm_bodies(template_bodies)
        return out

    def _run_dry(
        self, batch: MetaCircuitBatch, param_sets: np.ndarray
    ) -> MetaCircuitBatch:
        """Analytic path: emit shape-correct placeholders, no per-variant work.

        Fast-path emits empty QASM strings into ``qasm_bodies``; slow-path emits
        shared DAG references into ``circuit_bodies``.  The slot choice mirrors
        the real path so downstream stages (dry-aware or not) see the attribute
        they expect populated. ``parameters`` is left intact (unlike the real
        bound paths): :meth:`introspect` reports ``n_params`` from it, and dry
        traces are never routed through ``_batch_has_templates``.
        """
        n_param_sets = len(param_sets)
        out: MetaCircuitBatch = {}
        for key, node in batch.items():
            n_params = len(node.parameters)
            if self._fast_path:
                if n_params == 0:
                    bodies = tuple((tag, "") for tag, _ in node.circuit_bodies)
                else:
                    bodies = tuple(
                        ((*body_tag, (PARAM_SET_AXIS, i)), "")
                        for i in range(n_param_sets)
                        for body_tag, _ in node.circuit_bodies
                    )
                out[key] = node.set_qasm_bodies(bodies)
            else:
                if n_params == 0:
                    out[key] = node
                else:
                    bodies = tuple(
                        ((*body_tag, (PARAM_SET_AXIS, i)), body_dag)
                        for i in range(n_param_sets)
                        for body_tag, body_dag in node.circuit_bodies
                    )
                    out[key] = node.set_circuit_bodies(bodies)
        return out

    def introspect(
        self, batch: MetaCircuitBatch, env: PipelineEnv, token: StageToken
    ) -> dict[str, Any]:
        param_sets = np.asarray(env.param_sets, dtype=float)
        meta = next(iter(batch.values()), None)
        n_params = len(meta.parameters) if meta else 0
        return {
            "n_param_sets": len(param_sets),
            "n_params": n_params,
            "fast_path": self._fast_path,
            "template_path": self._template_path_enabled(env),
        }

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        return results
