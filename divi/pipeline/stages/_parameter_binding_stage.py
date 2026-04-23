# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit

from divi.circuits import MetaCircuit, build_template, dag_to_qasm_body, render_template
from divi.circuits._conversions import _format_bound_param
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
from divi.pipeline.stages._qem_stage import QEMStage

PARAM_SET_AXIS = "param_set"


def _validate_param_sets(env: PipelineEnv) -> np.ndarray:
    """Coerce ``env.param_sets`` to a 2D float array, raising on invalid shape."""
    param_sets = np.asarray(env.param_sets, dtype=float)
    if param_sets.ndim != 2:
        raise ValueError("ParameterBindingStage expects env.param_sets to be 2D.")
    return param_sets


def _iterate_bodies_over_param_sets(
    node: MetaCircuit,
    param_sets: np.ndarray,
    *,
    prepare: Callable[[MetaCircuit, DAGCircuit], Any],
    emit: Callable[[MetaCircuit, Any, np.ndarray], Any],
) -> list[tuple[tuple, Any]]:
    """Shared parametric loop for the real (non-dry) paths.

    Walks ``node.circuit_bodies`` once to precompute per-variant state via
    ``prepare``, then iterates ``param_sets`` producing one ``((*body_tag,
    (PARAM_SET_AXIS, param_set_idx)), emit(...))`` entry per (body, param set).
    Both the fast (QASM-template) and slow (DAG assign) paths plug into this
    helper via different ``prepare`` / ``emit`` callables.

    Dry mode skips this helper entirely: there is no per-variant state to
    cache, only shape-correct placeholders to emit — see
    :meth:`ParameterBindingStage._run_dry`.
    """
    n_params = len(node.parameters)
    prepared = tuple(
        (body_tag, prepare(node, body_dag))
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


def _fast_prepare(node: MetaCircuit, dag: DAGCircuit):
    """Build a parametric QASM template for one body DAG."""
    param_names = tuple(p.name for p in node.parameters)
    return build_template(dag_to_qasm_body(dag, precision=node.precision), param_names)


def _fast_emit(node: MetaCircuit, template, values: np.ndarray) -> str:
    """Render a concrete QASM string from a template + parameter values."""
    formatted = tuple(_format_bound_param(v, node.precision) for v in values)
    return render_template(template, formatted)


# ---------------------------------------------------------------------------
# Slow-path (DAG assign_parameters) prepare/emit pair.
# ---------------------------------------------------------------------------


def _slow_prepare(node: MetaCircuit, dag: DAGCircuit):
    """Convert a body DAG to a QuantumCircuit once, before the param-set loop."""
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
      set into ``meta.bound_circuit_bodies``.  The pipeline's compilation
      pass then reads the bound strings directly.
    * **Slow path** — when any downstream stage consumes body DAGs (e.g.
      ``PauliTwirlStage`` or ``QEMStage``). For each body variant and
      parameter set, :meth:`qiskit.circuit.QuantumCircuit.assign_parameters`
      emits a bound :class:`~qiskit.dagcircuit.DAGCircuit` written back to
      ``meta.circuit_bodies``. Slower, but preserves the DAG IR so
      downstream stages see concrete gate angles.

    :meth:`dry_expand` skips the prepare/emit helper entirely and emits
    shape-correct placeholders directly — the analytic path has no per-
    variant state to cache, so forcing it through the shared loop would
    only add dispatch noise.
    """

    @property
    def axis_name(self) -> str | None:
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
        run = self._run_fast if self._fast_path else self._run_slow
        return ExpansionResult(batch=run(batch, param_sets)), None

    def dry_expand(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> tuple[ExpansionResult, StageToken]:
        param_sets = _validate_param_sets(env)
        return ExpansionResult(batch=self._run_dry(batch, param_sets)), None

    def _run_fast(
        self, batch: MetaCircuitBatch, param_sets: np.ndarray
    ) -> dict[object, MetaCircuit]:
        out: dict[object, MetaCircuit] = {}
        for key, node in batch.items():
            if len(node.parameters) == 0:
                # Non-parametric: serialise each body once, no param-set expansion.
                bodies = tuple(
                    (tag, dag_to_qasm_body(dag, precision=node.precision))
                    for tag, dag in node.circuit_bodies
                )
                out[key] = node.set_bound_bodies(bodies)
                continue

            bound = _iterate_bodies_over_param_sets(
                node, param_sets, prepare=_fast_prepare, emit=_fast_emit
            )
            out[key] = node.set_bound_bodies(tuple(bound))
        return out

    def _run_slow(
        self, batch: MetaCircuitBatch, param_sets: np.ndarray
    ) -> dict[object, MetaCircuit]:
        out: dict[object, MetaCircuit] = {}
        for key, node in batch.items():
            if len(node.parameters) == 0:
                # No parameters to bind; nothing to rewrite.
                out[key] = node
                continue

            bound = _iterate_bodies_over_param_sets(
                node, param_sets, prepare=_slow_prepare, emit=_slow_emit
            )
            out[key] = node.set_circuit_bodies(tuple(bound))
        return out

    def _run_dry(
        self, batch: MetaCircuitBatch, param_sets: np.ndarray
    ) -> dict[object, MetaCircuit]:
        """Analytic path: emit shape-correct placeholders, no per-variant work.

        Fast-path emits empty QASM strings into ``bound_circuit_bodies``;
        slow-path emits shared DAG references into ``circuit_bodies``.  The
        slot choice mirrors the real path so downstream stages (dry-aware
        or not) see the attribute they expect populated.
        """
        n_param_sets = len(param_sets)
        out: dict[object, MetaCircuit] = {}
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
                out[key] = node.set_bound_bodies(bodies)
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
        }

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        return results
