# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from typing import Any

import numpy as np
from qiskit.converters import circuit_to_dag, dag_to_circuit

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
        param_sets = np.asarray(env.param_sets, dtype=float)

        if param_sets.ndim != 2:
            raise ValueError("ParameterBindingStage expects env.param_sets to be 2D.")

        if self._fast_path:
            out = self._expand_fast(batch, param_sets)
        else:
            out = self._expand_slow(batch, param_sets)
        return ExpansionResult(batch=out), None

    def _expand_fast(
        self,
        batch: MetaCircuitBatch,
        param_sets: np.ndarray,
    ) -> dict[object, MetaCircuit]:
        """Render bound QASM strings into ``meta.bound_circuit_bodies``."""
        out: dict[object, MetaCircuit] = {}

        for key, node in batch.items():
            # Non-parametric circuits: serialise each body once, no binding loop.
            if len(node.parameters) == 0:
                bodies = tuple(
                    (tag, dag_to_qasm_body(dag, precision=node.precision))
                    for tag, dag in node.circuit_bodies
                )
                out[key] = node.set_bound_bodies(bodies)
                continue

            precision = node.precision
            n_params = len(node.parameters)
            param_names = tuple(p.name for p in node.parameters)

            # Build one template per body variant — once, outside the
            # per-param-set loop.  ~0.25 ms per variant; amortised across
            # ``len(param_sets)`` binds.
            templates = tuple(
                (
                    tag,
                    build_template(
                        dag_to_qasm_body(dag, precision=precision),
                        param_names,
                    ),
                )
                for tag, dag in node.circuit_bodies
            )

            bound_bodies: list[tuple[tuple[tuple[str, object], ...], str]] = []
            for param_set_idx, param_values in enumerate(param_sets):
                if len(param_values) != n_params:
                    raise ValueError(
                        f"ParameterBindingStage expected {n_params} parameters "
                        f"for key '{key}', got {len(param_values)} in param set "
                        f"{param_set_idx}."
                    )
                formatted_values = tuple(
                    _format_bound_param(v, precision) for v in param_values
                )
                param_set_tag = (PARAM_SET_AXIS, param_set_idx)
                for qasm_tag, template in templates:
                    bound_tag = (*qasm_tag, param_set_tag)
                    bound_bodies.append(
                        (bound_tag, render_template(template, formatted_values))
                    )

            out[key] = node.set_bound_bodies(tuple(bound_bodies))

        return out

    def _expand_slow(
        self,
        batch: MetaCircuitBatch,
        param_sets: np.ndarray,
    ) -> dict[object, MetaCircuit]:
        """Emit bound DAGs by replacing ``meta.circuit_bodies``.

        Uses ``QuantumCircuit.assign_parameters(inplace=False)`` — the
        public Qiskit API that correctly handles ParameterExpressions,
        composite gates, and nested parameters.  Downstream body-consuming
        stages read the replaced ``circuit_bodies`` directly.
        """
        out: dict[object, MetaCircuit] = {}

        for key, node in batch.items():
            if len(node.parameters) == 0:
                # No parameters to bind; nothing to rewrite.
                out[key] = node
                continue

            n_params = len(node.parameters)
            parameters = node.parameters

            # Convert each body DAG to a QuantumCircuit once, outside the
            # per-param-set loop.  ``assign_parameters`` on a QC is the
            # cheapest public Qiskit path.
            qc_templates = tuple(
                (tag, dag_to_circuit(dag)) for tag, dag in node.circuit_bodies
            )

            bound_bodies: list = []
            for param_set_idx, param_values in enumerate(param_sets):
                if len(param_values) != n_params:
                    raise ValueError(
                        f"ParameterBindingStage expected {n_params} parameters "
                        f"for key '{key}', got {len(param_values)} in param set "
                        f"{param_set_idx}."
                    )
                binding = dict(zip(parameters, param_values))
                param_set_tag = (PARAM_SET_AXIS, param_set_idx)
                for qasm_tag, qc in qc_templates:
                    bound_qc = qc.assign_parameters(binding, inplace=False)
                    bound_dag = circuit_to_dag(bound_qc)
                    bound_tag = (*qasm_tag, param_set_tag)
                    bound_bodies.append((bound_tag, bound_dag))

            out[key] = node.set_circuit_bodies(tuple(bound_bodies))

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
