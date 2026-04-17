# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np

from divi.circuits import MetaCircuit, build_template, dag_to_qasm_body, render_template
from divi.pipeline.abc import (
    BundleStage,
    ChildResults,
    ExpansionResult,
    MetaCircuitBatch,
    PipelineEnv,
    StageToken,
)

PARAM_SET_AXIS = "param_set"


def _format_param(value: float, precision: int) -> str:
    """Format a numeric parameter for QASM insertion.

    Formats to *precision* decimal places, strips trailing zeros and dots,
    and normalises negative zero to ``"0"``.
    """
    s = f"{float(value):.{precision}f}".rstrip("0").rstrip(".")
    return "0" if s in {"-0", ""} else s


class ParameterBindingStage(BundleStage):
    """Bind ``env.param_sets`` into every circuit body via the template fast path.

    For each parametric ``MetaCircuit`` in the batch, this stage:

    1. Serialises every DAG body to a body-only parametric QASM string
       via :func:`~divi.circuits._conversions.dag_to_qasm_body` —
       one dump per body variant, amortised over ``env.param_sets``.
    2. Wraps each body in a :class:`~divi.circuits.QASMTemplate` for
       O(body length) string-substitution rendering.
    3. Produces one bound body string per ``(body variant × param set)``
       and stores them on ``meta.bound_circuit_bodies`` for
       :mod:`~divi.pipeline._compilation` to concatenate with the
       pre-serialised measurement QASMs.
    """

    @property
    def axis_name(self) -> str | None:
        return PARAM_SET_AXIS

    @property
    def stateful(self) -> bool:
        return True

    def __init__(self) -> None:
        super().__init__(name=type(self).__name__)

    def expand(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> tuple[ExpansionResult, StageToken]:
        out: dict[object, MetaCircuit] = {}
        param_sets = np.asarray(env.param_sets, dtype=float)

        if param_sets.ndim != 2:
            raise ValueError("ParameterBindingStage expects env.param_sets to be 2D.")

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
                    _format_param(v, precision) for v in param_values
                )
                param_set_tag = (PARAM_SET_AXIS, param_set_idx)
                for qasm_tag, template in templates:
                    bound_tag = (*qasm_tag, param_set_tag)
                    bound_bodies.append(
                        (bound_tag, render_template(template, formatted_values))
                    )

            out[key] = node.set_bound_bodies(tuple(bound_bodies))

        return ExpansionResult(batch=out), None

    def introspect(
        self, batch: MetaCircuitBatch, env: PipelineEnv, token: StageToken
    ) -> dict[str, Any]:
        param_sets = np.asarray(env.param_sets, dtype=float)
        meta = next(iter(batch.values()), None)
        n_params = len(meta.parameters) if meta else 0
        return {"n_param_sets": len(param_sets), "n_params": n_params}

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        return results
