# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np

from divi.circuits import MetaCircuit
from divi.circuits._qasm_template import render_template
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
    """BundleStage that binds env parameters into all circuit-body QASMs."""

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
            # Non-parametric circuits pass through unchanged.
            if len(node.symbols) == 0:
                out[key] = node
                continue

            precision = node.precision
            templates = node.circuit_body_templates
            n_symbols = len(node.symbols)

            bound_bodies: list[tuple[tuple[tuple[str, str], ...], str]] = []

            # For each param set, bind concrete values into every body variant.
            for param_set_idx, param_values in enumerate(param_sets):
                if len(param_values) != n_symbols:
                    raise ValueError(
                        f"ParameterBindingStage expected {n_symbols} parameters "
                        f"for key '{key}', got {len(param_values)} in param set "
                        f"{param_set_idx}."
                    )

                formatted_values = tuple(
                    _format_param(v, precision) for v in param_values
                )

                param_set_tag = (PARAM_SET_AXIS, param_set_idx)

                # Produce one bound body per (existing body variant × param set),
                # appending the param_set axis label to the QASM tag tuple.
                for qasm_tag, template in templates:
                    bound_tag = (*qasm_tag, param_set_tag)
                    bound_bodies.append(
                        (
                            bound_tag,
                            render_template(template, formatted_values),
                        )
                    )

            out[key] = node.set_circuit_bodies(tuple(bound_bodies))

        return ExpansionResult(batch=out), None

    def introspect(
        self, batch: MetaCircuitBatch, env: PipelineEnv, token: StageToken
    ) -> dict[str, Any]:
        param_sets = np.asarray(env.param_sets, dtype=float)
        meta = next(iter(batch.values()), None)
        n_params = len(meta.symbols) if meta else 0
        return {"n_param_sets": len(param_sets), "n_params": n_params}

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        return results
