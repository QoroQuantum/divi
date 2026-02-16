# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import re

import numpy as np

from divi.circuits import MetaCircuit
from divi.pipeline.abc import (
    BundleStage,
    ChildResults,
    ExpansionResult,
    MetaCircuitBatch,
    PipelineEnv,
    StageToken,
)

PARAM_SET_AXIS = "param_set"


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

            # Build a single regex that matches any symbol name in the QASM body,
            # so we can substitute all symbols in one pass per body string.
            bound_bodies: list[tuple[tuple[tuple[str, str], ...], str]] = []
            symbol_names = tuple(re.escape(str(symbol)) for symbol in node.symbols)
            pattern = re.compile("|".join(symbol_names))

            # For each param set, bind concrete values into every body variant.
            for param_set_idx, param_values in enumerate(param_sets):
                if len(param_values) != len(symbol_names):
                    raise ValueError(
                        f"ParameterBindingStage expected {len(symbol_names)} parameters "
                        f"for key '{key}', got {len(param_values)} in param set "
                        f"{param_set_idx}."
                    )

                # Format each value to the circuit's declared precision,
                # stripping trailing zeros (e.g. 1.5000 → "1.5", -0.0 → "0").
                formatted_values = []
                for value in param_values:
                    formatted = f"{float(value):.{precision}f}".rstrip("0").rstrip(".")
                    formatted_values.append(
                        "0" if formatted in {"-0", ""} else formatted
                    )
                mapping = dict(
                    zip(
                        symbol_names,
                        formatted_values,
                    )
                )
                param_set_tag = (PARAM_SET_AXIS, param_set_idx)

                # Produce one bound body per (existing body variant × param set),
                # appending the param_set axis label to the QASM tag tuple.
                for qasm_tag, qasm_body in node.circuit_body_qasms:
                    bound_tag = (*qasm_tag, param_set_tag)
                    bound_bodies.append(
                        (
                            bound_tag,
                            pattern.sub(lambda m: mapping[m.group(0)], qasm_body),
                        )
                    )

            out[key] = node.set_circuit_bodies(tuple(bound_bodies))

        return ExpansionResult(batch=out), None

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        return results
