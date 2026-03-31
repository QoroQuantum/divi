# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage that applies Pauli twirling to circuit bodies.

Pauli twirling inserts random Pauli gates around each two-qubit Clifford
gate (CNOT, CZ) so that coherent errors are converted into stochastic
Pauli noise.  The ideal circuit is unchanged; only the noise channel is
affected.

During *expand*, each circuit body is replaced by ``num_twirls`` randomised
copies.  During *reduce*, the expectation values from all copies are
averaged to produce a single result per original circuit.
"""

import cirq
from mitiq.pt import generate_pauli_twirl_variants

from divi.circuits import MetaCircuit
from divi.circuits._qasm_conversion import (
    _cirq_circuit_from_qasm,
    normalize_qasm_after_cirq,
)
from divi.pipeline.abc import (
    BundleStage,
    ChildResults,
    ExpansionResult,
    MetaCircuitBatch,
    PipelineEnv,
    StageToken,
)
from divi.pipeline.transformations import group_by_base_key

TWIRL_AXIS = "twirl"


class PauliTwirlStage(BundleStage):
    """Fan out each circuit body into Pauli-twirled copies and average on reduce.

    Args:
        num_twirls: Number of randomised copies per circuit body.
    """

    @property
    def axis_name(self) -> str | None:
        return TWIRL_AXIS

    @property
    def stateful(self) -> bool:
        return False

    def __init__(self, num_twirls: int = 100) -> None:
        super().__init__(name=type(self).__name__)
        self._num_twirls = num_twirls

    def expand(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> tuple[ExpansionResult, StageToken]:
        out: dict[object, MetaCircuit] = {}

        for parent_key, meta in batch.items():
            updated_bodies: list[tuple] = []

            for tag, body in meta.circuit_body_qasms:
                cirq_circuit = _cirq_circuit_from_qasm(body, meta.symbols)
                variants = generate_pauli_twirl_variants(
                    cirq_circuit, num_circuits=self._num_twirls
                )
                for twirl_idx, variant in enumerate(variants):
                    twirl_tag = (*tag, (self.axis_name, twirl_idx))
                    twirl_body = normalize_qasm_after_cirq(cirq.qasm(variant))
                    updated_bodies.append((twirl_tag, twirl_body))

            symbol_names = tuple(str(s) for s in meta.symbols)
            out[parent_key] = meta.set_circuit_bodies(
                tuple(updated_bodies), symbol_names=symbol_names
            )

        return ExpansionResult(batch=out), None

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        grouped = group_by_base_key(results, self.axis_name, indexed=False)
        reduced: ChildResults = {}
        for base_key, values in grouped.items():
            if isinstance(values[0], dict):
                # Per-obs expval dicts — average each observable independently
                obs_keys = values[0].keys()
                reduced[base_key] = {
                    k: sum(v[k] for v in values) / len(values) for k in obs_keys
                }
            else:
                reduced[base_key] = sum(values) / len(values)
        return reduced
