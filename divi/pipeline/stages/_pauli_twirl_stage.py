# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Pipeline stage that applies Pauli twirling to DAG circuit bodies.

Pauli twirling inserts random Pauli gates around each two-qubit Clifford
gate (CNOT, CZ) so that coherent errors are converted into stochastic
Pauli noise.  The ideal circuit is unchanged up to a measurement-invariant
global phase; only the noise channel is affected.

During *expand*, each DAG body is replaced by ``n_twirls`` randomized
copies.  During *reduce*, the expectation values from all copies are
averaged to produce a single result per original circuit.
"""

import copy
import random
from typing import Any

from divi.circuits import MetaCircuit
from divi.circuits._qem_passes import _TWIRL_DAG_TABLES, PauliTwirlPass
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
    """Fan out each DAG body into Pauli-twirled copies and average on reduce.

    Args:
        n_twirls: Number of randomized copies per circuit body.
        seed: Optional seed for deterministic twirl sampling (useful in tests).
    """

    @property
    def axis_name(self) -> str | None:
        return TWIRL_AXIS

    @property
    def stateful(self) -> bool:
        return False

    def __init__(self, n_twirls: int = 100, seed: int | None = None) -> None:
        super().__init__(name=type(self).__name__)
        self._n_twirls = n_twirls
        self._seed = seed

    def expand(
        self, batch: MetaCircuitBatch, env: PipelineEnv
    ) -> tuple[ExpansionResult, StageToken]:
        out: dict[object, MetaCircuit] = {}

        for parent_key, meta in batch.items():
            updated_bodies: list[tuple] = []

            for tag, dag in meta.circuit_bodies:
                # Pre-identify twirl-eligible node IDs once and map each
                # to its gate's sub-DAG table.  deepcopy preserves node
                # IDs, so the substitution loop can skip name lookups
                # and gate-table indexing on every copy.
                twirl_id_to_table = {
                    n._node_id: _TWIRL_DAG_TABLES[n.op.name]
                    for n in dag.op_nodes()
                    if n.op.name in _TWIRL_DAG_TABLES
                }

                for twirl_idx in range(self._n_twirls):
                    rng = (
                        random.Random(self._seed + twirl_idx)
                        if self._seed is not None
                        else None
                    )
                    dag_copy = copy.deepcopy(dag)
                    twirl_specs = [
                        (n, twirl_id_to_table[n._node_id])
                        for n in dag_copy.op_nodes()
                        if n._node_id in twirl_id_to_table
                    ]
                    twirled_dag = PauliTwirlPass(rng=rng)._apply(dag_copy, twirl_specs)
                    twirl_tag = (*tag, (self.axis_name, twirl_idx))
                    updated_bodies.append((twirl_tag, twirled_dag))

            out[parent_key] = meta.set_circuit_bodies(tuple(updated_bodies))

        return ExpansionResult(batch=out), None

    def introspect(
        self, batch: MetaCircuitBatch, env: PipelineEnv, token: StageToken
    ) -> dict[str, Any]:
        return {"n_twirls": self._n_twirls}

    def reduce(
        self, results: ChildResults, env: PipelineEnv, token: StageToken
    ) -> ChildResults:
        grouped = group_by_base_key(results, self.axis_name, indexed=False)
        reduced: ChildResults = {}
        for base_key, values in grouped.items():
            if isinstance(values[0], dict):
                # Per-obs expval dicts — average each observable independently.
                obs_keys = values[0].keys()
                reduced[base_key] = {
                    k: sum(v[k] for v in values) / len(values) for k in obs_keys
                }
            else:
                reduced[base_key] = sum(values) / len(values)
        return reduced
