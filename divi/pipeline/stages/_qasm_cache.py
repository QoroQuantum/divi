# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared DAG → parametric-QASM body cache for parameter-binding stages.

:class:`~divi.pipeline.stages.ParameterBindingStage` and
:class:`~divi.pipeline.stages.DataBindingStage` both serialise body DAGs to
parametric OpenQASM once and render values into the result. Serialisation is
the expensive step, so it is memoised here behind one bounded, thread-safe
cache shared by both stages.
"""

import threading
from collections import OrderedDict
from functools import lru_cache

from qiskit.dagcircuit import DAGCircuit

from divi.circuits import build_template, dag_to_qasm_body

# Bounded LRU memo for ``dag_to_qasm_body``. Keyed on ``id(dag)`` because
# ``DAGCircuit`` is unhashable; the tuple's DAG ref pins the object so
# ``id()`` cannot collide via GC.
_FAST_QASM_CACHE_MAXSIZE = 256
_FAST_QASM_CACHE: OrderedDict[tuple[int, int], tuple[DAGCircuit, str]] = OrderedDict()
_FAST_QASM_CACHE_LOCK = threading.Lock()


def _qasm_body_cached(dag: DAGCircuit, precision: int) -> str:
    key = (id(dag), precision)
    with _FAST_QASM_CACHE_LOCK:
        cached = _FAST_QASM_CACHE.get(key)
        if cached is not None and cached[0] is dag:
            _FAST_QASM_CACHE.move_to_end(key)
            return cached[1]
    # ``dag_to_qasm_body`` runs unlocked: it is CPU-heavy and idempotent, so
    # two threads racing on a cold key duplicate work but never corrupt state.
    body = dag_to_qasm_body(dag, precision=precision)
    with _FAST_QASM_CACHE_LOCK:
        _FAST_QASM_CACHE[key] = (dag, body)
        if len(_FAST_QASM_CACHE) > _FAST_QASM_CACHE_MAXSIZE:
            _FAST_QASM_CACHE.popitem(last=False)
    return body


# The body and symbol names are static across an optimizer loop, so the regex
# split in ``build_template`` is reused across stages and iterations.
_template_cached = lru_cache(maxsize=256)(build_template)
