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

from qiskit.dagcircuit import DAGCircuit

from divi.circuits import QASMTemplate, build_template, dag_to_qasm_body

# Bounded LRU memo for ``dag_to_qasm_body``. Keyed on ``id(dag)`` because
# ``DAGCircuit`` is unhashable; the tuple's DAG ref pins the object so
# ``id()`` cannot collide via GC.
_FAST_QASM_CACHE_MAXSIZE = 256
_FAST_QASM_CACHE: OrderedDict[tuple[int, int], tuple[DAGCircuit, str]] = OrderedDict()
_FAST_QASM_CACHE_LOCK = threading.Lock()

# Bounded LRU memo for ``build_template``. Keyed on the (body QASM, symbol
# names) pair — both hashable and the only inputs ``build_template`` depends
# on — so it is safe to reuse across stages and optimizer iterations.
_TEMPLATE_CACHE_MAXSIZE = 256
_TEMPLATE_CACHE: OrderedDict[tuple[str, tuple[str, ...]], QASMTemplate] = OrderedDict()
_TEMPLATE_CACHE_LOCK = threading.Lock()


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


def _template_cached(body_qasm: str, symbol_names: tuple[str, ...]) -> QASMTemplate:
    """Memoise :func:`~divi.circuits.build_template`.

    The parametric body, its symbol names, and precision are static across an
    optimizer loop, so the regex split that ``build_template`` performs is the
    same every cost evaluation. Both binding stages render values into a cached
    template instead of rebuilding it per iteration.
    """
    key = (body_qasm, symbol_names)
    with _TEMPLATE_CACHE_LOCK:
        cached = _TEMPLATE_CACHE.get(key)
        if cached is not None:
            _TEMPLATE_CACHE.move_to_end(key)
            return cached
    template = build_template(body_qasm, symbol_names)
    with _TEMPLATE_CACHE_LOCK:
        _TEMPLATE_CACHE[key] = template
        if len(_TEMPLATE_CACHE) > _TEMPLATE_CACHE_MAXSIZE:
            _TEMPLATE_CACHE.popitem(last=False)
    return template
