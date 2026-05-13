# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Concurrency stress for the module-level ``_FAST_QASM_CACHE``."""

import threading

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from divi.pipeline.stages._parameter_binding_stage import (
    _FAST_QASM_CACHE,
    _FAST_QASM_CACHE_MAXSIZE,
    _qasm_body_cached,
)


def _build_distinct_dags(n: int):
    """``n`` distinct two-qubit DAGs; structurally different so QASM bodies differ."""
    dags = []
    for i in range(n):
        qc = QuantumCircuit(2)
        qc.rx(0.1 + i, 0)
        qc.ry(0.2 + i, 1)
        qc.cx(0, 1)
        dags.append(circuit_to_dag(qc))
    return dags


def test_fast_qasm_cache_is_thread_safe():
    """Concurrent ``_qasm_body_cached`` must not raise or return inconsistent
    bodies for the same (dag, precision) pair."""
    distinct_dags = _build_distinct_dags(64)
    shared_qc = QuantumCircuit(2)
    shared_qc.h(0)
    shared_qc.cx(0, 1)
    shared_dag = circuit_to_dag(shared_qc)

    # Record the first body returned for each DAG so concurrent callers can
    # assert equality against it.
    first_body: dict[int, str] = {}
    first_body_lock = threading.Lock()

    n_threads = 16
    n_iters = 200
    barrier = threading.Barrier(n_threads)
    errors: list[BaseException] = []

    def worker(thread_id: int) -> None:
        try:
            barrier.wait()
            for i in range(n_iters):
                dag = distinct_dags[(thread_id * 17 + i) % len(distinct_dags)]
                body = _qasm_body_cached(dag, precision=8)
                key = id(dag)
                with first_body_lock:
                    expected = first_body.setdefault(key, body)
                assert body == expected, "cache returned divergent bodies for same DAG"
                shared_body = _qasm_body_cached(shared_dag, precision=8)
                with first_body_lock:
                    expected_shared = first_body.setdefault(id(shared_dag), shared_body)
                assert shared_body == expected_shared
        except BaseException as exc:  # pragma: no cover - exercised on race
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive(), "thread hung beyond 30s budget"

    assert errors == [], f"concurrency raised {len(errors)} exceptions: {errors[:3]}"


def test_fast_qasm_cache_stays_bounded_under_eviction_pressure():
    """Hammering the cache past ``_FAST_QASM_CACHE_MAXSIZE`` from many threads
    must keep the LRU within one entry of its bound (and never crash on
    concurrent ``popitem``)."""
    # Twice the cache size so eviction is unavoidable.
    n_dags = _FAST_QASM_CACHE_MAXSIZE * 2
    distinct_dags = _build_distinct_dags(n_dags)

    n_threads = 8
    barrier = threading.Barrier(n_threads)
    errors: list[BaseException] = []

    def worker(thread_id: int) -> None:
        try:
            barrier.wait()
            for i, dag in enumerate(distinct_dags):
                # Stagger so threads don't iterate in lockstep.
                idx = (thread_id * 31 + i) % n_dags
                _qasm_body_cached(distinct_dags[idx], precision=8)
        except BaseException as exc:  # pragma: no cover - exercised on race
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive()

    assert errors == [], f"concurrency raised {len(errors)} exceptions: {errors[:3]}"
    # The cache may temporarily exceed MAXSIZE by one between the set and the
    # subsequent popitem inside the lock; never by more than that.
    assert len(_FAST_QASM_CACHE) <= _FAST_QASM_CACHE_MAXSIZE + 1
