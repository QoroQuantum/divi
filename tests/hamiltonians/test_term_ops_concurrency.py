# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Concurrency stress for the module-level ``_PL_TO_SPO_CACHE``."""

import threading

import pennylane as qp
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians._term_ops import _from_spo, _spo_wires, _to_spo


def _build_distinct_spos(n: int) -> list[SparsePauliOp]:
    """``n`` distinct two-qubit SPOs, each with a unique coefficient."""
    return [
        SparsePauliOp.from_list([("IZ", 1.0 + i), ("ZI", 2.0 + i), ("ZZ", 3.0 + i)])
        for i in range(n)
    ]


def test_pl_to_spo_cache_is_thread_safe():
    """Concurrent ``_from_spo`` / ``_to_spo`` / ``_spo_wires`` must not raise
    or return inconsistent values."""
    distinct_spos = _build_distinct_spos(64)
    shared_spo = SparsePauliOp.from_list([("XX", 1.0), ("YY", 2.0)])

    n_threads = 16
    n_iters = 200
    barrier = threading.Barrier(n_threads)
    errors: list[BaseException] = []

    # Captured PL outputs for distinct SPOs, populated by writer threads.
    pl_for: dict[int, qp.operation.Operator] = {}
    pl_for_lock = threading.Lock()

    def writer(thread_id: int) -> None:
        try:
            barrier.wait()
            for i in range(n_iters):
                spo = distinct_spos[(thread_id * 17 + i) % len(distinct_spos)]
                pl = _from_spo(spo, range(spo.num_qubits))
                with pl_for_lock:
                    pl_for.setdefault(id(spo), pl)
                # Also exercise the shared SPO frequently.
                _from_spo(shared_spo, range(shared_spo.num_qubits))
        except BaseException as exc:  # pragma: no cover - exercised on race
            errors.append(exc)

    def reader() -> None:
        try:
            barrier.wait()
            for _ in range(n_iters):
                with pl_for_lock:
                    snapshot = list(pl_for.values())
                for pl in snapshot:
                    _to_spo(pl)
                    _spo_wires(pl)
        except BaseException as exc:  # pragma: no cover - exercised on race
            errors.append(exc)

    threads = [
        threading.Thread(target=writer, args=(i,)) for i in range(n_threads // 2)
    ] + [threading.Thread(target=reader) for _ in range(n_threads // 2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive(), "thread hung beyond 30s budget"

    assert errors == [], f"concurrency raised {len(errors)} exceptions: {errors[:3]}"


def test_to_spo_returns_cached_spo_after_from_spo_under_contention():
    """Once ``_from_spo`` has run for a given PL output, ``_to_spo`` on that
    PL must return the original SPO regardless of concurrent callers."""
    spos = _build_distinct_spos(32)
    pls = [_from_spo(s, range(s.num_qubits)) for s in spos]

    n_threads = 8
    n_iters = 500
    barrier = threading.Barrier(n_threads)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            barrier.wait()
            for i in range(n_iters):
                pl = pls[i % len(pls)]
                expected = spos[i % len(spos)]
                actual = _to_spo(pl)
                assert actual is expected
        except BaseException as exc:  # pragma: no cover - exercised on race
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == [], f"identity check failed: {errors[:3]}"
