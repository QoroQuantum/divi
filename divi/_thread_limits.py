# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Process-wide single-threaded sections for numerics too small to thread."""

import os
import sys
import threading
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from threadpoolctl import ThreadpoolController

# Visual C++'s OpenMP runtime shares one thread setting across the process.
_PROCESS_WIDE_OPENMP = sys.platform == "win32"


def _limits_calling_thread_only(lib: Any) -> bool:
    """Whether threadpoolctl's limit for ``lib`` applies to the calling thread only."""
    if lib.internal_api == "mkl":
        return True
    if _PROCESS_WIDE_OPENMP:
        return False
    return lib.internal_api == "openmp" or (
        lib.internal_api == "openblas" and lib.threading_layer == "openmp"
    )


def _split_by_limit_scope(
    controller: ThreadpoolController,
) -> tuple[ThreadpoolController, ThreadpoolController]:
    per_thread = [
        lib.filepath
        for lib in controller.lib_controllers
        if _limits_calling_thread_only(lib)
    ]
    process_wide = [
        lib.filepath
        for lib in controller.lib_controllers
        if lib.filepath not in per_thread
    ]
    return controller.select(filepath=per_thread), controller.select(
        filepath=process_wide
    )


class _ThreadHolds(threading.local):
    def __init__(self):
        self.filepaths: list[str] = []


# Holders of each process-wide library across every section, and the thread
# count it had before the first of them entered.
_lock = threading.Lock()
_holders: dict[str, int] = {}
_originals: dict[str, tuple[Any, int]] = {}
_thread_holds = _ThreadHolds()


def _restore(filepath: str) -> None:
    del _holders[filepath]
    lib, num_threads = _originals.pop(filepath)
    lib.set_num_threads(num_threads)


def _reset_in_child() -> None:
    """Keep only the forking thread's holds; the other threads are gone."""
    global _lock
    _lock = threading.Lock()
    held = Counter(_thread_holds.filepaths)
    for filepath in list(_holders):
        if held[filepath]:
            _holders[filepath] = held[filepath]
        else:
            _restore(filepath)


if sys.platform != "win32":
    os.register_at_fork(after_in_child=_reset_in_child)


class SingleThreadedSection:
    """Holds native thread pools at one thread while any thread is inside.

    Process-wide libraries are shared by every section: the first holder of a
    library sets its limit and the last to leave restores it. Each holder also
    limits per-thread libraries in its own thread. A forked child keeps a
    limit only if its thread was inside.

    Args:
        user_api: ``"blas"`` or ``"openmp"`` to limit one kind of library;
            ``None`` limits both.
    """

    def __init__(self, user_api: str | None = None):
        self._user_api = user_api

    def _discover(self) -> tuple[ThreadpoolController, ThreadpoolController]:
        controller = ThreadpoolController()
        if self._user_api is not None:
            controller = controller.select(user_api=self._user_api)
        return _split_by_limit_scope(controller)

    @contextmanager
    def hold(self) -> Iterator[None]:
        """Run the body with every selected thread pool at one thread."""
        # Discovered per hold, so libraries loaded since the last one count.
        per_thread, process_wide = self._discover()
        filepaths = [lib.filepath for lib in process_wide.lib_controllers]
        with _lock:
            for lib, filepath in zip(process_wide.lib_controllers, filepaths):
                if filepath not in _holders:
                    _holders[filepath] = 0
                    _originals[filepath] = (lib, lib.get_num_threads())
                    lib.set_num_threads(1)
                _holders[filepath] += 1
        _thread_holds.filepaths.extend(filepaths)
        try:
            with per_thread.limit(limits=1):
                yield
        finally:
            for filepath in filepaths:
                _thread_holds.filepaths.remove(filepath)
            with _lock:
                for filepath in filepaths:
                    _holders[filepath] -= 1
                    if _holders[filepath] == 0:
                        _restore(filepath)
