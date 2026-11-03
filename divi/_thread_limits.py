# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Process-wide single-threaded sections for numerics too small to thread."""

import os
import sys
import threading
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


class _ThreadDepth(threading.local):
    value = 0


class SingleThreadedSection:
    """Holds native thread pools at one thread while any thread is inside.

    Overlapping holders share the limit on process-wide libraries: the first
    to enter sets it and the last to leave restores it. Each also limits
    per-thread libraries in its own thread. A forked child keeps the limit
    only if its thread was inside.

    Args:
        user_api: ``"blas"`` or ``"openmp"`` to limit one kind of library;
            ``None`` limits both.
    """

    def __init__(self, user_api: str | None = None):
        self._user_api = user_api
        self._lock = threading.Lock()
        self._users = 0
        self._depth = _ThreadDepth()
        self._per_thread: ThreadpoolController | None = None
        self._process_wide_limit: Any = None
        if sys.platform != "win32":
            os.register_at_fork(after_in_child=self._reset_in_child)

    def _discover(self) -> tuple[ThreadpoolController, ThreadpoolController]:
        controller = ThreadpoolController()
        if self._user_api is not None:
            controller = controller.select(user_api=self._user_api)
        return _split_by_limit_scope(controller)

    def _reset_in_child(self) -> None:
        self._lock = threading.Lock()
        self._users = self._depth.value
        if self._users == 0 and self._process_wide_limit is not None:
            limit, self._process_wide_limit = self._process_wide_limit, None
            limit.restore_original_limits()

    @contextmanager
    def hold(self) -> Iterator[None]:
        """Run the body with every selected thread pool at one thread."""
        with self._lock:
            per_thread = self._per_thread
            if self._users == 0 or per_thread is None:
                # Rediscovered per burst, so libraries loaded since the last one count.
                per_thread, process_wide = self._discover()
                self._process_wide_limit = process_wide.limit(limits=1)
                self._per_thread = per_thread
            self._users += 1
        self._depth.value += 1
        try:
            with per_thread.limit(limits=1):
                yield
        finally:
            self._depth.value -= 1
            with self._lock:
                self._users -= 1
                if self._users == 0:
                    limit, self._process_wide_limit = self._process_wide_limit, None
                    limit.restore_original_limits()
