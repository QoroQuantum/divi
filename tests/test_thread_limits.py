# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import pytest
from threadpoolctl import ThreadpoolController, threadpool_info, threadpool_limits

from divi._thread_limits import SingleThreadedSection

_WAIT = 30


def _wait(event):
    if not event.wait(timeout=_WAIT):
        raise TimeoutError("the other holder never reached its checkpoint")


def _blas_threads():
    return {
        lib["num_threads"] for lib in threadpool_info() if lib["user_api"] == "blas"
    }


def _in_forked_child(fn):
    """Run ``fn`` in a forked child and return the ``repr`` of its result."""
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            payload = repr(fn())
        except BaseException as exc:
            payload = f"child raised {exc!r}"
        os.write(write_fd, payload.encode())
        os._exit(0)
    os.close(write_fd)
    with os.fdopen(read_fd) as reader:
        payload = reader.read()
    os.waitpid(pid, 0)
    return payload


class _PerThreadLimit(threading.local):
    num_threads = 4


class _FakeLibrary:
    """A thread pool whose limit is per calling thread or process-wide,
    reporting no threading layer, as FlexiBLAS and OpenMP runtimes do."""

    prefix = "libfake"

    def __init__(self, user_api, internal_api, *, per_thread):
        self.user_api = user_api
        self.internal_api = internal_api
        self.filepath = f"/fake/{internal_api}/{id(self)}"
        self.per_thread = per_thread
        self._shared = 4
        self._local = _PerThreadLimit()

    def get_num_threads(self):
        return self._local.num_threads if self.per_thread else self._shared

    def set_num_threads(self, num_threads):
        if self.per_thread:
            self._local.num_threads = num_threads
        else:
            self._shared = num_threads

    def info(self, debugging_info=False):
        return {
            "user_api": self.user_api,
            "internal_api": self.internal_api,
            "prefix": self.prefix,
            "filepath": self.filepath,
            "num_threads": self.get_num_threads(),
        }


class _FakeLayeredBlas(_FakeLibrary):
    """A fake BLAS library that reports its threading layer."""

    def __init__(self, internal_api, threading_layer, *, per_thread):
        super().__init__("blas", internal_api, per_thread=per_thread)
        self.threading_layer = threading_layer


def _fake_libraries(process_wide_openmp):
    return [
        _FakeLayeredBlas("mkl", "intel", per_thread=True),
        _FakeLayeredBlas("mkl", "tbb", per_thread=True),
        _FakeLayeredBlas("openblas", "openmp", per_thread=not process_wide_openmp),
        _FakeLayeredBlas("openblas", "pthreads", per_thread=False),
        _FakeLayeredBlas("openblas", "unknown", per_thread=False),
        _FakeLayeredBlas("blis", "openmp", per_thread=False),
        _FakeLibrary("blas", "flexiblas", per_thread=False),
        _FakeLibrary("openmp", "openmp", per_thread=not process_wide_openmp),
    ]


def _install_fake_libraries(mocker, process_wide_openmp):
    mocker.patch("divi._thread_limits._PROCESS_WIDE_OPENMP", process_wide_openmp)
    libs = _fake_libraries(process_wide_openmp)
    controller = ThreadpoolController()
    controller.lib_controllers = libs
    mocker.patch("divi._thread_limits.ThreadpoolController", return_value=controller)
    return libs


def _overlapping_holds(first_section, second_section, libs):
    """Thread counts while each section holds, from a third thread while both
    do, and after the first and then the second leaves."""
    entered = [threading.Event(), threading.Event()]
    release = [threading.Event(), threading.Event()]
    inside = {}

    def threads_now():
        return [lib.get_num_threads() for lib in libs]

    def hold(index, section):
        with section.hold():
            inside[index] = threads_now()
            entered[index].set()
            _wait(release[index])
        return threads_now()

    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            first = pool.submit(hold, 0, first_section)
            _wait(entered[0])
            second = pool.submit(hold, 1, second_section)
            _wait(entered[1])
            outside = threads_now()
            release[0].set()
            after_first = first.result(timeout=_WAIT)
            release[1].set()
            after_second = second.result(timeout=_WAIT)
        finally:
            for event in release:
                event.set()
    return inside, outside, after_first, after_second, threads_now()


class TestSingleThreadedSection:
    @pytest.mark.parametrize("user_api", [None, "blas"], ids=["all", "blas"])
    @pytest.mark.parametrize(
        "process_wide_openmp", [False, True], ids=["gnu-or-llvm-openmp", "msvc-openmp"]
    )
    def test_each_library_is_limited_in_the_scope_its_api_has(
        self, user_api, process_wide_openmp, mocker
    ):
        """Per-thread libraries are limited and restored in each holding
        thread; process-wide ones once for all overlapping holders, so the
        first to leave does not lift the limit from the others."""
        libs = _install_fake_libraries(mocker, process_wide_openmp)
        section = SingleThreadedSection(user_api)

        inside, outside, after_first, after_second, final = _overlapping_holds(
            section, section, libs
        )

        selected = [user_api in (None, lib.user_api) for lib in libs]
        assert inside[0] == inside[1] == [1 if s else 4 for s in selected]
        assert outside == [
            1 if s and not lib.per_thread else 4 for s, lib in zip(selected, libs)
        ]
        assert after_first == outside
        assert after_second == final == [4] * len(libs)

    def test_separate_sections_share_each_process_wide_limit(self, mocker):
        """A section entering while another holds must not save the limited
        count as the original, or the process keeps one thread after both."""
        libs = _install_fake_libraries(mocker, process_wide_openmp=False)

        _inside, outside, after_first, after_second, final = _overlapping_holds(
            SingleThreadedSection("blas"), SingleThreadedSection(), libs
        )

        assert outside == [4 if lib.per_thread else 1 for lib in libs]
        assert after_first == outside
        assert after_second == final == [4] * len(libs)

    @pytest.mark.skipif(sys.platform == "win32", reason="needs os.fork")
    @pytest.mark.parametrize("forked_inside", [False, True], ids=["outside", "inside"])
    def test_a_forked_child_stays_limited_only_if_its_thread_was_inside(
        self, forked_inside
    ):
        """Another thread's hold is gone in the child, so its limit must not
        outlive it there."""
        section = SingleThreadedSection("blas")
        entered, release = threading.Event(), threading.Event()

        def hold_until_released():
            with section.hold():
                entered.set()
                _wait(release)

        def hold_and_report():
            start = _blas_threads()
            with section.hold():
                inside = _blas_threads()
            return start, inside, _blas_threads()

        def fork():
            # Python warns about forking a multi-threaded process from 3.12.
            with (
                pytest.warns(DeprecationWarning, match="multi-threaded")
                if sys.version_info >= (3, 12)
                else nullcontext()
            ):
                return _in_forked_child(hold_and_report)

        with threadpool_limits(limits=2, user_api="blas"):
            before = _blas_threads()
            with ThreadPoolExecutor(max_workers=1) as pool:
                try:
                    running = pool.submit(hold_until_released)
                    _wait(entered)
                    if forked_inside:
                        with section.hold():
                            child = fork()
                    else:
                        child = fork()
                finally:
                    release.set()
                running.result(timeout=_WAIT)

        expected = {1} if forked_inside else before
        assert child == repr((expected, {1}, expected))
