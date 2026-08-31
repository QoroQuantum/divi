# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared program stubs for ``ProgramEnsemble`` tests."""

from divi.qprog.quantum_program import QuantumProgram
from divi.reporting._events import ProgressEvent
from divi.reporting._session import ProgressSession
from divi.reporting._state import ProgressState


class RecordingProgressSession(ProgressSession):
    """Real direct session that also records its typed input events."""

    def __init__(self, state: ProgressState) -> None:
        super().__init__(state, lambda _state, _affected: None, lambda: None)
        self.emitted: list[ProgressEvent] = []

    def emit(self, event: ProgressEvent) -> None:
        self.emitted.append(event)
        super().emit(event)


class _FakeRunResult:
    """Mimics a program instance returned by run() for mocked futures."""

    def __init__(self, circuit_count: int, run_time: float):
        self._total_circuit_count = circuit_count
        self._total_run_time = run_time


class _StubProgram(QuantumProgram):
    """Base for ProgramEnsemble test programs.

    Stubs the abstract methods and tracks a ``_ran`` flag; subclasses
    implement only ``run()`` (and any extra constructor args they need).
    """

    def __init__(self, *, backend, **kwargs):
        super().__init__(backend=backend, **kwargs)
        self._ran = False

    def has_results(self) -> bool:
        return self._ran

    def _generate_circuits(self, **kwargs):
        return []

    def _post_process_results(self, results):
        pass


class SimpleTestProgram(_StubProgram):
    """A simple mock program whose ``run()`` assigns preset counter values."""

    def __init__(self, circ_count: int, run_time: float, *, backend, **kwargs):
        super().__init__(backend=backend, **kwargs)
        self.circ_count = circ_count
        self.run_time = run_time

    def run(self):
        self._total_circuit_count = self.circ_count
        self._total_run_time = self.run_time
        self._ran = True
        return self


class FailingTestProgram(_StubProgram):
    """A mock program whose ``run()`` always raises."""

    def __init__(self, message: str = "program boom", *, backend, **kwargs):
        super().__init__(backend=backend, **kwargs)
        self.message = message

    def run(self):
        raise RuntimeError(self.message)
