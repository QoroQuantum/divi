# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared across the whole test suite."""

import itertools
import re
from threading import Event

import numpy as np
import pytest
from qiskit import QuantumCircuit

from divi.backends import AsyncJobBackend, ExecutionResult, JobStatus
from divi.circuits._payloads import flatten_bound_payloads
from divi.exceptions import ExecutionCancelledError
from divi.qprog.initial_states import (
    CustomPerQubitState,
    DickeState,
    InitialState,
    OnesState,
    SuperpositionState,
    WState,
    ZerosState,
)


class FakeAsyncBackend(AsyncJobBackend):
    """Async backend recording how it is driven; ``poll`` scripts its polling.

    ``poll(backend, execution_result, progress_callback, cancellation_event)``
    replaces the default of reporting ``RUNNING`` once and completing.
    """

    def __init__(self, *, poll=None, max_retries=None, cancel_error=None, run_time=2.5):
        super().__init__(shots=100, polling_interval=0.0, max_retries=max_retries)
        self._poll = poll
        self._cancel_error = cancel_error
        self._run_time = run_time
        self._job_ids = itertools.count(1)
        self._circuits: dict[str, dict[str, str]] = {}
        self.jobs: list[ExecutionResult] = []
        self.poll_calls: list[dict] = []
        self.fetched: list[ExecutionResult] = []
        self.cancelled_jobs: list[ExecutionResult] = []
        self.polling = Event()

    @property
    def supports_expval(self) -> bool:
        return False

    def submit_circuits(self, payloads, **kwargs) -> ExecutionResult:
        job = ExecutionResult(job_id=f"job-{next(self._job_ids)}")
        self._circuits[job.job_id] = flatten_bound_payloads(payloads)
        self.jobs.append(job)
        return job

    def poll_job_status(
        self,
        execution_result,
        loop_until_complete=False,
        verbose=True,
        progress_callback=None,
        cancellation_event=None,
    ):
        self.poll_calls.append(
            {
                "loop_until_complete": loop_until_complete,
                "verbose": verbose,
                "cancellation_event": cancellation_event,
            }
        )
        self.polling.set()
        if self._poll is not None:
            return self._poll(
                self, execution_result, progress_callback, cancellation_event
            )
        if progress_callback is not None:
            progress_callback(1, JobStatus.RUNNING.value)
        return JobStatus.COMPLETED

    def get_job_results(self, execution_result) -> ExecutionResult:
        self.fetched.append(execution_result)
        return ExecutionResult(
            results=[
                {"label": label, "results": {"00": 100}}
                for label in self._circuits[execution_result.job_id]
            ],
            run_time=self._run_time,
        )

    def cancel_job(self, execution_result):
        self.cancelled_jobs.append(execution_result)
        if self._cancel_error is not None:
            raise self._cancel_error


def block_until_cancelled(
    backend, execution_result, progress_callback, cancellation_event
):
    """Polling script: wait for the cancellation event, then raise."""
    if not cancellation_event.wait(timeout=10):
        raise TimeoutError("The cancellation event was never set.")
    raise ExecutionCancelledError("Polling interrupted.")


def raise_after_running(error_type):
    """Polling script: report ``RUNNING``, then raise ``error_type(job_id)``."""

    def poll(backend, execution_result, progress_callback, cancellation_event):
        if progress_callback is not None:
            progress_callback(1, JobStatus.RUNNING.value)
        raise error_type(execution_result.job_id)

    return poll


def report_status(status: str, attempt: int):
    """Polling script: report ``status`` at ``attempt``, then complete."""

    def poll(backend, execution_result, progress_callback, cancellation_event):
        if progress_callback is not None:
            progress_callback(attempt, status)
        return JobStatus.COMPLETED

    return poll


def exact_match(message: str) -> str:
    """``pytest.raises(match=...)`` pattern matching ``message`` and nothing else."""
    return f"^{re.escape(message)}$"


#: Water at its equilibrium geometry, in Ångström.
WATER_ATOM = "O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692"


def run_mean_field(atom: str, basis: str = "sto-3g", spin: int = 0):
    """A converged PySCF RHF mean field, ROHF when ``spin`` is non-zero; skips
    the test when PySCF is not installed."""
    gto = pytest.importorskip("pyscf.gto")
    scf = pytest.importorskip("pyscf.scf")
    molecule = gto.M(atom=atom, basis=basis, spin=spin, verbose=0)
    return (scf.ROHF if spin else scf.RHF)(molecule).run(verbose=0)


H2_ATOM = "H 0 0 0; H 0 0 0.74"
H3_CHAIN_ATOM = "H 0 0 0; H 0 0 0.9; H 0 0 1.8"


def run_mcscf(
    method,
    mean_field,
    n_orbitals,
    n_electrons,
    fcisolver=None,
    natorb=False,
    mo_coeff=None,
):
    """Run a quiet PySCF ``method``, with ``fcisolver`` replacing its FCI solver,
    from ``mo_coeff`` when given."""
    mc = method(mean_field, n_orbitals, n_electrons)
    mc.verbose = 0
    mc.natorb = natorb
    if fcisolver is not None:
        mc.fcisolver = fcisolver
    mc.kernel(mo_coeff)
    return mc


def build_degenerate_orbital_integrals(exchange, coulomb=1.0):
    """Two degenerate, non-hopping orbitals with an on-site Coulomb repulsion
    and an inter-orbital exchange term that, for ``exchange > 0``, puts the
    Sz=0 triplet component below the singlets -- a Hund's-rule-like case."""
    n_orb = 2
    one_body = np.zeros((n_orb, n_orb))
    two_body = np.zeros((n_orb,) * 4)
    two_body[0, 0, 0, 0] = coulomb
    two_body[1, 1, 1, 1] = coulomb
    two_body[0, 1, 1, 0] = exchange
    two_body[1, 0, 0, 1] = exchange
    return one_body, two_body, n_orb


def unitary(simulator, circuit: QuantumCircuit) -> np.ndarray:
    """``circuit``'s unitary, column ``j`` its output state on basis input ``j``,
    with qubit 0 the least-significant bit."""
    n = circuit.num_qubits
    columns = []
    for index in range(2**n):
        prepared = circuit.copy_empty_like()
        prepared.global_phase = 0
        for qubit in range(n):
            if index >> qubit & 1:
                prepared.x(qubit)
        columns.append(simulator.statevector(prepared.compose(circuit)))
    return np.column_stack(columns)


def same_unitary(
    simulator, a: QuantumCircuit, b: QuantumCircuit, atol: float = 1e-8
) -> bool:
    """Whether circuits ``a`` and ``b`` implement one unitary, up to a global
    phase."""
    return equal_up_to_phase(unitary(simulator, a), unitary(simulator, b), atol)


_QUBIT_VECTORS = {
    "0": np.array([1.0, 0.0]),
    "1": np.array([0.0, 1.0]),
    "+": np.array([1.0, 1.0]) / np.sqrt(2),
    "-": np.array([1.0, -1.0]) / np.sqrt(2),
}


def initial_state_amplitudes(state: InitialState, n_qubits: int) -> np.ndarray:
    """Amplitudes of ``state`` from its definition, not its circuit, with
    qubit 0 the least-significant bit. Dicke states span every wire."""
    uniform_chars = {ZerosState: "0", OnesState: "1", SuperpositionState: "+"}
    chars = (
        state.state_string
        if isinstance(state, CustomPerQubitState)
        else uniform_chars.get(type(state), "") * n_qubits
    )
    if chars:
        vector = np.ones(1)
        for char in chars:
            vector = np.kron(_QUBIT_VECTORS[char], vector)
        return vector
    indices = np.arange(2**n_qubits)
    if isinstance(state, DickeState) and state.n_qubits in (None, n_qubits):
        support = np.array(
            [int(i).bit_count() == state.hamming_weight for i in indices]
        )
    elif isinstance(state, WState):
        mask = (1 << state.block_size) - 1
        support = np.array(
            [
                all(
                    ((int(i) >> (block * state.block_size)) & mask).bit_count() == 1
                    for block in range(state.n_blocks)
                )
                for i in indices
            ]
        )
    else:
        raise ValueError(f"No amplitude definition for {state.name}.")
    return support / np.sqrt(support.sum())


def equal_up_to_phase(a: np.ndarray, b: np.ndarray, atol: float = 1e-8) -> bool:
    """Whether matrices or states ``a`` and ``b`` differ only by a global phase."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    pivot = np.unravel_index(np.argmax(np.abs(b)), b.shape)
    if abs(a[pivot]) < atol:
        return False
    phase = b[pivot] / a[pivot]
    return abs(abs(phase) - 1.0) <= atol and np.allclose(a * phase, b, atol=atol)
