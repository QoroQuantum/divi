# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import bisect
import heapq
import logging
import os
import threading
from collections.abc import Sequence
from functools import partial
from multiprocessing import Pool, current_process
from threading import Event
from typing import Literal
from warnings import warn

from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.providers import BackendV2
from qiskit.quantum_info import Pauli
from qiskit.transpiler.exceptions import TranspilerError
from qiskit_aer import AerSimulator
from qiskit_aer.library import SaveExpectationValue
from qiskit_aer.noise import NoiseModel

from divi._optional import import_optional
from divi.circuits._payloads import CircuitBatch, CircuitPayload, bound_circuits

from .._base import CircuitRunner, ExecutionResult
from .._cancellation import raise_if_cancelled
from .._pauli_serde import ham_ops_terms_for_circuit, pad_ham_ops
from .._shot_allocation import (
    ShotRange,
    bucket_by_shots,
    from_wire,
    per_circuit,
    validate,
)

logger = logging.getLogger(__name__)

# Suppress stevedore extension loading errors (harmless Qiskit v2/provider issue)
_stevedore_logger = logging.getLogger("stevedore.extension")
_stevedore_logger.setLevel(logging.CRITICAL)

# Lazy-loaded fake backends dictionary
_FAKE_BACKENDS_CACHE: dict[int, list] | None = None


def _load_fake_backends() -> dict[int, list]:
    """Lazy load and return the FAKE_BACKENDS dictionary."""
    global _FAKE_BACKENDS_CACHE
    if _FAKE_BACKENDS_CACHE is None:
        fk_prov = import_optional(
            "qiskit_ibm_runtime.fake_provider",
            extra="aer",
            capability="QiskitSimulator fake backends",
        )

        _FAKE_BACKENDS_CACHE = {
            5: [
                fk_prov.FakeManilaV2,
                fk_prov.FakeBelemV2,
                fk_prov.FakeLimaV2,
                fk_prov.FakeQuitoV2,
            ],
            7: [
                fk_prov.FakeOslo,
                fk_prov.FakePerth,
                fk_prov.FakeLagosV2,
                fk_prov.FakeNairobiV2,
            ],
            15: [fk_prov.FakeMelbourneV2],
            16: [fk_prov.FakeGuadalupeV2],
            20: [
                fk_prov.FakeAlmadenV2,
                fk_prov.FakeJohannesburgV2,
                fk_prov.FakeSingaporeV2,
                fk_prov.FakeBoeblingenV2,
            ],
            27: [
                fk_prov.FakeGeneva,
                fk_prov.FakePeekskill,
                fk_prov.FakeAuckland,
                fk_prov.FakeCairoV2,
            ],
        }
    return _FAKE_BACKENDS_CACHE


def _find_best_fake_backend(circuit: QuantumCircuit) -> list[type] | None:
    """Find the best fake backend for a given circuit based on qubit count.

    Args:
        circuit: QuantumCircuit to find a backend for.

    Returns:
        List of fake backend classes that support the circuit's qubit count, or None.
    """
    fake_backends = _load_fake_backends()
    keys = sorted(fake_backends.keys())
    pos = bisect.bisect_left(keys, circuit.num_qubits)
    return fake_backends[keys[pos]] if pos < len(keys) else None


# Public API for backward compatibility with tests
def __getattr__(name: str):
    """Lazy load FAKE_BACKENDS when accessed."""
    if name == "FAKE_BACKENDS":
        return _load_fake_backends()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _default_n_processes() -> int:
    """Get a reasonable default number of processes based on CPU count.

    Uses most available CPU cores (all minus 1, or 3/4 if many cores), with a
    minimum of 2 and maximum of 16. This provides good parallelism while leaving
    one core free for system processes.

    If running in a different thread or process (not the main thread/process),
    limits to 2 cores to avoid resource contention.

    Returns:
        int: Default number of processes to use.
    """
    # Check if we're running in a worker thread or subprocess
    is_main_thread = threading.current_thread() is threading.main_thread()
    is_main_process = current_process().name == "MainProcess"

    if not (is_main_thread and is_main_process):
        # Running in a different thread/process - limit to 2 cores
        return 2

    cpu_count = os.cpu_count() or 4
    if cpu_count <= 4:
        # For small systems, use all but 1 core
        return max(2, cpu_count - 1)
    elif cpu_count <= 16:
        # For medium systems, use all but 1 core
        return cpu_count - 1
    else:
        # For large systems, use 3/4 of cores, capped at 16
        return min(16, int(cpu_count * 0.75))


class QiskitSimulator(CircuitRunner):
    def __init__(
        self,
        n_processes: int | None = None,
        shots: int = 5000,
        simulation_seed: int | None = None,
        qiskit_backend: BackendV2 | Literal["auto"] | None = None,
        noise_model: NoiseModel | None = None,
        track_depth: bool = False,
        force_sampling: bool = False,
        optimization_level: int | None = None,
        _deterministic_execution: bool = False,
    ):
        """
        A parallel wrapper around Qiskit's AerSimulator using Qiskit's built-in parallelism.

        Args:
            n_processes (int | None, optional): Number of parallel processes to use for transpilation and
                simulation. If None, defaults to all-but-one core (<= 16 cores) or 3/4 of cores
                capped at 16 (> 16 cores); 2 when not running on the main thread/process.
                Controls both transpilation parallelism and execution parallelism;
                ``transpile`` runs serially for circuits submitted off the main thread. The execution
                parallelism mode (circuit or shot) is automatically selected based on workload
                characteristics.
            shots (int, optional): Number of shots to perform. Defaults to 5000.
            simulation_seed (int, optional): Seed for the random number generator to ensure reproducibility. Defaults to None.
            qiskit_backend (BackendV2 | Literal["auto"] | None, optional): A Qiskit backend to initiate the simulator from.
                If ``"auto"`` is passed, the best-fit most recent fake backend will be chosen for the given circuit.
                Defaults to None, resulting in noiseless simulation.
            noise_model (NoiseModel, optional): Qiskit noise model to use in simulation. Defaults to None.
            track_depth (bool, optional): If True, record circuit depth for each submitted batch.
                Access via :attr:`~divi.backends.CircuitRunner.depth_history` after execution. Defaults to False.
            force_sampling (bool, optional): If True, always use shot-based sampling
                even for expectation value measurements. Defaults to False.
            optimization_level (int | None, optional): Passed to
                :func:`~qiskit.compiler.transpile` for every circuit this backend
                runs. Defaults to None, leaving the choice to Qiskit.

                Optimisation rewrites a circuit by how compressible it is, which
                is not uniform across a batch, so under a noise model the
                executed circuits can accumulate different amounts of noise than
                the ones submitted. Protocols that compare circuits to each other
                — :class:`~divi.circuits.quepp.QuEPP` infers a rescaling factor
                from exactly that comparison — need ``optimization_level=0`` to
                stay faithful.
        """
        super().__init__(shots=shots, track_depth=track_depth)

        # Expval mode (save_expval) is incompatible with custom backends /
        # noise models — automatically fall back to shot-based sampling.
        if qiskit_backend is not None or noise_model is not None:
            force_sampling = True
        self._force_sampling = force_sampling

        if qiskit_backend and noise_model:
            warn(
                "Both `qiskit_backend` and `noise_model` have been provided."
                " `noise_model` will be ignored and the model from the backend will be used instead."
            )

        if n_processes is None:
            n_processes = _default_n_processes()
        elif n_processes < 1:
            raise ValueError(f"n_processes must be >= 1, got {n_processes}")
        self._n_processes = n_processes
        self.simulation_seed = simulation_seed
        self.qiskit_backend = qiskit_backend
        self.noise_model = noise_model
        self.optimization_level = optimization_level
        self._deterministic_execution = _deterministic_execution

    def set_seed(self, seed: int):
        """
        Set the random seed for circuit simulation.

        Args:
            seed (int): Seed value for the random number generator used in simulation.
        """
        self.simulation_seed = seed

    @property
    def n_processes(self) -> int:
        """
        Get the current number of parallel processes.

        Returns:
            int: Number of parallel processes configured.
        """
        return self._n_processes

    @n_processes.setter
    def n_processes(self, value: int):
        """
        Set the number of parallel processes (>= 1).

        Controls:
        - Transpilation parallelism
        - OpenMP thread limit
        - Circuit/Shot parallelism (auto-selected based on workload)
        """
        if value < 1:
            raise ValueError(f"n_processes must be >= 1, got {value}")
        self._n_processes = value

    @property
    def supports_expval(self) -> bool:
        """
        Whether the backend supports expectation value measurements.
        """
        return not self._force_sampling

    @property
    def is_async(self) -> bool:
        """
        Whether the backend executes circuits asynchronously.
        """
        return False

    def _resolve_backend(
        self, circuit: QuantumCircuit | None = None
    ) -> BackendV2 | None:
        """Resolve the backend from qiskit_backend setting."""
        if self.qiskit_backend == "auto":
            if circuit is None:
                raise ValueError(
                    "Circuit must be provided when qiskit_backend is 'auto'"
                )
            backend_list = _find_best_fake_backend(circuit)
            if backend_list is None:
                raise ValueError(
                    f"No fake backend available for circuit with {circuit.num_qubits} qubits. "
                    "Please provide an explicit backend or use a smaller circuit."
                )
            return backend_list[-1]()
        return self.qiskit_backend

    def _create_simulator(self, resolved_backend: BackendV2 | None) -> AerSimulator:
        """Create an AerSimulator instance from a resolved backend or noise model."""
        return (
            AerSimulator.from_backend(resolved_backend)
            if resolved_backend is not None
            else AerSimulator(noise_model=self.noise_model)
        )

    def _configure_simulator_parallelism(
        self, aer_simulator: AerSimulator, num_circuits: int
    ):
        """Configure AerSimulator parallelism options based on workload."""
        if self.simulation_seed is not None:
            aer_simulator.set_options(seed_simulator=self.simulation_seed)

        # Default to utilising all allocated processes for threads
        options = {"max_parallel_threads": self.n_processes}

        if num_circuits > 1:
            # Batch mode: parallelise experiments
            options.update(
                {
                    "max_parallel_experiments": min(num_circuits, self.n_processes),
                    "max_parallel_shots": 1,
                }
            )
        elif self.shots >= self.n_processes:
            # Single circuit, high shots: parallelise shots
            options.update(
                {
                    "max_parallel_experiments": 1,
                    "max_parallel_shots": self.n_processes,
                }
            )
        else:
            # Single circuit, low shots: default behaviour (usually serial shots)
            options.update(
                {
                    "max_parallel_experiments": 1,
                    "max_parallel_shots": 1,
                }
            )

        aer_simulator.set_options(**options)

    @staticmethod
    def _prepare_expval_circuit(
        circuit: QuantumCircuit, pauli_ops: list[str]
    ) -> QuantumCircuit:
        """Strip measurements and append ``save_expectation_value`` for each Pauli operator.

        Args:
            circuit: Qiskit circuit (may contain final measurements).
            pauli_ops: List of Pauli strings in divi convention (big-endian, q0 leftmost).

        Returns:
            New circuit with measurements removed and expectation-value save instructions.
        """
        qc = circuit.copy()
        qc.remove_final_measurements(inplace=True)
        for pauli_str in pauli_ops:
            # Reverse: divi big-endian (q0 leftmost) → Qiskit little-endian (q0 rightmost)
            qc.append(
                SaveExpectationValue(Pauli(pauli_str[::-1]), label=pauli_str),
                qargs=range(qc.num_qubits),
            )
        return qc

    def submit_circuits(
        self,
        payloads: Sequence[CircuitPayload] | CircuitBatch,
        *,
        ham_ops: str | None = None,
        circuit_ham_map: list[list[int]] | None = None,
        shot_groups: list[list[int]] | None = None,
        cancellation_event: Event | None = None,
        **kwargs,
    ) -> ExecutionResult:
        """Submit multiple circuits for parallel simulation using Qiskit's built-in parallelism.

        Args:
            payloads: Bound QASM payloads, one resolved circuit per parameter-set
                row — or a collection of already-resolved circuits.
            ham_ops: Semicolon-separated Pauli string for expectation value estimation,
                e.g. ``"ZI;IZ;XX"``. Multiple groups can be pipe-delimited when
                ``circuit_ham_map`` is provided. If None, runs in sampling mode.
                Terms shorter than a circuit are padded onto its first qubits,
                with a warning.
            circuit_ham_map: Each entry is ``[start, end)`` mapping a ``|``-group in
                ``ham_ops`` to a contiguous slice of circuits.
            shot_groups: Per-circuit shot allocation as ``[start, end, shots]``
                triples covering the iteration order of ``circuits``. When
                provided, overrides ``self.shots`` for each range; circuits
                sharing a shot count run in one Aer call. Sampling mode only;
                passing it with ``ham_ops`` raises ``ValueError``.
            cancellation_event: When set before this call, aborts dispatch.
                Aer's ``.run().result()`` cannot be interrupted mid-batch.
            **kwargs: Rejected with ``TypeError``.

        Returns:
            ExecutionResult containing either counts (sampling) or expectation values.
        """
        self._reject_unknown_options(kwargs)
        raise_if_cancelled(cancellation_event, "Qiskit batch cancelled before dispatch")
        self._reject_shot_groups_with_ham_ops(ham_ops, shot_groups)

        circuits = bound_circuits(payloads)
        n_circuits = len(circuits)
        logger.debug(
            f"Simulating {n_circuits} circuits with {self.n_processes} processes"
        )

        labels = list(circuits.keys())
        qiskit_circuits = [
            QuantumCircuit.from_qasm_str(qasm) for qasm in circuits.values()
        ]
        if self.track_depth:
            self._depth_history.append([qc.depth() for qc in qiskit_circuits])

        per_circuit_ops: list[list[str]] = []
        if ham_ops is not None:
            ham_ops = pad_ham_ops(
                ham_ops, circuit_ham_map, [qc.num_qubits for qc in qiskit_circuits]
            )
            per_circuit_ops = [
                ham_ops_terms_for_circuit(i, ham_ops, circuit_ham_map)
                for i in range(n_circuits)
            ]
            qiskit_circuits = [
                self._prepare_expval_circuit(qc, circuit_ops)
                for qc, circuit_ops in zip(qiskit_circuits, per_circuit_ops)
            ]

        resolved_backend = self._resolve_backend(
            max(qiskit_circuits, key=lambda qc: qc.num_qubits)
        )
        simulator = self._create_simulator(resolved_backend)
        self._configure_simulator_parallelism(simulator, n_circuits)
        # Qiskit parallelises transpile by forking, which can deadlock when
        # other threads (e.g. ensemble workers) run concurrently.
        on_main_thread = threading.current_thread() is threading.main_thread()
        transpiled = transpile(
            qiskit_circuits,
            simulator,
            num_processes=self.n_processes if on_main_thread else 1,
            optimization_level=self.optimization_level,
        )

        if ham_ops is not None:
            result = simulator.run(transpiled).result()
            return ExecutionResult(
                results=[
                    {
                        "label": label,
                        "results": {
                            op: float(result.data(i)[op]) for op in circuit_ops
                        },
                    }
                    for i, (label, circuit_ops) in enumerate(
                        zip(labels, per_circuit_ops)
                    )
                ],
                run_time=result.time_taken,
            )

        shot_ranges = (
            [ShotRange(0, n_circuits, self.shots)]
            if shot_groups is None
            else from_wire(shot_groups)
        )
        validate(shot_ranges, n_circuits)
        # Aer applies one shot count per run, so each distinct count is one run;
        # deterministic mode runs every circuit alone on its own seeded simulator.
        if self._deterministic_execution:
            per_circuit_shots = per_circuit(shot_ranges, n_circuits)
            runs = [([i], shots) for i, shots in enumerate(per_circuit_shots)]
        else:
            runs = [(idx, shots) for shots, idx in bucket_by_shots(shot_ranges).items()]

        counts: list[dict[str, int]] = [{}] * n_circuits
        run_time = 0.0
        for indices, shots in runs:
            if self._deterministic_execution:
                simulator = self._create_simulator(resolved_backend)
                if self.simulation_seed is not None:
                    seed = self.simulation_seed + indices[0]
                    simulator.set_options(seed_simulator=seed)
            batch = [transpiled[i] for i in indices]
            result = simulator.run(batch, shots=shots).result()
            run_time += result.time_taken
            parallel_experiments = result.metadata.get("parallel_experiments", 1)
            if parallel_experiments > 1 and self.simulation_seed is not None:
                logger.warning(
                    f"Parallel execution detected (parallel_experiments="
                    f"{parallel_experiments}, omp_nested="
                    f"{result.metadata.get('omp_nested', False)}). Results may not "
                    "be deterministic across different grouping strategies. "
                    "Consider enabling deterministic mode for deterministic results."
                )
            for offset, i in enumerate(indices):
                counts[i] = dict(result.get_counts(offset))

        return ExecutionResult(
            results=[
                {"label": label, "results": circuit_counts}
                for label, circuit_counts in zip(labels, counts)
            ],
            run_time=run_time,
        )

    @staticmethod
    def estimate_run_time_single_circuit(
        circuit: str,
        qiskit_backend: BackendV2 | Literal["auto"],
        **transpilation_kwargs,
    ) -> float:
        """
        Estimate the execution time of a quantum circuit on a given backend, accounting for parallel gate execution.

        Parameters:
            circuit: The quantum circuit to estimate execution time for as a QASM string.
            qiskit_backend: A Qiskit backend to use for gate time estimation.
            transpilation_kwargs: Forwarded to :func:`~qiskit.compiler.transpile`.
                Pass the same ``optimization_level`` the circuit will run at, or
                the estimate describes a different depth than the one executed.

        Returns:
            float: Estimated execution time in seconds.
        """
        qiskit_circuit = QuantumCircuit.from_qasm_str(circuit)

        if qiskit_backend == "auto":
            if not (backend_list := _find_best_fake_backend(qiskit_circuit)):
                raise ValueError(
                    f"No fake backend available for circuit with {qiskit_circuit.num_qubits} qubits. "
                    "Please provide an explicit backend or use a smaller circuit."
                )
            resolved_backend = backend_list[-1]()
        else:
            resolved_backend = qiskit_backend

        transpiled_circuit = transpile(
            qiskit_circuit, resolved_backend, **transpilation_kwargs
        )

        total_run_time_s = 0.0
        target = resolved_backend.target
        if target is None:
            raise RuntimeError(
                f"Backend {resolved_backend!r} has no transpiler target; "
                "cannot estimate run time."
            )
        durations = target.durations()

        for node in circuit_to_dag(transpiled_circuit).longest_path():
            if not isinstance(node, DAGOpNode) or not node.num_qubits:
                continue

            try:
                idx = tuple(q._index for q in node.qargs)
                duration = durations.get(node.name, idx, unit="s")
                total_run_time_s += duration
            except TranspilerError:
                if node.name != "barrier":
                    warn(f"Instruction duration not found: {node.name}")

        return total_run_time_s

    @staticmethod
    def estimate_run_time_batch(
        circuits: Sequence[str] | None = None,
        precomputed_durations: Sequence[float] | None = None,
        n_qpus: int = 5,
        **transpilation_kwargs,
    ) -> float:
        """
        Estimate the execution time of a quantum circuit on a given backend, accounting for parallel gate execution.

        Parameters:
            circuits (list[str]): The quantum circuits to estimate execution time for, as QASM strings.
            precomputed_durations (list[float]): A list of precomputed durations to use.
            n_qpus (int): Number of QPU nodes in the pre-supposed cluster we are estimating runtime against.

        Returns:
            float: Estimated execution time in seconds.
        """

        # Compute the run time estimates for each given circuit, in descending order
        if precomputed_durations is not None:
            estimated_run_times_sorted = sorted(precomputed_durations, reverse=True)
        elif circuits is not None:
            # Pin the worker count to ``_default_n_processes()`` so this
            # static helper inherits the same fork/thread-aware sizing the
            # instance uses, instead of defaulting to ``os.cpu_count()``
            # workers regardless of context.
            with Pool(processes=_default_n_processes()) as p:
                estimated_run_times = p.map(
                    partial(
                        QiskitSimulator.estimate_run_time_single_circuit,
                        qiskit_backend="auto",
                        **transpilation_kwargs,
                    ),
                    circuits,
                )
            estimated_run_times_sorted = sorted(estimated_run_times, reverse=True)
        else:
            raise ValueError(
                "estimate_run_time_batch requires either ``circuits`` or "
                "``precomputed_durations`` to be provided."
            )

        # Optimisation for trivial case
        if n_qpus >= len(estimated_run_times_sorted):
            return estimated_run_times_sorted[0] if estimated_run_times_sorted else 0.0

        # LPT (Longest Processing Time) scheduling using a min-heap of processor finish times
        processor_finish_times = [0.0] * n_qpus
        for run_time in estimated_run_times_sorted:
            heapq.heappush(
                processor_finish_times, heapq.heappop(processor_finish_times) + run_time
            )

        return max(processor_finish_times)
