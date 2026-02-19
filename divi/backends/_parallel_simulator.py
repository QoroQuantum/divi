# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import bisect
import heapq
import logging
import os
import threading
from collections.abc import Mapping, Sequence
from functools import partial
from multiprocessing import Pool, current_process
from typing import Any, Literal
from warnings import warn

from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.providers import Backend
from qiskit.transpiler.exceptions import TranspilerError
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from divi.backends import CircuitRunner
from divi.backends._execution_result import ExecutionResult

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
        # Import only when actually needed
        import qiskit_ibm_runtime.fake_provider as fk_prov

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


class ParallelSimulator(CircuitRunner):
    def __init__(
        self,
        n_processes: int | None = None,
        shots: int = 5000,
        simulation_seed: int | None = None,
        qiskit_backend: Backend | Literal["auto"] | None = None,
        noise_model: NoiseModel | None = None,
        _deterministic_execution: bool = False,
        track_depth: bool = False,
    ):
        """
        A parallel wrapper around Qiskit's AerSimulator using Qiskit's built-in parallelism.

        Args:
            n_processes (int | None, optional): Number of parallel processes to use for transpilation and
                simulation. If None, defaults to half the available CPU cores (min 2, max 8).
                Controls both transpilation parallelism and execution parallelism. The execution
                parallelism mode (circuit or shot) is automatically selected based on workload
                characteristics.
            shots (int, optional): Number of shots to perform. Defaults to 5000.
            simulation_seed (int, optional): Seed for the random number generator to ensure reproducibility. Defaults to None.
            qiskit_backend (Backend | Literal["auto"] | None, optional): A Qiskit backend to initiate the simulator from.
            If "auto" is passed, the best-fit most recent fake backend will be chosen for the given circuit.
            Defaults to None, resulting in noiseless simulation.
            noise_model (NoiseModel, optional): Qiskit noise model to use in simulation. Defaults to None.
            track_depth (bool, optional): If True, record circuit depth for each submitted batch.
                Access via :attr:`depth_history` after execution. Defaults to False.
        """
        super().__init__(shots=shots, track_depth=track_depth)

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
        return False

    @property
    def is_async(self) -> bool:
        """
        Whether the backend executes circuits asynchronously.
        """
        return False

    def _resolve_backend(self, circuit: QuantumCircuit | None = None) -> Backend | None:
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

    def _create_simulator(self, resolved_backend: Backend | None) -> AerSimulator:
        """Create an AerSimulator instance from a resolved backend or noise model."""
        return (
            AerSimulator.from_backend(resolved_backend)
            if resolved_backend is not None
            else AerSimulator(noise_model=self.noise_model)
        )

    def _execute_circuits_deterministically(
        self,
        circuit_labels: list[str],
        transpiled_circuits: list[QuantumCircuit],
        resolved_backend: Backend | None,
    ) -> list[dict[str, Any]]:
        """
        Execute circuits individually for debugging purposes.

        This method ensures deterministic results by running each circuit with its own
        simulator instance and the same seed. Used internally for debugging non-deterministic
        behavior in batch execution.

        Args:
            circuit_labels: List of circuit labels
            transpiled_circuits: List of transpiled QuantumCircuit objects
            resolved_backend: Resolved backend for simulator creation

        Returns:
            List of result dictionaries
        """
        results = []
        for i, (label, transpiled_circuit) in enumerate(
            zip(circuit_labels, transpiled_circuits)
        ):
            # Create a new simulator instance for each circuit with the same seed
            circuit_simulator = self._create_simulator(resolved_backend)

            if self.simulation_seed is not None:
                circuit_simulator.set_options(seed_simulator=self.simulation_seed + i)

            # Run the single circuit
            job = circuit_simulator.run(transpiled_circuit, shots=self.shots)
            circuit_result = job.result()
            counts = circuit_result.get_counts(0)
            results.append({"label": label, "results": dict(counts)})

        return results

    def _configure_simulator_parallelism(
        self, aer_simulator: AerSimulator, num_circuits: int
    ):
        """Configure AerSimulator parallelism options based on workload."""
        if self.simulation_seed is not None:
            aer_simulator.set_options(seed_simulator=self.simulation_seed)

        # Default to utilizing all allocated processes for threads
        options = {"max_parallel_threads": self.n_processes}

        if num_circuits > 1:
            # Batch mode: parallelize experiments
            options.update(
                {
                    "max_parallel_experiments": min(num_circuits, self.n_processes),
                    "max_parallel_shots": 1,
                }
            )
        elif self.shots >= self.n_processes:
            # Single circuit, high shots: parallelize shots
            options.update(
                {
                    "max_parallel_experiments": 1,
                    "max_parallel_shots": self.n_processes,
                }
            )
        else:
            # Single circuit, low shots: default behavior (usually serial shots)
            options.update(
                {
                    "max_parallel_experiments": 1,
                    "max_parallel_shots": 1,
                }
            )

        aer_simulator.set_options(**options)

    def submit_circuits(self, circuits: Mapping[str, str]) -> ExecutionResult:
        """
        Submit multiple circuits for parallel simulation using Qiskit's built-in parallelism.

        Uses Qiskit's native batch transpilation and execution, which handles parallelism
        internally.

        Args:
            circuits (dict[str, str]): Dictionary mapping circuit labels to OpenQASM
                string representations.

        Returns:
            ExecutionResult: Contains results directly (synchronous execution).
                Results are in the format: [{"label": str, "results": dict}, ...]
        """
        logger.debug(
            f"Simulating {len(circuits)} circuits with {self.n_processes} processes"
        )

        # 1. Parse Circuits
        circuit_labels = list(circuits.keys())
        qiskit_circuits = [
            QuantumCircuit.from_qasm_str(qasm) for qasm in circuits.values()
        ]

        if self.track_depth:
            self._depth_history.append([qc.depth() for qc in qiskit_circuits])

        # 2. Resolve Backend
        if self.qiskit_backend == "auto":
            max_qubits_circ = max(qiskit_circuits, key=lambda x: x.num_qubits)
            resolved_backend = self._resolve_backend(max_qubits_circ)
        else:
            resolved_backend = self._resolve_backend()

        # 3. Configure Simulator
        aer_simulator = self._create_simulator(resolved_backend)
        self._configure_simulator_parallelism(aer_simulator, len(qiskit_circuits))

        # 4. Transpile
        transpiled_circuits = transpile(
            qiskit_circuits, aer_simulator, num_processes=self.n_processes
        )

        # 5. Execute
        if self._deterministic_execution:
            results = self._execute_circuits_deterministically(
                circuit_labels, transpiled_circuits, resolved_backend
            )
            return ExecutionResult(results=results)

        job = aer_simulator.run(transpiled_circuits, shots=self.shots)
        batch_result = job.result()

        # Check for non-determinism warnings
        metadata = batch_result.metadata
        if (
            parallel_experiments := metadata.get("parallel_experiments", 1)
        ) > 1 and self.simulation_seed is not None:
            omp_nested = metadata.get("omp_nested", False)
            logger.warning(
                f"Parallel execution detected (parallel_experiments={parallel_experiments}, "
                f"omp_nested={omp_nested}). Results may not be deterministic across different "
                "grouping strategies. Consider enabling deterministic mode for "
                "deterministic results."
            )

        # 6. Format Results
        results = [
            {"label": label, "results": dict(batch_result.get_counts(i))}
            for i, label in enumerate(circuit_labels)
        ]
        return ExecutionResult(results=results)

    @staticmethod
    def estimate_run_time_single_circuit(
        circuit: str,
        qiskit_backend: Backend | Literal["auto"],
        **transpilation_kwargs,
    ) -> float:
        """
        Estimate the execution time of a quantum circuit on a given backend, accounting for parallel gate execution.

        Parameters:
            circuit: The quantum circuit to estimate execution time for as a QASM string.
            qiskit_backend: A Qiskit backend to use for gate time estimation.

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
        durations = resolved_backend.target.durations()

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
        if precomputed_durations is None:
            with Pool() as p:
                estimated_run_times = p.map(
                    partial(
                        ParallelSimulator.estimate_run_time_single_circuit,
                        qiskit_backend="auto",
                        **transpilation_kwargs,
                    ),
                    circuits,
                )
            estimated_run_times_sorted = sorted(estimated_run_times, reverse=True)
        else:
            estimated_run_times_sorted = sorted(precomputed_durations, reverse=True)

        # Optimization for trivial case
        if n_qpus >= len(estimated_run_times_sorted):
            return estimated_run_times_sorted[0] if estimated_run_times_sorted else 0.0

        # LPT (Longest Processing Time) scheduling using a min-heap of processor finish times
        processor_finish_times = [0.0] * n_qpus
        for run_time in estimated_run_times_sorted:
            heapq.heappush(
                processor_finish_times, heapq.heappop(processor_finish_times) + run_time
            )

        return max(processor_finish_times)
