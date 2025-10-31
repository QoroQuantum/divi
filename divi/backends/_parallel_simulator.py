# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import bisect
import heapq
import logging
from functools import partial
from multiprocessing import Pool
from typing import Literal
from warnings import warn

import qiskit_ibm_runtime.fake_provider as fk_prov
from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit.providers import Backend
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from divi.backends import CircuitRunner

logger = logging.getLogger(__name__)

FAKE_BACKENDS = {
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


def _find_best_fake_backend(circuit: QuantumCircuit):
    keys = sorted(FAKE_BACKENDS.keys())
    pos = bisect.bisect_left(keys, circuit.num_qubits)
    return FAKE_BACKENDS[keys[pos]] if pos < len(keys) else None


class ParallelSimulator(CircuitRunner):
    def __init__(
        self,
        n_processes: int = 2,
        shots: int = 5000,
        simulation_seed: int | None = None,
        qiskit_backend: Backend | Literal["auto"] | None = None,
        noise_model: NoiseModel | None = None,
        _deterministic_execution: bool = False,
    ):
        """
        A parallel wrapper around Qiskit's AerSimulator using Qiskit's built-in parallelism.

        Args:
            n_processes (int, optional): Number of parallel processes to use for transpilation and
                simulation. Defaults to 2. This sets both the transpile num_processes and
                AerSimulator's max_parallel_experiments.
            shots (int, optional): Number of shots to perform. Defaults to 5000.
            simulation_seed (int, optional): Seed for the random number generator to ensure reproducibility. Defaults to None.
            qiskit_backend (Backend | Literal["auto"] | None, optional): A Qiskit backend to initiate the simulator from.
            If "auto" is passed, the best-fit most recent fake backend will be chosen for the given circuit.
            Defaults to None, resulting in noiseless simulation.
            noise_model (NoiseModel, optional): Qiskit noise model to use in simulation. Defaults to None.
        """
        super().__init__(shots=shots)

        if qiskit_backend and noise_model:
            warn(
                "Both `qiskit_backend` and `noise_model` have been provided."
                " `noise_model` will be ignored and the model from the backend will be used instead."
            )

        self.n_processes = n_processes
        self.engine = "qiskit"
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

    def _execute_circuits_deterministically(
        self, circuit_labels: list[str], transpiled_circuits: list, resolved_backend
    ) -> list[dict]:
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
            if resolved_backend is not None:
                circuit_simulator = AerSimulator.from_backend(resolved_backend)
            else:
                circuit_simulator = AerSimulator(noise_model=self.noise_model)

            if self.simulation_seed is not None:
                circuit_simulator.set_option("seed_simulator", self.simulation_seed)

            # Run the single circuit
            job = circuit_simulator.run(transpiled_circuit, shots=self.shots)
            circuit_result = job.result()
            counts = circuit_result.get_counts(0)
            results.append({"label": label, "results": dict(counts)})

        return results

    def submit_circuits(self, circuits: dict[str, str]):
        """
        Submit multiple circuits for parallel simulation using Qiskit's built-in parallelism.

        Uses Qiskit's native batch transpilation and execution, which handles parallelism
        internally.

        Args:
            circuits (dict[str, str]): Dictionary mapping circuit labels to OpenQASM
                string representations.

        Returns:
            list[dict]: List of result dictionaries, each containing:
                - 'label' (str): Circuit identifier
                - 'results' (dict): Measurement counts as {bitstring: count}
        """
        logger.debug(
            f"Simulating {len(circuits)} circuits with {self.n_processes} processes"
        )

        # Convert QASM strings to QuantumCircuit objects
        circuit_labels = list(circuits.keys())
        qiskit_circuits = [
            QuantumCircuit.from_qasm_str(qasm) for qasm in circuits.values()
        ]

        # Determine backend for transpilation
        if self.qiskit_backend == "auto":
            # For "auto", find the maximum number of qubits across all circuits to determine backend
            max_qubits_circ = max(qiskit_circuits, key=lambda x: x.num_qubits)
            resolved_backend = _find_best_fake_backend(max_qubits_circ)[-1]()
        elif self.qiskit_backend is not None:
            resolved_backend = self.qiskit_backend
        else:
            resolved_backend = None

        # Create simulator
        if resolved_backend is not None:
            aer_simulator = AerSimulator.from_backend(resolved_backend)
        else:
            aer_simulator = AerSimulator(noise_model=self.noise_model)

        # Set simulator options for parallelism
        # Note: We don't set seed_simulator here because we need different seeds for each circuit
        # to ensure deterministic results when running multiple circuits in parallel
        aer_simulator.set_options(max_parallel_experiments=self.n_processes)

        # Batch transpile all circuits (Qiskit handles parallelism internally)
        transpiled_circuits = transpile(
            qiskit_circuits, aer_simulator, num_processes=self.n_processes
        )

        # Use deterministic execution for debugging if enabled
        if self._deterministic_execution:
            return self._execute_circuits_deterministically(
                circuit_labels, transpiled_circuits, resolved_backend
            )

        # Batch execution with metadata checking for non-deterministic behavior
        job = aer_simulator.run(transpiled_circuits, shots=self.shots)
        batch_result = job.result()

        # Check metadata to detect non-deterministic behavior
        metadata = batch_result.metadata
        parallel_experiments = metadata.get("parallel_experiments", 1)
        omp_nested = metadata.get("omp_nested", False)

        # If parallel execution is detected and we have a seed, warn about potential non-determinism
        if parallel_experiments > 1 and self.simulation_seed is not None:
            logger.warning(
                f"Parallel execution detected (parallel_experiments={parallel_experiments}, "
                f"omp_nested={omp_nested}). Results may not be deterministic across different "
                "grouping strategies. Consider enabling deterministic mode for "
                "deterministic results."
            )

        # Extract results and match with labels
        results = []
        for i, label in enumerate(circuit_labels):
            counts = batch_result.get_counts(i)
            results.append({"label": label, "results": dict(counts)})

        return results

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

        resolved_backend = (
            _find_best_fake_backend(qiskit_circuit)[-1]()
            if qiskit_backend == "auto"
            else qiskit_backend
        )

        transpiled_circuit = transpile(
            qiskit_circuit, resolved_backend, **transpilation_kwargs
        )

        dag = circuit_to_dag(transpiled_circuit)

        total_run_time_s = 0.0
        for node in dag.longest_path():
            if not isinstance(node, DAGOpNode):
                continue

            op_name = node.name

            # Determine qubit indices for the operation
            if node.num_clbits == 1:
                idx = (node.cargs[0]._index,)
            elif op_name != "measure" and node.num_qubits > 0:
                idx = tuple(qarg._index for qarg in node.qargs)
            else:
                # Skip operations without qubits or measurements without classical bits
                continue

            try:
                total_run_time_s += (
                    resolved_backend.instruction_durations.duration_by_name_qubits[
                        (op_name, idx)
                    ][0]
                )
            except KeyError:
                if op_name == "barrier":
                    continue
                warn(f"Instruction duration not found: {op_name}")

        return total_run_time_s

    @staticmethod
    def estimate_run_time_batch(
        circuits: list[str] | None = None,
        precomputed_durations: list[float] | None = None,
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

        # Just return the longest run time if there are enough QPUs
        if n_qpus >= len(estimated_run_times_sorted):
            return estimated_run_times_sorted[0]

        # Initialize processor queue with (total_run_time, processor_id)
        # Using a min heap to always get the processor that will be free first
        processors = [(0, i) for i in range(n_qpus)]
        heapq.heapify(processors)

        # Assign each task to the processor that will be free first
        for run_time in estimated_run_times_sorted:
            current_run_time, processor_id = heapq.heappop(processors)
            new_run_time = current_run_time + run_time
            heapq.heappush(processors, (new_run_time, processor_id))

        # The total run time is the maximum run time across all processors
        return max(run_time for run_time, _ in processors)
