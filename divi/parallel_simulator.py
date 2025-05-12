import heapq
import logging
from functools import partial
from multiprocessing import Pool
from typing import Optional
from warnings import warn

from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGOpNode
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeCairoV2, FakeWashingtonV2
from qiskit_ibm_runtime.fake_provider.fake_backend import FakeBackendV2

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ParallelSimulator:
    def __init__(self, n_processes: int = 2, n_qpus: int = 5):
        self.n_processes = n_processes
        self.engine = "qiskit"
        self.n_qpus = n_qpus

    @staticmethod
    def simulate_circuit(circuit_data, shots, simulation_seed):
        circuit_label, circuit = circuit_data

        qiskit_circuit = QuantumCircuit.from_qasm_str(circuit)

        aer_simulator = AerSimulator()
        aer_simulator.set_option("seed_simulator", simulation_seed)
        job = aer_simulator.run(qiskit_circuit, shots=shots)

        result = job.result()
        counts = result.get_counts(0)

        return {"label": circuit_label, "results": dict(counts)}

    def simulate(self, circuits, shots=1024, simulation_seed=None):
        logger.debug(
            f"Simulating {len(circuits)} circuits with {self.n_processes} processes"
        )
        with Pool(processes=self.n_processes) as pool:
            results = pool.starmap(
                self.simulate_circuit,
                [(circuit, shots, simulation_seed) for circuit in circuits.items()],
            )
        return results

    @staticmethod
    def estimate_run_time_single_circuit(
        circuit: str,
        backend: Optional[FakeBackendV2] = None,
        **transpilation_kwargs,
    ):
        """
        Estimate the execution time of a quantum circuit on a given backend, accounting for parallel gate execution.

        Parameters:
            circuit: The quantum circuit to estimate execution time for as a QASM string.
            backend: A Qiskit backend to use for gate time estimation.

        Returns:
            float: Estimated execution time in seconds.
        """
        qiskit_circuit = QuantumCircuit.from_qasm_str(circuit)

        if not backend:
            backend = (
                FakeCairoV2() if qiskit_circuit.num_qubits <= 27 else FakeWashingtonV2()
            )

        transpiled_circuit = transpile(qiskit_circuit, backend, **transpilation_kwargs)

        dag = circuit_to_dag(transpiled_circuit)

        total_run_time_s = 0.0
        for node in dag.longest_path():
            if not isinstance(node, DAGOpNode):
                continue

            op_name = node.name

            if node.num_clbits == 1:
                idx = (node.cargs[0]._index,)

            if op_name != "measure" and node.num_qubits > 0:
                idx = tuple(qarg._index for qarg in node.qargs)

            try:
                total_run_time_s += (
                    backend.instruction_durations.duration_by_name_qubits[
                        (op_name, idx)
                    ][0]
                )
            except KeyError:
                warn(f"Instruction duration not found: {op_name}")

        return total_run_time_s

    def estimate_run_time_batch(
        self,
        circuits: Optional[list[str]] = None,
        precomputed_duration: Optional[list[float]] = None,
        **transpilation_kwargs,
    ):
        """
        Estimate the execution time of a quantum circuit on a given backend, accounting for parallel gate execution.

        Parameters:
            circuits (list[str]): The quantum circuits to estimate execution time for, as QASM strings.
            precomputed_durations (list[float]): A list of precomputed durations to use.
        Returns:
            float: Estimated execution time in seconds.
        """

        # Compute the run time estimates for each given circuit, in descending order
        if precomputed_duration is None:
            with Pool() as p:
                estimated_run_times = p.map(
                    partial(
                        self.estimate_run_time_single_circuit, **transpilation_kwargs
                    ),
                    circuits,
                )
            estimated_run_times_sorted = sorted(estimated_run_times, reverse=True)
        else:
            estimated_run_times_sorted = sorted(precomputed_duration, reverse=True)

        # Just return the longest run time if there are enough QPUs
        if self.n_qpus >= len(estimated_run_times_sorted):
            return estimated_run_times_sorted[0]

        # Initialize processor queue with (total_run_time, processor_id)
        # Using a min heap to always get the processor that will be free first
        processors = [(0, i) for i in range(self.n_qpus)]
        heapq.heapify(processors)

        # Assign each task to the processor that will be free first
        for run_time in estimated_run_times_sorted:
            current_run_time, processor_id = heapq.heappop(processors)
            new_run_time = current_run_time + run_time
            heapq.heappush(processors, (new_run_time, processor_id))

        # The total run time is the maximum run time across all processors
        return max(run_time for run_time, _ in processors)
