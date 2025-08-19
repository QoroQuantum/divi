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

from divi.interfaces import CircuitRunner

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
    ):
        """
        A multi-process wrapper around Qiskit's AerSimulator.

        Args:
            n_processes (int, optional): Number of parallel processes to use for simulation. Defaults to 2.
            shots (int, optional): Number of shots to perform. Defaults to 5000.
            simulation_seed (int, optional): Seed for the random number generator to ensure reproducibility. Defaults to None.
            backend (Backend or "auto, optional): A Qiskit backend to initiate the simulator from. If "auto" is passed,
             the best-fit most recent fake backend will be chosen for the given circuit. Defaults to None, resulting in noiseless simulation.
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

    @staticmethod
    def simulate_circuit(
        circuit_data: tuple[str, str],
        shots: int,
        simulation_seed: int | None = None,
        qiskit_backend: Backend | None = None,
        noise_model: NoiseModel | None = None,
    ):
        circuit_label, circuit = circuit_data

        qiskit_circuit = QuantumCircuit.from_qasm_str(circuit)

        resolved_backend = (
            _find_best_fake_backend(qiskit_circuit)[-1]()
            if qiskit_backend == "auto"
            else qiskit_backend
        )

        aer_simulator = (
            AerSimulator.from_backend(resolved_backend)
            if qiskit_backend
            else AerSimulator(noise_model=noise_model)
        )
        transpiled_circuit = transpile(qiskit_circuit, aer_simulator)

        aer_simulator.set_option("seed_simulator", simulation_seed)
        job = aer_simulator.run(transpiled_circuit, shots=shots)

        result = job.result()
        counts = result.get_counts(0)

        return {"label": circuit_label, "results": dict(counts)}

    def set_seed(self, seed: int):
        self.simulation_seed = seed

    def submit_circuits(self, circuits: dict[str, str]):
        logger.debug(
            f"Simulating {len(circuits)} circuits with {self.n_processes} processes"
        )

        with Pool(processes=self.n_processes) as pool:
            results = pool.starmap(
                self.simulate_circuit,
                [
                    (
                        circuit,
                        self.shots,
                        self.simulation_seed,
                        self.qiskit_backend,
                        self.noise_model,
                    )
                    for circuit in circuits.items()
                ],
            )
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

            if node.num_clbits == 1:
                idx = (node.cargs[0]._index,)

            if op_name != "measure" and node.num_qubits > 0:
                idx = tuple(qarg._index for qarg in node.qargs)

            try:
                total_run_time_s += (
                    qiskit_backend.instruction_durations.duration_by_name_qubits[
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
        precomputed_duration: list[float] | None = None,
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
        if precomputed_duration is None:
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
            estimated_run_times_sorted = sorted(precomputed_duration, reverse=True)

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
