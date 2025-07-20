from divi.interfaces import CircuitRunner

from cunqa import getQPUs, gather
import math


def _allocate_circuits_among_qpus(qpus, circuits):
    """
    Allocate circuits to QPUs in fair-share batches while maintaining circuit order.

    Parameters:
        qpus (list): List of QPU identifiers.
        circuits (list): List of circuits to be allocated.

    Returns:
        dict: Mapping from QPU to list of assigned circuits.
    """
    allocation = {qpu: [] for qpu in qpus}
    num_qpus = len(qpus)
    total_circuits = len(circuits)

    batch_size = math.ceil(total_circuits / num_qpus)
    circuit_index = 0

    for _, qpu in enumerate(qpus):
        j = 0
        while j < batch_size and circuit_index < total_circuits:
            allocation[qpu].append(circuits[circuit_index])
            circuit_index += 1
            j += 1

    return allocation


class CUNQAService(CircuitRunner):

    def __init__(self, shots: int = 1000):
        """
        Initialize the CUNQAService with the number of shots.

        Args:
            shots (int): Number of shots for the quantum circuits.
        """
        super().__init__(shots=shots)
        self.shots = shots
        self.current_job = 0

    def submit_circuits(self, circuits: dict[str, str], **kwargs):
        qjobs = []
        qpus = getQPUs()

        if not qpus or len(qpus) == 0:
            raise RuntimeError("No QPUs available for execution.")

        circuit_allocations = _allocate_circuits_among_qpus(circuits, qpus)

        for qpu, circuit_list in circuit_allocations.items():
            for _, circuit_qasm in circuit_list:
                qjob = qpu.submit(circuit_qasm, transpile=True, shots=self.shots)
                qjobs.append(qjob)

        shot_results = gather(qjobs)
        return_results = []
        for i, shot_result in enumerate(shot_results):
            circuit_label = circuits[i][0]
            return_results.append({"label": circuit_label, "results": shot_result})
        return return_results
