import qiskit
import logging

from multiprocessing import Pool
from qiskit_aer import AerSimulator

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class ParallelSimulator:
    def __init__(self, num_processes=2):
        self.processes = num_processes
        self.engine = 'qiskit'

    @staticmethod
    def simulate_circuit(circuit_data, shots):
        circuit_label = circuit_data[0]
        circuit = circuit_data[1]
        qiskit_circuit = qiskit.QuantumCircuit.from_qasm_str(circuit)
        aer_simulator = AerSimulator()
        job = aer_simulator.run(qiskit_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts(0)
        return {'label': circuit_label, 'results': dict(counts)}

    def simulate(self, circuits, shots=1024):
        logger.debug(f"Simulating {len(circuits)} circuits with {self.processes} processes")
        with Pool(processes=self.processes) as pool:
            results = pool.starmap(self.simulate_circuit, [(
                circuit, shots) for circuit in circuits.items()])
        return results


if __name__ == "__main__":
    para_simulator = ParallelSimulator(num_processes=2)
    circuits = {'1': "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\ncreg meas[3];\nh q;\nmeasure q -> meas;",
                '2': "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\ncreg meas[3];\nh q;\nmeasure q -> meas;",
                '3': "OPENQASM 2.0;\ninclude \"qelib1.inc\";\nqreg q[3];\ncreg meas[3];\nh q;\nmeasure q -> meas;"}
    results = para_simulator.simulate(circuits, shots=1024)
    print(results)
