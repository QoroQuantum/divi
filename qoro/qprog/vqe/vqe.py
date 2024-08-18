import pennylane as qml
import numpy as npp
import warnings

from pennylane import numpy as np
from qprog.quantum_program import QuantumProgram
from circuit import Circuit
from enum import Enum


warnings.filterwarnings("ignore", category=UserWarning)


class Ansatze(Enum):
    UCCSD = "UCCSD"
    RY = "RY"
    RYRZ = "RYRZ"
    HW_EFFICIENT = "HW_EFFICIENT"
    LAYERED = "LAYERED"
    QAOA = "QAOA"
    HARTREE_FOCK = "HF"

    def describe(self):
        return self.name, self.value


class Optimizers(Enum):
    NELDER_MEAD = "Nelder-Mead"
    MONTE_CARLO = "Monte Carlo"

    def describe(self):
        return self.name, self.value


class VQE(QuantumProgram):
    def __init__(self, symbols, bond_lengths, coordinate_structure, optimizer="Nelder-Mead", ansatze=(Ansatze.HARTREE_FOCK,), *args, **kwargs) -> None:
        """
        Initialize the VQE problem.

        args:
            symbols (list): The symbols of the atoms in the molecule
            bond_lengths (list): The bond lengths to consider
            coordinate_structure (list): The coordinate structure of the molecule
            ansatze (list): The ansatze to use for the VQE problem
        """
        super().__init__(*args, **kwargs)
        self.symbols = symbols
        self.bond_lengths = bond_lengths
        self.num_qubits = 0
        self.circuits = {}
        self.results = {}
        self.ansatze = ansatze
        self.current_iteration = 0
        self.optimizer = optimizer

        self.coordinate_structure = coordinate_structure
        assert len(self.coordinate_structure) == len(
            self.symbols), "The number of symbols must match the number of coordinates"

        self.hamiltonian_ops = self._generate_hamiltonian_operations()
        self._generate_circuits()

    def _generate_hamiltonian_operations(self):
        """
        Generate the Hamiltonian operators for the given bond lengths.

        returns:
            (list) Hamiltonians for each bond length.
        """
        def all_equal(iterator):
            iterator = iter(iterator)
            try:
                first = next(iterator)
            except StopIteration:
                return True
            return all(first == x for x in iterator)

        hamiltonian_ops = []
        num_qubits = []
        for bond_length in self.bond_lengths:
            # Generate the Hamiltonian for the given bond length
            coordinates = []
            for coord_str in self.coordinate_structure:
                coordinates.append(
                    [coord_str[0] * bond_length, coord_str[1] * bond_length, coord_str[2] * bond_length])
            coordinates = npp.array(coordinates)
            molecule = qml.qchem.Molecule(self.symbols, coordinates)
            hamiltonian, qubits = qml.qchem.molecular_hamiltonian(molecule)
            hamiltonian_ops.append(hamiltonian)
            num_qubits.append(qubits)
        assert all_equal(
            num_qubits), "All Hamiltonians must have the same number of qubits"
        self.num_qubits = num_qubits[0]
        return hamiltonian_ops

    def _set_ansatz(self, ansatz, params, num_layers=1):
        if ansatz == Ansatze.UCCSD:
            self._add_uccsd_ansatz(num_layers)
        elif ansatz == Ansatze.HARTREE_FOCK:
            self._add_hartree_fock_ansatz(params, num_layers)
        elif ansatz == Ansatze.RY:
            self._add_ry_ansatz(num_layers)
        elif ansatz == Ansatze.RYRZ:
            self._add_ryrz_ansatz(num_layers)
        elif ansatz == Ansatze.HW_EFFICIENT:
            self._add_hw_efficient_ansatz(num_layers)
        elif ansatz == Ansatze.LAYERED:
            self._add_layered_ansatz(num_layers)
        elif ansatz == Ansatze.QAOA:
            self._add_qaoa_ansatz(num_layers)

    def _add_hw_efficient_ansatz(self, params, num_layers):
        qml.RX(params[0], wires=[0])

    def _add_ry_ansatz(self, params, num_layers):
        pass

    def _add_layered_ansatz(self,  params, num_layers):
        pass

    def _add_qaoa_ansatz(self,  params, num_layers):
        pass

    def _add_ryrz_ansatz(self,  params, num_layers):
        pass

    def _add_uccsd_ansatz(self, params, num_layers):
        pass

    def _add_hartree_fock_ansatz(self, params, num_layers):
        qml.RX(params[0], wires=[0])

    def _generate_circuits(self) -> None:
        """
        Generate the circuits for the VQE problem.

        In this method, we generate bulk circuits based on the selected parameters. 
        We generate circuits for each bond length and each ansatz and optimization choice.

        The structure of the circuits is as follows:
        - For each bond length:
            - For each ansatz:
                - Generate the circuit
        """

        def _determine_measurement_basis(H):
            """
            To go through with the post-processing of the VQE, we have to find the expectation values
            of the Hamiltonian pieces. Since these Hamiltonian pieces are different, it is significantly easier
            to find the expectation values for them when they are measured in their respective bases.

            args:
                H: The Hamiltonian to determine the measurement basis for, generally going to be a term
                from a larger Hamiltonian
            returns:
                (list): A list containing the wires the Hamiltonian is acting on, as well as the operator to set the measurement basis
                of.            
            """

            if isinstance(H, qml.X):
                return [(H.wires, qml.X)]
            if isinstance(H, qml.Y):
                return [(H.wires, qml.Y)]
            if isinstance(H, qml.operation.Tensor):
                return [(op.wires, type(op)) for op in H.non_identity_obs if isinstance(op, (qml.X, qml.Y))]

        def _add_measurement(wire, basis):
            """
            Add a measurement operation to the circuit.

            args:
                wire (qml.Wires): The wire to measure
                pauli (qml.Pauli): The Pauli operator to measure with
            """
            if basis == qml.X:
                qml.Hadamard(wires=wire)
            elif basis == qml.Y:
                qml.S(wires=wire).inv()

        def _prepare_circuit(ansatz, hamiltonian, params):
            self._set_ansatz(ansatz, params)
            measurement_basis = _determine_measurement_basis(hamiltonian)
            if measurement_basis is not None:
                for wire, pauli in measurement_basis:
                    _add_measurement(wire, pauli)
            return qml.sample()

        # TODO: number of params is dependant on the ansatz (layers and qubits)
        params = npp.random.rand(1, self.num_qubits)
        for i, _ in enumerate(self.bond_lengths):
            for ansatz in self.ansatze:
                self.circuits[(i, ansatz)] = []
                for j, hamiltonian in enumerate(self.hamiltonian_ops[i]):
                    device = qml.device(
                        "qiskit.aer", wires=self.num_qubits, shots=1000)
                    q_node = qml.QNode(_prepare_circuit, device)
                    q_node(ansatz, hamiltonian, params)
                    circuit = Circuit(device, tag=f"{i}_{ansatz.value}_{j}")
                    self.circuits[(i, ansatz)].append(circuit)

    def _run_iteration(self):
        pass


if __name__ == "__main__":
    vqe_problem = VQE(symbols=["H", "H"], bond_lengths=[
                      0.5, 1.0], coordinate_structure=[(1, 0, 0), (0, -1, 0)])

    for circuits in vqe_problem.circuits.values():
        for circuit in circuits:
            print(circuit.qasm_circuit)
            print("\n")
