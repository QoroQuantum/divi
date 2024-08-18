import pennylane as qml
import numpy as npp
import warnings

from pennylane import numpy as np
from qprog.quantum_program import QuantumProgram
from circuit import Circuit
from enum import Enum

try:
    import openfermionpyscf
except ImportError:
    warnings.warn(
        "openfermionpyscf not installed. Some functionality may be limited.")

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
    def __init__(self, symbols, bond_lengths, coordinate_structure, optimizer=Optimizers.NELDER_MEAD, ansatze=(Ansatze.HARTREE_FOCK,), *args, **kwargs) -> None:
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
        self.params = {}
        for ansatz in ansatze:
            self.params[ansatz] = []
        self.current_iteration = 0
        self.optimizer = optimizer
        self.shots = 1000

        self.coordinate_structure = coordinate_structure
        assert len(self.coordinate_structure) == len(
            self.symbols), "The number of symbols must match the number of coordinates"

        self.hamiltonian_ops = self._generate_hamiltonian_operations()

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
        num_electrons = []
        for bond_length in self.bond_lengths:
            # Generate the Hamiltonian for the given bond length
            coordinates = []
            for coord_str in self.coordinate_structure:
                coordinates.append(
                    [coord_str[0] * bond_length, coord_str[1] * bond_length, coord_str[2] * bond_length])
            coordinates = npp.array(coordinates)
            molecule = qml.qchem.Molecule(self.symbols, coordinates)
            hamiltonian, qubits = qml.qchem.molecular_hamiltonian(molecule)
            # , method="openfermion"
            hamiltonian_ops.append(hamiltonian)
            num_qubits.append(qubits)
            num_electrons.append(molecule.n_electrons)
        assert all_equal(
            num_qubits), "All Hamiltonians must have the same number of qubits"
        assert all_equal(
            num_electrons), "All Hamiltonians must have the same number of electrons"
        self.num_qubits = num_qubits[0]
        self.num_electrons = num_electrons[0]
        return hamiltonian_ops

    def _set_ansatz(self, ansatz, params, num_layers=1):
        def _add_hw_efficient_ansatz(params, num_layers):
            qml.RX(params[0], wires=[0])

        def _add_ry_ansatz(params, num_layers):
            p = 0
            for _ in range(num_layers):
                for j in range(self.num_qubits):
                    qml.RY(params[p], wires=[j])
                    p += 1

        def _add_layered_ansatz(params, num_layers):
            pass

        def _add_qaoa_ansatz(params, num_layers):
            pass

        def _add_ryrz_ansatz(params, num_layers):
            p = 0
            for _ in range(num_layers):
                for j in range(self.num_qubits):
                    qml.RY(params[p], wires=[j])
                    p += 1
                    qml.RZ(params[p], wires=[j])

        def _add_uccsd_ansatz(params, num_layers):
            pass

        def _add_hartree_fock_ansatz(params, num_layers):
            hf_state = np.array(
                [1 if i < self.num_electrons else 0 for i in range(self.num_qubits)])
            qml.BasisState(hf_state, wires=[i for i in range(self.num_qubits)])
            qml.DoubleExcitation(params, wires=range(self.num_qubits))

        if ansatz == Ansatze.UCCSD:
            _add_uccsd_ansatz(params, num_layers)
        elif ansatz == Ansatze.HARTREE_FOCK:
            _add_hartree_fock_ansatz(params, num_layers)
        elif ansatz == Ansatze.RY:
            _add_ry_ansatz(params, num_layers)
        elif ansatz == Ansatze.RYRZ:
            _add_ryrz_ansatz(params, num_layers)
        elif ansatz == Ansatze.HW_EFFICIENT:
            _add_hw_efficient_ansatz(params, num_layers)
        elif ansatz == Ansatze.LAYERED:
            _add_layered_ansatz(params, num_layers)
        elif ansatz == Ansatze.QAOA:
            _add_qaoa_ansatz(params, num_layers)

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
            of the pieces of the Hamiltonian. Since these pieces are different, it is easier
            to find the expectation values for them when they are measured in their respective bases.

            args:
                H: The Hamiltonian to determine the measurement basis for, generally going to be a term
                from a larger Hamiltonian
            returns:
                (list): A list containing the wires the Hamiltonian is acting on, as well as the operator to set the measurement basis
                of.            
            """

            if H.base.name == "PauliX":
                return [(H.wires, qml.X)]
            elif H.base.name == "PauliY":
                return [(H.wires, qml.Y)]
            elif H.base.name == "Prod":
                wires = H.base.wires
                obs = [ob.name for ob in H.base.obs]
                bases = []
                for i, wire in enumerate(wires):
                    if obs[i] == "PauliX":
                        bases.append(([wire], qml.X))
                    elif obs[i] == "PauliY":
                        bases.append(([wire], qml.Y))
                return bases

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
                qml.adjoint(qml.S(wires=wire))

        def _prepare_circuit(ansatz, hamiltonian, params):
            self._set_ansatz(ansatz, params)
            measurement_basis = _determine_measurement_basis(hamiltonian)
            if measurement_basis is not None:
                for wire, pauli in measurement_basis:
                    _add_measurement(wire, pauli)
            return qml.sample()

        # Generate a circuit for each bond length, ansatz, and parameter set grouping
        for i, _ in enumerate(self.bond_lengths):
            for ansatz in self.ansatze:
                self.circuits[(i, ansatz)] = []
                params_list = self.params[ansatz]
                for p, params in enumerate(params_list):
                    for j, hamiltonian in enumerate(self.hamiltonian_ops[i]):
                        device = qml.device(
                            "qiskit.aer", wires=self.num_qubits, shots=self.shots)
                        q_node = qml.QNode(_prepare_circuit, device)
                        q_node(ansatz, hamiltonian, params)
                        circuit = Circuit(
                            device, tag=f"{i}_{ansatz.value}_{p}_{j}")
                        self.circuits[(i, ansatz)].append(circuit)

    def _get_num_params_for_ansatz(self, ansatz):
        if ansatz == Ansatze.UCCSD:
            return self.num_qubits
        elif ansatz == Ansatze.HARTREE_FOCK:
            return 1
        elif ansatz == Ansatze.RY:
            return self.num_qubits
        elif ansatz == Ansatze.RYRZ:
            return 2 * self.num_qubits
        elif ansatz == Ansatze.HW_EFFICIENT:
            return 1
        elif ansatz == Ansatze.LAYERED:
            return 1
        elif ansatz == Ansatze.QAOA:
            return 1

    def run_iteration(self):
        assert self.hamiltonian_ops is not None and len(
            self.hamiltonian_ops) > 0, "Hamiltonian operators must be generated before running the VQE"

        if self.optimizer == Optimizers.NELDER_MEAD:
            num_param_sets = 1
        elif self.optimizer == Optimizers.MONTE_CARLO:
            num_param_sets = 10

        if self.current_iteration == 0:
            for ansatz in self.ansatze:
                num_params = self._get_num_params_for_ansatz(ansatz)
                self.params[ansatz] = [npp.random.uniform(
                    0, 2*np.pi, num_params) for _ in range(num_param_sets)]

        self._generate_circuits()


if __name__ == "__main__":
    vqe_problem = VQE(symbols=["H", "H"],
                      bond_lengths=[0.5, 1.0],
                      coordinate_structure=[(-1, -1, 0), (-1, 0.5, 0)],
                      ansatze=[Ansatze.HARTREE_FOCK, Ansatze.RYRZ])
    vqe_problem.run_iteration()
    c = 0

    for circuits in vqe_problem.circuits.values():
        for circuit in circuits:
            print(circuit.qasm_circuit)
            print("\n")
            c += 1

    print(f"Total circuits: {c}")
