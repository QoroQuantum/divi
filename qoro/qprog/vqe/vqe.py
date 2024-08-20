import pennylane as qml
import numpy as npp
import warnings

from pennylane import numpy as np
from qprog.quantum_program import QuantumProgram
from circuit import Circuit
from enum import Enum
from qoro_service import JobStatus
from qiskit.result import marginal_counts

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
        """
        Set the ansatz for the VQE problem.
        args:
            ansatz (Ansatze): The ansatz to use
            params (list): The parameters to use for the ansatz
            num_layers (int): The number of layers to use for the ansatz        
        """
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
            """
            Prepare the circuit for the VQE problem.
            args:   
                ansatz (Ansatze): The ansatz to use
                hamiltonian (qml.Hamiltonian): The Hamiltonian to use
                params (list): The parameters to use for the ansatz
            """
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
        """
        Get the number of parameters for the given ansatz.

        args:
            ansatz (Ansatze): The ansatz to get the number of parameters for
        """
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

    def run_iteration(self, store_data=False, data_file=None):
        """
        Run an iteration of the VQE problem.
        """
        assert self.hamiltonian_ops is not None and len(
            self.hamiltonian_ops) > 0, "Hamiltonian operators must be generated before running the VQE"

        if self.optimizer == Optimizers.NELDER_MEAD:
            num_param_sets = 1
        elif self.optimizer == Optimizers.MONTE_CARLO:
            num_param_sets = 3

        if self.current_iteration == 0:
            for ansatz in self.ansatze:
                num_params = self._get_num_params_for_ansatz(ansatz)
                self.params[ansatz] = [npp.random.uniform(
                    0, 2*np.pi, num_params) for _ in range(num_param_sets)]

        self._generate_circuits()

        if self.qoro_service is not None:
            job_circuits = {}
            for circuits in self.circuits.values():
                for circuit in circuits:
                    job_circuits[circuit.tag] = circuit.qasm_circuit
            job_id = self.qoro_service.send_circuits(
                job_circuits, shots=self.shots)
            self.job_id = job_id if job_id is not None else None
        if store_data:
            self.save_iteration(data_file)

    def post_process_results(self):
        """
        Post-process the results of the VQE problem.

        return:
            (dict) The energies for each bond length, ansatz, and parameter set grouping.
        """
        def process_results(results):
            processed_results = {}
            for r in results:
                processed_results[r["label"]] = r["results"]
            return processed_results

        def expectation_value(results):
            eigenvalue = 0
            total_shots = 0

            for key, val in results.items():
                if key.count("1") % 2 == 1:
                    eigenvalue += -val
                else:
                    eigenvalue += val
                total_shots += val

            return eigenvalue / total_shots

        status = self.qoro_service.job_status(
            self.job_id, loop_until_complete=True)
        if status != JobStatus.COMPLETED:
            raise Exception(
                "Job has not completed yet, cannot post-process results")
        results = self.qoro_service.get_job_results(self.job_id)
        results = process_results(results)
        energies = {}
        for i, _ in enumerate(self.bond_lengths):
            energies[i] = {}
            for ansatz in self.ansatze:
                energies[i][ansatz] = {}
                for p, _ in enumerate(self.params[ansatz]):
                    energies[i][ansatz][p] = 0
                    cur_result = {key: value for key, value in results.items(
                    ) if key.startswith(f"{i}_{ansatz.value}_{p}")}
                    marginal_results = []
                    for c in cur_result.keys():
                        ham_op_index = int(c.split("_")[-1])
                        ham_op = self.hamiltonian_ops[i][ham_op_index]
                        pair = (ham_op, cur_result[c], marginal_counts(
                            cur_result[c], ham_op.wires.tolist()))
                        marginal_results.append(pair)
                    for result in marginal_results:
                        energies[i][ansatz][p] += float(result[0].scalar) * \
                            expectation_value(result[2])
        return energies

    def save_iteration(self, data_file):
        """
        Save the current iteration of the VQE problem to a file.

        args:
            data_file (str): The file to save the iteration to.
        """
        import pickle
        with open(data_file, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def import_iteration(data_file):
        """
        Import an iteration of the VQE problem from a file.

        args:
            data_file (str): The file to import the iteration from.
        """
        import pickle
        with open(data_file, "rb") as f:
            return pickle.load(f)

    def visualize_results(self):
        """
        Visualize the results of the VQE problem.
        """
        pass


if __name__ == "__main__":
    from qoro_service import QoroService

    q_service = QoroService("6a539a765fe0b20f409b3c0bbd5d46875598f230")
    vqe_problem = VQE(symbols=["H", "H"],
                      bond_lengths=[0.5, 1.0],
                      coordinate_structure=[(-1, -1, 0), (-1, 0.5, 0)],
                      ansatze=[Ansatze.RYRZ],
                      optimizer=Optimizers.MONTE_CARLO,
                      qoro_service=q_service)
    vqe_problem.run_iteration()
    energies = vqe_problem.post_process_results()
    print(energies)

    c = 0
    for circuits in vqe_problem.circuits.values():
        for circuit in circuits:
            c += 1

    print(f"Total circuits: {c}")
