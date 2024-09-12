import pennylane as qml
import numpy as npp
import warnings
import time
import logging

from pennylane import numpy as np
from qprog.quantum_program import QuantumProgram
from circuit import Circuit
from enum import Enum
from qoro_service import JobStatus, JobTypes
from simulator.parallel_simulator import ParallelSimulator
from qiskit.result import marginal_counts
from multiprocessing import Pool
from scipy.optimize import minimize
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import openfermionpyscf
except ImportError:
    warnings.warn(
        "openfermionpyscf not installed. Some functionality may be limited.")

warnings.filterwarnings("ignore", category=UserWarning)

# Set up your logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)

# Suppress debug logs from external libraries
logging.getLogger().setLevel(logging.WARNING)

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

    def num_params(self, num_qubits):
        if self == Ansatze.UCCSD:
            return num_qubits
        elif self == Ansatze.HARTREE_FOCK:
            return 1
        elif self == Ansatze.RY:
            return num_qubits
        elif self == Ansatze.RYRZ:
            return 2 * num_qubits
        elif self == Ansatze.HW_EFFICIENT:
            # TODO
            return 1
        elif self == Ansatze.LAYERED:
            # TODO
            return 1
        elif self == Ansatze.QAOA:
            # TODO
            return 1


class Optimizers(Enum):
    NELDER_MEAD = "Nelder-Mead"
    MONTE_CARLO = "Monte Carlo"

    def describe(self):
        return self.name, self.value

    def num_param_sets(self):
        if self == Optimizers.NELDER_MEAD:
            return 1
        elif self == Optimizers.MONTE_CARLO:
            return 3

    def samples(self):
        if self == Optimizers.MONTE_CARLO:
            return 2
        return 1

    def update_params(self, params, iteration):
        if self == Optimizers.MONTE_CARLO:
            return [npp.random.normal(
                params, 1 / iteration, size=params.shape) for _ in range(self.num_param_sets())]
        else:
            raise NotImplementedError


class VQE(QuantumProgram):
    def __init__(self, symbols, bond_lengths, coordinate_structure, optimizer=Optimizers.NELDER_MEAD, ansatze=(Ansatze.HARTREE_FOCK,),  max_interations=10, shots=5000, *args, **kwargs) -> None:
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
        self.params_list = []
        self.current_iteration = 0
        self.optimizer = optimizer
        self.shots = shots
        self.job_type = JobTypes.EXECUTE
        self.max_iterations = max_interations
        self.energies = []

        self.coordinate_structure = coordinate_structure
        assert len(self.coordinate_structure) == len(
            self.symbols), "The number of symbols must match the number of coordinates"

        self.hamiltonian_ops = self._generate_hamiltonian_operations()

    def _reset_params(self):
        """
        Reset the parameters for the VQE problem.
        """
        self.params = {}
        for i in range(len(self.bond_lengths)):
            self.params[i] = {}
            for ansatz in self.ansatze:
                self.params[i][ansatz] = []

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
            raise NotImplementedError

        def _add_qaoa_ansatz(params, num_layers):
            raise NotImplementedError

        def _add_ryrz_ansatz(params, num_layers):
            p = 0
            for _ in range(num_layers):
                for j in range(self.num_qubits):
                    qml.RY(params[p], wires=[j])
                    p += 1
                    qml.RZ(params[p], wires=[j])

        def _add_uccsd_ansatz(params, num_layers):
            raise NotImplementedError

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

    def _generate_circuits(self, params=None) -> None:
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
        # TODO: This is very slow... Can we paralellize it? Or can we use Qiskit?

        if params is None:
            start = time.time()
            for i, _ in enumerate(self.bond_lengths):
                for ansatz in self.ansatze:
                    self.circuits[(i, ansatz)] = []
                    params_list = self.params[i][ansatz]

                    for p, params in enumerate(params_list):
                        for j, hamiltonian in enumerate(self.hamiltonian_ops[i]):
                            device = qml.device(
                                "qiskit.aer", wires=self.num_qubits, shots=self.shots)
                            # Maybe the two last parameters speed this up, has to be tested though
                            q_node = qml.QNode(_prepare_circuit, device, interface=None, diff_method=None)
                            q_node(ansatz, hamiltonian, params)
                            circuit = Circuit(
                                device, tag=f"{i}_{ansatz.value}_{p}_{j}")
                            self.circuits[(i, ansatz)].append(circuit)
            end = time.time()
            duration = end - start
            logger.debug(f"Execution time: {round(duration, 4)} seconds")
        else:
            for i, _ in enumerate(self.bond_lengths):
                for ansatz in self.ansatze:
                    self.circuits[(i, ansatz)] = []
                    for j, hamiltonian in enumerate(self.hamiltonian_ops[i]):
                        device = qml.device(
                            "qiskit.aer", wires=self.num_qubits, shots=self.shots)
                        q_node = qml.QNode(_prepare_circuit, device)
                        q_node(ansatz, hamiltonian, params)
                        circuit = Circuit(
                            device, tag=f"{i}_{ansatz.value}_{0}_{j}")
                        self.circuits[(i, ansatz)].append(circuit)

    def run(self, store_data=False, data_file=None, type=JobTypes.EXECUTE):
        """
        Run the VQE problem. The outputs are stored in the VQE object. Optionally, the data can be stored in a file.

        args:
            store_data (bool): Whether to store the data for the iteration
            data_file (str): The file to store the data in        
        """

        if self.optimizer == Optimizers.MONTE_CARLO:
            while self.current_iteration < self.max_iterations:
                logger.debug(f"Running iteration {self.current_iteration}")
                self.run_iteration(store_data, data_file, type)
        elif self.optimizer == Optimizers.NELDER_MEAD:
            def cost_function(params, bond_length_index, ansatz):
                self.params[bond_length_index][ansatz] = params
                self._generate_circuits(params)
                results, param = self._prepare_and_send_circuits()
                if param == 'job_id':
                    energies = self._post_process_results(job_id=results)
                elif param == 'circuit_results':
                    energies = self._post_process_results(results=results)
                self.energies.append(energies)
                return energies[bond_length_index][ansatz][0]

            def optimize_single(args):
                i, ansatz = args
                logger.debug(
                    'Running optimization for bond length:', i, ansatz)
                params = self.params[i][ansatz][0]
                result = minimize(cost_function, params, args=(
                    i, ansatz), method="Nelder-Mead", options={"maxiter": self.max_iterations})
                return i, ansatz, result.fun

            self._reset_params()
            num_param_sets = 1
            args = []
            energies = {}
            for i in range(len(self.bond_lengths)):
                energies[i] = {}
                for ansatz in self.ansatze:
                    energies[i][ansatz] = {}
                    num_params = ansatz.num_params(self.num_qubits)
                    self.params[i][ansatz] = [npp.random.uniform(
                        0, 2*np.pi, num_params) for _ in range(num_param_sets)]
                    args.append((i, ansatz))

            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(optimize_single, arg)
                           for arg in args]
                for future in futures:
                    i, ansatz, energy = future.result()
                    energies[i][ansatz][0] = energy

            return energies

    def run_iteration(self, store_data=False, data_file=None, type=JobTypes.EXECUTE):
        """
        Run an iteration of the VQE problem. The outputs are stored in the VQE object. Optionally, the data can be stored in a file.

        args:
            store_data (bool): Whether to store the data for the iteration
            data_file (str): The file to store the data in        
        """

        if self.current_iteration == self.max_iterations:
            raise Exception(
                "Maximum number of iterations reached, cannot run another iteration")

        assert self.hamiltonian_ops is not None and len(
            self.hamiltonian_ops) > 0, "Hamiltonian operators must be generated before running the VQE"

        self._run_optimize()
        self.params_list.append(self.params)
        self._generate_circuits()
        results, param = self._prepare_and_send_circuits()
        if param == 'job_id':
            self._post_process_results(job_id=results)
        elif param == 'circuit_results':
            self._post_process_results(results=results)

        if store_data:
            self.save_iteration(data_file)

    def _prepare_and_send_circuits(self):
        job_circuits = {}
        for circuits in self.circuits.values():
            for circuit in circuits:
                job_circuits[circuit.tag] = circuit.qasm_circuit

        if self.qoro_service is not None:
            job_id = self.qoro_service.send_circuits(
                job_circuits, shots=self.shots, job_type=self.job_type)
            self.job_id = job_id if job_id is not None else None
            return job_id, 'job_id'
        else:
            circuit_simulator = ParallelSimulator()
            circuit_results = circuit_simulator.simulate(
                job_circuits, shots=self.shots)
            return circuit_results, 'circuit_results'

    def _optimize(self):
        """
        Optimize the VQE problem.
        """
        if self.optimizer == Optimizers.NELDER_MEAD:
            raise NotImplementedError

        elif self.optimizer == Optimizers.MONTE_CARLO:
            for i, _ in enumerate(self.bond_lengths):
                for ansatz in self.ansatze:
                    energies = self.energies[self.current_iteration - 1][i][ansatz]
                    smallest_energy_keys = sorted(
                        energies, key=lambda k: energies[k])[:self.optimizer.samples()]
                    new_params = []
                    for key in smallest_energy_keys:
                        new_param_set = self.optimizer.update_params(
                            self.params[i][ansatz][int(key)], self.current_iteration)
                        new_params.extend(new_param_set)
                    self.params[i][ansatz] = new_params
        else:
            raise NotImplementedError

    def _run_optimize(self):
        """
        Run the optimization step for the VQE problem.
        """
        num_param_sets = self.optimizer.num_param_sets()
        if self.current_iteration == 0:
            self._reset_params()
            for i in range(len(self.bond_lengths)):
                for ansatz in self.ansatze:
                    num_params = ansatz.num_params(self.num_qubits)
                    self.params[i][ansatz] = [npp.random.uniform(
                        0, 2*np.pi, num_params) for _ in range(num_param_sets)]
        else:
            self._optimize()
        self.current_iteration += 1

    def _post_process_results(self, job_id=None, results=None):
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

        if job_id is not None and self.qoro_service is not None:
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
                for p, _ in enumerate(self.params[i][ansatz]):
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

        self.energies.append(energies)
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
        import matplotlib.pyplot as plt

        data = []
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        ansatz_list = list(Ansatze)
        for _, energies in enumerate(self.energies):
            for i, length in enumerate(self.bond_lengths):
                min_energies = []
                for ansatz in self.ansatze:
                    cur_energies = energies[i][ansatz]
                    min_energies.append(
                        (length, min(cur_energies.values()), colors[ansatz_list.index(ansatz)]))
                data.extend(min_energies)

        x, y, z = zip(*data)
        plt.scatter(x, y, color=z)

        plt.xlabel('Bond length')
        plt.ylabel('Energy level')
        plt.show()


if __name__ == "__main__":
    from qoro_service import QoroService

    # q_service = QoroService("71ec99c9c94cf37499a2b725244beac1f51b8ee4")
    q_service = None
    vqe_problem = VQE(symbols=["H", "H"],
                      bond_lengths=[0.5],
                      coordinate_structure=[(0, 0, -0.5), (0, 0, 0.5)],
                      ansatze=[Ansatze.HARTREE_FOCK],
                      optimizer=Optimizers.MONTE_CARLO,
                      qoro_service=q_service,
                      shots=500,
                      max_interations=4)

    vqe_problem.run()
    energies = vqe_problem.energies[vqe_problem.current_iteration - 1]
    ansatz = vqe_problem.ansatze[0]
    print(energies)
    for i in range(len(vqe_problem.bond_lengths)):
        print(energies[i][ansatz][0])
    # vqe_problem.visualize_results()

    # data = []
    # for energy in vqe_problem.energies:
    #     data.append(energy[Ansatze.HARTREE_FOCK][0])

    c = 0
    for circuits in vqe_problem.circuits.values():
        for circuit in circuits:
            c += 1

    print(f"Total circuits: {c}")
