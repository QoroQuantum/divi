import logging
import warnings
from enum import Enum

import numpy as np
import pennylane as qml
from qiskit.result import marginal_counts, sampled_expectation_value
from scipy.optimize import minimize

from divi.circuit import Circuit
from divi.qprog import QuantumProgram
from divi.qprog.optimizers import Optimizers
from divi.services.qoro_service import JobStatus

try:
    import openfermionpyscf
except ImportError:
    warnings.warn("openfermionpyscf not installed. Some functionality may be limited.")

warnings.filterwarnings("ignore", category=UserWarning)

# Set up your logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(ch)

# Suppress debug logs from external libraries
logging.getLogger().setLevel(logging.WARNING)


class VQEAnsatze(Enum):
    UCCSD = "UCCSD"
    RY = "RY"
    RYRZ = "RYRZ"
    HW_EFFICIENT = "HW_EFFICIENT"
    LAYERED = "LAYERED"
    QAOA = "QAOA"
    HARTREE_FOCK = "HF"

    def describe(self):
        return self.name, self.value

    def n_params(self, n_qubits, **kwargs):
        if self == VQEAnsatze.UCCSD:
            singles, doubles = qml.qchem.excitations(
                kwargs.pop("n_electrons"), n_qubits
            )
            s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
            return len(s_wires) + len(d_wires)
        elif self == VQEAnsatze.HARTREE_FOCK:
            return 1
        elif self == VQEAnsatze.RY:
            return n_qubits.num_qubits
        elif self == VQEAnsatze.RYRZ:
            return 2 * n_qubits.num_qubits
        elif self == VQEAnsatze.HW_EFFICIENT:
            # TODO
            return 1
        elif self == VQEAnsatze.LAYERED:
            # TODO
            return 1
        elif self == VQEAnsatze.QAOA:
            # TODO
            return 1


class VQE(QuantumProgram):
    def __init__(
        self,
        symbols,
        bond_length,
        coordinate_structure,
        optimizer=Optimizers.MONTE_CARLO,
        ansatz=VQEAnsatze.HARTREE_FOCK,
        max_iterations=10,
        shots=5000,
        **kwargs,
    ) -> None:
        """
        Initialize the VQE problem.

        Args:
            symbols (list): The symbols of the atoms in the molecule
            bond_length (float): The bond length to consider
            coordinate_structure (list): The coordinate structure of the molecule
            ansatz (VQEAnsatz): The ansatz to use for the VQE problem
            optimizer (Optimizers): The optimizer to use.
            max_iterations (int): Maximum number of iteration optimizers.
            shots (int): Number of shots for each circuit execution.
        """

        # Local Variables
        self.symbols = symbols
        self.bond_length = bond_length
        self.n_qubits = 0
        self.n_electrons = 0
        self.results = {}
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.shots = shots
        self.max_iterations = max_iterations
        self.coordinate_structure = coordinate_structure
        self.current_iteration = 0

        # Shared Variables
        self.energies = []
        if (m_list := kwargs.pop("energies", None)) is not None:
            self.energies = m_list

        assert len(self.coordinate_structure) == len(
            self.symbols
        ), "The number of symbols must match the number of coordinates"

        self.hamiltonian_ops = self._generate_hamiltonian_operations()

        super().__init__(**kwargs)

    def _reset_params(self):
        self.params = []

    def _generate_hamiltonian_operations(self):
        """
        Generate the Hamiltonian operators for the given bond lengths.

        Returns:
            (list) Hamiltonians for each bond length.
        """

        def all_equal(iterator):
            iterator = iter(iterator)
            try:
                first = next(iterator)
            except StopIteration:
                return True
            return all(first == x for x in iterator)

        n_qubits = []
        n_electrons = []

        # Generate the Hamiltonian for the given bond length
        coordinates = []
        for coord_str in self.coordinate_structure:
            coordinates.append(
                [
                    coord_str[0] * self.bond_length,
                    coord_str[1] * self.bond_length,
                    coord_str[2] * self.bond_length,
                ]
            )

        coordinates = np.array(coordinates)
        molecule = qml.qchem.Molecule(self.symbols, coordinates)
        hamiltonian, qubits = qml.qchem.molecular_hamiltonian(molecule)

        n_qubits.append(qubits)
        n_electrons.append(molecule.n_electrons)

        assert all_equal(
            n_qubits
        ), "All Hamiltonians must have the same number of qubits"
        assert all_equal(
            n_electrons
        ), "All Hamiltonians must have the same number of electrons"

        self.n_qubits = n_qubits[0]
        self.n_electrons = n_electrons[0]

        return hamiltonian

    def _set_ansatz(self, ansatz, params, num_layers=1):
        """
        Set the ansatz for the VQE problem.
        Args:
            ansatz (Ansatze): The ansatz to use
            params (list): The parameters to use for the ansatz
            num_layers (int): The number of layers to use for the ansatz
        """

        def _add_hw_efficient_ansatz(params, num_layers):
            qml.RX(params[0], wires=[0])

        def _add_ry_ansatz(params, num_layers):
            p = 0
            for _ in range(num_layers):
                for j in range(self.n_qubits):
                    qml.RY(params[p], wires=[j])
                    p += 1

        def _add_layered_ansatz(params, num_layers):
            raise NotImplementedError

        def _add_qaoa_ansatz(params, num_layers):
            raise NotImplementedError

        def _add_ryrz_ansatz(params, num_layers):
            p = 0
            for _ in range(num_layers):
                for j in range(self.n_qubits):
                    qml.RY(params[p], wires=[j])
                    p += 1
                    qml.RZ(params[p], wires=[j])

        def _add_uccsd_ansatz(params, num_layers):
            hf_state = qml.qchem.hf_state(self.n_electrons, self.n_qubits)
            singles, doubles = qml.qchem.excitations(self.n_electrons, self.n_qubits)
            s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
            qml.UCCSD(
                params,
                wires=[i for i in range(self.n_qubits)],
                s_wires=s_wires,
                d_wires=d_wires,
                init_state=hf_state,
            )

        def _add_hartree_fock_ansatz(params, num_layers):
            hf_state = np.array(
                [1 if i < self.n_electrons else 0 for i in range(self.n_qubits)]
            )

            qml.BasisState(hf_state, wires=[i for i in range(self.n_qubits)])
            qml.DoubleExcitation(params[0], wires=range(self.n_qubits))

        if ansatz == VQEAnsatze.UCCSD:
            _add_uccsd_ansatz(params, num_layers)
        elif ansatz == VQEAnsatze.HARTREE_FOCK:
            _add_hartree_fock_ansatz(params, num_layers)
        elif ansatz == VQEAnsatze.RY:
            _add_ry_ansatz(params, num_layers)
        elif ansatz == VQEAnsatze.RYRZ:
            _add_ryrz_ansatz(params, num_layers)
        elif ansatz == VQEAnsatze.HW_EFFICIENT:
            _add_hw_efficient_ansatz(params, num_layers)
        elif ansatz == VQEAnsatze.LAYERED:
            _add_layered_ansatz(params, num_layers)
        elif ansatz == VQEAnsatze.QAOA:
            _add_qaoa_ansatz(params, num_layers)

    def _generate_circuits(self, params=None):
        """
        Generate the circuits for the VQE problem.

        In this method, we generate bulk circuits based on the selected parameters.
        We generate circuits for each bond length and each ansatz and optimization choice.

        The structure of the circuits is as follows:
        - For each bond length:
            - For each ansatz:
                - Generate the circuit
        """

        def _prepare_circuit(ansatz, hamiltonian, params):
            """
            Prepare the circuit for the VQE problem.
            Args:
                ansatz (Ansatze): The ansatz to use
                hamiltonian (qml.Hamiltonian): The Hamiltonian to use
                params (list): The parameters to use for the ansatz
            """
            self._set_ansatz(ansatz, params)

            return [qml.sample(term) for term in hamiltonian]

        params = self.params if params is None else [params]

        for p, params_group in enumerate(params):
            qscript = qml.tape.make_qscript(_prepare_circuit)(
                self.ansatz, self.hamiltonian_ops, params_group
            )
            self.circuits.append(Circuit(qscript, tag_prefix=f"{p}"))

    def run(self, store_data=False, data_file=None):
        """
        Run the VQE problem. The outputs are stored in the VQE object. Optionally, the data can be stored in a file.

        Args:
            store_data (bool): Whether to store the data for the iteration
            data_file (str): The file to store the data in
        """
        if self.optimizer == Optimizers.MONTE_CARLO:
            while self.current_iteration < self.max_iterations:
                if self.hamiltonian_ops is None or len(self.hamiltonian_ops) == 0:
                    raise RuntimeError(
                        "Hamiltonian operators must be generated before running the VQE"
                    )

                logger.debug(f"Running iteration {self.current_iteration}")

                self._run_optimize()

                self._generate_circuits()
                self._dispatch_circuits_and_process_results(
                    store_data=store_data, data_file=data_file
                )

            return self.total_circuit_count

        elif self.optimizer == Optimizers.NELDER_MEAD:

            def cost_function(params):
                self._generate_circuits(params)
                energies = self._dispatch_circuits_and_process_results(
                    store_data=store_data, data_file=data_file
                )

                self.energies.append(energies)
                return energies[0]

            self._reset_params()

            n_params = self.ansatz.n_params(self.n_qubits, n_electrons=self.n_electrons)
            self.params = [
                np.random.uniform(-2 * np.pi, -2 * np.pi, n_params)
                for _ in range(self.optimizer.n_param_sets)
            ]

            minimize(
                cost_function,
                self.params[0],
                method="Nelder-Mead",
                options={"maxiter": self.max_iterations},
            )

            return self.total_circuit_count

    def _run_optimize(self):
        """
        Run the optimization step for the VQE problem.
        """
        n_param_sets = self.optimizer.n_param_sets

        if self.current_iteration == 0:
            self._reset_params()

            num_params = self.ansatz.n_params(
                self.n_qubits, n_electrons=self.n_electrons
            )
            self.params = [
                np.random.uniform(0, 2 * np.pi, num_params) for _ in range(n_param_sets)
            ]
        else:
            # Optimize the VQE problem.
            if self.optimizer == Optimizers.NELDER_MEAD:
                raise NotImplementedError

            elif self.optimizer == Optimizers.MONTE_CARLO:
                self.params = self.optimizer.compute_new_parameters(
                    self.params,
                    self.current_iteration,
                    losses=self.energies[-1],
                )
            else:
                raise NotImplementedError

        self.current_iteration += 1

    def _post_process_results(self, job_id=None, results=None):
        """
        Post-process the results of the VQE problem.

        Returns:
            (dict) The energies for each parameter set grouping.
        """

        def process_results(results):
            processed_results = {}
            for r in results:
                processed_results[r["label"]] = r["results"]
            return processed_results

        if job_id is not None and self.qoro_service is not None:
            status = self.qoro_service.job_status(self.job_id, loop_until_complete=True)
            if status != JobStatus.COMPLETED:
                raise Exception(
                    "Job has not completed yet, cannot post-process results"
                )
            results = self.qoro_service.get_job_results(self.job_id)

        results = process_results(results)
        energies = {}

        for p, _ in enumerate(self.params):
            energies[p] = 0
            cur_result = {
                key: value for key, value in results.items() if key.startswith(f"{p}")
            }

            marginal_results = []
            for c in cur_result.keys():
                ham_op_index = int(c.split("_")[-1])
                ham_op = self.hamiltonian_ops[ham_op_index]
                pair = (
                    ham_op,
                    cur_result[c],
                    marginal_counts(cur_result[c], ham_op.wires.tolist()),
                )
                marginal_results.append(pair)
            for result in marginal_results:
                exp_value = sampled_expectation_value(
                    result[2], "Z" * len(list(result[2].keys())[0])
                )
                energies[p] += float(result[0].scalar) * exp_value

        self.energies.append(energies)

        return energies
