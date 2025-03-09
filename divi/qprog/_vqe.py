from enum import Enum

import numpy as np
import pennylane as qml
import sympy as sp

from divi.circuit import MetaCircuit
from divi.qprog import QuantumProgram
from divi.qprog.optimizers import Optimizers


class VQEAnsatze(Enum):
    UCCSD = "UCCSD"
    RY = "RY"
    RYRZ = "RYRZ"
    HW_EFFICIENT = "HW_EFFICIENT"
    QAOA = "QAOA"
    HARTREE_FOCK = "HF"

    def describe(self):
        return self.name, self.value

    def n_params(self, n_qubits, **kwargs):
        if self in (VQEAnsatze.UCCSD, VQEAnsatze.HARTREE_FOCK):
            singles, doubles = qml.qchem.excitations(
                kwargs.pop("n_electrons"), n_qubits
            )
            s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

            return len(s_wires) + len(d_wires)
        elif self == VQEAnsatze.RY:
            return n_qubits
        elif self == VQEAnsatze.RYRZ:
            return 2 * n_qubits
        elif self == VQEAnsatze.HW_EFFICIENT:
            raise NotImplementedError
        elif self == VQEAnsatze.QAOA:
            return qml.QAOAEmbedding.shape(n_layers=1, n_wires=n_qubits)[1]


class VQE(QuantumProgram):
    def __init__(
        self,
        symbols,
        bond_length: float,
        coordinate_structure: list[tuple[float, float, float]],
        n_layers: int = 1,
        optimizer=Optimizers.MONTE_CARLO,
        ansatz=VQEAnsatze.HARTREE_FOCK,
        max_iterations=10,
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
        self.coordinate_structure = coordinate_structure

        if len(self.coordinate_structure) != len(self.symbols):
            raise ValueError(
                "The number of symbols must match the number of coordinates"
            )

        self.bond_length = bond_length
        self.n_layers = n_layers
        self.results = {}
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.max_iterations = max_iterations
        self.current_iteration = 0

        # Shared Variables
        self.losses = []
        if (m_list := kwargs.pop("losses", None)) is not None:
            self.losses = m_list

        self.hamiltonian = self._generate_hamiltonian_operations()

        self.expval_hamiltonian_metadata = {
            i: (term.wires, float(term.scalar))
            for i, term in enumerate(self.hamiltonian)
        }

        self._meta_circuits = self._create_meta_circuits()

        super().__init__(**kwargs)

    def _generate_hamiltonian_operations(self) -> qml.operation.Operator:
        """
        Generate the Hamiltonian operators for the given bond length.

        Returns:
            The Hamiltonian corresponding to the VQE problem.
        """

        coordinates = [
            (
                coord_0 * self.bond_length,
                coord_1 * self.bond_length,
                coord_2 * self.bond_length,
            )
            for (coord_0, coord_1, coord_2) in self.coordinate_structure
        ]

        coordinates = np.array(coordinates)
        molecule = qml.qchem.Molecule(self.symbols, coordinates)
        hamiltonian, qubits = qml.qchem.molecular_hamiltonian(molecule)

        self.n_qubits = qubits
        self.n_electrons = molecule.n_electrons

        self.n_params = self.ansatz.n_params(
            self.n_qubits, n_electrons=self.n_electrons
        )

        return hamiltonian

    def _create_meta_circuits(self):
        weights_syms = sp.symarray("w", (self.n_layers, self.n_params))

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

        return {
            "cost_circuit": MetaCircuit(
                qml.tape.make_qscript(_prepare_circuit)(
                    self.ansatz, self.hamiltonian, weights_syms
                ),
                symbols=weights_syms.flatten(),
            )
        }

    def _set_ansatz(self, ansatz: VQEAnsatze, params):
        """
        Set the ansatz for the VQE problem.
        Args:
            ansatz (Ansatze): The ansatz to use
            params (list): The parameters to use for the ansatz
            n_layers (int): The number of layers to use for the ansatz
        """

        def _add_hw_efficient_ansatz(params):
            raise NotImplementedError

        def _add_qaoa_ansatz(params):
            # This infers layers automatically from the parameters shape
            qml.QAOAEmbedding(
                features=[],
                weights=params.reshape(self.n_layers, -1),
                wires=range(self.n_qubits),
            )

        def _add_ry_ansatz(params):
            qml.layer(
                qml.AngleEmbedding,
                self.n_layers,
                params.reshape(self.n_layers, -1),
                wires=range(self.n_qubits),
                rotation="Y",
            )

        def _add_ryrz_ansatz(params):
            def _ryrz(params, wires):
                ry_rots, rz_rots = params.reshape(2, -1)
                qml.AngleEmbedding(ry_rots, wires=wires, rotation="Y")
                qml.AngleEmbedding(rz_rots, wires=wires, rotation="Z")

            qml.layer(
                _ryrz,
                self.n_layers,
                params.reshape(self.n_layers, -1),
                wires=range(self.n_qubits),
            )

        def _add_uccsd_ansatz(params):
            hf_state = qml.qchem.hf_state(self.n_electrons, self.n_qubits)

            singles, doubles = qml.qchem.excitations(self.n_electrons, self.n_qubits)
            s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

            qml.UCCSD(
                params.reshape(self.n_layers, -1),
                wires=range(self.n_qubits),
                s_wires=s_wires,
                d_wires=d_wires,
                init_state=hf_state,
                n_repeats=self.n_layers,
            )

        def _add_hartree_fock_ansatz(params):
            singles, doubles = qml.qchem.excitations(self.n_electrons, self.n_qubits)
            hf_state = qml.qchem.hf_state(self.n_electrons, self.n_qubits)

            qml.layer(
                qml.AllSinglesDoubles,
                self.n_layers,
                params.reshape(self.n_layers, -1),
                wires=range(self.n_qubits),
                hf_state=hf_state,
                singles=singles,
                doubles=doubles,
            )

            # Reset the BasisState operations after the first layer
            # for behaviour similar to UCCSD ansatz
            for op in qml.QueuingManager.active_context().queue[1:]:
                op._hyperparameters["hf_state"] = 0

        if ansatz in VQEAnsatze:
            locals()[f"_add_{ansatz.name.lower()}_ansatz"](params)
        else:
            raise ValueError(f"Invalid Ansatz Value. Got {ansatz}.")

    def _generate_circuits(self):
        """
        Generate the circuits for the VQE problem.

        In this method, we generate bulk circuits based on the selected parameters.
        We generate circuits for each bond length and each ansatz and optimization choice.

        The structure of the circuits is as follows:
        - For each bond length:
            - For each ansatz:
                - Generate the circuit
        """

        for p, params_group in enumerate(self._curr_params):
            circuit = self._meta_circuits[
                "cost_circuit"
            ].initialize_circuit_from_params(params_group, tag_prefix=f"{p}")

            self.circuits.append(circuit)

    def _run_optimization_circuits(self, store_data, data_file):
        if self.hamiltonian is None or len(self.hamiltonian) == 0:
            raise RuntimeError(
                "Hamiltonian operators must be generated before running the VQE"
            )

        return super()._run_optimization_circuits(store_data, data_file)
