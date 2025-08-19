# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from warnings import warn

import pennylane as qml
import sympy as sp

from divi.circuits import MetaCircuit
from divi.qprog import QuantumProgram
from divi.qprog.optimizers import Optimizer


class VQEAnsatz(Enum):
    UCCSD = "UCCSD"
    RY = "RY"
    RYRZ = "RYRZ"
    HW_EFFICIENT = "HW_EFFICIENT"
    QAOA = "QAOA"
    HARTREE_FOCK = "HF"

    def describe(self):
        return self.name, self.value

    def n_params(self, n_qubits, **kwargs):
        if self in (VQEAnsatz.UCCSD, VQEAnsatz.HARTREE_FOCK):
            singles, doubles = qml.qchem.excitations(
                kwargs.pop("n_electrons"), n_qubits
            )
            s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)

            return len(s_wires) + len(d_wires)
        elif self == VQEAnsatz.RY:
            return n_qubits
        elif self == VQEAnsatz.RYRZ:
            return 2 * n_qubits
        elif self == VQEAnsatz.HW_EFFICIENT:
            raise NotImplementedError
        elif self == VQEAnsatz.QAOA:
            return qml.QAOAEmbedding.shape(n_layers=1, n_wires=n_qubits)[1]


class VQE(QuantumProgram):
    def __init__(
        self,
        hamiltonian: qml.operation.Operator | None = None,
        molecule: qml.qchem.Molecule | None = None,
        n_electrons: int | None = None,
        n_layers: int = 1,
        ansatz=VQEAnsatz.HARTREE_FOCK,
        optimizer=Optimizer.MONTE_CARLO,
        max_iterations=10,
        **kwargs,
    ) -> None:
        """
        Initialize the VQE problem.

        Args:
            hamiltonain (pennylane.operation.Operator, optional): A Hamiltonian representing the problem.
            molecule (pennylane.qchem.Molecule, optional): The molecule representing the problem.
            n_electrons (int, optional): Number of electrons associated with the Hamiltonian.
                Only needs to be provided when a Hamiltonian is given.
            ansatz (VQEAnsatz): The ansatz to use for the VQE problem
            optimizer (Optimizers): The optimizer to use.
            max_iterations (int): Maximum number of iteration optimizers.
        """

        # Local Variables
        self.n_layers = n_layers
        self.results = {}
        self.ansatz = ansatz
        self.optimizer = optimizer
        self.max_iterations = max_iterations
        self.current_iteration = 0

        self._process_problem_input(
            hamiltonian=hamiltonian, molecule=molecule, n_electrons=n_electrons
        )

        super().__init__(**kwargs)

        self._meta_circuits = self._create_meta_circuits_dict()

    def _process_problem_input(self, hamiltonian, molecule, n_electrons):
        if hamiltonian is None and molecule is None:
            raise ValueError(
                "Either one of `molecule` and `hamiltonian` must be provided."
            )

        if hamiltonian is not None:
            if not isinstance(n_electrons, int) or n_electrons < 0:
                raise ValueError(
                    f"`n_electrons` is expected to be a non-negative integer. Got {n_electrons}."
                )

            self.n_electrons = n_electrons
            self.n_qubits = len(hamiltonian.wires)

        if molecule is not None:
            self.molecule = molecule
            hamiltonian, self.n_qubits = qml.qchem.molecular_hamiltonian(molecule)
            self.n_electrons = molecule.n_electrons

            if (n_electrons is not None) and self.n_electrons != n_electrons:
                warn(
                    "`n_electrons` is provided but not consistent with the molecule's. "
                    f"Got {n_electrons}, but molecule has {self.n_electrons}. "
                    "The molecular value will be used.",
                    UserWarning,
                )

        self.n_params = self.ansatz.n_params(
            self.n_qubits, n_electrons=self.n_electrons
        )

        self.cost_hamiltonian = self._clean_hamiltonian(hamiltonian)

    def _clean_hamiltonian(
        self, hamiltonian: qml.operation.Operator
    ) -> qml.operation.Operator:
        """
        Extracts the scalar from the Hamiltonian, and stores it in
        the `loss_constant` variable.

        Returns:
            The Hamiltonian without the scalar component.
        """

        constant_terms_idx = list(
            filter(
                lambda x: all(
                    isinstance(term, qml.I) for term in hamiltonian[x].terms()[1]
                ),
                range(len(hamiltonian)),
            )
        )

        self.loss_constant = sum(
            map(lambda x: hamiltonian[x].scalar, constant_terms_idx)
        )

        for idx in constant_terms_idx:
            hamiltonian -= hamiltonian[idx]

        return hamiltonian.simplify()

    def _create_meta_circuits_dict(self) -> dict[str, MetaCircuit]:
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

            # Even though in principle we want to sample from a state,
            # we are applying an `expval` operation here to make it compatible
            # with the pennylane transforms down the line, which complain about
            # the `sample` operation.
            return qml.expval(hamiltonian)

        return {
            "cost_circuit": self._meta_circuit_factory(
                qml.tape.make_qscript(_prepare_circuit)(
                    self.ansatz, self.cost_hamiltonian, weights_syms
                ),
                symbols=weights_syms.flatten(),
            )
        }

    def _set_ansatz(self, ansatz: VQEAnsatz, params):
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

        if ansatz in VQEAnsatz:
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
        if self.cost_hamiltonian is None or len(self.cost_hamiltonian) == 0:
            raise RuntimeError(
                "Hamiltonian operators must be generated before running the VQE"
            )

        return super()._run_optimization_circuits(store_data, data_file)
