# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from warnings import warn

import pennylane as qml
import sympy as sp

from divi.circuits import MetaCircuit
from divi.qprog import QuantumProgram
from divi.qprog.algorithms._ansatze import Ansatz
from divi.qprog.optimizers import MonteCarloOptimizer, Optimizer


class VQE(QuantumProgram):
    def __init__(
        self,
        hamiltonian: qml.operation.Operator | None = None,
        molecule: qml.qchem.Molecule | None = None,
        n_electrons: int | None = None,
        n_layers: int = 1,
        ansatz: Ansatz = None,
        optimizer: Optimizer | None = None,
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

        if not ansatz:
            raise ValueError("An ansatz must be provided.")

        # Local Variables
        self.n_layers = n_layers
        self.results = {}
        self.ansatz = ansatz
        self.max_iterations = max_iterations
        self.current_iteration = 0

        self.optimizer = optimizer if optimizer is not None else MonteCarloOptimizer()

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

        self.loss_constant = float(
            sum(map(lambda x: hamiltonian[x].scalar, constant_terms_idx))
        )

        for idx in constant_terms_idx:
            hamiltonian -= hamiltonian[idx]

        return hamiltonian.simplify()

    def _create_meta_circuits_dict(self) -> dict[str, MetaCircuit]:
        weights_syms = sp.symarray("w", (self.n_layers, self.n_params))

        def _prepare_circuit(hamiltonian, params):
            """
            Prepare the circuit for the VQE problem.
            Args:
                ansatz (Ansatze): The ansatz to use
                hamiltonian (qml.Hamiltonian): The Hamiltonian to use
                params (list): The parameters to use for the ansatz
            """
            self.ansatz.build(
                params,
                n_qubits=self.n_qubits,
                n_layers=self.n_layers,
                n_electrons=self.n_electrons,
            )

            # Even though in principle we want to sample from a state,
            # we are applying an `expval` operation here to make it compatible
            # with the pennylane transforms down the line, which complain about
            # the `sample` operation.
            return qml.expval(hamiltonian)

        return {
            "cost_circuit": self._meta_circuit_factory(
                qml.tape.make_qscript(_prepare_circuit)(
                    self.cost_hamiltonian, weights_syms
                ),
                symbols=weights_syms.flatten(),
            )
        }

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
