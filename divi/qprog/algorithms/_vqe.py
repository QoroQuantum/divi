# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from warnings import warn

import pennylane as qml
import sympy as sp

from divi.circuits import MetaCircuit
from divi.qprog import QuantumProgram
from divi.qprog.algorithms._ansatze import Ansatz, HartreeFockAnsatz
from divi.qprog.optimizers import MonteCarloOptimizer, Optimizer


class VQE(QuantumProgram):
    def __init__(
        self,
        hamiltonian: qml.operation.Operator | None = None,
        molecule: qml.qchem.Molecule | None = None,
        n_electrons: int | None = None,
        n_layers: int = 1,
        ansatz: Ansatz | None = None,
        optimizer: Optimizer | None = None,
        max_iterations=10,
        **kwargs,
    ) -> None:
        """
        Initialize the VQE problem.

        Args:
            hamiltonian (pennylane.operation.Operator, optional): A Hamiltonian
                representing the problem.
            molecule (pennylane.qchem.Molecule, optional): The molecule representing
                the problem.
            n_electrons (int, optional): Number of electrons associated with the
                Hamiltonian. Only needs to be provided when a Hamiltonian is given.
            n_layers (int, optional): Number of ansatz layers. Defaults to 1.
            ansatz (Ansatz, optional): The ansatz to use for the VQE problem.
                Defaults to HartreeFockAnsatz.
            optimizer (Optimizer, optional): The optimizer to use. Defaults to
                MonteCarloOptimizer.
            max_iterations (int, optional): Maximum number of optimization iterations.
                Defaults to 10.
            **kwargs: Additional keyword arguments passed to the parent QuantumProgram.
        """

        # Local Variables
        self.ansatz = HartreeFockAnsatz() if ansatz is None else ansatz
        self.n_layers = n_layers
        self.results = {}
        self.max_iterations = max_iterations
        self.current_iteration = 0

        self.optimizer = optimizer if optimizer is not None else MonteCarloOptimizer()

        self._process_problem_input(
            hamiltonian=hamiltonian, molecule=molecule, n_electrons=n_electrons
        )

        super().__init__(**kwargs)

        self._meta_circuits = self._create_meta_circuits_dict()

    @property
    def n_params(self):
        """
        Get the total number of parameters for the VQE ansatz.

        Returns:
            int: Total number of parameters (n_params_per_layer * n_layers).
        """
        return (
            self.ansatz.n_params_per_layer(self.n_qubits, n_electrons=self.n_electrons)
            * self.n_layers
        )

    def _process_problem_input(self, hamiltonian, molecule, n_electrons):
        """
        Process and validate the VQE problem input.

        Handles both Hamiltonian-based and molecule-based problem specifications,
        extracting the necessary information (n_qubits, n_electrons, hamiltonian).

        Args:
            hamiltonian: PennyLane Hamiltonian operator or None.
            molecule: PennyLane Molecule object or None.
            n_electrons: Number of electrons or None.

        Raises:
            ValueError: If neither hamiltonian nor molecule is provided.
            UserWarning: If n_electrons conflicts with the molecule's electron count.
        """
        if hamiltonian is None and molecule is None:
            raise ValueError(
                "Either one of `molecule` and `hamiltonian` must be provided."
            )

        if hamiltonian is not None:
            self.n_qubits = len(hamiltonian.wires)
            self.n_electrons = n_electrons

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
        weights_syms = sp.symarray(
            "w",
            (
                self.n_layers,
                self.ansatz.n_params_per_layer(
                    self.n_qubits, n_electrons=self.n_electrons
                ),
            ),
        )

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

            self._circuits.append(circuit)

    def _run_optimization_circuits(self, store_data, data_file):
        """
        Execute the circuits for the current optimization iteration.

        Validates that the Hamiltonian is properly set before running circuits.

        Args:
            store_data (bool): Whether to save iteration data.
            data_file (str): Path to file for saving data.

        Returns:
            dict: Loss values for each parameter set.

        Raises:
            RuntimeError: If the cost Hamiltonian is not set or empty.
        """
        if self.cost_hamiltonian is None or len(self.cost_hamiltonian) == 0:
            raise RuntimeError(
                "Hamiltonian operators must be generated before running the VQE"
            )

        return super()._run_optimization_circuits(store_data, data_file)
