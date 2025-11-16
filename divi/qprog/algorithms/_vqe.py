# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from warnings import warn

import numpy as np
import pennylane as qml
import sympy as sp

from divi.circuits import CircuitBundle, MetaCircuit
from divi.qprog.algorithms._ansatze import Ansatz, HartreeFockAnsatz
from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm
from divi.utils import clean_hamiltonian


class VQE(VariationalQuantumAlgorithm):
    """Variational Quantum Eigensolver (VQE) implementation.

    VQE is a hybrid quantum-classical algorithm used to find the ground state
    energy of a given Hamiltonian. It works by preparing a parameterized quantum
    state (ansatz) and optimizing the parameters to minimize the expectation
    value of the Hamiltonian.

    The algorithm can work with either:
    - A molecular Hamiltonian (for quantum chemistry problems)
    - A custom Hamiltonian operator

    Attributes:
        ansatz (Ansatz): The parameterized quantum circuit ansatz.
        n_layers (int): Number of ansatz layers.
        n_qubits (int): Number of qubits in the system.
        n_electrons (int): Number of electrons (for molecular systems).
        cost_hamiltonian (qml.operation.Operator): The Hamiltonian to minimize.
        loss_constant (float): Constant term extracted from the Hamiltonian.
        molecule (qml.qchem.Molecule): The molecule object (if applicable).
        optimizer (Optimizer): Classical optimizer for parameter updates.
        max_iterations (int): Maximum number of optimization iterations.
        current_iteration (int): Current optimization iteration.
    """

    def __init__(
        self,
        hamiltonian: qml.operation.Operator | None = None,
        molecule: qml.qchem.Molecule | None = None,
        n_electrons: int | None = None,
        n_layers: int = 1,
        ansatz: Ansatz | None = None,
        max_iterations=10,
        **kwargs,
    ) -> None:
        """Initialize the VQE problem.

        Args:
            hamiltonian (qml.operation.Operator | None): A Hamiltonian representing the problem. Defaults to None.
            molecule (qml.qchem.Molecule | None): The molecule representing the problem. Defaults to None.
            n_electrons (int | None): Number of electrons associated with the Hamiltonian.
                Only needed when a Hamiltonian is given. Defaults to None.
            n_layers (int): Number of ansatz layers. Defaults to 1.
            ansatz (Ansatz | None): The ansatz to use for the VQE problem.
                Defaults to HartreeFockAnsatz.
            max_iterations (int): Maximum number of optimization iterations. Defaults to 10.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(**kwargs)

        self.ansatz = HartreeFockAnsatz() if ansatz is None else ansatz
        self.n_layers = n_layers
        self.results = {}
        self.max_iterations = max_iterations
        self.current_iteration = 0

        self._eigenstate = None

        self._process_problem_input(
            hamiltonian=hamiltonian, molecule=molecule, n_electrons=n_electrons
        )

    @property
    def cost_hamiltonian(self) -> qml.operation.Operator:
        """The cost Hamiltonian for the VQE problem."""
        return self._cost_hamiltonian

    @property
    def n_params(self):
        """Get the total number of parameters for the VQE ansatz.

        Returns:
            int: Total number of parameters (n_params_per_layer * n_layers).
        """
        return (
            self.ansatz.n_params_per_layer(self.n_qubits, n_electrons=self.n_electrons)
            * self.n_layers
        )

    @property
    def eigenstate(self) -> np.ndarray | None:
        """Get the computed eigenstate as a NumPy array.

        Returns:
            np.ndarray | None: The array of bits of the lowest energy eigenstate,
                or None if not computed.
        """
        return self._eigenstate

    def _process_problem_input(self, hamiltonian, molecule, n_electrons):
        """Process and validate the VQE problem input.

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

        self._cost_hamiltonian, self.loss_constant = clean_hamiltonian(hamiltonian)
        if not self._cost_hamiltonian.operands:
            raise ValueError("Hamiltonian contains only constant terms.")

    def _create_meta_circuits_dict(self) -> dict[str, MetaCircuit]:
        """Create the meta-circuit dictionary for VQE.

        Returns:
            dict[str, MetaCircuit]: Dictionary containing the cost circuit template.
        """
        weights_syms = sp.symarray(
            "w",
            (
                self.n_layers,
                self.ansatz.n_params_per_layer(
                    self.n_qubits, n_electrons=self.n_electrons
                ),
            ),
        )

        def _prepare_circuit(hamiltonian, params, final_measurement=False):
            """Prepare the circuit for the VQE problem.

            Args:
                hamiltonian: The Hamiltonian to measure.
                params: The parameters for the ansatz.
                final_measurement (bool): Whether to perform final measurement.
            """
            self.ansatz.build(
                params,
                n_qubits=self.n_qubits,
                n_layers=self.n_layers,
                n_electrons=self.n_electrons,
            )

            if final_measurement:
                return qml.probs()

            # Even though in principle we want to sample from a state,
            # we are applying an `expval` operation here to make it compatible
            # with the pennylane transforms down the line, which complain about
            # the `sample` operation.
            return qml.expval(hamiltonian)

        return {
            "cost_circuit": self._meta_circuit_factory(
                source_circuit=qml.tape.make_qscript(_prepare_circuit)(
                    self._cost_hamiltonian, weights_syms, final_measurement=False
                ),
                symbols=weights_syms.flatten(),
            ),
            "meas_circuit": self._meta_circuit_factory(
                source_circuit=qml.tape.make_qscript(_prepare_circuit)(
                    self._cost_hamiltonian, weights_syms, final_measurement=True
                ),
                symbols=weights_syms.flatten(),
                grouping_strategy="wires",
            ),
        }

    def _generate_circuits(self) -> list[CircuitBundle]:
        """Generate the circuits for the VQE problem.

        Generates circuits for each parameter set in the current parameters.
        Each circuit is tagged with its parameter index for result processing.

        Returns:
            list[CircuitBundle]: List of CircuitBundle objects for execution.
        """
        circuit_type = (
            "cost_circuit" if not self._is_compute_probabilities else "meas_circuit"
        )

        return [
            self.meta_circuits[circuit_type].initialize_circuit_from_params(
                params_group, tag_prefix=f"{p}"
            )
            for p, params_group in enumerate(self._curr_params)
        ]

    def _post_process_results(self, results, **kwargs):
        """Post-process the results of the VQE problem.

        Args:
            results (dict[str, dict[str, int]]): The shot histograms of the quantum execution step.
            **kwargs: Additional keyword arguments.
                ham_ops (str): The Hamiltonian operators to measure, semicolon-separated.
                    Only needed when the backend supports expval.
        """
        if self._is_compute_probabilities:
            return self._process_probability_results(results)

        return super()._post_process_results(results, **kwargs)

    def _perform_final_computation(self, **kwargs):
        """Extract the eigenstate corresponding to the lowest energy found."""
        self.reporter.info(message="🏁 Computing Final Eigenstate 🏁\r")

        self._run_solution_measurement()

        if self._best_probs:
            best_measurement_probs = next(iter(self._best_probs.values()))
            eigenstate_bitstring = max(
                best_measurement_probs, key=best_measurement_probs.get
            )
            self._eigenstate = np.fromiter(eigenstate_bitstring, dtype=np.int32)

        self.reporter.info(message="🏁 Computed Final Eigenstate! 🏁\r\n")

        return self._total_circuit_count, self._total_run_time
