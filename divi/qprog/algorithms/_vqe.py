# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from warnings import warn

import numpy as np
import numpy.typing as npt
import pennylane as qml
import sympy as sp

from divi.circuits import MetaCircuit
from divi.hamiltonians import _clean_hamiltonian, _is_empty_hamiltonian
from divi.pipeline.stages import CircuitSpecStage
from divi.qprog.algorithms._ansatze import Ansatz, HartreeFockAnsatz
from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm


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

        # Build pipelines once (structure is fixed; only env changes per call)

        self._build_pipelines()

    def _build_pipelines(self) -> None:
        self._cost_pipeline = self._build_cost_pipeline(CircuitSpecStage())
        self._measurement_pipeline = self._build_measurement_pipeline()

    @property
    def n_params_per_layer(self):
        """Get the number of trainable parameters per ansatz layer.

        Returns:
            int: Parameters per layer for the current ansatz and qubit count.
        """
        return self.ansatz.n_params_per_layer(
            self.n_qubits, n_electrons=self.n_electrons
        )

    @property
    def eigenstate(self) -> npt.NDArray[np.int32] | None:
        """Get the computed eigenstate as a NumPy array.

        Returns:
            npt.NDArray[np.int32] | None: The array of bits of the lowest energy eigenstate,
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

        self._cost_hamiltonian, self.loss_constant = _clean_hamiltonian(hamiltonian)
        if _is_empty_hamiltonian(self._cost_hamiltonian):
            raise ValueError("Hamiltonian contains only constant terms.")

    def _create_meta_circuit_factories(self) -> dict[str, MetaCircuit]:
        """Create the meta-circuit factories for VQE.

        Returns:
            dict[str, MetaCircuit]: Dictionary containing cost and measurement circuit templates.
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

        ops = self.ansatz.build(
            weights_syms,
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            n_electrons=self.n_electrons,
        )

        symbols = weights_syms.flatten()
        return {
            "cost_circuit": MetaCircuit(
                source_circuit=qml.tape.QuantumScript(
                    ops=ops, measurements=[qml.expval(self._cost_hamiltonian)]
                ),
                symbols=symbols,
                precision=self._precision,
            ),
            "meas_circuit": MetaCircuit(
                source_circuit=qml.tape.QuantumScript(
                    ops=ops, measurements=[qml.probs()]
                ),
                symbols=symbols,
                precision=self._precision,
            ),
        }

    def _perform_final_computation(self, **kwargs):
        """Extract the eigenstate corresponding to the lowest energy found."""
        self.reporter.info(message="🏁 Computing Final Eigenstate 🏁", overwrite=True)

        self._run_solution_measurement()

        if self._best_probs:
            best_measurement_probs = next(iter(self._best_probs.values()))
            eigenstate_bitstring = max(
                best_measurement_probs, key=best_measurement_probs.get
            )
            self._eigenstate = np.fromiter(eigenstate_bitstring, dtype=np.int32)

        self.reporter.info(message="🏁 Computed Final Eigenstate! 🏁")

        return self._total_circuit_count, self._total_run_time

    def _save_subclass_state(self) -> dict[str, Any]:
        """Save VQE-specific runtime state."""
        return {
            "eigenstate": (
                self._eigenstate.tolist() if self._eigenstate is not None else None
            ),
        }

    def _load_subclass_state(self, state: dict[str, Any]) -> None:
        """Load VQE-specific state.

        Raises:
            KeyError: If any required state key is missing (indicates checkpoint corruption).
        """
        required_keys = ["eigenstate"]
        missing_keys = [key for key in required_keys if key not in state]
        if missing_keys:
            raise KeyError(
                f"Corrupted checkpoint: missing required state keys: {missing_keys}"
            )

        # eigenstate can be None (if not computed yet), but the key must exist
        eigenstate_list = state["eigenstate"]
        if eigenstate_list is not None:
            self._eigenstate = np.array(eigenstate_list, dtype=np.int32)
        else:
            self._eigenstate = None
