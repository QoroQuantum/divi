# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from warnings import warn

import numpy as np
import numpy.typing as npt
import pennylane as qp
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.quantum_info import SparsePauliOp

from divi.circuits import MetaCircuit
from divi.hamiltonians._term_ops import (
    _clean_hamiltonian_spo,
    _require_qiskit_num_qubits,
    to_spo,
)
from divi.qprog._solution_sampling_mixin import SolutionSamplingMixin
from divi.qprog.algorithms import (
    Ansatz,
    HartreeFockAnsatz,
    InitialState,
    QCCAnsatz,
    UCCSDAnsatz,
    ZerosState,
)
from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm


class VQE(SolutionSamplingMixin, VariationalQuantumAlgorithm):
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
        cost_hamiltonian: The Hamiltonian to minimize.
        loss_constant (float): Constant term extracted from the Hamiltonian.
        molecule: The molecule object (if applicable).
        optimizer: Classical optimizer for parameter updates.
        max_iterations (int): Maximum number of optimization iterations.
        current_iteration (int): Current optimization iteration.
    """

    def __init__(
        self,
        hamiltonian: qp.operation.Operator | SparsePauliOp | None = None,
        molecule: qp.qchem.Molecule | None = None,
        n_electrons: int | None = None,
        n_layers: int = 1,
        ansatz: Ansatz | None = None,
        initial_state: InitialState | None = None,
        max_iterations=10,
        **kwargs,
    ) -> None:
        """Initialize the VQE problem.

        Args:
            hamiltonian (qp.operation.Operator | None): A Hamiltonian representing the problem. Defaults to None.
            molecule (qp.qchem.Molecule | None): The molecule representing the problem. Defaults to None.
            n_electrons (int | None): Number of electrons associated with the Hamiltonian.
                Only needed when a Hamiltonian is given. Defaults to None.
            n_layers (int): Number of ansatz layers. Defaults to 1.
            ansatz (Ansatz | None): The ansatz to use for the VQE problem.
                Defaults to HartreeFockAnsatz.
            initial_state (InitialState | None): Initial state preparation.
                Pass an :class:`~divi.qprog.algorithms.InitialState` instance (e.g. ``ZerosState()``,
                ``SuperpositionState()``). Defaults to ``ZerosState()`` if None.
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

        # Resolve & store initial state (n_qubits is now set)
        if initial_state is None:
            initial_state = ZerosState()
        self.initial_state = initial_state

        if not isinstance(self.initial_state, ZerosState) and isinstance(
            self.ansatz, (HartreeFockAnsatz, QCCAnsatz, UCCSDAnsatz)
        ):
            warn(
                f"initial_state={self.initial_state!r} supplied with a chemistry "
                f"ansatz ({self.ansatz.name}) that embeds its own "
                f"reference-state preparation. The initial-state operators "
                f"will be prepended before the ansatz and may produce "
                f"unphysical circuits.",
                UserWarning,
                stacklevel=2,
            )

        # Build pipelines once (structure is fixed; only env changes per call)

        self._pipelines = self._build_pipelines()

    @property
    def n_params_per_layer(self):
        """Number of trainable parameters per ansatz layer.

        Returns:
            int: Parameters per layer for the current ansatz, qubit count,
            and electron count.
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
            self.n_qubits = (
                _require_qiskit_num_qubits(hamiltonian.num_qubits)
                if isinstance(hamiltonian, SparsePauliOp)
                else len(hamiltonian.wires)
            )
            self.n_electrons = n_electrons

        if molecule is not None:
            self.molecule = molecule
            hamiltonian, self.n_qubits = qp.qchem.molecular_hamiltonian(molecule)
            self.n_electrons = molecule.n_electrons

            if (n_electrons is not None) and self.n_electrons != n_electrons:
                warn(
                    "`n_electrons` is provided but not consistent with the molecule's. "
                    f"Got {n_electrons}, but molecule has {self.n_electrons}. "
                    "The molecular value will be used.",
                    UserWarning,
                )

        cost_spo = to_spo(hamiltonian)
        self.cost_hamiltonian, self.loss_constant = _clean_hamiltonian_spo(
            cost_spo, raise_on_constant=True
        )

    def _create_meta_circuit_factories(self) -> dict[str, MetaCircuit]:
        """Create the meta-circuit factories for VQE.

        Builds a single ``QuantumCircuit`` (initial state + ansatz) and
        wraps its DAG into both a cost ``MetaCircuit`` (carrying the
        cost ``SparsePauliOp`` as observable) and a measurement
        ``MetaCircuit`` (with all qubits measured).
        """
        n_params = self.ansatz.n_params_per_layer(
            self.n_qubits, n_electrons=self.n_electrons
        )
        weights = np.array(
            [ParameterVector(f"w_{i}", n_params) for i in range(self.n_layers)],
            dtype=object,
        )

        wires = list(range(self.n_qubits))
        qc = QuantumCircuit(self.n_qubits)
        qc.compose(self.initial_state.build(wires), inplace=True)
        qc.compose(
            self.ansatz.build(
                weights,
                n_qubits=self.n_qubits,
                n_layers=self.n_layers,
                n_electrons=self.n_electrons,
            ),
            inplace=True,
        )

        dag = circuit_to_dag(qc)
        flat_params = tuple(weights.flatten())
        return {
            "cost_circuit": MetaCircuit(
                circuit_bodies=(((), dag),),
                parameters=flat_params,
                observable=self.cost_hamiltonian,
                precision=self._precision,
            ),
            "sample_circuit": MetaCircuit(
                circuit_bodies=(((), dag),),
                parameters=flat_params,
                measured_wires=tuple(range(self.n_qubits)),
                precision=self._precision,
            ),
        }

    def sample_solution(
        self,
        params: npt.NDArray[np.float64] | None = None,
        **kwargs,
    ) -> "VQE":
        """Extract the eigenstate corresponding to the lowest energy found."""
        self.reporter.info(message="🏁 Computing Final Eigenstate 🏁", overwrite=True)

        super().sample_solution(params, **kwargs)

        if self._best_probs:
            best_measurement_probs = next(iter(self._best_probs.values()))
            eigenstate_bitstring = max(
                best_measurement_probs, key=best_measurement_probs.get
            )
            self._eigenstate = np.fromiter(eigenstate_bitstring, dtype=np.int32)

        self.reporter.info(message="🏁 Computed Final Eigenstate! 🏁")
        return self

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
