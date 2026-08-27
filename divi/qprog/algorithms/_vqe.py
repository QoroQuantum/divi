# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self, TypeAlias
from warnings import warn

import numpy as np
import numpy.typing as npt
from qiskit import transpile
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.converters import circuit_to_dag

from divi.circuits import MetaCircuit
from divi.circuits._conversions import _QISKIT_TO_QASM2
from divi.hamiltonians._molecular import molecular_hamiltonian
from divi.hamiltonians._term_ops import (
    ObservableInput,
    _clean_hamiltonian_spo,
    _require_qiskit_num_qubits,
    to_spo,
)
from divi.qprog._solution_sampling_mixin import SolutionSamplingMixin
from divi.qprog.algorithms import (
    Ansatz,
    HartreeFockAnsatz,
    InitialState,
    LUCJAnsatz,
    QCCAnsatz,
    UCCSDAnsatz,
    ZerosState,
)
from divi.qprog.variational_quantum_algorithm import VariationalQuantumAlgorithm

if TYPE_CHECKING:
    from pennylane.qchem import Molecule as PennyLaneMolecule
    from pyscf.gto import Mole as PySCFMolecule
    from pyscf.scf.hf import SCF as PySCFMeanField

    MoleculeInput: TypeAlias = PennyLaneMolecule | PySCFMolecule | PySCFMeanField
else:
    MoleculeInput: TypeAlias = Any


class VQE(SolutionSamplingMixin, VariationalQuantumAlgorithm):
    """Variational Quantum Eigensolver (VQE) implementation.

    VQE is a hybrid quantum-classical algorithm used to find the ground state
    energy of a given Hamiltonian. It works by preparing a parameterised quantum
    state (ansatz) and optimising the parameters to minimise the expectation
    value of the Hamiltonian.

    The algorithm can work with either:
    - A molecular Hamiltonian (for quantum chemistry problems)
    - A custom Hamiltonian operator

    Attributes:
        ansatz (Ansatz): The parameterised quantum circuit ansatz.
        n_layers (int): Number of ansatz layers.
        n_qubits (int): Number of qubits in the system.
        n_electrons (int): Number of electrons (for molecular systems).
        cost_hamiltonian: The Hamiltonian to minimise.
        loss_constant (float): Constant term extracted from the Hamiltonian.
        molecule: The molecule object (if applicable).
        optimizer: Classical optimizer for parameter updates.
        max_iterations (int): Maximum number of optimisation iterations.
        current_iteration (int): Current optimisation iteration.
    """

    def __init__(
        self,
        hamiltonian: ObservableInput | None = None,
        molecule: MoleculeInput | None = None,
        n_electrons: int | None = None,
        n_layers: int = 1,
        ansatz: Ansatz | None = None,
        initial_state: InitialState | None = None,
        max_iterations=10,
        n_alpha: int | None = None,
        n_beta: int | None = None,
        ansatz_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Initialise the VQE problem.

        Args:
            hamiltonian: A Hamiltonian representing the problem — a PennyLane
                operator, a Qiskit ``SparsePauliOp``, a divi Pauli-string dict,
                or an OpenFermion ``QubitOperator`` (requires the ``chem``
                extra). Defaults to None.
            molecule: The molecule representing the problem. Either a PennyLane
                ``qp.qchem.Molecule`` or a PySCF ``gto.Mole`` / restricted
                mean-field object (requires the ``chem`` extra). Defaults to None.
            n_electrons (int | None): Number of electrons associated with the Hamiltonian.
                Only needed when a Hamiltonian is given. Defaults to None.
            n_layers (int): Number of ansatz layers. Defaults to 1.
            ansatz (Ansatz | None): The ansatz to use for the VQE problem.
                Defaults to HartreeFockAnsatz.
            initial_state (InitialState | None): Initial state preparation.
                Pass an :class:`~divi.qprog.algorithms.InitialState` instance (e.g. ``ZerosState()``,
                ``SuperpositionState()``). Defaults to ``ZerosState()`` if None.
            max_iterations (int): Maximum number of optimisation iterations. Defaults to 10.
            n_alpha (int | None): Alpha electrons, for a spin-polarised
                reference. Must be given together with ``n_beta``; without both,
                ansatzes that need a reference determinant assume the
                closed-shell split. Defaults to None.
            n_beta (int | None): Beta electrons, under the same convention.
                Defaults to None.
            ansatz_kwargs (dict | None): Ansatz-specific options, forwarded to
                both ``n_params_per_layer`` and ``build`` so the parameter count
                and the circuit stay in agreement (e.g.
                ``{"trailing_rotation": True}`` for
                :class:`~divi.qprog.algorithms.LUCJAnsatz`). Defaults to None.
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

        if (n_alpha is None) != (n_beta is None):
            raise ValueError(
                "n_alpha and n_beta must be given together; got "
                f"n_alpha={n_alpha}, n_beta={n_beta}."
            )
        self._spin_kwargs: dict[str, int] = (
            {}
            if n_alpha is None or n_beta is None
            else {"n_alpha": n_alpha, "n_beta": n_beta}
        )
        # Merged into every ansatz call, so the count and the circuit agree.
        self._ansatz_kwargs: dict[str, Any] = dict(ansatz_kwargs or {})

        # Resolve & store initial state (n_qubits is now set)
        if initial_state is None:
            initial_state = ZerosState()
        self.initial_state = initial_state

        if not isinstance(self.initial_state, ZerosState) and isinstance(
            self.ansatz, (HartreeFockAnsatz, LUCJAnsatz, QCCAnsatz, UCCSDAnsatz)
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

    @property
    def n_params_per_layer(self):
        """Number of trainable parameters per ansatz layer.

        Returns:
            int: Parameters per layer for the current ansatz, qubit count,
            and electron count.
        """
        return self.ansatz.n_params_per_layer(
            self.n_qubits,
            n_electrons=self.n_electrons,
            **self._spin_kwargs,
            **self._ansatz_kwargs,
        )

    def _parameter_frequencies(self):
        """The ansatz's per-layer frequencies, repeated across layers."""
        per_layer = self.ansatz.parameter_frequencies(
            self.n_qubits,
            n_electrons=self.n_electrons,
            **self._spin_kwargs,
            **self._ansatz_kwargs,
        )
        return None if per_layer is None else list(per_layer) * self.n_layers

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
            hamiltonian: PennyLane operator, SparsePauliOp, or OpenFermion
                QubitOperator, or None.
            molecule: PennyLane Molecule, PySCF Mole / mean-field, or None.
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
            self.n_electrons = n_electrons

        if molecule is not None:
            self.molecule = molecule
            hamiltonian, self.n_electrons = molecular_hamiltonian(molecule)

            if (n_electrons is not None) and self.n_electrons != n_electrons:
                warn(
                    "`n_electrons` is provided but not consistent with the molecule's. "
                    f"Got {n_electrons}, but molecule has {self.n_electrons}. "
                    "The molecular value will be used.",
                    UserWarning,
                )

        cost_spo = to_spo(hamiltonian)
        self.n_qubits = _require_qiskit_num_qubits(cost_spo.num_qubits)
        self.cost_hamiltonian, self.loss_constant = _clean_hamiltonian_spo(
            cost_spo, raise_on_constant=True
        )

    def _cost_meta_from_ansatz(
        self, prefix: QuantumCircuit | None = None, /, **ansatz_kwargs
    ) -> MetaCircuit:
        """Wrap ``prefix`` plus a layered ansatz into a cost ``MetaCircuit``.

        The same keywords reach ``n_params_per_layer`` and ``build``, so the
        parameter count and the circuit cannot disagree. ``prefix`` is
        positional so an ansatz keyword of the same name cannot capture it.
        """
        n_params = self.ansatz.n_params_per_layer(
            self.n_qubits, n_electrons=self.n_electrons, **ansatz_kwargs
        )
        weights = np.array(
            [ParameterVector(f"w_{i}", n_params) for i in range(self.n_layers)],
            dtype=object,
        )

        qc = QuantumCircuit(self.n_qubits)
        if prefix is not None:
            qc.compose(prefix, inplace=True)
        qc.compose(
            self.ansatz.build(
                weights,
                n_qubits=self.n_qubits,
                n_layers=self.n_layers,
                n_electrons=self.n_electrons,
                **ansatz_kwargs,
            ),
            inplace=True,
        )

        # Lower to the gate set the QASM body emitter accepts. Ansatzes such as
        # LUCJAnsatz emit gates (e.g. xx_plus_yy) outside that basis;
        # optimization_level=0 keeps this a cheap gate-by-gate substitution.
        qc = transpile(
            qc,
            basis_gates=list(_QISKIT_TO_QASM2.keys()),
            optimization_level=0,
        )

        return MetaCircuit(
            circuit_bodies=(((), circuit_to_dag(qc)),),
            parameters=tuple(weights.flatten()),
            observable=self.cost_hamiltonian,
            precision=self._precision,
        )

    def _create_cost_circuit(self) -> MetaCircuit:
        """Create the cost MetaCircuit for VQE: initial state plus ansatz."""
        return self._cost_meta_from_ansatz(
            self.initial_state.build(list(range(self.n_qubits))),
            **self._spin_kwargs,
            **self._ansatz_kwargs,
        )

    def sample_solution(
        self,
        params: npt.NDArray[np.float64] | None = None,
        **kwargs,
    ) -> Self:
        """Extract the eigenstate corresponding to the lowest energy found."""
        self.reporter.info(message="🏁 Computing Final Eigenstate 🏁", overwrite=True)

        super().sample_solution(self._resolve_sample_params(params), **kwargs)

        if self._best_probs:
            best_measurement_probs = next(iter(self._best_probs.values()))
            eigenstate_bitstring = max(
                best_measurement_probs, key=best_measurement_probs.__getitem__
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
