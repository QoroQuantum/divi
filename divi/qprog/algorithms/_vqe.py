# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Self
from warnings import warn

import numpy as np
import numpy.typing as npt

from divi.circuits import MetaCircuit
from divi.hamiltonians._term_ops import (
    _clean_hamiltonian_spo,
    _require_qiskit_num_qubits,
)
from divi.qprog import VariationalQuantumAlgorithm
from divi.qprog.initial_states import InitialState, ZerosState
from divi.qprog.mixins import LayeredAnsatzMixin, SolutionSamplingMixin
from divi.qprog.problems import HamiltonianProblem
from divi.reporting._events import ProgressEvent

from ._ansatze import (
    Ansatz,
    ElectronCounts,
    HartreeFockAnsatz,
    LUCJAnsatz,
    QCCAnsatz,
    UCCSDAnsatz,
)


class VQE(SolutionSamplingMixin, LayeredAnsatzMixin, VariationalQuantumAlgorithm):
    """Variational Quantum Eigensolver (VQE) implementation.

    VQE is a hybrid quantum-classical algorithm used to find the ground state
    energy of a given Hamiltonian. It works by preparing a parameterised quantum
    state (ansatz) and optimising the parameters to minimise the expectation
    value of the Hamiltonian.

    The problem is a :class:`~divi.qprog.problems.HamiltonianProblem`: a bare
    qubit Hamiltonian, or an
    :class:`~divi.qprog.problems.MolecularProblem` built from a
    molecule or from integrals.

    Attributes:
        ansatz (Ansatz): The parameterised quantum circuit ansatz.
        n_layers (int): Number of ansatz layers.
        n_qubits (int): Number of qubits in the system.
        cost_hamiltonian: The Hamiltonian to minimise.
        loss_constant (float): Constant term extracted from the Hamiltonian.
        optimizer: Classical optimizer for parameter updates.
        max_iterations (int): Maximum number of optimisation iterations.
        current_iteration (int): Current optimisation iteration.
    """

    def __init__(
        self,
        problem: HamiltonianProblem,
        *,
        n_layers: int = 1,
        ansatz: Ansatz | None = None,
        initial_state: InitialState | None = None,
        max_iterations=10,
        **kwargs,
    ) -> None:
        """Initialise the VQE problem.

        Args:
            problem: The :class:`~divi.qprog.problems.HamiltonianProblem` to
                solve. Its electron counts, when set, reach ansatzes that
                prepare a reference determinant.
            n_layers (int): Number of ansatz layers. Defaults to 1.
            ansatz (Ansatz | None): The ansatz to use for the VQE problem, with
                its own options set on the instance (e.g.
                ``LUCJAnsatz(trailing_rotation=True)``). Defaults to
                HartreeFockAnsatz.
            initial_state (InitialState | None): Initial state preparation.
                Pass an :class:`~divi.qprog.initial_states.InitialState` instance (e.g. ``ZerosState()``,
                ``SuperpositionState()``). Defaults to ``ZerosState()`` if None.
            max_iterations (int): Maximum number of optimisation iterations. Defaults to 10.
            **kwargs: Additional keyword arguments passed to the parent class.

        Raises:
            TypeError: If ``problem`` is not a
                :class:`~divi.qprog.problems.HamiltonianProblem`.
            ValueError: If the problem's Hamiltonian contains only constant
                terms.
        """
        if not isinstance(problem, HamiltonianProblem):
            raise TypeError(
                "problem must be a HamiltonianProblem; got "
                f"{type(problem).__name__}. Wrap a bare operator in "
                "HamiltonianProblem, or a molecule in "
                "MolecularProblem.from_molecule."
            )
        super().__init__(
            ansatz=HartreeFockAnsatz() if ansatz is None else ansatz,
            n_layers=n_layers,
            **kwargs,
        )
        self.max_iterations = max_iterations
        self.current_iteration = 0

        self._problem = problem
        cost_spo = problem.hamiltonian
        self.n_qubits = _require_qiskit_num_qubits(cost_spo.num_qubits)
        self.cost_hamiltonian, self.loss_constant = _clean_hamiltonian_spo(
            cost_spo, raise_on_constant=True
        )

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

    def _ansatz_reference(self) -> ElectronCounts | None:
        """The problem's electron and spin counts, for chemistry ansätze."""
        if self._problem.n_electrons is None:
            return None
        has_spin = (
            self._problem.n_alpha is not None and self._problem.n_beta is not None
        )
        return ElectronCounts(
            self._problem.n_electrons,
            self._problem.n_alpha if has_spin else None,
            self._problem.n_beta if has_spin else None,
        )

    def _initialize_param_sets(self) -> npt.NDArray[np.float64]:
        """Start excitation ansätze at the Hartree–Fock reference.

        All-zero amplitudes make every excitation gate the identity, so
        :class:`~divi.qprog.algorithms.HartreeFockAnsatz` and
        :class:`~divi.qprog.algorithms.UCCSDAnsatz` start from exactly the
        Hartree–Fock state. Other ansätze start at random.
        """
        if isinstance(self.ansatz, (HartreeFockAnsatz, UCCSDAnsatz)):
            return np.zeros((self.optimizer.n_param_sets, self.n_params))
        return super()._initialize_param_sets()

    @property
    def eigenstate(self) -> npt.NDArray[np.int32] | None:
        """Get the computed eigenstate as a NumPy array.

        Returns:
            npt.NDArray[np.int32] | None: The array of bits of the lowest energy eigenstate,
                or None if not computed. Entry ``i`` is qubit ``i``, so for
                H2 the Hartree-Fock state reads ``[1, 1, 0, 0]``.
        """
        return self._results.get("eigenstate")

    def _create_cost_circuit(self) -> MetaCircuit:
        """Create the cost MetaCircuit for VQE: initial state plus ansatz."""
        return self._cost_meta_from_ansatz(
            self.cost_hamiltonian,
            self.initial_state.build(list(range(self.n_qubits))),
        )

    def sample_solution(
        self,
        params: npt.NDArray[np.float64] | None = None,
        **kwargs,
    ) -> Self:
        """Extract the eigenstate corresponding to the lowest energy found."""
        with self._ensure_progress_session(label="VQE solution sampling", total=None):
            self._progress_emitter(
                ProgressEvent.show(
                    self._progress_key, "🏁 Computing Final Eigenstate 🏁"
                )
            )

            super().sample_solution(self._resolve_sample_params(params), **kwargs)

            if best_probs := self._results.get("best_probs"):
                best_measurement_probs = next(iter(best_probs.values()))
                eigenstate_bitstring = max(
                    best_measurement_probs, key=best_measurement_probs.__getitem__
                )
                self._results["eigenstate"] = np.fromiter(
                    eigenstate_bitstring, dtype=np.int32
                )

            self._progress_emitter(
                ProgressEvent.show(
                    self._progress_key, "🏁 Computed Final Eigenstate! 🏁"
                )
            )
            return self

    def _save_subclass_state(self) -> dict[str, Any]:
        """Save VQE-specific runtime state."""
        state = super()._save_subclass_state()
        if "eigenstate" in self._results:
            state["eigenstate"] = self._results["eigenstate"].tolist()
        return state

    def _load_subclass_state(self, state: Mapping[str, Any]) -> None:
        """Load VQE-specific state."""
        super()._load_subclass_state(state)
        if "eigenstate" in state:
            self._results["eigenstate"] = np.array(state["eigenstate"], dtype=np.int32)
