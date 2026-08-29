# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Paper-faithful classical LUCJ preparation for LASSQD fragments."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, cast
from warnings import warn

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile

from divi.backends import CircuitRunner
from divi.hamiltonians._chem import requires_chem_extra
from divi.pipeline import sample_preprocessor
from divi.pipeline.stages import QiskitSpecStage
from divi.qprog._solution_sampling_mixin import _average_probabilities
from divi.qprog.quantum_program import QuantumProgram, reject_unclaimed_run_kwargs

from ._state import FragmentSpec

if TYPE_CHECKING:
    import ffsim
    from ffsim.optimize import minimize_linear_method
else:
    try:
        # optional chemistry extra
        import ffsim
        from ffsim.optimize import minimize_linear_method
    except ImportError:
        ffsim = None
        minimize_linear_method = None


_CC_MAX_CYCLE = 500


def _require_ffsim() -> None:
    """Raise the standard actionable error when ffsim is unavailable."""
    if ffsim is None or minimize_linear_method is None:
        raise ImportError(
            "LASSQD linear-method preparation requires the 'chem' extra; "
            "install it with `pip install qoro-divi[chem]`."
        )


@dataclass(frozen=True)
class LUCJPreparation:
    """Optimized fragment circuit and its working-basis Hamiltonian."""

    circuit: QuantumCircuit
    params: np.ndarray
    h_alpha: np.ndarray
    h_beta: np.ndarray
    two_body: np.ndarray
    orbital_rotation: np.ndarray


class LinearMethodFragmentProgram(QuantumProgram):
    """Classically prepare one fragment and submit only its final sample."""

    def __init__(
        self,
        h_alpha: np.ndarray,
        h_beta: np.ndarray,
        two_body: np.ndarray,
        spec: FragmentSpec,
        sampling_backend: CircuitRunner | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._input_h_alpha = np.asarray(h_alpha)
        self._input_h_beta = np.asarray(h_beta)
        self._input_two_body = np.asarray(two_body)
        self.spec = spec
        self._sampling_backend = sampling_backend
        self._preparation: LUCJPreparation | None = None
        self._best_probs: dict[int, dict[str, float]] = {}

    def run(self, **kwargs) -> Self:
        """Optimize the fragment classically, then sample its final circuit."""
        reject_unclaimed_run_kwargs(self, kwargs)
        self._preparation = prepare_lucj_fragment(
            self._input_h_alpha,
            self._input_h_beta,
            self._input_two_body,
            self.spec,
        )
        result = cast(
            "dict[int, dict[str, float] | list[dict[str, float]]]",
            self.evaluate(
                np.empty(0),
                sample_preprocessor(),
                backend=self._sampling_backend,
            ),
        )
        self._best_probs = {
            index: _average_probabilities(probabilities)
            for index, probabilities in result.items()
        }
        return self

    def has_results(self) -> bool:
        """Return whether final sampling probabilities are available."""
        return bool(self._best_probs)

    @property
    def best_params(self) -> np.ndarray:
        """Optimized ffsim LUCJ parameter vector."""
        return self._require_preparation().params

    @property
    def best_probs(self) -> dict[int, dict[str, float]]:
        """Normalized final-sampling probabilities."""
        return self._best_probs.copy()

    @property
    def h_alpha(self) -> np.ndarray:
        """Alpha one-body integrals in the sampled determinant basis."""
        return self._require_preparation().h_alpha

    @property
    def h_beta(self) -> np.ndarray:
        """Beta one-body integrals in the sampled determinant basis."""
        return self._require_preparation().h_beta

    @property
    def two_body(self) -> np.ndarray:
        """Two-body integrals in the sampled determinant basis."""
        return self._require_preparation().two_body

    @property
    def orbital_rotation(self) -> np.ndarray:
        """Rotation from the workflow fragment basis to the sampled basis."""
        return self._require_preparation().orbital_rotation

    def _spec_stage(self):
        return QiskitSpecStage()

    def _initial_spec(self) -> QuantumCircuit:
        return self._require_preparation().circuit

    def _require_preparation(self) -> LUCJPreparation:
        if self._preparation is None:
            raise RuntimeError("The fragment has not been prepared; call run() first.")
        return self._preparation


def paper_lucj_interaction_pairs(
    n_orbitals: int,
) -> tuple[
    list[tuple[int, int]],
    list[tuple[int, int]],
    list[tuple[int, int]],
]:
    """Return the spin-unbalanced local interaction graph used in the paper."""
    same_spin = [(p, p + 1) for p in range(n_orbitals - 1)]
    opposite_spin = [(p, p) for p in range(0, n_orbitals, 4)]
    return same_spin, opposite_spin, same_spin.copy()


def build_lucj_circuit(
    operator: Any,
    n_orbitals: int,
    n_electrons: tuple[int, int],
) -> QuantumCircuit:
    """Build the optimized ffsim circuit on Divi's interleaved spin wires."""
    _require_ffsim()

    circuit = QuantumCircuit(2 * n_orbitals)
    grouped_spin_wires = [
        *(circuit.qubits[2 * p] for p in range(n_orbitals)),
        *(circuit.qubits[2 * p + 1] for p in range(n_orbitals)),
    ]
    circuit.append(
        ffsim.qiskit.PrepareHartreeFockJW(n_orbitals, n_electrons),
        grouped_spin_wires,
    )
    circuit.append(
        ffsim.qiskit.UCJOpSpinUnbalancedJW(operator),
        grouped_spin_wires,
    )
    classical_bits = ClassicalRegister(2 * n_orbitals)
    circuit.add_register(classical_bits)
    for index, qubit in enumerate(circuit.qubits):
        circuit.measure(qubit, classical_bits[index])
    return transpile(
        circuit,
        basis_gates=["x", "u", "cx"],
        optimization_level=1,
    )


def _rotate_one_body(
    one_body: np.ndarray,
    orbital_rotation: np.ndarray,
) -> np.ndarray:
    """Rotate a spatial one-electron tensor into an MO basis."""
    return np.einsum(
        "pi,pq,qj->ij",
        orbital_rotation,
        one_body,
        orbital_rotation,
        optimize=True,
    )


def _rotate_integrals(
    one_body: np.ndarray,
    two_body: np.ndarray,
    orbital_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate spatial one- and two-electron integrals into an MO basis."""
    rotated_two_body = np.einsum(
        "pi,qj,pqrs,rk,sl->ijkl",
        orbital_rotation,
        orbital_rotation,
        two_body,
        orbital_rotation,
        orbital_rotation,
        optimize=True,
    )
    return _rotate_one_body(one_body, orbital_rotation), rotated_two_body


def _physical_spin_amplitudes(
    coupled_cluster: Any,
    spec: FragmentSpec,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Relabel PySCF's majority/minority amplitudes as physical alpha/beta.

    ROHF exposes the singly occupied majority channel first even when the
    fragment's physical majority is beta. ffsim instead interprets tuple order
    as alpha then beta, including the occupied and virtual axes of mixed-spin
    doubles.
    """
    t1_majority, t1_minority = (
        np.asarray(amplitude) for amplitude in coupled_cluster.t1
    )
    t2_majority, t2_mixed, t2_minority = (
        np.asarray(amplitude) for amplitude in coupled_cluster.t2
    )
    if spec.n_beta <= spec.n_alpha:
        return (t1_majority, t1_minority), (
            t2_majority,
            t2_mixed,
            t2_minority,
        )
    return (t1_minority, t1_majority), (
        t2_minority,
        t2_mixed.transpose(1, 0, 3, 2),
        t2_majority,
    )


def _require_finite_fragment_values(
    values: tuple[np.ndarray, ...],
    *,
    label: str,
    spec: FragmentSpec,
) -> None:
    """Reject invalid numerical preparation output with fragment context."""
    if not all(np.isfinite(value).all() for value in values):
        raise RuntimeError(
            f"LASSQD fragment {spec.orbitals} produced non-finite {label}."
        )


def rotate_rdms_to_fragment_basis(
    rdm1: np.ndarray,
    rdm2: np.ndarray,
    rdm1_alpha: np.ndarray,
    rdm1_beta: np.ndarray,
    orbital_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rotate SQD density matrices from the ROHF MO basis back to the fragment."""

    def rotate_one_body(density: np.ndarray) -> np.ndarray:
        return np.einsum(
            "ip,pq,jq->ij",
            orbital_rotation,
            density,
            orbital_rotation,
            optimize=True,
        )

    rotated_rdm2 = np.einsum(
        "ip,jq,kr,ls,pqrs->ijkl",
        orbital_rotation,
        orbital_rotation,
        orbital_rotation,
        orbital_rotation,
        rdm2,
        optimize=True,
    )
    return (
        rotate_one_body(rdm1),
        rotated_rdm2,
        rotate_one_body(rdm1_alpha),
        rotate_one_body(rdm1_beta),
    )


def _fragment_rohf(
    one_body: np.ndarray,
    two_body: np.ndarray,
    spec: FragmentSpec,
):
    """Solve the fragment ROHF problem whose orbitals define the LUCJ basis."""
    with requires_chem_extra("LASSQD linear-method preparation"):
        from pyscf import ao2mo, gto, scf

    n_orbitals = spec.n_orbitals
    molecule = gto.M(verbose=0)
    molecule.nelectron = spec.n_alpha + spec.n_beta
    # PySCF's ROHF convention puts the majority-spin amplitudes first;
    # ``_physical_spin_amplitudes`` restores physical alpha/beta tuple order.
    molecule.spin = abs(spec.n_alpha - spec.n_beta)
    molecule.nao = n_orbitals
    molecule.incore_anyway = True

    mean_field = scf.ROHF(molecule)
    mean_field.get_hcore = lambda *args: one_body
    mean_field.get_ovlp = lambda *args: np.eye(n_orbitals)
    mean_field._eri = ao2mo.restore(8, two_body, n_orbitals)
    mean_field.kernel()
    if not mean_field.converged:
        mean_field = mean_field.newton()
        mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError(f"ROHF did not converge for fragment {spec.orbitals}.")
    return mean_field


def _fragment_ccsd(mean_field: Any, spec: FragmentSpec):
    """Compute the paper's CCSD seed, retaining best amplitudes at the limit."""
    with requires_chem_extra("LASSQD linear-method preparation"):
        from pyscf import cc

    coupled_cluster = cc.CCSD(mean_field)
    coupled_cluster.max_cycle = _CC_MAX_CYCLE
    coupled_cluster.kernel()
    if not coupled_cluster.converged:
        warn(
            f"CCSD seed did not converge for fragment {spec.orbitals} after "
            f"{_CC_MAX_CYCLE} cycles; using its best available amplitudes, as "
            "in the LASSQD reference implementation.",
            UserWarning,
            stacklevel=2,
        )
    return coupled_cluster


def prepare_lucj_fragment(
    h_alpha: np.ndarray,
    h_beta: np.ndarray,
    two_body: np.ndarray,
    spec: FragmentSpec,
) -> LUCJPreparation:
    """Classically optimize the paper's one-repetition fragment LUCJ circuit.

    The sampled state uses the paper's alpha-channel preparation Hamiltonian.
    Both physical-spin one-body tensors are returned in that sampled orbital
    basis for the subsequent SQD diagonalisation.
    """
    _require_ffsim()

    mean_field = _fragment_rohf(h_alpha, two_body, spec)
    orbital_rotation = np.asarray(mean_field.mo_coeff)
    h_alpha_mo, two_body_mo = _rotate_integrals(h_alpha, two_body, orbital_rotation)
    h_beta_mo = _rotate_one_body(h_beta, orbital_rotation)

    coupled_cluster = _fragment_ccsd(mean_field, spec)
    t1, t2 = _physical_spin_amplitudes(coupled_cluster, spec)
    _require_finite_fragment_values((*t1, *t2), label="CCSD amplitudes", spec=spec)

    n_orbitals = spec.n_orbitals
    n_electrons = (spec.n_alpha, spec.n_beta)
    n_repetitions = 1
    interaction_pairs = paper_lucj_interaction_pairs(n_orbitals)
    seed_operator = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
        t2,
        t1=t1,
        n_reps=n_repetitions,
        interaction_pairs=interaction_pairs,
        optimize=True,
    )
    initial_params = seed_operator.to_parameters(interaction_pairs=interaction_pairs)
    _require_finite_fragment_values(
        (initial_params,), label="CCSD seed parameters", spec=spec
    )
    reference_state = ffsim.hartree_fock_state(n_orbitals, n_electrons)
    molecular_hamiltonian = ffsim.MolecularHamiltonian(
        h_alpha_mo,
        two_body_mo,
        0.0,
    )
    hamiltonian = ffsim.linear_operator(
        molecular_hamiltonian,
        norb=n_orbitals,
        nelec=n_electrons,
    )

    def params_to_vec(params: np.ndarray) -> np.ndarray:
        operator = ffsim.UCJOpSpinUnbalanced.from_parameters(
            params,
            norb=n_orbitals,
            n_reps=n_repetitions,
            interaction_pairs=interaction_pairs,
            with_final_orbital_rotation=True,
        )
        return ffsim.apply_unitary(
            reference_state,
            operator,
            norb=n_orbitals,
            nelec=n_electrons,
        )

    result = minimize_linear_method(
        params_to_vec,
        hamiltonian,
        x0=initial_params,
    )
    params = np.asarray(result.x, dtype=float)
    _require_finite_fragment_values(
        (params,), label="linear-method parameters", spec=spec
    )
    operator = ffsim.UCJOpSpinUnbalanced.from_parameters(
        params,
        norb=n_orbitals,
        n_reps=n_repetitions,
        interaction_pairs=interaction_pairs,
        with_final_orbital_rotation=True,
    )
    circuit = build_lucj_circuit(operator, n_orbitals, n_electrons)
    return LUCJPreparation(
        circuit=circuit,
        params=params,
        h_alpha=h_alpha_mo,
        h_beta=h_beta_mo,
        two_body=two_body_mo,
        orbital_rotation=orbital_rotation,
    )
