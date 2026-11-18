# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Classical CCSD-seeded LUCJ preparation, for LASSQD fragments and single
active spaces."""

import hashlib
import itertools
import math
import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Literal, Self
from warnings import warn

import ffsim
import ffsim.optimize
import ffsim.qiskit
import numpy as np
from pydantic import Field, model_validator
from pyscf import ao2mo, cc, gto, scf
from pyscf.scf import stability
from qiskit import ClassicalRegister, QuantumCircuit, transpile

from divi._thread_limits import SingleThreadedSection
from divi.backends import CircuitRunner
from divi.hamiltonians._molecular import without_chkfile
from divi.pipeline import sample_preprocessor
from divi.pipeline.stages import QiskitSpecStage
from divi.qprog import QuantumProgram
from divi.qprog._program_checkpoint import ProgramCheckpoint
from divi.qprog.checkpointing.write import fsync_directory
from divi.qprog.problems import MolecularProblem
from divi.reporting._events import ProgressEvent

from ._state import FragmentSpec, require_orthonormal

_CC_MAX_CYCLE = 500
# Re-optimisations from an unstable fragment ROHF solution's descent direction.
_STABILITY_RESTARTS = 3
# Starting occupations tried per fragment ROHF.
_ROHF_STARTS = 64
# Energy drop (Ha) below which following a reported ROHF instability finds a
# flat direction rather than a lower solution.
_FLAT_DIRECTION_TOL = 1e-9

# Fragment-sized arrays are too small for BLAS or OpenMP threading to pay.
_SINGLE_THREADED_NUMERICS = SingleThreadedSection()


#: Opposite-spin interactions of the LUCJ circuit: ``"heavy-hex"`` couples
#: every fourth orbital, as on IBM's heavy-hex lattice; ``"all"`` couples every
#: orbital.
LUCJInteraction = Literal["heavy-hex", "all"]


@dataclass(frozen=True)
class LUCJPreparation:
    """A CCSD-seeded LUCJ circuit and its Hamiltonian in the sampled basis.

    Attributes:
        circuit: Measured circuit on ``2 * n_orbitals`` interleaved qubits
            (qubit ``2p`` alpha, ``2p + 1`` beta).
        params: The circuit's ffsim LUCJ parameter vector.
        problem: The input problem rotated into the mean-field orbitals the
            circuit acts on; SQD must diagonalise this one.
        orbital_rotation: Columns are those orbitals in the input basis.
        mean_field_energy: Mean-field energy, :attr:`problem`'s constant
            included. With spin-dependent one-electron integrals, the energy
            of the alpha-channel Hamiltonian.
        ccsd_energy: CCSD energy, constant included, of the same Hamiltonian.
        ccsd_converged: Whether CCSD converged. If not, a warning was issued
            and the circuit is seeded from the last amplitudes.
    """

    circuit: QuantumCircuit
    params: np.ndarray
    problem: MolecularProblem
    orbital_rotation: np.ndarray
    mean_field_energy: float
    ccsd_energy: float
    ccsd_converged: bool


@dataclass(frozen=True)
class _PreparedFragment:
    params: np.ndarray
    problem: MolecularProblem
    orbital_rotation: np.ndarray


class _LUCJFragmentCheckpoint(ProgramCheckpoint):
    state_file: Literal["completed_state.npz"]
    state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    best_probs: dict[int, dict[str, float]]

    @model_validator(mode="after")
    def _validate_probabilities(self) -> Self:
        if any(
            not np.isfinite(probability) or probability < 0
            for probabilities in self.best_probs.values()
            for probability in probabilities.values()
        ):
            raise ValueError("Completed fragment probabilities must be finite")
        return self


class LUCJFragmentProgram(QuantumProgram):
    """Classically prepare one fragment's LUCJ circuit and submit only its sample.

    The circuit is the CCSD-seeded LUCJ operator, optimised by ffsim's linear
    method when ``run_linear_method`` is set.
    """

    def __init__(
        self,
        problem: MolecularProblem,
        spec: FragmentSpec,
        sampling_backend: CircuitRunner | None = None,
        run_linear_method: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.problem = problem
        self.spec = spec
        self._sampling_backend = sampling_backend
        self._run_linear_method = run_linear_method
        self._preparation: LUCJPreparation | None = None
        self._terminal_result: _PreparedFragment | None = None

    def run(self, **kwargs) -> Self:
        """Prepare the fragment classically, then sample its final circuit."""
        self._reject_unclaimed_run_kwargs(kwargs)
        self._preparation = prepare_lucj_fragment(
            self.problem,
            self.spec,
            report=self._show_progress,
            on_iteration=lambda energy: self._progress_emitter(
                ProgressEvent.advance(self._progress_key, loss=energy)
            ),
            run_linear_method=self._run_linear_method,
        )
        self._terminal_result = _PreparedFragment(
            params=self._preparation.params,
            problem=self._preparation.problem,
            orbital_rotation=self._preparation.orbital_rotation,
        )
        self._show_progress("Sampling the prepared circuit")
        result = self.evaluate(
            np.empty(0),
            sample_preprocessor(),
            backend=self._sampling_backend,
        )
        self._results["best_probs"] = {
            index: dict(probabilities) for index, probabilities in result.items()
        }
        return self

    def _show_progress(self, message: str) -> None:
        self._progress_emitter(ProgressEvent.show(self._progress_key, message))

    @property
    def best_params(self) -> np.ndarray:
        """The sampled circuit's ffsim LUCJ parameter vector."""
        return self._require_terminal_result().params

    @property
    def best_probs(self) -> dict[int, dict[str, float]]:
        """Normalised final-sampling probabilities."""
        return self._results.get("best_probs", {}).copy()

    @property
    def sampled_problem(self) -> MolecularProblem:
        """The fragment's Hamiltonian in the sampled determinant basis."""
        return self._require_terminal_result().problem

    @property
    def orbital_rotation(self) -> np.ndarray:
        """Rotation from the workflow fragment basis to the sampled basis."""
        return self._require_terminal_result().orbital_rotation

    def _make_checkpoint(
        self,
        checkpoint_dir: Path,
    ) -> _LUCJFragmentCheckpoint:
        result = self._require_terminal_result()
        artifact = "completed_state.npz"
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=checkpoint_dir, suffix=".npz", delete=False
            ) as handle:
                temporary_path = Path(handle.name)
                np.savez(
                    handle,
                    params=result.params,
                    h_alpha=result.problem.one_body,
                    h_beta=result.problem.one_body_beta,
                    two_body=result.problem.two_body,
                    orbital_rotation=result.orbital_rotation,
                )
                handle.flush()
                os.fsync(handle.fileno())
            with temporary_path.open("rb") as handle:
                state_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
            os.replace(temporary_path, checkpoint_dir / artifact)
            fsync_directory(checkpoint_dir)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return _LUCJFragmentCheckpoint.model_validate(
            {
                "program_type": type(self).__name__,
                "total_circuit_count": self.total_circuit_count,
                "total_run_time": self.total_run_time,
                "state_file": artifact,
                "state_sha256": state_sha256,
                "best_probs": self._results["best_probs"],
            }
        )

    def _restore_checkpoint(self, checkpoint_json: str, checkpoint_dir: Path) -> bool:
        checkpoint = _LUCJFragmentCheckpoint.model_validate_json(checkpoint_json)
        if checkpoint.program_type != type(self).__name__:
            raise ValueError("Checkpoint is for a different program type.")
        artifact = checkpoint_dir / checkpoint.state_file
        with artifact.open("rb") as handle:
            state_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        if state_sha256 != checkpoint.state_sha256:
            raise ValueError("Completed fragment state digest does not match metadata")

        with np.load(artifact, allow_pickle=False) as archive:
            required = {
                "params",
                "h_alpha",
                "h_beta",
                "two_body",
                "orbital_rotation",
            }
            if set(archive.files) != required:
                raise ValueError("Completed fragment state has missing or extra arrays")
            arrays = {name: np.asarray(archive[name]) for name in required}

        one_body_shape = self.problem.one_body.shape
        expected_shapes = {
            "h_alpha": one_body_shape,
            "h_beta": one_body_shape,
            "two_body": self.problem.two_body.shape,
            "orbital_rotation": one_body_shape,
        }
        if arrays["params"].ndim != 1 or any(
            arrays[name].shape != shape for name, shape in expected_shapes.items()
        ):
            raise ValueError("Completed fragment state has incompatible array shapes")
        if any(
            not np.issubdtype(array.dtype, np.number) or not np.all(np.isfinite(array))
            for array in arrays.values()
        ):
            raise ValueError(
                "Completed fragment state arrays must be finite numeric data"
            )
        require_orthonormal(
            arrays["orbital_rotation"], "Completed fragment state orbital_rotation"
        )

        self._terminal_result = _PreparedFragment(
            params=arrays["params"],
            problem=MolecularProblem(
                arrays["h_alpha"],
                arrays["two_body"],
                n_alpha=self.problem.n_alpha,
                n_beta=self.problem.n_beta,
                constant=self.problem.constant,
                one_body_beta=arrays["h_beta"],
            ),
            orbital_rotation=arrays["orbital_rotation"],
        )
        self._results["best_probs"] = checkpoint.best_probs
        return True

    def _spec_stage(self):
        return QiskitSpecStage()

    def _initial_spec(self) -> QuantumCircuit:
        return self._require_preparation().circuit

    def _require_preparation(self) -> LUCJPreparation:
        if self._preparation is None:
            raise RuntimeError("The fragment has not been prepared; call run() first.")
        return self._preparation

    def _require_terminal_result(self) -> _PreparedFragment:
        if self._terminal_result is None:
            raise RuntimeError("The fragment has not been prepared; call run() first.")
        return self._terminal_result


def build_lucj_interaction_pairs(
    n_orbitals: int, interaction: LUCJInteraction = "heavy-hex"
) -> tuple[
    list[tuple[int, int]],
    list[tuple[int, int]],
    list[tuple[int, int]],
]:
    """Alpha-alpha, alpha-beta and beta-beta pairs of the local LUCJ graph.

    Same-spin pairs join neighbouring orbitals; ``interaction`` picks the
    opposite-spin ones.

    Raises:
        ValueError: If ``interaction`` is neither ``"heavy-hex"`` nor ``"all"``.
    """
    if interaction not in ("heavy-hex", "all"):
        raise ValueError(
            f"interaction must be 'heavy-hex' or 'all'; got {interaction!r}."
        )
    same_spin = list(itertools.pairwise(range(n_orbitals)))
    stride = 4 if interaction == "heavy-hex" else 1
    opposite_spin = [(p, p) for p in range(0, n_orbitals, stride)]
    return same_spin, opposite_spin, same_spin.copy()


def build_lucj_circuit(
    operator: Any,
    n_orbitals: int,
    n_electrons: tuple[int, int],
) -> QuantumCircuit:
    """Build an ffsim LUCJ operator's circuit on Divi's interleaved spin wires."""

    circuit = QuantumCircuit(2 * n_orbitals)
    grouped_spin_wires = [
        *(circuit.qubits[2 * p] for p in range(n_orbitals)),
        *(circuit.qubits[2 * p + 1] for p in range(n_orbitals)),
    ]
    circuit.append(
        ffsim.qiskit.PrepareHartreeFockJW(n_orbitals, n_electrons),
        grouped_spin_wires,
    )
    gate = (
        ffsim.qiskit.UCJOpSpinBalancedJW
        if isinstance(operator, ffsim.UCJOpSpinBalanced)
        else ffsim.qiskit.UCJOpSpinUnbalancedJW
    )
    circuit.append(gate(operator), grouped_spin_wires)
    classical_bits = ClassicalRegister(2 * n_orbitals)
    circuit.add_register(classical_bits)
    for index, qubit in enumerate(circuit.qubits):
        circuit.measure(qubit, classical_bits[index])
    # PRE_INIT merges the orbital rotations before they are decomposed.
    return transpile(
        ffsim.qiskit.PRE_INIT.run(circuit),
        basis_gates=["x", "u", "cx"],
        optimization_level=1,
    )


def _physical_spin_amplitudes(
    coupled_cluster: Any,
    spec: FragmentSpec,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Relabel PySCF's majority/minority amplitudes as physical alpha/beta.

    PySCF lists the majority channel first even when it is beta; ffsim reads
    tuple order as alpha then beta, including the axes of mixed-spin doubles.
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
    values: Sequence[np.ndarray],
    *,
    label: str,
    spec: FragmentSpec,
) -> None:
    """Reject invalid numerical preparation output with fragment context."""
    if not all(np.isfinite(value).all() for value in values):
        raise RuntimeError(
            f"LUCJ preparation of fragment {spec.orbitals} produced non-finite {label}."
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


def _rohf_starts(one_body: np.ndarray, spec: FragmentSpec) -> list[np.ndarray]:
    """Starting densities for every occupation ROHF can represent, in the
    eigenbasis of ``one_body``, aufbau first; a fixed-seed sample of
    ``_ROHF_STARTS`` when there are more."""
    n_orbitals = spec.n_orbitals
    n_majority = max(spec.n_alpha, spec.n_beta)
    n_minority = min(spec.n_alpha, spec.n_beta)
    minority_per_majority = math.comb(n_majority, n_minority)
    n_occupations = math.comb(n_orbitals, n_majority) * minority_per_majority
    if n_occupations <= _ROHF_STARTS:
        occupations = [
            (singles, doubles)
            for singles in itertools.combinations(range(n_orbitals), n_majority)
            for doubles in itertools.combinations(singles, n_minority)
        ]
    else:
        generator = Random(0)
        ranks = {0}
        while len(ranks) < _ROHF_STARTS:
            ranks.add(generator.randrange(1, n_occupations))
        occupations = []
        for rank in sorted(ranks):
            majority_rank, minority_rank = divmod(rank, minority_per_majority)
            singles = _unrank_combination(range(n_orbitals), n_majority, majority_rank)
            doubles = _unrank_combination(singles, n_minority, minority_rank)
            occupations.append((singles, doubles))
    _, basis = np.linalg.eigh(one_body)

    def density(occupied):
        columns = basis[:, list(occupied)]
        return columns @ columns.T

    return [
        np.array([density(singles), density(doubles)])
        for singles, doubles in occupations
    ]


def _unrank_combination(items, size: int, rank: int) -> tuple[int, ...]:
    """Select a combination by its position in itertools' lexicographic order."""
    items = tuple(items)
    selected = []
    lower = 0
    for remaining in range(size, 0, -1):
        for index in range(lower, len(items) - remaining + 1):
            span = math.comb(len(items) - index - 1, remaining - 1)
            if rank < span:
                selected.append(items[index])
                lower = index + 1
                break
            rank -= span
    return tuple(selected)


def run_fragment_mean_field(
    method: Callable,
    one_body: np.ndarray,
    two_body: np.ndarray,
    spec: FragmentSpec,
):
    """A pyscf mean field (``scf.RHF``, ``scf.UHF`` or ``scf.ROHF``) over a
    fragment's own orthonormal orbitals and integrals.

    The molecule is a shell carrying the electron count and spin; the integrals
    are the fragment's, and the overlap is the identity. ROHF takes the spin as
    a magnitude, with the majority spin first.
    """
    n_orbitals = spec.n_orbitals
    molecule = gto.M(verbose=0)
    molecule.nelectron = spec.n_alpha + spec.n_beta
    molecule.spin = spec.n_alpha - spec.n_beta
    if method is scf.ROHF:
        molecule.spin = abs(molecule.spin)
    molecule.nao = n_orbitals
    molecule.incore_anyway = True

    mean_field: Any = without_chkfile(method(molecule))
    mean_field.get_hcore = lambda *args: one_body
    mean_field.get_ovlp = lambda *args: np.eye(n_orbitals)
    mean_field._eri = ao2mo.restore(8, two_body, n_orbitals)
    return mean_field


def _solve_fragment_rohf(
    one_body: np.ndarray,
    two_body: np.ndarray,
    spec: FragmentSpec,
):
    """Solve the fragment ROHF problem whose orbitals define the LUCJ basis.

    Open-shell fragments can have several stable ROHF solutions, so ROHF runs from
    each representable occupation and the lowest stable solution is kept.

    Raises:
        RuntimeError: If no start converges.
    """

    best = None
    for start in _rohf_starts(one_body, spec):
        mean_field = run_fragment_mean_field(scf.ROHF, one_body, two_body, spec)
        mean_field.kernel(dm0=start)
        if not mean_field.converged:
            mean_field = mean_field.newton()
            mean_field.kernel()
        if not mean_field.converged:
            continue
        for attempt in range(_STABILITY_RESTARTS + 1):
            mo_coeff, stable = stability.rohf_internal(
                mean_field, with_symmetry=False, return_status=True
            )
            if stable or attempt == _STABILITY_RESTARTS:
                break
            previous = mean_field.e_tot
            mean_field.kernel(dm0=mean_field.make_rdm1(mo_coeff, mean_field.mo_occ))
            if not mean_field.converged:
                break
            # A direction that lowers nothing is flat, as in a degenerate pair.
            if mean_field.e_tot > previous - _FLAT_DIRECTION_TOL:
                stable = True
                break
        if not mean_field.converged or not stable:
            continue
        if best is None or mean_field.e_tot < best.e_tot:
            best = mean_field
    if best is None:
        raise RuntimeError(f"ROHF did not converge for fragment {spec.orbitals}.")
    return best


def run_fragment_coupled_cluster(
    solver: Callable, name: str, mean_field: Any, spec: FragmentSpec
):
    """``solver`` (``cc.CCSD`` or ``cc.UCCSD``) run on a fragment's mean field.

    Non-convergence warns and keeps the best amplitudes reached.
    """
    coupled_cluster = solver(mean_field)
    coupled_cluster.max_cycle = _CC_MAX_CYCLE
    coupled_cluster.kernel()
    if not coupled_cluster.converged:
        warn(
            f"{name} did not converge for fragment {spec.orbitals} after "
            f"{_CC_MAX_CYCLE} cycles; using its best available amplitudes.",
            UserWarning,
            stacklevel=2,
        )
    return coupled_cluster


def prepare_lucj_fragment(
    problem: MolecularProblem,
    spec: FragmentSpec,
    report: Callable[[str], None] | None = None,
    on_iteration: Callable[[float], None] | None = None,
    run_linear_method: bool = True,
) -> LUCJPreparation:
    """Build the paper's one-repetition, spin-unbalanced fragment LUCJ circuit.

    :func:`prepare_lucj` with heavy-hex pairs and the refit factorisation.
    ``report`` receives each stage's name as it starts, and ``on_iteration``
    the energy after every linear-method iteration.
    """
    return _prepare_lucj(
        problem,
        spec,
        interaction="heavy-hex",
        n_reps=1,
        spin_balanced=False,
        refit=True,
        run_linear_method=run_linear_method,
        report=report,
        on_iteration=on_iteration,
    )


def prepare_lucj(
    problem: MolecularProblem,
    *,
    interaction: LUCJInteraction = "heavy-hex",
    n_reps: int = 1,
    spin_balanced: bool | None = None,
    refit: bool = True,
    run_linear_method: bool = False,
) -> LUCJPreparation:
    """Build a CCSD-seeded LUCJ circuit for SQD sampling (arXiv:2405.05068).

    The circuit acts on mean-field orbitals: ROHF, solved from several
    starting occupations and checked for stability, then RHF from its density
    for a spin-balanced circuit. Its parameters come from the CCSD amplitudes
    (restricted when spin-balanced, unrestricted otherwise), and from ffsim's
    linear method on the exact statevector with ``run_linear_method``. With
    spin-dependent one-electron integrals, the mean field and linear method
    use the alpha channel; the returned problem keeps both. Requires the
    ``chem`` extra.

    Args:
        problem: The active-space Hamiltonian.
        interaction: Opposite-spin pairs; same-spin pairs join neighbouring
            orbitals.
        n_reps: LUCJ layers. A spin-unbalanced operator fills them with
            alpha-beta or same-spin terms, whichever are larger; a balanced
            layer holds both.
        spin_balanced: One set of parameters for both spins. ``None`` chooses
            it when the spin counts and the alpha and beta one-electron
            integrals are equal.
        refit: Re-optimise the truncated, masked CCSD factorisation to match
            the amplitudes (ffsim's ``optimize``), rather than truncate the
            exact factorisation.
        run_linear_method: Optimise the seeded parameters with ffsim's linear
            method.

    Raises:
        ValueError: If ``spin_balanced`` is ``True`` for unequal spin counts or
            spin-dependent one-electron integrals.
        RuntimeError: If no mean field converges, or a stage produces
            non-finite values.
    """
    return _prepare_lucj(
        problem,
        FragmentSpec(
            orbitals=tuple(range(problem.n_orbitals)),
            n_alpha=problem.n_alpha,
            n_beta=problem.n_beta,
        ),
        interaction=interaction,
        n_reps=n_reps,
        spin_balanced=spin_balanced,
        refit=refit,
        run_linear_method=run_linear_method,
    )


def _is_spin_symmetric(problem: MolecularProblem) -> bool:
    return problem.n_alpha == problem.n_beta and np.array_equal(
        problem.one_body, problem.one_body_beta
    )


def _solve_restricted_mean_field(
    rohf: Any, problem: MolecularProblem, spec: FragmentSpec
):
    """RHF converged from a closed-shell ROHF solution's density.

    Raises:
        RuntimeError: If RHF does not converge, even by Newton iterations.
    """
    mean_field = run_fragment_mean_field(
        scf.RHF, problem.one_body, problem.two_body, spec
    )
    mean_field.kernel(dm0=np.sum(rohf.make_rdm1(), axis=0))
    if not mean_field.converged:
        mean_field = mean_field.newton()
        mean_field.kernel()
    if not mean_field.converged:
        raise RuntimeError(f"RHF did not converge for fragment {spec.orbitals}.")
    return mean_field


@_SINGLE_THREADED_NUMERICS.hold()
def _prepare_lucj(
    problem: MolecularProblem,
    spec: FragmentSpec,
    *,
    interaction: LUCJInteraction,
    n_reps: int,
    spin_balanced: bool | None,
    refit: bool,
    run_linear_method: bool,
    report: Callable[[str], None] | None = None,
    on_iteration: Callable[[float], None] | None = None,
) -> LUCJPreparation:
    announce = report if report is not None else (lambda message: None)
    report_iteration = on_iteration if on_iteration is not None else (lambda _: None)
    if spin_balanced is None:
        spin_balanced = _is_spin_symmetric(problem)
    elif spin_balanced and not _is_spin_symmetric(problem):
        raise ValueError(
            "spin_balanced=True needs equal alpha and beta electron counts and "
            "one-electron integrals."
        )

    announce("Fragment ROHF")
    mean_field = _solve_fragment_rohf(problem.one_body, problem.two_body, spec)
    if spin_balanced:
        mean_field = _solve_restricted_mean_field(mean_field, problem, spec)
    orbital_rotation = np.asarray(mean_field.mo_coeff)
    sampled_problem = problem.rotate_orbitals(orbital_rotation)

    announce("CCSD seed")
    n_orbitals = spec.n_orbitals
    n_electrons = (spec.n_alpha, spec.n_beta)
    pairs_aa, pairs_ab, pairs_bb = build_lucj_interaction_pairs(n_orbitals, interaction)
    build_operator: Callable[[np.ndarray], Any]
    if spin_balanced:
        coupled_cluster = run_fragment_coupled_cluster(
            cc.CCSD, "CCSD", mean_field, spec
        )
        t1, t2 = np.asarray(coupled_cluster.t1), np.asarray(coupled_cluster.t2)
        _require_finite_fragment_values((t1, t2), label="CCSD amplitudes", spec=spec)
        balanced_pairs = (pairs_aa, pairs_ab)
        initial_params = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            t2,
            t1=t1,
            n_reps=n_reps,
            interaction_pairs=balanced_pairs,
            optimize=refit,
        ).to_parameters(interaction_pairs=balanced_pairs)

        def build_operator(params: np.ndarray) -> Any:
            return ffsim.UCJOpSpinBalanced.from_parameters(
                params,
                norb=n_orbitals,
                n_reps=n_reps,
                interaction_pairs=balanced_pairs,
                with_final_orbital_rotation=True,
            )

    else:
        coupled_cluster = run_fragment_coupled_cluster(
            cc.UCCSD, "UCCSD", mean_field, spec
        )
        t1_spins, t2_spins = _physical_spin_amplitudes(coupled_cluster, spec)
        _require_finite_fragment_values(
            (*t1_spins, *t2_spins), label="CCSD amplitudes", spec=spec
        )
        unbalanced_pairs = (pairs_aa, pairs_ab, pairs_bb)
        initial_params = ffsim.UCJOpSpinUnbalanced.from_t_amplitudes(
            t2_spins,
            t1=t1_spins,
            n_reps=n_reps,
            interaction_pairs=unbalanced_pairs,
            optimize=refit,
        ).to_parameters(interaction_pairs=unbalanced_pairs)

        def build_operator(params: np.ndarray) -> Any:
            return ffsim.UCJOpSpinUnbalanced.from_parameters(
                params,
                norb=n_orbitals,
                n_reps=n_reps,
                interaction_pairs=unbalanced_pairs,
                with_final_orbital_rotation=True,
            )

    _require_finite_fragment_values(
        (initial_params,), label="CCSD seed parameters", spec=spec
    )
    params = initial_params
    if run_linear_method:
        reference_state = ffsim.hartree_fock_state(n_orbitals, n_electrons)
        hamiltonian = ffsim.linear_operator(
            ffsim.MolecularHamiltonian(
                sampled_problem.one_body, sampled_problem.two_body, 0.0
            ),
            norb=n_orbitals,
            nelec=n_electrons,
        )

        def params_to_vec(params: np.ndarray) -> np.ndarray:
            return ffsim.apply_unitary(
                reference_state,
                build_operator(params),
                norb=n_orbitals,
                nelec=n_electrons,
            )

        announce("Linear method")
        # Stops on the gradient alone.
        result = ffsim.optimize.minimize_linear_method(
            params_to_vec,
            hamiltonian,
            x0=initial_params,
            ftol=0.0,
            callback=lambda intermediate_result: report_iteration(
                float(intermediate_result.fun)
            ),
        )
        params = np.asarray(result.x, dtype=float)
        _require_finite_fragment_values(
            (params,), label="linear-method parameters", spec=spec
        )
    return LUCJPreparation(
        circuit=build_lucj_circuit(build_operator(params), n_orbitals, n_electrons),
        params=params,
        problem=sampled_problem,
        orbital_rotation=orbital_rotation,
        mean_field_energy=float(mean_field.e_tot) + problem.constant,
        ccsd_energy=float(coupled_cluster.e_tot) + problem.constant,
        ccsd_converged=bool(coupled_cluster.converged),
    )
