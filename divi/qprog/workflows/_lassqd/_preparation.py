# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Paper-faithful classical LUCJ preparation for LASSQD fragments."""

import contextlib
import hashlib
import itertools
import math
import os
import tempfile
import threading
from collections.abc import Callable
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
from pyscf import ao2mo, cc, gto, lib, scf
from pyscf.scf import stability
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from threadpoolctl import threadpool_limits

from divi.backends import CircuitRunner
from divi.pipeline import sample_preprocessor
from divi.pipeline.stages import QiskitSpecStage
from divi.qprog._program_checkpoint import ProgramCheckpoint
from divi.qprog.checkpointing import _fsync_directory
from divi.qprog.problems import MolecularProblem
from divi.qprog.quantum_program import (
    QuantumProgram,
    reject_unclaimed_run_kwargs,
)
from divi.reporting._events import ProgressEvent

from ._state import FragmentSpec, require_orthonormal

_CC_MAX_CYCLE = 500
# Re-optimisations from an unstable fragment ROHF solution's descent direction.
_STABILITY_RESTARTS = 3
# Starting occupations tried per fragment ROHF.
_ROHF_STARTS = 64

_PINNING_LOCK = threading.Lock()
_pinned_users = 0
_pinned_limits: threadpool_limits | None = None
_pinned_pyscf_threads = 0


@dataclass(frozen=True)
class LUCJPreparation:
    """Optimized fragment circuit and its working-basis Hamiltonian."""

    circuit: QuantumCircuit
    params: np.ndarray
    h_alpha: np.ndarray
    h_beta: np.ndarray
    two_body: np.ndarray
    orbital_rotation: np.ndarray


@dataclass(frozen=True)
class _PreparedFragment:
    params: np.ndarray
    h_alpha: np.ndarray
    h_beta: np.ndarray
    two_body: np.ndarray
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
        reject_unclaimed_run_kwargs(self, kwargs)
        self._preparation = prepare_lucj_fragment(
            self.problem.one_body,
            self.problem.one_body_beta,
            self.problem.two_body,
            self.spec,
            report=self._show_progress,
            on_iteration=lambda energy: self._progress_emitter(
                ProgressEvent.advance(self._progress_key, loss=energy)
            ),
            run_linear_method=self._run_linear_method,
        )
        self._terminal_result = _PreparedFragment(
            params=self._preparation.params,
            h_alpha=self._preparation.h_alpha,
            h_beta=self._preparation.h_beta,
            two_body=self._preparation.two_body,
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

    def has_results(self) -> bool:
        """Return whether final sampling probabilities are available."""
        return bool(self._results.get("best_probs"))

    @property
    def best_params(self) -> np.ndarray:
        """The sampled circuit's ffsim LUCJ parameter vector."""
        return self._require_terminal_result().params

    @property
    def best_probs(self) -> dict[int, dict[str, float]]:
        """Normalised final-sampling probabilities."""
        return self._results.get("best_probs", {}).copy()

    @property
    def h_alpha(self) -> np.ndarray:
        """Alpha one-body integrals in the sampled determinant basis."""
        return self._require_terminal_result().h_alpha

    @property
    def h_beta(self) -> np.ndarray:
        """Beta one-body integrals in the sampled determinant basis."""
        return self._require_terminal_result().h_beta

    @property
    def two_body(self) -> np.ndarray:
        """Two-body integrals in the sampled determinant basis."""
        return self._require_terminal_result().two_body

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
                    h_alpha=result.h_alpha,
                    h_beta=result.h_beta,
                    two_body=result.two_body,
                    orbital_rotation=result.orbital_rotation,
                )
                handle.flush()
                os.fsync(handle.fileno())
            with temporary_path.open("rb") as handle:
                state_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
            os.replace(temporary_path, checkpoint_dir / artifact)
            _fsync_directory(checkpoint_dir)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
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

        result = _PreparedFragment(**arrays)
        self._terminal_result = result
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
        ranks = [
            0,
            *sorted(Random(0).sample(range(1, n_occupations), _ROHF_STARTS - 1)),
        ]
        occupations = []
        for rank in ranks:
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


def _fragment_rohf(
    one_body: np.ndarray,
    two_body: np.ndarray,
    spec: FragmentSpec,
):
    """Solve the fragment ROHF problem whose orbitals define the LUCJ basis.

    Open-shell fragments have several stable ROHF solutions, so ROHF runs from
    each representable occupation and the lowest stable solution is kept.

    Raises:
        RuntimeError: If no start converges.
    """

    n_orbitals = spec.n_orbitals
    molecule = gto.M(verbose=0)
    molecule.nelectron = spec.n_alpha + spec.n_beta
    # PySCF's ROHF convention puts the majority-spin amplitudes first;
    # ``_physical_spin_amplitudes`` restores physical alpha/beta tuple order.
    molecule.spin = abs(spec.n_alpha - spec.n_beta)
    molecule.nao = n_orbitals
    molecule.incore_anyway = True
    eri = ao2mo.restore(8, two_body, n_orbitals)

    best = None
    for start in _rohf_starts(one_body, spec):
        mean_field = scf.ROHF(molecule)
        mean_field.get_hcore = lambda *args: one_body
        mean_field.get_ovlp = lambda *args: np.eye(n_orbitals)
        mean_field._eri = eri
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
            if stable:
                break
            if attempt == _STABILITY_RESTARTS:
                break
            mean_field.kernel(dm0=mean_field.make_rdm1(mo_coeff, mean_field.mo_occ))
            if not mean_field.converged:
                break
        if not mean_field.converged or not stable:
            continue
        if best is None or mean_field.e_tot < best.e_tot:
            best = mean_field
    if best is None:
        raise RuntimeError(f"ROHF did not converge for fragment {spec.orbitals}.")
    return best


def _fragment_ccsd(mean_field: Any, spec: FragmentSpec):
    """Compute the paper's CCSD seed, retaining best amplitudes at the limit."""

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


@contextlib.contextmanager
def _single_threaded_numerics():
    """Pin BLAS and OpenMP to one thread while any fragment preparation runs.

    Fragment-sized arrays are too small for threading to pay. The limits are
    process-wide, so concurrent preparations share one pinning, lifted when the
    last of them finishes.
    """
    global _pinned_users, _pinned_limits, _pinned_pyscf_threads
    with _PINNING_LOCK:
        if _pinned_users == 0:
            _pinned_limits = threadpool_limits(limits=1)
            _pinned_pyscf_threads = lib.num_threads()
            lib.num_threads(1)
        _pinned_users += 1
    try:
        yield
    finally:
        with _PINNING_LOCK:
            _pinned_users -= 1
            if _pinned_users == 0:
                lib.num_threads(_pinned_pyscf_threads)
                _pinned_limits.restore_original_limits()
                _pinned_limits = None


@_single_threaded_numerics()
def prepare_lucj_fragment(
    h_alpha: np.ndarray,
    h_beta: np.ndarray,
    two_body: np.ndarray,
    spec: FragmentSpec,
    report: Callable[[str], None] | None = None,
    on_iteration: Callable[[float], None] | None = None,
    run_linear_method: bool = True,
) -> LUCJPreparation:
    """Build the paper's one-repetition fragment LUCJ circuit from CCSD.

    The sampled state uses the paper's alpha-channel preparation Hamiltonian.
    Both physical-spin one-body tensors are returned in that sampled orbital
    basis for the subsequent SQD diagonalisation.

    With ``run_linear_method``, ffsim's linear method optimises the CCSD seed
    on the exact fragment statevector; without it, the seed is the circuit.

    ``report`` receives each stage's name as it starts, and ``on_iteration``
    the energy after every linear-method iteration.
    """
    announce = report if report is not None else (lambda message: None)
    report_iteration = on_iteration if on_iteration is not None else (lambda _: None)

    announce("Fragment ROHF")
    mean_field = _fragment_rohf(h_alpha, two_body, spec)
    orbital_rotation = np.asarray(mean_field.mo_coeff)
    h_alpha_mo, two_body_mo = _rotate_integrals(h_alpha, two_body, orbital_rotation)
    h_beta_mo = _rotate_one_body(h_beta, orbital_rotation)

    announce("CCSD seed")
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
    params = initial_params
    if run_linear_method:
        reference_state = ffsim.hartree_fock_state(n_orbitals, n_electrons)
        hamiltonian = ffsim.linear_operator(
            ffsim.MolecularHamiltonian(h_alpha_mo, two_body_mo, 0.0),
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
