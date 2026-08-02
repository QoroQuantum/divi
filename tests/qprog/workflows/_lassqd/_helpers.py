# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the LASSQD test modules."""

import itertools

import numpy as np
import pytest
import scipy.linalg
from pyscf import fci, gto, scf
from qiskit.quantum_info import SparsePauliOp

from divi.hamiltonians._chem import _spo_from_integrals
from divi.qprog import LASSQD, ReportingLevel
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.quantum_program import QuantumProgram
from divi.qprog.workflows._lassqd._active_space import localize_blocks
from divi.qprog.workflows._lassqd._integrals import (
    build_active_permutation,
    cached_ao_eri,
    cached_h_ao,
)
from divi.qprog.workflows._lassqd._state import FragmentSpec


def h2_molecule(bond_length=0.74, basis="sto-3g"):
    """Closed-shell H2: 2 electrons, spatial orbital count set by ``basis``."""
    return gto.M(
        atom=f"H 0 0 0; H 0 0 {bond_length}",
        basis=basis,
        verbose=0,
    )


def h4_chain():
    """Linear H4 in STO-3G, two well-separated H2 pairs.

    4 spatial orbitals / 4 electrons. Matches the reference repo's
    ``README.md`` example so parity numbers are comparable.
    """
    return gto.M(
        atom="H 0 0 0; H 0 0 0.74; H 0 0 2.0; H 0 0 2.74",
        basis="sto-3g",
        verbose=0,
    )


@pytest.fixture(scope="session")
def h4_chain_mean_field():
    """RHF mean field for ``h4_chain()``, computed once per test session."""
    return scf.RHF(h4_chain()).run(verbose=0)


@pytest.fixture(scope="session")
def h4_localized_blocks_seed0(h4_chain_mean_field):
    """``localize_blocks`` on ``h4_chain_mean_field`` under seed 0, computed
    once per test session for tests that only need the resulting localized
    orbitals rather than exercising ``localize_blocks`` itself."""
    mol = h4_chain_mean_field.mol
    mo_coeff = np.asarray(h4_chain_mean_field.mo_coeff)
    return localize_blocks(mol, mo_coeff, (0, 1), (2, 3), np.random.default_rng(0))


def h2o_molecule(basis="6-31g"):
    """Closed-shell water at its experimental geometry."""
    return gto.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis=basis,
        verbose=0,
    )


@pytest.fixture(scope="session")
def orbital_rotation_case():
    """A deliberately demanding ``optimize_orbitals`` argument set.

    H2O/6-31G has 13 spatial orbitals. Three are frozen core, five are active
    across two fragments of unequal size whose requested orbital indices are
    neither sorted nor contiguous (so ``build_active_permutation`` genuinely
    reorders them), and five are virtual -- 61 rotation pairs spanning all
    four rotation categories, with core-core Coulomb and exchange distinct.
    The fragmentation is one ``validate_fragment_specs`` accepts.

    The active RDMs are fixed-seed random, carrying exactly the permutation
    symmetries a real spatial RDM has and no more: ``rdm1`` is symmetric, and
    ``rdm2`` is symmetric under ``pqrs -> rspq`` and ``pqrs -> qpsr`` but is
    otherwise dense and unstructured, so a transposed index in the
    two-particle density cannot cancel against itself. Random rather than
    variationally optimal RDMs also leave the orbitals far from stationary,
    which is what makes the strict-improvement assertion meaningful.

    Returns the full positional argument list of ``optimize_orbitals``:
    ``(mol, mo_coeff, n_core, specs, rdm1_active, rdm2_active, ao_eri, h_ao)``.
    """
    mol = h2o_molecule()
    mean_field = scf.RHF(mol).run(verbose=0)
    mo_coeff = np.asarray(mean_field.mo_coeff)
    n_orbitals_total = mo_coeff.shape[1]

    specs = [
        FragmentSpec(orbitals=(4, 6), n_alpha=1, n_beta=1),
        FragmentSpec(orbitals=(2, 5, 7), n_alpha=1, n_beta=1),
    ]
    n_core = 3
    n_act = sum(spec.n_orbitals for spec in specs)
    permutation = build_active_permutation(specs, n_core, n_orbitals_total)

    rng = np.random.default_rng(20250801)
    root = rng.standard_normal((n_act, n_act))
    rdm1_active = 0.3 * (root @ root.T)
    rdm2_active = rng.standard_normal((n_act,) * 4)
    rdm2_active = rdm2_active + rdm2_active.transpose(2, 3, 0, 1)
    rdm2_active = rdm2_active + rdm2_active.transpose(1, 0, 3, 2)

    return (
        mol,
        mo_coeff[:, permutation],
        n_core,
        specs,
        rdm1_active,
        rdm2_active,
        cached_ao_eri(mol),
        cached_h_ao(mol),
    )


def uniform_full_space_probs(n_orb, n_alpha, n_beta):
    """A blocked-bitstring distribution covering every symmetry-allowed determinant."""
    probs = {}
    for alpha in itertools.combinations(range(n_orb), n_alpha):
        for beta in itertools.combinations(range(n_orb), n_beta):
            bits = ["0"] * n_orb + ["0"] * n_orb
            for p in alpha:
                bits[p] = "1"
            for p in beta:
                bits[n_orb + p] = "1"
            probs["".join(bits)] = 1.0
    total = sum(probs.values())
    return {k: v / total for k, v in probs.items()}


def dense_fci_energy(one_body, two_body, n_alpha, n_beta, constant=0.0):
    """Exact lowest eigenvalue for the given spatial-MO integrals via PySCF FCI."""
    n_orb = one_body.shape[0]
    energy, _ = fci.direct_spin1.kernel(one_body, two_body, n_orb, (n_alpha, n_beta))
    return energy + constant


#: Reference implementation's converged total energy and final MO-coefficient
#: trace for the H4 chain / two 2-orbital-fragment parity fixture below.
#:
#: Generated from a checkout of the research repo (``Fragmented-SQD``) at
#: commit ``b08d97a``, with two local, uncommitted fixes applied to
#: ``sqd_core.py`` and reverted immediately afterward: a sign fix in the
#: double-excitation Slater-Condon matrix element, and switching the
#: recovery branch's dedup from ``list(set(...))`` to ``sorted(set(...))``
#: so its sampling order is deterministic instead of depending on Python's
#: per-process string hash randomization. A pristine ``b08d97a`` checkout
#: will NOT reproduce these numbers.
#:
#: Regenerate with:
#:
#: .. code-block:: bash
#:
#:     cd Fragmented-SQD && uv run --with pyscf --with numpy --with scipy python -c "
#:     from pyscf import gto
#:     from lassqd import LASSQD
#:     mol = gto.M(atom='H 0 0 0; H 0 0 0.74; H 0 0 2.0; H 0 0 2.74', basis='sto-3g', verbose=0)
#:     solver = LASSQD(mol, [
#:         {'orbitals': [0, 1], 'n_alpha': 1, 'n_beta': 1},
#:         {'orbitals': [2, 3], 'n_alpha': 1, 'n_beta': 1},
#:     ], n_batches=5, batch_size=8, n_iterations=3, lambda_penalty=0.2)
#:     energy = solver.run_scf(max_macro_cycles=4, tol=1e-5)
#:     print('REFERENCE_ENERGY =', repr(energy))
#:     print('REFERENCE_MO_TRACE =', repr(float(solver.C.trace())))
#:     "
REFERENCE_ENERGY = -2.523619542803957
REFERENCE_MO_TRACE = -0.8246322519559883

#: Fixed, recognizable stand-in for a converged VQE's parameters, so tests
#: can assert the exact values that reach the following round's
#: ``FragmentState.params``.
_STUB_BEST_PARAMS = np.array([0.11, 0.22, 0.33])


class ExactSamplerVQE(QuantumProgram):
    """Stand-in for a fragment VQE whose distribution is exact.

    Mirrors the reference's own VQE emulation: diagonalize the fragment's own
    qubit Hamiltonian (the same ``SparsePauliOp`` a real VQE is built from,
    not the underlying spatial integrals) restricted to the correct
    alpha/beta particle-number sector, and square the ground-state amplitudes
    into a probability distribution. Gives the workflow a deterministic
    oracle with no sampling noise, routed through the same Jordan-Wigner
    qubit convention a real fragment VQE would use.
    """

    def __init__(
        self,
        hamiltonian: SparsePauliOp,
        spec: FragmentSpec,
        *,
        backend,
        program_id: str | None = None,
        progress_queue=None,
        best_params: np.ndarray = _STUB_BEST_PARAMS,
    ):
        super().__init__(
            backend=backend, program_id=program_id, progress_queue=progress_queue
        )
        self._hamiltonian = hamiltonian
        self._spec = spec
        self._best_probs: dict[int, dict[str, float]] = {}
        self._best_params = best_params
        self._has_results = False

    @property
    def best_probs(self) -> dict[int, dict[str, float]]:
        return self._best_probs

    @property
    def best_params(self) -> np.ndarray:
        return self._best_params

    def has_results(self) -> bool:
        return self._has_results

    def run(self, **kwargs) -> "ExactSamplerVQE":
        """Diagonalize the fragment's qubit Hamiltonian and sample its ground state.

        Builds the dense matrix from ``self._hamiltonian`` (a
        :class:`~qiskit.quantum_info.SparsePauliOp`), whose ``to_matrix()``
        basis-state index is Qiskit's native little-endian convention (qubit
        0 is the rightmost bit) -- the reverse of divi's own bitstring
        convention (qubit ``k`` is character ``k``) used elsewhere in this
        codebase. Restricts to the computational basis states matching the
        fragment's target alpha/beta electron counts -- exact for a
        molecular Hamiltonian, which conserves both counts separately -- and
        diagonalizes that subspace instead of building a separate spatial
        Slater-Condon matrix, so a Jordan-Wigner convention mismatch would
        surface here rather than only in the converter under test.

        Populates ``best_probs`` with a single parameter set's distribution,
        keyed by divi's interleaved qubit convention (qubit ``2p`` / ``2p +
        1`` are the alpha / beta spin-orbitals of spatial orbital ``p``).
        """
        n_orb = self._spec.n_orbitals
        n_alpha, n_beta = self._spec.n_alpha, self._spec.n_beta
        n_qubits = 2 * n_orb
        matrix = self._hamiltonian.to_matrix()

        valid_indices = []
        valid_bits = []
        for i in range(2**n_qubits):
            # Qiskit's to_matrix() indexes basis states little-endian (qubit
            # 0 rightmost); reverse to divi's convention (qubit k = char k).
            bits = format(i, f"0{n_qubits}b")[::-1]
            alpha_count = sum(int(bits[2 * p]) for p in range(n_orb))
            beta_count = sum(int(bits[2 * p + 1]) for p in range(n_orb))
            if alpha_count == n_alpha and beta_count == n_beta:
                valid_indices.append(i)
                valid_bits.append(bits)

        subspace = matrix[np.ix_(valid_indices, valid_indices)]
        _, eigenvectors = scipy.linalg.eigh(subspace)
        ground_state = np.asarray(eigenvectors)[:, 0]

        probs: dict[str, float] = {}
        for bits, amplitude in zip(valid_bits, ground_state):
            prob = float(np.abs(amplitude) ** 2)
            if prob <= 1e-12:
                continue
            probs[bits] = prob

        self._best_probs = {0: probs}
        self._has_results = True
        return self


def _build_exact_sampler_program(self, fragment, h_eff, g_frag, program_id, seed):
    """Replacement for ``LASSQD._build_fragment_program`` used by
    :func:`build_exact_sampler_lassqd`."""
    hamiltonian = _spo_from_integrals(h_eff, g_frag, constant=0.0)
    return ExactSamplerVQE(
        hamiltonian,
        fragment.spec,
        backend=self.backend,
        program_id=program_id,
        progress_queue=self._queue,
    )


def build_exact_sampler_lassqd(backend, mocker, seed=0, **overrides):
    """Build a fresh ``LASSQD`` ensemble whose fragment programs sample an
    exact ground state, patched with a fresh ``mocker`` per call so distinct
    instances don't share state.

    Builds two 2-orbital fragments on ``h4_chain()`` and monkeypatches
    ``LASSQD._build_fragment_program`` to return :class:`ExactSamplerVQE`
    instances instead of running real VQE optimizations, so ``update_state``
    is exercised against a deterministic, noise-free sampled distribution.

    Args:
        overrides: Extra keyword arguments forwarded to ``LASSQD``,
            overriding the defaults below (e.g. ``energy_tol``).

    Returns:
        ``(ensemble, state)``: the ensemble and its fresh initial state.
    """
    kwargs = dict(
        active_spaces=[
            FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1),
            FragmentSpec(orbitals=(2, 3), n_alpha=1, n_beta=1),
        ],
        optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        n_batches=2,
        batch_size=8,
        n_sqd_iterations=2,
        seed=seed,
        backend=backend,
        reporting_level=ReportingLevel.OFF,
    )
    kwargs.update(overrides)
    ensemble = LASSQD(h4_chain(), **kwargs)
    mocker.patch.object(LASSQD, "_build_fragment_program", _build_exact_sampler_program)

    return ensemble, ensemble.initial_state()


@pytest.fixture
def exact_sampler_lassqd(dummy_expval_backend, mocker):
    """The default ``exact_sampler`` ensemble, seeded at 0."""
    return build_exact_sampler_lassqd(dummy_expval_backend, mocker, seed=0)
