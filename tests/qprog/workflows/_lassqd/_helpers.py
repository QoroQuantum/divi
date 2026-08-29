# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the LASSQD test modules."""

import itertools
from dataclasses import fields
from typing import Self

import numpy as np
import pytest
import scipy.linalg
from pyscf import ao2mo, cc, fci, gto, scf
from qiskit.quantum_info import SparsePauliOp, Statevector

from divi.hamiltonians._chem import _spo_from_integrals
from divi.qprog import (
    LASSQD,
    FragmentationConfig,
    LASSQDPreparationMode,
    ReportingLevel,
    SQDConfig,
)
from divi.qprog.algorithms import Ansatz, UCCSDAnsatz
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.quantum_program import QuantumProgram
from divi.qprog.workflows._lassqd._active_space import localize_blocks
from divi.qprog.workflows._lassqd._integrals import (
    build_active_permutation,
    cached_ao_eri,
    cached_h_ao,
    fragment_effective_integrals,
    transform_integrals,
)
from divi.qprog.workflows._lassqd._state import FragmentSpec
from divi.qprog.workflows._lassqd._workflow import _compute_n_core

_FRAGMENTATION_FIELDS = {field.name for field in fields(FragmentationConfig)}
_SQD_FIELDS = {field.name for field in fields(SQDConfig)}


def lassqd_kwargs(**overrides):
    """Route flat keyword overrides into ``LASSQD``'s configuration objects.

    The test modules name individual knobs (``n_batches=2``,
    ``max_orbitals_per_fragment=4``); this assembles the two configs they belong
    to so each call site does not have to know which one owns what.
    """
    fragmentation = {
        name: overrides.pop(name)
        for name in list(overrides)
        if name in _FRAGMENTATION_FIELDS
    }
    sqd = {name: overrides.pop(name) for name in list(overrides) if name in _SQD_FIELDS}
    return dict(
        fragmentation=FragmentationConfig(**fragmentation),
        sqd=SQDConfig(**sqd),
        **overrides,
    )


def ansatz_energy(
    params, h_eff, g_frag, spec, ansatz: Ansatz | None = None, n_layers=1
):
    """Exact expectation value of a fragment Hamiltonian in an ansatz state."""
    ansatz = ansatz if ansatz is not None else UCCSDAnsatz()
    circuit = ansatz.build(
        params,
        2 * spec.n_orbitals,
        n_layers,
        n_electrons=spec.n_alpha + spec.n_beta,
        n_alpha=spec.n_alpha,
        n_beta=spec.n_beta,
    )
    hamiltonian = SparsePauliOp(_spo_from_integrals(h_eff, g_frag, constant=0.0))
    state = Statevector.from_instruction(circuit)
    return float(np.real(state.expectation_value(hamiltonian)))


def fragment_integrals(ensemble, mo_coeff, fragments, index):
    """``(h_alpha, h_beta, g_frag)`` for one fragment of a workflow state."""
    n_core = _compute_n_core(
        [fragment.spec for fragment in fragments], ensemble._mol.nelectron // 2
    )
    n_act = sum(fragment.spec.n_orbitals for fragment in fragments)
    integrals = transform_integrals(ensemble._mol, mo_coeff, n_core, n_act)
    return fragment_effective_integrals(integrals, fragments, index)


def embedded_fragment_ccsd(h_eff, g_frag, spec):
    """CCSD on a fragment's effective integrals, in the fragment's own basis.

    Mirrors the embedded mean field ``_ccsd_seed_params`` builds internally,
    including its identity ``mo_coeff``.
    """
    n_orb = spec.n_orbitals
    mol = gto.M(verbose=0)
    mol.nelectron = spec.n_alpha + spec.n_beta
    mol.incore_anyway = True

    mean_field = scf.RHF(mol)
    mean_field.get_hcore = lambda *args: h_eff
    mean_field.get_ovlp = lambda *args: np.eye(n_orb)
    mean_field._eri = ao2mo.restore(8, g_frag, n_orb)

    occupations = np.zeros(n_orb)
    occupations[: spec.n_alpha] = 2.0
    mean_field.mo_coeff = np.eye(n_orb)
    mean_field.mo_occ = occupations
    density = mean_field.make_rdm1()
    mean_field.mo_energy = np.diag(mean_field.get_fock(dm=density))
    mean_field.e_tot = mean_field.energy_tot(dm=density)
    mean_field.converged = True

    coupled_cluster = cc.CCSD(mean_field)
    coupled_cluster.kernel()
    return coupled_cluster


def h2_molecule(bond_length=0.74, basis="sto-3g"):
    """Closed-shell H2: 2 electrons, spatial orbital count set by ``basis``."""
    return gto.M(
        atom=f"H 0 0 0; H 0 0 {bond_length}",
        basis=basis,
        verbose=0,
    )


def h4_chain():
    """Linear H4 in STO-3G, two well-separated H2 pairs.

    4 spatial orbitals / 4 electrons, so FCI over the full space is exact in
    this basis and usable as a bound.
    """
    return gto.M(
        atom="H 0 0 0; H 0 0 0.74; H 0 0 2.0; H 0 0 2.74",
        basis="sto-3g",
        verbose=0,
    )


def h8_frontier_lassqd(backend=None, **overrides):
    """H8 split into two frontier-selected 4-orbital fragments.

    The seeding tests all want this same ensemble and differ only in the backend
    and whether the fragments are polarized.
    """
    kwargs = dict(
        n_active_orbitals=8,
        max_orbitals_per_fragment=4,
        seed=0,
        ansatz=UCCSDAnsatz(),
    )
    kwargs.update(overrides)
    return LASSQD(
        h8_chain(),
        optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        preparation_mode=LASSQDPreparationMode.VQE,
        backend=backend,
        reporting_level=ReportingLevel.OFF,
        **lassqd_kwargs(**kwargs),
    )


def h8_chain():
    """Uniform linear H8 in STO-3G.

    8 spatial orbitals / 8 electrons, which automatic fragmentation splits into
    two 4-orbital fragments holding 4 electrons each.
    """
    return gto.M(
        atom="; ".join(f"H 0 0 {index * 1.0:.1f}" for index in range(8)),
        basis="sto-3g",
        verbose=0,
    )


@pytest.fixture(scope="session")
def h4_chain_mean_field():
    """RHF mean field for ``h4_chain()``, computed once per test session."""
    return scf.RHF(h4_chain()).run(verbose=0)


@pytest.fixture(scope="session")
def h4_localized_blocks_seed0(h4_chain_mean_field):
    """``localize_blocks`` on ``h4_chain_mean_field`` under seed 0."""
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
    """An ``optimize_orbitals`` argument set spanning every rotation category.

    H2O/6-31G has 13 spatial orbitals: three frozen core, five active across
    two fragments of unequal size whose orbital indices are neither sorted nor
    contiguous, and five virtual -- 61 rotation pairs.

    The active RDMs are fixed-seed random, carrying the permutation symmetries
    of a real spatial RDM and no more: ``rdm1`` is symmetric and ``rdm2`` is
    symmetric under ``pqrs -> rspq`` and ``pqrs -> qpsr``, otherwise dense.

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


#: Energy and MO-coefficient trace of ``exact_sampler_lassqd`` over 4
#: macro-cycles. The trace fingerprints which of several equivalent orbital
#: solutions the optimizer reached, so it can move while the energy does not.
PRODUCT_STATE_ENERGY = -2.221584047820374
PRODUCT_STATE_MO_TRACE = -0.9605551058944303

#: Stand-in for a converged VQE's parameters.
_STUB_BEST_PARAMS = np.array([0.11, 0.22, 0.33])


class ExactSamplerVQE(QuantumProgram):
    """Stand-in for a fragment VQE whose distribution is exact.

    Diagonalizes the fragment's qubit Hamiltonian -- the same
    ``SparsePauliOp`` a real VQE is built from -- restricted to the correct
    alpha/beta particle-number sector, and squares the ground-state amplitudes
    into a probability distribution: a deterministic, noise-free oracle using
    the same Jordan-Wigner convention as a real fragment VQE.
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

    def run(self, **kwargs) -> Self:
        """Diagonalize the fragment's qubit Hamiltonian and sample its ground state.

        Restricts the dense matrix to the computational basis states matching
        the fragment's target alpha/beta electron counts, which a molecular
        Hamiltonian conserves separately, and diagonalizes that subspace.

        Populates ``best_probs`` keyed by divi's interleaved qubit convention
        (qubit ``2p`` / ``2p + 1`` are the alpha / beta spin-orbitals of
        spatial orbital ``p``).
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


def _build_exact_sampler_program(
    self, fragment, h_alpha, h_beta, g_frag, program_id, seed
):
    """Replacement for ``LASSQD._build_fragment_program`` used by
    :func:`build_exact_sampler_lassqd`."""
    hamiltonian = _spo_from_integrals(
        h_alpha, g_frag, constant=0.0, one_body_beta=h_beta
    )
    return ExactSamplerVQE(
        hamiltonian,
        fragment.spec,
        backend=self.backend,
        program_id=program_id,
        progress_queue=self._queue,
    )


def build_exact_sampler_lassqd(backend, mocker, seed=0, **overrides):
    """Build a fresh ``LASSQD`` ensemble whose fragment programs sample an
    exact ground state.

    Builds two 2-orbital fragments on ``h4_chain()`` and patches
    ``LASSQD._build_fragment_program`` to return :class:`ExactSamplerVQE`
    instances in place of real VQE optimizations.

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
        n_batches=2,
        batch_size=8,
        n_recovery_iterations=2,
        seed=seed,
        ansatz=UCCSDAnsatz(),
    )
    kwargs.update(overrides)
    ensemble = LASSQD(
        h4_chain(),
        optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
        preparation_mode=LASSQDPreparationMode.VQE,
        backend=backend,
        reporting_level=ReportingLevel.OFF,
        **lassqd_kwargs(**kwargs),
    )
    mocker.patch.object(LASSQD, "_build_fragment_program", _build_exact_sampler_program)

    return ensemble, ensemble.initial_state()


@pytest.fixture
def exact_sampler_lassqd(dummy_expval_backend, mocker):
    """The default ``exact_sampler`` ensemble, seeded at 0."""
    return build_exact_sampler_lassqd(dummy_expval_backend, mocker, seed=0)
