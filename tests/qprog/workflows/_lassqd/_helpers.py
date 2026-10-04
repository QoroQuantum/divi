# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the LASSQD test modules."""

import itertools
from dataclasses import fields
from functools import partial
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
    ReportingLevel,
    SQDConfig,
    VQEPreparation,
)
from divi.qprog.algorithms import Ansatz, UCCSDAnsatz
from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer
from divi.qprog.problems import MolecularProblem
from divi.qprog.quantum_program import QuantumProgram
from divi.qprog.workflows._lassqd._active_space import localize_blocks
from divi.qprog.workflows._lassqd._config import SecondOrderOrbitalSolve
from divi.qprog.workflows._lassqd._integrals import (
    assemble_active_rdms,
    build_active_permutation,
    cached_ao_eri,
    cached_h_ao,
    ciah_orbital_solve,
    fragment_effective_integrals,
    optimize_orbitals,
    transform_integrals,
)
from divi.qprog.workflows._lassqd._sqd import ci_string_to_int, compute_spatial_rdms
from divi.qprog.workflows._lassqd._state import FragmentSpec, FragmentState
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


def fragment_problem(h_alpha, h_beta, g_frag, spec):
    """The embedded problem ``LASSQD.create_programs`` hands a fragment program."""
    return MolecularProblem(
        h_alpha,
        g_frag,
        n_alpha=spec.n_alpha,
        n_beta=spec.n_beta,
        one_body_beta=h_beta,
    )


def embedded_fragment_ccsd(h_eff, g_frag, spec):
    """CCSD on a fragment's effective integrals, in the fragment's own basis.

    Mirrors the embedded mean field ``_ccsd_seed_params`` builds internally,
    including its identity ``mo_coeff``.
    """
    n_orb = spec.n_orbitals
    mol = gto.M(verbose=0)
    mol.incore_anyway = True

    mean_field = scf.RHF(mol)
    mean_field.get_hcore = lambda *args: h_eff
    mean_field._eri = ao2mo.restore(8, g_frag, n_orb)

    occupations = np.zeros(n_orb)
    occupations[: spec.n_alpha] = 2.0
    mean_field.mo_coeff = np.eye(n_orb)
    mean_field.mo_occ = occupations

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


def cobyla():
    """A fresh COBYLA optimizer."""
    return ScipyOptimizer(ScipyMethod.COBYLA)


def vqe_preparation(**overrides):
    """COBYLA-driven UCCSD fragment VQE, with ``overrides`` replacing fields."""
    return VQEPreparation(
        **({"optimizer": cobyla(), "ansatz": UCCSDAnsatz()} | overrides)
    )


def h8_frontier_lassqd(backend=None, **overrides):
    """H8 split into two frontier-selected 4-orbital fragments, prepared by
    :func:`vqe_preparation`."""
    kwargs = dict(
        n_active_orbitals=8,
        max_orbitals_per_fragment=4,
        seed=0,
    )
    kwargs.update(overrides)
    return LASSQD(
        MolecularProblem.from_molecule(h8_chain()),
        preparation=vqe_preparation(),
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


#: H4's whole active space as a single fragment.
H4_WHOLE_SPACE = FragmentSpec(orbitals=(0, 1, 2, 3), n_alpha=2, n_beta=2)

#: ``h8_chain()``'s atoms split into two four-atom halves.
H8_HALF_CHAINS = ([0, 1, 2, 3], [4, 5, 6, 7])


@pytest.fixture(scope="session")
def h2_mean_field():
    """RHF mean field for ``h2_molecule()``, computed once per test session."""
    return scf.RHF(h2_molecule()).run(verbose=0)


@pytest.fixture(scope="session")
def h4_chain_mean_field():
    """RHF mean field for ``h4_chain()``, computed once per test session."""
    return scf.RHF(h4_chain()).run(verbose=0)


def mo_integrals(mean_field):
    """``(one_body, two_body, n_orb, constant)`` over every MO of an RHF solve,
    with ``two_body`` in chemist order and ``constant`` the nuclear repulsion."""
    mol = mean_field.mol
    mo_coeff = np.asarray(mean_field.mo_coeff)
    n_orb = mo_coeff.shape[1]
    one_body = mo_coeff.T @ mean_field.get_hcore() @ mo_coeff
    two_body = ao2mo.restore(1, ao2mo.kernel(mol, mo_coeff), n_orb)
    return one_body, two_body, n_orb, float(mol.energy_nuc())


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

    The active RDMs are those of a product of fixed-seed random fragment
    wavefunctions, each spanning its fragment's whole determinant space, so
    they are physical while still dense.

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
    permutation = build_active_permutation(specs, n_core, n_orbitals_total)

    rng = np.random.default_rng(20250801)
    fragments = []
    for spec in specs:
        strings_alpha = _sector_strings(spec.n_orbitals, spec.n_alpha)
        strings_beta = _sector_strings(spec.n_orbitals, spec.n_beta)
        amplitudes = rng.standard_normal((len(strings_alpha), len(strings_beta)))
        amplitudes /= np.linalg.norm(amplitudes)
        rdm1, rdm2, rdm1_alpha, rdm1_beta = compute_spatial_rdms(
            strings_alpha, strings_beta, amplitudes, spec.n_orbitals
        )
        fragments.append(
            FragmentState(
                spec=spec,
                rdm1=rdm1,
                rdm2=rdm2,
                rdm1_alpha=rdm1_alpha,
                rdm1_beta=rdm1_beta,
            )
        )
    rdm1_active, rdm2_active = assemble_active_rdms(fragments)

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


#: ``ciah_orbital_solve`` under ``SecondOrderOrbitalSolve``'s default cap.
capped_ciah_orbital_solve = partial(
    ciah_orbital_solve, max_iterations=SecondOrderOrbitalSolve().max_iterations
)


@pytest.fixture(
    params=[optimize_orbitals, capped_ciah_orbital_solve], ids=["l-bfgs-b", "ciah"]
)
def orbital_solver(request):
    """Each orbital solve, sharing ``optimize_orbitals``'s signature."""
    return request.param


def _sector_strings(n_orb, n_electrons):
    """Every occupation string of one spin sector, ascending as SQD orders them."""
    strings = (
        "".join("1" if p in occupied else "0" for p in range(n_orb))
        for occupied in itertools.combinations(range(n_orb), n_electrons)
    )
    return sorted(strings, key=ci_string_to_int)


def build_energy_rdms(n_orb, n_core, rdm1_active, rdm2_active):
    """Dense full-register ``(D, d)`` whose contraction reproduces ``_total_energy``.

    ``E = E_nuc + sum(h * D) + 0.5 * sum(d * g)`` over the permuted MO register,
    with ``g`` in chemist order: core ``[0, n_core)`` doubly occupied, the active
    RDMs next, the virtual block zero.
    """
    n_act = rdm1_active.shape[0]
    active = slice(n_core, n_core + n_act)

    one_rdm = np.zeros((n_orb, n_orb))
    two_rdm = np.zeros((n_orb,) * 4)

    for i in range(n_core):
        one_rdm[i, i] = 2.0
    one_rdm[active, active] = rdm1_active

    for i in range(n_core):
        for j in range(n_core):
            two_rdm[i, i, j, j] += 4.0
            two_rdm[i, j, j, i] -= 2.0

    for i in range(n_core):
        two_rdm[active, active, i, i] += 2.0 * rdm1_active
        two_rdm[i, i, active, active] += 2.0 * rdm1_active
        two_rdm[active, i, i, active] -= rdm1_active
        two_rdm[i, active, active, i] -= rdm1_active

    two_rdm[active, active, active, active] += rdm2_active

    return one_rdm, two_rdm


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
PRODUCT_STATE_ENERGY = -2.221584055981204
PRODUCT_STATE_MO_TRACE = -0.0028071851266455727

#: Stand-in for a converged VQE's parameters.
_STUB_BEST_PARAMS = np.array([0.11, 0.22, 0.33])


class ExactSamplerVQE(QuantumProgram):
    """Stand-in for a fragment VQE whose distribution is exact.

    Diagonalizes the fragment's qubit Hamiltonian -- the same
    ``SparsePauliOp`` a real VQE is built from -- restricted to the correct
    alpha/beta particle-number sector, and squares the ground-state amplitudes
    into a probability distribution: a deterministic, noise-free oracle using
    the same Jordan-Wigner convention as a real fragment VQE. Like a fragment
    VQE, it exposes its problem's integrals and an identity orbital rotation.
    """

    def __init__(
        self,
        problem: MolecularProblem,
        spec: FragmentSpec,
        *,
        backend,
        best_params: np.ndarray = _STUB_BEST_PARAMS,
    ):
        super().__init__(backend=backend)
        self._problem = problem
        self._spec = spec
        self._best_params = best_params

    @property
    def best_probs(self) -> dict[int, dict[str, float]]:
        return self._results.get("best_probs", {})

    @property
    def best_params(self) -> np.ndarray:
        return self._best_params

    @property
    def h_alpha(self) -> np.ndarray:
        return self._problem.one_body

    @property
    def h_beta(self) -> np.ndarray:
        return self._problem.one_body_beta

    @property
    def two_body(self) -> np.ndarray:
        return self._problem.two_body

    @property
    def orbital_rotation(self) -> np.ndarray:
        return np.eye(self._spec.n_orbitals)

    def has_results(self) -> bool:
        return bool(self._results)

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
        matrix = self._problem.hamiltonian.to_matrix()

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

        self._results["best_probs"] = {0: probs}
        return self


def _build_exact_sampler_program(
    self, problem, fragment, *, backend, sampling_backend, seed, options
):
    """Replacement for a preparation's ``_build_program`` used by
    :func:`build_exact_sampler_lassqd`."""
    return ExactSamplerVQE(problem, fragment.spec, backend=backend)


def build_exact_sampler_lassqd(backend, mocker, seed=0, problem=None, **overrides):
    """Build a fresh ``LASSQD`` ensemble whose fragment programs sample an
    exact ground state.

    Builds two 2-orbital fragments on ``h4_chain()`` unless ``problem`` is
    given, and patches the preparation's ``_build_program`` to return
    :class:`ExactSamplerVQE` instances in place of real VQE optimizations.

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
        preparation=vqe_preparation(),
    )
    kwargs.update(overrides)
    ensemble = LASSQD(
        problem or MolecularProblem.from_molecule(h4_chain()),
        backend=backend,
        reporting_level=ReportingLevel.OFF,
        **lassqd_kwargs(**kwargs),
    )
    mocker.patch.object(
        type(ensemble.preparation), "_build_program", _build_exact_sampler_program
    )

    return ensemble, ensemble.initial_state()


@pytest.fixture
def exact_sampler_lassqd(dummy_expval_backend, mocker):
    """The default ``exact_sampler`` ensemble, seeded at 0."""
    return build_exact_sampler_lassqd(dummy_expval_backend, mocker, seed=0)
