# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""A PySCF active-space solver backed by QSCI."""

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
from qiskit.quantum_info import SparsePauliOp

from divi._optional import import_optional
from divi.backends import CircuitRunner
from divi.hamiltonians._openfermion import import_openfermion, qubit_operator_to_spo
from divi.hamiltonians._rdm import SpinRDMs, sum_spin_rdms
from divi.qprog.algorithms import VQE, Ansatz
from divi.qprog.optimizers import Optimizer
from divi.qprog.problems import MolecularProblem
from divi.qprog.quantum_program import QuantumProgram

from ._subspace import configs_to_limbs, unpack_bits
from ._workflow import QSCI, QSCIState

#: Widest active space PySCF's selected CI addresses, whose strings are 64-bit
#: words.
_MAX_PYSCF_ORBITALS = 63

#: Most entries of the dense alpha-by-beta vector PySCF's selected CI takes
#: (512 MiB of float64).
_MAX_SCI_VECTOR_ENTRIES = 1 << 26


class QSCISolver:
    """PySCF CASCI/CASSCF active-space solver backed by :class:`QSCI`.

    Assign an instance to the ``fcisolver`` of a PySCF ``mcscf.CASCI`` or
    ``mcscf.CASSCF`` object. Each :meth:`kernel` call runs a :class:`QSCI`
    ensemble on the active-space Hamiltonian whose sampling program is a
    :class:`~divi.qprog.algorithms.VQE`, trained afresh each call. The VQE's
    samples with the active space's alpha and beta electron counts seed the
    subspace, which QSCI then grows and diagonalises with an
    ``S^2 - S(S+1)`` penalty toward the lowest spin those counts allow,
    ``S = |n_alpha - n_beta| / 2``. The final
    :class:`QSCIState` takes the place of PySCF's CI vector, so ``mc.ci`` is
    that state. Requires the ``chem`` extra.

    Attributes:
        backend: Backend the QSCI ensembles run on.
        optimizer: Optimizer configuration; each VQE trains with its own copy.
        ansatz: Ansatz of every VQE, or None for
            :class:`~divi.qprog.algorithms.VQE`'s default.
        max_iterations: Optimisation iterations of each VQE.
        qsci_options: Further :class:`QSCI` arguments.
        vqe_kwargs: Further :class:`~divi.qprog.algorithms.VQE` arguments.
        nroots: States solved for; only 1 is supported.
        converged: Whether the last VQE run's optimizer reported success.
    """

    nroots: int = 1

    def __init__(
        self,
        *,
        backend: CircuitRunner,
        optimizer: Optimizer,
        ansatz: Ansatz | None = None,
        max_iterations: int = 10,
        qsci_options: Mapping[str, Any] | None = None,
        **vqe_kwargs: Any,
    ) -> None:
        """Configure the QSCI run of every :meth:`kernel` call.

        Args:
            backend: Backend the QSCI ensembles run on.
            optimizer: Optimizer configuration; each VQE trains with its own
                copy.
            ansatz: Ansatz of every VQE. Defaults to
                :class:`~divi.qprog.algorithms.VQE`'s default.
            max_iterations: Optimisation iterations of each VQE.
            qsci_options: Further :class:`QSCI` arguments, e.g.
                ``n_expansions`` or ``lambda_penalty``.
            **vqe_kwargs: Further :class:`~divi.qprog.algorithms.VQE` arguments,
                e.g. ``n_layers``.

        Raises:
            ValueError: If ``qsci_options`` sets ``config_filter`` or
                ``penalty``, which the solver derives from the active space's
                electron counts.
        """
        reserved = sorted({"config_filter", "penalty"} & set(qsci_options or {}))
        if reserved:
            raise ValueError(
                "QSCISolver derives config_filter and penalty from the active "
                f"space's electron counts; qsci_options sets {', '.join(reserved)}."
            )
        self.backend = backend
        self.optimizer = optimizer
        self.ansatz = ansatz
        self.max_iterations = max_iterations
        self.qsci_options = dict(qsci_options or {})
        self.vqe_kwargs = vqe_kwargs
        self.converged = False
        self._last_state: QSCIState | None = None

    def kernel(
        self,
        h1e: npt.ArrayLike,
        eri: npt.ArrayLike,
        norb: int,
        nelec: int | tuple[int, int],
        ci0: Any = None,
        ecore: float = 0.0,
        **kwargs: Any,
    ) -> tuple[float, QSCIState]:
        """Run QSCI on the active space, seeded by a VQE's samples.

        Args:
            h1e: One-body integrals, ``(norb, norb)``.
            eri: Two-body integrals in chemist order, in any PySCF packing.
            norb: Active orbitals.
            nelec: Active electrons, as a total or ``(n_alpha, n_beta)``.
            ci0: Unused.
            ecore: Core energy.
            **kwargs: Further PySCF solver options, unused.

        Returns:
            ``<H>`` of the final state with ``ecore`` included, a variational
            upper bound, and the :class:`QSCIState`.

        Raises:
            NotImplementedError: If ``nroots`` is not 1, or the integrals are
                unrestricted.
            ValueError: If no sample has the active space's electron counts.
        """
        if self.nroots != 1:
            raise NotImplementedError(
                f"QSCISolver finds one root; got nroots={self.nroots}."
            )
        if np.ndim(h1e) != 2:
            raise NotImplementedError(
                "QSCISolver needs restricted integrals; got separate alpha and "
                "beta ones."
            )
        problem = MolecularProblem.from_active_space(h1e, eri, norb, nelec, ecore)
        n_alpha, n_beta = problem.n_alpha, problem.n_beta

        def build_programs(backend: CircuitRunner) -> dict[str, QuantumProgram]:
            return {
                "vqe": VQE(
                    problem,
                    ansatz=self.ansatz,
                    optimizer=self.optimizer.copy(),
                    max_iterations=self.max_iterations,
                    backend=backend,
                    **self.vqe_kwargs,
                )
            }

        def is_in_sector(config: str) -> bool:
            return (
                config[0::2].count("1") == n_alpha and config[1::2].count("1") == n_beta
            )

        qsci = QSCI(
            problem.hamiltonian,
            build_programs,
            backend=self.backend,
            config_filter=is_in_sector,
            penalty=_build_spin_penalty(norb, n_alpha, n_beta),
            **self.qsci_options,
        )
        state = qsci.run().aggregate_results()
        result = qsci.programs["vqe"].optimize_result
        self.converged = result is not None and bool(result.success)
        self._last_state = state
        return state.energy, state

    def approx_kernel(
        self,
        h1e: npt.ArrayLike,
        eri: npt.ArrayLike,
        norb: int,
        nelec: int | tuple[int, int],
        ci0: Any = None,
        ecore: float = 0.0,
        **kwargs: Any,
    ) -> tuple[float, QSCIState]:
        """The last :meth:`kernel` result, for CASSCF's approximate CI steps.

        Runs :meth:`kernel` when there is no last result.
        """
        if self._last_state is None:
            return self.kernel(h1e, eri, norb, nelec, ci0, ecore, **kwargs)
        return self._last_state.energy, self._last_state

    def make_rdm12s(
        self, ci: QSCIState, norb: int, nelec: int | tuple[int, int]
    ) -> SpinRDMs:
        """Spin RDMs of ``ci``, in PySCF's ``make_rdm12s`` layout."""
        solver, civec = _build_selected_ci(ci, norb, "QSCISolver.make_rdm12s")
        return solver.make_rdm12s(civec, norb, nelec)

    def make_rdm1s(
        self, ci: QSCIState, norb: int, nelec: int | tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Alpha and beta 1-RDMs of ``ci``."""
        solver, civec = _build_selected_ci(ci, norb, "QSCISolver.make_rdm1s")
        return solver.make_rdm1s(civec, norb, nelec)

    def make_rdm12(
        self, ci: QSCIState, norb: int, nelec: int | tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Spin-summed 1- and 2-RDMs of ``ci``."""
        return sum_spin_rdms(self.make_rdm12s(ci, norb, nelec))

    def make_rdm1(
        self, ci: QSCIState, norb: int, nelec: int | tuple[int, int]
    ) -> np.ndarray:
        """Spin-summed 1-RDM of ``ci``."""
        dm1a, dm1b = self.make_rdm1s(ci, norb, nelec)
        return dm1a + dm1b

    def spin_square(
        self, ci: QSCIState, norb: int, nelec: int | tuple[int, int]
    ) -> tuple[float, float]:
        """``<S^2>`` and the multiplicity ``2S + 1`` of ``ci``."""
        solver, civec = _build_selected_ci(ci, norb, "QSCISolver.spin_square")
        return solver.spin_square(civec, norb, nelec)


def _build_spin_penalty(norb: int, n_alpha: int, n_beta: int) -> SparsePauliOp:
    """``S^2 - S(S+1)`` on ``2 * norb`` interleaved spin-orbital qubits, for
    ``S = |n_alpha - n_beta| / 2``."""
    openfermion = import_openfermion("openfermion", "QSCISolver.kernel")
    spin = abs(n_alpha - n_beta) / 2
    s_squared = qubit_operator_to_spo(
        openfermion.jordan_wigner(openfermion.s_squared_operator(norb)), 2 * norb
    )
    target = SparsePauliOp.from_list([("I" * (2 * norb), spin * (spin + 1))])
    return (s_squared - target).simplify()


def _build_selected_ci(
    state: QSCIState, norb: int, capability: str
) -> tuple[Any, np.ndarray]:
    """A PySCF selected-CI solver and ``state`` as its CI vector over the
    state's alpha and beta strings.

    Each configuration's sign changes by the number of alpha-beta transpositions
    from the Jordan-Wigner order, which interleaves the spins, to PySCF's, which
    puts every alpha before every beta.

    Raises:
        ValueError: If ``norb`` exceeds what PySCF's selected CI addresses, or
            the state's strings span more than ``_MAX_SCI_VECTOR_ENTRIES``
            alpha-beta pairs.
    """
    if norb > _MAX_PYSCF_ORBITALS:
        raise ValueError(
            f"PySCF's selected CI takes at most {_MAX_PYSCF_ORBITALS} active "
            f"orbitals; got {norb}."
        )
    selected_ci = import_optional(
        "pyscf.fci.selected_ci", extra="chem", capability=capability
    )
    bits = unpack_bits(configs_to_limbs(state.configs, 2 * norb), 2 * norb).astype(
        np.int64
    )
    alpha_bits, beta_bits = bits[:, 0::2], bits[:, 1::2]
    weights = np.int64(1) << np.arange(norb, dtype=np.int64)
    strings_alpha, rows = np.unique(alpha_bits @ weights, return_inverse=True)
    strings_beta, columns = np.unique(beta_bits @ weights, return_inverse=True)
    n_pairs = strings_alpha.size * strings_beta.size
    if n_pairs > _MAX_SCI_VECTOR_ENTRIES:
        raise ValueError(
            f"The state's {strings_alpha.size} alpha and {strings_beta.size} beta "
            f"strings span {n_pairs} pairs, more than the "
            f"{_MAX_SCI_VECTOR_ENTRIES} PySCF's dense selected-CI vector is "
            "built for here; lower max_subspace_size."
        )
    transpositions = (alpha_bits * (np.cumsum(beta_bits, axis=1) - beta_bits)).sum(
        axis=1
    )
    amplitudes = np.zeros((strings_alpha.size, strings_beta.size))
    amplitudes[rows, columns] = (
        np.where(transpositions % 2, -1.0, 1.0) * state.eigenvector
    )
    return selected_ci.SelectedCI(), selected_ci._as_SCIvector(
        amplitudes, (strings_alpha, strings_beta)
    )
