# SPDX-FileCopyrightText: 2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""A PySCF active-space solver backed by VQE."""

import copy
from typing import Any

import numpy as np
import numpy.typing as npt

from divi._optional import import_optional
from divi.backends import CircuitRunner
from divi.hamiltonians._rdm import SpinRDMs, sum_spin_rdms
from divi.qprog.optimizers import Optimizer
from divi.qprog.problems import MolecularProblem

from ._ansatze import Ansatz
from ._vqe import VQE


class VQEFCISolver:
    """PySCF CASCI/CASSCF active-space solver backed by :class:`VQE`.

    Assign an instance to the ``fcisolver`` of a PySCF ``mcscf.CASCI`` or
    ``mcscf.CASSCF`` object. Each :meth:`kernel` call trains a VQE on the
    active-space integrals, warm-started from the previous call's parameters,
    and measures its spin RDMs with :meth:`VQE.measure_rdms`. The trained VQE
    takes the place of PySCF's CI vector, so ``mc.ci`` is that program.
    Requires the ``chem`` extra.

    Attributes:
        backend: Backend the VQEs run on.
        optimizer: Optimizer configuration; each VQE trains with its own copy.
        ansatz: Ansatz of every VQE, or None for :class:`VQE`'s default.
        max_iterations: Optimisation iterations per :meth:`kernel` call.
        vqe_kwargs: Further :class:`VQE` arguments.
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
        **vqe_kwargs: Any,
    ) -> None:
        """Configure the VQE every :meth:`kernel` call trains.

        Args:
            backend: Backend the VQEs run on.
            optimizer: Optimizer configuration; each VQE trains with its own
                copy.
            ansatz: Ansatz of every VQE. Defaults to :class:`VQE`'s default.
            max_iterations: Optimisation iterations per :meth:`kernel` call.
            **vqe_kwargs: Further :class:`VQE` arguments, e.g. ``n_layers``.
        """
        self.backend = backend
        self.optimizer = optimizer
        self.ansatz = ansatz
        self.max_iterations = max_iterations
        self.vqe_kwargs = vqe_kwargs
        self.converged = False
        self._last_vqe: VQE | None = None
        self._last_energy = 0.0
        self._last_rdms: SpinRDMs | None = None

    def kernel(
        self,
        h1e: npt.ArrayLike,
        eri: npt.ArrayLike,
        norb: int,
        nelec: int | tuple[int, int],
        ci0: Any = None,
        ecore: float = 0.0,
        **kwargs: Any,
    ) -> tuple[float, VQE]:
        """Train a VQE on the active space and measure its energy and RDMs.

        Args:
            h1e: One-body integrals, ``(norb, norb)``.
            eri: Two-body integrals in chemist order, in any PySCF packing.
            norb: Active orbitals.
            nelec: Active electrons, as a total or ``(n_alpha, n_beta)``.
            ci0: Unused; the previous call's parameters seed the new run.
            ecore: Core energy.
            **kwargs: Further PySCF solver options, unused.

        Returns:
            The total energy from the measured RDMs, and the trained VQE.

        Raises:
            NotImplementedError: If ``nroots`` is not 1, or the integrals are
                unrestricted.
        """
        if self.nroots != 1:
            raise NotImplementedError(
                f"VQEFCISolver finds one root; got nroots={self.nroots}."
            )
        if np.ndim(h1e) != 2:
            raise NotImplementedError(
                "VQEFCISolver needs restricted integrals; got separate alpha and "
                "beta ones."
            )
        problem = MolecularProblem.from_active_space(h1e, eri, norb, nelec, ecore)
        vqe = VQE(
            problem,
            ansatz=self.ansatz,
            optimizer=self.optimizer.copy(),
            max_iterations=self.max_iterations,
            backend=self.backend,
            **self.vqe_kwargs,
        )
        vqe.run(
            initial_params=self._build_initial_params(vqe),
            perform_final_computation=False,
        )
        rdms = vqe.measure_rdms()
        dm1, dm2 = sum_spin_rdms(rdms)
        energy = float(
            problem.constant
            + np.einsum("pq,pq", problem.one_body, dm1)
            + 0.5 * np.einsum("pqrs,pqrs", problem.two_body, dm2)
        )
        result = vqe.optimize_result
        self.converged = result is not None and bool(result.success)
        self._last_vqe, self._last_energy, self._last_rdms = vqe, energy, rdms
        return energy, vqe

    def approx_kernel(
        self,
        h1e: npt.ArrayLike,
        eri: npt.ArrayLike,
        norb: int,
        nelec: int | tuple[int, int],
        ci0: Any = None,
        ecore: float = 0.0,
        **kwargs: Any,
    ) -> tuple[float, VQE]:
        """The last :meth:`kernel` result, for CASSCF's approximate CI steps.

        Runs :meth:`kernel` when there is no last result.
        """
        if self._last_vqe is None:
            return self.kernel(h1e, eri, norb, nelec, ci0, ecore, **kwargs)
        return self._last_energy, self._last_vqe

    def _build_initial_params(self, vqe: VQE) -> npt.NDArray[np.float64] | None:
        """The last run's best parameters, or None when there is no last run
        or its parameter count differs."""
        if self._last_vqe is None:
            return None
        params = self._last_vqe.best_params
        if params.shape != (vqe.n_params,):
            return None
        return params

    def make_rdm12s(self, ci: VQE, norb: int, nelec: int | tuple[int, int]) -> SpinRDMs:
        """Spin RDMs of ``ci``, in :meth:`VQE.measure_rdms`'s layout."""
        if ci is self._last_vqe and self._last_rdms is not None:
            return copy.deepcopy(self._last_rdms)
        return ci.measure_rdms()

    def make_rdm1s(
        self, ci: VQE, norb: int, nelec: int | tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Alpha and beta 1-RDMs of ``ci``."""
        return self.make_rdm12s(ci, norb, nelec)[0]

    def make_rdm12(
        self, ci: VQE, norb: int, nelec: int | tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Spin-summed 1- and 2-RDMs of ``ci``."""
        return sum_spin_rdms(self.make_rdm12s(ci, norb, nelec))

    def make_rdm1(self, ci: VQE, norb: int, nelec: int | tuple[int, int]) -> np.ndarray:
        """Spin-summed 1-RDM of ``ci``."""
        dm1a, dm1b = self.make_rdm1s(ci, norb, nelec)
        return dm1a + dm1b

    def spin_square(
        self, ci: VQE, norb: int, nelec: int | tuple[int, int]
    ) -> tuple[float, float]:
        """``<S^2>`` and the multiplicity ``2S + 1`` of ``ci``."""
        spin_op = import_optional(
            "pyscf.fci.spin_op", extra="chem", capability="VQEFCISolver.spin_square"
        )
        return spin_op.spin_square(ci, norb, nelec, fcisolver=self)
