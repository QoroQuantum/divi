# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Quantum-selected configuration interaction as a program ensemble."""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from itertools import islice
from numbers import Integral
from typing import NamedTuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from divi.backends import CircuitRunner
from divi.qprog.ensemble import ProgramEnsemble, ReportingLevel
from divi.qprog.quantum_program import QuantumProgram

from ._subspace import ConfigurationSubspace, limbs_to_configs


class ExpansionRecord(NamedTuple):
    """The subspace after one expansion step.

    Attributes:
        subspace_size: Configurations in the subspace.
        energy: Lowest eigenvalue of the Hamiltonian projected onto it.
        pt2_correction: Epstein-Nesbet second-order correction to ``energy``.
    """

    subspace_size: int
    energy: float
    pt2_correction: float


@dataclass(frozen=True, eq=False)
class QSCIState:
    """Result of a :class:`QSCI` round.

    Attributes:
        energy: Lowest eigenvalue in the final subspace, a variational upper
            bound on the ground-state energy.
        configs: The final subspace's configurations, in eigenvector order.
        eigenvector: Amplitude of each configuration.
        records: One :class:`ExpansionRecord` per expansion step, starting with
            the sampled seed subspace.
    """

    energy: float = float("inf")
    configs: tuple[str, ...] = ()
    eigenvector: np.ndarray | None = None
    records: tuple[ExpansionRecord, ...] = ()

    @property
    def pt2_energy(self) -> float:
        """``energy`` plus the final subspace's PT2 correction. Not variational."""
        return self.energy + self.records[-1].pt2_correction


def extrapolate_pt2_to_zero(
    records: tuple[ExpansionRecord, ...] | list[ExpansionRecord],
    *,
    n_points: int | None = None,
    max_correction: float | None = None,
) -> tuple[float, float, float]:
    """Estimate the exact ground-state energy by extrapolating to zero PT2.

    The energy is close to linear in the second-order correction, and the
    correction vanishes as the subspace approaches completeness, so the fit
    evaluated at zero correction estimates the limit. Early records sit far off
    the line: restrict the window with ``n_points`` (the last N records) or
    ``max_correction`` (records with ``|PT2|`` at or below it, in Hartree).

    Returns:
        ``(energy, slope, r_squared)``. Check the r-squared; the estimate only
        means something where the relationship is linear.

    Raises:
        ValueError: If fewer than two records remain in the window.
    """
    selected = list(records)
    if max_correction is not None:
        selected = [r for r in selected if abs(r.pt2_correction) <= max_correction]
    if n_points is not None:
        selected = selected[-n_points:]
    if len(selected) < 2:
        raise ValueError(
            f"Need at least two records to extrapolate, got {len(selected)}; "
            "widen n_points or max_correction, or allow more expansions."
        )

    corrections = np.array([r.pt2_correction for r in selected])
    energies = np.array([r.energy for r in selected])
    if np.ptp(corrections) < 1e-15:
        return float(energies.mean()), 0.0, 1.0

    slope, intercept = np.polyfit(corrections, energies, 1)
    residual = float(np.sum((energies - (slope * corrections + intercept)) ** 2))
    total = float(np.sum((energies - energies.mean()) ** 2))
    return float(intercept), float(slope), 1.0 - residual / total if total > 0 else 1.0


class QSCI(ProgramEnsemble):
    """Quantum-selected configuration interaction.

    Runs the sampling programs once, takes the most frequently measured
    bitstrings as a seed subspace, then grows it by the determinants the
    Hamiltonian connects to it most strongly, ranked by their Epstein-Nesbet
    second-order contribution. Each step diagonalises the Hamiltonian projected
    onto the subspace, which is done classically, so the register width is not
    limited by a statevector. Every step, the seed included, costs one pass of
    the Hamiltonian's terms over the subspace to rank the candidates.

    ``config_filter`` restricts only the seed. A Hamiltonian that conserves a
    quantity never connects a configuration to one that breaks it, so
    expansion from a filtered seed stays inside the same sector.

    Args:
        hamiltonian: Hermitian qubit Hamiltonian, of any width.
        program_factory: Builds the sampling programs from the backend, keyed by
            program id. Each must expose ``probabilities()``, as
            :class:`~divi.qprog.algorithms.TimeEvolution` does with no
            observable, and its bitstrings must give qubit ``q`` at character
            ``q``.
        backend: Backend the sampling programs run on.
        config_filter: Keeps a sampled bitstring in the seed when it returns
            true. ``None`` keeps all.
        max_seed_configs: Most-probable sampled bitstrings admitted to the seed.
        n_expansions: Expansion steps after the seed. ``0`` is plain QSCI.
        expansion_width: Configurations added per step.
        max_subspace_size: Stops expansion once the subspace reaches this size.
        epsilon_wf: Configurations whose eigenvector amplitude is at most this
            are pruned after each step.
        delta_conv: Expansion stops once a step lowers the energy by no more
            than this (Hartree).
        reporting_level: Forwarded to :class:`~divi.qprog.ProgramEnsemble`.

    Raises:
        ValueError: If a count is not a positive integer, or ``n_expansions``
            is negative; or if ``hamiltonian`` is empty or not Hermitian.
    """

    def __init__(
        self,
        hamiltonian: SparsePauliOp,
        program_factory: Callable[[CircuitRunner], dict[str, QuantumProgram]],
        *,
        backend: CircuitRunner,
        config_filter: Callable[[str], bool] | None = None,
        max_seed_configs: int = 1000,
        n_expansions: int = 10,
        expansion_width: int = 100,
        max_subspace_size: int = 50_000,
        epsilon_wf: float = 1e-5,
        delta_conv: float = 1e-6,
        reporting_level: ReportingLevel = ReportingLevel.COMPACT,
    ):
        for name, value, minimum in (
            ("max_seed_configs", max_seed_configs, 1),
            ("n_expansions", n_expansions, 0),
            ("expansion_width", expansion_width, 1),
            ("max_subspace_size", max_subspace_size, 1),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < minimum
            ):
                raise ValueError(
                    f"{name} must be an integer of at least {minimum}; got {value!r}."
                )
        super().__init__(backend=backend, reporting_level=reporting_level)

        self.hamiltonian = hamiltonian
        self.program_factory = program_factory
        self.config_filter = config_filter
        self.max_seed_configs = max_seed_configs
        self.n_expansions = n_expansions
        self.expansion_width = expansion_width
        self.max_subspace_size = max_subspace_size
        self.epsilon_wf = epsilon_wf
        self.delta_conv = delta_conv
        self._subspace = ConfigurationSubspace(hamiltonian)

    def initial_state(self) -> QSCIState:
        return QSCIState()

    def is_complete(self, state: QSCIState) -> bool:
        return state.eigenvector is not None

    def create_programs(self, state: QSCIState | None = None):
        super().create_programs()
        self._programs.update(self.program_factory(self.backend))

    def _seed_configs(self) -> list[str]:
        sampled: Counter[str] = Counter()
        for program in self.programs.values():
            sampled.update(program.probabilities())
        ranked = (
            config
            for config, _ in sampled.most_common()
            if self.config_filter is None or self.config_filter(config)
        )
        seed = list(islice(ranked, self.max_seed_configs))
        if not seed:
            raise ValueError("No sampled configuration passed config_filter.")
        return seed

    def update_state(self, state: QSCIState) -> QSCIState:
        """Seed the subspace from this round's samples and expand it."""
        subspace = self._subspace
        subspace.clear()
        subspace.add_configurations(self._seed_configs())
        subspace.extend_hamiltonian_matrix()
        subspace.solve_eigensystem()

        records: list[ExpansionRecord] = []
        while True:
            dets, contributions = subspace.external_contributions()
            records.append(
                ExpansionRecord(
                    len(subspace.configs), subspace.energy, float(contributions.sum())
                )
            )
            if (
                len(records) > self.n_expansions
                or len(subspace.configs) >= self.max_subspace_size
                or (
                    len(records) > 1
                    and records[-2].energy - records[-1].energy <= self.delta_conv
                )
            ):
                break

            width = min(
                self.expansion_width, self.max_subspace_size - len(subspace.configs)
            )
            strongest = np.argsort(-np.abs(contributions))[:width]
            if not subspace.add_configurations(
                limbs_to_configs(dets[strongest], subspace.n_qubits)
            ):
                break
            subspace.extend_hamiltonian_matrix()
            subspace.solve_eigensystem(warm_start=True)
            if subspace.prune(self.epsilon_wf):
                subspace.solve_eigensystem(warm_start=True)

        return QSCIState(
            subspace.energy,
            tuple(subspace.configs),
            subspace.eigenvector,
            tuple(records),
        )

    def aggregate_results(self) -> QSCIState:
        super().aggregate_results()
        return self.workflow_state
