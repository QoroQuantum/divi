# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import copy
from collections.abc import Sequence

import matplotlib.pyplot as plt
import pennylane as qml

from divi.backends import CircuitRunner
from divi.hamiltonians import TrotterizationStrategy
from divi.qprog.algorithms import InitialState, TimeEvolution
from divi.qprog.ensemble import ProgramEnsemble


class TimeEvolutionTrajectory(ProgramEnsemble):
    """Run TimeEvolution across multiple time points in parallel.

    Creates one :class:`~divi.qprog.algorithms.TimeEvolution` program per
    time point, executes them via :class:`ProgramEnsemble` (with optional
    batch-merged circuit submission), and aggregates results into a
    time-ordered mapping.

    Example::

        trajectory = TimeEvolutionTrajectory(
            hamiltonian=qml.PauliX(0) + qml.PauliX(1),
            time_points=[0.0, 0.5, 1.0, 1.5],
            backend=backend,
        )
        trajectory.create_programs()
        trajectory.run(blocking=True)
        results = trajectory.aggregate_results()
        # results: {0.0: {...}, 0.5: {...}, 1.0: {...}, 1.5: {...}}
    """

    _show_progress = True

    def __init__(
        self,
        hamiltonian: qml.operation.Operator,
        time_points: Sequence[float],
        *,
        backend: CircuitRunner,
        trotterization_strategy: TrotterizationStrategy | None = None,
        n_steps: int = 1,
        order: int = 1,
        initial_state: InitialState | None = None,
        observable: qml.operation.Operator | None = None,
        seed: int | None = None,
    ):
        """Initialize TimeEvolutionTrajectory.

        Args:
            hamiltonian: Hamiltonian to evolve under.
            time_points: List of evolution times. One program is created per
                time point.
            backend: Quantum circuit execution backend.
            trotterization_strategy: Strategy for term selection
                (``ExactTrotterization``, ``QDrift``). Defaults to ``ExactTrotterization()``.
                Deep-copied per program for thread safety.
            n_steps: Number of Trotter steps.
            order: Suzuki-Trotter order (1 or even).
            initial_state: Initial state preparation (:class:`InitialState` instance).
                Defaults to ``ZerosState()`` if None.
            observable: If None, measure probabilities; else expectation value.
            seed: Random seed for reproducible results.
        """
        super().__init__(backend=backend)

        time_points = list(time_points)
        if len(time_points) == 0:
            raise ValueError("time_points must not be empty.")
        if len(set(time_points)) != len(time_points):
            raise ValueError("time_points must not contain duplicates.")

        self._hamiltonian = hamiltonian
        self._time_points = time_points
        self._trotterization_strategy = trotterization_strategy
        self._n_steps = n_steps
        self._order = order
        self._initial_state = initial_state
        self._observable = observable
        self._seed = seed

    def create_programs(self):
        """Create one TimeEvolution program per time point."""
        super().create_programs()

        for t in self._time_points:
            prog_id = f"t={t}"
            self._programs[prog_id] = TimeEvolution(
                hamiltonian=self._hamiltonian,
                trotterization_strategy=copy.deepcopy(self._trotterization_strategy),
                time=t,
                n_steps=self._n_steps,
                order=self._order,
                initial_state=self._initial_state,
                observable=self._observable,
                backend=self.backend,
                seed=self._seed,
                program_id=prog_id,
                progress_queue=self._queue,
            )

    def aggregate_results(self) -> dict[float, dict | float]:
        """Aggregate results into a time-ordered mapping.

        Returns:
            dict mapping each time point to its result. The value is a
            ``dict[str, float]`` of probabilities when no observable is set,
            or a ``float`` expectation value otherwise.
        """
        self._check_ready_for_aggregation()
        return {t: self._programs[f"t={t}"].results for t in self._time_points}

    def visualize_results(self):
        """Plot the expectation-value trajectory over time.

        Requires that the trajectory was run with an ``observable``.

        Raises:
            RuntimeError: If no observable was set (probability mode).
        """
        if self._observable is None:
            raise RuntimeError(
                "plot() requires an observable. "
                "Set observable= when creating the trajectory."
            )

        results = self.aggregate_results()
        times = list(results.keys())
        values = list(results.values())

        plt.plot(times, values, marker="o")
        plt.xlabel("Time")
        plt.ylabel("Expectation value")
        plt.title("Time Evolution Trajectory")
        plt.show()
