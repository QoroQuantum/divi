# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
from collections.abc import Sequence
from dataclasses import replace

import matplotlib.pyplot as plt
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp

from divi.backends import CircuitRunner
from divi.circuits import MetaCircuit
from divi.hamiltonians import ExactTrotterization, TrotterizationStrategy, to_spo
from divi.qprog.algorithms import InitialState, TimeEvolution
from divi.qprog.ensemble import ProgramEnsemble

logger = logging.getLogger(__name__)

# Below this many time points the cost of building the parametric template
# (one symbolic ``_meta_circuit_factory`` call) outweighs the savings from
# sharing it across programs.
_CACHE_MIN_TIME_POINTS = 4


class TimeEvolutionTrajectory(ProgramEnsemble):
    """Run TimeEvolution across multiple time points in parallel.

    Creates one :class:`~divi.qprog.algorithms.TimeEvolution` program per
    time point, executes them via :class:`~divi.qprog.ensemble.ProgramEnsemble` (with optional
    batch-merged circuit submission), and aggregates results into a
    time-ordered mapping.

    Example::

        trajectory = TimeEvolutionTrajectory(
            hamiltonian=to_spo({"XI": 1.0, "IX": 1.0}),
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
        hamiltonian: SparsePauliOp,
        time_points: Sequence[float],
        *,
        backend: CircuitRunner,
        trotterization_strategy: TrotterizationStrategy | None = None,
        n_steps: int = 1,
        order: int = 1,
        initial_state: InitialState | None = None,
        observable: SparsePauliOp | None = None,
        seed: int | None = None,
        **kwargs,
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
            initial_state: Initial state preparation (:class:`~divi.qprog.algorithms.InitialState` instance).
                Defaults to ``ZerosState()`` if None.
            observable: If None, measure probabilities; else expectation value.
            seed: Random seed for reproducible results.
            **kwargs: Forwarded verbatim to every per-time-point
                :class:`~divi.qprog.algorithms.TimeEvolution`.  Use this for
                ``grouping_strategy``, ``shot_distribution``, ``precision``,
                and any other ``QuantumProgram`` / ``ObservableMeasuringMixin``
                kwarg that should apply uniformly across the trajectory.
                ``program_id`` and ``progress_queue`` are set internally and
                must not be passed here.
        """
        super().__init__(backend=backend)

        time_points = list(time_points)
        if len(time_points) == 0:
            raise ValueError("time_points must not be empty.")
        if len(set(time_points)) != len(time_points):
            raise ValueError("time_points must not contain duplicates.")

        for reserved in ("program_id", "progress_queue"):
            if reserved in kwargs:
                raise TypeError(
                    f"TimeEvolutionTrajectory sets {reserved!r} internally; "
                    f"do not pass it via kwargs."
                )

        self._hamiltonian = hamiltonian
        self._time_points = time_points
        self._trotterization_strategy = trotterization_strategy
        self._n_steps = n_steps
        self._order = order
        self._initial_state = initial_state
        self._observable = observable
        self._seed = seed
        self._extra_kwargs = kwargs

    def create_programs(self):
        """Create one TimeEvolution program per time point."""
        super().create_programs()

        template_meta, t_param = self._maybe_build_template()

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
                _template_meta=template_meta,
                _template_param=t_param,
                **self._extra_kwargs,
            )

    def _maybe_build_template(self) -> tuple[MetaCircuit | None, Parameter | None]:
        """Build a parametric MetaCircuit shared across all time-point
        programs, or return ``(None, None)`` when caching does not apply.

        The template is built only when the trotterization strategy is
        :class:`ExactTrotterization` (QDrift's per-program random sampling
        means each program's circuit topology differs) and the number of
        time points clears :data:`_CACHE_MIN_TIME_POINTS`. Any exception
        raised by the symbolic-time probe is logged at WARNING and the
        trajectory falls back to per-program circuit construction.
        """
        strategy = self._trotterization_strategy
        if strategy is not None and not isinstance(strategy, ExactTrotterization):
            return None, None
        if len(self._time_points) < _CACHE_MIN_TIME_POINTS:
            return None, None

        t_param = Parameter("t")
        try:
            probe = TimeEvolution(
                hamiltonian=self._hamiltonian,
                trotterization_strategy=copy.deepcopy(self._trotterization_strategy),
                time=t_param,
                n_steps=self._n_steps,
                order=self._order,
                initial_state=self._initial_state,
                observable=self._observable,
                backend=self.backend,
                seed=self._seed,
                **self._extra_kwargs,
            )
            template = probe._meta_circuit_factory(to_spo(self._hamiltonian), ham_id=0)
        except Exception:
            logger.warning(
                "TimeEvolutionTrajectory: parametric template build failed; "
                "falling back to per-program circuit construction.",
                exc_info=True,
            )
            return None, None

        if not template.circuit_bodies:
            logger.warning(
                "TimeEvolutionTrajectory: probe produced an empty MetaCircuit; "
                "falling back to per-program circuit construction."
            )
            return None, None

        return replace(template, parameters=(t_param,)), t_param

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
        values = [float(v) for v in results.values() if isinstance(v, (int, float))]

        plt.plot(times, values, marker="o")
        plt.xlabel("Time")
        plt.ylabel("Expectation value")
        plt.title("Time Evolution Trajectory")
        plt.show()
