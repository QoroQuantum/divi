# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import warnings
import weakref
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from threading import Event, Lock
from typing import TYPE_CHECKING, Any

from divi.circuits._payloads import CircuitBatch, CircuitPayload, bound_circuits
from divi.exceptions import ExecutionCancelledError

if TYPE_CHECKING:
    # For type checkers, assume maestro is always available — runtime code
    # that uses it is gated behind MaestroSimulator.__init__'s availability
    # check, so the type-checker view matches post-init invariants.
    # pyrefly: ignore[missing-import]  # ``maestro`` ships separately
    import maestro

    _maestro_import_error: ImportError | None = None
else:
    try:
        import maestro

        _maestro_import_error = None
    except ImportError as _err:
        maestro = None
        _maestro_import_error = _err

from .._base import CircuitRunner, ExecutionResult
from .._cancellation import raise_if_cancelled
from .._maestro_protocol import (
    MPS_AUTO_BOND_DIMENSION,
    MPS_QUBIT_THRESHOLD,
    counts_to_little_endian,
    expvals_from_result,
    qasm_n_qubits,
    strip_id_gates,
)
from .._pauli_serde import ham_ops_terms_for_circuit
from .._shot_allocation import per_circuit_or_none

logger = logging.getLogger(__name__)


def _run_with_cancellation(
    executor: ThreadPoolExecutor,
    fn: Callable[[Any], Any],
    items: Iterable[Any],
    cancellation_event: Event | None,
) -> list:
    """Run ``fn`` over ``items`` in order via ``executor`` with cancellation support.

    When ``cancellation_event`` is set between completed items, ``Future.cancel()``
    is called on every remaining future so unstarted ones never run — the
    shared per-instance ThreadPoolExecutor doesn't keep draining orphan work
    behind subsequent ``submit_circuits`` calls. Workers already in maestro's
    native call cannot be interrupted.
    """
    if cancellation_event is None:
        return list(executor.map(fn, items))
    futures = [executor.submit(fn, item) for item in items]
    out: list = []
    for fut in futures:
        if cancellation_event.is_set():
            # Inline the cleanup so a worker exception (any type — including
            # a hypothetical ExecutionCancelledError from a future Python
            # callable) propagates from ``fut.result()`` below without being
            # confused with our event-driven cancel.
            for f in futures:
                f.cancel()
            raise ExecutionCancelledError(
                "Maestro batch cancelled after partial completion"
            )
        out.append(fut.result())
    return out


def _resolve_noise_realizations(
    realizations: int | None, *, sampling: bool
) -> int | None:
    """Validate ``noise_realizations`` and resolve the backend dispatch.

    Returns ``None`` to signal "use the analytical backend" (expval only),
    or a positive ``int`` to forward to a Monte-Carlo entry point.

    Sampling has no analytical equivalent, so ``None`` collapses to ``1``
    (a single noise realisation).  Maestro's native default for
    ``noisy_execute`` is 64; divi deliberately uses 1 to avoid silently
    multiplying the shot budget.

    Zero or negative values raise ``ValueError`` — they have no meaningful
    MC interpretation and ``None`` already covers the analytical path.
    """
    if realizations is None:
        return 1 if sampling else None
    if realizations < 1:
        raise ValueError(
            f"noise_realizations must be None or a positive integer, got {realizations}."
        )
    return realizations


@dataclass(frozen=True)
class MaestroConfig:
    """Configuration object for :class:`MaestroSimulator`.

    Each field maps directly to an identically-named field on
    ``maestro.SimulatorConfig``; see the `maestro Python bindings guide
    <https://qoroquantum.github.io/maestro/d7/d01/python_guide.html#py_config>`_
    for the underlying semantics of each knob.  :attr:`mps_qubit_threshold`
    is Divi-specific and drives automatic Statevector → MatrixProductState
    selection.

    ``simulator_type`` and ``simulation_type`` accept the string names of the
    corresponding maestro enum members, e.g. ``"QCSim"``, ``"Gpu"``,
    ``"Statevector"``, ``"MatrixProductState"``.  ``None`` means "use maestro's
    default".
    """

    simulator_type: str | None = None
    """Maestro simulator type, e.g. ``"QCSim"`` or ``"Gpu"``.  ``None`` uses
    maestro's default (``"QCSim"``)."""

    simulation_type: str | None = None
    """Simulation method, e.g. ``"Statevector"`` or ``"MatrixProductState"``.
    ``None`` enables automatic selection based on qubit count."""

    max_bond_dimension: int | None = None
    """Maximum bond dimension for MPS simulation.  ``None`` uses maestro's
    default, except when auto-MPS is triggered (in which case 64 is used)."""

    singular_value_threshold: float | None = None
    """SVD truncation threshold for MPS simulation.  ``None`` uses maestro's
    default."""

    use_double_precision: bool = False
    """Use double-precision floating point.  Applies to the GPU MPS and
    tensor-network simulators; CPU simulation is already double precision."""

    precision: bool | None = None
    """Precision for Qiskit Aer — ``True`` selects double, ``False`` single, and
    ``None`` uses maestro's default.  Separate from
    :attr:`use_double_precision`, which covers the GPU simulators."""

    disable_optimized_swapping: bool = False
    """Disable MPS swap-cost optimisation."""

    lookahead_depth: int = -1
    """Lookahead depth for the MPS swap optimizer.  ``-1`` is maestro's default."""

    mps_measure_no_collapse: bool = True
    """If ``True``, use the non-collapsing MPS measurement algorithm; if
    ``False``, use the collapsing one."""

    pp_coefficient_threshold: float | None = None
    """Pauli-propagation coefficient truncation threshold.  Inert unless a trim
    or deduplication cadence is set."""

    pp_pauli_weight_threshold: int | None = None
    """Pauli-propagation maximum Pauli weight retained.  Ignored when at or
    above the qubit count, and inert unless a cadence is set."""

    pp_steps_between_trims: int | None = None
    """Gates between Pauli-propagation truncation passes, which drop each string
    independently.  Cheaper but markedly less accurate than
    :attr:`pp_steps_between_deduplications` at the same threshold."""

    pp_steps_between_deduplications: int | None = None
    """Gates between deduplication passes, which merge identical Pauli strings
    before applying the thresholds.  Preferred cadence when accuracy matters,
    and it takes precedence on gates where both cadences are due."""

    path_integral_threshold: float | None = None
    """Trim threshold for PathIntegral simulation.  ``None`` uses maestro's
    default (no trimming)."""

    mps_qubit_threshold: int = MPS_QUBIT_THRESHOLD
    """Qubit count above which automatic MPS selection kicks in.  Only active
    when :attr:`simulation_type` is ``None``; has no effect when
    ``simulation_type`` is set explicitly.  Divi-specific; not forwarded to
    ``maestro.SimulatorConfig``."""

    noise_model: "maestro.NoiseModel | None" = None
    """Maestro ``NoiseModel`` object.  ``None`` (default) disables noise —
    circuits run via ``simple_execute`` (sampling) or ``simple_estimate``
    (expval).  When set, dispatch routes to ``noisy_execute`` /
    ``noisy_estimate`` / ``noisy_estimate_montecarlo`` depending on
    :attr:`noise_realizations`.  Divi-specific; not forwarded to
    ``maestro.SimulatorConfig`` — Maestro keeps noise models separate from
    simulator config and accepts them positionally on the noisy entry points."""

    noise_seed: int = 42
    """Seed for Pauli-error sampling.  Consulted whenever execution routes
    through one of Maestro's stochastic noisy entry points
    (``noisy_execute`` or ``noisy_estimate_montecarlo``); the analytical
    ``noisy_estimate`` path ignores it.  Each circuit in a
    :meth:`MaestroSimulator.submit_circuits` batch is seeded with
    ``noise_seed + i`` (where ``i`` is the circuit's index in the input
    mapping) so circuits in the same batch get independent error patterns.

    Reproducibility scope: the seed pins the **Pauli error patterns**
    sampled from the noise model.  Expectation-value runs
    (``noisy_estimate_montecarlo``) are fully reproducible because the
    inner loop is analytical.  Noisy *sampling* runs (``noisy_execute``)
    are only partially reproducible: the same Pauli errors are injected,
    but the shot-count outcomes still vary across runs because Maestro's
    measurement sampler initialises its own RNG from system entropy on
    every call.

    Divi-specific; not forwarded to ``maestro.SimulatorConfig``."""

    noise_realizations: int | None = None
    """Number of Monte-Carlo noise realizations.  ``None`` (default) selects
    the analytical noisy backend when available:

    * **Expval** — ``maestro.noisy_estimate``, which applies exact Pauli
      damping coefficients to noiseless expectation values.  Deterministic.
    * **Sampling** — no analytical equivalent; falls back to one realisation
      (``noisy_execute`` with ``noise_realizations=1``).

    A positive ``int`` ``N`` selects Monte-Carlo backends:

    * **Expval** — ``maestro.noisy_estimate_montecarlo``, which runs ``N``
      independent Pauli-injection passes and averages the expectation values.
    * **Sampling** — ``maestro.noisy_execute``, which divides ``shots``
      across ``min(shots, N)`` batches, each with a freshly sampled noise
      pattern.  Total shot count is always ``shots``; if ``N > shots`` the
      effective realisation count is capped at ``shots``.

    Note that ``noise_realizations=1`` is **not** equivalent to ``None``
    for expval — the former is one random Pauli sampling, the latter is
    the exact analytical average.  Divi-specific; not forwarded to
    ``maestro.SimulatorConfig``."""

    def __post_init__(self):
        """Validate the Pauli-propagation knobs and warn about no-op combinations."""
        if (
            self.pp_coefficient_threshold is not None
            and self.pp_coefficient_threshold < 0
        ):
            raise ValueError(
                "pp_coefficient_threshold must be non-negative. "
                f"Got {self.pp_coefficient_threshold}."
            )

        if (
            self.pp_pauli_weight_threshold is not None
            and self.pp_pauli_weight_threshold < 0
        ):
            raise ValueError(
                "pp_pauli_weight_threshold must be non-negative. "
                f"Got {self.pp_pauli_weight_threshold}."
            )

        for name in ("pp_steps_between_trims", "pp_steps_between_deduplications"):
            cadence = getattr(self, name)
            # Maestro takes these modulo a gate index, so 0 divides by zero and
            # aborts the process with SIGFPE rather than raising.
            if cadence is not None and cadence < 1:
                raise ValueError(f"{name} must be a positive integer. Got {cadence}.")

    def override(self, other: "MaestroConfig") -> "MaestroConfig":
        """Return a new config overriding fields with non-default values from ``other``.

        "Non-default" here means a field whose value differs from the class
        default. :meth:`~divi.backends.ExecutionConfig.override` instead takes
        every field that is not ``None``; the two differ for any field whose
        default is something other than ``None``.
        """
        defaults = {f.name: f.default for f in fields(MaestroConfig)}
        merged = {f.name: getattr(self, f.name) for f in fields(MaestroConfig)}

        for f in fields(MaestroConfig):
            other_value = getattr(other, f.name)
            # Relies on != with the default sentinel.  Safe for scalar fields and
            # for noise_model because None is the default — any non-None object
            # evaluates != None as True.  If two non-None NoiseModel instances ever
            # need to be distinguished by value equality this logic would need
            # an identity check (``is not``) instead.
            if other_value != defaults[f.name]:
                merged[f.name] = other_value

        return MaestroConfig(**merged)

    def _resolve_simulation_type(self, n_qubits: int) -> str | None:
        """Choose simulation type based on qubit count when not explicitly set."""
        if self.simulation_type is not None:
            return self.simulation_type
        if n_qubits > self.mps_qubit_threshold:
            logger.info(
                "Circuit has %d qubits (> %d threshold), using MPS simulation.",
                n_qubits,
                self.mps_qubit_threshold,
            )
            return "MatrixProductState"
        return None

    def _to_maestro_config(self, n_qubits: int) -> "maestro.SimulatorConfig":
        """Build a ``maestro.SimulatorConfig`` for a batch of ``n_qubits`` circuits.

        Internal — the per-submission ``n_qubits`` drives auto-MPS selection.
        """
        kwargs: dict = {}

        if self.simulator_type is not None:
            kwargs["simulator_type"] = maestro.SimulatorType[self.simulator_type]

        resolved_sim_type = self._resolve_simulation_type(n_qubits)
        auto_mps = (
            self.simulation_type is None and resolved_sim_type == "MatrixProductState"
        )
        if resolved_sim_type is not None:
            kwargs["simulation_type"] = maestro.SimulationType[resolved_sim_type]

        if self.max_bond_dimension is not None:
            kwargs["max_bond_dimension"] = self.max_bond_dimension
        elif auto_mps:
            kwargs["max_bond_dimension"] = MPS_AUTO_BOND_DIMENSION

        if self.singular_value_threshold is not None:
            kwargs["singular_value_threshold"] = self.singular_value_threshold

        if self.use_double_precision:
            kwargs["use_double_precision"] = True

        if self.disable_optimized_swapping:
            kwargs["disable_optimized_swapping"] = True

        if self.lookahead_depth != -1:
            kwargs["lookahead_depth"] = self.lookahead_depth

        if not self.mps_measure_no_collapse:
            kwargs["mps_measure_no_collapse"] = False

        config = maestro.SimulatorConfig(**kwargs)

        # Maestro binds these as writable properties only; its constructor
        # does not accept them.
        property_settings = {
            "precision": self.precision,
            "pp_coefficient_threshold": self.pp_coefficient_threshold,
            "pp_pauli_weight_threshold": self.pp_pauli_weight_threshold,
            "pp_steps_between_trims": self.pp_steps_between_trims,
            "pp_steps_between_deduplications": self.pp_steps_between_deduplications,
            "path_integral_threshold": self.path_integral_threshold,
        }
        for name, value in property_settings.items():
            if value is not None:
                setattr(config, name, value)

        # Warned here, not in __post_init__: a config is also an override delta,
        # where a threshold and its cadence can arrive from opposite sides.
        thresholds_set = (
            self.pp_coefficient_threshold is not None
            or self.pp_pauli_weight_threshold is not None
        )
        cadence_set = (
            self.pp_steps_between_trims is not None
            or self.pp_steps_between_deduplications is not None
        )

        if thresholds_set and not cadence_set:
            warnings.warn(
                "pp_coefficient_threshold and pp_pauli_weight_threshold are only "
                "consulted during a truncation pass, so they have no effect unless "
                "pp_steps_between_deduplications or pp_steps_between_trims is set.",
                stacklevel=3,
            )

        if (thresholds_set or cadence_set) and resolved_sim_type not in (
            None,
            "PauliPropagator",
        ):
            warnings.warn(
                "The pp_* options only apply to PauliPropagator simulations; they "
                f"will be ignored with simulation_type={resolved_sim_type!r}.",
                stacklevel=3,
            )

        if (
            self.pp_pauli_weight_threshold is not None
            and self.pp_pauli_weight_threshold >= n_qubits
        ):
            warnings.warn(
                f"pp_pauli_weight_threshold={self.pp_pauli_weight_threshold} is at or "
                f"above the circuit's {n_qubits} qubits, which disables weight "
                "filtering.",
                stacklevel=3,
            )

        return config


def _shutdown_executor(executor: ThreadPoolExecutor) -> None:
    """Module-level finalizer callback for the per-instance fan-out pool.

    Lives at module scope (rather than as a method) so the
    :class:`weakref.finalize` registration does not capture a strong
    reference to the simulator instance, which would defeat GC.
    """
    executor.shutdown(wait=False)


class MaestroSimulator(CircuitRunner):
    """A CircuitRunner backend powered by qoro-maestro, Qoro's C++ quantum simulator.

    Supports multiple simulation methods (Statevector, MPS, Stabilizer, TensorNetwork,
    PauliPropagator), intelligent auto-routing, GPU acceleration, and native observable
    estimation.

    All maestro-level configuration — including noise — is carried in a
    :class:`MaestroConfig` object rather than as loose keyword arguments,
    matching the
    :class:`~divi.backends.ExecutionConfig` / :class:`~divi.backends.QoroService`
    pattern.  Pass a ``maestro.NoiseModel`` via :attr:`MaestroConfig.noise_model`
    (and tune :attr:`MaestroConfig.noise_seed` and
    :attr:`MaestroConfig.noise_realizations`) to route execution through
    Maestro's noisy entry points.

    .. note::

        Maestro's C++ extension must be loaded before other C++ libraries
        (Qiskit, PennyLane) to avoid initialisation order conflicts.  This
        is handled automatically by ``divi/__init__.py``.

    Args:
        shots: Number of measurement shots. Defaults to 5000.
        config: :class:`MaestroConfig` controlling simulator backend, simulation
            method, bond dimension, noise model, and related knobs.  Defaults
            to ``MaestroConfig()``.
        track_depth: Record circuit depth per submission. Defaults to False.
    """

    def __init__(
        self,
        shots: int = 5000,
        config: MaestroConfig | None = None,
        track_depth: bool = False,
    ):
        if maestro is None:
            raise ImportError(
                "qoro-maestro is required for MaestroSimulator but could not be imported."
            ) from _maestro_import_error

        super().__init__(shots=shots, track_depth=track_depth)
        self.config: MaestroConfig = config if config is not None else MaestroConfig()

        # Per-instance circuit fan-out pool, lazy-initialised on first
        # ``submit_circuits`` call.  Maestro's C++ entrypoints release the
        # GIL and use internal OpenMP threads, so we cap workers at cores/2
        # to leave headroom for that internal parallelism rather than
        # oversubscribing.  ``ThreadPoolExecutor.map`` is thread-safe across
        # concurrent submit calls — overlapping submissions multiplex
        # through the same worker pool instead of each spawning their own.
        self._executor: ThreadPoolExecutor | None = None
        self._executor_lock = Lock()
        self._executor_finalizer: weakref.finalize | None = None

    @property
    def supports_expval(self) -> bool:
        """Maestro supports native observable estimation."""
        return True

    @property
    def is_async(self) -> bool:
        """Maestro executes circuits synchronously."""
        return False

    def set_seed(self, seed: int) -> None:
        """No-op — maestro does not yet expose seeding from C++."""

    def _get_executor(self) -> ThreadPoolExecutor:
        """Return the per-instance circuit fan-out pool, creating it lazily.

        Sized once at first use; callers that submit fewer tasks than the
        worker count simply leave the extra workers idle (no per-call cost).
        """
        with self._executor_lock:
            if self._executor is None:
                n_workers = max(1, (os.cpu_count() or 2) // 2)
                executor = ThreadPoolExecutor(
                    max_workers=n_workers,
                    thread_name_prefix="maestro",
                )
                # Finalizer: shut the pool down when the simulator is GC'd
                # so its threads don't outlive the instance.  Use a static
                # callable (no ``self`` reference) so the weakref can
                # actually be collected.
                self._executor = executor
                self._executor_finalizer = weakref.finalize(
                    self, _shutdown_executor, executor
                )
            return self._executor

    def close(self) -> None:
        """Shut down the per-instance executor.

        Safe to call multiple times.  Called automatically when the
        instance is garbage-collected via :class:`weakref.finalize`, but
        callers that want deterministic cleanup (e.g. inside long-running
        services) can invoke this explicitly.

        ``shutdown(wait=True)`` runs **outside** ``_executor_lock`` — a
        concurrent ``submit_circuits`` on another thread can grab the lock
        and lazily re-create a fresh pool while the old one drains, instead
        of serialising behind a slow shutdown.  Subsequent submits therefore
        observe ``close()`` as "release current pool; new pool created on
        demand".
        """
        with self._executor_lock:
            executor = self._executor
            finalizer = self._executor_finalizer
            # Detach the finalizer before zeroing attributes so a GC pass
            # interleaving these two writes can't fire the callback.
            if finalizer is not None:
                finalizer.detach()
            self._executor = None
            self._executor_finalizer = None
        if executor is not None:
            executor.shutdown(wait=True)

    def submit_circuits(
        self,
        payloads: Sequence[CircuitPayload] | CircuitBatch,
        *,
        ham_ops: str | None = None,
        circuit_ham_map: list[list[int]] | None = None,
        shot_groups: list[list[int]] | None = None,
        cancellation_event: Event | None = None,
        **kwargs,
    ) -> ExecutionResult:
        """Submit quantum circuits for execution on the maestro simulator.

        Args:
            payloads: Bound QASM payloads, one resolved circuit per parameter-set
                row — or a collection of already-resolved circuits.
            ham_ops: Semicolon-separated Pauli string for expectation value estimation,
                e.g. ``"ZI;IZ;XX"``. If None, runs in sampling mode.
            circuit_ham_map: Maps circuit index ranges to observable groups for
                heterogeneous batches. Each inner list contains circuit indices
                belonging to that observable group.
            shot_groups: Per-circuit shot allocation as ``[start, end, shots]``
                triples covering the iteration order of ``circuits``. Sampling
                mode only — ignored when ``ham_ops`` is provided because
                maestro's ``simple_estimate`` computes expectation values
                analytically.
            cancellation_event: When set, aborts further dispatch and raises
                :class:`~divi.exceptions.ExecutionCancelledError`. Workers
                already in maestro's native call are not interrupted.
            **kwargs: Ignored — accepted so callers using the generic
                :class:`~divi.backends.CircuitRunner` interface can forward
                unrelated options without breaking.

        Returns:
            ExecutionResult containing either counts (sampling) or expectation values.
        """
        raise_if_cancelled(
            cancellation_event,
            "Maestro batch cancelled before any circuit was dispatched",
        )
        self._reject_shot_groups_with_ham_ops(ham_ops, shot_groups)

        circuits = bound_circuits(payloads)
        circuit_labels = list(circuits.keys())
        qasm_strings = list(circuits.values())

        self._record_qasm_depths(qasm_strings)

        # Determine max qubit count for automatic simulation type selection.
        max_qubits = max(
            qasm_n_qubits(qasm, label)
            for label, qasm in zip(circuit_labels, qasm_strings)
        )

        # Pre-process: strip id gates (not supported by maestro's QASM parser).
        qasm_strings = [strip_id_gates(q) for q in qasm_strings]

        sim_config = self.config._to_maestro_config(n_qubits=max_qubits)

        executor = self._get_executor()

        if ham_ops is None:
            per_circuit_shots = per_circuit_or_none(shot_groups, len(circuit_labels))

            def _run_sample(item):
                i, label, qasm = item
                shots = (
                    per_circuit_shots[i]
                    if per_circuit_shots is not None
                    else self.shots
                )
                if self.config.noise_model is None:
                    raw = maestro.simple_execute(qasm, config=sim_config, shots=shots)
                else:
                    # Noisy functions require a parsed maestro Circuit, not a raw QASM string.
                    circuit_parser = maestro.QasmToCirc()
                    maestro_circuit = circuit_parser.parse_and_translate(qasm)
                    realizations = _resolve_noise_realizations(
                        self.config.noise_realizations, sampling=True
                    )
                    raw = maestro.noisy_execute(
                        maestro_circuit,
                        self.config.noise_model,
                        config=sim_config,
                        shots=shots,
                        noise_realizations=realizations,
                        # Derive a per-circuit seed so circuits in a batch don't
                        # all receive the same Pauli error pattern.
                        seed=self.config.noise_seed + i,
                    )
                return {
                    "label": label,
                    "results": counts_to_little_endian(raw["counts"]),
                }

            items = [
                (i, label, qasm)
                for i, (label, qasm) in enumerate(zip(circuit_labels, qasm_strings))
            ]
            results = _run_with_cancellation(
                executor, _run_sample, items, cancellation_event
            )
        else:
            # Expectation value mode — strip measurement gates so they don't
            # collapse the statevector before expectation values are computed.
            def _run_estimate(item):
                i, label, qasm = item
                terms = ham_ops_terms_for_circuit(i, ham_ops, circuit_ham_map)
                pauli_string = ";".join(terms)
                if self.config.noise_model is None:
                    raw = maestro.simple_estimate(
                        qasm,
                        observables=pauli_string,
                        config=sim_config,
                    )
                else:
                    # Noisy functions require a parsed maestro Circuit, not a raw QASM string.
                    circuit_parser = maestro.QasmToCirc()
                    maestro_circuit = circuit_parser.parse_and_translate(qasm)
                    realizations = _resolve_noise_realizations(
                        self.config.noise_realizations, sampling=False
                    )
                    if realizations is None:
                        raw = maestro.noisy_estimate(
                            maestro_circuit,
                            observables=pauli_string,
                            noise_model=self.config.noise_model,
                            config=sim_config,
                        )
                    else:
                        raw = maestro.noisy_estimate_montecarlo(
                            maestro_circuit,
                            observables=pauli_string,
                            noise_model=self.config.noise_model,
                            noise_realizations=realizations,
                            # Derive a per-circuit seed so circuits in a batch don't
                            # all receive the same Pauli error pattern.
                            seed=self.config.noise_seed + i,
                            config=sim_config,
                        )
                return {
                    "label": label,
                    "results": expvals_from_result(raw, terms),
                }

            items = [
                (i, label, qasm)
                for i, (label, qasm) in enumerate(zip(circuit_labels, qasm_strings))
            ]
            results = _run_with_cancellation(
                executor, _run_estimate, items, cancellation_event
            )

        return ExecutionResult(results=results)
