# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import copy
import logging
import os
import warnings
import weakref
import zlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from typing import Any

# pyrefly: ignore[missing-import]  # ``maestro`` ships as a compiled wheel
import maestro
from pydantic import BaseModel, ConfigDict, SkipValidation, model_validator

from divi.circuits._payloads import CircuitBatch, CircuitPayload, bound_circuits
from divi.exceptions import ExecutionCancelledError

from .._base import CircuitRunner, ExecutionResult
from .._cancellation import raise_if_cancelled
from .._maestro_protocol import (
    MPS_AUTO_BOND_DIMENSION,
    MPS_QUBIT_THRESHOLD,
    counts_to_little_endian,
    expvals_from_result,
    id_gates_as_noise_sites,
    qasm_n_qubits,
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


def _circuit_seed(seed: int | None, label: str) -> int | None:
    """Per-circuit seed keyed on the label, independent of batch position."""
    return None if seed is None else zlib.crc32(label.encode(), seed)


def _result_entry(
    label: str, results: Any, raw: Mapping[str, Any], result_key: str
) -> dict[str, Any]:
    """One circuit's result entry, with maestro's other outputs as metadata."""
    metadata = {key: value for key, value in raw.items() if key != result_key}
    return {"label": label, "results": results, "metadata": metadata}


_SEED_LIMIT = 2**32

# Simulator/simulation pairs on which Maestro applies noise as exact channels.
_EXACT_CHANNEL_BACKENDS = frozenset(
    {
        (None, "DensityMatrix"),
        (None, "MatrixProductOperator"),
        ("QCSim", "DensityMatrix"),
        ("QCSim", "MatrixProductOperator"),
        ("Gpu", "DensityMatrix"),
        ("Gpu", "MatrixProductOperator"),
        ("QiskitAer", "DensityMatrix"),
    }
)

TRUNCATION_MODES = ("relative_max", "discarded_weight")
KRAUS_COMPLETENESS_CHECKS = ("ignore", "warn", "strict")

# Maestro applies at most one SVD solver per backend, so two flags set in the
# same group would silently make the winner depend on maestro's ordering.
GPU_SVD_FLAG_GROUPS = tuple(
    tuple(
        f"{prefix}_use_{solver}" for solver in ("gesvd", "gesvdj", "gesvdp", "gesvdr")
    )
    for prefix in ("mps", "mpo", "tensor_network")
)

# Forwarded only when True — maestro's own defaults are False.
BOOLEAN_FLAG_FIELDS = (
    "mpo_restore_trace_after_truncation",
    "mpo_hermitize_after_truncation",
    *(name for group in GPU_SVD_FLAG_GROUPS for name in group),
)


class MaestroConfig(BaseModel):
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

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

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

    truncation_mode: str | None = None
    """SVD truncation convention for MPS and MPO simulation — ``"relative_max"``
    (keep singular values above ``singular_value_threshold`` times the largest)
    or ``"discarded_weight"`` (discard the smallest until their cumulative
    squared weight reaches the threshold).  ``None`` uses maestro's default,
    ``"discarded_weight"``.  Only QCSim and the GPU backend support
    ``"relative_max"``; Qiskit Aer raises if it is requested."""

    seed: int | None = None
    """Seed for maestro's simulation.  Each circuit gets its own seed derived
    from this one and its label.  ``None`` seeds from system entropy."""

    gpu_device: int | None = None
    """CUDA-visible device ordinal for the ``"Gpu"`` simulator type.  ``None``
    uses maestro's default device."""

    distributed_options: dict[str, str] | None = None
    """Settings for the ``"DistributedGpu"`` and ``"DistributedMpiGpu"``
    simulator types, applied before the state is allocated.  Keys start with
    ``distributed_`` or ``mpi_`` and values are strings, e.g.
    ``{"distributed_devices": "0,1"}``; an MPI communicator is passed as
    ``{"mpi_communicator": str(comm.py2f())}``.  ``None`` uses maestro's
    defaults."""

    mpo_kraus_completeness_check: str | None = None
    """How the MPO simulator reacts to Kraus operators that do not sum to the
    identity — ``"ignore"``, ``"warn"`` or ``"strict"`` (raise).  ``None`` uses
    maestro's default."""

    mpo_restore_trace_after_truncation: bool = False
    """Rescale the MPO to unit trace after each truncation pass."""

    mpo_hermitize_after_truncation: bool = False
    """Make the MPO Hermitian again after each truncation pass."""

    mps_use_gesvd: bool = False
    """Select the ``gesvd`` GPU SVD solver for MPS truncation."""

    mps_use_gesvdj: bool = False
    """Select the Jacobi ``gesvdj`` GPU SVD solver for MPS truncation."""

    mps_use_gesvdp: bool = False
    """Select the polar ``gesvdp`` GPU SVD solver for MPS truncation."""

    mps_use_gesvdr: bool = False
    """Select the randomised ``gesvdr`` GPU SVD solver for MPS truncation."""

    mpo_use_gesvd: bool = False
    """Select the ``gesvd`` GPU SVD solver for MPO truncation."""

    mpo_use_gesvdj: bool = False
    """Select the Jacobi ``gesvdj`` GPU SVD solver for MPO truncation."""

    mpo_use_gesvdp: bool = False
    """Select the polar ``gesvdp`` GPU SVD solver for MPO truncation."""

    mpo_use_gesvdr: bool = False
    """Select the randomised ``gesvdr`` GPU SVD solver for MPO truncation."""

    tensor_network_use_gesvd: bool = False
    """Select the ``gesvd`` GPU SVD solver for tensor-network truncation."""

    tensor_network_use_gesvdj: bool = False
    """Select the Jacobi ``gesvdj`` GPU SVD solver for tensor-network
    truncation."""

    tensor_network_use_gesvdp: bool = False
    """Select the polar ``gesvdp`` GPU SVD solver for tensor-network
    truncation."""

    tensor_network_use_gesvdr: bool = False
    """Select the randomised ``gesvdr`` GPU SVD solver for tensor-network
    truncation."""

    mps_qubit_threshold: int = MPS_QUBIT_THRESHOLD
    """Qubit count above which automatic MPS selection kicks in.  Only active
    when :attr:`simulation_type` is ``None``; has no effect when
    ``simulation_type`` is set explicitly.  Divi-specific; not forwarded to
    ``maestro.SimulatorConfig``."""

    noise_model: SkipValidation["maestro.NoiseModel | None"] = None
    """Maestro ``NoiseModel`` to simulate with.  When set, circuits run through
    ``maestro.full_noise_execute`` (sampling) or ``maestro.full_noise_estimate``
    (expectation values); ``None`` runs them noiseless."""

    noise_seed: int | None = None
    """Seed passed to Maestro's noisy entry points, derived per circuit like
    :attr:`seed`.  ``None`` leaves the noise seeded by :attr:`seed`."""

    noise_realizations: int | None = None
    """``noise_realizations`` passed to Maestro's noisy entry points.  ``None``
    uses one on backends where every channel of the model is exact, and
    Maestro's default otherwise."""

    @model_validator(mode="after")
    def _validate_knobs(self):
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

        for name, allowed in (
            ("truncation_mode", TRUNCATION_MODES),
            ("mpo_kraus_completeness_check", KRAUS_COMPLETENESS_CHECKS),
        ):
            value = getattr(self, name)
            if value is not None and value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}. Got {value!r}.")

        if self.gpu_device is not None and self.gpu_device < 0:
            raise ValueError(
                f"gpu_device must be a non-negative integer. Got {self.gpu_device}."
            )

        for name in ("seed", "noise_seed"):
            value = getattr(self, name)
            if value is not None and not 0 <= value < _SEED_LIMIT:
                raise ValueError(
                    f"{name} must be an integer in [0, 2**32). Got {value}."
                )

        if self.noise_realizations is not None and self.noise_realizations < 1:
            raise ValueError(
                "noise_realizations must be None or a positive integer. "
                f"Got {self.noise_realizations}."
            )

        if self.noise_model is not None and not isinstance(
            self.noise_model, maestro.NoiseModel
        ):
            raise ValueError(
                "noise_model must be a maestro.NoiseModel. "
                f"Got {type(self.noise_model).__name__}."
            )

        if self.distributed_options is not None:
            invalid = sorted(
                key
                for key in self.distributed_options
                if not key.startswith(("distributed_", "mpi_"))
            )
            if invalid:
                raise ValueError(
                    "distributed_options keys must start with 'distributed_' or "
                    f"'mpi_'. Got {invalid}."
                )

        for group in GPU_SVD_FLAG_GROUPS:
            enabled = [name for name in group if getattr(self, name)]
            if len(enabled) > 1:
                raise ValueError(
                    f"At most one of {group} may be set. Got {tuple(enabled)}."
                )

        return self

    def override(self, other: "MaestroConfig") -> "MaestroConfig":
        """Return a new config overriding fields with non-default values from ``other``.

        "Non-default" here means a field whose value differs from the class
        default. :meth:`~divi.backends.ExecutionConfig.override` instead takes
        every field that is not ``None``; the two differ for any field whose
        default is something other than ``None``.
        """
        merged = dict(self)

        for name, spec in MaestroConfig.model_fields.items():
            other_value = getattr(other, name)
            # Relies on != with the default sentinel.  Safe for scalar fields and
            # for noise_model because None is the default — any non-None object
            # evaluates != None as True.  If two non-None NoiseModel instances ever
            # need to be distinguished by value equality this logic would need
            # an identity check (``is not``) instead.
            if other_value != spec.default:
                merged[name] = other_value

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

    def _uses_exact_channels(self, n_qubits: int) -> bool:
        """Whether Maestro applies noise as exact channels for ``n_qubits`` circuits."""
        return (
            self.simulator_type,
            self._resolve_simulation_type(n_qubits),
        ) in _EXACT_CHANNEL_BACKENDS

    def _noise_realization_kwargs(self, n_qubits: int) -> dict[str, int]:
        """``noise_realizations`` for Maestro's noisy entry points.

        Unset, one realisation is used where every channel of the model is
        exact, since repeats would redo the same simulation; otherwise
        Maestro's default applies.
        """
        if self.noise_realizations is not None:
            return {"noise_realizations": self.noise_realizations}
        noise_model = self.noise_model
        stochastic = noise_model.has_coherent() or noise_model.has_correlated()
        if self._uses_exact_channels(n_qubits) and not stochastic:
            return {"noise_realizations": 1}
        return {}

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

        for name in (
            "singular_value_threshold",
            "truncation_mode",
            "seed",
            "gpu_device",
            "distributed_options",
        ):
            value = getattr(self, name)
            if value is not None:
                kwargs[name] = value

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
            "mpo_kraus_completeness_check": self.mpo_kraus_completeness_check,
        }
        for name, value in property_settings.items():
            if value is not None:
                setattr(config, name, value)

        for name in BOOLEAN_FLAG_FIELDS:
            if getattr(self, name):
                setattr(config, name, True)

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

    Runs circuits on any of maestro's simulator and simulation types, and
    estimates observables natively.

    All maestro-level configuration — including noise — is carried in a
    :class:`MaestroConfig` object rather than as loose keyword arguments,
    matching the
    :class:`~divi.backends.ExecutionConfig` / :class:`~divi.backends.QoroService`
    pattern.

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
        force_sampling: If True, route observable measurements through
            shot-based sampling instead of maestro's native estimation, e.g.
            to see a noise model's readout errors.  Defaults to False.
    """

    def __init__(
        self,
        shots: int = 5000,
        config: MaestroConfig | None = None,
        track_depth: bool = False,
        force_sampling: bool = False,
    ):
        super().__init__(shots=shots, track_depth=track_depth)
        self.config: MaestroConfig = config if config is not None else MaestroConfig()
        self._force_sampling = force_sampling

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
        """Maestro supports native observable estimation unless sampling is forced."""
        return not self._force_sampling

    @property
    def is_async(self) -> bool:
        """Maestro executes circuits synchronously."""
        return False

    def set_seed(self, seed: int) -> None:
        """Seed maestro's simulation RNG.

        Rebinds ``config`` with :attr:`MaestroConfig.seed` set, so the seed
        reaches every subsequent submission; a seed already on the config is
        overwritten.  Assigning a new ``config`` afterwards discards it.

        Args:
            seed: Non-negative seed value.
        """
        self.config = MaestroConfig.model_validate(dict(self.config) | {"seed": seed})

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
                mode only; passing it with ``ham_ops`` raises ``ValueError``.
            cancellation_event: When set, aborts further dispatch and raises
                :class:`~divi.exceptions.ExecutionCancelledError`. Workers
                already in maestro's native call are not interrupted.
            **kwargs: Ignored — accepted so callers using the generic
                :class:`~divi.backends.CircuitRunner` interface can forward
                unrelated options without breaking.

        Returns:
            ExecutionResult containing either counts (sampling) or expectation
            values, with maestro's other outputs for each circuit under
            ``"metadata"``.
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

        config = self.config
        noise_model = config.noise_model
        noise_kwargs: dict[str, int] = {}
        if noise_model is not None:
            qasm_strings = [id_gates_as_noise_sites(q) for q in qasm_strings]
            if ham_ops is not None and noise_model.has_readout_error():
                warnings.warn(
                    "Readout errors do not affect expectation values; use "
                    "force_sampling=True to include them.",
                    stacklevel=2,
                )
            noise_kwargs = config._noise_realization_kwargs(max_qubits)

        base_config = config._to_maestro_config(n_qubits=max_qubits)
        per_circuit_shots = (
            per_circuit_or_none(shot_groups, len(circuit_labels))
            if ham_ops is None
            else None
        )

        def _run(item):
            i, label, qasm = item
            sim_config = base_config
            if config.seed is not None:
                sim_config = copy.copy(base_config)
                sim_config.seed = _circuit_seed(config.seed, label)

            circuit, noisy_kwargs = None, noise_kwargs
            if noise_model is not None:
                # The noisy entry points take a parsed circuit, not QASM.
                circuit = maestro.QasmToCirc().parse_and_translate(qasm)
                noise_seed = _circuit_seed(config.noise_seed, label)
                if noise_seed is not None:
                    noisy_kwargs = noise_kwargs | {"seed": noise_seed}

            if ham_ops is None:
                shots = (
                    self.shots if per_circuit_shots is None else per_circuit_shots[i]
                )
                raw = (
                    maestro.simple_execute(qasm, config=sim_config, shots=shots)
                    if noise_model is None
                    else maestro.full_noise_execute(
                        circuit,
                        noise_model,
                        config=sim_config,
                        shots=shots,
                        **noisy_kwargs,
                    )
                )
                counts = counts_to_little_endian(raw["counts"])
                return _result_entry(label, counts, raw, "counts")

            terms = ham_ops_terms_for_circuit(i, ham_ops, circuit_ham_map)
            observables = ";".join(terms)
            raw = (
                maestro.simple_estimate(
                    qasm, observables=observables, config=sim_config
                )
                if noise_model is None
                else maestro.full_noise_estimate(
                    circuit,
                    observables=observables,
                    noise_model=noise_model,
                    config=sim_config,
                    **noisy_kwargs,
                )
            )
            expvals = expvals_from_result(raw, terms)
            return _result_entry(label, expvals, raw, "expectation_values")

        items = [
            (i, label, qasm)
            for i, (label, qasm) in enumerate(zip(circuit_labels, qasm_strings))
        ]
        results = _run_with_cancellation(
            self._get_executor(), _run, items, cancellation_event
        )
        return ExecutionResult(results=results)
