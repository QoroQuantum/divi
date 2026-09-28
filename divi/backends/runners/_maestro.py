# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import copy
import os
import re
import warnings
import weakref
import zlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from typing import Any

import maestro
from pydantic import BaseModel, ConfigDict, SkipValidation, model_validator

from divi.circuits._payloads import CircuitBatch, CircuitPayload, bound_circuits
from divi.exceptions import ExecutionCancelledError
from divi.qasm import count_qubits

from .._base import CircuitRunner, ExecutionResult
from .._cancellation import raise_if_cancelled
from .._config import describe_unknown, reject_unknown_reset
from .._pauli_serde import ham_ops_terms_for_circuit, pad_ham_ops
from .._shot_allocation import per_circuit_or_none


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


def _circuit_seed(seed: int | None, label: str, qasm: str) -> int | None:
    """Per-circuit seed keyed on the label and QASM."""
    if seed is None:
        return None
    return zlib.crc32(qasm.encode(), zlib.crc32(label.encode(), seed))


def _result_entry(
    label: str, results: Any, raw: Mapping[str, Any], result_key: str
) -> dict[str, Any]:
    """One circuit's result entry, with maestro's other outputs as metadata."""
    metadata = {key: value for key, value in raw.items() if key != result_key}
    return {"label": label, "results": results, "metadata": metadata}


_ID_GATE_RE = re.compile(r"\bid\s+(q\[\d+\])\s*;")


def id_gates_as_noise_sites(qasm: str) -> str:
    """Rewrite ``id`` gates as ``u3(0,0,0)``, which Maestro's noise injection
    treats as a gate; it adds no noise after ``id`` itself."""
    return _ID_GATE_RE.sub(r"u3(0,0,0) \1;", qasm)


# Simulator/simulation pairs on which Maestro applies noise as exact channels.
_EXACT_CHANNEL_BACKENDS = frozenset(
    {
        ("QCSim", "DensityMatrix"),
        ("QCSim", "MatrixProductOperator"),
        ("Gpu", "DensityMatrix"),
        ("Gpu", "MatrixProductOperator"),
        ("QiskitAer", "DensityMatrix"),
    }
)

_SIMULATOR_OPTIONS = frozenset(maestro.SimulatorConfig._fields)
# Per-run settings a maestro user may look for on the config.
_RUN_OPTIONS = frozenset({"shots", "force_sampling"})
_ENUMS = {
    "simulator_type": maestro.SimulatorType,
    "simulation_type": maestro.SimulationType,
}


class MaestroConfig(BaseModel):
    """Configuration object for :class:`MaestroSimulator` and cloud Maestro runs.

    Simulator options are ``maestro.SimulatorConfig``'s own, given as keyword
    arguments with its names, defaults and validation; see the `maestro Python
    bindings guide
    <https://qoroquantum.github.io/maestro/d7/d01/python_guide.html#py_config>`_
    for what each one does::

        MaestroConfig(simulation_type="MatrixProductState", max_bond_dimension=32)

    The simulation ``seed`` is one of these options too, e.g.
    ``MaestroConfig(seed=42)``. ``simulator_type`` and ``simulation_type``
    accept the maestro enum members or their names. Options left unset use
    maestro's defaults, on :class:`MaestroSimulator` and
    :class:`~divi.backends.QoroService` alike, and read back as those defaults.
    :meth:`from_simulator_config` builds a config from an existing
    ``maestro.SimulatorConfig``.

    Raises:
        ValueError: If an option name is unknown, or maestro rejects an
            option's value.
    """

    model_config = ConfigDict(frozen=True, extra="allow", arbitrary_types_allowed=True)

    noise_model: SkipValidation["maestro.NoiseModel | None"] = None
    """Maestro ``NoiseModel`` to simulate with.  When set, circuits run through
    ``maestro.full_noise_execute`` (sampling) or ``maestro.full_noise_estimate``
    (expectation values); ``None`` runs them noiseless."""

    noise_seed: int | None = None
    """Seed for the injected noise, derived per circuit from its label and QASM.
    ``None`` leaves maestro to seed the noise from the simulator ``seed``."""

    noise_realizations: int | None = None
    """``noise_realizations`` passed to Maestro's noisy entry points.  ``None``
    uses one on backends where every channel of the model is exact, and
    Maestro's default otherwise."""

    @model_validator(mode="before")
    @classmethod
    def _normalise_options(cls, data: Any) -> Any:
        """Reject unknown option names and resolve enum names to members."""
        if not isinstance(data, Mapping):
            return data
        if unknown := describe_unknown(
            data, _SIMULATOR_OPTIONS | cls.model_fields.keys()
        ):
            run_options = sorted(data.keys() & _RUN_OPTIONS)
            hint = (
                f" {run_options} are set on MaestroSimulator or JobConfig, not here."
                if run_options
                else ""
            )
            raise ValueError(
                f"MaestroConfig got unknown options {unknown}.{hint} Simulator "
                f"options are maestro.SimulatorConfig's: {sorted(_SIMULATOR_OPTIONS)}."
            )
        data = dict(data)
        for name, enum in _ENUMS.items():
            value = data.get(name)
            if isinstance(value, str):
                if value not in enum.__members__:
                    raise ValueError(
                        f"{name} must be one of {sorted(enum.__members__)}. "
                        f"Got {value!r}."
                    )
                data[name] = enum.__members__[value]
        return data

    @model_validator(mode="after")
    def _validate_with_maestro(self):
        """Have maestro validate the options when the config is built."""
        self._simulator_config()
        return self

    @classmethod
    def from_simulator_config(
        cls, simulator_config: "maestro.SimulatorConfig", **fields: Any
    ) -> "MaestroConfig":
        """Build a config holding the options ``simulator_config`` changes
        from maestro's defaults.

        Args:
            simulator_config: The ``maestro.SimulatorConfig`` to copy options
                from.
            **fields: Further ``MaestroConfig`` arguments, such as
                ``noise_model``; options given here take precedence.
        """
        defaults = maestro.SimulatorConfig()
        options = {
            name: value
            for name in _SIMULATOR_OPTIONS
            if (value := getattr(simulator_config, name)) != getattr(defaults, name)
        }
        return cls(**options | fields)

    def __getattr__(self, name: str) -> Any:
        """Read an option this config leaves unset as maestro's default."""
        try:
            # pyrefly: ignore[missing-attribute]
            return super().__getattr__(name)
        except AttributeError:
            if name in _SIMULATOR_OPTIONS:
                return getattr(self._simulator_config(), name)
            raise

    def _simulator_options(self) -> dict[str, Any]:
        """Every simulator option this config sets, by maestro field name."""
        return dict(self.model_extra or {})

    def _simulator_config(self) -> "maestro.SimulatorConfig":
        """A fresh ``maestro.SimulatorConfig`` holding this config's options."""
        options = self._simulator_options()
        try:
            return maestro.SimulatorConfig(**options)
        except TypeError as exc:
            raise ValueError(
                f"maestro.SimulatorConfig rejected the options {options}: {exc}"
            ) from exc

    def _set_fields(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.model_fields_set}

    def override(self, **fields: Any) -> "MaestroConfig":
        """Return a copy with ``fields`` set, e.g.
        ``config.override(max_bond_dimension=32)``."""
        return MaestroConfig(**self._set_fields() | fields)

    def reset(self, *names: str) -> "MaestroConfig":
        """Return a copy with ``names`` back at their defaults, e.g.
        ``config.reset("max_bond_dimension")``; ``reset("noise_model")``
        turns noise off."""
        reject_unknown_reset(names, _SIMULATOR_OPTIONS | type(self).model_fields.keys())
        return MaestroConfig(
            **{k: v for k, v in self._set_fields().items() if k not in names}
        )

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> "MaestroConfig":
        """Copy, validating ``update`` as :meth:`override` does."""
        copied = super().model_copy(deep=deep)
        return copied.override(**update) if update else copied

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | _SIMULATOR_OPTIONS)

    def _uses_exact_channels(self) -> bool:
        """Whether Maestro applies noise as exact channels on this config's backend."""
        config = self._simulator_config()
        backend = (config.simulator_type.name, config.simulation_type.name)
        return backend in _EXACT_CHANNEL_BACKENDS

    def _noise_realization_kwargs(self) -> dict[str, int]:
        """``noise_realizations`` for Maestro's noisy entry points.

        Unset, one realisation is used where every channel of the model is
        exact, since repeats would redo the same simulation; otherwise
        Maestro's default applies.
        """
        if self.noise_realizations is not None:
            return {"noise_realizations": self.noise_realizations}
        noise_model = self.noise_model
        stochastic = noise_model.has_coherent() or noise_model.has_correlated()
        if self._uses_exact_channels() and not stochastic:
            return {"noise_realizations": 1}
        return {}


def require_maestro_config(value: Any, name: str) -> None:
    """Raise unless ``value`` is a :class:`MaestroConfig`."""
    if isinstance(value, MaestroConfig):
        return
    hint = (
        "; convert it with MaestroConfig.from_simulator_config(...)"
        if isinstance(value, maestro.SimulatorConfig)
        else ""
    )
    raise TypeError(
        f"{name} must be a MaestroConfig, got {type(value).__name__}{hint}."
    )


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
    :class:`MaestroConfig` object rather than as loose keyword arguments; the
    same object configures cloud runs on :class:`~divi.backends.QoroService`.

    .. note::

        Maestro's C++ extension must be loaded before other C++ libraries
        (Qiskit, PennyLane) to avoid initialisation order conflicts.  This
        is handled automatically by ``divi/__init__.py``.

    Args:
        shots: Number of measurement shots. Defaults to 5000.
        maestro_config: :class:`MaestroConfig` controlling simulator backend,
            simulation method, bond dimension, noise model, and related
            options.  Defaults to ``MaestroConfig()``, maestro's defaults
            throughout.  Configs are frozen; change one by assigning a copy,
            e.g. ``sim.maestro_config = sim.maestro_config.override(seed=7)``.
        track_depth: Record circuit depth per submission. Defaults to False.
        force_sampling: If True, route observable measurements through
            shot-based sampling instead of maestro's native estimation, e.g.
            to see a noise model's readout errors.  Defaults to False.
    """

    def __init__(
        self,
        shots: int = 5000,
        maestro_config: MaestroConfig | None = None,
        track_depth: bool = False,
        force_sampling: bool = False,
    ):
        super().__init__(shots=shots, track_depth=track_depth)
        self.maestro_config = (
            maestro_config if maestro_config is not None else MaestroConfig()
        )
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
    def maestro_config(self) -> MaestroConfig:
        """The Maestro settings every submission runs with."""
        return self._maestro_config

    @maestro_config.setter
    def maestro_config(self, value: MaestroConfig) -> None:
        require_maestro_config(value, "maestro_config")
        self._maestro_config = value

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

        Rebinds ``maestro_config`` with its ``seed`` option set, so the seed
        reaches every subsequent submission; a seed already on the config is
        overwritten.  Assigning a new ``maestro_config`` afterwards discards it.

        Args:
            seed: Non-negative seed value.
        """
        self.maestro_config = self.maestro_config.override(seed=seed)

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
                e.g. ``"ZI;IZ;XX"``. If None, runs in sampling mode. Terms shorter
                than a circuit are padded onto its first qubits, with a warning.
            circuit_ham_map: Maps circuit index ranges to observable groups for
                heterogeneous batches. Each inner list contains circuit indices
                belonging to that observable group.
            shot_groups: Per-circuit shot allocation as ``[start, end, shots]``
                triples covering the iteration order of ``circuits``. Sampling
                mode only; passing it with ``ham_ops`` raises ``ValueError``.
            cancellation_event: When set, aborts further dispatch and raises
                :class:`~divi.exceptions.ExecutionCancelledError`. Workers
                already in maestro's native call are not interrupted.
            **kwargs: Rejected with ``TypeError``.

        Returns:
            ExecutionResult containing either counts (sampling) or expectation
            values, with maestro's other outputs for each circuit under
            ``"metadata"``.
        """
        self._reject_unknown_options(kwargs)
        raise_if_cancelled(
            cancellation_event,
            "Maestro batch cancelled before any circuit was dispatched",
        )
        self._reject_shot_groups_with_ham_ops(ham_ops, shot_groups)

        circuits = bound_circuits(payloads)
        circuit_labels = list(circuits.keys())
        qasm_strings = list(circuits.values())
        if ham_ops is not None:
            ham_ops = pad_ham_ops(
                ham_ops, circuit_ham_map, [count_qubits(q) for q in qasm_strings]
            )

        self._record_qasm_depths(qasm_strings)

        config = self.maestro_config
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
            noise_kwargs = config._noise_realization_kwargs()

        base_config = config._simulator_config()
        per_circuit_shots = (
            per_circuit_or_none(shot_groups, len(circuit_labels))
            if ham_ops is None
            else None
        )

        def _run(item):
            i, label, qasm = item
            sim_config = base_config
            if base_config.seed is not None:
                sim_config = copy.copy(base_config)
                sim_config.seed = _circuit_seed(base_config.seed, label, qasm)

            circuit, noisy_kwargs = None, noise_kwargs
            if noise_model is not None:
                # The noisy entry points take a parsed circuit, not QASM.
                circuit = maestro.QasmToCirc().parse_and_translate(qasm)
                noise_seed = _circuit_seed(config.noise_seed, label, qasm)
                if noise_seed is not None:
                    noisy_kwargs = noise_kwargs | {"noise_seed": noise_seed}

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
                # Maestro puts q[0] leftmost; Qiskit (little-endian) puts it rightmost.
                counts = {bits[::-1]: n for bits, n in raw["counts"].items()}
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
            expvals = dict(zip(terms, raw["expectation_values"]))
            return _result_entry(label, expvals, raw, "expectation_values")

        items = [
            (i, label, qasm)
            for i, (label, qasm) in enumerate(zip(circuit_labels, qasm_strings))
        ]
        results = _run_with_cancellation(
            self._get_executor(), _run, items, cancellation_event
        )
        return ExecutionResult(results=results)
