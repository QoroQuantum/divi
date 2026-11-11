# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import atexit
import copy
import functools
import os
import re
import time
import warnings
import zlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from itertools import pairwise
from numbers import Integral
from threading import Event, Lock
from typing import Any

import maestro
import numpy as np
from pydantic import BaseModel, ConfigDict, SkipValidation, model_validator
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from divi.backends._base import CircuitRunner, ExecutionResult
from divi.backends._cancellation import raise_if_cancelled
from divi.backends._config import describe_unknown, reject_unknown_reset
from divi.backends._pauli_serde import ham_ops_terms_per_circuit, pad_ham_ops
from divi.backends._shot_allocation import expand_shot_groups
from divi.circuits._conversions import (
    _sparse_pauli_op_to_ham_string,
    lower_to_basis_gates,
)
from divi.circuits._payloads import CircuitBatch, CircuitPayload, flatten_bound_payloads
from divi.circuits._qasm_template import bind_template, build_template
from divi.exceptions import IncrementalEvolutionUnavailableError, VJPUnavailableError
from divi.hamiltonians._term_ops import _assert_hermitian_spo
from divi.qasm import count_clbits, count_qubits


def _run_with_cancellation(
    executor: ThreadPoolExecutor,
    fn: Callable[[Any], Any],
    items: Iterable[Any],
    cancellation_event: Event | None,
) -> list:
    """Run ``fn`` over ``items`` via ``executor``, returning results in item order.

    On a worker error or a set ``cancellation_event``, unstarted items are
    cancelled; running ones finish.
    """
    items = list(items)
    futures: dict[Future, int] = {}
    out: list = [None] * len(items)
    try:
        for i, item in enumerate(items):
            futures[executor.submit(fn, item)] = i
        for future in as_completed(futures):
            raise_if_cancelled(
                cancellation_event, "Maestro batch cancelled after partial completion"
            )
            out[futures[future]] = future.result()
    finally:
        for future in futures:
            future.cancel()
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


@functools.lru_cache(maxsize=256)
def _compile_adjoint_template(
    circuit: str, parameter_names: tuple[str, ...]
) -> tuple["maestro.AdjointProgram", np.ndarray, np.ndarray] | str:
    """Compile ``circuit`` for adjoint differentiation, or say why it cannot be.

    Returns the program with the ``(angle_matrix, angle_offset)`` that give its
    gate angles as ``angle_matrix @ values + angle_offset``. The map is read from
    programs compiled at a base point and at steps from it, then confirmed at a
    second point. When every angle follows at most one parameter, the steps move
    all parameters at once and then the parameters whose index has each bit set,
    so each angle's responses spell out its parameter in ``log2`` compiles;
    otherwise each parameter is stepped on its own.
    """
    lines = circuit.splitlines(keepends=True)
    while lines and (not lines[-1].strip() or lines[-1].lstrip().startswith("measure")):
        lines.pop()
    template = build_template("".join(lines), parameter_names)
    n_params = len(parameter_names)

    def compile_at(values: np.ndarray) -> "maestro.AdjointProgram":
        return maestro.AdjointProgram(
            maestro.QasmToCirc().parse_and_translate(bind_template(template, values))
        )

    base, check = np.random.default_rng(0).uniform(-1.0, 1.0, (2, n_params))
    try:
        program = compile_at(base)
        base_angles = program.parameters

        def responses(steps: np.ndarray) -> np.ndarray | None:
            moved = [compile_at(base + step).parameters for step in steps]
            if any(angles.shape != base_angles.shape for angles in moved):
                return None
            return np.array(moved).reshape(len(steps), -1) - base_angles

        codes = np.arange(1, n_params + 1)
        probes = np.vstack(
            [np.ones(n_params)]
            + [(codes >> bit) & 1 for bit in range(n_params.bit_length())]
        )
        probed = responses(probes)
        if probed is None:
            return "The circuit's gates change with its parameter values."
        slope = probed[0]
        bits = np.rint(probed[1:] / np.where(slope == 0, 1.0, slope)).astype(int)
        owner = (bits << np.arange(len(bits))[:, None]).sum(axis=0) - 1
        follows_one = (slope != 0) & (owner >= 0) & (owner < n_params)
        angle_matrix = np.zeros((len(base_angles), n_params))
        angle_matrix[np.flatnonzero(follows_one), owner[follows_one]] = slope[
            follows_one
        ]
        if not np.allclose(probed, probes @ angle_matrix.T, rtol=0.0, atol=1e-12):
            stepped = responses(np.eye(n_params))
            if stepped is None:
                return "The circuit's gates change with its parameter values."
            angle_matrix = stepped.T
        angle_offset = base_angles - angle_matrix @ base

        check_angles = angle_matrix @ check + angle_offset
        compiled_check = compile_at(check).parameters
        if compiled_check.shape != check_angles.shape or not np.allclose(
            compiled_check, check_angles, rtol=0.0, atol=1e-12
        ):
            return "The circuit's gate angles are not affine in its parameters."
    except (ValueError, MemoryError) as exc:
        return f"Maestro cannot differentiate the circuit: {exc}"
    return program, angle_matrix, angle_offset


def _compile_adjoint_programs(
    payloads: Sequence[CircuitPayload],
) -> list[tuple["maestro.AdjointProgram", np.ndarray, np.ndarray]]:
    """Each payload's adjoint program with the map from its parameters to the
    program's gate angles.

    Raises:
        VJPUnavailableError: If a circuit cannot be differentiated.
    """
    compiled = []
    for payload in payloads:
        program = _compile_adjoint_template(payload.circuit, payload.parameter_names)
        if isinstance(program, str):
            raise VJPUnavailableError(program)
        compiled.append(program)
    return compiled


def _vjp_inputs(
    payload: CircuitPayload,
    angle_matrix: np.ndarray,
    angle_offset: np.ndarray,
    cotangents: Mapping[str, np.ndarray],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """A payload's row labels, their gate angles and their cotangents."""
    labels = [label for label, _values in payload.parameter_sets]
    rows = np.array(
        [values for _label, values in payload.parameter_sets], dtype=np.float64
    ).reshape(len(labels), len(payload.parameter_names))
    weights = np.array([cotangents[label] for label in labels], dtype=np.float64)
    return labels, rows @ angle_matrix.T + angle_offset, weights


def _vjp_result_entries(
    labels: Sequence[str],
    terms: Sequence[str],
    raw: Mapping[str, Any],
    angle_matrix: np.ndarray,
    time_taken: float,
) -> list[dict]:
    """Result entries from an adjoint VJP, its gradients taken back to the
    payload's parameters."""
    gradients = np.asarray(raw["vjp"], dtype=np.float64) @ angle_matrix
    return [
        {
            "label": label,
            "results": dict(zip(terms, map(float, values))),
            "metadata": {"vjp": gradient, "time_taken": time_taken},
        }
        for label, values, gradient in zip(labels, raw["expectation_values"], gradients)
    ]


# Compiled programs are native objects; free them before interpreter shutdown.
atexit.register(_compile_adjoint_template.cache_clear)


_ID_GATE_RE = re.compile(r"\bid\s+(q\[\d+\])\s*;")


def id_gates_as_noise_sites(qasm: str) -> str:
    """Rewrite ``id`` gates as ``u3(0,0,0)``, which Maestro's noise injection
    treats as a gate; it adds no noise after ``id`` itself."""
    return _ID_GATE_RE.sub(r"u3(0,0,0) \1;", qasm)


def _prepare_noisy_circuits(
    config: "MaestroConfig", qasm_strings: list[str], *, estimating: bool
) -> list[str]:
    """The circuits as a noisy run takes them, warning when estimating would
    hide the model's readout errors."""
    noise_model = config.noise_model
    if noise_model is None:
        return qasm_strings
    if estimating and noise_model.has_readout_error():
        warnings.warn(
            "Readout errors do not affect expectation values; use "
            "force_sampling=True to include them.",
            stacklevel=3,
        )
    return [id_gates_as_noise_sites(q) for q in qasm_strings]


def _lower_for_statevector(
    config: "MaestroConfig",
    circuit: "str | QuantumCircuit",
    include_idle_qubits: bool,
) -> tuple[str, float, list[int]]:
    """``circuit`` as basis-gate QASM ready for a statevector, the global phase
    OpenQASM 2 drops, and the indices of the qubits it acts on.

    Raises:
        ValueError: If ``config`` has a noise model, which a state vector
            cannot carry.
    """
    if config.noise_model is not None:
        raise ValueError(
            "statevector cannot apply the configured noise model; use "
            "expectation_value, or a maestro_config without noise_model."
        )
    source = (
        QuantumCircuit.from_qasm_str(circuit) if isinstance(circuit, str) else circuit
    )
    # Lowered here, not only on export, so its decompositions' phase is kept.
    lowered = lower_to_basis_gates(source)
    active = {
        qubit
        for instruction in lowered.data
        if instruction.operation.name != "barrier"
        for qubit in instruction.qubits
    }
    if include_idle_qubits and (idle := [q for q in lowered.qubits if q not in active]):
        # Padded first, so any measurement stays terminal.
        padded = lowered.copy_empty_like()
        padded.global_phase = 0.0
        for qubit in idle:
            padded.rz(0.0, qubit)
        padded.compose(lowered, inplace=True)
        lowered = padded
    (qasm,) = flatten_bound_payloads([lowered]).values()
    active_indices = sorted(lowered.find_bit(qubit).index for qubit in active)
    return qasm, float(lowered.global_phase), active_indices


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
    <https://qoroquantum.github.io/maestro/api/configuration.html#all-simulatorconfig-options>`_
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
    leaves Maestro to choose: one where it applies every channel exactly, and
    its sampling default otherwise."""

    @model_validator(mode="before")
    @classmethod
    def _normalise_options(cls, data: Any) -> Any:
        """Reject unknown option names, resolve enum names to members and
        hold numpy values as plain Python."""
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
        data = {
            name: (
                value.tolist() if isinstance(value, np.ndarray | np.generic) else value
            )
            for name, value in data.items()
        }
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

    def _noise_realization_kwargs(self) -> dict[str, int]:
        """``noise_realizations`` for Maestro's noisy entry points, when set."""
        if self.noise_realizations is None:
            return {}
        return {"noise_realizations": self.noise_realizations}


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


def _fit_counts_width(counts: Mapping[str, int], width: int) -> dict[str, int]:
    """Maestro's counts with each key fitted to ``width`` bits, c[0] leftmost.

    Maestro may size keys by the highest classical bit it touched.
    """
    if not width:
        return dict(counts)
    return {bits[:width].ljust(width, "0"): n for bits, n in counts.items()}


def _expectation_value(
    runner: CircuitRunner, circuit: str | QuantumCircuit, observable: SparsePauliOp
) -> float:
    """``<observable>`` in the output state of ``circuit``, estimated by ``runner``.

    Raises:
        ValueError: If ``observable`` is not Hermitian.
    """
    _assert_hermitian_spo(observable)
    terms = _sparse_pauli_op_to_ham_string(observable)
    result = runner.submit_circuits({"circuit": circuit}, ham_ops=terms)
    if result.results is None:
        raise RuntimeError("Backend returned no results")
    expvals = result.results[0]["results"]
    return float(
        sum(
            coeff.real * expvals[term]
            for term, coeff in zip(terms.split(";"), observable.coeffs)
        )
    )


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
            throughout.  Configurations are frozen; change one by assigning a copy,
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

        # Shared by concurrent submissions; each native call also runs its own OpenMP team.
        self._executor: ThreadPoolExecutor | None = None
        self._executor_lock = Lock()

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
    def supports_vjp(self) -> bool:
        """Whether estimation is native and noiseless, as adjoint differentiation needs.

        Maestro itself refuses simulators it cannot differentiate on.
        """
        return self.supports_expval and self.maestro_config.noise_model is None

    def submit_vjp(
        self,
        payloads: Sequence[CircuitPayload],
        *,
        ham_ops: str,
        cotangents: Mapping[str, np.ndarray],
        cancellation_event: Event | None = None,
    ) -> ExecutionResult:
        """Expectation values and vector-Jacobian products by adjoint differentiation.

        One native call per payload. See :meth:`~divi.backends.CircuitRunner.submit_vjp`. Terms shorter than
        a circuit are padded onto its first qubits, as in :meth:`submit_circuits`.

        Raises:
            VJPUnavailableError: If :attr:`supports_vjp` is False, or Maestro cannot
                differentiate a payload's circuit on the configured simulator.
        """
        if not self.supports_vjp:
            raise VJPUnavailableError(
                "MaestroSimulator computes vector-Jacobian products only with "
                "native, noiseless estimation."
            )
        config = self.maestro_config._simulator_config()
        raise_if_cancelled(
            cancellation_event,
            "Maestro batch cancelled before any circuit was dispatched",
        )
        compiled = _compile_adjoint_programs(payloads)
        terms = pad_ham_ops(
            ham_ops, None, [count_qubits(payload.circuit) for payload in payloads]
        ).split(";")
        self._record_payload_depths(payloads)

        def _run(item) -> list[dict]:
            payload, (program, angle_matrix, angle_offset) = item
            labels, angles, weights = _vjp_inputs(
                payload, angle_matrix, angle_offset, cotangents
            )
            start = time.perf_counter()
            try:
                raw = program.vjp(terms, weights, angles, config=config)
            except (ValueError, MemoryError) as exc:
                raise VJPUnavailableError(
                    f"Maestro cannot differentiate the circuit: {exc}"
                ) from exc
            time_taken = (time.perf_counter() - start) / len(labels)
            return _vjp_result_entries(labels, terms, raw, angle_matrix, time_taken)

        per_payload = _run_with_cancellation(
            self._get_executor(),
            _run,
            list(zip(payloads, compiled)),
            cancellation_event,
        )
        results = [entry for entries in per_payload for entry in entries]
        return ExecutionResult(
            results=results,
            run_time=sum(entry["metadata"]["time_taken"] for entry in results),
        )

    @property
    def supports_incremental_evolution(self) -> bool:
        """Whether estimation is native and noiseless, as incremental evolution needs."""
        return self.supports_expval and self.maestro_config.noise_model is None

    def submit_incremental_evolution(
        self,
        initial_circuit: str,
        step_circuit: str,
        *,
        steps: Sequence[int],
        ham_ops: str,
        cancellation_event: Event | None = None,
    ) -> ExecutionResult:
        """Expectation values at each step count from one native evolution.

        See :meth:`~divi.backends.CircuitRunner.submit_incremental_evolution`.
        Terms shorter than the register are padded onto its first qubits, as in
        :meth:`submit_circuits`.

        Raises:
            IncrementalEvolutionUnavailableError: If
                :attr:`supports_incremental_evolution` is False.
            ValueError: If ``steps`` is not a non-empty, strictly increasing
                list of non-negative integers.
        """
        if not self.supports_incremental_evolution:
            raise IncrementalEvolutionUnavailableError(
                "MaestroSimulator evolves incrementally only with native, "
                "noiseless estimation."
            )
        steps = list(steps)
        if not (
            steps
            and all(isinstance(s, Integral) and not isinstance(s, bool) for s in steps)
            and steps[0] >= 0
            and all(a < b for a, b in pairwise(steps))
        ):
            raise ValueError(
                "steps must be non-empty, non-negative and strictly increasing "
                f"integers; got {steps}."
            )
        raise_if_cancelled(
            cancellation_event,
            "Maestro incremental evolution cancelled before dispatch",
        )
        width = max(count_qubits(initial_circuit), count_qubits(step_circuit))
        ham_ops = pad_ham_ops(ham_ops, None, [width])
        self._record_qasm_depths([initial_circuit, step_circuit])
        parser = maestro.QasmToCirc()
        raw = maestro.incremental_evolve(
            parser.parse_and_translate(initial_circuit),
            parser.parse_and_translate(step_circuit),
            [int(step) for step in steps],
            ham_ops,
            config=self.maestro_config._simulator_config(),
        )

        terms = ham_ops.split(";")
        results = [
            {"label": str(step), "results": dict(zip(terms, values, strict=True))}
            for step, values in zip(
                raw["steps"], raw["expectation_values"], strict=True
            )
        ]
        return ExecutionResult(results=results, run_time=raw["time_taken"])

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
                self._executor = ThreadPoolExecutor(
                    max_workers=n_workers,
                    thread_name_prefix="maestro",
                )
            return self._executor

    def close(self) -> None:
        """Shut down the per-instance executor.

        Safe to call multiple times.  The pool's idle threads also exit once
        the instance is garbage-collected; callers that want deterministic
        cleanup (e.g. inside long-running services) can invoke this explicitly.

        ``shutdown(wait=True)`` runs **outside** ``_executor_lock`` — a
        concurrent ``submit_circuits`` on another thread can grab the lock
        and lazily re-create a fresh pool while the old one drains, instead
        of serialising behind a slow shutdown.  Subsequent submits therefore
        observe ``close()`` as "release current pool; new pool created on
        demand".
        """
        with self._executor_lock:
            executor = self._executor
            self._executor = None
        if executor is not None:
            executor.shutdown(wait=True)

    def submit_circuits(
        self,
        payloads: Sequence[CircuitPayload] | CircuitBatch,
        *,
        ham_ops: str | None = None,
        circuit_ham_map: Sequence[list[int]] | None = None,
        shot_groups: Sequence[list[int]] | None = None,
        cancellation_event: Event | None = None,
        **kwargs,
    ) -> ExecutionResult:
        """Submit quantum circuits for execution on the maestro simulator,
        longest first.

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

        circuits = flatten_bound_payloads(payloads)
        circuit_labels = list(circuits.keys())
        qasm_strings = list(circuits.values())
        if ham_ops is not None:
            ham_ops = pad_ham_ops(
                ham_ops, circuit_ham_map, [count_qubits(q) for q in qasm_strings]
            )

        self._record_qasm_depths(qasm_strings)

        config = self.maestro_config
        noise_model = config.noise_model
        qasm_strings = _prepare_noisy_circuits(
            config, qasm_strings, estimating=ham_ops is not None
        )
        noise_kwargs = config._noise_realization_kwargs()

        base_config = config._simulator_config()
        per_circuit_shots = expand_shot_groups(shot_groups, len(circuit_labels))
        if ham_ops is not None:
            per_circuit_terms = ham_ops_terms_per_circuit(
                len(circuit_labels), ham_ops, circuit_ham_map
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
                counts = _fit_counts_width(raw["counts"], count_clbits(qasm))
                return _result_entry(label, counts, raw, "counts")

            terms = per_circuit_terms[i]
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

        # Statement count stands in for runtime: the longest circuits start first.
        cost_proxy = [qasm.count(";") for qasm in qasm_strings]
        items = sorted(
            (
                (i, label, qasm)
                for i, (label, qasm) in enumerate(zip(circuit_labels, qasm_strings))
            ),
            key=lambda item: cost_proxy[item[0]],
            reverse=True,
        )
        out = _run_with_cancellation(
            self._get_executor(), _run, items, cancellation_event
        )
        entry_by_index = {
            i: entry for (i, _label, _qasm), entry in zip(items, out, strict=True)
        }
        results = [entry_by_index[i] for i in range(len(items))]
        return ExecutionResult(
            results=results,
            run_time=sum(entry["metadata"]["time_taken"] for entry in results),
        )

    def statevector(
        self, circuit: str | QuantumCircuit, include_idle_qubits: bool = True
    ) -> np.ndarray:
        """Output amplitudes of ``circuit``, ignoring its final measurements, with
        qubit 0 the least-significant bit of the basis index.

        Args:
            circuit: OpenQASM 2.0 string or Qiskit circuit.
            include_idle_qubits: Span every qubit of the register. ``False`` keeps
                only the qubits a gate acts on, in ascending index order, so the
                vector is ``2**n_active`` long.

        Raises:
            ValueError: If the configuration has a noise model, which a state
                vector cannot carry.
        """
        qasm, global_phase, _active = _lower_for_statevector(
            self.maestro_config, circuit, include_idle_qubits
        )
        amplitudes = maestro.get_statevector(
            maestro.QasmToCirc().parse_and_translate(qasm),
            config=self.maestro_config._simulator_config(),
        )
        return np.exp(1j * global_phase) * np.asarray(amplitudes)

    def expectation_value(
        self, circuit: str | QuantumCircuit, observable: SparsePauliOp
    ) -> float:
        """``<observable>`` in the output state of ``circuit``.

        Raises:
            ValueError: If ``observable`` has a non-real coefficient, so is not
                Hermitian and has no real expectation value.
        """
        return _expectation_value(self, circuit, observable)
