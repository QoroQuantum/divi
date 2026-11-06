# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Run circuits on Maestro and IBM resources managed by QRMI."""

import importlib
import json
import logging
import os
import re
import weakref
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import repeat
from threading import Event, Lock
from typing import Any, Self

import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import BitArray
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager

from divi._optional import import_if_available, import_optional
from divi.backends._base import CircuitRunner, ExecutionResult
from divi.backends._cancellation import raise_if_cancelled
from divi.backends._config import DeviceConfig, describe_unknown, require_device_config
from divi.backends._pauli_serde import ham_ops_terms_per_circuit, pad_ham_ops
from divi.backends._shot_allocation import (
    bucket_by_shots,
    expand_shot_groups,
    group_shots_into_ranges,
)
from divi.circuits._payloads import CircuitBatch, CircuitPayload, flatten_bound_payloads
from divi.exceptions import VJPUnavailableError
from divi.qasm import count_clbits, count_qubits

from ._maestro import (
    MaestroConfig,
    _bind_parameter_sets,
    _circuit_seed,
    _compile_adjoint_programs,
    _expectation_value,
    _little_endian_counts,
    _lower_for_statevector,
    _prepare_noisy_circuits,
    _result_entry,
    _vjp_inputs,
    _vjp_result_entries,
    require_maestro_config,
)

logger = logging.getLogger(__name__)

# Native request names for ``maestro.SimulatorType`` members.
_NATIVE_BACKENDS = {
    "QCSim": "qcsim",
    "CompositeQCSim": "composite_qcsim",
    "QiskitAer": "aer",
    "CompositeQiskitAer": "composite_aer",
    "Gpu": "gpu",
    "DistributedGpu": "distributed_gpu",
    "DistributedMpiGpu": "distributed_mpi_gpu",
    "QuestSim": "quest",
}

# ``noise_realizations`` that Maestro's ``full_noise_execute`` and
# ``full_noise_estimate`` default to, which a local run leaves in force.
_LOCAL_EXECUTE_REALIZATIONS = 64
_LOCAL_ESTIMATE_REALIZATIONS = 100

# What the server reports for a session it no longer has, or QRMI for a
# resource holding none.
_LOST_SESSION_MARKERS = ("session_not_found", "No session")

# Task failures after which a local run would also have given up on adjoint
# differentiation.
_VJP_UNAVAILABLE_CODES = frozenset({"insufficient_resources"})

# Seconds between status polls: a local socket, or IBM's REST API.
_MAESTRO_POLLING_INTERVAL = 0.05
_IBM_POLLING_INTERVAL = 1.0

# ``generate_preset_pass_manager`` arguments, by IBM device key.
_IBM_TRANSPILE_KEYS = {
    "TRANSPILE_OPTIMIZATION_LEVEL": "optimization_level",
    "TRANSPILE_LAYOUT_METHOD": "layout_method",
    "TRANSPILE_ROUTING_METHOD": "routing_method",
    "TRANSPILE_SEED": "seed_transpiler",
    "TRANSPILE_APPROXIMATION_DEGREE": "approximation_degree",
    "TRANSPILE_INITIAL_LAYOUT": "initial_layout",
}

# ``SamplerV2`` option groups and names, by IBM device key.
_IBM_SAMPLER_KEYS = {
    "DYNAMICAL_DECOUPLING_ENABLE": ("dynamical_decoupling", "enable"),
    "DYNAMICAL_DECOUPLING_SEQUENCE_TYPE": ("dynamical_decoupling", "sequence_type"),
    "DYNAMICAL_DECOUPLING_SCHEDULING_METHOD": (
        "dynamical_decoupling",
        "scheduling_method",
    ),
    "DYNAMICAL_DECOUPLING_EXTRA_SLACK_DISTRIBUTION": (
        "dynamical_decoupling",
        "extra_slack_distribution",
    ),
    "TWIRLING_ENABLE_GATES": ("twirling", "enable_gates"),
    "TWIRLING_ENABLE_MEASURE": ("twirling", "enable_measure"),
    "TWIRLING_NUM_RANDOMIZATIONS": ("twirling", "num_randomizations"),
    "TWIRLING_SHOTS_PER_RANDOMIZATION": ("twirling", "shots_per_randomization"),
    "TWIRLING_STRATEGY": ("twirling", "strategy"),
    "EXECUTION_REP_DELAY": ("execution", "rep_delay"),
    "EXECUTION_INIT_QUBITS": ("execution", "init_qubits"),
}

_IBM_DEVICE_KEYS = frozenset(
    {*_IBM_TRANSPILE_KEYS, *_IBM_SAMPLER_KEYS, "USE_FRACTIONAL_GATES"}
)

# The level Qoro's IBM service compiles at without aggressive compiling.
_IBM_OPTIMIZATION_LEVEL = 2


class QRMITaskFailedError(RuntimeError):
    """A task the resource ran to a failure, with the server's error code."""

    def __init__(self, message: str, code: str | None):
        super().__init__(message)
        self.code = code


@dataclass
class _Lease:
    """The session a backend acquired, shared with its finaliser."""

    session: str | None = None


def _release_session(resource: Any, lock: Lock, lease: _Lease) -> None:
    """Release ``lease``'s session, if any; a refusal is logged, not raised."""
    with lock:
        session, lease.session = lease.session, None
        if session is None:
            return
        try:
            resource.release(session)
        except RuntimeError as exc:
            logger.warning("Could not release QRMI session %s: %s", session, exc)


def _import_qrmi() -> Any:
    """The ``qrmi`` module, with ``qrmi.maestro`` where the build has it."""
    qrmi = import_if_available("qrmi")
    if qrmi is None:
        raise ImportError(
            "QRMIBackend needs qrmi; maestro-local resources need Qoro's fork "
            "(https://github.com/QoroQuantum/qrmi)."
        )
    if hasattr(qrmi.ResourceType, "MaestroLocal"):
        importlib.import_module("qrmi.maestro")
    return qrmi


def _import_ibm_runtime() -> tuple[Callable[..., Any], Any, Any]:
    """qrmi's IBM backend factory, and Qiskit Runtime's job encoder and result
    decoder."""
    runtime = import_optional(
        "qiskit_ibm_runtime",
        extra="aer",
        capability="QRMIBackend on IBM resources",
        hint='Installing "qrmi[ibm]" provides it too.',
    )
    from qiskit_ibm_runtime.decoders import result_decoder

    return (
        importlib.import_module("qrmi.primitives.ibm").get_backend,
        runtime.RuntimeEncoder,
        result_decoder.ResultDecoder,
    )


def _ibm_settings(
    config: DeviceConfig | None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], bool]:
    """Pass manager arguments, ``SamplerV2`` options and fractional-gate flag
    from ``config``.

    Raises:
        ValueError: If ``config`` has a key QRMI cannot apply.
    """
    payload = config.to_payload() if config is not None else {}
    if unknown := describe_unknown(payload, _IBM_DEVICE_KEYS):
        raise ValueError(
            f"QRMIBackend cannot apply device_config keys {unknown} on IBM "
            "resources; it takes the TRANSPILE_*, DYNAMICAL_DECOUPLING_*, "
            "TWIRLING_*, EXECUTION_REP_DELAY, EXECUTION_INIT_QUBITS and "
            "USE_FRACTIONAL_GATES keys."
        )
    transpile: dict[str, Any] = {"optimization_level": _IBM_OPTIMIZATION_LEVEL}
    sampler: dict[str, dict[str, Any]] = {}
    for key, value in payload.items():
        if key in _IBM_TRANSPILE_KEYS:
            transpile[_IBM_TRANSPILE_KEYS[key]] = value
        elif key in _IBM_SAMPLER_KEYS:
            group, name = _IBM_SAMPLER_KEYS[key]
            sampler.setdefault(group, {})[name] = value
    return transpile, sampler, bool(payload.get("USE_FRACTIONAL_GATES"))


def _native_method(simulation_type: Any) -> str:
    """``MatrixProductState`` -> ``matrix_product_state``."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", simulation_type.name).lower()


def _simulator_block(config: MaestroConfig) -> dict[str, Any]:
    """The request's ``simulator`` object for ``config``.

    Raises:
        ValueError: If the server has no native name for the simulator type.
    """
    backend = _NATIVE_BACKENDS.get(config.simulator_type.name)
    if backend is None:
        raise ValueError(
            f"A Maestro server cannot run simulator_type={config.simulator_type.name}."
        )
    options = config._simulator_options()
    for name in ("simulator_type", "simulation_type", "seed"):
        options.pop(name, None)
    return {
        "backend": backend,
        "method": _native_method(config.simulation_type),
        "options": options,
    }


def _noise_block(
    config: MaestroConfig, label: str, qasm: str, estimating: bool
) -> dict[str, Any] | None:
    """The request's ``noise`` object, or None when no channel acts on the circuit."""
    if config.noise_model is None:
        return None
    channels = config.noise_model.native_channels(
        count_qubits(qasm), include_readout=not estimating
    )
    if not channels:
        return None
    noise = {
        "channels": channels,
        "realizations": config._noise_realization_kwargs().get(
            "noise_realizations",
            _LOCAL_ESTIMATE_REALIZATIONS if estimating else _LOCAL_EXECUTE_REALIZATIONS,
        ),
    }
    if (seed := _circuit_seed(config.noise_seed, label, qasm)) is not None:
        noise["seed"] = seed
    return noise


def _request(
    operation: str,
    qasm: str,
    simulator: dict[str, Any],
    *,
    num_clbits: int = 0,
    **fields: Any,
) -> dict[str, Any]:
    """A native request for ``operation`` on ``qasm``, with the ``fields`` that
    are not None."""
    circuit: dict[str, Any] = {"source": qasm, "num_qubits": count_qubits(qasm)}
    if num_clbits:
        circuit["num_clbits"] = num_clbits
    return {
        "schema_version": 2,
        "operation": operation,
        "circuit": circuit,
        "simulator": simulator,
        **{name: value for name, value in fields.items() if value is not None},
    }


def _circuit_request(
    qasm: str,
    simulator: dict[str, Any],
    *,
    seed: int | None,
    shots: int,
    terms: list[str] | None,
    noise: dict[str, Any] | None,
) -> dict[str, Any]:
    """An ``execute`` request for ``shots``, or an ``estimate`` of ``terms``."""
    execution = {} if seed is None else {"seed": seed}
    if terms is not None:
        return _request(
            "estimate",
            qasm,
            simulator,
            execution=execution,
            observables=terms,
            noise=noise,
        )
    return _request(
        "execute",
        qasm,
        simulator,
        num_clbits=count_clbits(qasm),
        execution=execution | {"shots": shots},
        noise=noise,
    )


def _timed_result(results: list[dict]) -> ExecutionResult:
    """``results`` with the server's per-circuit times summed."""
    return ExecutionResult(
        results=results,
        run_time=sum(entry["metadata"]["time_taken"] for entry in results),
    )


def _active_amplitudes(amplitudes: np.ndarray, active: Sequence[int]) -> np.ndarray:
    """The entries of a full-register state in which only ``active`` qubits vary.

    Idle qubits stay in ``|0>``, so these entries carry the whole state.
    """
    ks = np.arange(1 << len(active))
    indices = np.zeros_like(ks)
    for bit, qubit in enumerate(active):
        indices |= ((ks >> bit) & 1) << qubit
    return amplitudes[indices]


class QRMIBackend(CircuitRunner):
    """Run circuits on a quantum resource that QRMI manages.

    On a ``maestro-local`` resource, each circuit is a native Maestro request;
    results match :class:`~divi.backends.MaestroSimulator`'s for the same
    Maestro version, configuration and seeds.

    On an IBM resource, each batch is compiled for the device and sampled in
    one ``SamplerV2`` job per shot count.

    Args:
        resource: A ``maestro-local`` or IBM ``qrmi.QuantumResource``. A
            session its scheduler exported as
            ``<resource_id>_QRMI_JOB_ACQUISITION_TOKEN`` is used and never
            released.
        shots: Shots per circuit. Defaults to 5000.
        maestro_config: :class:`~divi.backends.MaestroConfig`, as for
            :class:`~divi.backends.MaestroSimulator`. Maestro only; defaults
            to ``MaestroConfig()``.
        device_config: :class:`~divi.backends.DeviceConfig` of IBM device keys.
            IBM only; see :attr:`device_config`.
        track_depth: Record per-circuit logical depth.
        force_sampling: Sample observables instead of estimating them.
        polling_interval: Seconds between status polls. Defaults to 0.05 on
            Maestro and 1 on IBM.

    Raises:
        ValueError: If the resource type is unsupported, a config is given for
            the other resource kind, or ``device_config`` has a key QRMI cannot
            apply.
    """

    def __init__(
        self,
        resource: Any,
        *,
        shots: int = 5000,
        maestro_config: MaestroConfig | None = None,
        device_config: DeviceConfig | None = None,
        track_depth: bool = False,
        force_sampling: bool = False,
        polling_interval: float | None = None,
    ):
        super().__init__(shots=shots, track_depth=track_depth)
        self._qrmi = _import_qrmi()
        types = self._qrmi.ResourceType
        self._resource = resource
        self._resource_type = resource.resource_type()
        is_maestro = self._resource_type == getattr(types, "MaestroLocal", None)
        if not is_maestro and self._resource_type not in (
            types.IBMQuantumSystem,
            types.IBMQiskitRuntimeService,
            types.IBMQuantumComputeService,
        ):
            raise ValueError(
                "QRMIBackend runs maestro-local and IBM resources; got "
                f"{self._resource_type}. maestro-local needs Qoro's fork of qrmi "
                "(https://github.com/QoroQuantum/qrmi)."
            )
        # A QRMI resource's methods take `&mut self`: one call at a time.
        self._resource_lock = Lock()
        # A pass manager keeps each run's state on itself: one run at a time.
        self._compile_lock = Lock()

        self._maestro_config = MaestroConfig() if is_maestro else None
        if not is_maestro:
            self._ibm_backend_factory, self._job_encoder, self._result_decoder = (
                _import_ibm_runtime()
            )
        self.device_config = device_config
        if maestro_config is not None:
            self.maestro_config = maestro_config

        self._force_sampling = force_sampling
        if polling_interval is None:
            polling_interval = (
                _MAESTRO_POLLING_INTERVAL if is_maestro else _IBM_POLLING_INTERVAL
            )
        self._polling_interval = polling_interval
        self._lease = _Lease()
        self._borrowed_session = (
            f"{resource.resource_id()}_QRMI_JOB_ACQUISITION_TOKEN" in os.environ
        )
        weakref.finalize(
            self, _release_session, resource, self._resource_lock, self._lease
        )

    @property
    def maestro_config(self) -> MaestroConfig | None:
        """Maestro settings for every submission; ``None`` on IBM."""
        return self._maestro_config

    @maestro_config.setter
    def maestro_config(self, value: MaestroConfig) -> None:
        if not self._is_maestro:
            raise ValueError(
                "maestro_config applies to maestro-local resources; this "
                f"resource is {self._resource_type}. Pass device_config instead."
            )
        require_maestro_config(value, "maestro_config")
        self._maestro_config = value

    @property
    def device_config(self) -> DeviceConfig | None:
        """IBM device keys for every submission; ``None`` on Maestro.

        Takes ``TRANSPILE_*`` (optimisation level 2 by default),
        ``DYNAMICAL_DECOUPLING_*``, ``TWIRLING_*``, ``EXECUTION_REP_DELAY``,
        ``EXECUTION_INIT_QUBITS`` and ``USE_FRACTIONAL_GATES``; other keys
        raise ``ValueError``.
        """
        return self._device_config

    @device_config.setter
    def device_config(self, value: DeviceConfig | None) -> None:
        if value is not None:
            if self._is_maestro:
                raise ValueError(
                    "device_config applies to IBM resources; this resource is "
                    f"{self._resource_type}. Pass maestro_config instead."
                )
            require_device_config(value)
        settings = _ibm_settings(value)
        with self._compile_lock:
            self._device_config = value
            self._ibm_job_settings = settings
            self._pass_manager: Any = None

    @property
    def supports_expval(self) -> bool:
        """Maestro estimates observables natively unless sampling is forced."""
        return self._is_maestro and not self._force_sampling

    @property
    def supports_vjp(self) -> bool:
        """Whether estimation is native and noiseless, as adjoint differentiation needs.

        The server itself refuses simulators it cannot differentiate on.
        """
        config = self._maestro_config
        return (
            self.supports_expval and config is not None and config.noise_model is None
        )

    @property
    def _is_maestro(self) -> bool:
        return self._maestro_config is not None

    def set_seed(self, seed: int) -> None:
        """Seed Maestro's simulation RNG; no effect on IBM.

        Rebinds ``maestro_config`` with ``seed`` set; assigning a new
        ``maestro_config`` afterwards discards it.

        Args:
            seed: Non-negative seed value.
        """
        if self._maestro_config is not None:
            self.maestro_config = self._maestro_config.override(seed=seed)

    def close(self) -> None:
        """Release the session, if this backend acquired one.

        Call it once the backend is no longer needed, or use the backend as a
        context manager, so the resource is free for other jobs.
        """
        _release_session(self._resource, self._resource_lock, self._lease)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

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
        """Run each circuit as a Maestro task, or the batch as IBM jobs.

        Args:
            payloads: Bound circuits, as payloads or any collection of
                already-resolved circuits.
            ham_ops: Semicolon-separated Pauli terms, e.g. ``"ZI;IZ"``. Switches
                to Maestro's analytic estimator, returning expectation values
                instead of counts. Terms shorter than a circuit are padded onto
                its first qubits, with a warning. Maestro only.
            circuit_ham_map: Assigns each ``|``-delimited ``ham_ops`` group to a
                slice of the circuit list.
            shot_groups: Per-circuit ``[start, end, shots]`` triples. Sampling
                only.
            cancellation_event: Checked between circuits, and while one runs.
            **kwargs: Rejected with ``TypeError``.

        Returns:
            ExecutionResult of counts per label, or expectation values keyed by
            Pauli term, with the resource's other outputs for each circuit under
            ``"metadata"``.

        Raises:
            ValueError: If ``ham_ops`` is given for an IBM resource, or an IBM
                circuit has no classical bits.
            QRMITaskFailedError: If the resource runs a task to a failure.
        """
        self._reject_unknown_options(kwargs)
        raise_if_cancelled(
            cancellation_event,
            "QRMI batch cancelled before any circuit was dispatched",
        )
        self._reject_shot_groups_with_ham_ops(ham_ops, shot_groups)
        config = self._maestro_config
        if ham_ops is not None and config is None:
            raise self._maestro_only("ham_ops")

        circuits = flatten_bound_payloads(payloads)
        if not circuits:
            raise ValueError("submit_circuits requires at least one payload.")
        labels = list(circuits)
        qasm_strings = list(circuits.values())
        if ham_ops is not None:
            ham_ops = pad_ham_ops(
                ham_ops, circuit_ham_map, [count_qubits(q) for q in qasm_strings]
            )
        self._record_qasm_depths(qasm_strings)

        shots = expand_shot_groups(shot_groups, len(labels))
        if shots is None:
            shots = [self.shots] * len(labels)
        if config is None:
            return self._sample_on_ibm(labels, qasm_strings, shots, cancellation_event)

        qasm_strings = _prepare_noisy_circuits(
            config, qasm_strings, estimating=ham_ops is not None
        )
        terms = (
            None
            if ham_ops is None
            else ham_ops_terms_per_circuit(len(labels), ham_ops, circuit_ham_map)
        )
        return self._run_on_maestro(
            config, labels, qasm_strings, shots, terms, cancellation_event
        )

    def submit_vjp(
        self,
        payloads: Sequence[CircuitPayload],
        *,
        ham_ops: str,
        cotangents: Mapping[str, np.ndarray],
        cancellation_event: Event | None = None,
    ) -> ExecutionResult:
        """Expectation values and vector-Jacobian products by adjoint differentiation.

        One ``adjoint_vjp`` task per payload. See
        :meth:`~divi.backends.CircuitRunner.submit_vjp`. Terms shorter than a
        circuit are padded onto its first qubits, as in :meth:`submit_circuits`.

        Raises:
            VJPUnavailableError: If :attr:`supports_vjp` is False, a payload's
                circuit cannot be differentiated, or the server refuses to on
                the configured simulator or runs out of memory.
        """
        config = self._maestro_config
        if config is None or not self.supports_vjp:
            raise VJPUnavailableError(
                "QRMIBackend computes vector-Jacobian products only on "
                "maestro-local resources, with native, noiseless estimation."
            )
        simulator = _simulator_block(config)
        raise_if_cancelled(
            cancellation_event,
            "QRMI batch cancelled before any circuit was dispatched",
        )
        compiled = _compile_adjoint_programs(payloads)
        terms = pad_ham_ops(
            ham_ops, None, [count_qubits(payload.circuit) for payload in payloads]
        ).split(";")
        if self.track_depth:
            self._record_qasm_depths(_bind_parameter_sets(payloads))

        results: list[dict] = []
        for payload, (_, angle_matrix, angle_offset) in zip(payloads, compiled):
            raise_if_cancelled(
                cancellation_event, "QRMI batch cancelled after partial completion"
            )
            labels, angles, weights = _vjp_inputs(
                payload, angle_matrix, angle_offset, cotangents
            )
            # Any bound row: parameter_rows supplies every row's gate angles.
            circuit = next(_bind_parameter_sets([payload]))
            request = _request(
                "adjoint_vjp",
                circuit,
                simulator,
                observables=terms,
                parameter_rows=angles.tolist(),
                cotangents=weights.tolist(),
            )
            try:
                raw = self._run_request(request, cancellation_event)
            except self._qrmi.UnsupportedFunctionError as exc:
                raise VJPUnavailableError(
                    f"Maestro cannot differentiate the circuit: {exc}"
                ) from exc
            except QRMITaskFailedError as exc:
                if exc.code not in _VJP_UNAVAILABLE_CODES:
                    raise
                raise VJPUnavailableError(
                    f"Maestro cannot differentiate the circuit: {exc}"
                ) from exc
            results.extend(
                _vjp_result_entries(
                    labels, terms, raw, angle_matrix, raw["time_taken"] / len(labels)
                )
            )
        return _timed_result(results)

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
            ValueError: On an IBM resource, or with a noise model, which a
                state vector cannot carry.
        """
        config = self._maestro_config
        if config is None:
            raise self._maestro_only("statevector")
        qasm, global_phase, active = _lower_for_statevector(
            config, circuit, include_idle_qubits
        )
        raw = self._run_request(
            _request("statevector", qasm, _simulator_block(config)), None
        )
        amplitudes = np.asarray(raw["amplitudes"], dtype=float) @ np.array([1, 1j])
        if not include_idle_qubits:
            amplitudes = _active_amplitudes(amplitudes, active)
        return np.exp(1j * global_phase) * amplitudes

    def expectation_value(
        self, circuit: str | QuantumCircuit, observable: SparsePauliOp
    ) -> float:
        """``<observable>`` in the output state of ``circuit``.

        Raises:
            ValueError: If ``observable`` is not Hermitian, or on an IBM
                resource.
        """
        return _expectation_value(self, circuit, observable)

    def _maestro_only(self, operation: str) -> ValueError:
        return ValueError(
            f"{operation} needs a maestro-local resource; this resource is "
            f"{self._resource_type}."
        )

    def _run_on_maestro(
        self,
        config: MaestroConfig,
        labels: Sequence[str],
        qasm_strings: Sequence[str],
        shots: Sequence[int],
        terms: list[list[str]] | None,
        cancellation_event: Event | None,
    ) -> ExecutionResult:
        """Run each circuit as its own Maestro task, in order."""
        simulator = _simulator_block(config)
        results: list[dict] = []
        for label, qasm, count, circuit_terms in zip(
            labels, qasm_strings, shots, terms or repeat(None)
        ):
            raise_if_cancelled(
                cancellation_event, "QRMI batch cancelled after partial completion"
            )
            request = _circuit_request(
                qasm,
                simulator,
                seed=_circuit_seed(config.seed, label, qasm),
                shots=count,
                terms=circuit_terms,
                noise=_noise_block(
                    config, label, qasm, estimating=circuit_terms is not None
                ),
            )
            raw = self._run_request(request, cancellation_event)
            if circuit_terms is None:
                counts = _little_endian_counts(raw["counts"], count_clbits(qasm))
                results.append(_result_entry(label, counts, raw, "counts"))
            else:
                expvals = dict(zip(circuit_terms, raw["expectation_values"]))
                results.append(_result_entry(label, expvals, raw, "expectation_values"))
        return _timed_result(results)

    def _sample_on_ibm(
        self,
        labels: Sequence[str],
        qasm_strings: Sequence[str],
        shots: Sequence[int],
        cancellation_event: Event | None,
    ) -> ExecutionResult:
        """Compile the circuits for the device and sample them, sent as QPY, in
        one job per shot count."""
        if unmeasured := [
            label for label, qasm in zip(labels, qasm_strings) if not count_clbits(qasm)
        ]:
            raise ValueError(
                "IBM resources return measurement counts, but circuits "
                f"{unmeasured} have no classical bits."
            )
        compiled, sampler_options = self._compile_for_ibm(
            [QuantumCircuit.from_qasm_str(qasm) for qasm in qasm_strings]
        )
        by_shots = bucket_by_shots(group_shots_into_ranges(shots))
        entries: dict[int, dict] = {}
        for count, indices in by_shots.items():
            job = {
                "pubs": [[compiled[index], None, count] for index in indices],
                "shots": count,
                "options": sampler_options,
                "version": 2,
                "support_qiskit": True,
            }
            payload = self._qrmi.Payload.QiskitPrimitive(
                input=json.dumps(job, cls=self._job_encoder), program_id="sampler"
            )
            raw = self._run_task(payload, cancellation_event)
            for index, pub in zip(indices, self._result_decoder.decode(raw)):
                # Registers joined in declaration order, the first rightmost.
                bits = BitArray.concatenate_bits(list(pub.data.values()))
                entries[index] = {
                    "label": labels[index],
                    "results": bits.get_counts(),
                    "metadata": dict(pub.metadata),
                }
        return ExecutionResult(
            results=[entries[index] for index in range(len(labels))],
            backend_jobs=len(by_shots),
        )

    def _compile_for_ibm(
        self, circuits: list[QuantumCircuit]
    ) -> tuple[list[QuantumCircuit], dict[str, dict[str, Any]]]:
        """``circuits`` compiled for the device, and the ``SamplerV2`` options.

        The pass manager is built on first use after each ``device_config``
        change; only fetching the device's target takes the resource lock.
        """
        with self._compile_lock:
            transpile, sampler_options, fractional_gates = self._ibm_job_settings
            if self._pass_manager is None:
                with self._resource_lock:
                    backend = self._ibm_backend_factory(
                        self._resource, use_fractional_gates=fractional_gates
                    )
                self._pass_manager = generate_preset_pass_manager(
                    backend=backend, **transpile
                )
            return self._pass_manager.run(circuits), sampler_options

    def _run_request(self, request: dict, cancellation_event: Event | None) -> dict:
        """Run one native Maestro request and return the server's reply."""
        payload = self._qrmi.maestro.request_payload(request)
        return json.loads(self._run_task(payload, cancellation_event))

    def _run_task(self, payload: Any, cancellation_event: Event | None) -> str:
        """Run one task and return its result.

        Holds the resource lock throughout, so concurrent callers queue. A
        cancellation that arrived while this thread waited stops it before
        dispatch.
        """
        with self._resource_lock:
            raise_if_cancelled(
                cancellation_event, "QRMI batch cancelled before dispatch"
            )
            task_id = self._start_task(payload)
            status = self._await_task(task_id, cancellation_event)
            return self._task_outcome(task_id, status)

    def _ensure_session(self) -> None:
        """Acquire a session unless one is held or borrowed."""
        if self._lease.session is None and not self._borrowed_session:
            self._lease.session = self._resource.acquire()
            logger.info("Acquired QRMI session %s", self._lease.session)

    def _replace_lost_session(self) -> None:
        """Acquire a fresh session, which this backend then owns."""
        self._lease.session = self._resource.acquire()
        self._borrowed_session = False
        logger.info("Replaced a lost QRMI session with %s", self._lease.session)

    def _start_task(self, payload: Any) -> str:
        """Start a task, replacing a session the server lost."""
        self._ensure_session()
        try:
            return self._resource.task_start(payload)
        except self._qrmi.ConfigError as exc:
            if not any(marker in str(exc) for marker in _LOST_SESSION_MARKERS):
                raise
            logger.info("QRMI session lost (%s)", exc)
            self._replace_lost_session()
            return self._resource.task_start(payload)

    def _await_task(self, task_id: str, cancellation_event: Event | None) -> Any:
        """Poll until the task ends, stopping it on any other exit."""
        status_type = self._qrmi.TaskStatus
        finished = (status_type.Completed, status_type.Failed, status_type.Cancelled)
        event = cancellation_event if cancellation_event is not None else Event()
        try:
            while True:
                raise_if_cancelled(
                    event, f"QRMI batch cancelled while task {task_id} was running"
                )
                if (status := self._resource.task_status(task_id)) in finished:
                    return status
                event.wait(self._polling_interval)
        except BaseException:
            self._stop_task(task_id)
            raise

    def _task_outcome(self, task_id: str, status: Any) -> str:
        """An ended task's result, or ``QRMITaskFailedError`` with why it failed.

        An IBM job is deleted once read, and kept if its result cannot be read.
        """
        if status != self._qrmi.TaskStatus.Completed:
            try:
                code, reason = self._task_failure(task_id)
            finally:
                self._delete_ibm_job(task_id)
            raise QRMITaskFailedError(
                f"Task {task_id} ended as {status}: {reason}", code
            )
        try:
            value = self._resource.task_result(task_id).value
        except BaseException:
            if not self._is_maestro:
                logger.error(
                    "IBM job %s finished but its result could not be read; the "
                    "job is kept on the service.",
                    task_id,
                )
            raise
        self._delete_ibm_job(task_id)
        return value

    def _task_failure(self, task_id: str) -> tuple[str | None, str]:
        """The resource's error code for a failed task, and its account of why."""
        if self._is_maestro:
            return self._maestro_failure(task_id)
        return None, self._ibm_failure(task_id)

    def _maestro_failure(self, task_id: str) -> tuple[str | None, str]:
        """The server's error code and message for a failed task, then its logs."""
        try:
            logs = json.loads(self._resource.task_logs(task_id))
        except (RuntimeError, ValueError) as exc:
            return None, f"no diagnostics available ({exc})"
        error = logs.get("error")
        if isinstance(error, dict) and "message" in error:
            code = error.get("code")
            reason = f"{code}: {error['message']}"
        else:
            code, reason = None, logs.get("failure_message") or "no failure recorded"
        output = (logs.get("logs") or "").strip()
        return code, f"{reason}\n{output}" if output else reason

    def _ibm_failure(self, task_id: str) -> str:
        """IBM's code, message and solution for a failed job, then its logs."""
        try:
            self._resource.task_result(task_id)
            reason = "no failure recorded"
        except self._qrmi.TaskNotReadyError as exc:
            reason = str(exc)
        try:
            logs = self._resource.task_logs(task_id).strip()
        except RuntimeError:
            logs = ""
        return f"{reason}\n{logs}" if logs else reason

    def _delete_ibm_job(self, task_id: str) -> None:
        """Delete a finished IBM job; ``task_stop`` deletes on IBM."""
        if not self._is_maestro:
            self._stop_task(task_id)

    def _stop_task(self, task_id: str) -> None:
        """Ask the resource to stop a task; a refusal is logged, not raised."""
        try:
            self._resource.task_stop(task_id)
        except RuntimeError as exc:
            logger.warning("Could not stop task %s: %s", task_id, exc)
