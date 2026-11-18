# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import itertools
import json
import logging
import os
import platform
import re
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from enum import Enum
from http import HTTPStatus
from importlib import metadata
from threading import Event
from typing import Any, TypeVar

import numpy as np
import requests
from dotenv import dotenv_values, find_dotenv
from requests.adapters import HTTPAdapter, Retry
from rich.console import Console

from divi.backends._base import AsyncJobBackend, ExecutionResult
from divi.backends._cancellation import (
    _cancel_job_on_abort,
    _sigint_to_event,
    raise_if_cancelled,
)
from divi.backends._config import (
    _TARGET_FIELDS,
    DeviceConfig,
    JobConfig,
    require_device_config,
)
from divi.backends._job_status import (
    InsufficientCreditsError,
    JobCancelledError,
    JobFailedError,
    JobStatus,
    JobTimedOutError,
    QoroJobError,
)
from divi.backends._pauli_serde import compress_ham_ops, compress_text, pad_ham_ops
from divi.backends._results_processing import (
    _decode_histogram_b64,
    _reverse_bitstrings,
)
from divi.backends._saved_defaults import defaults_path, load_defaults
from divi.backends._systems import (
    QPUSystem,
    SimulatorCluster,
    get_qpu_system,
    get_simulator_cluster,
    parse_qpu_systems,
    parse_simulator_clusters,
    update_qpu_systems_cache,
    update_simulator_clusters_cache,
)
from divi.circuits._payloads import (
    CircuitBatch,
    CircuitPayload,
    flatten_bound_payloads,
    is_bound,
    normalise_to_payloads,
)
from divi.exceptions import (
    CharacterizationSubmitError,
    CircuitTooWideError,
    ExecutionCancelledError,
    VJPUnavailableError,
)
from divi.qasm import (
    _format_validation_error_with_context,
    count_qubits,
    validate_qasm,
)

from ._maestro import MaestroConfig, require_maestro_config
from ._maestro_payload import maestro_config_from_payload, maestro_config_to_payload
from ._qoro_adjoint import run_adjoint_job

API_URL = "https://app.qoroquantum.net/api"
_MAX_CIRCUITS_PER_REQUEST = 3000

T = TypeVar("T")


def _build_user_agent() -> str:
    try:
        version = metadata.version("qoro-divi")
    except metadata.PackageNotFoundError:
        version = "unknown"
    return (
        f"divi/{version} (python={platform.python_version()}) "
        f"{requests.utils.default_user_agent()}"
    )


USER_AGENT = _build_user_agent()

session = requests.Session()
retry_configuration = Retry(
    total=5,
    backoff_factor=0.1,
    # Gateway errors, including Cloudflare's 52x origin errors.
    status_forcelist=[502, 503, 504, 520, 521, 522, 523, 524],
    allowed_methods=["GET", "DELETE"],
    raise_on_status=False,
)

session.mount("http://", HTTPAdapter(max_retries=retry_configuration))
session.mount("https://", HTTPAdapter(max_retries=retry_configuration))

# Retries only a connection that never opened, which the server never saw.
connect_retry_configuration = Retry(
    total=3, connect=3, read=False, other=0, backoff_factor=0.5
)


def _send_once(method: str, url: str, **kwargs) -> requests.Response:
    """Send a request on a fresh connection, never resending it once delivered."""
    with requests.Session() as fresh:
        fresh.mount("http://", HTTPAdapter(max_retries=connect_retry_configuration))
        fresh.mount("https://", HTTPAdapter(max_retries=connect_retry_configuration))
        return fresh.request(method, url, **kwargs)


logger = logging.getLogger(__name__)


def _raise_with_details(resp: requests.Response):
    try:
        data = resp.json()
        body = json.dumps(data, ensure_ascii=False)
    except ValueError:
        # Non-JSON response (e.g. Cloudflare HTML error page) — don't
        # dump hundreds of lines of HTML on the user.
        text = resp.text or ""
        if "<html" in text.lower():
            body = (
                f"(HTML error page from {resp.url or 'server'}; "
                f"upstream may be down or timed out)"
            )
        else:
            # Keep non-HTML text but truncate if very long
            body = text[:500] + ("..." if len(text) > 500 else "")
    msg = f"{resp.status_code} {resp.reason}: {body}"
    raise requests.HTTPError(msg, response=resp)


def _config_body(
    maestro_config: MaestroConfig | None, device_config: DeviceConfig | None
) -> dict[str, dict]:
    """The ``maestro_config`` / ``device_config`` keys of a request body; a
    ``DeviceConfig`` with no options set is left out."""
    body: dict[str, dict] = {}
    if maestro_config is not None:
        body["maestro_config"] = maestro_config_to_payload(maestro_config)
    if device_config is not None and (device := device_config.to_payload()):
        body["device_config"] = device
    return body


def _by_config_key(specs: Iterable[Mapping]) -> dict[str, dict[str, Any]]:
    """Re-key a blueprint's field specs by the config key they describe."""
    return {
        spec["name"]: {k: v for k, v in spec.items() if k != "name"} for spec in specs
    }


def _is_recoverable_characterization_error(exc: Exception) -> bool:
    """Whether an existing job may need inspection after this request failed."""
    if isinstance(
        exc,
        (requests.exceptions.Timeout, requests.exceptions.ConnectionError),
    ):
        return True
    return (
        isinstance(exc, requests.exceptions.HTTPError)
        and exc.response is not None
        and exc.response.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR
    )


_QUBIT_LIMIT_PATTERN = re.compile(r"exceeds qubit limit (\d+)")


def _circuit_too_wide(exc: requests.HTTPError) -> CircuitTooWideError | None:
    """The typed error for a 400 refusing a circuit's width, else ``None``."""
    response = exc.response
    if response is None or response.status_code != HTTPStatus.BAD_REQUEST:
        return None
    if (match := _QUBIT_LIMIT_PATTERN.search(str(exc))) is None:
        return None
    return CircuitTooWideError(str(exc), limit=int(match[1]), response=response)


def _encode_parameter_set(
    label: str,
    values: Sequence[float],
    cotangents: Mapping[str, np.ndarray] | None,
) -> dict[str, Any]:
    """A ``parameter_sets`` row as sent, with its cotangent when given."""
    row: dict[str, Any] = {"label": label, "values": [float(v) for v in values]}
    if cotangents is not None:
        row["cotangent"] = [float(w) for w in cotangents[label]]
    return row


class JobType(Enum):
    """Type of job to execute on the Qoro Service."""

    EXECUTE = "EXECUTE"
    """Run circuits and return measurement count histograms."""

    EXPECTATION = "EXPECTATION"
    """Compute expectation values for Hamiltonian operators."""

    CHARACTERIZE = "VALIDATE"
    """Submit a QUBO/HUBO for characterisation (no simulator/QPU needed).

    The wire value remains ``"VALIDATE"`` for server compatibility.
    """


class MaxRetriesReachedError(Exception):
    """Exception raised when the maximum number of retries is reached."""

    def __init__(self, job_id, retries):
        self.job_id = job_id
        self.retries = retries
        self.message = (
            f"Maximum retries reached: {retries} retries attempted for job {job_id}"
        )
        super().__init__(self.message)


_DEFAULT_SIMULATOR_CLUSTER = SimulatorCluster(name="qoro_maestro")
_DEFAULT_JOB_CONFIG = JobConfig(
    simulator_cluster=_DEFAULT_SIMULATOR_CLUSTER,
    use_circuit_packing=False,
)


class QoroService(AsyncJobBackend):
    """A client for interacting with the Qoro Quantum Service API.

    This class provides methods to submit circuits, check job status,
    and retrieve results from the Qoro platform.

    Resolves parameters server-side, so the pipeline can hand it parametric
    QASM-encoded :class:`~divi.circuits.CircuitPayload` objects and skip binding
    near-identical circuits locally on variational sweeps.
    """

    def __init__(
        self,
        auth_token: str | None = None,
        job_config: JobConfig | None = None,
        maestro_config: MaestroConfig | None = None,
        device_config: DeviceConfig | None = None,
        polling_interval: float = 3.0,
        max_retries: int | None = None,
        track_depth: bool = False,
        use_saved_defaults: bool = True,
    ):
        """Initialises the QoroService client.

        Args:
            auth_token (str | None, optional):
                The authentication token for the Qoro API. If not provided,
                it will be read from ``QORO_API_KEY`` in a ``.env`` file,
                falling back to the ``QORO_API_KEY`` environment variable.
            job_config (JobConfig | None, optional):
                A JobConfig object containing default job settings. Its set
                fields override the saved defaults' ``job_config`` (see below);
                with neither, the ``qoro_maestro`` simulator cluster is used. If
                the result has neither ``simulator_cluster`` nor
                ``qpu_system``, it defaults to ``qoro_maestro`` with a warning.
            maestro_config (MaestroConfig | None, optional):
                Default Maestro settings for submitted jobs, the same
                :class:`~divi.backends.MaestroConfig` a local
                :class:`~divi.backends.MaestroSimulator` takes. Every call to
                :meth:`submit_circuits` on a simulator target uses it unless
                given its own ``maestro_config``. Replaces the saved default.
            device_config (DeviceConfig | None, optional):
                Default hardware options for jobs on a QPU target, used unless
                :meth:`submit_circuits` is given its own ``device_config``.
                Replaces the saved default.
            polling_interval (float, optional):
                The interval in seconds for polling job status. Defaults to 3.0.
            max_retries (int | None, optional):
                The maximum number of retries for polling. ``None`` (the
                default) polls indefinitely until the job reaches a terminal
                state or polling is cancelled. Pass an integer to cap the
                number of attempts.
            track_depth (bool, optional):
                If True, record circuit depth for each submitted batch.
                Access via :attr:`~divi.backends.CircuitRunner.depth_history` after execution. Defaults to False.
            use_saved_defaults (bool, optional):
                Whether to start from the defaults ``divi-config-helper`` saved
                in ``qoro.json`` in divi's user config folder. Defaults to True.

        Raises:
            ValueError: If no API key is found, or the saved defaults are
                invalid or name a target this key cannot use; the message names
                the file.
        """

        # Set up auth_token first (needed for API calls like fetch_simulator_clusters)
        if auth_token is None:
            try:
                env_path = find_dotenv(usecwd=True)
                auth_token = dotenv_values(env_path)["QORO_API_KEY"]
            except KeyError:
                auth_token = os.environ.get("QORO_API_KEY")
        if auth_token is None:
            raise ValueError(
                "Qoro API key not provided nor found in a .env file "
                "or QORO_API_KEY environment variable."
            )

        self.auth_token = "Bearer " + auth_token

        # Fetch available systems (needs auth_token to be set)
        self.fetch_qpu_systems()
        self.fetch_simulator_clusters()

        saved_job, saved_maestro, saved_device = (
            load_defaults() if use_saved_defaults else (None, None, None)
        )
        passed = job_config._set_fields() if job_config is not None else {}
        if saved_job is None:
            self.job_config = _DEFAULT_JOB_CONFIG if job_config is None else job_config
        else:
            try:
                self.job_config = saved_job.override(**passed)
            except ValueError as exc:
                if passed.keys() & _TARGET_FIELDS:
                    raise
                raise ValueError(f"{defaults_path()}: {exc}") from exc

        self.maestro_config = (
            maestro_config if maestro_config is not None else saved_maestro
        )
        self.device_config = (
            device_config if device_config is not None else saved_device
        )

        super().__init__(
            shots=self.job_config.shots,
            track_depth=track_depth,
            polling_interval=polling_interval,
            max_retries=max_retries,
        )

    @property
    def supports_expval(self) -> bool:
        """
        Whether the backend supports expectation value measurements.
        """
        target = self._resolved_target(self.job_config)
        return target.supports_expval and not self.job_config.force_sampling

    @staticmethod
    def _resolved_target(job_config: JobConfig) -> SimulatorCluster | QPUSystem:
        target = job_config.simulator_cluster or job_config.qpu_system
        if not isinstance(target, (SimulatorCluster, QPUSystem)):
            raise TypeError(
                f"The job target must be a SimulatorCluster or QPUSystem; got {target!r}."
            )
        return target

    @property
    def shots(self) -> int:
        """Shots per circuit, from the service's ``job_config``."""
        return self.job_config.shots

    @property
    def job_config(self) -> JobConfig:
        """The service's default job configuration."""
        return self._job_config

    @job_config.setter
    def job_config(self, value: JobConfig) -> None:
        if not isinstance(value, JobConfig):
            raise TypeError(
                f"job_config must be a JobConfig, got {type(value).__name__}."
            )
        self._job_config = self._resolve_and_validate_target(value)

    @property
    def maestro_config(self) -> MaestroConfig | None:
        """The service's default Maestro settings."""
        return self._maestro_config

    @maestro_config.setter
    def maestro_config(self, value: MaestroConfig | None) -> None:
        if value is not None:
            require_maestro_config(value, "maestro_config")
        self._maestro_config = value

    @property
    def device_config(self) -> DeviceConfig | None:
        """The service's default hardware options for QPU jobs."""
        return self._device_config

    @device_config.setter
    def device_config(self, value: DeviceConfig | None) -> None:
        if value is not None:
            require_device_config(value)
        self._device_config = value

    @property
    def resolves_parameters(self) -> bool:
        """The Qoro backend substitutes parameter values server-side."""
        return True

    @property
    def supports_vjp(self) -> bool:
        """Whether estimation is native and noiseless, as adjoint differentiation needs.

        The service itself refuses simulators it cannot differentiate on.
        """
        config = self.maestro_config
        return self.supports_expval and (config is None or config.noise_model is None)

    def _resolve_and_validate_target(self, config: JobConfig) -> JobConfig:
        """Ensures the config has a valid target, resolving strings if needed.

        If neither ``simulator_cluster`` nor ``qpu_system`` is set, defaults to
        the ``qoro_maestro`` simulator cluster with a warning.
        """
        if config.simulator_cluster is None and config.qpu_system is None:
            warnings.warn(
                "No simulator_cluster or qpu_system specified in JobConfig. "
                f"Defaulting to simulator cluster '{_DEFAULT_SIMULATOR_CLUSTER.name}'.",
                stacklevel=2,
            )
            return config.model_copy(
                update={"simulator_cluster": _DEFAULT_SIMULATOR_CLUSTER}
            )

        if isinstance(config.simulator_cluster, str):
            resolved = get_simulator_cluster(config.simulator_cluster)
            return config.model_copy(update={"simulator_cluster": resolved})

        if isinstance(config.qpu_system, str):
            resolved = get_qpu_system(config.qpu_system)
            return config.model_copy(update={"qpu_system": resolved})

        return config

    def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> requests.Response:
        """
        Make an authenticated HTTP request to the Qoro API.

        This internal method centralises all API communication, handling authentication
        headers and error responses consistently. A POST is sent once on a fresh
        connection, since a resend could apply it twice; other methods are
        retried on transport failure or gateway error.

        Args:
            method (str): HTTP method to use (e.g., 'get', 'post', 'delete').
            endpoint (str): API endpoint path (without base URL).
            **kwargs: Additional arguments to pass to requests.Session.request(), such as
                'json', 'timeout', 'params', etc.

        Returns:
            requests.Response: The HTTP response object from the API.

        Raises:
            requests.exceptions.HTTPError: If the response status code is 400 or above.
        """
        url = f"{API_URL}/{endpoint}"

        headers = {"Authorization": self.auth_token, "User-Agent": USER_AGENT}

        if method.upper() == "POST":
            headers["Content-Type"] = "application/json"

        # Allow overriding default headers
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))

        requester = _send_once if method.upper() == "POST" else session.request
        response = requester(method, url, headers=headers, **kwargs)

        # Raise with comprehensive error details if request failed
        if response.status_code >= 400:
            _raise_with_details(response)

        return response

    def _extract_job_id(self, execution_result: ExecutionResult) -> str:
        job_id = execution_result.job_id
        if job_id is None:
            raise ValueError(
                "ExecutionResult must have a job_id. "
                "This ExecutionResult appears to be from a synchronous backend."
            )
        return job_id

    def test_connection(self):
        """
        Test the connection to the Qoro API.

        Sends a simple GET request to verify that the API is reachable and
        the authentication token is valid.

        Returns:
            requests.Response: The response from the API ping endpoint.

        Raises:
            requests.exceptions.HTTPError: If the connection fails or authentication
                is invalid.
        """
        return self._make_request("get", "", timeout=10)

    def fetch_qpu_systems(self) -> list[QPUSystem]:
        """
        Get the list of available QPU systems from the Qoro API.

        Returns:
            List of QPUSystem objects.
        """
        response = self._make_request("get", "qpusystem/", timeout=10)
        systems = parse_qpu_systems(response.json())
        update_qpu_systems_cache(systems)
        return systems

    def fetch_simulator_clusters(self) -> list[SimulatorCluster]:
        """
        Get the list of available simulator clusters from the Qoro API.

        Returns:
            List of SimulatorCluster objects.
        """
        response = self._make_request("get", "simulatorcluster/", timeout=10)
        clusters = parse_simulator_clusters(response.json())
        update_simulator_clusters_cache(clusters)
        return clusters

    def fetch_vendor_blueprints(self) -> dict[str, dict[str, Any]]:
        """
        Get the config shape each QPU vendor accepts from the Qoro API.

        Describes shape only: no stored values are returned, and never a
        credential. Each QPU carries its own stored values, set on the Qoro
        dashboard.

        Returns:
            Blueprints keyed by vendor name (e.g. ``"ibm"``, ``"iqm"``), each
            holding a ``label`` plus ``credentials`` and ``device`` maps of
            config key to the spec the service gives it, such as its
            ``kind``, ``choices``, ``default`` and bounds::

                blueprints["ibm"]["device"]["TRANSPILE_OPTIMIZATION_LEVEL"]["default"]

            A vendor whose shape is not modelled yet reports empty maps.
        """
        response = self._make_request("get", "vendor-config-blueprint/", timeout=10)
        return {
            vendor: {
                "label": blueprint["label"],
                "credentials": _by_config_key(blueprint["credentials"]),
                "device": _by_config_key(blueprint["device"]),
            }
            for vendor, blueprint in response.json().items()
        }

    def get_credit_balance(self) -> dict:
        """
        Get the current credit balance for the authenticated user.

        Returns:
            dict: A dictionary containing the credit account information::

                {
                    "balance": "500.00",
                    "total_used": "0",
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": "2026-01-01T00:00:00Z"
                }

        Raises:
            requests.exceptions.HTTPError: If the request fails (e.g., 401
                if the token is deactivated).
        """
        response = self._make_request("get", "credits/", timeout=10)
        return response.json()

    def get_credit_transactions(self, page: int = 1, page_size: int = 20) -> dict:
        """
        Get paginated credit transaction history for the authenticated user.

        Args:
            page (int, optional): Page number to retrieve. Defaults to 1.
            page_size (int, optional): Number of transactions per page.
                Defaults to 20. Maximum is 100.

        Returns:
            dict: A paginated response containing transaction records::

                {
                    "count": 1,
                    "total_pages": 1,
                    "next": null,
                    "previous": null,
                    "results": [
                        {
                            "id": 1,
                            "amount": "500.00",
                            "balance_after": "500.00",
                            "transaction_type": "PURCHASE",
                            "description": "...",
                            "job_id": null,
                            "created_at": "2026-01-01T00:00:00Z"
                        }
                    ]
                }

        Raises:
            requests.exceptions.HTTPError: If the request fails (e.g., 401
                if the token is deactivated).
        """
        response = self._make_request(
            "get",
            "credits/transactions/",
            params={"page": page, "page_size": page_size},
            timeout=10,
        )
        return response.json()

    def _require_expval(self, job_config: JobConfig) -> None:
        """Raise if ``job_config`` cannot run an expectation-value job."""
        target = self._resolved_target(job_config)
        if job_config.force_sampling:
            reason = "its job_config sets force_sampling=True"
        elif not target.supports_expval:
            reason = f"'{target.name}' does not estimate expectation values"
        else:
            return
        raise ValueError(f"ham_ops cannot be used on this job: {reason}.")

    def _configs_for_target(
        self,
        job_config: JobConfig,
        maestro_config: MaestroConfig | None,
        device_config: DeviceConfig | None,
    ) -> tuple[MaestroConfig | None, DeviceConfig | None]:
        """The ``(maestro_config, device_config)`` a job on ``job_config``'s
        target takes.

        A simulator job takes only Maestro settings and a QPU job only device
        options, so the other kind's service-level default is skipped; a
        per-call config aimed at the other kind of target raises.
        """
        if isinstance(job_config.qpu_system, QPUSystem):
            if maestro_config is not None:
                raise ValueError(
                    "maestro_config applies to simulator targets; this job "
                    f"targets QPU system '{job_config.qpu_system.name}'. Pass "
                    "device_config for hardware options instead."
                )
            if device_config is None:
                return None, self.device_config
            require_device_config(device_config)
            return None, device_config
        if device_config is not None:
            raise ValueError(
                "device_config applies to QPU targets; this job targets a "
                "simulator cluster. Pass maestro_config instead."
            )
        if maestro_config is None:
            return self.maestro_config, None
        require_maestro_config(maestro_config, "maestro_config")
        return maestro_config, None

    @staticmethod
    def _resolve_job_type(job_type: JobType | None, ham_ops: str | None) -> JobType:
        """Resolve the job type, inferring EXPECTATION from ``ham_ops``."""
        if ham_ops is None:
            if job_type is not None and job_type is not JobType.EXECUTE:
                raise ValueError(
                    "submit_circuits runs EXECUTE jobs, or EXPECTATION jobs when "
                    f"ham_ops is given; got job_type {job_type.name} without ham_ops."
                )
            return JobType.EXECUTE
        if job_type is not None and job_type != JobType.EXPECTATION:
            raise ValueError(
                "Hamiltonian operators are only supported for EXPECTATION job type."
            )
        return JobType.EXPECTATION

    def submit_circuits(
        self,
        payloads: Sequence[CircuitPayload] | CircuitBatch,
        *,
        ham_ops: str | None = None,
        circuit_ham_map: Sequence[list[int]] | None = None,
        shot_groups: Sequence[list[int]] | None = None,
        job_type: JobType | None = None,
        maestro_config: MaestroConfig | None = None,
        device_config: DeviceConfig | None = None,
        job_config: JobConfig | None = None,
        cancellation_event: Event | None = None,
        **kwargs,
    ) -> ExecutionResult:
        """
        Submit payloads to the Qoro API for execution.

        A single ``job/init/`` call returns the ``job_id``, then one or more
        ``add_circuits/`` calls upload the payload; only the last is marked
        ``finalized``.

        Parametric payloads travel as a compressed ``circuit_template`` plus the
        parameter matrix, and the backend resolves one circuit per row. Bound
        payloads (the degenerate no-parameter case) travel as a ``label ->
        circuit`` mapping instead. Either way results come back keyed by the
        labels supplied in ``parameter_sets``.

        An ensemble that merges programs into one call (``BatchMode.MERGED``)
        sends their circuits bound, so those runs take the mapping shape
        regardless of the parameters they started with.

        The two shapes resolve floats differently: the backend substitutes
        via ``str(value)`` on the decoded float, preserving the full
        Python repr, whereas bound QASM was rendered locally at the
        :class:`~divi.circuits.MetaCircuit`'s precision. Values that
        round-trip through both formatters give
        byte-identical circuits; others differ only in the trailing digits.

        Args:
            payloads (Sequence[CircuitPayload] | CircuitBatch):
                One :class:`~divi.circuits.CircuitPayload` per ``(body, measurement)``
                variant in the compiled batch, each carrying its own
                ``parameter_sets`` rows pre-labelled with deterministic
                ``BranchKey``-derived labels — or a collection of
                already-resolved circuits: a mapping of unique circuit ID →
                QASM string or :class:`~qiskit.circuit.QuantumCircuit`, or a
                bare sequence labelled by positional index.
            ham_ops (str | None, optional):
                String representing the Hamiltonian operators to measure, semicolon-separated.
                Each term is a combination of Pauli operators, e.g. "XYZ;XXZ;ZIZ".
                Groups can be pipe-delimited (e.g. "XYZ;XXZ|ZI;IZ") and assigned
                to circuits with ``circuit_ham_map``. If None, no Hamiltonian
                operators will be measured.
                Terms shorter than a circuit are padded onto its first qubits,
                with a warning.
            circuit_ham_map (list[list[int]] | None, optional):
                One ``[start, end)`` circuit slice per ``|``-group of
                ``ham_ops``, covering every circuit once; without it every
                circuit takes all terms. Bound payloads only.
            shot_groups (list[list[int]] | None, optional):
                Per-circuit shot allocation as ``[start, end, shots]`` triples
                covering the circuits. Mutually exclusive with ``shots``;
                bound payloads only.
            job_type (JobType | None, optional):
                Type of job to execute: EXECUTE, or EXPECTATION with
                ``ham_ops``. If not provided, inferred from ``ham_ops``.
            maestro_config (MaestroConfig | None, optional):
                Maestro settings for this submission on a simulator target,
                used in place of the service's ``maestro_config``, e.g.
                ``service.maestro_config.override(max_bond_dimension=32)``.
                When omitted, the service's is used; a QPU submission skips it.
            device_config (DeviceConfig | None, optional):
                Hardware options for this submission on a QPU target, used in
                place of the service's ``device_config``. When omitted, the
                service's is used; a simulator submission skips it. The
                service rejects an option or value that no QPU vendor accepts;
                :meth:`fetch_vendor_blueprints` lists them.
            job_config (JobConfig | None, optional):
                Job settings for this submission, used in place of the
                service's ``job_config``, e.g.
                ``service.job_config.override(shots=2000)``.
            cancellation_event (Event | None, optional):
                Accepted for :class:`~divi.backends.CircuitRunner` interface
                parity; submission itself is unaffected. Pass the same Event
                to :meth:`poll_job_status` to interrupt the polling loop.
            **kwargs: Rejected with ``TypeError``.

        Raises:
            ValueError: If any circuit is not valid QASM, ``job_type`` does
                not match the presence of ``ham_ops``, ``ham_ops`` is given
                for a job whose target or ``force_sampling`` rules out
                expectation values, ``ham_ops`` and ``shot_groups`` are both
                given, ``circuit_ham_map`` does not cover every circuit exactly
                once, ``circuit_ham_map`` or ``shot_groups`` is given with
                parametric payloads, or a config is aimed at the other kind of
                target (``maestro_config`` on a QPU job, or ``device_config``
                on a simulator job).
            ~divi.exceptions.CircuitTooWideError: If the API refuses a circuit
                for exceeding the account's qubit limit.
            requests.exceptions.HTTPError: If any API request fails, including
                ``job/init/`` refusing a ``device_config`` option or value.

        Returns:
            ExecutionResult: Contains job_id for asynchronous execution. Use the job_id
                to poll for results using backend.poll_job_status() and get_job_results().
        """
        self._reject_unknown_options(kwargs)
        payloads = normalise_to_payloads(payloads)
        if not payloads:
            raise ValueError("submit_circuits requires at least one payload.")

        self._reject_shot_groups_with_ham_ops(ham_ops, shot_groups)
        job_type = self._resolve_job_type(job_type, ham_ops)
        bound = is_bound(payloads)
        if not bound and circuit_ham_map is not None:
            raise ValueError(
                "circuit_ham_map is not supported for parametric payloads because "
                "their ordering does not align with the bound-circuit flat "
                "index ranges it references. Bind the parameters first if your "
                "batch needs |-delimited ham_ops with circuit_ham_map."
            )
        if not bound and shot_groups is not None:
            raise ValueError(
                "shot_groups is not supported for parametric payloads for the "
                "same reason as circuit_ham_map; per-circuit shot allocation "
                "would need re-indexing into the payload order."
            )

        if ham_ops is not None:
            widths = [
                width
                for payload in payloads
                for width in itertools.repeat(
                    count_qubits(payload.circuit), len(payload.parameter_sets)
                )
            ]
            ham_ops = pad_ham_ops(ham_ops, circuit_ham_map, widths)

        job_config = (
            self.job_config
            if job_config is None
            else self._resolve_and_validate_target(job_config)
        )
        if ham_ops is not None:
            self._require_expval(job_config)
        maestro_config, device_config = self._configs_for_target(
            job_config, maestro_config, device_config
        )
        per_circuit = {
            "ham_ops": ham_ops,
            "circuit_ham_map": circuit_ham_map,
            "shot_groups": shot_groups,
        }
        build_call_plan = (
            self._build_bound_call_plan if bound else self._build_parametric_call_plan
        )
        shots = job_config.shots if ham_ops is None and shot_groups is None else None

        return self._dispatch_job(
            build_call_plan(payloads, per_circuit),
            job_config=job_config,
            maestro_config=maestro_config,
            device_config=device_config,
            job_type=job_type,
            shots=shots,
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

        Submits one ``EXPECTATION`` job in which every parameter set carries its
        cotangent, and waits for it. See
        :meth:`~divi.backends.CircuitRunner.submit_vjp`. Terms shorter than a
        circuit are padded onto its first qubits, as in :meth:`submit_circuits`.

        Raises:
            VJPUnavailableError: If :attr:`supports_vjp` is False, a payload's
                circuit cannot be differentiated, or the service fails the job.
        """
        if not self.supports_vjp:
            raise VJPUnavailableError(
                "QoroService computes vector-Jacobian products only with "
                "native, noiseless estimation."
            )
        return run_adjoint_job(
            self,
            JobType.EXPECTATION,
            payloads,
            ham_ops=ham_ops,
            cotangents=cotangents,
            cancellation_event=cancellation_event,
        )

    def _slice_requests(
        self, items: Sequence[T], per_circuit: Mapping[str, Any]
    ) -> Iterator[tuple[Sequence[T], dict[str, Any]]]:
        """Each request's share of at most :data:`_MAX_CIRCUITS_PER_REQUEST`
        items, with ``per_circuit`` encoded for that share alone."""
        decoded = [
            (kwarg, values)
            for kwarg in self.per_circuit_kwargs
            if (values := kwarg.decode(per_circuit, len(items))) is not None
        ]
        for start in range(0, len(items), _MAX_CIRCUITS_PER_REQUEST):
            part = slice(start, start + _MAX_CIRCUITS_PER_REQUEST)
            fields: dict[str, Any] = {}
            for kwarg, values in decoded:
                fields |= kwarg.encode(values[part])
            if "ham_ops" in fields:
                fields["observables"] = compress_ham_ops(fields.pop("ham_ops"))
            yield items[part], fields

    def _build_bound_call_plan(
        self,
        payloads: Sequence[CircuitPayload],
        per_circuit: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """``add_circuits/`` fragments for resolved circuits."""
        circuits = flatten_bound_payloads(payloads)
        for key, circuit in circuits.items():
            try:
                validate_qasm(circuit)
            except SyntaxError as e:
                msg = _format_validation_error_with_context(circuit, e)
                raise ValueError(f"Circuit '{key}' is not a valid QASM: {msg}") from e

        items = [(label, compress_text(qasm)) for label, qasm in circuits.items()]
        call_plan = [
            {"circuits": dict(chunk), **fields}
            for chunk, fields in self._slice_requests(items, per_circuit)
        ]
        self._record_qasm_depths(circuits.values())
        return call_plan

    def _build_parametric_call_plan(
        self,
        payloads: Sequence[CircuitPayload],
        per_circuit: Mapping[str, Any],
        cotangents: Mapping[str, np.ndarray] | None = None,
    ) -> list[dict[str, Any]]:
        """``add_circuits/`` fragments for templates and their parameter rows,
        each row carrying its cotangent when ``cotangents`` is given."""
        call_plan: list[dict[str, Any]] = []
        for payload in payloads:
            if not isinstance(payload.circuit, str):
                raise TypeError(
                    "QoroService resolves parameters server-side and needs "
                    f"QASM-encoded payloads; got {type(payload.circuit).__name__}."
                )
            # One check per template, not per resolved row: the rows differ
            # only in the substituted values.
            try:
                validate_qasm(payload.circuit, payload.parameter_names)
            except SyntaxError as e:
                msg = _format_validation_error_with_context(payload.circuit, e)
                raise ValueError(f"Circuit template is not valid QASM: {msg}") from e

            template = {
                "circuit_template": compress_text(payload.circuit),
                "parameter_names": list(payload.parameter_names),
            }
            rows = [
                _encode_parameter_set(label, values, cotangents)
                for label, values in payload.parameter_sets
            ]
            call_plan += [
                {**template, "parameter_sets": chunk, **fields}
                for chunk, fields in self._slice_requests(rows, per_circuit)
            ]
        self._record_payload_depths(payloads)
        return call_plan

    def _dispatch_job(
        self,
        call_plan: Sequence[dict[str, Any]],
        *,
        job_config: JobConfig,
        maestro_config: MaestroConfig | None,
        device_config: DeviceConfig | None,
        job_type: JobType,
        shots: int | None = None,
    ) -> ExecutionResult:
        """Open a job, upload every payload fragment, return its ``job_id``.

        The plan is complete before ``job/init/`` runs, so the last fragment
        is known up front and is the only one marked ``finalized``. Neither
        request is resent once delivered: a resend after a lost response would
        open a second job or append the circuits twice. A failed upload cancels the job it
        opened.

        Raises:
            ~divi.exceptions.CircuitTooWideError: If the API refuses a circuit
                for exceeding the account's qubit limit.
        """
        init_payload: dict[str, Any] = {
            "tag": job_config.tag,
            "job_type": job_type.value,
            "use_packing": job_config.use_circuit_packing or False,
        }
        if isinstance(job_config.simulator_cluster, SimulatorCluster):
            init_payload["simulator_cluster"] = job_config.simulator_cluster.name
        elif isinstance(job_config.qpu_system, QPUSystem):
            init_payload["qpu_system_name"] = job_config.qpu_system.name
        init_payload |= _config_body(maestro_config, device_config)

        init_response = self._make_request(
            "post", "job/init/", json=init_payload, timeout=100
        )
        job_id = init_response.json()["job_id"]

        with _cancel_job_on_abort(self, ExecutionResult(job_id=job_id)):
            try:
                for i, fragment in enumerate(call_plan):
                    payload = {
                        "mode": "append",
                        **fragment,
                        "finalized": "true" if i == len(call_plan) - 1 else "false",
                    }
                    if shots is not None:
                        payload["shots"] = shots
                    self._make_request(
                        "post",
                        f"job/{job_id}/add_circuits/",
                        json=payload,
                        timeout=100,
                    )
            except requests.HTTPError as exc:
                if too_wide := _circuit_too_wide(exc):
                    raise too_wide from exc
                raise

        return ExecutionResult(job_id=job_id, backend_jobs=len(call_plan))

    def delete_job(self, execution_result: ExecutionResult) -> requests.Response:
        """
        Delete a job from the Qoro Database.

        Args:
            execution_result: An ExecutionResult instance with a job_id to delete.
        Returns:
            requests.Response: The response from the API.
        Raises:
            ValueError: If the ExecutionResult does not have a job_id.
        """
        job_id = self._extract_job_id(execution_result)
        return self._make_request(
            "delete",
            f"job/{job_id}",
            timeout=50,
        )

    def cancel_job(self, execution_result: ExecutionResult) -> requests.Response:
        """
        Cancel a job on the Qoro Service.

        Args:
            execution_result: An ExecutionResult instance with a job_id to cancel.
        Returns:
            requests.Response: The response from the API. Use response.json() to get
                the cancellation details (status, job_id, circuits_cancelled).
        Raises:
            ValueError: If the ExecutionResult does not have a job_id.
            requests.exceptions.HTTPError: If the cancellation fails (e.g., 403 Forbidden,
                or 409 Conflict if job is not in a cancellable state).
        """
        job_id = self._extract_job_id(execution_result)
        return self._make_request(
            "post",
            f"job/{job_id}/cancel/",
            timeout=50,
        )

    def _get_execution_config(self, execution_result: ExecutionResult) -> dict:
        job_id = self._extract_job_id(execution_result)
        response = self._make_request(
            "get", f"job/{job_id}/execution_config/", timeout=50
        )
        return response.json()

    def get_maestro_config(
        self, execution_result: ExecutionResult
    ) -> MaestroConfig | None:
        """Retrieve a job's Maestro settings.

        Args:
            execution_result: An ExecutionResult instance whose ``job_id``
                identifies the target job.

        Returns:
            MaestroConfig | None: The job's Maestro settings, or ``None`` when
                it has none set.

        Raises:
            ValueError: If the ExecutionResult does not have a job_id.
            requests.exceptions.HTTPError:
                - 404: No such job.
        """
        data = self._get_execution_config(execution_result).get("maestro_config")
        return None if data is None else maestro_config_from_payload(data)

    def get_device_config(
        self, execution_result: ExecutionResult
    ) -> DeviceConfig | None:
        """Retrieve a job's hardware options.

        Args:
            execution_result: An ExecutionResult instance whose ``job_id``
                identifies the target job.

        Returns:
            DeviceConfig | None: The job's hardware options, or ``None`` when
                it has none set.

        Raises:
            ValueError: If the ExecutionResult does not have a job_id.
            requests.exceptions.HTTPError:
                - 404: No such job.
        """
        data = self._get_execution_config(execution_result).get("device_config")
        return None if data is None else DeviceConfig.from_payload(data)

    def get_job_results(self, execution_result: ExecutionResult) -> ExecutionResult:
        """
        Get the results of a job from the Qoro Database.

        Args:
            execution_result: An ExecutionResult instance with a job_id to fetch results for.

        Returns:
            ExecutionResult: A new ExecutionResult instance with results and the
                job's ``run_time`` populated. Counts are keyed with c[0]
                leftmost; a QPU job's counts arrive with c[0] rightmost and
                are reversed.

        Raises:
            ValueError: If the ExecutionResult does not have a job_id.
            requests.exceptions.HTTPError: If the job results are not available
                (e.g., job is still running) or if the request fails.
        """
        job_id = self._extract_job_id(execution_result)

        all_results: list[dict] = []
        histograms: list[dict] = []
        limit = 100
        offset = 0

        while True:
            try:
                response = self._make_request(
                    "get",
                    f"job/{job_id}/resultsV2/?limit={limit}&offset={offset}",
                    timeout=100,
                )
            except requests.exceptions.HTTPError as e:
                # Provide a more specific error message for 400 Bad Request
                if e.response.status_code == HTTPStatus.BAD_REQUEST:
                    raise requests.exceptions.HTTPError(
                        "400 Bad Request: Job results not available, likely job is still running"
                    ) from e
                # Re-raise any other HTTP error
                raise e

            data = response.json()

            for result in data["results"]:
                encoded = result["results"]
                result["results"] = _decode_histogram_b64(encoded)
                if encoded and encoded.get("payload"):
                    histograms.append(result)
            all_results.extend(data["results"])

            if data.get("next") is None:
                break
            offset += limit

        if histograms:
            meta = self._make_request("get", f"job/{job_id}/meta/", timeout=100)
            # QPUs key counts with c[0] rightmost.
            if meta.json()["qpu_system_name"] is not None:
                for result in histograms:
                    result["results"] = _reverse_bitstrings(result["results"])

        status = self._make_request("get", f"job/{job_id}/status/", timeout=200)
        return execution_result.with_results(
            all_results, run_time=float(status.json().get("run_time", 0))
        )

    def poll_job_status(
        self,
        execution_result: ExecutionResult,
        loop_until_complete: bool = False,
        verbose: bool = True,
        progress_callback: Callable[[int, str], None] | None = None,
        cancellation_event: Event | None = None,
    ) -> JobStatus:
        """Inspect a Qoro job once, or wait and raise on terminal failure.

        Waiting callers without a ``cancellation_event`` get a SIGINT funnel,
        and the remote job is cancelled if the wait ends for any reason except
        the job itself finishing. Wrappers that pass their own event retain
        cleanup ownership.

        Args:
            execution_result: An ExecutionResult instance with a job_id to check.
            loop_until_complete: Return after one request when ``False``. When
                ``True``, wait for ``COMPLETED`` or raise for terminal failure.
            verbose (bool, optional): If True, prints polling status to the logger.
            progress_callback (Callable, optional): A function for updating progress bars.
                Takes `(retry_count, status)`.
            cancellation_event (Event, optional): When provided, the polling loop
                waits on this Event between attempts instead of plain
                ``time.sleep`` so that ``Event.set()`` interrupts the next sleep
                window and raises :class:`~divi.exceptions.ExecutionCancelledError`.
                An in-flight HTTP request is **not** interrupted. If that request
                reports ``COMPLETED``, completion takes precedence; otherwise
                cancellation takes precedence as soon as the request returns.
                Worst-case cancellation latency is bounded by the request timeout.

        Returns:
            The current status for a one-shot poll, or ``COMPLETED`` when waiting.

        Raises:
            ValueError: If the ExecutionResult does not have a job_id.
            ExecutionCancelledError: If ``cancellation_event`` is set while
                polling or during a polling-interval wait.
            JobFailedError: If the job reaches ``FAILED``.
            JobCancelledError: If the scheduler reports ``CANCELLED``.
            InsufficientCreditsError: If the job lacks execution credits.
            JobTimedOutError: If the server-side execution deadline expires.
        """
        job_id = self._extract_job_id(execution_result)

        if not loop_until_complete:
            response = self._make_request("get", f"job/{job_id}/status/", timeout=200)
            return JobStatus(response.json()["status"])

        owns_event = cancellation_event is None
        event = Event() if owns_event else cancellation_event

        polling_status = None
        if progress_callback:
            update_fn = progress_callback
        elif verbose:
            polling_status = Console(file=None).status("", spinner="aesthetic")
            polling_status.start()

            def update_polling_status(retry_count, job_status):
                cap = "∞" if self.max_retries is None else self.max_retries
                status_msg = (
                    f"Job [cyan]{job_id.split('-')[0]}[/cyan] is {job_status}. "
                    f"Polling attempt {retry_count} / {cap}"
                )
                polling_status.update(status_msg)

            update_fn = update_polling_status
        else:

            def ignore_status(retry_count, job_status):
                pass

            update_fn = ignore_status

        try:
            with ExitStack() as stack:
                if owns_event:
                    stack.enter_context(_sigint_to_event(event))
                    stack.enter_context(_cancel_job_on_abort(self, execution_result))
                terminal_errors = {
                    JobStatus.FAILED: JobFailedError,
                    JobStatus.CANCELLED: JobCancelledError,
                    JobStatus.INSUFFICIENT_CREDITS: InsufficientCreditsError,
                    JobStatus.TIMED_OUT: JobTimedOutError,
                }

                attempts = (
                    itertools.count(1)
                    if self.max_retries is None
                    else range(1, self.max_retries + 1)
                )
                cancelled = f"Polling cancelled for job {job_id}."
                for retry_count in attempts:
                    raise_if_cancelled(event, cancelled)

                    response = self._make_request(
                        "get", f"job/{job_id}/status/", timeout=200
                    )
                    status = JobStatus(response.json()["status"])

                    if status is JobStatus.COMPLETED:
                        return status
                    raise_if_cancelled(event, cancelled)
                    if error_type := terminal_errors.get(status):
                        raise error_type(job_id)

                    update_fn(retry_count, status.value)

                    if event.wait(self.polling_interval):
                        raise ExecutionCancelledError(cancelled)

                raise MaxRetriesReachedError(job_id, self.max_retries)
        finally:
            if polling_status:
                polling_status.stop()

    # ------------------------------------------------------------------ #
    # QUBO / HUBO Characterisation
    # ------------------------------------------------------------------ #

    def characterize_and_validate(
        self,
        qubo: dict | None = None,
        *,
        reference_states: Sequence[str] | None = None,
        options: dict | None = None,
        job_id: str | None = None,
        tag: str = "divi-characterize",
    ) -> dict:
        """Submit a QUBO for characterisation, or fetch an existing result.

        Submit mode initialises the job and queues the analysis, which runs
        asynchronously on the server. Both modes then wait for the job to reach
        a terminal status before retrieving its stored result. Fetching an
        existing job does not charge credits.

        Args:
            qubo: Legacy comma-key QUBO/HUBO dict or a ``factored_v1``
                envelope. Required in submit mode.
            reference_states: One or more binary reference bitstrings. Required
                in submit mode by the characterisation engine.
            options: Optional server options: ``ansatz``, ``constraints``,
                ``n_qubits``, and the ``cost_qubo``/``penalty_qubo`` split.
            job_id: Existing characterisation job to wait for and fetch. When
                set, submit-mode arguments are ignored.
            tag: Job tag used during initialisation.

        Returns:
            The raw characterisation response. A failed or cancelled fetch
            returns a minimal ``{"job_id": ..., "status": ...}`` response so
            the high-level wrapper can raise the corresponding domain error.

        Raises:
            ValueError: If submit mode lacks a QUBO or reference state.
            ~divi.exceptions.CharacterizationSubmitError: If a request after
                job creation fails ambiguously, or polling exhausts
                ``max_retries``. The exception carries the existing ``job_id``
                and failed ``phase``.
            ~divi.backends.InsufficientCreditsError: If the account lacks the
                credits to run the job, at submission or while waiting.
            requests.exceptions.HTTPError: For definite client-side rejections
                and failures before a recoverable job ID exists.

        .. note::
            Initialisation and submission are never resent once delivered,
            because both mutate server state. Fetching by ``job_id`` is free.
            Waiting shows polling progress and is bounded by the service's
            ``max_retries`` (unbounded by default). Interrupting the wait does
            not cancel the job; fetch it later by ``job_id``.
        """
        if job_id is None:
            if qubo is None:
                raise ValueError(
                    "characterize_and_validate() requires either 'qubo' (to submit a "
                    "new job) or 'job_id' (to fetch an existing result)."
                )
            if reference_states is None or len(reference_states) == 0:
                raise ValueError(
                    "characterize_and_validate() requires at least one reference "
                    "state when submitting a new job."
                )

            init_resp = self._make_request(
                "post",
                "job/init/",
                json={"job_type": JobType.CHARACTERIZE.value, "tag": tag},
                timeout=100,
            )
            job_id = init_resp.json()["job_id"]
            if not isinstance(job_id, str):
                raise ValueError(
                    "Characterisation initialisation returned an invalid job_id."
                )
            logger.info(
                "Characterisation job %s created. If this call does not return, "
                "fetch the result with this id before resubmitting.",
                job_id,
            )

            submit_payload: dict = {
                "qubo": qubo,
                "reference_states": [str(state) for state in reference_states],
            }
            if options:
                submit_payload["options"] = options

            try:
                self._make_request(
                    "post",
                    f"job/{job_id}/submit_qubo/",
                    json=submit_payload,
                    timeout=100,
                )
            except requests.RequestException as exc:
                if (
                    exc.response is not None
                    and exc.response.status_code == HTTPStatus.PAYMENT_REQUIRED
                ):
                    raise InsufficientCreditsError(job_id) from exc
                if not _is_recoverable_characterization_error(exc):
                    raise
                raise CharacterizationSubmitError(
                    job_id, exc, phase="submission"
                ) from exc

        try:
            # Passing an event leaves the job running if the wait is interrupted.
            self.poll_job_status(
                ExecutionResult(job_id=job_id),
                loop_until_complete=True,
                cancellation_event=Event(),
            )
        except QoroJobError as exc:
            return {"job_id": job_id, "status": exc.status}
        except MaxRetriesReachedError as exc:
            raise CharacterizationSubmitError(
                job_id, exc, phase="status polling"
            ) from exc
        except requests.RequestException as exc:
            if not _is_recoverable_characterization_error(exc):
                raise
            raise CharacterizationSubmitError(
                job_id, exc, phase="status polling"
            ) from exc

        try:
            result_resp = self._make_request(
                "get",
                f"job/{job_id}/validation_result/",
                timeout=100,
            )
        except requests.RequestException as exc:
            if not _is_recoverable_characterization_error(exc):
                raise
            raise CharacterizationSubmitError(
                job_id, exc, phase="result retrieval"
            ) from exc

        data = result_resp.json()
        data.setdefault("job_id", job_id)
        return data

    def _fetch_characterization_html(self, job_id: str) -> str:
        """Fetch the server-rendered HTML report for a characterisation job.

        Used by :meth:`CharacterizationResult._repr_html_` to lazily render
        the result in Jupyter. The endpoint returns a self-contained HTML
        fragment (inline CSS, no external assets).
        """
        resp = self._make_request(
            "get",
            f"job/{job_id}/validation_result/html/",
            timeout=100,
        )
        return resp.text
