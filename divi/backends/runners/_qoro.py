# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import gzip
import itertools
import json
import logging
import os
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import replace
from enum import Enum
from http import HTTPStatus
from threading import Event
from typing import Any, TypeVar

import requests
from dotenv import dotenv_values, find_dotenv
from qiskit import QuantumCircuit
from requests.adapters import HTTPAdapter, Retry
from rich.console import Console

from divi.circuits._payloads import (
    CircuitBatch,
    CircuitPayload,
    as_payloads,
    bound_circuits,
    is_bound,
)
from divi.exceptions import CharacterizationSubmitError, ExecutionCancelledError
from divi.qasm import (
    _format_validation_error_with_context,
    is_valid_qasm,
    validate_qasm,
)

from .._base import CircuitRunner, ExecutionResult
from .._cancellation import _auto_cancellation_scope
from .._config import ExecutionConfig, JobConfig
from .._pauli_serde import compress_ham_ops
from .._results_processing import _decode_histogram_b64
from .._shot_allocation import (
    from_wire,
    restrict_to_chunk,
    to_wire,
    validate,
)
from .._systems import (
    QPUSystem,
    SimulatorCluster,
    get_qpu_system,
    get_simulator_cluster,
    parse_qpu_systems,
    parse_simulator_clusters,
    update_qpu_systems_cache,
    update_simulator_clusters_cache,
)

API_URL = "https://app.qoroquantum.net/api"
_MAX_PAYLOAD_SIZE_MB = 0.95

T = TypeVar("T")

session = requests.Session()
retry_configuration = Retry(
    total=5,
    backoff_factor=0.1,
    status_forcelist=[502],
    allowed_methods=["GET", "POST", "DELETE"],
)

session.mount("http://", HTTPAdapter(max_retries=retry_configuration))
session.mount("https://", HTTPAdapter(max_retries=retry_configuration))

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


def _greedy_size_chunks(
    items: Iterable[T],
    item_size: Callable[[T], int],
    base_overhead: int,
    max_bytes: int,
) -> list[list[T]]:
    """Greedily pack *items* into chunks whose estimated size stays under the cap.

    A chunk is flushed before the item that would overflow it, so a single item
    larger than ``max_bytes`` still lands in a chunk of its own rather than
    being dropped — the caller is responsible for rejecting that case up front
    if the server cannot accept it.
    """
    chunks: list[list[T]] = []
    current: list[T] = []
    current_size = base_overhead

    for item in items:
        size = item_size(item)
        if current and current_size + size > max_bytes:
            chunks.append(current)
            current = []
            current_size = base_overhead
        current.append(item)
        current_size += size

    if current:
        chunks.append(current)

    return chunks


class JobStatus(Enum):
    """Status of a job on the Qoro Service."""

    PENDING = "PENDING"
    """Job is queued and waiting to be processed."""

    RUNNING = "RUNNING"
    """Job is currently being executed."""

    COMPLETED = "COMPLETED"
    """Job has finished successfully."""

    FAILED = "FAILED"
    """Job execution encountered an error."""

    CANCELLED = "CANCELLED"
    """Job was cancelled before completion."""


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
_DEFAULT_SHOTS = 1000

_DEFAULT_JOB_CONFIG = JobConfig(
    shots=_DEFAULT_SHOTS,
    simulator_cluster=_DEFAULT_SIMULATOR_CLUSTER,
    use_circuit_packing=False,
)


class QoroService(CircuitRunner):
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
        execution_config: ExecutionConfig | None = None,
        polling_interval: float = 3.0,
        max_retries: int | None = None,
        track_depth: bool = False,
    ):
        """Initialises the QoroService client.

        Args:
            auth_token (str | None, optional):
                The authentication token for the Qoro API. If not provided,
                it will be read from ``QORO_API_KEY`` in a ``.env`` file,
                falling back to the ``QORO_API_KEY`` environment variable.
            job_config (JobConfig | None, optional):
                A JobConfig object containing default job settings. If not
                provided, a default configuration will be created. If the
                job_config has neither ``simulator_cluster`` nor ``qpu_system``,
                it defaults to the ``qoro_maestro`` simulator cluster with a
                warning.
            execution_config (ExecutionConfig | None, optional):
                Default execution configuration for submitted jobs. When
                provided, every call to :meth:`submit_circuits` will use
                this config unless an explicit ``execution_config`` argument
                overrides it.
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
        self.polling_interval = polling_interval
        self.max_retries = max_retries

        # Fetch available systems (needs auth_token to be set)
        self.fetch_qpu_systems()
        self.fetch_simulator_clusters()

        # Set up job config
        if job_config is None:
            job_config = _DEFAULT_JOB_CONFIG

        self.job_config = job_config

        self.execution_config = execution_config

        shots = (
            self.job_config.shots
            if self.job_config.shots is not None
            else _DEFAULT_SHOTS
        )
        super().__init__(shots=shots, track_depth=track_depth)

    @property
    def supports_expval(self) -> bool:
        """
        Whether the backend supports expectation value measurements.
        """
        target = self.job_config.simulator_cluster or self.job_config.qpu_system
        if not isinstance(target, (SimulatorCluster, QPUSystem)):
            raise RuntimeError(
                "JobConfig target is unresolved; this should have been resolved "
                "by _resolve_and_validate_target before reaching here."
            )
        return target.supports_expval and not self.job_config.force_sampling

    @property
    def job_config(self) -> JobConfig:
        """The service's default job configuration."""
        return self._job_config

    @job_config.setter
    def job_config(self, value: JobConfig) -> None:
        self._job_config = self._resolve_and_validate_target(value)

    @property
    def execution_config(self) -> ExecutionConfig | None:
        """The service's default execution configuration."""
        return self._execution_config

    @execution_config.setter
    def execution_config(self, value: ExecutionConfig | None) -> None:
        self._execution_config = value

    @property
    def is_async(self) -> bool:
        """
        Whether the backend executes circuits asynchronously.
        """
        return True

    @property
    def resolves_parameters(self) -> bool:
        """The Qoro backend substitutes parameter values server-side."""
        return True

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
            return replace(config, simulator_cluster=_DEFAULT_SIMULATOR_CLUSTER)

        if isinstance(config.simulator_cluster, str):
            resolved = get_simulator_cluster(config.simulator_cluster)
            return replace(config, simulator_cluster=resolved)

        if isinstance(config.qpu_system, str):
            resolved = get_qpu_system(config.qpu_system)
            return replace(config, qpu_system=resolved)

        return config

    def _make_request(
        self,
        method: str,
        endpoint: str,
        *,
        retry: bool = True,
        **kwargs,
    ) -> requests.Response:
        """
        Make an authenticated HTTP request to the Qoro API.

        This internal method centralises all API communication, handling authentication
        headers and error responses consistently.

        Args:
            method (str): HTTP method to use (e.g., 'get', 'post', 'delete').
            endpoint (str): API endpoint path (without base URL).
            retry (bool): When ``True`` (the default), the request goes
                through the session that has the retry adapter mounted.
                Set to ``False`` for state-mutating endpoints where a
                retry would target a job already past its initial state.
            **kwargs: Additional arguments to pass to requests.request(), such as
                'json', 'timeout', 'params', etc.

        Returns:
            requests.Response: The HTTP response object from the API.

        Raises:
            requests.exceptions.HTTPError: If the response status code is 400 or above.
        """
        url = f"{API_URL}/{endpoint}"

        headers = {"Authorization": self.auth_token}

        if method.upper() in ["POST", "PUT", "PATCH"]:
            headers["Content-Type"] = "application/json"

        # Allow overriding default headers
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))

        requester = session.request if retry else requests.request
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

    @staticmethod
    def _compress_data(value: str) -> str:
        return base64.b64encode(gzip.compress(value.encode("utf-8"))).decode("utf-8")

    def _split_circuits(self, circuits: Mapping[str, str]) -> list[dict[str, str]]:
        """
        Splits circuits into chunks by estimating payload size with a simplified,
        consistent overhead calculation.
        Assumes that BASE64 encoding produces ASCI characters, which are 1 byte each.
        """
        compressed = [
            (key, self._compress_data(value)) for key, value in circuits.items()
        ]
        chunks = _greedy_size_chunks(
            compressed,
            # 6 bytes of JSON punctuation per entry: two quote pairs, a colon
            # and a comma.
            lambda item: len(item[0]) + len(item[1]) + 6,
            # The opening and closing curly braces.
            base_overhead=2,
            max_bytes=int(_MAX_PAYLOAD_SIZE_MB * 1024 * 1024),
        )
        return [dict(chunk) for chunk in chunks]

    def _resolve_job_config(self, override: JobConfig | None) -> JobConfig:
        """Layer service defaults under an optional per-call override."""
        if not override:
            return self.job_config
        return self._resolve_and_validate_target(self.job_config.override(override))

    def _resolve_execution_config(
        self, override: ExecutionConfig | None
    ) -> ExecutionConfig | None:
        """Layer service defaults under an optional per-call override."""
        if override is None:
            return self.execution_config
        if self.execution_config is None:
            return override
        return self.execution_config.override(override)

    @staticmethod
    def _validate_ham_group(group: str) -> None:
        """Check one ``;``-delimited observable group is well formed."""
        valid_paulis = {"I", "X", "Y", "Z"}
        terms = group.split(";")
        if not all(terms):
            raise ValueError(
                "Hamiltonian operators must be non-empty semicolon-separated strings."
            )
        ham_ops_length = len(terms[0])
        if not all(len(term) == ham_ops_length for term in terms):
            raise ValueError("All Hamiltonian operators must have the same length.")
        if not all(all(c in valid_paulis for c in term) for term in terms):
            raise ValueError(
                "Hamiltonian operators must contain only I, X, Y, Z characters."
            )

    @staticmethod
    def _job_type_for(job_type: JobType | None, ham_ops: str | None) -> JobType:
        """Resolve the job type, inferring EXPECTATION from ``ham_ops``."""
        if ham_ops is None:
            return job_type if job_type is not None else JobType.EXECUTE
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
        circuit_ham_map: list[list[int]] | None = None,
        shot_groups: list[list[int]] | None = None,
        job_type: JobType | None = None,
        override_execution_config: ExecutionConfig | None = None,
        override_job_config: JobConfig | None = None,
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

        Running an ensemble with ``batch_submissions=True`` merges circuits
        across programs and always sends them bound, so those runs take the
        mapping shape regardless of the parameters they started with.

        The two shapes resolve floats differently: the backend substitutes
        via ``str(value)`` on the decoded float, preserving the full
        Python repr, whereas bound QASM was rendered locally at the
        :class:`~divi.circuits.MetaCircuit`'s precision (8 decimals by
        default). Values that round-trip through both formatters give
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
                Multiple groups can be pipe-delimited (e.g. "XYZ;XXZ|ZI;IZ") when
                ``circuit_ham_map`` is provided to assign each group to a slice of
                circuits. If None, no Hamiltonian operators will be measured.
            circuit_ham_map (list[list[int]] | None, optional):
                Maps each ``|``-delimited group in ``ham_ops`` to a ``[start, end)``
                slice of the ordered circuit list.  Must have the same length as
                ``ham_ops.split("|")``.  When None, a single ``ham_ops`` group is
                applied to all circuits. Bound payloads only — the parametric
                ordering does not align with the flat index ranges it references.
            shot_groups (list[list[int]] | None, optional):
                Per-circuit shot allocation as ``[start, end, shots]`` triples
                covering the iteration order of the circuits. Mutually exclusive
                with the service-level ``shots`` field. When provided, ranges
                spanning multiple internal chunks are re-indexed automatically.
                Bound payloads only, for the same reason as ``circuit_ham_map``.
            job_type (JobType | None, optional):
                Type of job to execute (EXECUTE or EXPECTATION).
                If not provided, defaults to EXECUTE.
            override_execution_config (ExecutionConfig | None, optional):
                Execution configuration override for this submission. When
                provided, its non-None fields override the service-level
                ``execution_config`` set in the constructor. When omitted, the
                service-level default is used (if any). The merged config is
                sent inline to ``job/init`` as ``execution_configuration``.
            override_job_config (JobConfig | None, optional):
                Configuration object to override the service's default settings.
                If not provided, default values are used.
            cancellation_event (Event | None, optional):
                Accepted for :class:`~divi.backends.CircuitRunner` interface
                parity; submission itself is unaffected. Pass the same Event
                to :meth:`poll_job_status` to interrupt the polling loop.
            **kwargs:
                Accepted to match the ``CircuitRunner.submit_circuits``
                signature but not used by this backend. Any extra keyword
                arguments are ignored.

        Raises:
            ValueError: If any circuit is not valid QASM.
            requests.exceptions.HTTPError: If any API request fails.

        Returns:
            ExecutionResult: Contains job_id for asynchronous execution. Use the job_id
                to poll for results using backend.poll_job_status() and get_job_results().
        """
        payloads = as_payloads(payloads)
        if not payloads:
            raise ValueError("submit_circuits requires at least one payload.")

        if ham_ops is not None and shot_groups is not None:
            raise ValueError(
                "shot_groups is incompatible with ham_ops: EXPECTATION jobs "
                "compute expectation values analytically on the backend and "
                "ignore shot counts. Pass exactly one."
            )

        if ham_ops is not None:
            # Each |-delimited group is validated independently.
            ham_groups = ham_ops.split("|")
            for group in ham_groups:
                self._validate_ham_group(group)

            if circuit_ham_map is not None and len(circuit_ham_map) != len(ham_groups):
                raise ValueError(
                    f"circuit_ham_map length ({len(circuit_ham_map)}) must match "
                    f"number of ham_ops groups ({len(ham_groups)})."
                )

        job_config = self._resolve_job_config(override_job_config)
        call_plan = (
            self._bound_call_plan(payloads, shot_groups)
            if is_bound(payloads)
            else self._parametric_call_plan(
                payloads, ham_ops, circuit_ham_map, shot_groups
            )
        )

        return self._dispatch_job(
            call_plan,
            job_config=job_config,
            execution_config=self._resolve_execution_config(override_execution_config),
            job_type=self._job_type_for(job_type, ham_ops),
            ham_ops=ham_ops,
            circuit_ham_map=circuit_ham_map,
        )

    def _bound_call_plan(
        self, payloads: Sequence[CircuitPayload], shot_groups: list[list[int]] | None
    ) -> list[dict[str, Any]]:
        """Chunk resolved circuits into ``add_circuits/`` payload fragments."""
        circuits = bound_circuits(payloads)

        shot_ranges = None
        if shot_groups is not None:
            shot_ranges = from_wire(shot_groups)
            validate(shot_ranges, len(circuits))

        for key, circuit in circuits.items():
            if not is_valid_qasm(circuit):
                try:
                    validate_qasm(circuit)
                except SyntaxError as e:
                    msg = _format_validation_error_with_context(circuit, e)
                    raise ValueError(
                        f"Circuit '{key}' is not a valid QASM: {msg}"
                    ) from e

        if self.track_depth:
            self._depth_history.append(
                [
                    QuantumCircuit.from_qasm_str(qasm).depth()
                    for qasm in circuits.values()
                ]
            )

        chunks = self._split_circuits(circuits)
        # Per-chunk starting offset into the global circuit list, used to
        # re-index ``shot_groups`` when chunking.
        offsets = itertools.accumulate((len(c) for c in chunks), initial=0)

        call_plan = []
        for chunk, offset in zip(chunks, offsets):
            fragment: dict[str, Any] = {"circuits": chunk}
            if shot_ranges is not None:
                fragment["shot_groups"] = to_wire(
                    restrict_to_chunk(shot_ranges, offset, len(chunk))
                )
            call_plan.append(fragment)
        return call_plan

    @staticmethod
    def _split_payload_parameter_sets(
        compressed_template_b64: str,
        parameter_names: tuple[str, ...],
        parameter_sets: tuple[tuple[str, tuple[float, ...]], ...],
    ) -> list[list[tuple[str, tuple[float, ...]]]]:
        """Split a ``CircuitPayload``'s ``parameter_sets`` into chunks bounded
        by :data:`_MAX_PAYLOAD_SIZE_MB`.

        Each chunk re-uses the same already-compressed ``circuit_template``
        and ``parameter_names``; only ``parameter_sets`` is split.  The size
        estimate is an intentionally loose upper bound on the JSON body's
        character count — the JSON encoder may add whitespace and per-value
        floats round up rather than down, so a small cushion keeps us under
        the cap.  This mirrors :meth:`_split_circuits` for the bound path.
        """
        max_payload_bytes = int(_MAX_PAYLOAD_SIZE_MB * 1024 * 1024)
        # Fixed per-call overhead: compressed template + parameter_names JSON
        # + JSON structural keys (circuit_template, parameter_names,
        # parameter_sets, mode, finalized, shots/observables).
        parameter_names_bytes = (
            sum(len(n) for n in parameter_names) + 4 * len(parameter_names) + 2
        )
        # 512-byte cushion absorbs key names, observables blob, and whitespace.
        fixed_overhead = len(compressed_template_b64) + parameter_names_bytes + 512

        if fixed_overhead >= max_payload_bytes:
            raise ValueError(
                "Compressed circuit_template "
                f"({len(compressed_template_b64)} bytes) alone exceeds the "
                f"per-request payload cap ({_MAX_PAYLOAD_SIZE_MB} MB); "
                "reduce the template size or split the program."
            )

        def row_size(row: tuple[str, tuple[float, ...]]) -> int:
            # One row's JSON: {"label": "<label>", "values": [v0, v1, ...]},
            # ≈ 26 + len(label) + per-value chars + trailing ", ".
            label, values = row
            values_size = sum(len(repr(float(v))) + 2 for v in values)
            return 26 + len(label) + values_size + 2

        return _greedy_size_chunks(
            parameter_sets, row_size, fixed_overhead, max_payload_bytes
        )

    def _parametric_call_plan(
        self,
        payloads: Sequence[CircuitPayload],
        ham_ops: str | None,
        circuit_ham_map: list[list[int]] | None,
        shot_groups: list[list[int]] | None,
    ) -> list[dict[str, Any]]:
        """Chunk parameter matrices into ``add_circuits/`` payload fragments.

        Each fragment reuses the same compressed ``circuit_template``; only
        ``parameter_sets`` is split, mirroring how :meth:`_bound_call_plan`
        splits the circuit mapping.
        """
        if circuit_ham_map is not None:
            raise ValueError(
                "circuit_ham_map is not supported for parametric payloads because "
                "their ordering does not align with the bound-circuit flat "
                "index ranges it references. Bind the parameters first if your "
                "batch needs |-delimited ham_ops with circuit_ham_map."
            )
        if shot_groups is not None:
            raise ValueError(
                "shot_groups is not supported for parametric payloads for the "
                "same reason as circuit_ham_map; per-circuit shot allocation "
                "would need re-indexing into the payload order."
            )
        if ham_ops is not None and "|" in ham_ops:
            raise ValueError(
                "|-delimited ham_ops groups require circuit_ham_map; not "
                "supported for parametric payloads."
            )

        call_plan: list[dict[str, Any]] = []
        for payload in payloads:
            if not isinstance(payload.circuit, str):
                raise TypeError(
                    "QoroService resolves parameters server-side and needs "
                    f"QASM-encoded payloads; got {type(payload.circuit).__name__}."
                )
            # One check per template, not per resolved row: the rows differ
            # only in the substituted values.
            if not is_valid_qasm(payload.circuit, payload.parameter_names):
                try:
                    validate_qasm(payload.circuit, payload.parameter_names)
                except SyntaxError as e:
                    msg = _format_validation_error_with_context(payload.circuit, e)
                    raise ValueError(
                        f"Circuit template is not valid QASM: {msg}"
                    ) from e

            compressed = self._compress_data(payload.circuit)
            for chunk in self._split_payload_parameter_sets(
                compressed, payload.parameter_names, payload.parameter_sets
            ):
                call_plan.append(
                    {
                        "circuit_template": compressed,
                        "parameter_names": list(payload.parameter_names),
                        "parameter_sets": [
                            {"label": label, "values": list(values)}
                            for label, values in chunk
                        ],
                    }
                )
        return call_plan

    def _dispatch_job(
        self,
        call_plan: list[dict[str, Any]],
        *,
        job_config: JobConfig,
        execution_config: ExecutionConfig | None,
        job_type: JobType,
        ham_ops: str | None,
        circuit_ham_map: list[list[int]] | None,
    ) -> ExecutionResult:
        """Open a job, upload every payload fragment, return its ``job_id``.

        The plan is complete before ``job/init/`` runs, so the last fragment
        is known up front and is the only one marked ``finalized``.
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
        if execution_config is not None:
            init_payload["execution_configuration"] = execution_config.to_payload()

        init_response = self._make_request(
            "post", "job/init/", json=init_payload, timeout=100
        )
        if init_response.status_code not in [HTTPStatus.OK, HTTPStatus.CREATED]:
            _raise_with_details(init_response)
        job_id = init_response.json()["job_id"]

        compressed_ham_ops = compress_ham_ops(ham_ops) if ham_ops is not None else None

        for i, fragment in enumerate(call_plan):
            payload: dict[str, Any] = {
                **fragment,
                "mode": "append",
                "finalized": "true" if i == len(call_plan) - 1 else "false",
            }
            if compressed_ham_ops is not None:
                payload["observables"] = compressed_ham_ops
                if circuit_ham_map is not None:
                    payload["circuit_ham_map"] = circuit_ham_map
            elif "shot_groups" not in payload:
                payload["shots"] = job_config.shots

            response = self._make_request(
                "post", f"job/{job_id}/add_circuits/", json=payload, timeout=100
            )
            if response.status_code != HTTPStatus.OK:
                _raise_with_details(response)

        return ExecutionResult(results=None, job_id=job_id)

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

    def set_execution_config(
        self,
        execution_result: ExecutionResult,
        config: ExecutionConfig,
    ) -> dict:
        """Set or overwrite the execution configuration for a job.

        The job must be in ``PENDING`` status. Re-calling this method
        overwrites any previously set configuration.

        Args:
            execution_result: An ExecutionResult instance whose ``job_id``
                identifies the target job.
            config: The execution configuration to attach.

        Returns:
            dict: The API response containing ``status``, ``job_id`` and
                ``execution_configuration``.

        Raises:
            ValueError: If the ExecutionResult does not have a job_id.
            requests.exceptions.HTTPError:
                - 400: Validation errors (unknown ``api_meta`` keys, wrong
                  types, payload too large).
                - 403: ``bond_dimension`` exceeds the user's tier cap.
                - 409: Job is not in ``PENDING`` status.
        """
        job_id = self._extract_job_id(execution_result)
        response = self._make_request(
            "post",
            f"job/{job_id}/execution_config/",
            json=config.to_payload(),
            timeout=50,
        )
        return response.json()

    def get_execution_config(
        self,
        execution_result: ExecutionResult,
    ) -> ExecutionConfig:
        """Retrieve the execution configuration for an existing job.

        Args:
            execution_result: An ExecutionResult instance whose ``job_id``
                identifies the target job.

        Returns:
            ExecutionConfig: The execution configuration attached to the job.

        Raises:
            ValueError: If the ExecutionResult does not have a job_id.
            requests.exceptions.HTTPError:
                - 404: No execution configuration exists for this job.
        """
        job_id = self._extract_job_id(execution_result)
        response = self._make_request(
            "get",
            f"job/{job_id}/execution_config/",
            timeout=50,
        )
        data = response.json()
        return ExecutionConfig.from_response(data["execution_configuration"])

    def get_job_results(self, execution_result: ExecutionResult) -> ExecutionResult:
        """
        Get the results of a job from the Qoro Database.

        Args:
            execution_result: An ExecutionResult instance with a job_id to fetch results for.

        Returns:
            ExecutionResult: A new ExecutionResult instance with results populated.

        Raises:
            ValueError: If the ExecutionResult does not have a job_id.
            requests.exceptions.HTTPError: If the job results are not available
                (e.g., job is still running) or if the request fails.
        """
        job_id = self._extract_job_id(execution_result)

        all_results: list[dict] = []
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
                result["results"] = _decode_histogram_b64(result["results"])
            all_results.extend(data["results"])

            if data.get("next") is None:
                break
            offset += limit

        # Return a new ExecutionResult with results populated
        return execution_result.with_results(all_results)

    def poll_job_status(
        self,
        execution_result: ExecutionResult,
        loop_until_complete: bool = False,
        on_complete: Callable[[dict], None] | None = None,
        verbose: bool = True,
        progress_callback: Callable[[int, str], None] | None = None,
        cancellation_event: Event | None = None,
    ) -> JobStatus:
        """
        Get the status of a job and optionally execute a function on completion.

        When ``loop_until_complete`` is ``True`` and the caller does not supply a
        ``cancellation_event``, the service installs a SIGINT funnel for the
        duration of the wait and best-effort cancels the remote job on Ctrl+C —
        so direct callers (e.g. ``service.poll_job_status(..., loop_until_complete=True)``
        in a script) get the same clean cancellation UX as pipeline-driven callers.
        Wrappers that pass their own event (the pipeline) opt out and retain
        cleanup ownership.

        Args:
            execution_result: An ExecutionResult instance with a job_id to check.
            loop_until_complete (bool): If True, polls until the job is complete or failed.
            on_complete (Callable, optional): A function called with the decoded
                final status payload when the job reaches a terminal state.
                Consumers read ``run_time`` from it to accumulate
                :attr:`~divi.qprog.QuantumProgram.total_run_time`.
            verbose (bool, optional): If True, prints polling status to the logger.
            progress_callback (Callable, optional): A function for updating progress bars.
                Takes `(retry_count, status)`.
            cancellation_event (Event, optional): When provided, the polling loop
                waits on this Event between attempts instead of plain
                ``time.sleep`` so that ``Event.set()`` interrupts the next sleep
                window and raises :class:`~divi.exceptions.ExecutionCancelledError`.  An
                in-flight HTTP request is **not** interrupted — worst-case
                cancellation latency is bounded by the per-request ``timeout``
                rather than the polling interval.

        Returns:
            JobStatus: The current job status.

        Raises:
            ValueError: If the ExecutionResult does not have a job_id.
            ExecutionCancelledError: If ``cancellation_event`` was set during
                a polling-interval wait.
        """
        job_id = self._extract_job_id(execution_result)

        # Take ownership of cancellation lifecycle for direct callers
        # (loop wait, no caller event); otherwise pass the caller's event
        # through unchanged via a no-op context.
        scope = (
            _auto_cancellation_scope(self, execution_result)
            if loop_until_complete and cancellation_event is None
            else nullcontext(cancellation_event)
        )

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
            update_fn = lambda _, __: None

        try:
            with scope as cancellation_event:
                if not loop_until_complete:
                    response = self._make_request(
                        "get", f"job/{job_id}/status/", timeout=200
                    )
                    return JobStatus(response.json()["status"])

                terminal_statuses = {
                    JobStatus.COMPLETED,
                    JobStatus.FAILED,
                    JobStatus.CANCELLED,
                }

                attempts = (
                    itertools.count(1)
                    if self.max_retries is None
                    else range(1, self.max_retries + 1)
                )
                for retry_count in attempts:
                    if cancellation_event is not None and cancellation_event.is_set():
                        raise ExecutionCancelledError(
                            f"Polling cancelled for job {job_id}."
                        )

                    response = self._make_request(
                        "get", f"job/{job_id}/status/", timeout=200
                    )
                    payload = response.json()
                    status = JobStatus(payload["status"])

                    if status in terminal_statuses:
                        if on_complete:
                            on_complete(payload)
                        return status

                    update_fn(retry_count, status.value)

                    if cancellation_event is not None:
                        if cancellation_event.wait(self.polling_interval):
                            raise ExecutionCancelledError(
                                f"Polling cancelled for job {job_id}."
                            )
                    else:
                        time.sleep(self.polling_interval)

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
        reference_states: list[str] | None = None,
        options: dict | None = None,
        job_id: str | None = None,
        tag: str = "divi-characterize",
    ) -> dict:
        """Submit a QUBO for characterisation, or fetch an existing result.

        Submit mode performs init, synchronous analysis submission, and result
        retrieval. Fetch mode waits for an existing job to reach a terminal
        status before retrieving its stored result, without charging credits.

        Args:
            qubo: Legacy comma-key QUBO/HUBO dict or a ``factored_v1``
                envelope. Required in submit mode.
            reference_states: One or more binary reference bitstrings. Required
                in submit mode by the Composer characterisation engine.
            options: Optional server options, including ``preset``, ``analysis``,
                ``ansatz``, ``subspace``, ``constraints``, and ``n_qubits``.
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
                job creation fails ambiguously. The exception carries the
                existing ``job_id`` and failed ``phase``.
            requests.exceptions.HTTPError: For definite client-side rejections
                and failures before a recoverable job ID exists.

        .. note::
            Initialisation and submission deliberately disable HTTP retries
            because both mutate server state. Fetching by ``job_id`` is free.
        """
        if job_id is not None:
            try:
                status = self.poll_job_status(
                    ExecutionResult(job_id=job_id),
                    loop_until_complete=True,
                    verbose=False,
                )
            except requests.RequestException as exc:
                if not _is_recoverable_characterization_error(exc):
                    raise
                raise CharacterizationSubmitError(
                    job_id, exc, phase="status polling"
                ) from exc

            if status != JobStatus.COMPLETED:
                return {"job_id": job_id, "status": status.value}
        else:
            if qubo is None:
                raise ValueError(
                    "characterize_and_validate() requires either 'qubo' (to submit a "
                    "new job) or 'job_id' (to fetch an existing result)."
                )
            if not reference_states:
                raise ValueError(
                    "characterize_and_validate() requires at least one reference "
                    "state when submitting a new job."
                )

            init_resp = self._make_request(
                "post",
                "job/init/",
                retry=False,
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
                "fetch the result with this id rather than resubmitting.",
                job_id,
            )

            submit_payload: dict = {
                "qubo": qubo,
                "reference_states": reference_states,
            }
            if options:
                submit_payload["options"] = options

            try:
                self._make_request(
                    "post",
                    f"job/{job_id}/submit_qubo/",
                    retry=False,
                    json=submit_payload,
                    timeout=300,
                )
            except requests.RequestException as exc:
                if not _is_recoverable_characterization_error(exc):
                    raise
                raise CharacterizationSubmitError(
                    job_id, exc, phase="submission"
                ) from exc

        assert job_id is not None
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
