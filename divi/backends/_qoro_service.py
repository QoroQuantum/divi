# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import base64
import gzip
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, fields
from enum import Enum
from http import HTTPStatus

import requests
from dotenv import dotenv_values
from requests.adapters import HTTPAdapter, Retry

from divi.backends import CircuitRunner
from divi.backends._qpu_system import QPU, QPUSystem
from divi.extern.cirq import is_valid_qasm

API_URL = "https://app.qoroquantum.net/api"
_MAX_PAYLOAD_SIZE_MB = 0.95

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


def _decode_qh1_b64(encoded: dict) -> dict[str, int]:
    """
    Decode a {'encoding':'qh1','n_bits':N,'payload':base64} histogram
    into a dict with bitstring keys -> int counts.

    Returns {} if payload is empty.
    """
    if not encoded or not encoded.get("payload"):
        return {}

    if encoded.get("encoding") != "qh1":
        raise ValueError(f"Unsupported encoding: {encoded.get('encoding')}")

    blob = base64.b64decode(encoded["payload"])
    hist_int = _decompress_histogram(blob)
    return {str(k): v for k, v in hist_int.items()}


def _uleb128_decode(data: bytes, pos: int = 0) -> tuple[int, int]:
    x = 0
    shift = 0
    while True:
        if pos >= len(data):
            raise ValueError("truncated varint")
        b = data[pos]
        pos += 1
        x |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            break
        shift += 7
    return x, pos


def _int_to_bitstr(x: int, n_bits: int) -> str:
    return format(x, f"0{n_bits}b")


def _rle_bool_decode(data: bytes, pos=0) -> tuple[list[bool], int]:
    num_runs, pos = _uleb128_decode(data, pos)
    if num_runs == 0:
        return [], pos
    first_val = data[pos] != 0
    pos += 1
    total, val = [], first_val
    for _ in range(num_runs):
        ln, pos = _uleb128_decode(data, pos)
        total.extend([val] * ln)
        val = not val
    return total, pos


def _decompress_histogram(buf: bytes) -> dict[str, int]:
    if not buf:
        return {}
    pos = 0
    if buf[pos : pos + 3] != b"QH1":
        raise ValueError("bad magic")
    pos += 3
    n_bits = buf[pos]
    pos += 1
    unique, pos = _uleb128_decode(buf, pos)
    total_shots, pos = _uleb128_decode(buf, pos)

    num_gaps, pos = _uleb128_decode(buf, pos)
    gaps = []
    for _ in range(num_gaps):
        g, pos = _uleb128_decode(buf, pos)
        gaps.append(g)

    idxs, acc = [], 0
    for i, g in enumerate(gaps):
        acc = g if i == 0 else acc + g
        idxs.append(acc)

    rb_len, pos = _uleb128_decode(buf, pos)
    is_one, _ = _rle_bool_decode(buf[pos : pos + rb_len], 0)
    pos += rb_len

    extras_len, pos = _uleb128_decode(buf, pos)
    extras = []
    for _ in range(extras_len):
        e, pos = _uleb128_decode(buf, pos)
        extras.append(e)

    counts, it = [], iter(extras)
    for flag in is_one:
        counts.append(1 if flag else next(it) + 2)

    hist = {_int_to_bitstr(i, n_bits): c for i, c in zip(idxs, counts)}

    # optional integrity check
    if sum(counts) != total_shots:
        raise ValueError("corrupt stream: shot sum mismatch")
    if len(counts) != unique:
        raise ValueError("corrupt stream: unique mismatch")
    return hist


def _raise_with_details(resp: requests.Response):
    try:
        data = resp.json()
        body = json.dumps(data, ensure_ascii=False)
    except ValueError:
        body = resp.text
    msg = f"{resp.status_code} {resp.reason}: {body}"
    raise requests.HTTPError(msg)


class JobStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class JobType(Enum):
    EXECUTE = "EXECUTE"
    SIMULATE = "SIMULATE"
    EXPECTATION = "EXPECTATION"
    CIRCUIT_CUT = "CIRCUIT_CUT"


@dataclass(frozen=True)
class JobConfig:
    """Configuration for a Qoro Service job.

    Attributes:
        shots: Number of shots for the job.
        qpu_system_name: Name of the QPU system to use.
        use_circuit_packing: Whether to use circuit packing optimization.
        tag: Tag to associate with the job for identification.
        job_type: Type of job to execute.
    """

    shots: int | None = None
    qpu_system_name: str | QPUSystem | None = None
    use_circuit_packing: bool | None = None
    tag: str = "default"

    def override(self, other: "JobConfig") -> "JobConfig":
        """Creates a new config by overriding attributes with non-None values.

        This method ensures immutability by always returning a new `JobConfig` object
        and leaving the original instance unmodified.

        Args:
            other: Another JobConfig instance to take values from. Only non-None
                   attributes from this instance will be used for the override.

        Returns:
            A new JobConfig instance with the merged configurations.
        """
        current_attrs = {f.name: getattr(self, f.name) for f in fields(self)}

        for f in fields(other):
            other_value = getattr(other, f.name)
            if other_value is not None:
                current_attrs[f.name] = other_value

        return JobConfig(**current_attrs)

    def __post_init__(self):
        """Sanitizes and validates the configuration."""
        if self.shots is not None and self.shots <= 0:
            raise ValueError(f"Shots must be a positive integer. Got {self.shots}.")

        if self.qpu_system_name is not None:
            if isinstance(self.qpu_system_name, QPUSystem):
                # Use object.__setattr__ to modify the attribute on a frozen dataclass
                object.__setattr__(self, "qpu_system_name", self.qpu_system_name.name)
            elif not isinstance(self.qpu_system_name, str):
                raise TypeError(
                    "Expected a QPUSystem instance or str, got "
                    f"{type(self.qpu_system_name)}"
                )

        if self.use_circuit_packing is not None and not isinstance(
            self.use_circuit_packing, bool
        ):
            raise TypeError(f"Expected a bool, got {type(self.use_circuit_packing)}")


class MaxRetriesReachedError(Exception):
    """Exception raised when the maximum number of retries is reached."""

    def __init__(self, retries):
        self.retries = retries
        self.message = f"Maximum retries reached: {retries} retries attempted"
        super().__init__(self.message)


def _parse_qpu_systems(json_data: list) -> list[QPUSystem]:
    return [
        QPUSystem(
            name=system_data["name"],
            qpus=[QPU(**qpu) for qpu in system_data.get("qpus", [])],
            access_level=system_data["access_level"],
        )
        for system_data in json_data
    ]


_DEFAULT_QPU_SYSTEM = "qoro_maestro"

_DEFAULT_JOB_CONFIG = JobConfig(
    shots=1000, qpu_system_name=_DEFAULT_QPU_SYSTEM, use_circuit_packing=False
)


class QoroService(CircuitRunner):
    """A client for interacting with the Qoro Quantum Service API.

    This class provides methods to submit circuits, check job status,
    and retrieve results from the Qoro platform.
    """

    def __init__(
        self,
        auth_token: str | None = None,
        config: JobConfig | None = None,
        polling_interval: float = 3.0,
        max_retries: int = 5000,
    ):
        """Initializes the QoroService client.

        Args:
            auth_token (str | None, optional):
                The authentication token for the Qoro API. If not provided,
                it will be read from the QORO_API_KEY in a .env file.
            config (JobConfig | None, optional):
                A JobConfig object containing default job settings. If not
                provided, a default configuration will be created.
            polling_interval (float, optional):
                The interval in seconds for polling job status. Defaults to 3.0.
            max_retries (int, optional):
                The maximum number of retries for polling. Defaults to 5000.
        """
        if config is None:
            config = _DEFAULT_JOB_CONFIG
        self.config = config

        super().__init__(shots=self.config.shots)

        if auth_token is None:
            try:
                auth_token = dotenv_values()["QORO_API_KEY"]
            except KeyError:
                raise ValueError("Qoro API key not provided nor found in a .env file.")

        self.auth_token = "Bearer " + auth_token
        self.polling_interval = polling_interval
        self.max_retries = max_retries

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Make an authenticated HTTP request to the Qoro API.

        This internal method centralizes all API communication, handling authentication
        headers and error responses consistently.

        Args:
            method (str): HTTP method to use (e.g., 'get', 'post', 'delete').
            endpoint (str): API endpoint path (without base URL).
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

        response = session.request(method, url, headers=headers, **kwargs)

        # Generic error handling for non-OK statuses
        if response.status_code >= 400:
            raise requests.exceptions.HTTPError(
                f"API Error: {response.status_code} {response.reason} for URL {response.url}"
            )

        return response

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
        return _parse_qpu_systems(response.json())

    @staticmethod
    def _compress_data(value) -> bytes:
        return base64.b64encode(gzip.compress(value.encode("utf-8"))).decode("utf-8")

    def _split_circuits(self, circuits: dict[str, str]) -> list[dict[str, str]]:
        """
        Splits circuits into chunks by estimating payload size with a simplified,
        consistent overhead calculation.
        Assumes that BASE64 encoding produces ASCI characters, which are 1 byte each.
        """
        max_payload_bytes = _MAX_PAYLOAD_SIZE_MB * 1024 * 1024
        circuit_chunks = []
        current_chunk = {}

        # Start with size 2 for the opening and closing curly braces '{}'
        current_chunk_size_bytes = 2

        for key, value in circuits.items():
            compressed_value = self._compress_data(value)

            item_size_bytes = len(key) + len(compressed_value) + 6

            # If adding this item would exceed the limit, finalize the current chunk.
            # This check only runs if the chunk is not empty.
            if current_chunk and (
                current_chunk_size_bytes + item_size_bytes > max_payload_bytes
            ):
                circuit_chunks.append(current_chunk)

                # Start a new chunk
                current_chunk = {}
                current_chunk_size_bytes = 2

            # Add the new item to the current chunk and update its size
            current_chunk[key] = compressed_value
            current_chunk_size_bytes += item_size_bytes

        # Add the last remaining chunk if it's not empty
        if current_chunk:
            circuit_chunks.append(current_chunk)

        return circuit_chunks

    def submit_circuits(
        self,
        circuits: dict[str, str],
        job_type: JobType = JobType.SIMULATE,
        override_config: JobConfig | None = None,
    ) -> str:
        """
        Submit quantum circuits to the Qoro API for execution.

        This method first initializes a job and then sends the circuits in
        one or more chunks, associating them all with a single job ID.

        Args:
            circuits (dict[str, str]):
                Dictionary mapping unique circuit IDs to QASM circuit strings.
            job_type (JobType, optional):
                Type of job to execute (e.g., SIMULATE, EXECUTE, EXPECTATION, CIRCUIT_CUT).
                Defaults to JobType.SIMULATE.
            override_config (JobConfig | None, optional):
                Configuration object to override the service's default settings.
                If not provided, default values are used.

        Raises:
            ValueError: If more than one circuit is submitted for a CIRCUIT_CUT job,
                        or if any circuit is not valid QASM.
            requests.exceptions.HTTPError: If any API request fails.

        Returns:
            str: The job ID for the created job.
        """
        # 1. Create final job configuration by layering configurations:
        #    service defaults -> user overrides
        job_config = (
            job_config.override(override_config) if override_config else job_config
        )

        # 2. Validate circuits
        if job_type == JobType.CIRCUIT_CUT and len(circuits) > 1:
            raise ValueError("Only one circuit allowed for circuit-cutting jobs.")

        for key, circuit in circuits.items():
            if not (err := is_valid_qasm(circuit)):
                raise ValueError(f"Circuit '{key}' is not a valid QASM: {err}")

        # 3. Initialize the job without circuits to get a job_id
        init_payload = {
            "shots": job_config.shots,
            "tag": job_config.tag,
            "job_type": job_type.value,
            "qpu_system_name": job_config.qpu_system_name,
            "use_packing": job_config.use_circuit_packing,
        }

        init_response = self._make_request(
            "post", "job/init/", json=init_payload, timeout=100
        )
        if init_response.status_code not in [HTTPStatus.OK, HTTPStatus.CREATED]:
            _raise_with_details(init_response)
        job_id = init_response.json()["job_id"]

        # 4. Split circuits and add them to the created job
        circuit_chunks = self._split_circuits(circuits)
        num_chunks = len(circuit_chunks)

        for i, chunk in enumerate(circuit_chunks):
            is_last_chunk = i == num_chunks - 1
            add_circuits_payload = {
                "circuits": chunk,
                "shots": self.config.shots,
                "mode": "append",
                "finalized": "true" if is_last_chunk else "false",
            }

            add_circuits_response = self._make_request(
                "post",
                f"job/{job_id}/add_circuits/",
                json=add_circuits_payload,
                timeout=100,
            )
            if add_circuits_response.status_code != HTTPStatus.OK:
                _raise_with_details(add_circuits_response)

        return job_id

    def delete_job(self, job_id: str) -> requests.Response:
        """
        Delete a job from the Qoro Database.

        Args:
            job_id: The ID of the job to be deleted.
        Returns:
            requests.Response: The response from the API.
        """
        return self._make_request(
            "delete",
            f"job/{job_id}",
            timeout=50,
        )

    def get_job_results(self, job_id: str) -> list[dict]:
        """
        Get the results of a job from the Qoro Database.

        Args:
            job_id: The ID of the job to get results from.
        Returns:
            list[dict]: The results of the job, with histograms decoded.
        """
        try:
            response = self._make_request(
                "get",
                f"job/{job_id}/resultsV2/?limit=100&offset=0",
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

        # If the request was successful, process the data
        data = response.json()
        for result in data["results"]:
            result["results"] = _decode_qh1_b64(result["results"])
        return data["results"]

    def poll_job_status(
        self,
        job_id: str,
        loop_until_complete: bool = False,
        on_complete: Callable[[requests.Response], None] | None = None,
        verbose: bool = True,
        poll_callback: Callable[[int, str], None] | None = None,
    ) -> str | JobStatus:
        """
        Get the status of a job and optionally execute a function on completion.

        Args:
            job_id: The ID of the job to check.
            loop_until_complete (bool): If True, polls until the job is complete or failed.
            on_complete (Callable, optional): A function to call with the final response
                object when the job finishes.
            verbose (bool, optional): If True, prints polling status to the logger.
            poll_callback (Callable, optional): A function for updating progress bars.
                Takes `(retry_count, status)`.

        Returns:
            str | JobStatus: The current job status as a string if not looping,
            or a JobStatus enum member (COMPLETED or FAILED) if looping.
        """
        # Decide once at the start which update function to use
        if poll_callback:
            update_fn = poll_callback
        elif verbose:
            CYAN = "\033[36m"
            RESET = "\033[0m"

            update_fn = lambda retry_count, status: logger.info(
                rf"Job {CYAN}{job_id.split('-')[0]}{RESET} is {status}. Polling attempt {retry_count} / {self.max_retries}\r",
                extra={"append": True},
            )
        else:
            update_fn = lambda _, __: None

        if not loop_until_complete:
            response = self._make_request("get", f"job/{job_id}/status/", timeout=200)
            return response.json()["status"]

        for retry_count in range(1, self.max_retries + 1):
            response = self._make_request("get", f"job/{job_id}/status/", timeout=200)
            status = response.json()["status"]

            if status == JobStatus.COMPLETED.value:
                if on_complete:
                    on_complete(response)
                return JobStatus.COMPLETED

            if status == JobStatus.FAILED.value:
                if on_complete:
                    on_complete(response)
                return JobStatus.FAILED

            update_fn(retry_count, status)
            time.sleep(self.polling_interval)

        raise MaxRetriesReachedError(self.max_retries)
