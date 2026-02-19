# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import random
import re

import matplotlib

# Configure matplotlib to use Agg backend for testing
matplotlib.use("Agg")

# Suppress stevedore extension loading errors (Qiskit v2 compatibility issue)
# These occur when IBM backend plugins fail to load due to ProviderV1 removal.
# Must be set early, before any qiskit-ibm-runtime imports, to be effective in all test processes.
_stevedore_logger = logging.getLogger("stevedore.extension")
_stevedore_logger.setLevel(logging.CRITICAL)

import pytest
from dotenv import load_dotenv

from divi.backends import CircuitRunner, ExecutionResult, ParallelSimulator
from divi.pipeline import PipelineEnv


class DummySimulator(CircuitRunner):
    def __init__(self, shots, seed=42):
        super().__init__(shots=shots)
        self._rng = random.Random(seed)

    @property
    def is_async(self):
        return False

    @property
    def supports_expval(self):
        return False

    def submit_circuits(self, circuits):
        res = []
        for label, qasm in circuits.items():
            match = re.search(r"qreg q\[(\d+)\]", qasm)
            if not match:
                raise RuntimeError("QASM missing qreg for some reason")
            n_qubits = int(match.group(1))

            res.append(
                {
                    "label": label,
                    "results": {
                        "0" * n_qubits: 50 * self._rng.randint(1, 5),
                        "1" * n_qubits: 50 * self._rng.randint(1, 5),
                    },
                }
            )

        return ExecutionResult(results=res)


class DummyExpvalBackend(CircuitRunner):
    """Backend that supports expectation values (for PCE expval-mode tests)."""

    @property
    def is_async(self):
        return False

    @property
    def supports_expval(self):
        return True

    def submit_circuits(self, circuits, **kwargs):
        return ExecutionResult(results=[])


@pytest.fixture
def dummy_simulator():
    return DummySimulator(shots=1)


@pytest.fixture
def dummy_expval_backend():
    return DummyExpvalBackend(shots=100)


@pytest.fixture
def dummy_pipeline_env(dummy_expval_backend):
    """PipelineEnv with dummy expval backend (for pipeline tests)."""
    return PipelineEnv(backend=dummy_expval_backend)


@pytest.fixture
def default_test_simulator():
    return ParallelSimulator(shots=5000, _deterministic_execution=True)


def is_assertion_error(err, *_) -> bool:
    return isinstance(err, AssertionError)


def pytest_addoption(parser):
    parser.addoption(
        "--api-key",
        action="store",
        default=None,
        help="API key for authentication (can also be set via QORO_API_KEY environment variable)",
    )
    parser.addoption(
        "--run-api-tests",
        action="store_true",
        default=False,
        help="Run tests that require an API key.",
    )


@pytest.fixture(scope="module")
def api_key(request):
    if not request.config.getoption("--run-api-tests"):
        pytest.skip("Skipping API tests. Use --run-api-tests to run them.")

    # Load .env file if it exists
    load_dotenv()

    # Check command line option first, then environment variable
    key = request.config.getoption("--api-key")
    if not key:
        key = os.getenv("QORO_API_KEY")

    # Skip if no key is found
    if not key:
        pytest.skip(
            "Skipping API tests: API key not provided. Set QORO_API_KEY or use --api-key option."
        )

    # Setup code
    print(f"\nSetup: Initializing resources with API key: {key[:8]}...")

    yield key

    # Teardown code
    print(f"\nTeardown: Cleaning up resources initialized with API key: {key[:8]}...")
