# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import os
import random
import re

import matplotlib

# Configure matplotlib to use Agg backend for testing
matplotlib.use("Agg")


import pytest
from dotenv import load_dotenv

from divi.backends import CircuitRunner, ParallelSimulator


class DummySimulator(CircuitRunner):
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
                        "0" * n_qubits: 50 * random.randint(1, 5),
                        "1" * n_qubits: 50 * random.randint(1, 5),
                    },
                }
            )

        return res


@pytest.fixture
def dummy_simulator():
    return DummySimulator(shots=1)


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
