# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import logging
import os
import random
import re
import warnings
from pathlib import Path

# Pin BLAS to one thread per process. Must be set before numpy loads, since BLAS
# reads its thread count at library load. Under ``-n auto`` each xdist worker
# would otherwise spawn one BLAS thread per core, oversubscribing the machine by
# the square of the core count and making the parallel suite slower than serial.
for _blas_threads_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_blas_threads_var, "1")

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

import divi.backends as backends
from divi.backends import (
    CircuitRunner,
    ExecutionResult,
    MaestroSimulator,
)
from divi.circuits._payloads import bound_circuits
from divi.circuits.quepp import SymbolicAngleWarning
from divi.pipeline import DiviPerformanceWarning, PipelineEnv
from divi.qprog.optimizers import MonteCarloOptimizer


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Suppress only Python's repeated JAX warning during worker forks."""
    config.addinivalue_line(
        "filterwarnings",
        r"ignore:os\.fork\(\) was called\. os\.fork\(\) is incompatible with "
        r"multithreaded code, and JAX is multithreaded, so this will likely "
        r"lead to a deadlock\.:RuntimeWarning",
    )


def pytest_ignore_collect(collection_path: Path, config):
    ai_tests = Path(__file__).parent / "ai"
    in_ai_tests = collection_path == ai_tests or ai_tests in collection_path.parents
    return in_ai_tests and importlib.util.find_spec("bm25s") is None


@pytest.fixture
def suppress_quepp_warnings():
    """Suppress QuEPP shallow-circuit and signal-destroyed warnings.

    Use on tests that exercise QuEPP with intentionally small circuits
    where the warnings are expected but not under test.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"QuEPP:.*shallow circuits")
        warnings.filterwarnings("ignore", message=r"QuEPP:.*signal destroyed")
        warnings.filterwarnings("ignore", message=r"QuEPP Monte Carlo:.*non-diagonal")
        warnings.filterwarnings("ignore", message=r"QuEPP:.*zero diagonal Pauli paths")
        # The η diagnostics: undefined, sign-inverted, or small enough to
        # amplify the noisy residual. Small test circuits hit all three.
        warnings.filterwarnings(
            "ignore", message=r"(?s)QuEPP:.*no Pauli path with a non-negligible"
        )
        warnings.filterwarnings("ignore", message=r"(?s)QuEPP:.*negative η")
        warnings.filterwarnings(
            "ignore", message=r"(?s)QuEPP:.*amplify the noisy residual"
        )
        warnings.filterwarnings("ignore", message=r"QuEPP Monte Carlo:.*expects only")
        # By category: a symbolic circuit is the norm in these tests, and the
        # options QuEPP disables need concrete angles.
        warnings.simplefilter("ignore", SymbolicAngleWarning)
        yield


@pytest.fixture
def suppress_pipeline_perf_warnings():
    """Suppress :class:`~divi.pipeline.DiviPerformanceWarning` during a test.

    Use when constructing a pipeline that intentionally exercises a
    legal-but-slow configuration (e.g. exhaustive QuEPP sampling or
    ParameterBindingStage placed before QEMStage).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DiviPerformanceWarning)
        yield


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

    def submit_circuits(self, payloads, **kwargs):
        res = []
        for label, qasm in bound_circuits(payloads).items():
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

    def submit_circuits(self, payloads, **kwargs):
        return ExecutionResult(results=[])


@pytest.fixture
def make_dummy_simulator():
    def _make(shots, seed=42):
        return DummySimulator(shots=shots, seed=seed)

    return _make


@pytest.fixture
def dummy_simulator(make_dummy_simulator):
    return make_dummy_simulator(1)


@pytest.fixture
def dummy_expval_backend():
    return DummyExpvalBackend(shots=100)


@pytest.fixture
def dummy_pipeline_env(dummy_expval_backend):
    """PipelineEnv with dummy expval backend (for pipeline tests)."""
    return PipelineEnv(backend=dummy_expval_backend)


@pytest.fixture
def dummy_sampling_pipeline_env(make_dummy_simulator):
    """PipelineEnv on a sampling backend, for tests that need the qwc grouping.

    An expval-capable backend promotes qwc to the analytic ``_backend_expval``
    path, which submits one circuit and allocates no per-group shots — so a test
    inspecting groups or shot allocation has to ask for a backend that samples.
    """
    return PipelineEnv(backend=make_dummy_simulator(300))


@pytest.fixture
def default_test_simulator():
    return MaestroSimulator(shots=5000)


@pytest.fixture
def sampling_test_simulator():
    """A real backend that samples rather than evaluating observables analytically.

    Needed by tests about observable grouping or per-group shot allocation: on an
    expval-capable backend the measurement stage promotes to the analytic path,
    which submits one circuit and allocates no per-group shots.
    """
    try:
        simulator_cls = backends.QiskitSimulator
    except ImportError as exc:
        pytest.skip(str(exc))
    return simulator_cls(force_sampling=True, shots=1200)


@pytest.fixture
def default_optimizer():
    """A fresh ``MonteCarloOptimizer`` for tests that don't care which optimizer
    they use (``VariationalQuantumAlgorithm`` now requires one explicitly)."""
    return MonteCarloOptimizer()


@pytest.fixture(scope="session")
def qp():
    """PennyLane, skipped when the ``pennylane`` extra is not installed.

    For front-door tests sitting in a module whose other tests are
    PennyLane-free; a wholly PennyLane module takes one ``importorskip``
    at the top instead.
    """
    return pytest.importorskip("pennylane")


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


@pytest.fixture(scope="module")
def locked_account_key(request):
    if not request.config.getoption("--run-api-tests"):
        pytest.skip("Skipping API tests. Use --run-api-tests to run them.")

    load_dotenv()

    key = os.getenv("LOCKED_ACCOUNT_KEY")

    if not key:
        pytest.skip(
            "Skipping locked account test: LOCKED_ACCOUNT_KEY not found in .env or environment."
        )

    yield key
