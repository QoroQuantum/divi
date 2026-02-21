Testing
=======

This guide explains Divi's comprehensive testing framework and how to run tests effectively.

Running Tests
-------------

**Basic Test Execution**

To run all tests:

.. code-block:: bash

   poetry run pytest

To run tests with coverage:

.. code-block:: bash

   poetry run pytest --cov=divi

To run specific test files:

.. code-block:: bash

   poetry run pytest tests/test_hamiltonians.py

**Parallel Testing** (Recommended for CI/Large Test Suites)

Divi supports parallel test execution using `pytest-xdist <https://pytest-xdist.readthedocs.io/>`_:

.. code-block:: bash

   poetry run pytest -n auto    # Auto-detect CPU cores
   poetry run pytest -n 4       # Use 4 workers
   poetry run pytest -n 0       # Disable parallel execution

**Test Markers**

Divi uses custom pytest markers to categorize tests:

.. code-block:: bash

   poetry run pytest -m "requires_api_key"  # API-dependent tests only
   poetry run pytest -m "algo"              # Algorithm tests only
   poetry run pytest -m "e2e"               # End-to-end tests only
   poetry run pytest -m "not e2e"           # Skip slow e2e tests

Available markers:
- ``requires_api_key``: Tests requiring QORO_API_KEY
- ``algo``: Algorithm-related tests
- ``e2e``: Slow end-to-end integration tests

**API Testing**

Tests requiring cloud API access are conditionally executed:

.. code-block:: bash

   # Run API tests (requires QORO_API_KEY environment variable)
   poetry run pytest --run-api-tests

   # Run API tests with specific key
   poetry run pytest --run-api-tests --api-key your-key-here

Test Structure
--------------

Tests are organized in the ``tests/`` directory:

- ``tests/qprog/`` - Tests for quantum programs and algorithms
- ``tests/circuits/`` - Tests for circuit operations and QEM
- ``tests/backends/`` - Tests for backend implementations
- ``tests/pipeline/`` - Tests for the circuit pipeline and stages
- ``tests/reporting/`` - Tests for progress reporters
- ``tests/test_hamiltonians.py`` - Tests for Hamiltonian utilities and Trotterization
- ``tests/conftest.py`` - Shared fixtures and configuration

**Key Fixtures Available:**

- ``dummy_simulator``: Fast mock simulator returning random counts (for unit tests)
- ``dummy_expval_backend``: Mock backend that supports expectation values (for PCE tests)
- ``dummy_pipeline_env``: ``PipelineEnv`` wrapping the expval backend (for pipeline tests)
- ``default_test_simulator``: Deterministic ``ParallelSimulator`` (for integration tests)
- ``api_key``: Fixture providing API key for cloud tests (module-scoped)

Writing Tests
-------------

**Test Guidelines:**

1. **Follow naming convention**: ``test_*.py`` files, ``test_*`` functions
2. **Use descriptive names**: ``test_quantum_program_initialization``
3. **Test both success and failure cases**
4. **Use appropriate fixtures** for common test data
5. **Mark slow tests**: Use ``@pytest.mark.e2e`` for integration tests

**Example Test with Fixtures:**

The ``dummy_simulator`` fixture is defined in ``tests/conftest.py`` and provides
a fake backend (no real circuit execution). Use it so tests do not depend on a
real backend. Common test data (e.g. a minimal molecule) can be created inline
or in a shared fixture:

.. code-block:: python

   import numpy as np
   import pytest
   import pennylane as qml
   from divi.qprog import VQE
   from divi.qprog.algorithms import HartreeFockAnsatz

   def test_vqe_initialization(dummy_simulator):
       """Test that VQE initializes correctly with mock backend."""
       # Minimal H2 molecule for testing (or use a fixture from conftest)
       mol = qml.qchem.Molecule(
           symbols=["H", "H"],
           coordinates=np.array([[0.0, 0.0, -0.6614], [0.0, 0.0, 0.6614]]),
       )
       vqe = VQE(
           molecule=mol,
           ansatz=HartreeFockAnsatz(),
           backend=dummy_simulator,
       )
       assert vqe.n_params_per_layer is not None
       assert vqe.n_layers is not None

   @pytest.mark.requires_api_key
   def test_cloud_backend_integration(api_key):
       """Test integration with cloud backend."""
       # This test only runs with --run-api-tests
       pass

**Mocking with pytest-mock**

Divi uses `pytest-mock <https://pytest-mock.readthedocs.io/>`_ for clean mocking in tests. The ``mocker`` fixture provides easy access to Python's ``unittest.mock``:

.. code-block:: python

   import pytest
   from divi.backends import ExecutionResult, QoroService

   def test_qoro_service_submission(mocker):
       """Test QoroService circuit submission with mocked API."""
       # Mock the requests library
       mock_post = mocker.patch('requests.post')
       mock_post.return_value.json.return_value = {"job_id": "test-123"}
       mock_post.return_value.status_code = 200

       service = QoroService()
       result = service.submit_circuits({"test": "OPENQASM 2.0; qreg q[1];"})

       assert isinstance(result, ExecutionResult)
       assert result.job_id == "test-123"
       mock_post.assert_called_once()

   def test_backend_error_handling(mocker):
       """Test error handling in backend operations."""
       # Mock a failing API call
       mocker.patch('requests.post', side_effect=ConnectionError("API unavailable"))

       service = QoroService()
       with pytest.raises(ConnectionError):
           service.submit_circuits({"test": "circuit"})

**Test Configuration**

Key pytest settings in ``pytest.ini``:

- Custom markers for test categorization (``requires_api_key``, ``algo``, ``e2e``)
- Deprecation and syntax warning filters for cleaner output

**Coverage Reporting**

Generate detailed coverage reports:

.. code-block:: bash

   poetry run pytest --cov=divi --cov-report=html --cov-report=term-missing

This generates:
- Terminal output with missing line numbers
- HTML report in ``htmlcov/index.html``
- Detailed coverage analysis by file
