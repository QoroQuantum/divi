Testing
=======

This guide explains Divi's comprehensive testing framework and how to run tests effectively.

Running Tests
-------------

**Basic Test Execution**

To run all tests:

.. code-block:: bash

   uv run pytest -n auto

To run tests with coverage:

.. code-block:: bash

   uv run pytest -n auto --cov=divi

To run specific test files:

.. code-block:: bash

   uv run pytest tests/hamiltonians/

**Parallel Testing** (Recommended for CI/Large Test Suites)

Divi supports parallel test execution using `pytest-xdist <https://pytest-xdist.readthedocs.io/>`_:

.. code-block:: bash

   uv run pytest -n auto    # Auto-detect CPU cores
   uv run pytest -n 4       # Use 4 workers
   uv run pytest -n 0       # Disable parallel execution

**Test Markers**

Divi uses custom pytest markers to categorize tests:

.. code-block:: bash

   uv run pytest -m "requires_api_key"  # API-dependent tests only
   uv run pytest -m "e2e"               # End-to-end tests only
   uv run pytest -m "not e2e"           # Skip slow e2e tests

Available markers:
- ``requires_api_key``: Tests requiring QORO_API_KEY
- ``e2e``: Slow end-to-end integration tests

**API Testing**

Tests requiring cloud API access are conditionally executed:

.. code-block:: bash

   # Run API tests (requires QORO_API_KEY environment variable)
   uv run pytest --run-api-tests

   # Run API tests with specific key
   uv run pytest --run-api-tests --api-key your-key-here

Test Structure
--------------

Tests under ``tests/`` mirror the ``divi/`` package layout. The only
non-obvious split is within ``tests/qprog/``:

- ``tests/qprog/problems/`` - one file per ``QAOAProblem`` subclass
- ``tests/qprog/algorithms/`` - algorithm-generic behavior (QAOA pipeline, QDrift, VQE, PCE)
- ``tests/qprog/workflows/`` - workflow orchestrators (``PartitioningProgramEnsemble``, ``VQEHyperparameterSweep``)

Shared test infrastructure
~~~~~~~~~~~~~~~~~~~~~~~~~~

Besides ``test_*.py`` files, packages use three supporting module types.
Keep their roles separate — do not put fixtures in ``_helpers.py`` or
import from ``conftest.py`` in test code.

``conftest.py``
   Pytest **fixtures** and hooks (``@pytest.fixture``, ``pytest_addoption``).
   Discovered automatically; scoped by directory (``tests/conftest.py`` is
   global, ``tests/backends/conftest.py`` is backend-only, etc.).

``_helpers.py``
   **Explicit-import** utilities: factories, constants, assertion helpers,
   and shared test doubles (e.g. ``DummySpecStage``, ``ExpvalBackendSpy``).
   Import with ``from tests.<pkg>._helpers import …``.

``_contracts.py`` (or ``_<domain>_contracts.py`` when more specific)
   Shared **`verify_*`** behavioural checks used across multiple test
   modules in the same package (e.g. ``verify_basic_program_ensemble_behaviour``,
   ``verify_depth_tracking_*`` for ``CircuitRunner``). Use a descriptive
   module name when ``_contracts.py`` would be too vague.

Fixtures that only one test file needs may stay in that file. Promote them
to ``conftest.py`` when a second consumer appears in the same subtree.

Current locations:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Path
     - Role
   * - ``tests/conftest.py``
     - Global simulators, API key, warning suppression
   * - ``tests/backends/conftest.py``
     - Qoro service fixtures
   * - ``tests/backends/_helpers.py``
     - Qoro mock/result factories
   * - ``tests/backends/_circuit_runner_contracts.py``
     - ``CircuitRunner`` depth-tracking ``verify_*``
   * - ``tests/pipeline/_helpers.py``
     - Stage doubles, meta builders, backend spies
   * - ``tests/qprog/_program_contracts.py``
     - ``QuantumProgram`` / VQA / ensemble behavioural contracts
   * - ``tests/qprog/_optimizer_contracts.py``
     - Optimizer save/load behavioural contracts
   * - ``tests/qprog/conftest.py``
     - ``optimizer`` / ``checkpointing_optimizer`` parametrized fixtures
   * - ``tests/qprog/problems/_helpers.py``
     - Shared QUBO/graph test data
   * - ``tests/hamiltonians/_helpers.py``
     - PennyLane parity assertions
   * - ``tests/qprog/algorithms/_helpers.py``
     - Circuit gate introspection helpers
   * - ``tests/ai/conftest.py``
     - Doc/indexer/LLM fixtures

**Key Fixtures Available:**

- ``dummy_simulator``: Fast mock simulator returning random counts (for unit tests)
- ``dummy_expval_backend``: Mock backend that supports expectation values (for PCE tests)
- ``dummy_pipeline_env``: ``PipelineEnv`` wrapping the expval backend (for pipeline tests)
- ``default_test_simulator``: ``MaestroSimulator`` with 5000 shots (for integration tests)
- ``api_key``: Fixture providing API key for cloud tests (module-scoped)

Problem Test Structure
~~~~~~~~~~~~~~~~~~~~~~

Each ``QAOAProblem`` subclass has a dedicated test file under ``tests/qprog/problems/``.
Tests within each file follow a layered structure, from isolated units to full integration:

1. **Utility functions** — standalone helpers (QUBO construction, validation, decoding, repair)
2. **Problem class** — construction, properties, input validation, error handling
3. **QAOAProblem protocol hooks** — ``decompose``, ``extend_solution``, ``evaluate_global_solution``,
   ``initial_solution_size``, ``postprocess_candidates``
4. **QAOA integration** — Problem + QAOA together (initialization, e2e runs, checkpointing)
5. **PartitioningProgramEnsemble integration** — Problem + ensemble (program creation, aggregation, e2e)

Example layout for a new problem:

.. skip: next

.. code-block:: python

   # tests/qprog/problems/test_my_problem.py

   # --- 1. Utility functions ---
   class TestBuildMyQubo:
       def test_returns_square_matrix(self): ...
       def test_rejects_invalid_input(self): ...

   # --- 2. Problem class ---
   class TestMyProblem:
       def test_cost_hamiltonian(self): ...
       def test_is_feasible(self): ...
       def test_compute_energy(self): ...
       def test_decode_fn(self): ...

   # --- 3. QAOAProblem protocol hooks ---
   class TestDecomposeMyProblem:
       def test_decompose_returns_subproblems(self): ...
       def test_extend_solution(self): ...

   # --- 4. QAOA integration ---
   class TestMyProblemQAOA:
       def test_runs_via_qaoa(self, default_test_simulator): ...

       @pytest.mark.e2e
       def test_e2e_solution(self, default_test_simulator): ...

   # --- 5. Ensemble integration ---
   class TestMyProblemEnsemble:
       def test_create_programs(self, dummy_simulator): ...

       @pytest.mark.e2e
       def test_partitioning_e2e(self, default_test_simulator): ...

Generic ``PartitioningProgramEnsemble`` behavior (error handling, hook delegation) is tested
once in ``tests/qprog/workflows/test_partitioning_ensemble.py`` using mocked problems — do
not duplicate those tests in problem-specific files.

Writing Tests
-------------

**Test Guidelines:**

1. **Follow naming convention**: ``test_*.py`` files, ``test_*`` functions
2. **Use descriptive names**: ``test_quantum_program_initialization``
3. **Test both success and failure cases**
4. **Use appropriate fixtures** for common test data
5. **Mark slow tests**: Use ``@pytest.mark.e2e`` for integration tests
6. **Keep infrastructure roles separate**: fixtures in ``conftest.py``,
   explicit helpers in ``_helpers.py``, shared ``verify_*`` in ``_contracts.py``

**Example Test with Fixtures:**

The ``dummy_simulator`` fixture is defined in ``tests/conftest.py`` and provides
a fake backend (no real circuit execution). Use it so tests do not depend on a
real backend. Common test data (e.g. a minimal molecule) can be created inline
or in a shared fixture:

.. code-block:: python

   import numpy as np
   import pytest
   import pennylane as qp
   from divi.qprog import VQE
   from divi.qprog.algorithms import HartreeFockAnsatz

   def test_vqe_initialization(dummy_simulator):
       """Test that VQE initializes correctly with mock backend."""
       # Minimal H2 molecule for testing (or use a fixture from conftest)
       mol = qp.qchem.Molecule(
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

- Custom markers for test categorization (``requires_api_key``, ``e2e``)
- Deprecation and syntax warning filters for cleaner output

**Coverage Reporting**

Generate detailed coverage reports:

.. code-block:: bash

   uv run pytest --cov=divi --cov-report=html --cov-report=term-missing

This generates:
- Terminal output with missing line numbers
- HTML report in ``htmlcov/index.html``
- Detailed coverage analysis by file
