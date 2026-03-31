Backends Guide
==============

Divi provides a flexible backend system that allows you to run quantum programs on different execution environments, from local simulators to cloud-based quantum hardware. This guide will walk you through the available backends and how to choose the right one for your needs.

Backend Architecture
--------------------

All backends in Divi implement the :class:`CircuitRunner` interface, providing a consistent API regardless of the underlying execution environment. This powerful abstraction allows you to develop your quantum programs locally and then switch to a different backend—like cloud hardware—with a single line of code.

Understanding ExecutionResult
------------------------------

All backend :meth:`submit_circuits` methods return an :class:`ExecutionResult` object, which provides a unified interface for handling both synchronous and asynchronous execution.

**For Synchronous Backends** (like :class:`MaestroSimulator` and :class:`QiskitSimulator`):
   Results are available immediately after submission:

   .. code-block:: python

      from divi.backends import MaestroSimulator

      backend = MaestroSimulator()
      result = backend.submit_circuits({"circuit_0": qasm_string})

      # Access results directly
      for circuit_result in result.results:
          label = circuit_result["label"]
          counts = circuit_result["results"]
          print(f"{label}: {counts}")

**For Asynchronous Backends** (like :class:`QoroService`):
   For cloud-based backends, you need to wait for the job to complete and then fetch the results:

   .. code-block:: python

      from divi.backends import QoroService

      service = QoroService()
      result = service.submit_circuits({"circuit_0": qasm_string})

      # Wait for the job to complete
      service.poll_job_status(result, loop_until_complete=True)

      # Fetch the results
      completed_result = service.get_job_results(result)

      # Access the results
      for circuit_result in completed_result.results:
          label = circuit_result["label"]
          counts = circuit_result["results"]
          print(f"{label}: {counts}")

**Note:** When using high-level algorithms like :class:`VQE` or :class:`QAOA`, you don't interact with :class:`ExecutionResult` directly — the algorithm handles submission and result fetching. The examples above show the patterns for when you call ``submit_circuits`` yourself.

**Result Format:**
   The ``results`` attribute is a list of dictionaries, each containing:

   - ``label`` (str): The circuit label from your input dictionary
   - ``results`` (dict): The execution results (bitstring counts for sampling mode, or expectation values for expectation mode)

   Example:

   .. code-block:: python

      [
          {"label": "circuit_0", "results": {"00": 500, "11": 500}},
          {"label": "circuit_1", "results": {"01": 1000}}
      ]

Available Backends
------------------

Divi comes with three primary backends out of the box:

* :class:`MaestroSimulator` — A high-performance local simulator, recommended as the default for development and testing.
* :class:`QiskitSimulator` — A convenience wrapper around Qiskit's ``AerSimulator`` with simplified noise modeling and thread-count control. Use this when you need noisy simulation.
* :class:`QoroService` — A cloud-based quantum computing service for accessing powerful simulators and real quantum hardware.

Let's dive into each one.

MaestroSimulator
-----------------

The :class:`MaestroSimulator` is the recommended backend for local development, testing, and research. It is powered by Qoro's C++ quantum simulator (``qoro-maestro``) and provides fast, accurate simulation with automatic Statevector-to-MPS fallback for larger circuits.

**Key Features:**

* **High Performance**: Significantly faster than Qiskit Aer across typical circuit sizes (auto-MPS keeps it competitive beyond 22 qubits).
* **Auto Method Selection**: Automatically switches from Statevector to MPS for circuits exceeding 22 qubits (configurable via ``mps_qubit_threshold``).
* **Multiple Simulation Methods**: Statevector, MPS, Stabilizer, TensorNetwork, PauliPropagator.


Getting Started
^^^^^^^^^^^^^^^

.. code-block:: python

   from divi.backends import MaestroSimulator

   # Default — auto-selects Statevector or MPS based on circuit size
   backend = MaestroSimulator()

   # Explicit MPS for large circuits
   backend = MaestroSimulator(
       shots=5000,
       simulation_type="MatrixProductState",
       max_bond_dimension=64,
   )


QiskitSimulator
------------------

The :class:`QiskitSimulator` is a convenience wrapper around Qiskit's ``AerSimulator`` with simplified thread-count control and noise configuration. Use it when you need to model realistic hardware noise — for example, when developing error mitigation strategies or benchmarking algorithm robustness.


Examples
^^^^^^^^

.. code-block:: python

   from divi.backends import QiskitSimulator

   # Reproducible noisy simulation
   backend = QiskitSimulator(
       shots=10000,
       n_processes=8,
       qiskit_backend="auto", # Auto-select a Qiskit fake backend by qubit count
       simulation_seed=42     # Deterministic results for debugging
   )

   # Noisy simulation to mimic real hardware
   from qiskit_ibm_runtime.fake_provider import FakeManilaV2
   backend = QiskitSimulator(
       shots=5000,
       qiskit_backend=FakeManilaV2(),  # Use a fake backend with a realistic noise model
       n_processes=2
   )

QoroService
------------

The :class:`QoroService` provides access to cloud-based quantum computing resources, including advanced simulation (tensor networks and more) and real quantum hardware. It supports **sampling mode** (measurement counts) and **expectation mode** (Pauli expectation values, simulation-only). Circuit packing (``JobConfig.use_circuit_packing``) can batch circuits for efficiency.

Submitting and Monitoring Jobs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from divi.backends import QoroService

   service = QoroService()

   # Sampling mode — submit circuits, poll, fetch results
   result = service.submit_circuits({"c0": qasm_string_1, "c1": qasm_string_2})
   service.poll_job_status(result, loop_until_complete=True)
   completed = service.get_job_results(result)
   # [{'label': 'c0', 'results': {'0011': 2000}}, ...]

   # Expectation mode — pass ham_ops (semicolon-separated Pauli terms)
   result = service.submit_circuits({"c0": qasm_string}, ham_ops="XYZ;XXZ;ZIZ")
   service.poll_job_status(result, loop_until_complete=True)
   completed = service.get_job_results(result)
   # [{'label': 'c0', 'results': {'XYZ': 0.5, 'XXZ': -0.3, 'ZIZ': 1.0}}]

   # Cancel a job
   service.cancel_job(result)

.. note::

   **Bitstring Ordering**: :class:`QoroService` returns bitstrings in **Little Endian** ordering (least significant bit first, rightmost bit is qubit 0), but Hamiltonian operators passed via the ``ham_ops`` parameter should follow **Big Endian** ordering (most significant bit first, leftmost bit is qubit 0). For example, a 4-qubit system with qubits labeled 0-3: the bitstring ``"0011"`` in results represents qubit 0=1, qubit 1=1, qubit 2=0, qubit 3=0 (reading right to left), while the Hamiltonian operator ``"ZIZI"`` applies Z to qubit 0, I to qubit 1, Z to qubit 2, and I to qubit 3 (reading left to right).

Configuring Jobs with :class:`JobConfig`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`QoroService` uses a :class:`JobConfig` object to manage settings for job submissions. You can configure it in two ways:

1.  **Default Configuration**: Set a default :class:`JobConfig` when you initialize the service. This configuration will apply to all jobs unless you override it.
2.  **Override Configuration**: For a specific job, you can provide an ``override_job_config`` to the ``submit_circuits`` method.

.. code-block:: python

   from divi.backends import QoroService, JobConfig

   # 1. Set a custom default configuration for the service
   default_config = JobConfig(
       shots=500,
       simulator_cluster="qoro_maestro",
       use_circuit_packing=True,
       tag="default_run"
   )
   service = QoroService(job_config=default_config)

   # 2. Override the default configuration for a single job
   override = JobConfig(shots=2000, tag="high_shot_run")
   execution_result = service.submit_circuits(circuits, override_job_config=override)

   # This job will run with 2000 shots and the tag 'high_shot_run',
   # but will still use 'qoro_maestro' and circuit packing from the default config.

You can also update the service's default configuration after construction:

.. code-block:: python

   # Update the service's default job configuration
   service.job_config = JobConfig(shots=2000, simulator_cluster="qoro_maestro")

   # Update the service's default execution configuration
   service.execution_config = ExecutionConfig(bond_dimension=512)

The ``job_config`` setter automatically resolves string target names and
defaults to the ``qoro_maestro`` simulator cluster when neither
``simulator_cluster`` nor ``qpu_system`` is set, just like the constructor does.

Execution Configuration
^^^^^^^^^^^^^^^^^^^^^^^

Control the simulator backend, simulation method, bond dimension, and runtime
metadata for your jobs using :class:`ExecutionConfig`. Like :class:`JobConfig`,
you can configure it in two ways:

1.  **Default Configuration**: Set a default :class:`ExecutionConfig` when you initialize the service. This configuration will apply to all jobs unless you override it.
2.  **Per-submission Override**: Pass an ``execution_config`` to ``submit_circuits`` to override the default for a single job. Non-None fields in the override take precedence.

.. code-block:: python

   from divi.backends import (
       QoroService, ExecutionConfig, Simulator, SimulationMethod
   )

   # 1. Set a service-level default execution configuration
   default_exec = ExecutionConfig(
       bond_dimension=256,
       simulator=Simulator.QCSim,
       simulation_method=SimulationMethod.MatrixProductState,
   )
   service = QoroService(execution_config=default_exec)

   # All submissions use the default execution config
   result = service.submit_circuits(circuits)

   # 2. Override specific fields for a single submission
   override = ExecutionConfig(bond_dimension=512, api_meta={"optimization_level": 2})
   result = service.submit_circuits(circuits, override_execution_config=override)
   # Uses bond_dimension=512 and api_meta from the override,
   # but keeps simulator and simulation_method from the default.

   # Retrieve the configuration to verify
   retrieved = service.get_execution_config(result)
   print(retrieved.bond_dimension)  # 512

All ``ExecutionConfig`` fields are optional; only the fields you provide are
sent to the service. You can update the configuration later with
``set_execution_config`` as long as the job is still ``PENDING``; each call
replaces the previous execution configuration for that job.

.. note::

   Execution configuration can only be set on jobs in **PENDING** status. Attempting to set it on a running or completed job will raise a ``409 Conflict`` error.

.. warning::

   The ``bond_dimension`` field is subject to tier-based caps. Free-tier users are limited to a maximum of 32. Exceeding the cap returns a ``403 Forbidden`` error.

The ``api_meta`` field accepts a dictionary of runtime pass-through metadata. See the API reference for the full list of allowed keys (e.g. ``optimization_level``, ``resilience_level``, ``max_execution_time``).

.. _Backend Selection Guide:

Backend Selection Guide
-----------------------

Choosing the right backend depends on what stage of development you're in.

* **For Development and Testing**, use :class:`MaestroSimulator`. For noisy simulation, use :class:`QiskitSimulator` with Qiskit noise models.
* **For Production Runs**, use :class:`QoroService` for cloud simulation, real quantum hardware, and scalable execution.
* **For Research**, start with :class:`MaestroSimulator` for prototyping, then use :class:`QoroService` for validation against real hardware.

Backend Comparison
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 27 27 26
   :stub-columns: 1

   * - Feature
     - MaestroSimulator
     - QiskitSimulator
     - QoroService
   * - **Use Case**
     - Default local simulation
     - Noisy simulation
     - Production & real hardware
   * - **Simulation Engine**
     - Qoro C++ (qoro-maestro)
     - Qiskit Aer
     - Cloud (Maestro / Aer / hardware)
   * - **Noise Support**
     - No
     - Qiskit fake backends & noise models
     - Hardware noise (real QPUs)
   * - **Seed / Reproducibility**
     - Not yet supported
     - ``simulation_seed`` parameter
     - N/A

Depth Tracking
--------------

All backends support ``track_depth=True`` to record circuit depths across submissions:

.. code-block:: python

   backend = MaestroSimulator(track_depth=True)
   # After running: backend.average_depth(), backend.std_depth()

Common Issues and Solutions
---------------------------

* **Slow MaestroSimulator at >22 qubits**: The auto-MPS threshold handles this automatically. If you need to tune it, set ``mps_qubit_threshold`` or explicitly use ``simulation_type="MatrixProductState"`` with a suitable ``max_bond_dimension``.
* **Slow QiskitSimulator**: Increase ``n_processes`` or reduce ``shots``.
* **High memory usage with QiskitSimulator**: Reduce ``n_processes`` or ``shots``.
* **Non-reproducible results**: :class:`MaestroSimulator` does not yet support seeded sampling. Use :class:`QiskitSimulator` with ``simulation_seed`` when you need exact reproducibility.
* **Job queue delays on QoroService**: Use local simulation for development; submit cloud jobs during off-peak hours.

Next Steps
----------

* Try the runnable examples in the `tutorials/ <https://github.com/QoroQuantum/divi/tree/main/tutorials>`_ directory.
* Learn about :doc:`improving_results_qem` for improving your results on noisy hardware.
