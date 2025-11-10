Backends Guide
==============

Divi provides a flexible backend system that allows you to run quantum programs on different execution environments, from local simulators to cloud-based quantum hardware. This guide will walk you through the available backends and how to choose the right one for your needs.

Backend Architecture
--------------------

All backends in Divi implement the :class:`CircuitRunner` interface, providing a consistent API regardless of the underlying execution environment. This powerful abstraction allows you to develop your quantum programs locally and then switch to a different backend—like cloud hardware—with a single line of code.

Available Backends
------------------

Divi comes with two primary backends out of the box:

* **ParallelSimulator**: A high-performance local simulator with parallel execution capabilities, perfect for development and testing.
* **QoroService**: A cloud-based quantum computing service for accessing powerful simulators and real quantum hardware.

Let's dive into each one.

ParallelSimulator
-----------------

The ``ParallelSimulator`` is your go-to backend for local development, testing, and research. It's designed for speed and flexibility, allowing you to iterate quickly without needing an internet connection.

**Key Features:**

* **Fast Execution**: Optimized simulation with parallel processing to take full advantage of your local machine.
* **Noise Modeling**: Simulate realistic noise conditions by integrating with Qiskit's fake backends.
* **Flexible Configuration**: Easily customize the number of shots, parallel processes, and noise models.
* **Local Execution**: Runs entirely on your machine, no cloud access required.

Getting Started with ParallelSimulator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using the simulator is straightforward. You can create a default instance or configure it to your specific needs.

.. code-block:: python

   from divi.backends import ParallelSimulator

   # A basic simulator with default settings
   backend = ParallelSimulator()

   # A configured simulator for better accuracy and speed
   backend = ParallelSimulator(
       shots=1000,
       n_processes=4
   )

Advanced Configuration
^^^^^^^^^^^^^^^^^^^^^^

You can tune the ``ParallelSimulator`` for different scenarios, like maximizing performance or simulating a noisy environment.

.. code-block:: python

   # High-performance configuration for production-level simulations
   backend = ParallelSimulator(
       shots=10000,           # Increase measurement shots for higher precision
       n_processes=8,        # Use more parallel processes
       qiskit_backend="auto", # Let Divi auto-select the best available simulator
       seed=42               # Set a random seed for reproducible results
   )

   # Noisy simulation to mimic real hardware
   from qiskit_ibm_runtime.fake_provider import FakeManilaV2
   backend = ParallelSimulator(
       shots=5000,
       qiskit_backend=FakeManilaV2(),  # Use a fake backend with a realistic noise model
       n_processes=2
   )

QoroService
-----------

The ``QoroService`` provides access to cloud-based quantum computing resources, including advanced simulation services with greater bandwidth and a wider variety of simulation types (such as tensor networks), as well as real quantum hardware. While ``ParallelSimulator`` is ideal for local prototyping, ``QoroService`` offers production-grade simulation capabilities and hardware access. The service supports two execution modes: **sampling mode** for measurement histograms (available on both simulation and hardware) and **expectation mode** for expectation values (currently simulation-only).

**Key Features:**

* **Advanced Simulation**: Access production-grade simulation services with greater bandwidth and a variety of simulation types, including tensor networks, beyond what's available in local prototyping.
* **Real Hardware**: Run your algorithms on actual quantum computers.
* **Scalable Execution**: The service is designed to handle large queues of jobs efficiently.
* **Circuit Packing**: Enable circuit packing optimization via ``JobConfig.use_circuit_packing`` to improve execution efficiency by combining multiple circuits into optimized batches.
* **Job Configuration**: Use ``JobConfig`` to configure job settings including shots, QPU system selection, circuit packing, and tags. Set default configurations at service initialization or override them per job.
* **Job Management**: Track job status (PENDING, RUNNING, COMPLETED, FAILED, CANCELLED), poll for completion with configurable intervals, retrieve results, and delete jobs.

Getting Started with QoroService
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use the service, you'll first need to initialize it and test your connection.

.. code-block:: python

   from divi.backends import QoroService, JobType

   # Initialize the service (API keys are loaded from your environment)
   service = QoroService()

   # Test your connection to the service
   service.test_connection()

Execution Modes
^^^^^^^^^^^^^^^

The ``QoroService`` supports two distinct execution modes:

1. **Sampling Mode** (circuit-only input): Submit circuits without Hamiltonian operators.
   The service executes the circuits with a specified number of shots and returns
   measurement histograms (bitstring counts). This mode works with both simulation
   (``JobType.SIMULATE``) and real hardware (``JobType.EXECUTE``).

2. **Expectation Mode** (circuit with Pauli terms): Submit circuits along with
   Hamiltonian operators specified as semicolon-separated Pauli terms (e.g., ``"XYZ;XXZ;ZIZ"``).
   The service automatically uses ``JobType.EXPECTATION`` and returns expectation values
   for each Pauli term. **Note**: Expectation mode is currently only available on simulation
   backends, not on real hardware.

Submitting and Monitoring Jobs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The workflow for submitting circuits depends on which execution mode you're using.

**Sampling Mode Example:**

.. code-block:: python

   # Prepare your circuits as a dictionary
   circuits = {
       "circuit_1": qasm_string_1,
       "circuit_2": qasm_string_2
   }

   # Submit the job in sampling mode (no ham_ops parameter)
   job_id = service.submit_circuits(
       circuits,
       job_type=JobType.SIMULATE  # Can also use JobType.EXECUTE for real hardware
   )

   # Monitor the execution until completion
   service.poll_job_status(job_id, loop_until_complete=True)

   # Retrieve your results (returns histograms with bitstring counts)
   results = service.get_job_results(job_id)
   # Example output shape:
   # [{'label': 'circuit_0', 'results': {'0011': 2000}},
   #  {'label': 'circuit_1', 'results': {'0011': 2000}}, ...]

**Expectation Mode Example:**

.. code-block:: python

   # Prepare your circuits and Pauli operators
   circuits = {
       "circuit_1": qasm_string_1
   }

   # Define Hamiltonian operators as semicolon-separated Pauli terms
   # Each term must have the same length and contain only I, X, Y, Z
   ham_ops = "XYZ;XXZ;ZIZ"

   # Submit the job in expectation mode (ham_ops automatically sets JobType.EXPECTATION)
   # Note: This mode is only available on simulation backends, not real hardware
   # The QPU system in your JobConfig should be a simulation system (e.g., "qoro_maestro")
   job_id = service.submit_circuits(
       circuits,
       ham_ops=ham_ops
       # job_type is automatically set to JobType.EXPECTATION when ham_ops is provided
   )

   # Monitor the execution until completion
   service.poll_job_status(job_id, loop_until_complete=True)

   # Retrieve your results (returns expectation values for each Pauli term)
   results = service.get_job_results(job_id)
   # Example output shape:
   # [{'label': 'circuit_0', 'results': {'IIII': 1.0, 'XXXX': 0.0, 'YYYY': 0.0, 'ZZZZ': 1.0}}]

.. note::

   **Bitstring Ordering**: ``QoroService`` returns bitstrings in **Little Endian** ordering (least significant bit first, rightmost bit is qubit 0), but Hamiltonian operators passed via the ``ham_ops`` parameter should follow **Big Endian** ordering (most significant bit first, leftmost bit is qubit 0). For example, a 4-qubit system with qubits labeled 0-3: the bitstring ``"0011"`` in results represents qubit 0=1, qubit 1=1, qubit 2=0, qubit 3=0 (reading right to left), while the Hamiltonian operator ``"ZIZI"`` applies Z to qubit 0, I to qubit 1, Z to qubit 2, and I to qubit 3 (reading left to right).

Configuring Jobs with JobConfig
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``QoroService`` uses a ``JobConfig`` object to manage settings for job submissions. You can configure it in two ways:

1.  **Default Configuration**: Set a default ``JobConfig`` when you initialize the service. This configuration will apply to all jobs unless you override it.
2.  **Override Configuration**: For a specific job, you can provide an ``override_config`` to the ``submit_circuits`` method.

.. code-block:: python

   from divi.backends import QoroService, JobConfig

   # 1. Set a custom default configuration for the service
   default_config = JobConfig(
       shots=500,
       qpu_system="qoro_maestro",
       use_circuit_packing=True,
       tag="default_run"
   )
   service = QoroService(config=default_config)

   # 2. Override the default configuration for a single job
   override = JobConfig(shots=2000, tag="high_shot_run")
   job_id = service.submit_circuits(circuits, override_config=override)

   # This job will run with 2000 shots and the tag 'high_shot_run',
   # but will still use 'qoro_maestro' and circuit packing from the default config.

Backend Selection Guide
-----------------------

Choosing the right backend depends on what stage of development you're in.

* **For Development and Testing**, use ``ParallelSimulator``. It offers fast iteration cycles, easy debugging, and is completely free.
* **For Production Runs**, use ``QoroService``. It provides access to real quantum hardware, scalable execution, and advanced features.
* **For Research**, it's often best to use both. Start with ``ParallelSimulator`` for rapid prototyping and then use ``QoroService`` for final validation and to compare simulated results against real hardware.

Backend Comparison
------------------

The best choice of backend depends on your specific needs. Here's a summary of the key differences:

.. list-table::
   :header-rows: 1
   :widths: 25 37 38
   :stub-columns: 1

   * - Feature
     - ParallelSimulator
     - QoroService
   * - **Use Case**
     - Development & Prototyping
     - Production (Simulation & Real Hardware)
   * - **Speed**
     - Fast (Local CPU)
     - High-throughput (Cloud)
   * - **Accuracy**
     - Ideal
     - Ideal (simulation) / Real-world (hardware)
   * - **Cost**
     - Free
     - Pay-per-use
   * - **Scalability**
     - Limited by local hardware
     - High (Cloud infrastructure)
   * - **Availability**
     - Always (Local)
     - Queue-dependent

Best Practices
--------------

1.  **Start Local**: Always begin your development and testing with the ``ParallelSimulator``.
2.  **Monitor Resources**: Keep an eye on your circuit counts and execution times to avoid unexpected costs.
3.  **Choose the Right Backend**: Select your backend based on your specific problem requirements.
4.  **Handle Errors Gracefully**: Implement proper error handling and fallbacks in your code.
5.  **Optimize Your Configuration**: Tune your backend parameters to get the best performance for your use case.

Common Issues and Solutions
---------------------------

* **Slow Simulation**: Increase ``n_processes``, reduce ``shots`` for testing.
* **High Memory Usage**: Reduce ``n_processes``, process circuits in smaller batches.
* **Job Queue Delays**: Submit jobs during off-peak hours or use local simulation for development.
* **Connection Problems**: Check your internet connection, verify your API credentials, and implement retry logic.

Next Steps
----------

* Try the runnable examples in the `tutorials/ <https://github.com/QoroQuantum/divi/tree/main/tutorials>`_ directory.
* Learn about :doc:`error_mitigation` for improving your results on noisy hardware.
