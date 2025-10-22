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

When you're ready to move from simulation to real hardware, the ``QoroService`` provides access to cloud-based quantum computing resources.

**Key Features:**

* **Real Hardware**: Run your algorithms on actual quantum computers.
* **Scalable Execution**: The service is designed to handle large queues of jobs efficiently.
* **Circuit Cutting**: Automatically decompose large circuits that wouldn't otherwise fit on a QPU.
* **Job Management**: A robust system for tracking and managing your quantum jobs.

Getting Started with QoroService
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use the service, you'll first need to initialize it and test your connection.

.. code-block:: python

   from divi.backends import QoroService, JobType

   # Initialize the service (API keys are loaded from your environment)
   service = QoroService()

   # Test your connection to the service
   service.test_connection()

Submitting and Monitoring Jobs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The workflow for submitting circuits is straightforward: you send your circuits and then monitor the job until it's complete.

.. code-block:: python

   # Prepare your circuits as a dictionary
   circuits = {
       "circuit_1": qasm_string_1,
       "circuit_2": qasm_string_2
   }

   # Submit the job to the service
   job_ids = service.submit_circuits(
       circuits,
       job_type=JobType.SIMULATE  # Specify the job type
   )

   # Monitor the execution until completion
   service.poll_job_status(job_ids, loop_until_complete=True)

   # Retrieve your results
   results = service.get_job_results(job_ids)

The service supports different job types depending on your needs:

.. code-block:: python

   from divi.backends import JobType

   # Standard simulation jobs
   job_ids = service.submit_circuits(circuits, job_type=JobType.SIMULATE)

   # Execution jobs for running on real hardware
   job_ids = service.submit_circuits(circuits, job_type=JobType.EXECUTE)

   # Estimation jobs for quick cost analysis
   job_ids = service.submit_circuits(circuits, job_type=JobType.ESTIMATE)

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
     - Production & Real Hardware
   * - **Speed**
     - Fast (Local CPU)
     - High-throughput (Cloud)
   * - **Accuracy**
     - Ideal (Noiseless)
     - Real-world (Hardware noise)
   * - **Cost**
     - Free
     - Pay-per-use
   * - **Scalability**
     - Limited by local hardware
     - High (Cloud infrastructure)
   * - **Noise**
     - Simulated (Configurable)
     - Physical (Real hardware)
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

* Try the runnable examples in the `tutorials/ <https://github.com/qoro-quantum/divi/tree/main/tutorials>`_ directory.
* Learn about :doc:`error_mitigation` for improving your results on noisy hardware.
