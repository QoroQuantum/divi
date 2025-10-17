Backends Guide
==============

Divi provides a flexible backend system that allows you to run quantum programs on different execution environments, from local simulators to cloud-based quantum hardware.

Backend Architecture
--------------------

All backends in Divi implement the :class:`CircuitRunner` interface, providing a consistent API regardless of the underlying execution environment. This allows you to switch between different backends without changing your quantum program code.

Available Backends
------------------

**ParallelSimulator**
   High-performance local simulator with parallel execution capabilities.

**QoroService**
   Cloud-based quantum computing service for accessing performant simulators and real quantum hardware.

ParallelSimulator
-----------------

The ParallelSimulator is the recommended backend for development, testing, and research. It provides:

- **Fast Execution**: Optimized simulation with parallel processing
- **Noise Modeling**: Realistic noise simulation using Qiskit backends
- **Flexible Configuration**: Customizable shots, processes, and noise models
- **Local Execution**: No internet connection required

Basic Usage:

.. code-block:: python

   from divi.backends import ParallelSimulator

   # Basic simulator
   backend = ParallelSimulator()

   # Configured simulator
   backend = ParallelSimulator(
       shots=1000,
       n_processes=4,
       qiskit_backend="qasm_simulator"
   )

Configuration Options:

.. code-block:: python

   # High-performance configuration
   backend = ParallelSimulator(
       shots=10000,           # Number of measurement shots
       n_processes=8,        # Number of parallel processes
       qiskit_backend="auto", # Auto-select best simulator
       seed=42               # Random seed for reproducibility
   )

   # Noisy simulation
   from qiskit_ibm_runtime.fake_provider import FakeManilaV2
   backend = ParallelSimulator(
       shots=5000,
       qiskit_backend=FakeManilaV2(),  # Use fake backend with noise
       n_processes=2
   )

QoroService
-----------

QoroService provides access to cloud-based quantum computing resources:

- **Real Hardware**: Access to actual quantum computers
- **Scalable Execution**: Handle large job queues
- **Circuit Cutting**: Automatic circuit decomposition for large problems
- **Job Management**: Track and manage quantum jobs

Basic Usage:

.. code-block:: python

   from divi.backends import QoroService, JobType

   # Initialize service
   service = QoroService()

   # Test connection
   service.test_connection()

Submitting Circuits:

.. code-block:: python

   # Prepare circuits
   circuits = {
       "circuit_1": qasm_string_1,
       "circuit_2": qasm_string_2
   }

   # Submit jobs
   job_ids = service.submit_circuits(
       circuits,
       job_type=JobType.SIMULATE
   )

   # Monitor execution
   service.poll_job_status(job_ids, loop_until_complete=True)

   # Retrieve results
   results = service.get_job_results(job_ids)

Job Types:

.. code-block:: python

   from divi.backends import JobType

   # Standard simulation jobs
   job_ids = service.submit_circuits(circuits, job_type=JobType.SIMULATE)

   # Execution jobs
   job_ids = service.submit_circuits(circuits, job_type=JobType.EXECUTE)

   # Estimation jobs
   job_ids = service.submit_circuits(circuits, job_type=JobType.ESTIMATE)

Backend Selection Guide
-----------------------

**For Development and Testing**
   Use ParallelSimulator:
   - Fast iteration cycles
   - No external dependencies
   - Easy debugging
   - Cost-effective

**For Production Runs**
   Use QoroService:
   - Real quantum hardware
   - Scalable execution
   - Professional support
   - Advanced features

**For Research**
   Use both backends:
   - ParallelSimulator for rapid prototyping
   - QoroService for final validation
   - Compare results across backends

Performance Optimization
------------------------

**ParallelSimulator Optimization**

.. code-block:: python

   # Optimize for speed
   backend = ParallelSimulator(
       n_processes=min(8, os.cpu_count()),  # Use available cores
       shots=1000,                          # Balance accuracy vs speed
       qiskit_backend="qasm_simulator"      # Fastest simulator
   )

   # Optimize for accuracy
   backend = ParallelSimulator(
       shots=10000,                         # More shots for better statistics
       n_processes=2,                       # Fewer processes for stability
       qiskit_backend="statevector_simulator"  # Exact simulation
   )

**QoroService Optimization**

.. code-block:: python

   # Batch circuit submission
   service = QoroService()

   # Submit multiple circuits at once
   large_circuit_batch = {f"circuit_{i}": circuit for i, circuit in enumerate(circuits)}
   job_ids = service.submit_circuits(large_circuit_batch)

   # For large circuit batches, consider submitting in smaller groups
   if len(circuits) > 20:
       # Split into smaller batches if needed
       batch_size = 10
       for i in range(0, len(circuits), batch_size):
           batch = dict(list(circuits.items())[i:i+batch_size])
           job_ids = service.submit_circuits(batch)

Backend Comparison
------------------

+------------------+------------------+------------------+
| Feature          | ParallelSimulator| QoroService      |
+==================+==================+==================+
| Execution Speed  | Very Fast        | Variable         |
| Accuracy         | Perfect          | Hardware-limited |
| Cost             | Free             | Pay-per-use      |
| Scalability      | Limited          | High             |
| Noise            | Configurable     | Real hardware    |
| Availability     | Always           | Queue-dependent  |
+------------------+------------------+------------------+

Error Handling
--------------

**Connection Issues**

.. code-block:: python

   try:
       service = QoroService()
       service.test_connection()
   except ConnectionError as e:
       print(f"Connection failed: {e}")
       # Fall back to local simulator
       backend = ParallelSimulator()

**Job Failures**

.. code-block:: python

   try:
       results = service.get_job_results(job_ids)
   except JobFailedError as e:
       print(f"Job failed: {e}")
       # Retry with smaller batches for large circuits
       if len(circuits) > 20:
           batch_size = 10
           for i in range(0, len(circuits), batch_size):
               batch = dict(list(circuits.items())[i:i+batch_size])
               job_ids = service.submit_circuits(batch)

**Timeout Handling**

.. code-block:: python

   import time

   start_time = time.time()
   timeout = 300  # 5 minutes

   while time.time() - start_time < timeout:
       status = service.get_job_status(job_ids)
       if all(s == JobStatus.COMPLETED for s in status):
           break
       time.sleep(10)
   else:
       print("Job timeout - consider splitting into smaller batches")

Best Practices
--------------

1. **Start Local**: Always test with ParallelSimulator first
2. **Monitor Resources**: Track circuit counts and execution times
3. **Use Appropriate Backends**: Choose based on problem requirements
4. **Handle Errors**: Implement proper error handling and fallbacks
5. **Optimize Configuration**: Tune backend parameters for your use case

Common Issues and Solutions
---------------------------

**Slow Simulation**
   - Increase n_processes (up to CPU core count)
   - Reduce shots for testing
   - Use faster qiskit_backend

**High Memory Usage**
   - Reduce n_processes
   - Process circuits in smaller batches
   - Use less memory-intensive simulators

**Job Queue Delays**
   - Submit jobs during off-peak hours
   - Consider local simulation for development

**Connection Problems**
   - Check internet connection
   - Verify API credentials
   - Implement retry logic with exponential backoff

Next Steps
----------

- Try the runnable examples in the `tutorials/ <https://github.com/qoro-quantum/divi/tree/main/tutorials>`_ directory
- Learn about :doc:`error_mitigation` for improving results
