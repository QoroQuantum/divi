Backends Guide
==============

Divi's execution layer is built around :class:`~divi.backends.CircuitRunner`: every backend exposes the same submission API so programs can swap simulators or the cloud service without changing algorithm code. This guide covers the bundled runners, :class:`~divi.backends.ExecutionResult`, and Qoro job configuration.

Backend Architecture
--------------------

All backends in Divi implement the :class:`~divi.backends.CircuitRunner` interface, providing a consistent API regardless of the underlying execution environment. This powerful abstraction allows you to develop your quantum programs locally and then switch to a different backend, like cloud hardware, with a single line of code.

Understanding ExecutionResult
------------------------------

All backend :meth:`~divi.backends.CircuitRunner.submit_circuits` methods return an :class:`~divi.backends.ExecutionResult` object, which provides a unified interface for handling both synchronous and asynchronous execution.

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

**For Synchronous Backends** (like :class:`~divi.backends.MaestroSimulator` and :class:`~divi.backends.QiskitSimulator`):
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

**For Asynchronous Backends** (like :class:`~divi.backends.QoroService`):
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

**Note:** When using high-level algorithms such as :class:`~divi.qprog.algorithms.VQE` or :class:`~divi.qprog.algorithms.QAOA`, you do not handle :class:`~divi.backends.ExecutionResult` yourself; the :doc:`circuit pipeline <pipelines>` submits circuits and collects results. The examples above are for direct :meth:`~divi.backends.CircuitRunner.submit_circuits` use.

Available Backends
------------------

Divi ships three :class:`~divi.backends.CircuitRunner` implementations:

* :class:`~divi.backends.MaestroSimulator` — A high-performance local simulator, recommended as the default for development and testing.  Supports Pauli-channel noise via ``maestro.NoiseModel`` (see :ref:`noisy-simulation-maestro`).
* :class:`~divi.backends.QiskitSimulator` — A convenience wrapper around Qiskit's ``AerSimulator`` with thread-count control.  Use this when you need device-calibrated noise from a Qiskit fake backend or an arbitrary ``qiskit_aer.noise.NoiseModel``.
* :class:`~divi.backends.QoroService` — A cloud-based quantum computing service for accessing powerful simulators and real quantum hardware.

MaestroSimulator
-----------------

:class:`~divi.backends.MaestroSimulator` is the recommended runner for local development, testing, and research. It is powered by Qoro's C++ quantum simulator (``qoro-maestro``) and automatically selects between Statevector and MatrixProductState methods based on circuit width.

**Key Features:**

* **Native C++ Core**: Backed by ``qoro-maestro``, a compiled simulator designed for low per-circuit overhead.
* **Auto Method Selection**: Switches from Statevector to MatrixProductState for circuits exceeding 22 qubits (configurable via :class:`~divi.backends.MaestroConfig`'s ``mps_qubit_threshold``), so a single backend handles both narrow and wide registers.
* **Multiple Simulation Methods**: Statevector, MatrixProductState, Stabilizer, TensorNetwork, PauliPropagator.
* **Pauli-channel noise**: Analytical exact-mean expectation values or Monte-Carlo sampling under a ``maestro.NoiseModel``, configured directly on :class:`~divi.backends.MaestroConfig` — see :ref:`noisy-simulation-maestro`.


.. _configuring-maestrosimulator:

Configuring MaestroSimulator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Simulator configuration is carried by a dedicated
:class:`~divi.backends.MaestroConfig` object — the same pattern
:class:`~divi.backends.QoroService` uses with
:class:`~divi.backends.ExecutionConfig`, so the mental model is the same
whether you run locally or in the cloud.  Field semantics are documented in
the `maestro Python bindings guide <https://qoroquantum.github.io/maestro/d7/d01/python_guide.html#py_config>`_,
since each :class:`~divi.backends.MaestroConfig` field maps directly to the
identically-named field on maestro's ``SimulatorConfig``.

Constructing :class:`~divi.backends.MaestroSimulator` with no config gives you
automatic simulation-method selection: Statevector for narrow circuits, MPS
above ``mps_qubit_threshold`` (default 22 qubits) so wide registers do not try
to store a full statevector.

.. code-block:: python

   from divi.backends import MaestroSimulator, MaestroConfig

   # Default — auto-selects Statevector or MPS based on circuit size
   backend = MaestroSimulator()

   # Explicit MPS for large circuits
   backend = MaestroSimulator(
       shots=5000,
       config=MaestroConfig(
           simulation_type="MatrixProductState",
           max_bond_dimension=64,
       ),
   )


.. _noisy-simulation-maestro:

Noisy simulation
^^^^^^^^^^^^^^^^

Pass a ``maestro.NoiseModel`` via :class:`~divi.backends.MaestroConfig` to
enable Pauli-channel noise.

**Building a noise model**

Each ``set_*`` method on a ``NoiseModel`` *adds* a Pauli channel to the
qubits it touches; calling more than one composes them, it does not
overwrite.

.. code-block:: python

   import maestro

   # Start with uniform 1 % depolarizing on every qubit of a 2-qubit circuit.
   noise_model = maestro.NoiseModel()
   noise_model.set_all_depolarizing(num_qubits=2, p=0.01)

   # Add stronger dephasing on top, per-qubit (composes with the above).
   noise_model.set_dephasing(qubit=0, p=0.005)
   noise_model.set_dephasing(qubit=1, p=0.02)

   # Add an asymmetric Pauli channel on qubit 0 (also composes).
   noise_model.set_qubit_noise(qubit=0, px=0.002, py=0.001, pz=0.003)

**Choosing an execution mode**

:attr:`~divi.backends.MaestroConfig.noise_model` and
:attr:`~divi.backends.MaestroConfig.noise_realizations` together select
one of five Maestro entry points:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - ``noise_realizations``
     - Backend
     - Notes
   * - *(not set)*
     - ``simple_execute`` / ``simple_estimate``
     - No noise applied.
   * - ``None`` *(expval mode, default when* ``noise_model`` *is set)*
     - ``noisy_estimate``
     - Exact analytical mean: applies per-Pauli damping coefficients to
       noiseless expectation values.  Deterministic; zero Monte-Carlo overhead.
   * - ``None`` *(sampling mode)*
     - ``noisy_execute`` with 1 realization
     - One random Pauli error pattern per circuit; counts are stochastic.
       For statistical accuracy set an explicit count (e.g. ``noise_realizations=20``).
   * - Positive ``int`` *N* *(expval mode)*
     - ``noisy_estimate_montecarlo``
     - *N* independent Pauli-injection passes; mean of expectation values.
       Converges to analytical as *N* → ∞.
   * - Positive ``int`` *N* *(sampling mode)*
     - ``noisy_execute`` with *N* batches
     - Total shots is always ``shots``; distributed across
       ``min(shots, N)`` batches, each with a fresh noise pattern.

.. note::

   ``noise_realizations=1`` (expval) is **not** the same as ``noise_realizations=None``:
   the former is one random Pauli sampling; the latter is the exact analytical mean.

**Analytical noisy expectation values**

.. code-block:: python

   from divi.backends import MaestroSimulator, MaestroConfig

   # noise_realizations defaults to None → analytical noisy_estimate
   sim = MaestroSimulator(config=MaestroConfig(noise_model=noise_model))
   result = sim.submit_circuits({"c0": qasm_string}, ham_ops="ZI;IZ")
   expvals = result.results[0]["results"]   # {"ZI": <float>, "IZ": <float>}

**Monte Carlo noisy expectation values**

.. code-block:: python

   sim_mc = MaestroSimulator(
       config=MaestroConfig(
           noise_model=noise_model,
           noise_realizations=20,
           noise_seed=42,
       )
   )
   result = sim_mc.submit_circuits({"c0": qasm_string}, ham_ops="ZI;IZ")
   expvals = result.results[0]["results"]

**Noisy sampling (counts mode)**

.. code-block:: python

   sim_sample = MaestroSimulator(
       shots=5000,
       config=MaestroConfig(
           noise_model=noise_model,
           noise_realizations=10,
           noise_seed=0,
       ),
   )
   result = sim_sample.submit_circuits({"c0": qasm_string})
   counts = result.results[0]["results"]
   # 5000 shots distributed across 10 random Pauli error patterns.

For reproducibility under noisy execution see :ref:`Operational notes <operational-notes-shot-reproducibility>` below.


QiskitSimulator
------------------

:class:`~divi.backends.QiskitSimulator` wraps Qiskit's ``AerSimulator`` with thread-count control and Qiskit-native noise configuration.  Use it when you need device-calibrated noise from a Qiskit fake backend, or when you have an existing ``qiskit_aer.noise.NoiseModel`` you want to run as-is.  For Pauli-channel noise written from scratch, :ref:`MaestroSimulator's noisy paths <noisy-simulation-maestro>` are usually faster.


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

:class:`~divi.backends.QoroService` talks to the Qoro cloud API, giving programs access to advanced simulators, tensor-network backends, and real QPUs. It supports two execution modes: **sampling mode** (measurement counts) and **expectation mode** (Pauli expectation values, simulation-only).

**Two layers of batching**

QoroService participates in two complementary batching mechanisms:

1. **Backend-level packing** (:attr:`~divi.backends.JobConfig.use_circuit_packing`,
   enabled by default) — within a single ``submit_circuits`` call, circuits
   are packed into one cloud job, amortizing the per-job scheduler cost.
2. **Ensemble-level merging** (:class:`~divi.qprog.ensemble.BatchConfig` on
   :meth:`~divi.qprog.ensemble.ProgramEnsemble.run`) — merges submissions
   from multiple programs into one ``submit_circuits`` call.  See
   :ref:`circuit-batching`.

The two compose: ensemble-level merging yields one large
``submit_circuits`` call per flush, which backend-level packing then sends
as a single cloud job.

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

   **Bitstring Ordering**: :class:`~divi.backends.QoroService` returns bitstrings in **Little Endian** ordering (least significant bit first, rightmost bit is qubit 0), but Hamiltonian operators passed via the ``ham_ops`` parameter should follow **Big Endian** ordering (most significant bit first, leftmost bit is qubit 0). For example, a 4-qubit system with qubits labeled 0-3: the bitstring ``"0011"`` in results represents qubit 0=1, qubit 1=1, qubit 2=0, qubit 3=0 (reading right to left), while the Hamiltonian operator ``"ZIZI"`` applies Z to qubit 0, I to qubit 1, Z to qubit 2, and I to qubit 3 (reading left to right).

Configuring Jobs with JobConfig
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~divi.backends.QoroService` uses a :class:`~divi.backends.JobConfig` object to manage settings for job submissions. You can configure it in two ways:

1.  **Default Configuration**: Set a default :class:`~divi.backends.JobConfig` when you initialize the service. This configuration will apply to all jobs unless you override it.
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

   from divi.backends import ExecutionConfig, JobConfig

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
metadata for your jobs using :class:`~divi.backends.ExecutionConfig`. Like :class:`~divi.backends.JobConfig`,
you can configure it in two ways:

1.  **Default Configuration**: Set a default :class:`~divi.backends.ExecutionConfig` when you initialize the service. This configuration will apply to all jobs unless you override it.
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

The ``api_meta`` field accepts runtime pass-through metadata. Allowed keys are documented on :class:`~divi.backends.ExecutionConfig` in :doc:`../api_reference/backends` (e.g. ``optimization_level``, ``resilience_level``, ``max_execution_time``).

.. _Backend Selection Guide:

Backend Selection Guide
-----------------------

Choosing the right backend depends on what stage of development you're in.

* **For Development and Testing**, use :class:`~divi.backends.MaestroSimulator` — for exact noiseless simulation, for analytical Pauli-channel noise, or for fast Monte Carlo over a hand-written noise model.
* **For Qiskit-native noise**, use :class:`~divi.backends.QiskitSimulator` — its strength is plugging into Qiskit fake backends and arbitrary ``qiskit_aer.noise.NoiseModel`` instances built from device calibration data.
* **For Production Runs**, use :class:`~divi.backends.QoroService` for cloud simulation, real quantum hardware, and scalable execution.
* **For Research**, start with :class:`~divi.backends.MaestroSimulator` for prototyping, then use :class:`~divi.backends.QoroService` for validation against real hardware.

Backend Comparison
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 27 27 26
   :stub-columns: 1

   * - Feature
     - :class:`~divi.backends.MaestroSimulator`
     - :class:`~divi.backends.QiskitSimulator`
     - :class:`~divi.backends.QoroService`
   * - **Use Case**
     - Default local simulation; Pauli-channel noise
     - Qiskit-native noise (fake backends, calibrated models)
     - Production & real hardware
   * - **Simulation Engine**
     - Qoro C++ (qoro-maestro)
     - Qiskit Aer
     - Cloud (Maestro / Aer / hardware)
   * - **Noise Support**
     - ``maestro.NoiseModel`` (Pauli channels)
     - Qiskit fake backends & noise models
     - Hardware noise (real QPUs)
   * - **Seed / Reproducibility**
     - ``noise_seed`` (noisy paths only)
     - ``simulation_seed`` parameter
     - N/A
   * - **Depth Tracking**
     - ``track_depth=True``
     - ``track_depth=True``
     - ``track_depth=True``

Depth Tracking
--------------

All backends accept ``track_depth=True`` on construction to record per-batch depths on :class:`~divi.backends.CircuitRunner`. After submissions, use :meth:`~divi.backends.CircuitRunner.average_depth`, :meth:`~divi.backends.CircuitRunner.std_depth`, and :meth:`~divi.backends.CircuitRunner.clear_depth_history` as needed.

.. code-block:: python

   backend = MaestroSimulator(track_depth=True)

Operational notes
-----------------

* **MaestroSimulator and many qubits**: See :ref:`Configuring MaestroSimulator <configuring-maestrosimulator>` above for the auto-MPS threshold and the :class:`~divi.backends.MaestroConfig` fields that control it (``mps_qubit_threshold``, ``simulation_type``, ``max_bond_dimension``).  Note that switching to MPS changes memory and runtime scaling — it is not a generic "make it faster" switch.
* **QiskitSimulator**: ``n_processes`` and ``shots`` trade throughput, memory, and statistical noise; there is no single knob—balance them for your machine and accuracy needs.

.. _operational-notes-shot-reproducibility:

* **Shot reproducibility**: :meth:`~divi.backends.MaestroSimulator.set_seed` is a no-op — Maestro's noiseless simulators seed their measurement RNG from system entropy.  :attr:`~divi.backends.MaestroConfig.noise_seed` pins **Pauli error patterns** for noisy execution (each circuit gets ``noise_seed + i``), so noisy expval runs are fully reproducible; noisy sampling counts still vary because Maestro's measurement sampler re-seeds from entropy each call.  For reproducible noiseless counts, use :class:`~divi.backends.QiskitSimulator` with ``simulation_seed``.
* **QoroService latency**: Client-side wait time is dominated by how you poll; tune ``polling_interval`` and ``max_retries`` on :class:`~divi.backends.QoroService`. For fast inner loops, use a local simulator; cloud queue time is outside the client library.

Next Steps
----------

* `tutorials/qasm_thru_service.py <https://github.com/QoroQuantum/divi/blob/main/tutorials/qasm_thru_service.py>`_ and `tutorials/backend_properties_conversion.py <https://github.com/QoroQuantum/divi/blob/main/tutorials/backend_properties_conversion.py>`_ — Qoro submission and backend-from-metadata workflows
* :doc:`../api_reference/backends` — full ``CircuitRunner``, ``JobConfig``, and ``ExecutionConfig`` reference
* :doc:`pipelines` — how programs drive backends through the pipeline
* :doc:`improving_results_qem` — error mitigation on noisy hardware
