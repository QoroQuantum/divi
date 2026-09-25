Backends
========

Divi's execution layer is built around :class:`~divi.backends.CircuitRunner`:
every backend exposes the same submission API, so programs can change execution
targets without changing algorithm logic.

.. list-table:: Choose a backend
   :header-rows: 1
   :widths: 24 38 38

   * - Backend
     - Start here when
     - Move elsewhere when
   * - :class:`~divi.backends.MaestroSimulator`
     - Developing locally, running noiseless simulations, or modelling
       a hand-written ``maestro.NoiseModel``.
     - You need Qiskit-native calibration data or a cloud target.
   * - :class:`~divi.backends.QiskitSimulator`
     - Reusing a Qiskit Aer noise model or fake-backend calibration.
     - You need Maestro's local methods or managed cloud execution.
   * - :class:`~divi.backends.QoroService`
     - Running managed simulators or quantum hardware through the cloud.
     - You are still iterating rapidly on local code.

Most algorithm users can continue directly to the relevant backend section.
The next section is for callers using ``submit_circuits`` without a
:class:`~divi.qprog.QuantumProgram`.

Backend Architecture
--------------------

All backends in Divi implement the :class:`~divi.backends.CircuitRunner` interface, providing a consistent API regardless of the underlying execution environment. This powerful abstraction allows you to develop your quantum programs locally and then switch to a different backend, like cloud hardware, with a single line of code.

Direct Submission and ``ExecutionResult``
-----------------------------------------

All backend :meth:`~divi.backends.CircuitRunner.submit_circuits` methods return an :class:`~divi.backends.ExecutionResult` object, which provides a unified interface for handling both synchronous and asynchronous execution.

**Accepted Input:**
   Circuits may be given as a mapping of label → circuit, or as a bare
   sequence whose labels become the positional indices ``"0"``, ``"1"``, …
   Each circuit is either an OpenQASM string or a Qiskit
   ``QuantumCircuit``, and the two may be mixed freely:

   .. code-block:: python

      backend.submit_circuits([qiskit_circuit_a, qiskit_circuit_b])
      backend.submit_circuits({"ansatz": qiskit_circuit, "reference": qasm_string})

**Result Format:**
   The ``results`` attribute is a list of dictionaries, each containing:

   - ``label`` (str): The circuit label from your input dictionary, or the
     positional index when circuits were passed as a sequence
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

* :class:`~divi.backends.MaestroSimulator` — A high-performance local simulator, recommended as the default for development and testing.  Supports noise via ``maestro.NoiseModel`` (see :ref:`noisy-simulation-maestro`).
* :class:`~divi.backends.QiskitSimulator` — A convenience wrapper around Qiskit's ``AerSimulator`` with thread-count control.  Use this when you need device-calibrated noise from a Qiskit fake backend or an arbitrary ``qiskit_aer.noise.NoiseModel``.
* :class:`~divi.backends.QoroService` — A cloud-based quantum computing service for accessing powerful simulators and real quantum hardware.

MaestroSimulator
-----------------

:class:`~divi.backends.MaestroSimulator` is the recommended runner for local development, testing, and research. It is powered by Qoro's C++ quantum simulator (``qoro-maestro``) and automatically selects between Statevector and MatrixProductState methods based on circuit width.

.. _configuring-maestrosimulator:

Configuring MaestroSimulator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Options live in :class:`~divi.backends.MaestroConfig` and map directly to the
`maestro Python bindings <https://qoroquantum.github.io/maestro/d7/d01/python_guide.html#py_config>`_.
Without a config, Maestro uses Statevector below ``mps_qubit_threshold``
(default 22 qubits) and MPS above it.

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

Pass a ``maestro.NoiseModel`` via
:attr:`~divi.backends.MaestroConfig.noise_model`. Divi runs noisy circuits
through Maestro's ``full_noise_execute`` (sampling) and
``full_noise_estimate`` (expectation values); see the
`maestro Python guide <https://qoroquantum.github.io/maestro/d7/d01/python_guide.html>`_
for the channels a ``NoiseModel`` offers. Readout errors only show in sampled
counts, so divi warns when a model with them is used for expectation values;
pass ``force_sampling=True`` to :class:`~divi.backends.MaestroSimulator` to
include them.

.. code-block:: python

   import maestro

   noise_model = maestro.NoiseModel()
   noise_model.set_all_depolarizing(num_qubits=2, p=0.01)
   noise_model.set_readout_error(qubit=0, p_meas1_prep0=0.02, p_meas0_prep1=0.05)

:attr:`~divi.backends.MaestroConfig.noise_realizations` and
:attr:`~divi.backends.MaestroConfig.noise_seed` are passed to those entry
points; see their reference entries for what happens when they are unset.

.. code-block:: python

   from divi.backends import MaestroConfig, MaestroSimulator

   backend = MaestroSimulator(
       shots=5000,
       config=MaestroConfig(
           noise_model=noise_model,
           noise_realizations=20,
           noise_seed=42,
       ),
   )

For reproducibility under noisy execution see :ref:`Operational notes <operational-notes-shot-reproducibility>` below.


QiskitSimulator
------------------

:class:`~divi.backends.QiskitSimulator` wraps Qiskit's ``AerSimulator`` with thread-count control and Qiskit-native noise configuration.  Use it when you need device-calibrated noise from a Qiskit fake backend, or when you have an existing ``qiskit_aer.noise.NoiseModel`` you want to run as-is.  For a noise model written from scratch, :ref:`MaestroSimulator's noisy paths <noisy-simulation-maestro>` are usually faster.

.. note::

   Requires the ``aer`` extra: ``pip install "qoro-divi[aer]"``. The default
   local simulator, :class:`~divi.backends.MaestroSimulator`, is part of the
   core install — see :ref:`optional-extras`.

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

**Controlling transpilation**

``optimization_level`` is forwarded to :func:`~qiskit.compiler.transpile` for
every circuit the backend runs, and defaults to ``None`` — Qiskit's own choice.

Under a noise model this is worth setting deliberately. Optimisation rewrites a
circuit by how compressible it is, which is not uniform across a batch: a
Clifford circuit collapses much further than one holding arbitrary rotations. So
the circuits that execute can accumulate different amounts of noise than the
ones you submitted. Protocols that compare circuits to each other are sensitive
to this — :class:`~divi.circuits.quepp.QuEPP` infers its rescaling factor from
exactly that comparison, so it wants ``optimization_level=0``:

.. code-block:: python

   from qiskit_aer.noise import NoiseModel, depolarizing_error

   noise_model = NoiseModel()
   noise_model.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ["cx"])

   backend = QiskitSimulator(
       shots=5000,
       noise_model=noise_model,
       optimization_level=0,  # keep executed circuits faithful to submitted ones
   )

The same applies to a mirror-circuit benchmark: at any level above 0 a noiseless
``U · U†`` can be optimised away to nothing.

QoroService
------------

:class:`~divi.backends.QoroService` talks to the Qoro cloud API, giving programs access to advanced simulators, tensor-network backends, and real QPUs. It supports two execution modes: **sampling mode** (measurement counts) and **expectation mode** (Pauli expectation values, simulation-only).

**Two layers of batching**

QoroService participates in two complementary batching mechanisms:

1. **QPU-side circuit packing** (:attr:`~divi.backends.JobConfig.use_circuit_packing`,
   enabled by default) — packs circuits together onto the target QPU. This
   does not change how many cloud jobs are submitted; it can reduce the
   number of QPU jobs scheduled.
2. **Ensemble-level merging** (:class:`~divi.qprog.ensemble.BatchConfig` on
   :meth:`~divi.qprog.ensemble.ProgramEnsemble.run`) — merges submissions
   from multiple programs into one ``submit_circuits`` call.  See
   :ref:`circuit-batching`.

The two compose: ensemble-level merging yields one large
``submit_circuits`` call per flush, and QPU-side packing then decides how
those circuits are grouped onto QPU jobs.

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

   **Bitstring Ordering**: :class:`~divi.backends.QoroService` returns bitstrings in **Little Endian** ordering (least significant bit first, rightmost bit is qubit 0), but Hamiltonian operators passed via the ``ham_ops`` parameter should follow **Big Endian** ordering (most significant bit first, leftmost bit is qubit 0). For example, a 4-qubit system with qubits labelled 0-3: the bitstring ``"0011"`` in results represents qubit 0=1, qubit 1=1, qubit 2=0, qubit 3=0 (reading right to left), while the Hamiltonian operator ``"ZIZI"`` applies Z to qubit 0, I to qubit 1, Z to qubit 2, and I to qubit 3 (reading left to right).

Configuring Jobs with JobConfig
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~divi.backends.QoroService` uses a :class:`~divi.backends.JobConfig` object to manage settings for job submissions. You can configure it in two ways:

1.  **Default Configuration**: Set a default :class:`~divi.backends.JobConfig` when you initialise the service. This configuration will apply to all jobs unless you override it.
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

   from divi.backends import JobConfig, MaestroConfig

   # Update the service's default job configuration
   service.job_config = JobConfig(shots=2000, simulator_cluster="qoro_maestro")

   # Update the service's default Maestro settings
   service.maestro_config = MaestroConfig(max_bond_dimension=512)

The ``job_config`` setter automatically resolves string target names and
defaults to the ``qoro_maestro`` simulator cluster when neither
``simulator_cluster`` nor ``qpu_system`` is set, just like the constructor does.

Maestro Configuration
^^^^^^^^^^^^^^^^^^^^^

Cloud Maestro runs take the same :class:`~divi.backends.MaestroConfig` as a
local :class:`~divi.backends.MaestroSimulator`, noise model included, so a
config moves between the two unchanged. Like :class:`~divi.backends.JobConfig`,
you can set it in two ways:

1.  **Default Configuration**: Pass ``maestro_config`` when you initialise the service. It applies to every job unless you override it.
2.  **Per-submission Override**: Pass ``override_maestro_config`` to ``submit_circuits``. Its fields that differ from the class defaults take precedence (see :meth:`~divi.backends.MaestroConfig.override`).

.. code-block:: python

   from divi.backends import MaestroConfig, QoroService

   # 1. Set a service-level default
   default_maestro = MaestroConfig(
       simulator_type="QCSim",
       simulation_type="MatrixProductState",
       max_bond_dimension=256,
   )
   service = QoroService(maestro_config=default_maestro)

   # All submissions use the default
   result = service.submit_circuits(circuits)

   # 2. Override specific fields for a single submission
   override = MaestroConfig(max_bond_dimension=512)
   result = service.submit_circuits(circuits, override_maestro_config=override)
   # Uses max_bond_dimension=512 from the override, but keeps the simulator
   # and simulation type from the default.

   # Retrieve the configuration to verify
   print(service.get_maestro_config(result).max_bond_dimension)  # 512

A job runs on a simulator or on hardware, never both, so it takes a
:class:`~divi.backends.MaestroConfig` or a :class:`~divi.backends.DeviceConfig`
but not the other: passing one aimed at the other kind of target raises
``ValueError``. A service-level ``maestro_config`` default is left out of QPU
submissions.

For hardware options on a QPU target, see :ref:`device-settings-per-job` below.

Inspecting Vendor Configuration Blueprints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each QPU vendor that Qoro can route to — IBM, IQM, Amazon Braket — accepts its
own set of configuration keys, such as a transpilation level or an error
mitigation toggle. :meth:`~divi.backends.QoroService.fetch_vendor_blueprints`
reports which keys each vendor accepts, so you can see what is tunable on the
hardware you are targeting without consulting the dashboard.

The reply describes *shape only*. It carries no stored values and never a
credential, and it is keyed for direct lookup: one dictionary per vendor, each
holding a ``label`` plus ``credentials`` and ``device`` maps of configuration
key to its ``kind``, ``required``, ``secret``, ``choices`` and ``default``.

.. code-block:: python

   from divi.backends import QoroService

   service = QoroService()
   blueprints = service.fetch_vendor_blueprints()

   # Read one option directly
   print(blueprints["ibm"]["device"]["TRANSPILE_LEVEL"]["default"])   # 2
   print(blueprints["ibm"]["device"]["USE_TWIRLING"]["choices"])      # ['true', 'false']

   # Or list everything one vendor exposes
   for key, spec in blueprints["ibm"]["device"].items():
       requirement = "required" if spec["required"] else f"default={spec['default']!r}"
       print(f"{key}: {spec['kind']}, {requirement}")

A vendor whose configuration shape Qoro does not yet describe reports empty
``credentials`` and ``device`` maps:

.. code-block:: python

   print(blueprints["braket"]["device"])   # {}

.. note::

   A blueprint reports what a vendor *accepts*, not what your target is
   currently set to — the stored values live on the dashboard. Nor does the QPU
   system listing say which vendor a given system routes to, so pair this with
   :meth:`~divi.backends.QoroService.fetch_qpu_systems` for orientation rather
   than treating it as a description of a specific run.

.. _device-settings-per-job:

Hardware Options for One Job
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`~divi.backends.DeviceConfig` carries a job's hardware options, such as
the transpiler's optimisation level or layout and routing methods. Pass it to
``submit_circuits`` for a job on a QPU target:

.. code-block:: python

   from divi.backends import DeviceConfig, JobConfig

   result = service.submit_circuits(
       circuits,
       device_config=DeviceConfig(optimization_level=1, routing_method="sabre"),
       override_job_config=JobConfig(qpu_system="my_qpu"),
   )

Options you leave unset are not sent, and the service rejects options it does
not recognise. :meth:`~divi.backends.QoroService.get_device_config` reads them
back.

.. _Backend Selection Guide:

Analytic vs Sampling Execution
------------------------------

Each backend declares a ``supports_expval`` capability.  When it is ``True``,
expectation values are computed analytically from the state representation and
shot-based options — ``shots`` and ``shot_distribution`` — do not change the
(exact) result.  Each local backend and :class:`~divi.backends.QoroService`
reports ``supports_expval=False`` (i.e. samples) when sampling is forced:
``MaestroSimulator(force_sampling=True)``,
``QiskitSimulator(force_sampling=True)`` (also implied by passing a
``qiskit_backend`` or ``noise_model``), or ``JobConfig(force_sampling=True)`` for
the cloud service.  Setting ``shot_distribution`` on an expval-native backend is
not silent — it emits a :class:`UserWarning`.  See :ref:`adaptive-shot-allocation`
for how shot allocation interacts with this capability.

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
     - Default local simulation; ``maestro.NoiseModel`` noise
     - Qiskit-native noise (fake backends, calibrated models)
     - Production & real hardware
   * - **Simulation Engine**
     - Qoro C++ (qoro-maestro)
     - Qiskit Aer
     - Cloud (Maestro / Aer / hardware)
   * - **Noise Support**
     - ``maestro.NoiseModel``
     - Qiskit fake backends & noise models
     - Hardware noise (real QPUs)
   * - **Seed / Reproducibility**
     - ``seed``, and optionally a separate ``noise_seed`` for noisy paths
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

* **Shot reproducibility**: :attr:`~divi.backends.MaestroConfig.seed` makes MaestroSimulator runs, noisy or not, repeat exactly; :meth:`~divi.backends.MaestroSimulator.set_seed` sets it on an existing simulator.  Each circuit gets its own seed derived from its label, so circuits in a batch draw independently.  Left unset, runs draw from system entropy.
* **QoroService latency**: Client-side wait time is dominated by how you poll; tune ``polling_interval`` and ``max_retries`` on :class:`~divi.backends.QoroService`. For fast inner loops, use a local simulator; cloud queue time is outside the client library.

Next Steps
----------

* `tutorials/backends/qasm_thru_service.py <https://github.com/QoroQuantum/divi/blob/main/tutorials/backends/qasm_thru_service.py>`_ and `tutorials/backends/backend_properties_conversion.py <https://github.com/QoroQuantum/divi/blob/main/tutorials/backends/backend_properties_conversion.py>`_ — Qoro submission and backend-from-metadata workflows
* :doc:`../api_reference/backends` — full ``CircuitRunner``, ``JobConfig``, ``MaestroConfig``, and ``DeviceConfig`` reference
* :doc:`pipelines` — how programs drive backends through the pipeline
* :doc:`../algorithms/improving_results_qem` — error mitigation on noisy hardware
