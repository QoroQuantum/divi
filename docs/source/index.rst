Divi Documentation
==================

Divi is a Python library for building and running quantum programs at scale.
It sits above circuit-level frameworks like PennyLane and Qiskit and handles the
orchestration that practitioners usually write by hand: circuit generation,
batching, error mitigation, parameter optimisation, and result aggregation.

Why Divi?
---------

* **Batteries-included algorithms** — ready-to-run
  :class:`~divi.qprog.algorithms.VQE` for chemistry,
  :class:`~divi.qprog.algorithms.QAOA` / :class:`~divi.qprog.algorithms.PCE` for
  combinatorial optimisation, :class:`~divi.qprog.algorithms.TimeEvolution` for
  dynamics, and :class:`~divi.qprog.algorithms.QNN` for quantum machine
  learning — or bring your own circuit with
  :class:`~divi.qprog.algorithms.CustomVQA`.
* **Structured pipelines** — an *expand → execute → reduce* model automates the
  path from a high-level program to executed circuits, with inspectable stages for
  compilation, batching, and error mitigation. See
  :doc:`execution_workflows/pipelines`.
* **Program ensembles** — run many variational programs in parallel under one
  :class:`~divi.qprog.ensemble.ProgramEnsemble`, optionally over multiple
  adaptive rounds where each round's programs are chosen from what the last
  round measured, with automatic circuit batching and aggregation. Built-in workflows cover hyperparameter sweeps, graph
  partitioning, and time-evolution trajectories. See
  :doc:`execution_workflows/program_ensembles`.
* **Swap backends without changing code** — develop and simulate locally with
  :class:`~divi.backends.MaestroSimulator`, use
  :class:`~divi.backends.QiskitSimulator` for Qiskit-native or
  device-calibrated noise models, and scale up on the cloud via
  :class:`~divi.backends.QoroService` — all behind the same
  :class:`~divi.backends.CircuitRunner` interface. See
  :doc:`execution_workflows/backends`.
* **Integrated error mitigation** — Zero-Noise Extrapolation and QuEPP plug
  directly into the variational loop, not as a post-processing step. See
  :doc:`algorithms/improving_results_qem`.

New to Divi? Start with the :doc:`quickstart` for a five-minute VQE example and
a tour of the built-in algorithms, then explore :guilabel:`Algorithms` or
:guilabel:`Execution & Workflows` in the sidebar.

Installation
============

Divi can be installed using uv (recommended) or pip.

If you have uv installed:

.. code-block:: bash

   uv add qoro-divi

Or if you want to install from source:

.. code-block:: bash

   git clone https://github.com/QoroQuantum/divi.git
   cd divi
   uv sync

Alternatively, you can install using pip:

.. code-block:: bash

   pip install qoro-divi

Nightly Builds
--------------

Nightly development builds are published daily from ``main``. To install the latest nightly:

.. code-block:: bash

   pip install qoro-divi --pre

Or pin a specific nightly by date:

.. code-block:: bash

   pip install qoro-divi==0.8.0.dev20260305

.. note::

   Nightly builds may contain unstable or experimental features.
   For production use, stick with the stable release (``pip install qoro-divi``).

.. toctree::
   :maxdepth: 1

   quickstart

.. toctree::
   :maxdepth: 1
   :caption: Algorithms

   algorithms/ground_state_energy_estimation_vqe
   algorithms/localized_active_space_sqd
   algorithms/combinatorial_optimization_qaoa_pce
   algorithms/routing
   algorithms/hamiltonian_time_evolution
   algorithms/quantum_neural_networks
   algorithms/improving_results_qem

.. toctree::
   :maxdepth: 1
   :caption: Execution & Workflows

   execution_workflows/core_concepts
   execution_workflows/backends
   execution_workflows/optimizers
   execution_workflows/resuming_long_runs
   execution_workflows/visualization
   execution_workflows/program_ensembles
   execution_workflows/framework_integration
   execution_workflows/pipelines
   execution_workflows/pipeline_authoring

.. toctree::
   :maxdepth: 1
   :caption: Tools

   tools/divi_ai
   tools/qubo_characterization

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api_reference/qprog/index
   api_reference/program_ensembles
   api_reference/backends
   api_reference/circuits
   api_reference/pipeline
   api_reference/reporting
   api_reference/visualization
   api_reference/hamiltonians
   api_reference/exceptions

.. toctree::
   :maxdepth: 1
   :caption: Development

   development/contributing
   development/building_docs
   development/testing

Indices and tables
==================

* :ref:`genindex`
