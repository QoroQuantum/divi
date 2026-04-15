Divi Documentation
==================

Divi is a Python library for building and running quantum programs at scale.
It sits above circuit-level frameworks like PennyLane and Qiskit and handles the
orchestration that practitioners usually write by hand: circuit generation,
batching, error mitigation, parameter optimization, and result aggregation.

Why Divi?
---------

* **Structured pipelines** — an *expand → execute → reduce* model automates the
  path from a high-level program to executed circuits, with inspectable stages for
  compilation, batching, and error mitigation. See :doc:`user_guide/pipelines`.
* **Program ensembles** — run many variational programs in parallel under one
  :class:`~divi.qprog.ensemble.ProgramEnsemble`, with automatic circuit batching
  and aggregation. Built-in workflows cover hyperparameter sweeps, graph
  partitioning, and time-evolution trajectories. See :doc:`user_guide/program_ensembles`.
* **Swap backends without changing code** — develop locally against
  :class:`~divi.backends.MaestroSimulator`, simulate noise with
  :class:`~divi.backends.QiskitSimulator`, and scale up on the cloud via
  :class:`~divi.backends.QoroService` — all behind the same
  :class:`~divi.backends.CircuitRunner` interface. See :doc:`user_guide/backends`.
* **Integrated error mitigation** — Zero-Noise Extrapolation and QuEPP plug
  directly into the variational loop, not as a post-processing step. See
  :doc:`user_guide/improving_results_qem`.

New to Divi? Start with the :doc:`quickstart` for a five-minute VQE example,
then work through the :guilabel:`User Guide` in the sidebar.

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
   :caption: User Guide

   user_guide/core_concepts
   user_guide/ground_state_energy_estimation_vqe
   user_guide/combinatorial_optimization_qaoa_pce
   user_guide/hamiltonian_time_evolution
   user_guide/routing
   user_guide/backends
   user_guide/optimizers
   user_guide/program_ensembles
   user_guide/improving_results_qem
   user_guide/visualization
   user_guide/pipelines
   user_guide/resuming_long_runs

.. toctree::
   :maxdepth: 1
   :caption: Tools

   tools/divi_ai

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
   api_reference/utils
   api_reference/types

.. toctree::
   :maxdepth: 1
   :caption: Development

   development/contributing
   development/building_docs
   development/testing

Indices and tables
==================

* :ref:`genindex`
