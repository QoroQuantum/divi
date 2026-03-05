Divi Documentation
==================

Welcome to the Divi documentation! Divi is a Python library to automate generating, parallelizing, and executing quantum programs.

Installation
============

Divi can be installed using Poetry (recommended) or pip.

If you have Poetry installed:

.. code-block:: bash

   poetry add qoro-divi

Or if you want to install from source:

.. code-block:: bash

   git clone https://github.com/QoroQuantum/divi.git
   cd divi
   poetry install

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
   user_guide/pipelines
   user_guide/ground_state_energy_estimation_vqe
   user_guide/combinatorial_optimization_qaoa_pce
   user_guide/hamiltonian_time_evolution
   user_guide/backends
   user_guide/program_batches
   user_guide/improving_results_zne
   user_guide/optimizers
   user_guide/resuming_long_runs

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api_reference/qprog
   api_reference/program_batches
   api_reference/backends
   api_reference/circuits
   api_reference/pipeline
   api_reference/reporting
   api_reference/utils

.. toctree::
   :maxdepth: 1
   :caption: Development

   development/contributing
   development/building_docs
   development/testing

Indices and tables
==================

* :ref:`genindex`
