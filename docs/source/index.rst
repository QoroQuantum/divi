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

   git clone https://github.com/qoro-quantum/divi.git
   cd divi
   poetry install

Alternatively, you can install using pip:

.. code-block:: bash

   pip install qoro-divi

.. toctree::
   :maxdepth: 1

   quickstart

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   user_guide/core_concepts
   user_guide/vqe
   user_guide/qaoa
   user_guide/backends
   user_guide/program_batches
   user_guide/error_mitigation
   user_guide/optimizers

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api_reference/qprog
   api_reference/program_batches
   api_reference/backends
   api_reference/utils
   api_reference/circuits
   api_reference/reporting

.. toctree::
   :maxdepth: 1
   :caption: Development

   development/contributing
   development/building_docs
   development/testing

Indices and tables
==================

* :ref:`genindex`
