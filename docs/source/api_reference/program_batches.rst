Program Batches
================

The ``divi.qprog.batch`` module provides powerful program batch capabilities for running multiple quantum programs in parallel, managing large-scale hyperparameter sweeps, and handling complex workflows.

Overview
--------

Program batches in Divi enable you to:

- **Parallel Execution**: Run multiple quantum programs simultaneously
- **Hyperparameter Sweeps**: Systematically explore parameter spaces
- **Large Problem Decomposition**: Break down complex problems into manageable subproblems
- **Progress Monitoring**: Track execution across multiple programs with rich progress bars

Core Architecture
-----------------

.. autoclass:: divi.qprog.batch.ProgramBatch
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
   :special-members: __init__

**Usage:**

.. code-block:: python

   from divi.qprog.batch import ProgramBatch
   from divi.backends import ParallelSimulator

   class MyBatch(ProgramBatch):
       def create_programs(self):
           # Create multiple quantum programs
           self.programs = {
               "program_1": VQE(...),
               "program_2": QAOA(...),
               "program_3": VQE(...)
           }

       def run(self):
           # Execute all programs in parallel
           return super().run()

   # Run program batch
   batch = MyBatch(backend=ParallelSimulator())
   results = batch.run()

Workflows
---------

VQE Hyperparameter Sweeps
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: divi.qprog.workflows.VQEHyperparameterSweep
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Molecule Transformer
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: divi.qprog.workflows.MoleculeTransformer
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Graph Partitioning QAOA
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: divi.qprog.workflows.GraphPartitioningQAOA
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Partitioning Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: divi.qprog.workflows.PartitioningConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

QUBO Partitioning QAOA
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: divi.qprog.workflows.QUBOPartitioningQAOA
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Progress Monitoring
-------------------

Program batches include sophisticated progress monitoring capabilities with rich progress bars that show multiple program tracking, phase information, and error reporting.

.. automodule:: divi.qprog.batch
   :members:
   :undoc-members:
   :show-inheritance:
