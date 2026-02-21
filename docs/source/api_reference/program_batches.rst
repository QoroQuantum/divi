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
   :special-members: __init__

**Usage:**

.. code-block:: python

   from divi.qprog.batch import ProgramBatch
   from divi.qprog import VQE, QAOA
   from divi.backends import ParallelSimulator

   class MyBatch(ProgramBatch):
       def create_programs(self):
           super().create_programs()  # Required: initializes internal state
           # Create multiple quantum programs
           self.programs = {
               "program_1": VQE(...),
               "program_2": QAOA(...),
               "program_3": VQE(...),
           }

       def run(self, blocking=True):
           return super().run(blocking=blocking)

   # Run program batch: create programs first, then execute
   batch = MyBatch(backend=ParallelSimulator())
   batch.create_programs()
   batch.run(blocking=True)

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

Beam Search Aggregation
~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: divi.qprog.batch.beam_search_aggregate
