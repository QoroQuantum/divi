Program Ensembles
==================

The ``divi.qprog.ensemble`` module provides powerful program ensemble capabilities for running multiple quantum programs in parallel, managing large-scale hyperparameter sweeps, and handling complex workflows.

Overview
--------

Program ensembles in Divi enable you to:

- **Parallel Execution**: Run multiple quantum programs simultaneously
- **Hyperparameter Sweeps**: Systematically explore parameter spaces
- **Large Problem Decomposition**: Break down complex problems into manageable subproblems
- **Progress Monitoring**: Track execution across multiple programs with rich progress bars

Core Architecture
-----------------

.. autoclass:: divi.qprog.ensemble.ProgramEnsemble
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: divi.qprog.ensemble.BatchMode
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: divi.qprog.ensemble.BatchConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

**Usage:**

.. code-block:: python

   from divi.qprog.ensemble import ProgramEnsemble
   from divi.qprog import VQE, QAOA, BatchConfig, BatchMode
   from divi.backends import ParallelSimulator

   class MyEnsemble(ProgramEnsemble):
       def create_programs(self):
           super().create_programs()  # Required: initializes internal state
           # Create multiple quantum programs
           self._programs = {
               "program_1": VQE(...),
               "program_2": QAOA(...),
               "program_3": VQE(...),
           }

       def aggregate_results(self):
           super().aggregate_results()
           # Collect results from all programs
           ...

   # Run program ensemble: create programs first, then execute
   ensemble = MyEnsemble(backend=ParallelSimulator())
   ensemble.create_programs()
   ensemble.run(blocking=True)

   # Disable circuit batching for local simulators
   ensemble.run(blocking=True, batch_config=BatchConfig(mode=BatchMode.OFF))

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
