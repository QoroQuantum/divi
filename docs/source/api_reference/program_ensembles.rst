Program Ensembles
==================

The ``divi.qprog.ensemble`` module coordinates parallel execution of multiple
quantum programs with automatic circuit batching and progress tracking.

Core Architecture
-----------------

:class:`~divi.qprog.ensemble.ProgramEnsemble` is the abstract base shared by every
workflow class. :class:`~divi.qprog.ensemble.BatchConfig` and
:class:`~divi.qprog.ensemble.BatchMode` configure how circuits are grouped for
execution.

.. automodapi:: divi.qprog.ensemble
   :no-heading:
   :no-inheritance-diagram:
   :no-inherited-members:
   :include-all-objects:

Workflows
---------

Concrete workflow classes build on :class:`~divi.qprog.ensemble.ProgramEnsemble`.
:class:`~divi.qprog.workflows.VQEHyperparameterSweep` orchestrates parameterized VQE
runs over a grid of inputs; :class:`~divi.qprog.workflows.PartitioningProgramEnsemble`
decomposes a large graph problem into solvable sub-problems;
:class:`~divi.qprog.workflows.TimeEvolutionTrajectory` runs a sequence of time-evolution
steps to build a trajectory.

.. automodapi:: divi.qprog.workflows
   :no-heading:
   :no-inheritance-diagram:
   :no-inherited-members:
   :include-all-objects:

Partitioning Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~divi.qprog.problems.GraphPartitioningConfig` parameterizes how
:class:`~divi.qprog.workflows.PartitioningProgramEnsemble` splits a graph; it lives in
``divi.qprog.problems`` and is documented on the
:doc:`qprog/problems` reference page.
