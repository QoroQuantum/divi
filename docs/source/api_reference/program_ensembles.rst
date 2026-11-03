Program Ensembles
==================

The ``divi.qprog.ensembles`` package coordinates parallel execution of multiple
quantum programs with automatic circuit batching and progress tracking.

Core Architecture
-----------------

:class:`~divi.qprog.ensembles.ProgramEnsemble` is the abstract base shared by every
ensemble class. :class:`~divi.qprog.ensembles.BatchConfig` and
:class:`~divi.qprog.ensembles.BatchMode` configure how circuits are grouped for
execution.

Concrete Ensembles
------------------

:class:`~divi.qprog.ensembles.VQEHyperparameterSweep` orchestrates parameterised VQE
runs over a grid of inputs; :class:`~divi.qprog.ensembles.PartitioningProgramEnsemble`
decomposes a large graph, QUBO or portfolio problem into solvable sub-problems;
:class:`~divi.qprog.ensembles.TimeEvolutionTrajectory` runs a sequence of time-evolution
steps to build a trajectory; :class:`~divi.qprog.ensembles.lassqd.LASSQD` computes
molecular energies and excited states fragment by fragment, with its
configuration and result types on the :doc:`lassqd` page.

Aggregation Strategies
----------------------

:class:`~divi.qprog.ensembles.PartitioningProgramEnsemble` combines per-partition
candidates into global solutions using an
:class:`~divi.qprog.ensembles.AggregationStrategy`.
:class:`~divi.qprog.ensembles.BeamSearchStrategy` (the default) explores candidates
left-to-right across partitions; :class:`~divi.qprog.ensembles.HierarchicalStrategy` solves
groups of partitions independently and merges them pairwise.

Partitioning Configuration
--------------------------

:class:`~divi.qprog.problems.GraphPartitioningConfig` and
:class:`~divi.qprog.problems.QUBOPartitioningConfig` parameterise how
:class:`~divi.qprog.ensembles.PartitioningProgramEnsemble` splits graph problems
and QUBO or portfolio problems respectively; they live in ``divi.qprog.problems``
and are documented on the
:doc:`qprog/problems` reference page.

Reference
---------

.. automodapi:: divi.qprog.ensembles
   :no-heading:
   :no-inheritance-diagram:
   :no-inherited-members:
   :include-all-objects:
