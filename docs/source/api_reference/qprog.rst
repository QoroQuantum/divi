Quantum Programs (qprog)
=========================

The ``divi.qprog`` module contains the core quantum programming abstractions for building and executing quantum algorithms.

Overview
--------

The quantum programming module provides a high-level interface for quantum algorithm development, supporting both single-instance problems and large-scale hyperparameter sweeps. At its core is the :class:`QuantumProgram` abstract base class that defines the common interface for all quantum algorithms.

Core Architecture
-----------------

.. automodule:: divi.qprog
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: BatchConfig, BatchMode, ProgramEnsemble

Core Classes
------------

.. autoclass:: divi.qprog.QuantumProgram
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __init__

.. autoclass:: divi.qprog.VariationalQuantumAlgorithm
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Early Stopping
~~~~~~~~~~~~~~

.. autoclass:: divi.qprog.EarlyStopping
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: divi.qprog.early_stopping.StopReason
   :members:
   :undoc-members:
   :show-inheritance:

Algorithms
----------

Divi provides implementations of popular quantum algorithms with a focus on scalability and ease of use.

.. automodule:: divi.qprog.algorithms
   :members:
   :undoc-members:
   :show-inheritance:

VQE Algorithm
~~~~~~~~~~~~~

.. autoclass:: divi.qprog.algorithms.VQE
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: cost_hamiltonian

.. autoattribute:: divi.qprog.algorithms.VQE.cost_hamiltonian
   :no-index:

Problem Classes
~~~~~~~~~~~~~~~

QAOA accepts a :class:`Problem` instance that encapsulates the optimisation
objective, mixer, initial state, and solution decoding.  Divi provides concrete
classes for common graph and binary optimisation problems.

.. autoclass:: divi.qprog.problems.QAOAProblem
   :members:
   :show-inheritance:

.. autoclass:: divi.qprog.problems.MaxCutProblem
   :show-inheritance:
   :special-members: __init__
   :no-index:

.. autoclass:: divi.qprog.problems.MaxCliqueProblem
   :show-inheritance:
   :special-members: __init__
   :no-index:

.. autoclass:: divi.qprog.problems.MaxIndependentSetProblem
   :show-inheritance:
   :special-members: __init__
   :no-index:

.. autoclass:: divi.qprog.problems.MinVertexCoverProblem
   :show-inheritance:
   :special-members: __init__
   :no-index:

.. autoclass:: divi.qprog.problems.MaxWeightCycleProblem
   :show-inheritance:
   :special-members: __init__
   :no-index:

.. autoclass:: divi.qprog.problems.MaxWeightMatchingProblem
   :members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

.. autoclass:: divi.qprog.problems.BinaryOptimizationProblem
   :members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

.. autoclass:: divi.qprog.problems.TSPProblem
   :members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

.. autoclass:: divi.qprog.problems.CVRPProblem
   :members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

QAOA Algorithm
~~~~~~~~~~~~~~

.. autoclass:: divi.qprog.algorithms.QAOA
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: cost_hamiltonian

.. autoattribute:: divi.qprog.algorithms.QAOA.cost_hamiltonian
   :no-index:

Iterative QAOA
~~~~~~~~~~~~~~

.. autoclass:: divi.qprog.algorithms.IterativeQAOA
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: divi.qprog.algorithms.InterpolationStrategy
   :members:
   :undoc-members:
   :no-index:

.. autofunction:: divi.qprog.algorithms._iterative_qaoa.interpolate_qaoa_params

Trotterization Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~

QAOA uses a trotterization strategy to evolve the cost Hamiltonian. The default is
:class:`~divi.qprog.ExactTrotterization`; :class:`~divi.qprog.QDrift` provides
randomized sampling for shallower circuits at the cost of more circuits per iteration.

.. autoclass:: divi.hamiltonians.TrotterizationStrategy
   :members:
   :undoc-members:
   :no-index:

.. autoclass:: divi.hamiltonians.ExactTrotterization
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: divi.hamiltonians.QDrift
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

PCE Algorithm
~~~~~~~~~~~~~

.. autoclass:: divi.qprog.algorithms.PCE
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: cost_hamiltonian

Time Evolution
~~~~~~~~~~~~~~

.. autoclass:: divi.qprog.algorithms.TimeEvolution
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Custom VQA
~~~~~~~~~~

.. autoclass:: divi.qprog.algorithms.CustomVQA
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: cost_hamiltonian, param_shape

.. autoattribute:: divi.qprog.algorithms.CustomVQA.cost_hamiltonian
   :no-index:

.. autoattribute:: divi.qprog.algorithms.CustomVQA.param_shape
   :no-index:

VQE Ansätze
~~~~~~~~~~~

.. autoclass:: divi.qprog.algorithms.HartreeFockAnsatz
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: divi.qprog.algorithms.UCCSDAnsatz
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: divi.qprog.algorithms.QAOAAnsatz
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: divi.qprog.algorithms.HardwareEfficientAnsatz
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: divi.qprog.algorithms.GenericLayerAnsatz
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: divi.qprog.algorithms.Ansatz
   :members:
   :undoc-members:
   :show-inheritance:

Optimizers
----------

Divi provides multiple optimization strategies for quantum algorithm parameter tuning, from classical gradient-based methods to quantum-inspired approaches.

.. automodule:: divi.qprog.optimizers
   :members:
   :undoc-members:
   :show-inheritance:

Scipy Optimizer
~~~~~~~~~~~~~~~

.. autoclass:: divi.qprog.optimizers.ScipyOptimizer
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
   :special-members: __init__

Monte Carlo Optimizer
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: divi.qprog.optimizers.MonteCarloOptimizer
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
   :special-members: __init__

GridSearchOptimizer
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: divi.qprog.optimizers.GridSearchOptimizer
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
   :special-members: __init__

Workflows
---------

Divi provides workflow classes for managing large-scale quantum computations, including hyperparameter sweeps and graph partitioning.

.. automodule:: divi.qprog.workflows
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: VQEHyperparameterSweep, MoleculeTransformer, GraphPartitioningConfig, PartitioningProgramEnsemble

VQE Hyperparameter Sweep
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: divi.qprog.workflows.VQEHyperparameterSweep
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

.. autoclass:: divi.qprog.workflows.MoleculeTransformer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

Partitioning Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: divi.qprog.problems.GraphPartitioningConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

Checkpointing
-------------

Divi provides comprehensive checkpointing support for saving and resuming optimization state.

Checkpoint Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: divi.qprog.checkpointing.CheckpointConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Checkpoint Information
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: divi.qprog.checkpointing.CheckpointInfo
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Checkpoint Utilities
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: divi.qprog.checkpointing.resolve_checkpoint_path

.. autofunction:: divi.qprog.checkpointing.get_checkpoint_info

.. autofunction:: divi.qprog.checkpointing.list_checkpoints

.. autofunction:: divi.qprog.checkpointing.get_latest_checkpoint

.. autofunction:: divi.qprog.checkpointing.cleanup_old_checkpoints

Checkpoint Exceptions
~~~~~~~~~~~~~~~~~~~~~

.. autoexception:: divi.qprog.checkpointing.CheckpointError
   :show-inheritance:

.. autoexception:: divi.qprog.checkpointing.CheckpointNotFoundError
   :show-inheritance:

.. autoexception:: divi.qprog.checkpointing.CheckpointCorruptedError
   :show-inheritance:

Exceptions
----------

.. automodule:: divi.qprog.exceptions
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:
