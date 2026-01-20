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

QAOA Algorithm
~~~~~~~~~~~~~~

.. autoclass:: divi.qprog.algorithms.QAOA
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: cost_hamiltonian, mixer_hamiltonian

.. autoattribute:: divi.qprog.algorithms.QAOA.cost_hamiltonian
   :no-index:

.. autoattribute:: divi.qprog.algorithms.QAOA.mixer_hamiltonian
   :no-index:

PCE Algorithm
~~~~~~~~~~~~~

.. autoclass:: divi.qprog.algorithms.PCE
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :exclude-members: cost_hamiltonian

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

Graph Problem Types
~~~~~~~~~~~~~~~~~~~

.. autoclass:: divi.qprog.algorithms.GraphProblem
   :members:
   :undoc-members:
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

Workflows
---------

Divi provides workflow classes for managing large-scale quantum computations, including hyperparameter sweeps and graph partitioning.

.. automodule:: divi.qprog.workflows
   :members:
   :undoc-members:
   :show-inheritance:

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

Graph Partitioning QAOA
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: divi.qprog.workflows.GraphPartitioningQAOA
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

.. autoclass:: divi.qprog.workflows.PartitioningConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

QUBO Partitioning QAOA
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: divi.qprog.workflows.QUBOPartitioningQAOA
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
