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

.. autoclass:: divi.qprog.algorithms.VQE
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: divi.qprog.algorithms.QAOA
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Graph Problem Types
~~~~~~~~~~~~~~~~~~~

.. autoclass:: divi.qprog.algorithms.GraphProblem
   :members:
   :undoc-members:
   :no-index:

VQE Ansatze
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

Exceptions
----------

.. automodule:: divi.qprog.exceptions
   :members:
   :undoc-members:
   :show-inheritance:
