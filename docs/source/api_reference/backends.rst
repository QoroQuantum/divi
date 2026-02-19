Backends
========

The ``divi.backends`` module provides interfaces for running quantum circuits on different backends, from local simulators to cloud-based quantum hardware.

Backend Architecture
--------------------

All backends implement the :class:`CircuitRunner` interface, providing a consistent API for circuit execution. This abstraction allows switching between different execution environments without changing quantum program code.

.. autoclass:: divi.backends.CircuitRunner
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Execution Results
-----------------

All :meth:`CircuitRunner.submit_circuits` methods return an :class:`ExecutionResult` object, which provides a unified interface for handling both synchronous and asynchronous backend responses.

.. autoclass:: divi.backends.ExecutionResult
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

Core Backend Classes
--------------------

.. autoclass:: divi.backends.ParallelSimulator
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: divi.backends.QoroService
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

Job Management
--------------

.. autoclass:: divi.backends.JobConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

.. autoclass:: divi.backends.JobStatus
   :members:
   :undoc-members:

.. autoclass:: divi.backends.JobType
   :members:
   :undoc-members:

Execution Configuration
-----------------------

.. autoclass:: divi.backends.ExecutionConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

.. autoclass:: divi.backends.Simulator
   :members:
   :undoc-members:

.. autoclass:: divi.backends.SimulationMethod
   :members:
   :undoc-members:

.. automodule:: divi.backends
   :members:
   :undoc-members:
   :show-inheritance:
