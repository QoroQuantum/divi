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

.. autoclass:: divi.backends.JobStatus
   :members:
   :undoc-members:

.. autoclass:: divi.backends.JobType
   :members:
   :undoc-members:

.. automodule:: divi.backends
   :members:
   :undoc-members:
   :show-inheritance:
