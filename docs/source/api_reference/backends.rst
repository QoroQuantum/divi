Backends
========

The ``divi.backends`` module provides interfaces for running quantum circuits on
different backends, from local simulators to cloud-based quantum hardware.

All backends implement the :class:`~divi.backends.CircuitRunner` interface,
providing a consistent API for circuit execution. This abstraction allows
switching between different execution environments without changing quantum
program code.

All :meth:`~divi.backends.CircuitRunner.submit_circuits` methods return an
:class:`~divi.backends.ExecutionResult` object, which provides a unified
interface for handling both synchronous and asynchronous backend responses.

.. automodapi:: divi.backends
   :no-heading:
   :no-inheritance-diagram:
   :no-inherited-members:
   :include-all-objects:
