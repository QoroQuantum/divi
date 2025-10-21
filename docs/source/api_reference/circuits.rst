Circuits
========

The ``divi.circuits`` module provides circuit abstractions for quantum program generation, execution, and error mitigation.

Core Circuit Classes
--------------------

.. warning::
   **Developer-Facing Classes**: The core circuit classes (``Circuit`` and ``MetaCircuit``) are intended for advanced users and developers. Most users should interact with circuits through higher-level APIs in the ``divi.qprog`` module.

.. autoclass:: divi.circuits.Circuit
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

.. autoclass:: divi.circuits.MetaCircuit
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

QASM Integration
----------------

.. automodule:: divi.circuits.qasm
   :members:
   :undoc-members:
   :show-inheritance:

QASM Generation Function
~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: divi.circuits.qasm.to_openqasm
   :no-index:

Error Mitigation Protocols
--------------------------

Divi provides quantum error mitigation (QEM) capabilities to improve the accuracy of quantum computations in the presence of noise.

.. automodule:: divi.circuits.qem
   :members:
   :undoc-members:
   :show-inheritance:

QEM Protocol Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~

All error mitigation protocols in Divi inherit from the :class:`QEMProtocol` base class, providing a consistent interface for different mitigation techniques.

.. autoclass:: divi.circuits.qem.QEMProtocol
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
   :special-members: __init__

Zero Noise Extrapolation (ZNE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: divi.circuits.qem.ZNE
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
   :no-index:

No Mitigation Protocol
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: divi.circuits.qem._NoMitigation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: divi.circuits
   :members:
   :undoc-members:
   :show-inheritance:
