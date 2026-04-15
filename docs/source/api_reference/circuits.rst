Circuits
========

The ``divi.circuits`` module provides circuit abstractions for quantum program
generation, execution, and error mitigation.

.. warning::
   **Developer-Facing Classes**: The core circuit class ``MetaCircuit`` is
   intended for advanced users and developers. Most users should interact with
   circuits through higher-level APIs in the ``divi.qprog`` module.

Core
----

.. automodapi:: divi.circuits
   :no-heading:
   :no-inheritance-diagram:
   :no-inherited-members:
   :include-all-objects:

Error Mitigation Protocols
--------------------------

Divi provides quantum error mitigation (QEM) capabilities to improve the
accuracy of quantum computations in the presence of noise. All protocols
inherit from :class:`~divi.circuits.qem.QEMProtocol`.

.. automodapi:: divi.circuits.qem
   :no-heading:
   :no-inheritance-diagram:
   :no-inherited-members:
   :include-all-objects:

Quantum Enhanced Pauli Propagation (QuEPP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodapi:: divi.circuits.quepp
   :no-heading:
   :no-inheritance-diagram:
   :no-inherited-members:
   :include-all-objects:
