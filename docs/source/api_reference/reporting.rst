Reporting
=========

The public ``divi.reporting`` API controls Divi's optional Rich logging
handler. Progress rendering is managed internally by each operation.

Usage
-----

Importing Divi does not configure logging. Call
:func:`~divi.reporting.enable_logging` to add Divi's Rich handler and
:func:`~divi.reporting.disable_logging` to remove it again. These functions
leave application-owned handlers unchanged. When the existing effective
threshold would suppress the requested level, ``enable_logging()`` temporarily
lowers the ``divi`` logger threshold; ``disable_logging()`` restores the old
explicit level unless the application changed it in the meantime.
These logging controls do not select or suppress progress rendering. Set
``DIVI_DISABLE_PROGRESS`` to ``1``, ``true``, ``yes``, or ``on`` to silence
progress output for both standalone programs and ensembles; ordinary Divi log
messages are unaffected.

.. code-block:: python

   from divi.reporting import disable_logging, enable_logging

   # Opt in to Rich-formatted Divi logs.
   enable_logging()

   # Remove only the handler installed above.
   disable_logging()

.. automodapi:: divi.reporting
   :no-heading:
   :no-inheritance-diagram:
   :no-inherited-members:
   :include-all-objects:
