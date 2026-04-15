Reporting
=========

The ``divi.reporting`` module provides logging, progress reporting, and
visualization functionality for quantum program execution.

Usage
-----

Logging is enabled automatically on import; call
:func:`~divi.reporting.disable_logging` to silence it.

.. code-block:: python

   from divi.reporting import enable_logging, disable_logging

   # Enable logging (called automatically on import)
   enable_logging()

   # Disable logging if needed
   disable_logging()

Advanced Features
-----------------

.. warning::
   **Developer-Facing Features**: The progress bar and progress reporting systems are
   intended for advanced users and developers who need custom logging and progress
   reporting. Most users will not need to interact with these features directly.

:func:`~divi.reporting.make_progress_bar` creates a customized Rich progress bar
designed for quantum program execution tracking, with custom columns (job name,
progress bar, status, elapsed time), a conditional spinner, emoji status
indicators, Jupyter adaptation, and job-polling support for service-based
backends.

Divi also includes a progress reporting system for real-time feedback during
long-running quantum computations. The system supports both console logging
(:class:`~divi.reporting.LoggingProgressReporter`) and queue-based progress
updates (:class:`~divi.reporting.QueueProgressReporter`) for integration with
external monitoring systems.

.. automodapi:: divi.reporting
   :no-heading:
   :no-inheritance-diagram:
   :no-inherited-members:
   :include-all-objects:
