Reporting
=========

The ``divi.reporting`` module provides comprehensive logging, progress reporting, and visualization functionality for quantum program execution.

Logging System
--------------

.. autofunction:: divi.reporting.enable_logging

.. autofunction:: divi.reporting.disable_logging

Global Logging Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configures logging for the entire Divi package with appropriate levels and formatters:

**Usage:**

.. code-block:: python

   from divi.reporting import enable_logging, disable_logging

   # Enable logging (called automatically on import)
   enable_logging()

   # Disable logging if needed
   disable_logging()

Advanced Features
-----------------

.. warning::
   **Developer-Facing Features**: The progress bar and progress reporting systems are intended for advanced users and developers who need custom logging and progress reporting. Most users will not need to interact with these features directly.

Progress Bar System
~~~~~~~~~~~~~~~~~~~

Rich Progress Bar Creation
^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``make_progress_bar`` function creates a customized Rich progress bar specifically designed for quantum program execution tracking:

.. autofunction:: divi.reporting.make_progress_bar

**Features:**

- **Custom Columns**: Job name, progress bar, completion status, elapsed time
- **Spinner Animation**: Conditional spinner that stops when job completes
- **Status Indicators**: Visual status with emojis (Success ✅, Failed ❌, etc.)
- **Jupyter Support**: Automatically adapts refresh behavior for notebook environments
- **Job Polling**: Shows polling attempts and job status for service-based backends

Progress Reporting System
^^^^^^^^^^^^^^^^^^^^^^^^^

Divi includes a sophisticated progress reporting system that provides real-time feedback during long-running quantum computations. The system supports both console logging and queue-based progress updates for integration with external monitoring systems.

Logging Progress Reporter
~~~~~~~~~~~~~~~~~~~~~~~~~

Console-based progress reporter that provides formatted output during quantum program execution:

.. autoclass:: divi.reporting.LoggingProgressReporter
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

Queue Progress Reporter
~~~~~~~~~~~~~~~~~~~~~~~

Thread-safe progress reporter that sends updates through a queue for external monitoring:

.. autoclass:: divi.reporting.QueueProgressReporter
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automodule:: divi.reporting
   :members:
   :undoc-members:
   :show-inheritance:
