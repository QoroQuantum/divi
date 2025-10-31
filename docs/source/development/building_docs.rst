Building Documentation
=======================

This guide explains how to build and maintain the Divi documentation.

Prerequisites
-------------

Make sure you have the documentation dependencies installed:

.. code-block:: bash

   poetry install --with docs

Building the Documentation
--------------------------

**Standard Build**

To build the HTML documentation:

.. code-block:: bash

   cd docs
   make build

The built documentation will be available in ``docs/build/html/``.

**Live Development Server** (Recommended for Development)

For active documentation development, use the live reload server:

.. code-block:: bash

   cd docs
   make dev

This will:
- Automatically rebuild when files change
- Serve documentation at ``http://localhost:8000``
- Provide live reloading for faster iteration

**Serving Built Documentation**

To serve already-built documentation:

.. code-block:: bash

   cd docs
   make serve

Auto-generating API Documentation
---------------------------------

The API documentation is automatically generated from docstrings using `Sphinx's autodoc extension <https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html>`_. When you add or modify docstrings in your code, the documentation will be updated the next time you build.

Key Files
---------

- ``docs/source/conf.py`` - Sphinx configuration
- ``docs/source/index.rst`` - Main documentation index
- ``docs/source/api_reference/`` - Auto-generated API docs
- ``docs/Makefile`` - Build commands

Writing Good Docstrings
-----------------------

Follow these guidelines for docstrings that work well with Sphinx:

1. **Use Google-style docstrings** (recommended) or NumPy-style
2. **Include type hints** - Sphinx will automatically link them
3. **Document all public methods and classes**
4. **Include examples** where helpful

Example:

.. code-block:: python

   def optimize_circuit(self, circuit: Circuit, method: str = "default") -> Circuit:
       """Optimize a quantum circuit using the specified method.

       Args:
           circuit: The quantum circuit to optimize
           method: Optimization method to use

       Returns:
           The optimized circuit

       Raises:
           ValueError: If the method is not supported
       """
       pass

**Additional Makefile Commands**

The documentation Makefile provides several useful commands:

.. code-block:: bash

   cd docs
   make help          # Show all available commands
   make clean         # Remove all build files
   make test          # Run all quality checks (spelling, linkcheck, coverage)
   make spelling      # Check for spelling errors
   make linkcheck     # Check for broken links
   make coverage      # Run a documentation coverage check
   make open          # Open the built documentation in your browser
   make install       # Install documentation dependencies
