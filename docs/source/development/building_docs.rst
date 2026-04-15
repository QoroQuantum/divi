Building Documentation
=======================

This guide explains how to build and maintain the Divi documentation.

Prerequisites
-------------

Make sure you have the documentation dependencies installed:

.. code-block:: bash

   uv sync --group docs

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

.. skip: next

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

Documentation code snippets (Sybil)
-----------------------------------

Python examples in ``*.rst`` files are executed in CI (see the **Build Docs** workflow)
using `Sybil <https://sybil.readthedocs.io/>`_ and ``docs/source/conftest.py``. This
catches stale imports and API drift in the user guide.

**Run locally**

.. code-block:: bash

   cd docs
   make test-snippets

By default this runs **all** snippet examples (no early exit). To stop at the first
failure while debugging:

.. code-block:: bash

   make test-snippets-fast

or ``PYTEST_ADDOPTS=-x make test-snippets``.

**Authoring**

- Prefer **runnable** ``.. code-block:: python`` sections. Use ``.. skip: next`` only
  when the block is pseudocode, uses ``...`` placeholders, or needs symbols you do
  not want to wire up yet.
- Keep skipped blocks as ``python`` (not ``text``) for consistent highlighting unless
  the content is not Python at all.
- Shared setup for many pages belongs in ``docs/source/conftest.py`` (namespace
  injection, ceilings on iterations/shots, ``DocStubQoroService`` for cloud examples).
- For ``literalinclude`` from the test suite, point at files that are already run by
  ``pytest`` so examples stay single-sourced.

Pattern quick reference
^^^^^^^^^^^^^^^^^^^^^^^

**Runnable snippet** — Default. Use a normal ``.. code-block:: python`` that runs as
written in the doc test session.

**Invisible setup** — Use ``.. invisible-code-block: python`` (single colon after
``invisible-code-block``; double ``::`` breaks Sybil’s lexer) for shared state between
nearby examples when repeating setup would hurt readability. The code still runs in
Sybil; it is not shown in the built HTML.

**Skip marker** — Use ``.. skip: next`` only for intentionally non-runnable blocks
(pseudocode, placeholders, or examples that are too heavy for CI doc-test budgets).

Finding ``skip`` directives
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Line-number inventories go stale; search when auditing:

.. code-block:: shell

   rg '\.\. skip:' docs/source

Invisible snippets: when and how
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``.. invisible-code-block: python`` when:

- A section has several short examples that share the same objects (for example, one
  optimization run reused by several plotting calls).
- The shared setup is runnable and needed for tests, but repetitive in the rendered guide.
- You want to replace ``.. skip: next`` continuation fragments with executable docs.

Avoid invisible snippets when:

- The hidden code is long enough that readers need to see it to understand the example.
- A visible minimal setup would be clearer than hidden state.
- The setup is shared across many pages (prefer ``docs/source/conftest.py`` namespace
  injection instead).

``user_guide/visualization.rst`` uses invisible setup to define shared scan inputs once
and keep later examples short while still runnable.
