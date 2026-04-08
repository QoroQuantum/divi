Contributing
=============

Thank you for your interest in contributing to Divi! This guide will help you get started with our development workflow.

Getting Started
---------------

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:

   .. code-block:: bash

      git clone https://github.com/your-username/divi.git
      cd divi

3. **Install development dependencies**:

   .. code-block:: bash

      uv sync --group dev --group testing --group docs

4. **Set up pre-commit hooks** (recommended):

   .. code-block:: bash

      uv run pre-commit install

5. **Create a new branch** for your changes:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

Development Environment
-----------------------

Divi uses `uv <https://docs.astral.sh/uv/>`_ for dependency management with several groups:

- **dev**: Core development tools (`Black <https://github.com/psf/black>`_, `isort <https://github.com/pycqa/isort>`_, `pre-commit <https://pre-commit.com/>`_)
- **testing**: Testing framework and utilities (`pytest <https://pytest.org/>`_, `pytest-mock <https://pytest-mock.readthedocs.io/>`_)
- **docs**: Documentation building (`Sphinx <https://www.sphinx-doc.org/>`_, `nbsphinx <https://nbsphinx.readthedocs.io/>`_)
- **jupyter**: Jupyter notebook support (`ipywidgets <https://ipywidgets.readthedocs.io/>`_)

Install specific groups as needed:

.. code-block:: bash

   uv sync --group dev --group testing  # For core development
   uv sync --group docs                # For documentation work

Code Quality & Style
--------------------

We maintain high code quality through automated tools:

Pre-commit Hooks (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pre-commit hooks automatically run quality checks before each commit:

- **Black**: Code formatting (`Black <https://github.com/psf/black>`_)
- **isort**: Import sorting (`isort <https://github.com/pycqa/isort>`_ with Black-compatible profile)
- **License Headers**: Automatic SPDX license header insertion
- **File Validation**: JSON, YAML, TOML syntax checking
- **Line Endings**: CRLF/tab enforcement

Manual Quality Checks
~~~~~~~~~~~~~~~~~~~~~

If you prefer manual control:

   .. code-block:: bash

      uv run black .
      uv run isort .
      uv run pytest

License Compliance
------------------

All Python files must include SPDX license headers. The pre-commit hook automatically adds them, but you can also add manually:

.. code-block:: python

   # SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
   #
   # SPDX-License-Identifier: Apache-2.0

Pull Request Process
--------------------

1. **Write tests** for your changes (see :doc:`testing`)
2. **Update documentation** if needed (see :doc:`building_docs`)
3. **Ensure pre-commit hooks pass** (or run quality checks manually)
4. **Run the test suite**: ``uv run pytest``
5. **Submit a pull request** with a clear description

**Pro Tips:**
- Use descriptive commit messages
- Reference related issues in your PR description
- Keep PRs focused on a single feature/fix
- Ensure all CI checks pass

For more details, see our `Contributing Guidelines <https://github.com/QoroQuantum/divi/blob/main/CONTRIBUTING.md>`_.
