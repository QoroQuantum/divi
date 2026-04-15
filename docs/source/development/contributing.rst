Contributing
=============

The full contributing guide — setup, code style, testing, commit conventions,
and pull request process — lives in
`CONTRIBUTING.md <https://github.com/QoroQuantum/divi/blob/main/CONTRIBUTING.md>`_
at the repository root. The highlights:

.. code-block:: bash

   git clone https://github.com/QoroQuantum/divi.git
   cd divi
   uv sync                # installs dev, testing, and docs groups
   pre-commit install

Run the test suite with ``uv run pytest -n auto``. Commit messages follow
the `Conventional Commits <https://www.conventionalcommits.org/>`_ specification
(enforced by the commit-msg hook).

.. seealso::

   :doc:`testing` for pytest conventions and markers, :doc:`building_docs`
   for the Sphinx/Makefile workflow.
