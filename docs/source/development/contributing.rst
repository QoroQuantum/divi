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

Optional Dependencies
---------------------

Packages from the optional extras (``chem``, ``pennylane``, ``aer``,
``qubo-decompose``) are never imported at module level. Go through
``divi._optional`` instead:

- **Using the package:** call ``import_optional`` where the package is needed
  and work with the module it returns. If the extra is missing, it raises an
  ``ImportError`` naming the extra to install; ``hint=`` appends extra advice.
- **Recognising an input:** call ``module_if_imported`` and check the input with
  ``isinstance``. It reads ``sys.modules`` and never imports, since an instance
  of an optional package's class can only exist once that package is loaded.
- **Type hints:** import under ``if TYPE_CHECKING:``.
- **Lazy public exports:** a package ``__getattr__`` calls ``import_optional``,
  then imports the internal module that depends on the package. Code behind
  that gate, or behind an ``isinstance`` check on the package's own objects,
  uses ordinary imports.

.. code-block:: python

   from divi._optional import import_optional, module_if_imported


   def build_mean_field(mol):
       scf = import_optional("pyscf.scf", extra="chem", capability="Building a mean field")
       return scf.RHF(mol)


   def is_qnode(candidate):
       qp = module_if_imported("pennylane")
       return qp is not None and isinstance(candidate, qp.QNode)

Avoid module-level ``try``/``except ImportError`` fallbacks and hand-written
"requires the extra" messages.

.. seealso::

   :doc:`testing` for pytest conventions and markers, :doc:`building_docs`
   for the Sphinx/Makefile workflow.
