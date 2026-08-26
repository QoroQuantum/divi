Installation
============

Divi's core install is deliberately small. Heavier dependencies that only a
subset of features needs are published as **optional extras**, so you only
download what your workflow actually uses.

Core Install
------------

If you have uv installed (recommended):

.. code-block:: bash

   uv add qoro-divi

Alternatively, with pip:

.. code-block:: bash

   pip install qoro-divi

Or from source:

.. code-block:: bash

   git clone https://github.com/QoroQuantum/divi.git
   cd divi
   uv sync

The core install covers local simulation with
:class:`~divi.backends.MaestroSimulator`, cloud execution via
:class:`~divi.backends.QoroService`, VQE and QAOA, program ensembles, circuit
pipelines, and every optimiser except
:class:`~divi.qprog.optimizers.PymooOptimizer`.

.. _optional-extras:

Optional Extras
---------------

.. list-table::
   :header-rows: 1
   :widths: 18 34 12 36

   * - Extra
     - Unlocks
     - Adds
     - Works without it
   * - ``aer``
     - :class:`~divi.backends.QiskitSimulator` — Qiskit Aer noise models and
       IBM fake-backend calibration data.
     - ~38 MB
     - :class:`~divi.backends.MaestroSimulator` (the default local simulator)
       and :class:`~divi.backends.QoroService`.
   * - ``qubo-decompose``
     - Partitioned QUBO solving via the D-Wave ``hybrid`` package — the
       ``decomposer`` argument and
       :meth:`~divi.qprog.problems.BinaryOptimizationProblem.decompose`,
       D-Wave's own ``EnergyImpactDecomposer`` and ``SplatComposer``, and
       Divi's :class:`~divi.qprog.problems.CommunityDecomposer`.
     - ~145 MB
     - Unpartitioned QUBO and HUBO problems — cost Hamiltonian, mixer, and
       solving without partitioning.
   * - ``chem``
     - Molecule inputs via PySCF, OpenFermion Hamiltonians,
       :class:`~divi.qprog.algorithms.UCCSDAnsatz`, and
       :class:`~divi.qprog.workflows.LASSQD`.
     - ~322 MB
     - VQE from a ``SparsePauliOp`` or a Pauli-string dictionary, and every
       non-chemistry ansatz.
   * - ``ai``
     - The ``divi-ai`` offline documentation assistant. See :doc:`tools/divi_ai`.
     - ~144 MB
     - Everything except ``divi-ai``.
   * - ``jupyter``
     - Notebook progress widgets.
     - ~72 MB
     - All reporting outside notebooks.

Sizes are approximate installed-on-disk figures for the extra and the
dependencies it alone pulls in; they vary by platform and Python version.

Install one extra, or several at once:

.. code-block:: bash

   pip install "qoro-divi[aer]"
   pip install "qoro-divi[aer,chem]"

   uv add "qoro-divi[aer,chem]"

To pull in everything:

.. code-block:: bash

   pip install "qoro-divi[all]"

.. note::

   ``all`` includes ``ai``, which depends on ``llama-cpp-python``. On platforms
   with no prebuilt wheel it is compiled from source and needs a C++ toolchain.
   If that is a problem, name the extras you want individually instead.

Working from a source checkout with uv, the equivalents are:

.. code-block:: bash

   uv sync --extra aer --extra chem
   uv sync --all-extras
   uv sync --all-extras --no-extra ai

.. note::

   ``--no-extra`` only takes effect alongside ``--all-extras``.

Missing an Extra
----------------

Features behind an extra are imported lazily, so a missing dependency surfaces
when you first reach for the feature rather than at ``import divi``. The error
names both the feature and the command that fixes it:

.. code-block:: text

   ImportError: QiskitSimulator requires the 'aer' extra; install it with
   `pip install qoro-divi[aer]`. Divi's default simulator, MaestroSimulator,
   is included in the core install.

Nightly Builds
--------------

Nightly development builds are published daily from ``main``. To install the
latest nightly:

.. code-block:: bash

   pip install qoro-divi --pre

Or pin a specific nightly by date:

.. code-block:: bash

   pip install qoro-divi==0.8.0.dev20260305

.. note::

   Nightly builds may contain unstable or experimental features.
   For production use, stick with the stable release (``pip install qoro-divi``).
