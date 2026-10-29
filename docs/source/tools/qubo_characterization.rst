QUBO Characterisation Service
=============================

The Qoro **QUBO Characterisation Service** analyses a QUBO or HUBO. It
compares QAOA against a classical baseline on the same problem, states whether
and how to run QAOA, and returns optimised angles. The analysis is classical
and does not use quantum hardware.

Quick Start
-----------

The helper creates a cloud job, waits for it to finish, and returns a
:class:`~divi.backends.characterization.CharacterizationResult`.

.. skip: next

.. code-block:: python

   import numpy as np

   from divi.backends import QoroService
   from divi.backends.characterization import (
       CharacterizationOptions,
       characterize_and_validate,
   )
   from divi.qprog.problems import BinaryOptimizationProblem

   problem = BinaryOptimizationProblem(np.array([[-1, 2], [0, -1]]))

   result = characterize_and_validate(
       problem,
       reference_states=["01", "10"],
       service=QoroService(),
       options=CharacterizationOptions(ansatz={"layers": 2}),
   )

   print(result)  # the summary and every recommendation
   result         # in a notebook: the full HTML report

Set ``QORO_API_KEY`` in the environment or a ``.env`` file first.

``reference_states`` is required and must contain at least one binary
candidate with one bit per problem variable. The report covers them; they are
not constraints, do not configure a warm start, and do not need to be proven
optima.

The service minimises every submitted objective: reported energies are costs,
and lower is better. One problem variable corresponds to one qubit.

Reading the Result
------------------

``result.summary`` is one sentence on whether and how to run QAOA, and
``result.recommendations`` lists the recommendations behind it. In a notebook,
the result renders the full HTML report.

The result exposes the report's sections as dictionaries, as the service sends
them: ``problem``, ``classical``, ``qaoa``, ``angles``, ``references``,
``limits``, and, when the request asks for them, ``constraints`` and
``penalty``. ``result.report`` holds the whole report.

Shortcuts read individual values:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Shortcut
     - Meaning
   * - ``recommended_layers``
     - The recommended QAOA layer count.
   * - ``gap_closed``
     - Share of the gap from random guessing to the optimum that QAOA's mean
       energy closes at the requested layers: 0 is no better than guessing,
       1 is the optimum.
   * - ``recommended_lambda``
     - The recommended penalty weight, for a problem with a penalty.
   * - ``qaoa_initial_params(layers)``
     - The angles at a layer count (default: the recommended one), in the
       layout :class:`~divi.qprog.algorithms.QAOA`'s ``run`` expects.

Starting QAOA from the service's angles:

.. skip: next

.. code-block:: python

   from divi.qprog import QAOA

   qaoa = QAOA(problem, n_layers=result.recommended_layers)
   qaoa.run(initial_params=result.qaoa_initial_params())

Each recommendation is a dict with ``level`` (``"warning"``, ``"action"`` or
``"info"``, in that order), ``topic`` and plain ``text``:

.. skip: next

.. code-block:: python

   for rec in result.recommendations:
       if rec["level"] == "action":
           print(f'{rec["topic"]}: {rec["text"]}')

Configuring the Run
-------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Purpose
   * - ``ansatz``
     - ``layers`` (the requested layer count) and ``mixer``: ``"x"``,
       ``"xy"`` (Hamming-weight preserving) or ``"I"`` (no mixer).
   * - ``constraints``
     - Constraint descriptors for feasibility diagnostics; they do not
       modify the QUBO.
   * - ``n_qubits``
     - Declares the full width when some variables do not appear in any term.

Divi rejects unknown keys and malformed values before creating a job. See
:class:`~divi.backends.characterization.CharacterizationOptions` for the
complete schema.

Penalties and Constraints
-------------------------

When the problem is built with a penalty, either by passing
:class:`~divi.qprog.problems.LinearConstraint`\ s through ``constraints=``
(see :ref:`linear-constraints`) or with ``penalty=``, divi sends its cost and
penalty separately and the report gains a ``penalty`` section: the penalty
weight thresholds, a recommended weight, and the weight the submitted QUBO
itself uses.

.. skip: next

.. code-block:: python

   problem = BinaryOptimizationProblem(
       cost_terms,
       penalty=penalty_terms,
       penalty_weight=10.0,
   )
   options = CharacterizationOptions(
       constraints=[
           {"type": "max_cardinality", "bound": 3},
           {"type": "inequality", "bound": 10, "weights": {0: 4, 1: 5, 2: 7}},
       ],
   )

Supported constraint types are ``max_cardinality``, ``min_cardinality``,
``eq_cardinality``, ``inequality``, and ``equality``. Weighted constraints use
``weights={qubit_index: weight}``; cardinality constraints may restrict their
scope with ``qubits``.

Job Lifecycle and Recovery
--------------------------

Jobs move through ``PENDING``, ``RUNNING``, and a terminal
``COMPLETED``, ``FAILED``, ``CANCELLED``, ``INSUFFICIENT_CREDITS``, or
``TIMED_OUT`` state. Submission only queues the job. The convenience helper
then polls until the job is terminal, showing progress, and fetches the report.
The service's ``polling_interval`` sets the poll rate and ``max_retries`` caps
the polls; it is unbounded by default.

``Ctrl-C`` stops the wait without cancelling the job. Its id is logged when the
job is created, so you can fetch it later.

Initialisation and submission are not retried because both mutate server
state. After job creation, an ambiguous submission, polling (including
exhausting ``max_retries``), or result-fetch failure raises
:exc:`~divi.exceptions.CharacterizationSubmitError`. It preserves the existing
``job_id`` and identifies the failed ``phase``.

:func:`~divi.backends.characterization.get_characterization_result` fetches
that job by id:

.. skip: next

.. code-block:: python

   from divi.backends import QoroService
   from divi.backends.characterization import get_characterization_result
   from divi.exceptions import CharacterizationSubmitError

   try:
       result = characterize_and_validate(
           problem,
           reference_states=reference_states,
           service=QoroService(),
       )
   except CharacterizationSubmitError as exc:
       result = get_characterization_result(exc.job_id, service=QoroService())

Fetching waits through intermediate states and costs no credits.
``FAILED`` raises :exc:`~divi.exceptions.CharacterizationFailedError`;
``CANCELLED`` raises :exc:`~divi.exceptions.ExecutionCancelledError`;
``INSUFFICIENT_CREDITS`` raises :exc:`~divi.backends.InsufficientCreditsError`,
including when the account is rejected at submission; ``TIMED_OUT`` raises
:exc:`~divi.backends.JobTimedOutError`.
A completed job without a report also raises.

Cost
----

Characterisation is priced by variable count. Failed jobs are not billed, and
fetching an existing result is free. See the
`dashboard <https://dash.qoroquantum.net/>`_ for current tiers.

See Also
--------

* ``tutorials/backends/characterize_maxcut_qubo.py`` for an end-to-end
  MaxCut example.
* :class:`~divi.backends.characterization.CharacterizationResult` for the
  result's sections and shortcuts.
* :class:`~divi.backends.characterization.CharacterizationOptions` for the
  complete configuration schema.
* :doc:`../algorithms/combinatorial_optimization_qaoa_pce` for the QAOA
  workflow.
