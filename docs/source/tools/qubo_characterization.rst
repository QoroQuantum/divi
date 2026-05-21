QUBO Characterization Service
=============================

The Qoro **QUBO Characterization Service** is a hosted diagnostic that
analyzes a QUBO or HUBO before — or instead of — running QAOA. It returns
a structural quality score, an optimal-parameter sweep, hardness metrics,
and actionable recommendations, all from a single short call.

.. note::

   Characterization runs on Qoro's hosted cloud service via
   :class:`~divi.backends.QoroService`. See :ref:`pricing` below for
   credit cost.

When to Use It
--------------

Characterization is useful whenever you would otherwise *guess* at a QAOA
setup. Common scenarios:

* **Before tuning a QAOA**, to find good initial ``γ``, ``β`` values via
  a server-side parameter sweep and skip a long optimizer warm-up.
* **When picking a penalty multiplier** for constrained problems —
  auto-tuning returns a recommended ``λ`` and flags whether the current
  value is well-tuned.
* **To decide whether a QUBO is worth running at all** — the quality
  score and hardness metrics surface formulations that are unlikely to
  succeed at shallow depth regardless of how long you optimize.
* **For HUBO problems**, where ``BinaryOptimizationProblem`` already
  accepts higher-order terms; characterization works on the same
  canonical form without extra conversion.

The diagnostic itself does **not** execute QAOA on hardware. It runs
classically on Qoro's cloud and completes within a few seconds.

Quick Start
-----------

The single-call entry point is
:func:`~divi.backends.characterize_and_validate`:

.. skip: next

.. code-block:: python

   import numpy as np
   from divi.backends import (
       CharacterizationOptions,
       QoroService,
       characterize_and_validate,
   )
   from divi.qprog.problems import BinaryOptimizationProblem

   # A small QUBO (any shape BinaryOptimizationProblem accepts is fine —
   # ndarray, sparse, BQM, HUBO dict).
   problem = BinaryOptimizationProblem(np.array([[-1, 2], [0, -1]]))

   result = characterize_and_validate(
       problem,
       target_states=["01", "10"],   # known/expected solutions, if any
       service=QoroService(),
       options=CharacterizationOptions(parameter_sweep=True),
   )

   result.display()           # rich console report
   print(result.quality_score)

Viewing the Report
------------------

A characterization result carries two views of the same data, chosen by
output medium:

* **Terminal** — ``result.display()`` prints a styled console report
  (panels, gauges, tables) using ``rich``. Use this from scripts or
  REPLs.
* **Jupyter** — evaluating ``result`` directly in a cell triggers
  ``_repr_html_`` and renders a server-styled HTML report inline. The
  HTML is fetched once when the result is created and cached on the
  object as the ``html`` attribute.

The HTML report is **self-contained** — inline CSS, no external assets
— so you can serialize ``result.html`` and email, log, or embed it as
a standalone artifact:

.. skip: next

.. code-block:: python

   Path("report.html").write_text(result.html)

If the HTML endpoint is unreachable at fetch time, the result is still
returned but ``result.html`` is an empty string and a warning is logged.
The structured fields (``quality_score``, ``recommendations``, etc.)
remain available either way.

What You Get Back
-----------------

:func:`~divi.backends.characterize_and_validate` returns a
:class:`~divi.backends.CharacterizationResult` with typed properties for
the most useful fields:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Property
     - What it means
   * - ``quality_score``
     - Composite metric (0–100) of how amenable the QUBO is to QAOA,
       derived from spectral and concentration features. A high
       ``quality_score`` indicates a well-conditioned formulation; it
       **does not predict the approximation ratio at any specific
       circuit depth**, and in particular says nothing about whether
       ``p=1`` will solve the problem.
   * - ``concentration_ratio``
     - Probability mass on target states relative to the uniform baseline.
       ``1.0`` is uniform; values around ``1.5×`` or below are flagged as
       too uniform and suggest deeper circuits.
   * - ``approximation_ratio``
     - The server's estimate of the approximation ratio at the returned
       ``best_parameters``. Comparable to your own QAOA's AR **only** at
       the same depth and ansatz configuration — comparing a ``p=1``
       sweep result against a ``p=3`` run, for example, is not
       meaningful.
   * - ``best_parameters``
     - The ``γ`` / ``β`` returned by the parameter sweep (if requested).
   * - ``feasibility_rate``
     - Fraction of sampled states satisfying all declared constraints.
   * - ``penalty_recommendation``, ``is_well_tuned``
     - For constrained problems: a recommended ``λ`` and whether the
       current value is well-tuned.
   * - ``hardness``
     - Static QUBO-structure metrics — ``difficulty``, ``spectral_gap``,
       ``condition_number``. Independent of any quantum run.
   * - ``recommendations``
     - Server-generated interpretive notes (see
       :ref:`reading-recommendations` below).

Every field is ``None`` when the corresponding analysis was not requested.

Configuring the Run
-------------------

Pass a :class:`~divi.backends.CharacterizationOptions` to control which
sub-analyses run. All fields are optional; the default-constructed object
runs a basic report.

.. skip: next

.. code-block:: python

   from divi.backends import CharacterizationOptions

   opts = CharacterizationOptions(
       parameter_sweep=True,             # sweep γ, β server-side
       sensitivity=True,                 # per-qubit fragility report
       auto_tune=True,                   # recommend a penalty λ
       ansatz={"mixer": "x", "layers": 1},
   )

Setting ``sensitivity=True`` adds a per-qubit fragility breakdown to the
result's ``sensitivity`` field, identifying which variable assignments
have the largest impact on the objective — useful for spotting brittle
encodings that small perturbations can flip.

One constructor rule is worth knowing up front:
``parameter_sweep=True`` is mutually exclusive with fixed ``gamma`` /
``beta`` — pick one mode. The constructor raises ``ValueError``
immediately if you set both, so the error surfaces locally before any
API call.

Penalty Tuning for Constrained Problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For QUBOs that encode constraints as quadratic penalties, you can split
the formulation into the cost-only and penalty-only halves and pass them
both. The service then evaluates whether the chosen multiplier dominates
the cost term enough to enforce constraints — and recommends an
adjustment when it does not.

.. skip: next

.. code-block:: python

   opts = CharacterizationOptions(
       cost_qubo=cost_problem,           # BinaryOptimizationProblem
       penalty_qubo=penalty_problem,     # BinaryOptimizationProblem
       constraints=[...],                # constraint descriptors
       auto_tune=True,
   )

   result = characterize_and_validate(
       problem, target_states=[...], service=QoroService(), options=opts
   )
   print(result.penalty_recommendation, result.is_well_tuned)

.. _reading-recommendations:

Reading Recommendations Programmatically
----------------------------------------

``result.recommendations`` is always a list (empty when no rules fired or
the job did not complete). Each entry is a dict with four keys:

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Key
     - Value
   * - ``level``
     - ``"info"``, ``"warn"``, or ``"action"``. ``action`` recommends a
       concrete change; ``warn`` flags a risk; ``info`` is contextual.
   * - ``metric``
     - Which report field drove the rule — one of ``"quality_score"``,
       ``"difficulty"``, ``"feasibility_rate"``, ``"penalty_tuning"``,
       ``"concentration_ratio"``.
   * - ``text``
     - Plain-text message, suitable for terminal or log output.
   * - ``html``
     - Same message with inline ``<strong>`` markup, used by the
       notebook renderer. ``text`` and ``html`` always carry the same
       content; choose by output medium.

Acting on them in code:

.. skip: next

.. code-block:: python

   for rec in result.recommendations:
       if rec["level"] == "action":
           logger.warning("[%s] %s", rec["metric"], rec["text"])

Recommendations are deterministic given the same report — the server
applies a fixed rule set (quality-score tiers, a hardness-difficulty
trigger, a feasibility-rate threshold, a penalty-tuning check, and a
low-concentration warning) — so they are safe to gate workflows on.

Re-fetching a Previous Result
-----------------------------

Every characterization run has a ``job_id`` that you can hand back to
:func:`~divi.backends.get_characterization_result` to retrieve the stored
result without re-running the analysis.

.. skip: next

.. code-block:: python

   from divi.backends import QoroService, get_characterization_result

   result = get_characterization_result(
       "4d0550f5-ffb0-...",
       service=QoroService(),
   )
   result.display()

This costs **no credits** and is the recommended pattern for notebooks
and CI runs where you want to inspect a known result without re-billing.

.. _pricing:

Pricing
-------

Characterization is priced per submission, scaling with the QUBO's qubit
count. A balance check runs *before* submission, and credits are deducted
only when the job completes successfully — failed jobs are not billed.
Fetching a result by ``job_id`` does not cost anything. See the
`dashboard <https://dash.qoroquantum.net/>`_ for the current credit tiers.

See Also
--------

* :doc:`../user_guide/combinatorial_optimization_qaoa_pce` — the QAOA
  workflow that characterization is designed to support.
* :class:`~divi.backends.CharacterizationResult` — full property list.
* :class:`~divi.backends.CharacterizationOptions` — every configurable
  field, including constraint and ansatz schemas.
* :class:`~divi.backends.QoroService` — the cloud client that drives the
  underlying API calls.
