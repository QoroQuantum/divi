QUBO Characterization Service
=============================

The Qoro **QUBO Characterization Service** is a hosted diagnostic that
answers a decision-first question before you spend QAOA runs on a
problem: *is quantum worth it here?* It runs a classical reference
solver alongside a QAOA parameter sweep on the same QUBO/HUBO, then
returns a verdict, the classical baseline it was measured against, a
real solution-quality number (:attr:`~divi.backends.CharacterizationResult.approximation_ratio`),
hardness metrics, and actionable recommendations — all from a single
short call.

.. note::

   Characterization runs on Qoro's hosted cloud service via
   :class:`~divi.backends.QoroService`. See :ref:`pricing` below for
   credit cost.

When to Use It
--------------

Characterization is useful whenever you would otherwise *guess* at a QAOA
setup, or spend real QAOA runs finding out something a classical solver
could have told you in seconds. Common scenarios:

* **Before running QAOA at all**, to get a :attr:`~divi.backends.CharacterizationResult.verdict`
  — ``classically_easy``, ``marginal``, or ``promising`` — backed by a
  real classical baseline, so you don't burn quantum runs on a QUBO a
  greedy or simulated-annealing solver already solves.
* **Before tuning a QAOA**, to find good initial ``γ``, ``β`` values via
  a server-side parameter sweep and skip a long optimizer warm-up.
* **When picking a penalty multiplier** for constrained problems —
  auto-tuning returns a guaranteed-safe ``λ`` and an empirical minimum
  feasible ``λ``, and flags whether the current value is well-tuned.
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

   result.display()                        # rich console report
   print(result.verdict["verdict"])        # "classically_easy" / "marginal" / "promising"
   print(result.approximation_ratio)       # QAOA solution quality, 0-1

.. _the-verdict:

The Verdict: Is Quantum Worth It?
----------------------------------

:attr:`~divi.backends.CharacterizationResult.verdict` is the headline
result. It is a dict with three parts:

* ``verdict`` — ``"classically_easy"`` (a classical solver already
  reaches the optimum or near it), ``"marginal"`` (QAOA and classical
  solvers land close together), or ``"promising"`` (QAOA's swept
  approximation ratio beats the classical baseline by a meaningful
  margin).
* ``rationale`` — a human-readable sentence explaining the call, e.g.
  *"QAOA AR 0.87 exceeds the classical baseline 0.80."* ("AR" is the
  approximation ratio described below — 1.0 is optimal.)
* ``qaoa_approximation_ratio`` and ``classical_best_energy`` — the raw
  numbers behind the rationale.

The verdict is only as good as the reference it is measured against —
:attr:`~divi.backends.CharacterizationResult.classical_baseline`.
That dict reports what cheap classical solvers achieve on the same
QUBO: ``greedy_energy``, ``sa_energy`` (simulated annealing),
``best_energy`` (the best of the two), ``distinct_optima``, and, for
small problems, the exact ``exact_ground_energy`` from brute-force
enumeration. When a continuous relaxation was computed, its bound is
also exposed as :attr:`~divi.backends.CharacterizationResult.relaxation_bound`
— a provable lower bound on the true minimum, independent of any
classical heuristic; the closer it sits to ``best_energy``, the more
confidence you can have that the classical baseline is already
near-optimal. This baseline is what turns
:attr:`~divi.backends.CharacterizationResult.approximation_ratio` from
an isolated number into an interpretable one — an AR of 0.87 only
tells you QAOA is *worth it* if you also know a classical solver on the
same problem tops out at, say, 0.80.

``approximation_ratio`` is computed as

.. math::

   r = \frac{\langle C \rangle - C_{\max}}{C_{\min} - C_{\max}} \in [0, 1]

where :math:`\langle C \rangle` is the QAOA ansatz's expected cost at
the swept ``best_parameters``, and :math:`C_{\min}`, :math:`C_{\max}`
are the exact optimal and worst-case costs. ``r = 1`` means QAOA
reaches the optimum at that depth; this is the same quantity you would
compute from your own QAOA run, evaluated server-side so you get it
before spending any circuits.

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
The structured fields (``verdict``, ``quality_score``, ``recommendations``,
etc.) remain available either way.

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
   * - ``verdict``
     - Decision-first summary — ``classically_easy`` / ``marginal`` /
       ``promising`` — plus a ``rationale`` and the comparison numbers.
       See :ref:`the-verdict` above.
   * - ``classical_baseline``
     - Greedy and simulated-annealing energies on the same QUBO, the
       best of the two, and (for small problems) the exact ground
       energy. The reference that makes ``approximation_ratio``
       meaningful.
   * - ``relaxation_bound``
     - Continuous relaxation bound (e.g. LP/SDP) on the optimum, when
       computed — a provable lower bound independent of any classical
       heuristic. Read alongside ``classical_baseline``.
   * - ``approximation_ratio``
     - The real approximation ratio QAOA reaches at the swept
       ``best_parameters`` — ``1.0`` is the exact optimum. The headline
       solution-quality number; interpret it against
       ``classical_baseline``.
   * - ``quality_score``
     - QAOA amenability score (0–100) at the best swept parameters —
       an alias for ``target_achievability`` when a sweep ran, falling
       back to ``formulation_quality`` otherwise. **Not** the solution
       quality; use ``approximation_ratio`` for that.
   * - ``formulation_quality``
     - Structural amenability score (0–100), independent of any
       parameter sweep — a scale-invariant composite of cost gap,
       ground-state degeneracy, density, and weight balance. Tells you
       how well-formed the QUBO is for QAOA in general.
   * - ``target_achievability``
     - QAOA quality (0–100) at the best swept parameters specifically
       — target-dependent, requires ``parameter_sweep=True``.
   * - ``concentration_ratio``
     - Probability mass on target states relative to the uniform
       baseline. ``1.0`` is uniform; values around ``1.5×`` or below
       are flagged as too uniform and suggest deeper circuits.
   * - ``best_parameters``
     - The ``γ`` / ``β`` returned by the parameter sweep (if requested).
   * - ``cost_gap``, ``cost_gap_normalized``, ``ground_state_degeneracy``, ``global_flip_symmetric``, ``treewidth_estimate``, ``frustration_index``
     - Cost-spectrum hardness metrics — see
       :ref:`cost-spectrum-metrics` below.
   * - ``feasibility_rate``
     - Fraction of sampled states satisfying all declared constraints.
   * - ``constraint_diagnostics``
     - Per-constraint violation rate and a redundancy flag, one entry
       per declared constraint.
   * - ``penalty_lambda_min_feasible``, ``penalty_lambda_safe``, ``penalty_recommendation``, ``is_well_tuned``
     - Penalty guidance for constrained problems — see
       :ref:`penalty-tuning` below.
   * - ``recommendations``
     - Server-generated interpretive notes (see
       :ref:`reading-recommendations` below).

Every field is ``None`` when the corresponding analysis was not requested.

.. _cost-spectrum-metrics:

Hardness Metrics: Cost Spectrum, Not Matrix Spectrum
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``hardness`` metrics are computed from the QUBO's **cost spectrum**
— the distribution of objective values over bit assignments — rather
than the eigenvalues of the QUBO matrix itself:

* ``cost_gap`` — energy gap between the best and second-best distinct
  assignment.
* ``cost_gap_normalized`` — ``cost_gap`` divided by the full energy
  range ``E_max - E_min``; scale-invariant, so use this (not the raw
  ``cost_gap``) to compare across differently-scaled formulations.
* ``ground_state_degeneracy`` — number of assignments tied for optimal
  (exact for small problems).
* ``global_flip_symmetric`` — whether flipping every bit maps the best
  solution to another optimum. When ``True``, a standard X-mixer QAOA
  state stays in a fixed global-parity eigenspace at any depth, so this
  particular degeneracy cannot be resolved by adding layers alone.
* ``treewidth_estimate`` — upper bound on the interaction-graph
  treewidth (min-fill heuristic), a proxy for how much QAOA's
  entangling structure has to cover.
* ``frustration_index`` — fraction of couplings unsatisfiable at the
  best solution.

These are **scale-invariant**: rescaling every coefficient in a QUBO by
a constant factor changes none of them, because they depend only on
the relative ordering and structure of assignments, not on the
magnitude of the objective. This is the reason they replaced the
earlier matrix eigenvalue metrics (spectral gap, condition number),
which shift under a rescaling that leaves the underlying optimization
problem unchanged. The matrix eigenvalue gap is still computed
server-side for transparency but is not surfaced as a client property.

Configuring the Run
--------------------

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

.. _penalty-tuning:

Penalty Tuning for Constrained Problems
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For QUBOs that encode constraints as quadratic penalties, you can split
the formulation into the cost-only and penalty-only halves and pass them
both. With ``auto_tune=True``, the service returns an **interval**
rather than a single number:

* ``penalty_lambda_min_feasible`` — the empirical smallest ``λ`` at
  which the optimum becomes feasible in the swept report.
* ``penalty_lambda_safe`` — a guaranteed-sufficient bound derived from
  the Lucas (2014) and Glover–Kochenberger–Du penalty-sizing rules,
  which relate ``λ`` to the QUBO's coefficient magnitudes rather than
  to any single sampled run. See `Lucas, "Ising formulations of many NP
  problems" <https://arxiv.org/abs/1302.5843>`_ and `Glover, Kochenberger
  & Du, "A Tutorial on Formulating and Using QUBO Models"
  <https://arxiv.org/abs/1811.11538>`_.

``penalty_recommendation`` carries the safe bound — use it directly if
you want a single number; use the two-sided interval if you want to
trade off penalty strength against QAOA landscape difficulty (larger
``λ`` enforces feasibility more strongly but flattens the cost
landscape QAOA has to search).

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
   print(result.penalty_lambda_min_feasible, result.penalty_lambda_safe)
   print(result.constraint_diagnostics)

``constraint_diagnostics`` breaks the aggregate ``feasibility_rate``
down per constraint — each entry carries an ``index``, the constraint
``type``, its ``violation_rate``, and an ``is_redundant`` flag for
constraints that never bind given the others.

.. _reading-recommendations:

Reading Recommendations Programmatically
-----------------------------------------

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
     - Which report field drove the rule — e.g. ``"quality_score"``,
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
-------------------------------

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
