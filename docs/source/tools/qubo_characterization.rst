QUBO Characterization Service
=============================

The Qoro **QUBO Characterization Service** is a hosted diagnostic that
answers a decision-first question before you spend QAOA runs on a
problem: *is quantum worth it here?* It runs a classical reference
solver alongside a QAOA parameter sweep on the same QUBO/HUBO, then
returns an analysis regime and structural certificate, the classical
baseline it was measured against, an achievable upper-bound
approximation ratio (:attr:`~divi.backends.characterization.CharacterizationResult.approximation_ratio`),
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

* **Before running QAOA at all**, to check the
  :attr:`~divi.backends.characterization.CharacterizationResult.certificate` — does a
  provable classical shortcut exist (bounded treewidth, submodular
  couplings), or do literature markers say low-depth QAOA is unlikely to
  help (high frustration, global bit-flip symmetry)? — backed by a real
  classical baseline, so you don't burn quantum runs on a QUBO a greedy
  or simulated-annealing solver already solves.
* **Before tuning a QAOA**, to find good initial ``γ``, ``β`` values via
  a server-side parameter sweep and skip a long optimizer warm-up.
* **When picking a penalty multiplier** for constrained problems —
  penalty tuning returns a guaranteed-safe ``λ`` and an empirical minimum
  feasible ``λ``, and flags whether the current value is well-tuned.
* **For HUBO problems**, where ``BinaryOptimizationProblem`` already
  accepts higher-order terms; characterization works on the same
  canonical form without extra conversion.

The diagnostic itself does **not** execute QAOA on hardware. It runs
classically on Qoro's cloud and completes within a few seconds.

Quick Start
-----------

The single-call entry point is
:func:`~divi.backends.characterization.characterize_and_validate`:

.. skip: next

.. code-block:: python

   import numpy as np
   from divi.backends import QoroService
   from divi.backends.characterization import (
       CharacterizationOptions,
       characterize_and_validate,
   )
   from divi.qprog.problems import BinaryOptimizationProblem

   # A small QUBO (any shape BinaryOptimizationProblem accepts is fine —
   # ndarray, sparse, BQM, HUBO dict).
   problem = BinaryOptimizationProblem(np.array([[-1, 2], [0, -1]]))

   result = characterize_and_validate(
       problem,
       reference_states=["01", "10"],   # optional reference solution bitstrings
       service=QoroService(),
       options=CharacterizationOptions(parameter_sweep=True),
   )

   result.display()                        # rich console report
   print(result.regime, result.confidence) # e.g. "estimate", "estimated"
   print(result.certificate)               # certified_easy / no_lowdepth_advantage_expected / uncertain
   print(result.approximation_ratio)       # achievable upper bound, 0-1 (None if regime == "refuse")

``reference_states`` are reference solution bitstrings for
reference-dependent diagnostics. They are not constraints, not the warm start,
and they do not have to be proven optima. If you do not have candidate or known
solutions, pass ``[]``; Composer can derive a classical reference solution when
an analysis branch needs one.

.. _the-certificate:

The Certificate: Is Quantum Worth It?
---------------------------------------

Characterization answers "is quantum worth it here?" with three
independent pieces of information rather than a single ternary label:

* :attr:`~divi.backends.characterization.CharacterizationResult.regime` — how
  :math:`\langle C \rangle` (the QAOA ansatz's expected cost) was
  computed, based on the size of the interaction light cone:
  ``"exact"`` (small enough to compute exactly), ``"structured"``
  (exact via exploited structure, e.g. low treewidth), ``"estimate"``
  (truncated Pauli-propagation, with an error bound), or ``"refuse"``
  (the light cone is wider than the feasibility budget, so no
  cheap-and-correct estimate exists — no approximation ratio is given;
  see :attr:`~divi.backends.characterization.CharacterizationResult.refuse_reason`).
* :attr:`~divi.backends.characterization.CharacterizationResult.confidence` — the
  confidence behind that regime: ``"proven"`` (exact light cone),
  ``"estimated"`` (± :attr:`~divi.backends.characterization.CharacterizationResult.approximation_ratio_error_bound`
  truncation error), or ``"open"`` (refused).
* :attr:`~divi.backends.characterization.CharacterizationResult.certificate` — a dict of
  independently-firing structural flags, not a ternary verdict:

  * ``certified_easy`` — a provable classical shortcut exists (bounded
    treewidth or submodular couplings); ``easy_witnesses`` names which
    one fired.
  * ``no_lowdepth_advantage_expected`` — literature markers suggest
    low-depth QAOA is unlikely to help (high frustration, global
    bit-flip symmetry); ``lowdepth_markers`` names which one fired.
  * ``uncertain`` — neither of the above fired. When set, the optional
    :attr:`~divi.backends.characterization.CharacterizationResult.quantum_curiosity`
    sub-dict reports a ``status``, ``depth_to_escape_locality``, and a
    ``next_step`` suggestion.
  * ``structural`` (optional) — ``is_psd``, ``rank``, ``submodular``,
    ``bounded_treewidth``, also exposed directly as
    :attr:`~divi.backends.characterization.CharacterizationResult.is_psd` and
    :attr:`~divi.backends.characterization.CharacterizationResult.rank`.

The certificate never claims a QUBO is "promising" for quantum — it
only rules easy cases in or out. A ``"refuse"`` regime means the
interaction light cone is wider than the feasibility budget at the
requested depth, so no cheap-and-correct assessment exists — this is
driven by the light-cone width (which grows with circuit depth and
connectivity), **not** by coupling density, and does not mean the QUBO
is intractable or uninteresting. The
:attr:`~divi.backends.characterization.CharacterizationResult.refuse_reason`
distinguishes ``"lightcone_too_wide"`` (an a-priori size limit) from
``"estimate_unreachable"`` (the truncated estimator could not reach its
error tolerance within budget), and
:attr:`~divi.backends.characterization.CharacterizationResult.regime_diagnostics`
reports the light-cone sizes. Hardness, penalty, and classical-baseline
data are still returned in full.

.. note::

   A few **cost-spectrum** hardness fields (``cost_gap``,
   ``cost_gap_normalized``, ``ground_state_degeneracy``) are exact only for
   small problems and return ``None`` above the exact-enumeration size (check
   ``hardness["cost_spectrum_estimated"]``). This is independent of the
   ``"refuse"`` regime — a large problem can be ``"structured"`` and still have
   these particular fields estimated/omitted; ``difficulty``, ``frustration_index``,
   and the certificate remain available.

The certificate is only as useful as the reference it is measured
against — :attr:`~divi.backends.characterization.CharacterizationResult.classical_baseline`.
That dict reports what cheap classical solvers achieve on the same
QUBO: ``greedy_energy``, ``sa_energy`` (simulated annealing),
``best_energy`` (the best of the two), ``distinct_optima``, and, for
small problems, the exact ``exact_ground_energy`` from brute-force
enumeration. When a continuous relaxation was computed, its bound is
also exposed as :attr:`~divi.backends.characterization.CharacterizationResult.relaxation_bound`
— a provable lower bound on the true minimum, independent of any
classical heuristic; the closer it sits to ``best_energy``, the more
confidence you can have that the classical baseline is already
near-optimal.

When the regime allows it (anything but ``"refuse"``),
:attr:`~divi.backends.characterization.CharacterizationResult.approximation_ratio` is
computed as

.. math::

   r = \frac{\langle C \rangle - C_{\max}}{C_{\min} - C_{\max}} \in [0, 1]

evaluated at the uniform :math:`|{+}\rangle` state by the light-cone
engine, where :math:`\langle C \rangle` is the ansatz's expected cost
at the swept ``best_parameters`` and :math:`C_{\min}`, :math:`C_{\max}`
are the exact optimal and worst-case costs. This is an **achievable
upper bound** — what a real, cold-started QAOA run can reach at that
depth, not a live measurement or a guarantee any particular run gets
there — paired with
:attr:`~divi.backends.characterization.CharacterizationResult.approximation_ratio_error_bound`
for the ``±ε`` uncertainty band (``0`` in the ``"exact"``/``"structured"``
regimes, positive for the truncated ``"estimate"`` regime). It is
``None`` in the ``"refuse"`` regime: the server declined to estimate
rather than ship an unreliable number. This baseline turns
``approximation_ratio`` from an isolated number into an interpretable
one — an upper bound of 0.87 means little if a classical solver on the
same problem already tops out at 0.80.

.. note::

   For **penalty-encoded constrained problems**, the ratio is taken over the
   full penalized cost spectrum, whose worst-case states are the
   constraint-violating ones. A high ``approximation_ratio`` there means "close
   to the best penalized energy", **not** "this fraction of the way to a good
   *feasible* solution" — read it alongside ``feasibility_rate``. Because the
   spectrum is stretched by ``penalty_weight``, the ratio also shifts with the
   penalty multiplier, so it is not comparable across different weights.

Warm-starting a real QAOA run
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The swept angles come back in two places, for two uses:

* :attr:`~divi.backends.characterization.CharacterizationResult.best_parameters`
  — the single best ``γ``/``β`` from the **p=1** sweep. Use it to warm-start a
  one-layer run.
* :attr:`~divi.backends.characterization.CharacterizationResult.ar_vs_depth` —
  a per-depth curve; entry ``[p-1]`` carries the ``gammas``/``betas`` lists for a
  ``p``-layer circuit. Use it to warm-start at the
  :attr:`~divi.backends.characterization.CharacterizationResult.recommended_min_layers`.

The result assembles either of these into the exact ``initial_params`` layout
``QAOA.run`` expects — call
:meth:`~divi.backends.characterization.CharacterizationResult.qaoa_initial_params`
and pass it straight through, so warm-starting is one line:

.. skip: next

.. code-block:: python

   p = result.recommended_min_layers
   qaoa = QAOA(problem, n_layers=p, optimizer=ScipyOptimizer(), backend=MaestroSimulator())
   qaoa.run(initial_params=result.qaoa_initial_params())

Under the hood that is a flat row ordered **per layer, cost angle then mixer
angle** — ``[γ₀, β₀, γ₁, β₁, …]``, shape ``(1, 2 * n_layers)``. Only the shape
is validated, so if you build it by hand match this order exactly; the helper
does it for you. ``qaoa_initial_params()`` returns ``None`` in the ``"refuse"``
regime or when no sweep ran, so guard it before use.

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
The structured fields (``regime``, ``certificate``, ``quality_score``,
``recommendations``, etc.) remain available either way.

What You Get Back
-----------------

:func:`~divi.backends.characterization.characterize_and_validate` returns a
:class:`~divi.backends.characterization.CharacterizationResult` with typed properties for
the most useful fields:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Property
     - What it means
   * - ``regime``
     - How :math:`\langle C \rangle` was computed, or whether it could
       be — ``exact`` / ``structured`` / ``estimate`` / ``refuse``. See
       :ref:`the-certificate` above.
   * - ``confidence``
     - Confidence behind ``regime`` — ``proven`` / ``estimated`` /
       ``open``.
   * - ``certificate``
     - Independently-firing structural flags (``certified_easy``,
       ``no_lowdepth_advantage_expected``, ``uncertain``) with
       witnesses — never a ternary "promising" call. See
       :ref:`the-certificate` above.
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
     - Achievable **upper bound** on the approximation ratio at the
       swept ``best_parameters`` (± ``approximation_ratio_error_bound``)
       — ``1.0`` is the exact optimum. Not a live-run guarantee; ``None``
       in the ``refuse`` regime. Interpret it against
       ``classical_baseline``.
   * - ``approximation_ratio_error_bound``
     - The ``±ε`` uncertainty band on ``approximation_ratio`` — ``0``
       for the exact light-cone computation (``exact``/``structured``
       regimes), positive for the truncated estimate (``estimate``
       regime).
   * - ``quality_score``
     - QAOA amenability score (0–100) at the best swept parameters —
       an alias for ``reference_concentration_score`` when a sweep ran, falling
       back to ``formulation_quality`` otherwise. **Not** the solution
       quality; use ``approximation_ratio`` for that.
   * - ``formulation_quality``
     - Structural amenability score (0–100), independent of any
       parameter sweep — a scale-invariant composite of cost gap,
       ground-state degeneracy, density, and weight balance. Tells you
       how well-formed the QUBO is for QAOA in general.
   * - ``reference_concentration_score``
     - QAOA quality (0–100) at the best swept parameters specifically
       — reference-dependent, requires ``parameter_sweep=True``.
   * - ``concentration_ratio``
     - Probability mass on reference states relative to the uniform
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

Pass a :class:`~divi.backends.characterization.CharacterizationOptions` to control which
sub-analyses run. All fields are optional; the default-constructed object
runs a basic report.

.. skip: next

.. code-block:: python

   from divi.backends.characterization import CharacterizationOptions

   opts = CharacterizationOptions(
       parameter_sweep=True,             # sweep γ, β server-side
       structural_sensitivity=True,                 # per-qubit flip-cost criticality report
       penalty_tuning=True,                   # recommend a penalty λ
       ansatz={"mixer": "x", "layers": 1},
       subspace={"auto_warmstart": True, "max_variable_qubits": 12},
   )

``ansatz`` controls the QAOA circuit shape only: currently the mixer and
layer count. Supported mixers are ``"x"``, ``"xy"``, and ``"I"`` (identity/no
mixer, useful as a diagnostic baseline). ``subspace`` controls how the
service chooses or bounds the simulated/search subspace, including warm-start
behavior and the number of variable qubits to explore.

Setting ``structural_sensitivity=True`` adds a per-qubit flip-cost criticality
breakdown to the result's ``structural_sensitivity`` field, identifying which variables
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

.. important::

   Declaring a constraint in ``options.constraints`` does **not** encode it into
   the QUBO. Those descriptors drive the feasibility/penalty *diagnostics* only;
   you must still build the penalty terms yourself (e.g. slack bits for an
   inequality, a squared cardinality penalty) and pass them via
   ``BinaryOptimizationProblem(cost, penalty=..., penalty_weight=...)``. The
   service tunes and reports on the penalty you supply; it does not synthesize one.

For QUBOs that encode constraints as quadratic penalties, you can split
the formulation into the cost-only and penalty-only halves and pass them
both. With ``penalty_tuning=True``, the service returns an **interval**
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

   problem = BinaryOptimizationProblem(
       cost_terms,                       # objective/cost QUBO or HUBO
       penalty=penalty_terms,            # penalty-only QUBO or HUBO
       penalty_weight=10.0,              # current multiplier in the full QUBO
   )
   opts = CharacterizationOptions(
       constraints=[...],                # constraint descriptors
       penalty_tuning=True,
   )

   result = characterize_and_validate(
       problem, reference_states=[...], service=QoroService(), options=opts
   )
   print(result.penalty_lambda_min_feasible, result.penalty_lambda_safe)
   print(result.constraint_diagnostics)

``constraint_diagnostics`` breaks the aggregate ``feasibility_rate``
down per constraint — each entry carries an ``index``, the constraint
``type``, its ``violation_rate``, and an ``is_redundant`` flag.

.. note::

   ``feasibility_rate``, ``violation_rate`` and ``is_redundant`` are
   **QAOA-distribution weighted** — they measure the probability mass on
   (in)feasible states under the diagnostic p=1 ansatz, *not* the raw
   fraction of the search space that is feasible. So ``is_redundant`` means
   "carries negligible QAOA probability mass," not "logically implied by the
   other constraints." And ``penalty_lambda_min_feasible`` is only exact for
   small problems: above ~15 variables it is a subspace-search estimate
   (check ``penalty_lambda_min_feasible_estimated``) — rely on
   ``penalty_lambda_safe`` / ``penalty_recommendation``, which are analytic.

.. _constraint-schema:

Constraint Descriptors
^^^^^^^^^^^^^^^^^^^^^^^

Each entry in ``constraints`` is a plain dict:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - ``type``
     - Meaning (with ``bound``)
   * - ``max_cardinality``
     - at most ``bound`` variables set to 1
   * - ``min_cardinality``
     - at least ``bound`` variables set to 1
   * - ``eq_cardinality``
     - exactly ``bound`` variables set to 1
   * - ``inequality``
     - weighted sum ``Σ wᵢ·xᵢ ≤ bound``; requires ``weights``
   * - ``equality``
     - weighted sum ``Σ wᵢ·xᵢ == bound``; requires ``weights``

``weights`` is a dict ``{qubit_index: weight}``; the optional ``qubits``
key restricts a cardinality constraint to a subset of variables (defaults
to all). All indices must lie in ``[0, n_qubits)`` or the request is
rejected with a clear error.

.. skip: next

.. code-block:: python

   constraints = [
       {"type": "max_cardinality", "bound": 3},
       {"type": "inequality", "bound": 10, "weights": {0: 4, 1: 5, 2: 7}},
   ]

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
:func:`~divi.backends.characterization.get_characterization_result` to retrieve the stored
result without re-running the analysis.

.. skip: next

.. code-block:: python

   from divi.backends import QoroService
   from divi.backends.characterization import get_characterization_result

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
* :class:`~divi.backends.characterization.CharacterizationResult` — full property list.
* :class:`~divi.backends.characterization.CharacterizationOptions` — every configurable
  field, including constraint, ansatz, and subspace schemas.
* :class:`~divi.backends.QoroService` — the cloud client that drives the
  underlying API calls.
