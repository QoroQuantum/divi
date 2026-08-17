QUBO Characterization Service
=============================

The Qoro **QUBO Characterization Service** analyzes a QUBO or HUBO before you
spend quantum runs on it. It compares classical baselines, inspects the QAOA
landscape, and reports whether a cheap classical route already makes QAOA a
poor choice.

It can rule QAOA out or identify risks, but it cannot prove that quantum
execution will help. The analysis is classical and does not use quantum
hardware.

.. _quickstart-characterization:

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
       options=CharacterizationOptions(preset="deep"),
   )

   result.display()
   print(result.regime, result.confidence)
   print(result.certificate)
   print(result.classical_baseline)
   print(result.approximation_ratio)

Set ``QORO_API_KEY`` in the environment or a ``.env`` file first.

``reference_states`` is required and must contain at least one binary
candidate with one bit per problem variable. The states drive
reference-dependent diagnostics. They are not constraints, do not configure
the warm start, and do not need to be proven optima. A classical heuristic
candidate is sufficient.

When to Use It
--------------

Use characterization before QAOA when you need to:

* decide whether a classical shortcut already makes a quantum run unnecessary;
* choose initial QAOA angles without spending optimizer iterations on a blind
  search;
* understand whether density, frustration, or symmetry makes low-depth QAOA
  unlikely to help;
* size a penalty multiplier for a constrained formulation; or
* analyze a HUBO already accepted by ``BinaryOptimizationProblem``.

The service minimizes every submitted objective: reported energies are costs,
and lower is better. One problem variable corresponds to one qubit.

Characterization is diagnostic, not a benchmark of a particular QPU. It is
most useful as a filter before execution and as a source of initial
configuration for the QAOA run that follows.

Reading the Result
------------------

Read the result in this order:

1. Check ``regime`` and ``confidence`` to learn whether the QAOA estimate
   was exact, bounded, or unavailable.
2. With ``preset="deep"``, inspect ``certificate`` for a classical
   shortcut or low-depth warning.
3. Compare ``classical_baseline`` with ``approximation_ratio``.
4. If QAOA remains reasonable, use ``best_parameters``,
   ``qaoa_initial_params()``, and ``recommendations`` to configure it.

``certificate`` contains independent flags:

* ``low_treewidth`` and ``submodular`` identify possible efficient
  classical solution methods.
* ``greedy_solved`` and ``sa_solved`` report whether a cheap baseline
  reached the best known energy.
* ``high_frustration``, ``bitflip_symmetric``, and
  ``qaoa_low_depth_limited`` flag known low-depth QAOA risks.

A false flag only means that the corresponding shortcut or warning was not
detected. It is not evidence of quantum advantage.

Use ``result.display()`` for a terminal report, ``result.html`` in a
notebook, and ``result.report`` for the raw response. See
:class:`~divi.backends.characterization.CharacterizationResult` for every
property.

Report groups
^^^^^^^^^^^^^

The report is easier to understand as a few groups rather than as a flat list
of properties:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Group
     - Important fields
   * - Execution context
     - ``regime``, ``confidence``, ``regime_diagnostics``, and
       ``approximation_ratio_error_bound``.
   * - QAOA landscape
     - ``best_parameters``, ``energy_landscape``,
       ``state_probabilities``, and ``concentration_ratio``.
   * - Classical comparison
     - ``classical_baseline``, ``exact_ground_energy``,
       ``approximation_ratio``, and ``certificate``.
   * - Formulation quality
     - ``quality_score``, ``hardness``, and
       ``structural_sensitivity``.
   * - Constraints
     - ``feasibility_rate``, ``constraint_diagnostics``, and the
       penalty thresholds.
   * - Guidance
     - ``recommendations`` and ``qaoa_initial_params()``.

Some fields are optional by design. For example, a fast preset has no parameter
sweep, a refused QAOA regime has no approximation ratio, and a large problem
may have estimated spectrum endpoints but no exact gap. Interpret ``None``
together with the preset, regime, and estimation flags rather than as a failed
job.

``quality_score`` and ``hardness["difficulty"]`` summarize different
things. Quality describes properties such as coefficient scaling and
formulation structure. Difficulty combines the cost spectrum and other
problem diagnostics. Neither is a prediction of quantum advantage.

The certificate also separates mathematical statements from heuristics:
``low_treewidth`` and ``submodular`` can identify classical algorithms
with formal guarantees, while baseline and low-depth flags report what the
service observed or estimated for this formulation. Keep that distinction
when gating an automated workflow.

.. _sizing:

.. _cost-spectrum-metrics:

Scale and Exactness
^^^^^^^^^^^^^^^^^^^

Two different sizes affect the report:

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Threshold
     - Effect
   * - ``n <= 18``
     - The cost spectrum, gap, degeneracy, and ground energy are exact.
   * - ``n > 18``
     - Spectrum extremes are estimated. Check
       ``hardness["cost_spectrum_estimated"]``; gap fields may be ``None``.
   * - light cone ``k <= 18``
     - Exact QAOA expectation: ``regime`` is ``"exact"`` or
       ``"structured"``.
   * - ``18 < k <= 24``
     - ``regime="estimate"`` with an error bound.
   * - ``k > 24``
     - ``regime="refuse"``; no approximation ratio is returned, but other
       diagnostics remain available.

Sparsity therefore matters more than total variable count. A bounded-degree
QUBO can stay in the exact local regime at thousands of variables, while a
small dense problem can exceed the light-cone range.

The approximation ratio uses cost-spectrum endpoints, not eigenvalues of the
QUBO coefficient matrix. Matrix diagnostics describe numerical structure only.

The cost-spectrum fields have the following meanings:

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Field
     - Meaning
   * - ``cost_min`` / ``cost_max``
     - Best and worst objective values used to normalize the QAOA expectation.
   * - ``cost_gap``
     - Difference between the lowest distinct energy levels, when the spectrum
       is exact.
   * - ``ground_state_degeneracy``
     - Number of exact minimizers, when full enumeration is available.
   * - ``cost_spectrum_estimated``
     - Whether the endpoints came from sampling rather than enumeration.
   * - ``approximation_ratio_error_bound``
     - Error in the QAOA expectation calculation. It does not include error in
       estimated spectrum endpoints.

That last distinction matters: an error bound of zero means the expectation was
computed exactly, but the ratio can still use estimated ``cost_min`` and
``cost_max``. Check both fields before comparing ratios across problems.

Above 15 variables, feasibility and concentration are usually evaluated in a
greedy-selected subspace. Treat those values as conditional on that subspace,
not as exhaustive rates over every bitstring.

.. _analysis-presets:

Configuring the Run
-------------------

Choose a preset by goal:

.. list-table::
   :header-rows: 1
   :widths: 20 52 28

   * - Preset
     - Purpose
     - Additional analysis
   * - ``"fast"``
     - Short structural check
     - no sweep or certificate
   * - ``"standard"``
     - Structural check plus angle selection
     - parameter sweep
   * - ``"deep"``
     - Go/no-go assessment
     - sweep and certificate

Omitting ``preset`` uses the server's ``"standard"`` configuration.
Presets have the same credit cost; they differ in analysis time and output.

Common controls:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Purpose
   * - ``n_qubits``
     - Declares the full width when some variables do not appear in any term.
   * - ``parameter_sweep``
     - Overrides the preset. Fixed ``gamma`` and ``beta`` cannot be
       combined with an explicitly enabled sweep.
   * - ``gamma_range``, ``beta_range``
     - Overrides a sweep grid as ``(start, stop, count)``.
   * - ``ansatz``
     - Configures ``mixer`` and positive ``layers``.
   * - ``subspace``
     - Configures automatic warm-start or a manual variable subspace.
   * - ``constraints``
     - Adds feasibility diagnostics; it does not modify the QUBO.
   * - ``penalty_tuning``
     - Tunes separately supplied cost and penalty components.

Explicit booleans override the preset. Leaving them as ``None`` lets the
server decide.

This distinction matters when options come from a configuration file:
``parameter_sweep=None`` means "use the preset", while
``parameter_sweep=True`` pins the behavior even if the preset changes.
Overriding a preset in an unexpected direction emits a ``UserWarning`` but
the explicit value is honored.

Fixed ``gamma`` and ``beta`` do not implicitly disable a preset's sweep.
Use ``preset="fast"`` for a fixed-point evaluation without a sweep, or set
``parameter_sweep=False`` explicitly. Conversely, explicit sweep ranges only
matter when a sweep runs.

``n_qubits`` should be provided when isolated variables exist or when
constraint indices refer to variables absent from the objective terms. Divi
uses it to validate reference-state width, manual subspaces, and any constraint
indices it can check locally.

Warm-start and subspaces
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from divi.backends.characterization import CharacterizationOptions

   automatic = CharacterizationOptions(
       preset="deep",
       subspace={
           "auto_warmstart": True,
           "solver": "greedy",
           "max_variable_qubits": 12,
           "restarts": 5,
       },
   )

   manual = CharacterizationOptions(
       n_qubits=6,
       subspace={
           "auto_warmstart": False,
           "base_bitstring": "001010",
           "variable_qubits": [0, 2, 5],
       },
   )

For 15 qubits or fewer, the service normally uses exact full-space evaluation.
Set ``auto_warmstart=True`` explicitly when you need warm-start outputs.
``structural_sensitivity=True`` produces flip-cost scores only on the
automatic warm-start path.

Divi rejects unknown keys, incompatible modes, malformed bitstrings, duplicate
indices, and invalid sizes before creating a job. See
:class:`~divi.backends.characterization.CharacterizationOptions` for the
complete schema.

.. _penalty-tuning:

Penalty Tuning and Constraints
------------------------------

Constraint descriptors are diagnostic metadata. They do **not** encode a
constraint into the QUBO. Build the penalty terms yourself and supply cost-only
and penalty-only components:

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
           {
               "type": "inequality",
               "bound": 10,
               "weights": {0: 4, 1: 5, 2: 7},
           },
       ],
       penalty_tuning=True,
   )

   result = characterize_and_validate(
       problem,
       reference_states=["001"],
       service=QoroService(),
       options=options,
   )

Supported types are ``max_cardinality``, ``min_cardinality``,
``eq_cardinality``, ``inequality``, and ``equality``. Weighted
constraints use ``weights={qubit_index: weight}``; cardinality constraints
may restrict their scope with ``qubits``.

Penalty tuning returns two minimum thresholds:

* ``penalty_lambda_min_feasible`` is the smallest empirically feasible value
  observed by the tuner.
* ``penalty_lambda_safe`` is conservative and guaranteed. Use
  ``lambda >= penalty_lambda_safe``.

``penalty_recommendation`` is the larger value. For larger problems, the
empirical threshold may be estimated; check
``penalty_lambda_min_feasible_estimated``.

``constraint_diagnostics`` reports each descriptor's violation rate and
whether it appears redundant. These values are weighted by the diagnostic QAOA
distribution: "redundant" means that the constraint carries negligible
violating probability mass in that distribution, not that it is logically
implied by the other constraints.

The service tunes the penalty you supply; it does not synthesize missing
penalty terms. A larger multiplier strengthens feasibility but can also widen
the energy scale and make the QAOA landscape harder to optimize.

Reading Recommendations Programmatically
-----------------------------------------

``result.recommendations`` is a list of structured rules. Each entry has:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Key
     - Meaning
   * - ``level``
     - ``"info"``, ``"warn"``, or ``"action"``.
   * - ``metric``
     - The report field that triggered the rule.
   * - ``text``
     - Plain-text guidance for logs and terminals.
   * - ``html``
     - Equivalent guidance with inline markup for notebook output.

.. skip: next

.. code-block:: python

   actionable = [
       rec for rec in result.recommendations if rec["level"] == "action"
   ]
   for rec in actionable:
       print(f'{rec["metric"]}: {rec["text"]}')

The list is empty when no rule applies. Failed, cancelled, or unfinished jobs
raise instead of returning an empty result, so an empty recommendation list
does not hide a job failure.

Job Lifecycle and Recovery
--------------------------

Jobs move through ``PENDING``, ``RUNNING``, and a terminal
``COMPLETED``, ``FAILED``, or ``CANCELLED`` state. The convenience
helper waits synchronously.

Initialization and submission are not retried because both mutate server
state. After job creation, an ambiguous submission, polling, or result-fetch
failure raises :exc:`~divi.exceptions.CharacterizationSubmitError`. It
preserves the existing ``job_id`` and identifies the failed ``phase``.

Recover that job instead of submitting another one:

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
``CANCELLED`` raises :exc:`~divi.exceptions.ExecutionCancelledError`.
A completed job without a usable report also raises.

.. _pricing:

Cost
----

Characterization is priced by variable count. Preset choice does not affect the
charge, failed jobs are not billed, and fetching an existing result is free.
See the `dashboard <https://dash.qoroquantum.net/>`_ for current tiers.

See Also
--------

* ``tutorials/backends/characterize_maxcut_qubo.py`` for an end-to-end
  MaxCut example.
* :class:`~divi.backends.characterization.CharacterizationResult` for result
  properties.
* :class:`~divi.backends.characterization.CharacterizationOptions` for the
  complete configuration schema.
* :doc:`../algorithms/combinatorial_optimization_qaoa_pce` for the QAOA
  workflow.
