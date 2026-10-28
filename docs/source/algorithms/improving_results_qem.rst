Improving Results with Error Mitigation
========================================

Divi provides built-in quantum error mitigation (QEM). Two built-in protocols
ship with the library:

- **Zero Noise Extrapolation (ZNE)** — runs circuits at artificially increased
  noise levels and extrapolates to the zero-noise limit.
- **Quantum Enhanced Pauli Propagation (QuEPP)** — decomposes the circuit into
  Clifford Pauli paths, simulates them classically, and corrects the noisy
  quantum result with an empirical rescaling factor.

Pass either protocol into variational programs (for example :class:`~divi.qprog.algorithms.VQE`
or :class:`~divi.qprog.algorithms.QAOA`) with the ``qem_protocol`` argument. You can also
subclass :class:`~divi.circuits.qem.QEMProtocol` for custom mitigation; see
`Custom Error Mitigation Protocols`_ below.

.. list-table:: Mitigation protocols
   :header-rows: 1
   :widths: 30 70

   * - Protocol
     - Cost driver
   * - ZNE
     - Scale factors × measurement groups
   * - QuEPP
     - Path count × twirls, plus the target circuit

:meth:`~divi.qprog.QuantumProgram.dry_run` shows structural circuit expansion;
it cannot predict the number of shots needed to reach a particular statistical
error bar.

Zero Noise Extrapolation (ZNE)
------------------------------

Divi's ZNE runs the target circuit at several amplified noise levels and
extrapolates the per-scale expectation values back to the zero-noise limit.
Folding and extrapolation are both built-in — :class:`~divi.circuits.zne.ZNE`
ships with global-unitary folding (:func:`~divi.circuits.zne.global_fold`)
by default and uses :class:`~divi.circuits.zne.RichardsonExtrapolator`
unless a custom extrapolator is provided.  Both integer and fractional
scale factors are supported. :func:`~divi.circuits.zne.local_fold` folds
per gate.

**Basic Usage:**

.. dashboard-example: zne

.. code-block:: python

   from divi.circuits.zne import ZNE, RichardsonExtrapolator
   from divi.qprog import VQE
   from divi.qprog.optimizers import MonteCarloOptimizer
   from divi.qprog.problems import MolecularProblem
   from divi.backends import MaestroConfig, MaestroSimulator
   import maestro
   from pyscf import gto

   # Create a ZNE protocol with three noise scale factors.  The default
   # folding function is global unitary folding, which supports both
   # integer (e.g. [1, 3, 5]) and fractional (e.g. [1.0, 1.5, 2.0])
   # scale factors.
   scale_factors = [1, 3, 5]
   zne_protocol = ZNE(
       scale_factors=scale_factors,
       extrapolator=RichardsonExtrapolator(),
   )

   # Apply to VQE
   h2_molecule = gto.M(
       atom="H 0 0 -0.6614; H 0 0 0.6614", basis="sto-3g", unit="Bohr"
   )

   problem = MolecularProblem.from_molecule(h2_molecule)

   # Depolarising gate noise, simulated exactly as a density matrix
   n_qubits = problem.hamiltonian.num_qubits
   noise_model = maestro.NoiseModel()
   noise_model.set_all_1q_gate_depolarizing(n_qubits, 0.005)
   noise_model.set_all_2q_gate_depolarizing(n_qubits, 0.01)
   backend = MaestroSimulator(
       maestro_config=MaestroConfig(
           simulation_type="DensityMatrix", noise_model=noise_model
       )
   )

   vqe = VQE(
       problem,
       qem_protocol=zne_protocol,
       optimizer=MonteCarloOptimizer(),
       backend=backend,
       max_iterations=10,
   )

   vqe.run()
   print(f"Mitigated energy: {vqe.best_loss:.6f}")

The default :func:`~divi.circuits.zne.global_fold` is deterministic.
:func:`~divi.circuits.zne.local_fold` folds per gate:

.. skip: next

.. code-block:: python

   from divi.circuits.zne import ZNE, local_fold

   # Per-gate folding with fractional scale factors
   zne_local = ZNE(
       scale_factors=[1.0, 1.25, 1.5, 1.75, 2.0],
       folding_fn=local_fold,
   )

Pass ``selection="from_left"`` or ``"from_right"`` through
``functools.partial`` for deterministic local folding; ``exclude=`` can isolate
specific gate classes.

.. note::
   With ``d`` foldable gates, effective scales are quantised in increments of
   ``2/d``. ZNE forwards the effective scale and warns when requested scales
   collapse to one value.

Quantum Enhanced Pauli Propagation (QuEPP)
------------------------------------------

QuEPP implements the Clifford Perturbation Theory protocol of
`Majumder et al. (2026) <https://arxiv.org/abs/2603.14485>`_. It computes
low-order Clifford paths classically and uses noisy hardware to estimate the
residual. Select it with ``qem_protocol=QuEPP(truncation_order=2)``.

**Parameters:**

- ``truncation_order`` *(int, default 2)* — Maximum CPT expansion order *K*.
  Higher *K* includes more Pauli paths (cost grows combinatorially with the
  number of non-Clifford gates).  It governs every enumerated path set: always
  with ``sampling="exhaustive"``, and on symbolic circuits otherwise (see
  ``sampling``).
- ``coefficient_threshold`` *(float, optional)* — Prune paths whose absolute
  weight falls below this threshold during enumeration (disabled on symbolic
  circuits, whose angle magnitudes are unknown).
  The weight is the path's trigonometric product, independent of the
  observable's coefficients, so the threshold means the same thing whatever
  scale the Hamiltonian is written in.
- ``sampling`` — ``"auto"`` *(default)* picks per circuit. Variational programs
  (VQE/QAOA) present a *symbolic* circuit at mitigation time — error mitigation
  runs before parameter binding — so ``auto`` enumerates paths up to
  ``truncation_order`` once and reuses them for every parameter set, and
  ``n_samples`` has no effect. A :class:`~divi.qprog.algorithms.TimeEvolution`
  at a fixed ``time`` has concrete angles, so ``auto`` draws ``n_samples``
  random paths there and the knobs swap roles: ``n_samples`` sets the cost and
  ``truncation_order`` does not enter. ``"montecarlo"`` insists on sampling and
  warns when a symbolic circuit forces enumeration; ``"exhaustive"`` always
  enumerates, binding parameters before mitigation so it sees concrete angles.
- ``n_samples`` *(int, default 200)* — Monte Carlo path budget, used only when
  a concrete circuit is sampled (see ``sampling``).
- ``seed`` *(int, optional)* — RNG seed for Monte Carlo reproducibility.
- ``n_twirls`` *(int, default 10)* — Pauli twirl count; ``0`` disables twirling.

ZNE vs QuEPP
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Property
     - ZNE
     - QuEPP
   * - Noise model required?
     - No
     - No
   * - Classical pre-computation
     - None
     - Clifford simulation of ensemble
   * - Circuit overhead
     - ``len(scale_factors)`` — one circuit per scale factor, with no separate
       unmitigated extra (include ``1.0`` in the list if you want one)
     - ``(1 + surviving Pauli paths) × n_twirls``. The default
       ``n_twirls=10`` multiplies the entire path fan-out by 10; use a dry run
       for the actual count
   * - Observable required?
     - No
     - Yes (used for classical simulation)

Estimating Circuit Cost with Dry Run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``format_dry_run(program.dry_run())`` shows QEM paths, Clifford simulations,
and twirl fan-out. QuEPP's fan-out is ``n_paths + 1`` because it also submits
the unmitigated circuit. See :ref:`dry-run` for the complete report contract.

.. warning::

   For concrete-angle programs that actually use Monte Carlo, path counts are
   sampled estimates. Default previews report pre-mitigation circuit
   shape. Use ``force_circuit_generation=True`` for post-rewrite depth and gate
   counts; measurement basis changes add further depth.

.. note::

   ``shot_distribution`` caps shots per mitigation variant, not per evaluation.
   Use :attr:`~divi.pipeline.DryRunReport.total_shots` for the full cost.

Signal Destruction and Automatic Fallback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

QuEPP corrects the noisy quantum result by dividing by the empirical rescaling
factor η.  When noise is so severe that η drops below a safety threshold
(``min_eta=0.1``), the ``1/η`` correction would amplify noise rather than
suppress it.  In this case QuEPP **falls back to the raw noisy value** for that
observable group and emits a summary warning after the evaluation:

.. code-block:: text

   UserWarning: QuEPP: signal destroyed — η fell below the safety threshold
   and mitigation fell back to the raw noisy value. Consider increasing shots
   or reducing noise.

The η diagnostics carry no per-run counts, so each fires at most once for a
whole optimisation rather than once per iteration.

Three further conditions get their own warning:

- **No classical signal.** Every Pauli path of an observable has a negligible
  classical expectation value, so η has no denominator to form and the raw
  noisy value is returned.
- **Negative η.** The noisy Clifford ensemble came back with the opposite sign
  to the exact one. No rescaling repairs that.
- **Amplifying η.** η cleared ``min_eta`` but is still small enough that
  ``1/η`` magnifies the noisy residual several-fold.

QuEPP does not warn on a wide spread in the per-circuit factors. The
diagnostics it does emit are an undefined, negative, or small η.

Shallow Circuit Warning
~~~~~~~~~~~~~~~~~~~~~~~

QuEPP's correction relies on the CPT expansion being a small perturbation of the
target circuit.  When the truncation order K replaces a large fraction of the
non-Clifford rotations, path circuits differ too much from the target for
reliable η estimation.  QuEPP emits a warning when ``K / n_rotations > 33%``:

.. code-block:: text

   UserWarning: QuEPP: truncation order K=2 replaces a large fraction of the
   4 non-Clifford rotations (50%). Mitigation quality may degrade on shallow
   circuits — consider reducing truncation_order or using a deeper circuit.

.. _qem-quepp-pauli-sums:

How QuEPP Handles Sums of Pauli Terms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For an observable that is a sum of Pauli terms, QuEPP treats each term
separately and adds the results. The path weights belong to a specific term,
and each path circuit is evaluated against **that term's** Pauli, not against
the whole observable. So QuEPP measures each Pauli term separately, on both the
classical and the noisy side.

This adds no target-circuit variants and does not change the configured shot
split. Grouping still uses the same distinct Pauli strings, and path circuits
remain deduplicated across terms.

You do not configure any of this, and it does not surface anywhere you look:
:meth:`~divi.circuits.quepp.QuEPP.reduce` maps the per-term values back and
returns one mitigated value per observable you asked for, and a dry run reports
the same group and Pauli-term counts either way, because grouping deduplicates
the Pauli set before it counts them.

Performance Considerations
--------------------------

- **QuEPP**: Cost grows with path count (Monte Carlo budget or exhaustive
  enumeration), twirls, and circuit size.
- **QuEPP on a many-term Hamiltonian**: Pauli-term handling does not by itself
  add target-circuit variants, but each term gets its own paths and
  its own weight (see :ref:`qem-quepp-pauli-sums`). QuEPP warns once ``1/η``
  alone exceeds 5.
- **Budget**: Mitigation increases total shots or circuit evaluations.

.. _qem-multi-observable:

Multi-Observable Programs
-------------------------

Programs that accept several observables in one run (for example
:class:`~divi.qprog.algorithms.TimeEvolution` with
``observable=[O1, O2, ...]`` — see :ref:`time-evolution-multi-observable`)
amortise mitigation cost across the group:

- **ZNE** folds the target circuit once per scale factor for the *whole*
  observable set, not once per observable.  The submitted count is
  ``#scales × #measurement groups``, so observables that commute share a group
  and cost nothing extra, while a non-commuting one adds a group and its own
  ``#scales`` circuits — the fan-out follows the grouping, not the raw
  observable count.
- **QuEPP** shares the target circuit across all observables and dedupes
  path DAGs across observables that produce coincident branches, so the
  classical Clifford simulation is reused.

Both protocols return one mitigated value per input observable, in input
order.


Custom Error Mitigation Protocols
---------------------------------

Protocol authors subclass :class:`~divi.circuits.qem.QEMProtocol` and provide:

- ``name`` — the unique pipeline-axis identifier;
- ``expand(dag, observable)`` — circuit variants plus a
  ``QEMContext`` dictionary;
- ``reduce(quantum_results, context)`` — one mitigated value per requested
  observable; and
- optionally ``post_reduce(contexts)`` for evaluation-wide diagnostics.

``expand`` may consume its input DAG, so copy it explicitly when returning
independent variants. A protocol that overrides measurements owns the remapping
back to the caller’s requested observable order. See
:class:`~divi.circuits.qem.QEMProtocol` and the ZNE and QuEPP implementations
for the complete extension contract. Pipeline integration is automatic through
:class:`~divi.pipeline.stages.QEMStage`.


Next Steps
----------

- :doc:`../api_reference/circuits` — ``QEMProtocol``, ``ZNE``, and QuEPP
- :doc:`../execution_workflows/program_ensembles` — running many mitigated programs together
- :doc:`../execution_workflows/pipelines` — how QEM fits into the circuit pipeline
