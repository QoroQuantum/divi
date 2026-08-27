Improving Results with Error Mitigation
========================================

Divi provides built-in quantum error mitigation (QEM) to improve results from
noisy quantum hardware. Two built-in protocols ship with the library:

- **Zero Noise Extrapolation (ZNE)** — runs circuits at artificially increased
  noise levels and extrapolates to the zero-noise limit.
- **Quantum Enhanced Pauli Propagation (QuEPP)** — decomposes the circuit into
  Clifford Pauli paths, simulates them classically, and corrects the noisy
  quantum result with an empirical rescaling factor.

Pass either protocol into variational programs (for example :class:`~divi.qprog.algorithms.VQE`
or :class:`~divi.qprog.algorithms.QAOA`) with the ``qem_protocol`` argument. You can also
subclass :class:`~divi.circuits.qem.QEMProtocol` for custom mitigation; see
`Custom Error Mitigation Protocols`_ below.

.. list-table:: Choosing a mitigation protocol
   :header-rows: 1
   :widths: 18 32 25 25

   * - Protocol
     - Reach for it when
     - Cost driver
     - Main limitation
   * - ZNE
     - The backend noise can be amplified predictably and several folded
       evaluations are affordable.
     - Scale factors × measurement groups
     - Extrapolation becomes unstable when amplification does not track the
       target noise.
   * - QuEPP
     - Clifford-path simulation is tractable and the residual signal survives
       the target noise.
     - Path count × twirls, plus the target circuit
     - Signal destruction and cancellation can amplify statistical error.

Use :meth:`~divi.qprog.QuantumProgram.dry_run` before a long mitigated run. It
shows structural circuit expansion; it cannot predict the number of shots
needed to reach a particular statistical error bar.

Zero Noise Extrapolation (ZNE)
------------------------------

Divi's ZNE runs the target circuit at several amplified noise levels and
extrapolates the per-scale expectation values back to the zero-noise limit.
Folding and extrapolation are both built-in — :class:`~divi.circuits.zne.ZNE`
ships with global-unitary folding (:func:`~divi.circuits.zne.global_fold`)
by default and uses :class:`~divi.circuits.zne.RichardsonExtrapolator`
unless a custom extrapolator is provided.  Both integer and fractional
scale factors are supported; for per-gate folding on deep circuits or
scales close to 1, switch to :func:`~divi.circuits.zne.local_fold`.

**Basic Usage:**

.. dashboard-example: zne

.. code-block:: python

   from divi.circuits.zne import ZNE, RichardsonExtrapolator
   from divi.qprog import VQE
   from divi.qprog.optimizers import MonteCarloOptimizer
   from divi.backends import QiskitSimulator
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

   vqe = VQE(
       molecule=h2_molecule,
       qem_protocol=zne_protocol,
       optimizer=MonteCarloOptimizer(),
       backend=QiskitSimulator(qiskit_backend="auto"),
       max_iterations=10,
   )

   vqe.run()
   print(f"Mitigated energy: {vqe.best_loss:.6f}")

Use two or three scales for lower cost and more scales for a richer
extrapolation fit. Additional scales also increase cost and statistical noise;
they do not guarantee greater accuracy. The default :func:`~divi.circuits.zne.global_fold` is
deterministic and suits widely spaced scales. For deep circuits, scales near 1,
or per-gate control, use :func:`~divi.circuits.zne.local_fold`:

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
  number of non-Clifford gates).  It governs path enumeration for
  ``sampling="exhaustive"`` **and** for the montecarlo fallback on symbolic
  circuits (see ``sampling``).
- ``coefficient_threshold`` *(float, optional)* — Prune paths whose absolute
  weight falls below this threshold during DFS enumeration (``sampling="exhaustive"``
  only; disabled on symbolic circuits, whose angle magnitudes are unknown).
  The weight is the path's trigonometric product, independent of the
  observable's coefficients, so the threshold means the same thing whatever
  scale the Hamiltonian is written in.
- ``sampling`` — ``"exhaustive"`` enumerates paths up to ``truncation_order``
  (deterministic; cost grows with order and circuit size).  ``"montecarlo"``
  *(default)* draws ``n_samples`` random paths, **but only on concrete
  (parameter-bound) circuits**.  Variational programs (VQE/QAOA) present a
  *symbolic* circuit at mitigation time — error mitigation runs before parameter
  binding — so ``montecarlo`` warns and falls back to exhaustive enumeration
  governed by ``truncation_order``; ``n_samples`` has no effect in that case.
  A :class:`~divi.qprog.algorithms.TimeEvolution` at a fixed ``time`` has
  concrete angles, so montecarlo stays live there and the knobs swap roles:
  ``n_samples`` sets the cost and ``truncation_order`` does not enter.
- ``n_samples`` *(int, default 200)* — Monte Carlo path budget, used only on the
  concrete-circuit montecarlo path (see ``sampling``).
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
   * - Best for
     - Coherent gate noise
     - Uniform noise (e.g. readout error)
   * - Observable required?
     - No
     - Yes (used for classical simulation)

Estimating Circuit Cost with Dry Run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use :meth:`~divi.qprog.QuantumProgram.dry_run` before committing to mitigation:
``format_dry_run(program.dry_run())`` shows QEM paths, Clifford simulations,
and twirl fan-out. QuEPP's fan-out is ``n_paths + 1`` because it also submits
the unmitigated circuit. See :ref:`dry-run` for the complete report contract.

.. warning::

   For concrete-angle programs that actually use Monte Carlo, path counts are
   sampled estimates; budget with headroom or use ``sampling="exhaustive"``.
   Default previews report pre-mitigation circuit
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

If you see this warning frequently, consider:

- **Increasing the number of shots** to reduce statistical noise in η.
- **Enabling Pauli twirling** (``n_twirls > 0``) to convert coherent noise into
  stochastic noise that QuEPP handles more gracefully.
- **Lowering the noise level** (e.g. using a less noisy backend or reducing
  circuit depth).

Three further conditions get their own warning, because each calls for a
different response:

- **No classical signal.** Every Pauli path of an observable has a negligible
  classical expectation value, so η has no denominator to form and the raw
  noisy value is returned. Check that the observable's coefficients are not all
  negligible, and that the circuit's final Clifford layer leaves the
  back-propagated Pauli diagonal.
- **Negative η.** The noisy Clifford ensemble came back with the opposite sign
  to the exact one. No rescaling repairs that, so it points at a misconfigured
  backend or a noise level past the protocol's usable range rather than at a
  recoverable estimate.
- **Amplifying η.** η cleared ``min_eta`` but is still small enough that
  ``1/η`` magnifies the noisy residual several-fold. The mitigated value is
  then *noisier* than the unmitigated one even though it is less biased.

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

This typically occurs on small circuits (< 10 qubits) where the number of
non-Clifford rotations is comparable to K.  The paper validates QuEPP on
49-qubit circuits with hundreds of rotations.

If you see this warning:

- **Reduce truncation_order** to ``K=1`` to cut the number of enumerated
  branches.  (``sampling="montecarlo"`` does *not* help here: on the symbolic
  circuits variational programs produce it falls back to exhaustive enumeration
  anyway — see ``sampling`` above.)
- If the problem already requires more non-Clifford rotations, the warning may
  disappear; do not deepen a circuit solely to satisfy QuEPP.
- **Use ZNE instead** for shallow circuits where QuEPP is unreliable.

.. _qem-quepp-assumptions:

What QuEPP Assumes About the Noise
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

QuEPP infers one rescaling factor from how much noise the Clifford ensemble
suffers relative to the target. That is sound only when the ensemble is a fair
proxy for the target, which fails in ways worth knowing:

- **Coherent (unitary) errors.** A systematic over-rotation accumulates with
  depth in the target, while the ensemble's exact ±π/2 Clifford replacements
  are far less sensitive to it. η therefore comes back close to 1 and the
  correction barely fires; at larger coherent strengths the result can be worse
  than the unmitigated value. Keep ``n_twirls > 0`` on hardware: Pauli twirling
  converts coherent errors into the stochastic channel QuEPP assumes, which is
  why ``n_twirls`` defaults to 10.
- **Non-unital noise.** Amplitude damping and thermal relaxation bias the state
  toward :math:`|0\rangle`, which *shifts* an expectation value rather than
  merely shrinking it. A multiplicative ``1/η`` cannot undo an additive shift,
  so the mitigated value can be further from the truth than the unmitigated one
  while η itself looks healthy.
- **Noise carried only by the rotations being replaced.** A cos branch
  substitutes an identity for a rotation, so error attached exclusively to the
  rotation gate has no counterpart in the ensemble and cannot be inferred from
  it. This is a limit of the protocol rather than of any particular backend.
- **Transpiler optimisation.** The ensemble circuits are Clifford, so an
  optimising transpiler compresses them much further than the target, and they
  then see less noise than it does. On
  :class:`~divi.backends.QiskitSimulator` pass ``optimization_level=0``
  alongside your noise model; on hardware, transpile the ensemble and the
  target the same way.

A wide spread in the per-circuit factors is not on its own a sign of trouble, so
QuEPP does not warn on it. The diagnostics it does emit — an undefined,
negative, or small η — are the ones that indicate an unusable correction.

.. _qem-quepp-pauli-sums:

How QuEPP Handles Sums of Pauli Terms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most real observables — a molecular Hamiltonian, a QUBO cost operator — are
sums :math:`H = \sum_i c_i P_i` rather than single Pauli strings. Pauli
propagation back-propagates *one* Pauli at a time, so QuEPP treats each term
separately and adds the results:

.. math::

   \langle H \rangle = \sum_i c_i \sum_p w_{i,p}\,
   \mathrm{Tr}\!\left[\rho\, C_{i,p}(P_i)\right]

The path weights :math:`w_{i,p}` belong to a specific term, and each path
circuit is evaluated against **that term's** Pauli — not against the whole
observable. So QuEPP measures each Pauli term separately, on both the classical
and the noisy side.

This adds no target-circuit variants and does not change the configured shot
split. Grouping still uses the same distinct Pauli strings, and path circuits
remain deduplicated across terms. Higher reconstruction variance can nonetheless
require more shots for a target accuracy.

You do not configure any of this, and it does not surface anywhere you look:
:meth:`~divi.circuits.quepp.QuEPP.reduce` maps the per-term values back and
returns one mitigated value per observable you asked for, and a dry run reports
the same group and Pauli-term counts either way, because grouping deduplicates
the Pauli set before it counts them.

Performance Considerations
--------------------------

- **ZNE**: Expect roughly one backend evaluation per scale factor per
  unmitigated evaluation (plus extrapolation overhead on the classical side).
- **QuEPP**: Cost grows with path count (Monte Carlo budget or exhaustive
  enumeration), twirls, and circuit size. Classical Clifford simulation of
  paths is comparatively cheap next to quantum shots.
- **QuEPP on a many-term Hamiltonian**: Pauli-term handling does not by itself
  add target-circuit variants, but each term gets its own paths and
  its own weight (see :ref:`qem-quepp-pauli-sums`), so the CPT sum has more,
  individually larger, terms than a per-observable sum would. The reconstruction
  is unbiased either way, but its variance is higher, and reaching a given
  accuracy on a chemistry Hamiltonian can take more shots than a single-Pauli
  observable needs. ``1/η`` is a floor on how much the error bar grows: the
  noisy residual's own shot noise is weighted by the path weights, so a
  reconstruction leaning on heavy cancellation grows it further. QuEPP warns
  once ``1/η`` alone exceeds 5.
- **Budget**: Mitigation increases total shots or circuit evaluations; use
  :meth:`~divi.qprog.QuantumProgram.dry_run` to preview expansion before a long
  run.

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
  path DAGs across observables that produce coincident branches, so a
  large fraction of the classical Clifford simulation is reused.

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
