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
   import pennylane as qp
   import numpy as np

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
   h2_molecule = qp.qchem.Molecule(
       symbols=["H", "H"],
       coordinates=np.array([[0.0, 0.0, -0.6614], [0.0, 0.0, 0.6614]])
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

**Configuration Options** (same imports as in **Basic Usage** above):

.. code-block:: python

   # Light mitigation (faster, 2 scale factors)
   light_zne = ZNE(
       scale_factors=[1, 3],
       extrapolator=RichardsonExtrapolator(),
   )

   # Heavy mitigation (more accurate, 5 scale factors)
   heavy_zne = ZNE(
       scale_factors=[1, 3, 5, 7, 9],
       extrapolator=RichardsonExtrapolator(),
   )

**Choosing a folding strategy.**  The default
:func:`~divi.circuits.zne.global_fold` folds the entire circuit
(``U · (U†·U)^k · L†·L``, with the tail ``L`` handling fractional
remainders); it is deterministic and a sensible first choice when scale
factors are widely spaced.  For deep circuits, scales close to 1, or
finer-grained noise scaling, swap in
:func:`~divi.circuits.zne.local_fold`, which folds each gate
independently (``G · (G†·G)^k``) and distributes fractional remainders
across a random subset of gates:

.. skip: next

.. code-block:: python

   from divi.circuits.zne import ZNE, local_fold

   # Per-gate folding with fractional scale factors
   zne_local = ZNE(
       scale_factors=[1.0, 1.25, 1.5, 1.75, 2.0],
       folding_fn=local_fold,
   )

``local_fold`` accepts keyword arguments via ``functools.partial`` for
deterministic output (``selection="from_left"`` / ``"from_right"``) or
to skip gates during folding — for example, excluding 2-qubit gates to
isolate single-qubit noise, or excluding everything except ``cx`` to
target 2-qubit gate errors specifically:

.. skip: next

.. code-block:: python

   from functools import partial
   from divi.circuits.zne import ZNE, local_fold

   zne_selective = ZNE(
       scale_factors=[1.0, 1.5, 2.0],
       folding_fn=partial(local_fold, selection="from_left", exclude={"cx"}),
   )

.. note::
   The achievable scale factors form a discrete grid of granularity
   ``2/d`` where ``d`` is the number of foldable gates.  For very small
   ``d`` a requested non-integer scale may snap to a different value;
   ZNE forwards the *effective* scale to the extrapolator so
   extrapolation stays unbiased, and warns if two requested scales
   collapse to the same effective value.

Quantum Enhanced Pauli Propagation (QuEPP)
------------------------------------------

QuEPP is a hybrid classical-quantum protocol based on Clifford Perturbation
Theory (CPT) from `Majumder et al. (2026) <https://arxiv.org/abs/2603.14485>`_.

It works by decomposing the target circuit into a set of Clifford circuits
(Pauli paths) whose expectation values can be computed exactly with a classical
simulator.  The low-order paths capture most of the signal; the residual
higher-order contribution is estimated from the noisy quantum hardware and
corrected with a rescaling factor derived from comparing noisy and ideal values
on the ensemble circuits.

**Basic Usage:**

.. skip: next

.. code-block:: python

   from divi.circuits.quepp import QuEPP
   from divi.qprog import VQE
   from divi.qprog.optimizers import MonteCarloOptimizer
   from divi.backends import QiskitSimulator
   import pennylane as qp
   import numpy as np

   h2_molecule = qp.qchem.Molecule(
       symbols=["H", "H"],
       coordinates=np.array([[0.0, 0.0, -0.6614], [0.0, 0.0, 0.6614]])
   )

   vqe = VQE(
       molecule=h2_molecule,
       qem_protocol=QuEPP(truncation_order=2),
       backend=QiskitSimulator(qiskit_backend="auto"),
       optimizer=MonteCarloOptimizer(),
       max_iterations=10,
   )

   vqe.run()
   print(f"Mitigated energy: {vqe.best_loss:.6f}")

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
     - ``(1 + surviving Pauli paths) × n_twirls`` — twirling is a *separate*
       stage and ``n_twirls`` defaults to **10**, so ``truncation_order=2``
       costs ×400, not ×40.  ``C(n, 1) + ... + C(n, K_T)`` (with ``n`` the
       non-Clifford rotation count and ``K_T`` the truncation order) bounds the
       paths before pruning; read the real factors off a dry run, which shows
       ``QEMStage`` and ``PauliTwirlStage`` separately
   * - Best for
     - Coherent gate noise
     - Uniform noise (e.g. readout error)
   * - Observable required?
     - No
     - Yes (used for classical simulation)

Estimating Circuit Cost with Dry Run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Error mitigation can multiply the number of circuits significantly.  Use
:meth:`~divi.qprog.QuantumProgram.dry_run` to preview the per-stage expansion
before committing to a full run, and pipe the returned reports through
:func:`~divi.pipeline.format_dry_run` to render them as a tree:

.. code-block:: python

   import numpy as np
   import pennylane as qp

   from divi.backends import QiskitSimulator
   from divi.circuits.quepp import QuEPP
   from divi.pipeline import format_dry_run
   from divi.qprog import VQE
   from divi.qprog.optimizers import MonteCarloOptimizer

   h2_molecule = qp.qchem.Molecule(
       symbols=["H", "H"],
       coordinates=np.array([(0.0, 0.0, 0.0), (0.0, 0.0, 0.74)]),
   )

   vqe = VQE(
       molecule=h2_molecule,
       qem_protocol=QuEPP(truncation_order=2, n_twirls=10),
       backend=QiskitSimulator(qiskit_backend="auto"),
       optimizer=MonteCarloOptimizer(),
   )

   # Collect the analytic reports and render a per-stage factor tree per pipeline.
   format_dry_run(vqe.dry_run())

The QEM-relevant entries are how many Pauli paths QuEPP generates, the
Clifford simulation count, and the twirl fan-out — use these to tune
``truncation_order`` and ``n_twirls`` before spending any shots.  The stage's
fan-out is one greater than its reported ``n_paths``: the unmitigated circuit is
submitted alongside the paths, so ``n_paths: 9`` shows as ``×10``.
(``coefficient_threshold`` is disabled on symbolic circuits, which is every
variational program, so a dry run will not show it doing anything.)  See the
:ref:`dry-run` section of the pipelines guide for how to read the per-stage
factor tree (fan-out ``×K`` vs grouping reduction ``÷K``) and for programmatic
access to the reports.

.. warning::

   **Circuit counts are exact except on a sampled path; the reported depth and
   gate counts are not.**  Monte Carlo path selection (a concrete-angle program,
   see ``sampling``) draws its paths at random and deduplicates them, so the
   surviving count is a random variable: two programs sharing one protocol
   instance, or one protocol reused across runs, will not draw the same number.
   The preview reports a sample from an independent stream — reproducible, close,
   but **not** a prediction of any particular run — and says so, marking the stage
   ``path_count: sampled (an estimate, not an exact count)`` and the compact row
   ``sampled count``.  Size a budget for such a run with headroom, or switch to
   ``sampling="exhaustive"``, whose count is deterministic.  Everywhere else the
   count is exact.

   On the default analytic path, mitigation is previewed as placeholders, so the
   ``Summary`` shape figures describe the circuits *entering* the mitigation
   stage.  The error is not a bound in either direction — folding understates
   depth (a 5× fold reads as unfolded), while path substitution can overstate it.
   Pass ``force_circuit_generation=True`` to expand and measure the real circuits
   when you are sizing a hardware budget.

   Reported depth also excludes the basis-change and measurement layer that
   grouping appends, which adds a small amount on top.

.. note::

   On a sampling backend, ``shot_distribution`` caps the shot budget per
   *mitigation variant*, not across the evaluation.  Every variant a protocol
   emits — a folded copy, a Pauli path, a twirl — is its own circuit drawing the
   full capped budget, so a protocol emitting three variants spends the cap three
   times.  Read :attr:`~divi.pipeline.DryRunReport.total_shots` for the figure
   rather than assuming the cap is global.

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
whole optimization rather than once per iteration.

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
- **Use a deeper circuit** (more qubits or Trotter steps).
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
- **Transpiler optimization.** The ensemble circuits are Clifford, so an
  optimizing transpiler compresses them much further than the target, and they
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

This costs nothing extra in circuits or shots. Grouping builds its measurement
basis from the *set* of distinct Pauli strings, which is the same set either
way, so the submitted circuits and the shot split are identical to measuring
``H`` as one observable. Path circuits depend only on the branch choices, never
on the term, so they are still deduplicated across terms.

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
- **QuEPP on a many-term Hamiltonian**: each Pauli term gets its own paths and
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

You can implement custom error mitigation strategies by inheriting from
:class:`~divi.circuits.qem.QEMProtocol`.  The protocol operates on Qiskit
:class:`~qiskit.dagcircuit.DAGCircuit` bodies — the same IR the rest of the
pipeline uses — and must implement three members:

.. code-block:: python

   import copy
   from collections.abc import Sequence
   from typing import Any

   import numpy as np
   from qiskit.dagcircuit import DAGCircuit
   from divi.backends import MaestroSimulator
   from divi.circuits.qem import QEMContext, QEMProtocol

   class WeightedAveraging(QEMProtocol):
       """A simple protocol that runs the circuit twice and averages results."""

       @property
       def name(self) -> str:
           return "weighted_avg"

       def expand(self, dag: DAGCircuit, observable=None):
           """Return circuits to execute and a reduce-time context.

           ``expand`` *consumes* the input ``dag`` — implementations may
           mutate it, and downstream stages may mutate the returned DAGs
           in place.  When you need multiple distinct variants, deep-copy
           the dag explicitly (as shown below); reusing the same reference
           would cause later edits to affect every slot it appears in.
           The optional ``observable`` argument carries the observable being
           measured (as a Qiskit
           :class:`~qiskit.quantum_info.SparsePauliOp`) — hybrid protocols
           like QuEPP use it for classical pre-computation.
           """
           # Run the circuit twice as two independent DAG copies so later
           # pipeline stages can mutate each one without interference.
           return (copy.deepcopy(dag), dag), {}

       def reduce(
           self, quantum_results: Sequence[Any], context: QEMContext
       ) -> list[float]:
           """Combine the quantum results into one mitigated value per observable.

           ``quantum_results`` has one entry per circuit returned by ``expand``, in
           the same order, and each entry is itself a list of per-observable
           expectation values — so the average is taken across circuits, position
           by position, and the return value is a list, not a scalar.
           """
           per_circuit = [np.atleast_1d(r) for r in quantum_results]
           return list(np.mean(per_circuit, axis=0))

   # Pass the custom protocol when constructing any variational program
   vqe = VQE(
       molecule=h2_molecule,
       qem_protocol=WeightedAveraging(),
       optimizer=MonteCarloOptimizer(),
       backend=MaestroSimulator(),
   )

**Key Members to Implement:**

- ``name`` *(property)* — Unique protocol name used as the pipeline axis identifier
- ``expand(dag, observable)`` — Generate one or more Qiskit
  :class:`~qiskit.dagcircuit.DAGCircuit` bodies to execute on the quantum
  backend and a ``QEMContext`` carrying any classical side-channel data for
  the reduce phase.  Return a ``tuple[tuple[DAGCircuit, ...], QEMContext]``.
- ``reduce(quantum_results, context)`` — Combine the per-circuit results with
  the ``QEMContext`` into a ``list[float]``, one mitigated value per
  **requested** observable.  Each entry of ``quantum_results`` is itself a
  sequence of expectation values, one per observable the measurement stage was
  asked for — which is the requested set unless the protocol declared an
  ``OBSERVABLE_OVERRIDE`` (see below).
- ``post_reduce(contexts)`` *(optional)* — Called once after all per-group
  ``reduce`` calls in an evaluation.  Override to inspect the collected contexts
  and emit summary diagnostics (e.g. QuEPP's η diagnostics).
  The default implementation is a no-op.

A protocol that needs finer-grained measurements than the caller requested can
set the ``OBSERVABLE_OVERRIDE`` key on its context to a
``tuple[SparsePauliOp, ...]``; the QEM stage applies it to the emitted circuits,
so the measurement stage reports one value per entry.  The protocol then owns
the remapping — its ``reduce`` must still return one value per *requested*
observable.  QuEPP uses this to measure each Pauli term of a sum separately
(see :ref:`qem-quepp-pauli-sums`).

.. note::
   When a ``qem_protocol`` is provided, the :doc:`circuit pipeline <pipelines>`
   automatically wraps it in a :class:`~divi.pipeline.stages.QEMStage`.
   During execution, ``expand`` is called in the pipeline's *expand* pass and
   ``reduce`` is called in the *reduce* pass — you don't need to manage
   pipeline integration yourself.

Next Steps
----------

- :doc:`../api_reference/circuits` — ``QEMProtocol``, ``ZNE``, and QuEPP
- :doc:`program_ensembles` — running many mitigated programs together
- :doc:`pipelines` — how QEM fits into the circuit pipeline
