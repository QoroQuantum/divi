Program Ensembles and Workflows
================================

A :class:`~divi.qprog.ensemble.ProgramEnsemble` runs multiple quantum programs
in parallel — handling scheduling, circuit batching, progress tracking, and
result aggregation.  Typical use cases include parameter sweeps, molecular
dissociation curves, problem decomposition, and algorithm comparison.

An ensemble can also run over several *rounds*, choosing each round's programs
from what the previous round measured — the basis for iterative refinement and
other adaptive workflows. See `The Workflow Lifecycle`_.

Use this page in increasing order of control:

1. choose a ready-made workflow under `Built-in Ensemble Workflows`_;
2. inspect its results, or aggregate a partitioning workflow;
3. use `The Workflow Lifecycle`_ for adaptive rounds; and
4. use `Custom Ensemble Workflows`_ and `Circuit Batching`_ only when you need
   custom orchestration or dispatch control.

Built-in Ensemble Workflows
----------------------------

Divi provides several ready-made ensemble workflows.  Each is covered in
detail on its own page — this section gives a quick overview and links.

**VQE Hyperparameter Sweeps**
   :class:`~divi.qprog.workflows.VQEHyperparameterSweep` runs VQE across multiple
   molecular configurations (bond lengths, ansätze) in parallel.
   See :doc:`../algorithms/ground_state_energy_estimation_vqe` for configuration and examples.

**Time Evolution Trajectories**
   :class:`~divi.qprog.workflows.TimeEvolutionTrajectory` runs one time-evolution
   program per time point and collects expectation values into a trajectory.
   See :doc:`../algorithms/hamiltonian_time_evolution` for full details.

**Problem Decomposition (Graph / QUBO / Matching)**
   :class:`~divi.qprog.workflows.PartitioningProgramEnsemble` decomposes a
   large :class:`~divi.qprog.problems.QAOAProblem` into sub-problems, solves
   each partition with QAOA (or PCE / IterativeQAOA), and stitches the
   per-partition results into a global solution using a configurable
   aggregation strategy (see `Aggregation Strategies`_).  Graph, QUBO, and
   matching partitioning are all covered in
   :doc:`../algorithms/combinatorial_optimization_qaoa_pce`.

**Localised Active-Space SQD**
   :class:`~divi.qprog.workflows.LASSQD` partitions a molecule's active space
   into fragments, runs one VQE per fragment against its own
   mean-field-embedded effective Hamiltonian, and recovers the ground state
   via sample-based quantum diagonalisation. See
   :doc:`../algorithms/localized_active_space_sqd` for fragment specification, automatic
   fragmentation, and the accuracy characteristics of the reported energy.

Aggregation Strategies
----------------------

For :class:`~divi.qprog.workflows.PartitioningProgramEnsemble`, each solved
partition returns several candidate
bitstrings ranked by probability. An *aggregation strategy* stitches these
per-partition candidates into a single global solution — the choice of strategy
controls the quality/cost trade-off across the full problem. ``aggregate_results``
and ``get_top_solutions`` accept a ``strategy`` — an
:class:`~divi.qprog.AggregationStrategy` — and default to
:class:`~divi.qprog.BeamSearchStrategy`.

Beam search
~~~~~~~~~~~

:class:`~divi.qprog.BeamSearchStrategy` explores candidates left-to-right across
partitions, keeping the best partial solutions at each step. It takes two
parameters:

- ``beam_width`` — how many partial solutions are kept after each partition step.
- ``n_partition_candidates`` — how many candidates to extract from each partition (defaults to ``beam_width``).

.. skip: next

.. code-block:: python

   from divi.qprog import BeamSearchStrategy

   # Greedy (default): single best candidate per partition
   solution, energy = qaoa_partition.aggregate_results(
       strategy=BeamSearchStrategy(beam_width=1)
   )

   # Beam search: keep the top 5 partial solutions after each partition step
   solution, energy = qaoa_partition.aggregate_results(
       strategy=BeamSearchStrategy(beam_width=5)
   )

   # Wider candidate pool with narrow beam: consider 10 candidates per partition
   # but only keep the best 3 partial solutions after each step
   solution, energy = qaoa_partition.aggregate_results(
       strategy=BeamSearchStrategy(beam_width=3, n_partition_candidates=10)
   )

   # Exhaustive: try all candidate combinations (expensive for many partitions)
   solution, energy = qaoa_partition.aggregate_results(
       strategy=BeamSearchStrategy(beam_width=None)
   )

**When to use beam search**

- **Greedy** (``beam_width=1``): Fast, good for problems with low inter-partition coupling.
- **Bounded beam** (``beam_width=k``): Good trade-off for problems with moderate coupling between partitions. Start with ``beam_width=3`` and increase if solution quality improves.
- **Exhaustive** (``beam_width=None``): Guarantees the global optimum across all candidate combinations, but scales exponentially with the number of partitions.

.. tip::

   The two settings are independent, and either can be the larger one. Raising ``n_partition_candidates`` above ``beam_width`` lets each partition propose many alternatives (wider local search) while the beam stays narrow. Raising ``beam_width`` above ``n_partition_candidates`` goes the other way: the beam holds *combinations* of per-partition choices, so across ``P`` partitions it can retain up to ``n_partition_candidates ** P`` of them. That is usually the cheaper way to widen the search, because the number of candidates fetched per partition — and therefore the branching factor at every step — stays small.

Hierarchical aggregation
~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~divi.qprog.HierarchicalStrategy` is a divide-and-conquer alternative:
partitions are split into groups of ``group_size``, each group is solved
independently with a beam of width ``max_per_group``, and the resulting group
pools are combined in a pairwise merge tree. Deferring cross-group commitment
lets each group keep prefixes that a single left-to-right beam would prune
early. The trade-off is that interactions spanning partitions in *different*
groups are only scored at merge time, not during per-group pruning.

The knobs trade solution quality against cost:

- ``group_size`` — partitions per group (larger groups defer more commitment).
- ``k_per_partition`` — candidates fetched from each partition.
- ``max_per_group`` — solutions kept per group and per merge level; the main quality/cost dial.
- ``merge_width`` — limits how many solutions from each group are paired during a merge step, capping each merge's cost to ``merge_width`` squared. Lower it to tame cost on problems with many partitions; the default (``None``) uses all ``max_per_group`` entries.

Unlike beam search, ``top_n`` does **not** widen the search for the hierarchical
strategy — it only sets how many of the final solutions are returned. Calling
:meth:`~divi.qprog.workflows.PartitioningProgramEnsemble.get_top_solutions` with a
larger ``n`` never inflates the search cost.

.. skip: next

.. code-block:: python

   from divi.qprog import HierarchicalStrategy

   solution, energy = qaoa_partition.aggregate_results(
       strategy=HierarchicalStrategy(
           group_size=4, k_per_partition=20, max_per_group=200, merge_width=50
       )
   )

**When to use hierarchical aggregation**

- **Many partitions with localised coupling**: when strongly-coupled partitions can land in the same group, each group explores more alternatives before committing to a cross-group assignment — recovering combinations a greedy left-to-right beam would prune early.
- **Finer cost control**: ``merge_width`` caps per-merge cost independently of ``max_per_group``, a knob beam search does not offer.
- **Prefer beam search** when coupling is global or unpredictable across partition boundaries: groups are formed in partition order, so the grouping then provides little benefit over a wider beam.

.. _ensemble-top-n:

Top-N Solutions
---------------

:class:`~divi.qprog.workflows.PartitioningProgramEnsemble` exposes a
``get_top_solutions`` method that returns multiple ranked global solutions. It
accepts the same ``strategy`` argument.

.. skip: next

.. code-block:: python

   from divi.qprog import BeamSearchStrategy

   top_solutions = qaoa_partition.get_top_solutions(
       n=5, strategy=BeamSearchStrategy(beam_width=5, n_partition_candidates=10)
   )

   # Return type is problem-dependent:
   #   Graph  → list[(node_indices, energy)]
   #   QUBO   → list[(solution_array, energy)]
   for rank, (solution, energy) in enumerate(top_solutions, 1):
       print(f"{rank}. Energy: {energy:.6f}, Solution: {solution}")

This is useful when you want to inspect alternative solutions or post-process
candidates with domain-specific constraints. Beam search widens its beam to at
least ``n`` when needed; hierarchical returns the ``n`` best from its final
merged pool (without widening the search, as noted above).

For constrained problems such as maximum-weight matching, partition boundaries
can produce globally invalid raw candidates even when each partition candidate
is locally valid. ``aggregate_results`` keeps the default forgiving behaviour and
repairs matching conflicts. To inspect only raw candidates that are already
valid globally, use ``get_top_solutions(..., strict=True)``. The returned list
may contain fewer than ``n`` entries.

.. _ensemble-sample-solution:

Sampling from Pre-Trained Parameters
------------------------------------

:meth:`~divi.qprog.ensemble.ProgramEnsemble.sample_solution` mirrors the
standalone
:meth:`~divi.qprog.SolutionSamplingMixin.sample_solution` across every
sub-program in one call. Use it when you already have trained parameters for
each partition (e.g. from a prior ``run()``, a loaded checkpoint, or an
external training routine) and only need to re-sample — no EXPECTATION jobs
are dispatched.

Two usage paths:

* ``params_per_program=None`` — each sub-program uses its own
  ``_best_params``. After a prior ``run()`` on the same ensemble, just call
  ``ensemble.sample_solution(blocking=True)``.
* ``params_per_program={program_id: params, ...}`` — pass explicit
  per-partition parameters. Unknown program IDs raise :class:`ValueError`;
  program IDs present in the ensemble but missing from the dict fall back to
  that program's own ``_best_params`` and emit a single
  :class:`UserWarning` listing all fallbacks (silence with
  ``suppress_strict_warning=True``).

.. skip: next

.. code-block:: python

   # Re-sample a previously trained partitioning ensemble and aggregate
   # the global solution — without re-paying for the optimizer.
   ensemble.sample_solution(blocking=True)
   solution, energy = ensemble.aggregate_results(
       strategy=BeamSearchStrategy(beam_width=3)
   )

   # Or: bring trained parameters in from elsewhere (per partition).
   ensemble.sample_solution(
       params_per_program={pid: params[pid] for pid in pids},
       blocking=True,
   )

The full lifecycle infrastructure (executor pool, merged batching,
progress UI, cancellation, ``blocking`` / non-blocking semantics) is
shared with :meth:`~divi.qprog.ensemble.ProgramEnsemble.run`. No sub-program
mutates its own optimizer-side state (``best_params``, ``losses_history``,
``current_iteration``).

Separate Optimisation and Sampling Backends
--------------------------------------------

Programs that expose :meth:`~divi.qprog.SolutionSamplingMixin.sample_solution`
accept an optional ``sampling_backend`` in addition to their normal ``backend``
(also readable as a ``sampling_backend`` property). The normal backend
executes optimisation and other circuit evaluations; the sampling backend
executes only the final solution measurement. If ``sampling_backend`` is
omitted, both phases use ``backend`` as before.

Backend selection follows this precedence, for both a standalone program's
``sample_solution`` and ``ensemble.sample_solution``:

1. A one-call ``backend=...`` passed to ``sample_solution``.
2. The configured ``sampling_backend``.
3. The normal ``backend``.

Solution-sampling ensembles accept the same ``sampling_backend`` constructor
argument — step 2 above is then the *ensemble's* ``sampling_backend``. Every
sub-program must expose ``sample_solution`` once it is set —
:meth:`~divi.qprog.ensemble.ProgramEnsemble.run` raises ``TypeError`` upfront
otherwise — and ``run`` trains every sub-program before sampling any of them.
Final measurements still participate in the ensemble's merged batching,
cancellation, progress reporting, and accounting.

.. skip: next

.. code-block:: python

   ensemble = PartitioningProgramEnsemble(
       ...,
       backend=optimisation_backend,
       sampling_backend=sampling_backend,
   )
   ensemble.run()

.. note::

   A sub-program's own ``sampling_backend`` is ignored while it runs inside an
   ensemble — only the ensemble's applies. It matters only when that
   sub-program's ``sample_solution`` is called directly, outside an ensemble.

For example, a one-off re-sampling job can use a third backend without changing
the ensemble's persistent configuration:

.. skip: next

.. code-block:: python

   ensemble.sample_solution(
       backend=one_off_sampling_backend,
       blocking=True,
   )

In merged mode, every sub-program's final measurement is submitted together to
the selected backend, and each sub-program's own ``backend`` is restored once
``join()`` returns — including on failure and cancellation.

.. warning::

   :meth:`~divi.qprog.ensemble.ProgramEnsemble.run_one_round` splits the same
   way when ``sampling_backend`` is set, but only for ``blocking=True``:
   sampling needs training to finish first, so ``blocking=False`` raises
   ``ValueError``.


.. _ensemble-lifecycle:

The Workflow Lifecycle
----------------------

:meth:`~divi.qprog.ensemble.ProgramEnsemble.run` is the entry point for every
ensemble. It blocks until the workflow finishes and returns the ensemble, so
results are ready to aggregate as soon as it returns:

.. skip: next

.. code-block:: python

   results = ensemble.run().aggregate_results()

Internally ``run()`` drives a loop of *rounds*. Each round materialises a fresh
program map, executes it, and folds the results into a workflow state that the
next round can use. The loop runs in this order:

1. ``initial_state()`` — once, producing the first state.
2. ``is_complete(state)`` — if ``True``, stop here.
3. ``create_programs(state)`` — materialise this round's programs.
4. Execute those programs in parallel.
5. ``update_state(state)`` — fold the results into the next state.
6. Back to step 2.

The check at step 2 happens *before* each round, including the first, so a
state that already satisfies ``is_complete`` runs zero rounds. One required
hook and three optional ones fill in that loop:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Hook
     - Responsibility
   * - ``create_programs(state)``
     - Populate ``self.programs`` for the coming round. **Required.**
   * - ``initial_state()``
     - The state handed to the first round. Defaults to ``None``.
   * - ``update_state(state)``
     - Reduce the finished round's results into the next state.
   * - ``is_complete(state)``
     - ``True`` to stop. Defaults to stopping after one round.

Because the defaults stop after a single round, a one-shot ensemble only has
to implement ``create_programs`` — which is why the built-in workflows behave
as simple parallel dispatchers.

.. important::

   Each round replaces the program map, so after ``run()`` only the **final**
   round's programs remain in ``self.programs``. A multi-round ensemble must
   carry anything it needs across rounds in the state returned by
   ``update_state`` — the inherited ``aggregate_results`` sees the last round
   only.

Multi-round workflows
~~~~~~~~~~~~~~~~~~~~~

Override the remaining hooks to make an ensemble adaptive — each round's
programs can depend on what the previous round measured. Below, each round
samples bond lengths around the best geometry the last round found and halves
the search width. ``bond_lengths_around``, ``make_vqe``, and ``best_geometry``
stand in for your own code; only the four hooks matter here:

.. skip: next

.. code-block:: python

   class SelfRefiningSweep(ProgramEnsemble):
       def initial_state(self):
           return {"center": 0.74, "spread": 0.10}

       def create_programs(self, state):
           super().create_programs()
           self.programs = {
               f"d_{i}": make_vqe(d, self.backend)
               for i, d in enumerate(bond_lengths_around(state))
           }

       def update_state(self, state):
           # self.programs still holds the round that just finished.
           return {
               "center": best_geometry(self.programs),
               "spread": state["spread"] / 2,
           }

       def is_complete(self, state):
           return state["spread"] < 1e-3

       def aggregate_results(self):
           return self.workflow_state

   ensemble = SelfRefiningSweep(backend).run(max_rounds=8)

``max_rounds`` caps the loop even when ``is_complete`` never returns ``True``.
Leave it as ``None`` to run until convergence.

Interrupting a running workflow with :kbd:`Ctrl-C` cancels the current round and
stops the loop: the round is recorded as ``CANCELLED`` and no further round
starts. The state is left at the last round that completed, so a partial round's
results never reach it — whether the interrupt landed in the dispatch or in
``update_state``.

.. note::

   ``batch_config`` is fixed for the whole run while the program count can
   change per round, so an ensemble that grows across rounds can hit the
   wait-for-all barrier limit partway through. Set ``max_batch_size`` or
   ``max_concurrent_programs`` explicitly (see :ref:`circuit-batching`) when
   later rounds may produce many more programs than the first.

Inspecting what happened
~~~~~~~~~~~~~~~~~~~~~~~~

After ``run()`` returns, three attributes describe the workflow:

- ``workflow_state`` — the latest state, converged or not.
- ``stop_reason`` — a :class:`~divi.qprog.WorkflowStatus`: ``COMPLETE``,
  ``MAX_ROUNDS``, ``FAILED``, or ``CANCELLED``, and ``None`` before the first
  ``run()``.
- ``round_history`` — a tuple of immutable
  :class:`~divi.qprog.RoundRecord` entries, one per round.

Each record carries the round number, program count, the round's circuit and
runtime *deltas*, a status, and — for a failed round — the formatted
exception:

.. skip: next

.. code-block:: python

   ensemble.run(max_rounds=5)

   print(ensemble.stop_reason)     # e.g. WorkflowStatus.MAX_ROUNDS
   for record in ensemble.round_history:
       print(record.number, record.program_count, record.circuit_count, record.status)

``total_circuit_count`` and ``total_run_time`` remain lifetime totals across
every round and every run; the per-round deltas live in ``round_history``.
Starting a new ``run()`` resets the workflow state and round history but keeps
those lifetime totals.

When a round fails — materialising its programs, executing them, or reducing
them — ``run()`` records the failed round and then *raises*, so ``FAILED`` is
something you catch, not something you find on a returned ensemble. The
exception is whatever the round raised, so catch broadly:

.. skip: next

.. code-block:: python

   try:
       ensemble.run(max_rounds=5)
   except Exception:
       failed = ensemble.round_history[-1]
       print(f"round {failed.number} failed: {failed.error}")

Controlling a single round
~~~~~~~~~~~~~~~~~~~~~~~~~~

:meth:`~divi.qprog.ensemble.ProgramEnsemble.run_one_round` dispatches an
already-materialised program map exactly once, and is the lower-level control
surface behind ``run()``. Prefer ``run()``; reach for this only when you need
to dispatch without blocking, or to drive rounds yourself:

.. skip: next

.. code-block:: python

   ensemble.create_programs()
   ensemble.run_one_round(blocking=False)
   # ... do other work ...
   ensemble.join()

If you materialise a program map yourself and then call ``run()``, that map is
used as the first round rather than being rebuilt — so calling
``create_programs()`` beforehand is safe. It just isn't needed: ``run()``
materialises each round for you.

See :ref:`ensemble-sample-solution` for how ``sampling_backend`` restricts
``blocking=False`` here.

.. _ensemble-reporting:

Progress Reporting
------------------

Every ensemble accepts a ``reporting_level`` controlling how much live progress
is rendered. It is a :class:`~divi.qprog.ReportingLevel`:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Level
     - Rows shown
   * - ``COMPACT``
     - Workflow/round, preparation, and batch rows. Successful program rows are
       hidden; failing ones are revealed so they stay diagnosable. The default.
   * - ``FULL``
     - Workflow/round, preparation, batch, and every program row.
   * - ``OFF``
     - No display at all. ``round_history`` is still recorded.

Reach for ``FULL`` when you need to watch one specific program — say a
partition that keeps failing — and for ``OFF`` in notebooks, logging
pipelines, or scripts whose stdout you parse.

.. skip: next

.. code-block:: python

   from divi.qprog import ReportingLevel

   ensemble = PartitioningProgramEnsemble(
       problem=problem,
       n_layers=2,
       backend=backend,
       optimizer=optimizer,
       reporting_level=ReportingLevel.FULL,
   )

The equivalent string works too — ``reporting_level="full"`` is accepted and
coerced — and an unrecognised value raises :class:`ValueError` rather than
silently falling back.

Setting the ``DIVI_DISABLE_PROGRESS`` environment variable to a truthy value
(``1``, ``true``, ``yes``, ``on``) suppresses progress output from standalone
programs and ensembles regardless of the level — useful in CI. Ordinary log
messages are unaffected, and round history is still retained.

Custom Ensemble Workflows
-------------------------

Subclass :class:`~divi.qprog.ensemble.ProgramEnsemble` and implement
``create_programs``. Override the state hooks for adaptive rounds and
``aggregate_results`` for a custom return shape. The multi-round example above
is schematic; ``make_vqe`` and its geometry helpers stand for application code.
A one-round subclass needs only this contract:

.. skip: next

.. code-block:: python

   class MyEnsemble(ProgramEnsemble):
       def create_programs(self, state=None):
           super().create_programs()
           self.programs = {"job": make_program(self.backend)}

.. _ensemble-dry-run:

Inspecting an Ensemble Before Running It
----------------------------------------

:meth:`~divi.qprog.ensemble.ProgramEnsemble.dry_run` traverses every
sub-program without execution. Call ``create_programs()``, then render the
nested report:

.. skip: next

.. code-block:: python

   from divi.pipeline import format_dry_run

   ensemble.create_programs()
   reports = ensemble.dry_run()
   format_dry_run(reports)

``format_dry_run`` auto-selects a layout from the program count; pass
``style="compact"``, ``"grouped"``, or ``"verbose"`` to force one.

``grouped`` requires matching widths, parameter counts, and objectives;
differences appear as ``mixed (…)`` or ``factor_range``. Recurring and one-time
pipelines remain separate. See :ref:`dry-run` for report semantics.

Cancellation and Batch Status
-----------------------------

Reporting levels and display suppression are described in
:ref:`ensemble-reporting`. During a batched run, an additional status line shows
the merged circuit count, participating programs, polling state, and backend job
identifier.

Pressing :kbd:`Ctrl-C` cancels in-flight backend work and stops the current
round. Completed rounds remain in ``round_history``; a partial round does not
update workflow state or start another round. Cancellation is cooperative, so
work already completing inside a backend may still return before shutdown.

.. _circuit-batching:

Circuit Batching
----------------

By default, :meth:`~divi.qprog.ensemble.ProgramEnsemble.run` merges submissions
through :class:`~divi.qprog.ensemble.BatchConfig`.

**How it works**

Each iteration, the coordinator:

1. Collects circuit submissions from all active programs (barrier-based flush).
2. Merges them into a single payload with namespaced circuit tags.
3. Submits the merged payload to the real backend in one call.
4. Polls for results once (instead of N times).
5. Demultiplexes the results back to each program by tag prefix.

**When to use batching**

- **Cloud backends**: batching reduces API calls, queue slots, and polling.
- **Local simulators**: batching adds synchronisation without network savings.

**Limiting batch size**

By default the coordinator waits for **all** active programs to submit before
merging circuits.  For large ensembles this can produce very large merged jobs.
Use ``max_batch_size`` to cap the number of circuits per flush:

.. skip: next

.. code-block:: python

   from divi.qprog import BatchConfig

   # Flush as soon as 50 circuits are pending (partial flush)
   ensemble.run(batch_config=BatchConfig(max_batch_size=50))

At ``max_batch_size`` the coordinator flushes immediately. The limit controls
merge granularity, not the size of one program's submission.

**Cloud submission with one merged job**

Set ``max_concurrent_programs=-1`` to include the entire ensemble in one cloud
submission, bypassing the default 256-program barrier cap:

.. skip: next

.. code-block:: python

   from divi.qprog import BatchConfig

   # All programs run concurrently -> single merged backend submission.
   ensemble.run(
       batch_config=BatchConfig(max_concurrent_programs=-1),
   )

Use a positive integer to cap concurrency. Combine it with ``max_batch_size``
when each program emits many circuits:

.. skip: next

.. code-block:: python

   ensemble.run(
       batch_config=BatchConfig(
           max_concurrent_programs=20,   # how many programs run at once
           max_batch_size=1024,          # cap circuits per merged call
       ),
   )

Values above 1024 warn because users often intend ``max_batch_size`` instead;
the explicit ``-1`` opt-in is silent.

**Disabling batching**

Pass ``BatchConfig(mode=BatchMode.OFF)`` to disable batching entirely:

.. skip: next

.. code-block:: python

   from divi.qprog import BatchConfig, BatchMode

   # Each program submits circuits independently
   ensemble.run(batch_config=BatchConfig(mode=BatchMode.OFF))

   # Merged submissions (default)
   ensemble.run()

Next Steps
----------

- :doc:`backends` — backend configuration and performance tuning.
- :doc:`resuming_long_runs` — checkpointing the state of supported variational
  programs. Ensemble workflow state is not currently checkpointed as a unit.
- :doc:`visualization` — result visualisation, including :meth:`~divi.qprog.workflows.VQEHyperparameterSweep.visualize_results`.
