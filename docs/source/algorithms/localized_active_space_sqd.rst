Localised Active-Space SQD (LASSQD)
====================================

:class:`~divi.qprog.workflows.LASSQD` estimates the ground-state energy of a
molecule whose active space is too large for one VQE. It first partitions the
active space. Each macro-cycle then:

1. runs one embedded VQE per fragment;
2. recovers fragment states with sample-based quantum diagonalisation (SQD);
3. reassembles the fragment reduced density matrices (RDMs); and
4. re-optimises the molecular orbitals against the active-space RDM.

The cycle repeats until the total energy converges.

**When to reach for it**: an active space that fits comfortably in one VQE (a
handful of orbitals) is better served by :class:`~divi.qprog.algorithms.VQE`
directly — see :doc:`ground_state_energy_estimation_vqe`. LASSQD trades exact
treatment of inter-fragment correlation for the ability to scale the active
space past what a single circuit's qubit count and optimizer can support,
by splitting it into VQE-sized pieces. Read
:ref:`lassqd-accuracy-characteristics` below before treating its output as
a chemistry-grade energy.

LASSQD requires the ``chem`` extra: ``pip install qoro-divi[chem]``. It
accepts a PySCF ``gto.Mole`` or restricted (closed-shell) mean-field object
— not a PennyLane ``qchem.Molecule`` — runs (or reuses) the RHF calculation,
and only supports closed-shell molecules.

Because :class:`~divi.qprog.workflows.LASSQD` subclasses
:class:`~divi.qprog.ensemble.ProgramEnsemble`, its multi-round execution
model, progress reporting, and circuit-batching behaviour are the ones
described in :doc:`../execution_workflows/program_ensembles`; this page covers only what is
specific to LASSQD.

A Single Fragment: Comparable to FCI
--------------------------------------

The simplest configuration puts the whole active space in one fragment. The
reassembled RDM is then the fragment's own, with no product structure to it, so
the energy is directly comparable to a full configuration interaction (FCI)
calculation on the same active space:

.. code-block:: python

   from pyscf import gto
   from divi.backends import MaestroSimulator
   from divi.qprog import LASSQD, FragmentationConfig, FragmentSpec, SQDConfig
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer

   mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)

   ensemble = LASSQD(
       mol,
       fragmentation=FragmentationConfig(
           active_spaces=[FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)],
       ),
       sqd=SQDConfig(n_batches=4, batch_size=128, n_recovery_iterations=1),
       optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
       max_iterations=5,
       seed=0,
       backend=MaestroSimulator(shots=500),
   )
   ensemble.run(max_rounds=2)
   print(f"Energy: {ensemble.energy:.6f} Ha")

.. important::

   Examples here run below :class:`~divi.qprog.workflows.SQDConfig`'s defaults
   (``n_batches=15``, ``batch_size=170``, ``n_recovery_iterations=6``) for
   speed.

   Each batch samples at most ``batch_size`` alpha and beta half-strings; their
   Cartesian-product subspace therefore contains at most ``batch_size**2``
   determinants. Sampling is without replacement. ``n_batches`` subspaces
   compete rather than pool, so it buys attempts, not size.
   ``carryover_cutoff`` keeps what was seen across iterations and is on by
   default; setting it to ``None`` gives conventional SQD, whose energies
   oscillate rather than converge (:ref:`lassqd-carryover`).

   ``stop_reason == COMPLETE`` means the energy stopped *changing*, not that it
   is accurate. An energy equal to the mean field means the subspace held only
   the reference determinant; the workflow warns when that happens.

For this H2 example, the configured subspace can recover the FCI result when it
samples the relevant determinants. At ``batch_size=32`` it instead returns the
mean-field energy because the correlated determinant carries about 1% of the
distribution.

Explicit Fragment Specification
--------------------------------

:class:`~divi.qprog.workflows.FragmentSpec` names one fragment: which spatial
orbitals belong to it and how many alpha and beta electrons are assigned to
it. Indices refer to canonical RHF molecular orbitals, not spatial positions.
Fragments must be disjoint and closed shell, with valid electron counts. Pass
known splits through ``active_spaces``. For the four-orbital H4 tutorial:

.. code-block:: python

   from divi.qprog import FragmentationConfig, FragmentSpec

   fragmentation = FragmentationConfig(
       active_spaces=[
           FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1),
           FragmentSpec(orbitals=(2, 3), n_alpha=1, n_beta=1),
       ],
   )

.. note::

   This occupied/virtual split cuts through both H2 units and is intentionally
   poor. The H4 tutorial compares it with a weak-coupling split.

Automatic Fragmentation
------------------------

Automatic fragmentation controls orbital selection, partitioning, and spin
through :class:`~divi.qprog.workflows.FragmentationConfig`.

**Which orbitals.** Pass exactly one of these (each also mutually exclusive with
``active_spaces``):

- ``n_active_orbitals`` — selects around the HOMO-LUMO gap.
- ``active_orbitals`` — explicit MO columns, useful when character matters more
  than energy; pair with an appropriate mean field such as PySCF AVAS.

**Partitioning.** The default localises orbitals and merges them by coupling,
capped by ``max_orbitals_per_fragment`` (default 4); ``coupling_threshold``
(default ``1e-3``) prunes weak edges. ``fragment_atoms`` instead assigns each
localised orbital to an atom-defined fragment.

**Spin.** ``local_spins`` supplies signed ``2S`` values per atom-defined
fragment while preserving electron counts. These values must sum to zero,
corresponding to total ``S_z=0``. It requires ``fragment_atoms`` because
automatic graph-fragment order is unstable.

.. code-block:: python

   from divi.qprog import FragmentationConfig

   fragmentation = FragmentationConfig(
       n_active_orbitals=4,
       max_orbitals_per_fragment=2,
   )

.. _lassqd-rounds-and-results:

Rounds Are Macro-Cycles
-------------------------

One LASSQD round runs every fragment VQE and SQD recovery, reassembles the RDM,
and optimises the orbitals. ``run(max_rounds=N)`` caps macro-cycles; ``None``
runs until consecutive energies differ by less than ``energy_tol`` (default
``1e-6`` Ha). Key results are:

- ``ensemble.energy`` — the converged (or latest) total energy.
- ``ensemble.workflow_state`` — orbitals, fragment states, and current/previous
  energies.
- ``ensemble.round_history`` — program count and circuit/runtime deltas.
- ``ensemble.round_reports`` — energy change, SQD subspace sizes, orbital-solve
  diagnostics, and timings. ``report.summary()`` renders one line.
- ``ensemble.stop_reason`` — ``COMPLETE``, ``MAX_ROUNDS``, ``FAILED``, or
  ``CANCELLED``.

Reports are recorded after reduction. An interrupted reduction can therefore
appear in ``round_history`` without a corresponding report. General lifecycle
and cancellation semantics are in :doc:`../execution_workflows/program_ensembles`.

.. _lassqd-accuracy-characteristics:

Accuracy Characteristics
--------------------------

**The energy is a variational upper bound.** Reassembling the per-fragment
RDMs reproduces the reduced density matrices of a product of fragment states,
including the cross-fragment Coulomb and exchange blocks, so the reported
``energy`` is a genuine expectation value and cannot fall below a CASCI or FCI
calculation on the same active space.

**What fragmenting costs is inter-fragment correlation.** A product of fragment
states cannot describe correlation *between* fragments, and the error grows with
how strongly they interact. Measured against a CASCI reference on the same
active space: essentially exact for well-separated H\ :sub:`2` pairs, tens of
mHa on coupled hydrogen chains, and — on an N\ :sub:`2` triple bond in a
six-orbital active space — around 50 mHa near equilibrium, rising to roughly
285 mHa at a bond length of 3.0 Å once the bond is fully broken. Pick fragments
along weak interactions, not through bonds.

Two consequences worth knowing before you trust a number:

* **Error smoothness matters more than its size** for relative energies. On
  H\ :sub:`4` separation and symmetric-stretch curves the error is smooth and
  monotone; on N\ :sub:`2` it is neither. Across 1.1–3.0 Å it moves by 12–28 mHa
  between adjacent geometries near equilibrium and by about 175 mHa between 2.5
  and 3.0 Å, and it does not decrease monotonically. Automatic fragmentation is
  part of the reason: the layout it picks changes along the curve, so adjacent
  points are not always solving the same partitioning. Such a curve is unusable
  for reaction energies even though each point is a valid bound.
* **More fragments is not automatically worse.** A single fragment spanning a
  wide active space asks more of the sampling than several narrow ones, and can
  come out less accurate despite being the more expressive ansatz. Compare
  layouts on your own system rather than assuming.

.. _lassqd-carryover:

Carrying Configurations Between Recovery Iterations
-----------------------------------------------------

Without retention, each recovery iteration diagonalises only the configurations
it just sampled, so a determinant found early is lost as soon as sampling moves
on. ``carryover_cutoff`` keeps the ones carrying real weight — the determinants
of the winning batch whose coefficient exceeds that fraction of the largest —
and extends later iterations' subspaces with them (arXiv:2512.14936). It
defaults to ``1e-5``; pass ``None`` for conventional SQD:

.. code-block:: python

   from divi.qprog import SQDConfig

   sqd = SQDConfig(
       n_batches=2,
       batch_size=4,
       n_recovery_iterations=4,
       carryover_cutoff=1e-2,
       max_carryover=64,
   )

Because the halves are retained separately and the subspace is rebuilt as their
product, this reintroduces determinant *combinations* that were never sampled
together. Which strings survive is ranked by marginal weight over the whole
subspace, not only over the determinants that cleared the cutoff.

Two knobs bound the growth. ``max_carryover`` caps how many alpha and beta
strings retention holds **per spin sector**; carried strings join each batch's
own sampled halves rather than replacing them, so a cap of ``k`` bounds a
batch's subspace at ``(k + batch_size) ** 2`` determinants. ``max_dim`` caps each
sector outright, so the subspace never exceeds the product of the two limits:

.. code-block:: python

   # At most 12 alpha x 12 beta = 144 determinants per batch, whatever the budget.
   config = SQDConfig(carryover_cutoff=1e-2, max_dim=12)

When a cap binds, strings are kept in priority order: reference, then carried by
descending weight, then this batch's halves by descending sample count.

.. warning::

   Leaving both caps unset lets the retained set grow every iteration, and the
   subspace with it, quadratically — the relative cutoff prunes little on its
   own. The projected matrices are dense, so a fragment with a large determinant
   space can exhaust memory. Set ``max_dim`` there.

This helps where sampling reaches a small fraction of a fragment's determinant
space. Where sampling already covers that space — small fragments, or a generous
``batch_size`` — there is nothing left to add and it changes nothing. Check
:attr:`~divi.qprog.workflows.LASSQDRoundReport.subspace_sizes` against the
fragment's full determinant count to see which regime you are in.

Retention is scoped to one fragment solve. A determinant is a statement about a
particular orbital basis, and every round re-optimises the orbitals, so carrying
bitstrings across rounds would require mapping them into the new basis first.

.. _lassqd-subspace-floor:

Guaranteeing a Floor, and Stopping Early
------------------------------------------

Three further knobs on :class:`~divi.qprog.workflows.SQDConfig` shape the
subspace rather than its size.

``include_reference`` (on by default) keeps the fragment's aufbau reference
determinant in every batch. Adding a determinant to a variational subspace can
only lower the projected minimum, so a fragment's SQD energy cannot land above
its own reference however the sampling went. Turn it off to make the subspace
exactly what was sampled.

``symmetrize_spin`` pools the alpha and beta halves together, so a sampled
``|1001>`` also offers ``|0110>`` and both singlet and triplet combinations can
be formed. Ignored on spin-polarised fragments, where exchanging the sectors is
not a symmetry.

``recovery_energy_tol`` and ``recovery_occupancies_tol`` end a fragment's
recovery once both its energy and its orbital occupancies stop moving. Both
default to ``0.0``, so recovery spends every iteration unless you opt in. Neither
is ``LASSQD``'s own ``energy_tol``, which ends the macro-cycle.

.. warning::

   Carryover improves the subspace non-monotonically, so a settled iteration does
   not mean the next one had nothing to add. Stopping early cost 2.6 mHa on the
   diiron complex of arXiv:2512.14936 and 11.7 mHa on a four-orbital H4 fragment.
   Enable it only once you have confirmed that recovery, rather than the orbital
   solve, is what your runs spend their time on.

Choosing an Ansatz
--------------------

Use the default :class:`~divi.qprog.algorithms.UCCSDAnsatz` unless you have
validated :class:`~divi.qprog.algorithms.LUCJAnsatz`—the local unitary
cluster-Jastrow alternative—at your fragment size. Both use fragment CCSD data.
UCCSD reads amplitudes directly; LUCJ derives rotation and Coulomb parameters by
double factorisation and generally uses more parameters.

SQD needs coverage, not merely a low ansatz energy. An ansatz concentrated on
one determinant starves recovery. Compare per-fragment subspace sizes in
:attr:`~divi.qprog.workflows.LASSQDRoundReport.subspace_sizes` against the
full determinant count.

For the truncated LUCJ circuit used in arXiv:2405.05068 and arXiv:2512.14936,
set ``ansatz_kwargs={"trailing_rotation": True}`` with ``n_layers=1``. To reduce
parameters, use ``shared_spin_params``, limit ``rotation_depth``, or specify
``same_spin_pairs`` / ``opposite_spin_pairs``. ``shared_spin_params`` cannot use
the factorised seed and therefore starts from the optimizer initialisation.

Next Steps
------------

- The full tutorial in
  `tutorials/chemistry/lassqd_h4.py <https://github.com/QoroQuantum/divi/blob/main/tutorials/chemistry/lassqd_h4.py>`_
  runs the two-fragment H4 example to convergence and compares against CASCI.
- :doc:`../execution_workflows/program_ensembles` for the shared multi-round execution model,
  progress reporting, and circuit batching.
- :doc:`ground_state_energy_estimation_vqe` for single-fragment-sized active
  spaces that don't need fragmentation at all.
