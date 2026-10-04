Localised Active-Space SQD (LASSQD)
====================================

:class:`~divi.qprog.workflows.LASSQD` estimates the ground-state energy of a
molecule whose active space is too large for one circuit. It first partitions
the active space. Each macro-cycle then:

1. prepares each fragment's LUCJ circuit classically from CCSD, without
   optimisation by default
   (:class:`~divi.qprog.workflows.LinearMethodPreparation` optionally refines
   it);
2. samples each final circuit once on a quantum backend;
3. recovers fragment states with sample-based quantum diagonalisation (SQD);
4. reassembles the fragment reduced density matrices (RDMs); and
5. re-optimises the molecular orbitals against the active-space RDM.

The cycle repeats until the total energy converges.

**When to reach for it**: an active space that fits comfortably in one VQE (a
handful of orbitals) is better served by :class:`~divi.qprog.algorithms.VQE`
directly — see :doc:`ground_state_energy_estimation_vqe`. LASSQD trades exact
treatment of inter-fragment correlation for the ability to scale the active
space past what a single circuit's qubit count can support by splitting it into
smaller pieces. Read
:ref:`lassqd-accuracy-characteristics` below before treating its output as
a chemistry-grade energy.

LASSQD requires the ``chem`` extra: ``pip install qoro-divi[chem]``. Build its
input with :meth:`~divi.qprog.problems.MolecularProblem.from_molecule`, passed
a PySCF ``gto.Mole`` or restricted (closed-shell) mean-field object; LASSQD
runs (or reuses) the RHF calculation from that input. Because each macro-cycle
re-optimises the molecular orbitals in the atomic-orbital basis, LASSQD rejects
a :class:`~divi.qprog.problems.MolecularProblem` built from a PennyLane
``qchem.Molecule`` or from bare integrals — both carry no atomic-orbital basis
to re-optimise into. Only closed-shell (RHF) molecules are supported.

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
   from divi.qprog.problems import MolecularProblem

   mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)

   ensemble = LASSQD(
       MolecularProblem.from_molecule(mol),
       fragmentation=FragmentationConfig(
           active_spaces=[FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)],
       ),
       sqd=SQDConfig(n_batches=4, batch_size=128, n_recovery_iterations=1),
       seed=0,
       backend=MaestroSimulator(shots=500),
   )
   print(type(ensemble.preparation).__name__)  # CCSDPreparation
   ensemble.run(max_rounds=2)
   print(f"Energy: {ensemble.energy:.6f} Ha")

LUCJ, CCSD and VQE Preparation
--------------------------------

The two LUCJ preparations, ``CCSDPreparation`` and ``LinearMethodPreparation``,
seed a one-repetition spin-unbalanced LUCJ operator from
each fragment's CCSD amplitudes, fitted subject to the local interaction graph
of arXiv:2512.14936, and send only the final computational-basis sample to
``backend``. Independent fragment preparations remain separate ensemble
programs and can run in parallel.

:class:`~divi.qprog.workflows.CCSDPreparation`, the default, samples the
CCSD-seeded operator as it is, with no optimisation and without building the
fragment statevector: its classical cost is CCSD's and it submits one sampling
job per fragment per round. Without the optimisation, the sampled state is CCSD's quality,
and carryover across rounds (:ref:`lassqd-carryover`) does more of the work.

:class:`~divi.qprog.workflows.LinearMethodPreparation` follows
arXiv:2512.14936: ffsim optimises the CCSD-seeded operator classically with the
linear method before the sample. The linear method works on the fragment's
exact statevector, whose size grows combinatorially with the fragment, so it is
limited to fragments that can be simulated classically.

For a spin-polarised fragment, ROHF and ffsim use the paper's single one-body
preparation Hamiltonian. SQD diagonalisation receives both physical-spin
one-body tensors, rotated into the sampled orbital basis, so the signed
embedding field remains present during recovery. Non-finite CCSD amplitudes or
linear-method parameters raise ``RuntimeError`` with the affected fragment
before a circuit is submitted.

VQE preparation is available as an explicit alternative. Pass
:class:`~divi.qprog.workflows.VQEPreparation` with an optimizer, and optionally
another Divi ansatz and ``max_iterations`` (default 10). ``backend`` then runs
the VQE expectation-value loop and ``sampling_backend`` can run the separate
final SQD sample:

.. skip: next

.. code-block:: python

   from divi.backends import MaestroSimulator, QiskitSimulator
   from divi.qprog import ScipyMethod, ScipyOptimizer, UCCSDAnsatz, VQEPreparation

   ensemble = LASSQD(
       ...,
       preparation=VQEPreparation(
           ScipyOptimizer(ScipyMethod.COBYLA), ansatz=UCCSDAnsatz()
       ),
       backend=MaestroSimulator(shots=500),
       sampling_backend=QiskitSimulator(force_sampling=True, shots=4000),
   )
   ensemble.run(max_rounds=2)

If ``sampling_backend`` is omitted, VQE optimisation and final sampling both
use ``backend``. With the LUCJ preparations, ``sampling_backend`` simply
overrides ``backend`` for the one final sample.

:class:`~divi.qprog.workflows.VQEPreparation` uses
:class:`~divi.qprog.algorithms.UCCSDAnsatz` when ``ansatz`` is omitted. The
configured strategy is available as ``ensemble.preparation``.

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

For this H2 example, the configured subspace recovers the FCI result when it
samples the relevant determinants. A smaller ``batch_size`` can miss a
correlated determinant that carries little of the distribution and return the
mean-field energy instead.

Explicit Fragment Specification
--------------------------------

:class:`~divi.qprog.workflows.FragmentSpec` names one fragment: which spatial
orbitals belong to it and how many alpha and beta electrons are assigned to
it. Indices refer to canonical RHF molecular orbitals, not spatial positions.
Fragments must be disjoint, carry valid electron counts, and list their
occupied orbitals before their virtual ones, since each fragment's reference
determinant fills its orbitals in the order given. Pass known splits through
``active_spaces``. For the four-orbital H4 tutorial:

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

One LASSQD round prepares and samples every fragment, runs SQD recovery,
reassembles the RDM, and optimises the orbitals. ``run(max_rounds=N)`` caps
macro-cycles; ``None`` runs until consecutive energies differ by less than
``energy_tol`` (default ``1e-6`` Ha) and that round's orbital solve converged,
meaning its orbital-gradient norm is at most ``sqrt(energy_tol)``
(:attr:`~divi.qprog.workflows.LASSQDRoundReport.orbital_converged`). Key
results are:

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

.. _lassqd-orbital-update:

Orbital Update
----------------

Each macro-cycle ends by re-optimising the molecular orbitals against the
reassembled active-space RDM. ``orbital_update`` selects the solver:

- :class:`~divi.qprog.workflows.SecondOrderOrbitalSolve` (default) takes PySCF
  CIAH augmented-Hessian steps, with Hessian-vector products from finite
  differences of the analytic orbital gradient. ``max_iterations`` (default 50)
  caps the steps per macro-cycle.
- :class:`~divi.qprog.workflows.FullOrbitalSolve` minimises with SciPy's
  L-BFGS-B. ``max_iterations`` defaults to ``None``, which leaves SciPy's own
  cap.

A capped solve returns its best orbitals and reports the round as not
converged, which blocks completion. Each round's iteration count, gradient
norm and convergence flag are in ``ensemble.round_reports``; the configured
solver is ``ensemble.orbital_update``.

.. _lassqd-accuracy-characteristics:

Accuracy Characteristics
--------------------------

**The energy is a variational upper bound.** Reassembling the per-fragment
RDMs reproduces the reduced density matrices of a product of fragment states,
including the cross-fragment Coulomb and exchange blocks, so the reported
``energy`` is a genuine expectation value and cannot fall below CASSCF with the
same active-space size, or FCI in the full basis. CASCI on fixed orbitals is
not a bound, since LASSQD re-optimises the orbitals.

**What fragmenting costs is inter-fragment correlation.** A product of fragment
states cannot describe correlation *between* fragments, and the error grows with
how strongly they interact: small for well-separated fragments, larger for
coupled ones, and largest when a fragment boundary cuts through a bond —
splitting a triple bond is the worst case, and the error grows as the bond is
stretched. Pick fragments along weak interactions, not through bonds.

Two consequences worth knowing before you trust a number:

* **Error smoothness matters more than its size** for relative energies. Along
  a curve that separates weakly interacting fragments the error varies
  smoothly; along one that breaks a bond between fragments it can jump between
  adjacent geometries and need not change monotonically. Automatic
  fragmentation contributes: the layout it picks can change along the curve, so
  adjacent points are not always solving the same partitioning. Such a curve is
  unusable for reaction energies even though each point is a valid bound.
* **More fragments is not automatically worse.** A single fragment spanning a
  wide active space asks more of the sampling than several narrow ones, and can
  come out less accurate despite being the more expressive ansatz. Compare
  layouts on your own system rather than assuming.

.. _lassqd-carryover:

Carrying Configurations Between Iterations and Rounds
-------------------------------------------------------

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

Retention also crosses macro-cycles. The strings retained from a fragment's best
result seed the first recovery iteration of its next round; from there the
cutoff decides again what stays. A determinant is a statement about a particular
orbital basis, and every round re-optimises the orbitals and re-prepares the
fragment in a new basis, so the carried strings are first mapped into it through
the atomic-orbital overlap of the old and new orbitals. A determinant in one
basis is a superposition in another, so the mapping keeps the nearest one: each
new orbital takes the occupation of one old orbital. ``carryover_mapping``
chooses how they are paired:

- ``'assignment'`` (default) pairs old and new orbitals one-to-one by maximum
  total overlap, so every carried string keeps its electron count.
- ``'argmax'`` gives each new orbital the old orbital it overlaps most, as the
  reference implementation does. Two new orbitals can then share an old one;
  strings whose electron count that changes are dropped.

.. code-block:: python

   sqd = SQDConfig(carryover_cutoff=1e-3, carryover_mapping="argmax")

Setting ``carryover_cutoff=None`` turns retention off within and across rounds.

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
   not mean the next one had nothing to add, and stopping early can cost
   accuracy. Enable it only once you have confirmed that recovery, rather than the orbital
   solve, is what your runs spend their time on.

Choosing a Preparation
------------------------

Choose :class:`~divi.qprog.workflows.LinearMethodPreparation` while the
fragments can be simulated classically, and
:class:`~divi.qprog.workflows.CCSDPreparation` once they cannot. Of the three,
only :class:`~divi.qprog.workflows.VQEPreparation` improves on the CCSD seed at
that size, at the cost of a quantum optimisation loop. Optimising it on a
:class:`~divi.backends.MaestroSimulator` configured for matrix-product-state
simulation and passing the device as ``sampling_backend`` keeps the device to
one job per fragment per round.

Use :class:`~divi.qprog.workflows.VQEPreparation` to run a Divi ansatz in a
quantum optimisation loop. It defaults to
:class:`~divi.qprog.algorithms.UCCSDAnsatz` and accepts
:class:`~divi.qprog.algorithms.LUCJAnsatz` or another compatible ansatz.

SQD needs coverage, not merely a low ansatz energy. An ansatz concentrated on
one determinant starves recovery. Compare per-fragment subspace sizes in
:attr:`~divi.qprog.workflows.LASSQDRoundReport.subspace_sizes` against the
full determinant count.

The LUCJ preparations fix the paper's one-repetition topology, including its
final orbital rotation and local interaction pairs. The ``n_layers`` and
``ansatz_kwargs`` keywords configure only
:class:`~divi.qprog.workflows.VQEPreparation`, and ``LASSQD`` rejects them with
the other preparations at construction. Set the VQE iteration count with
``VQEPreparation(max_iterations=...)``.

Next Steps
------------

- The full tutorial in
  `tutorials/chemistry/lassqd_h4.py <https://github.com/QoroQuantum/divi/blob/main/tutorials/chemistry/lassqd_h4.py>`_
  runs the two-fragment H4 example to convergence and compares against CASCI.
- :doc:`../execution_workflows/program_ensembles` for the shared multi-round execution model,
  progress reporting, and circuit batching.
- :doc:`ground_state_energy_estimation_vqe` for single-fragment-sized active
  spaces that don't need fragmentation at all.
