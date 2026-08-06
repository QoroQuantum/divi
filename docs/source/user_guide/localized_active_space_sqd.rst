Localized Active-Space SQD (LASSQD)
====================================

:class:`~divi.qprog.workflows.LASSQD` estimates the ground-state energy of a
molecule whose active space is too large to optimize as a single VQE. It
partitions the active space into fragments, runs one VQE per fragment against
its own mean-field-embedded effective Hamiltonian, recovers each fragment's
ground state from the VQE's sampled bitstring distribution via sample-based
quantum diagonalization (SQD), reassembles the fragments' reduced density
matrices (RDMs) into an active-space RDM, and re-optimizes the molecular
orbitals against it. This cycle — VQE, SQD recovery, RDM reassembly, orbital
optimization — repeats until the total energy converges.

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
model, progress reporting, and circuit-batching behavior are the ones
described in :doc:`program_ensembles`; this page covers only what is
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

   ``batch_size`` is how much of the determinant space one iteration sees; the
   subspace holds at most its square, and each batch draws *without*
   replacement so those are distinct configurations. ``n_batches`` subspaces
   compete rather than pool, so it buys attempts, not size.
   ``carryover_cutoff`` keeps what was seen across iterations and is on by
   default; setting it to ``None`` gives conventional SQD, whose energies
   oscillate rather than converge (:ref:`lassqd-carryover`).

   ``stop_reason == COMPLETE`` means the energy stopped *changing*, not that it
   is accurate. An energy equal to the mean field means the subspace held only
   the reference determinant; the workflow warns when that happens.

One fragment spanning the whole space is just SQD on that space, so the snippet
above reaches FCI. At ``batch_size=32`` it returns the mean-field energy
instead: the correlated determinant carries about 1% of the distribution here,
so a few dozen samples usually miss it.

Explicit Fragment Specification
--------------------------------

:class:`~divi.qprog.workflows.FragmentSpec` names one fragment: which spatial
orbitals belong to it and how many alpha and beta electrons are assigned to
it. Orbital indices are canonical RHF molecular-orbital indices, in
energy order — not spatial positions; localization only happens in the
automatic branch below. Fragments must be disjoint, closed-shell
(``n_alpha == n_beta``), and their electron counts must fall within
``[0, len(orbitals)]``. Pass a list of specs as
``active_spaces`` when you already know how the active space should split
— for example, splitting a linear H4 chain's canonical orbitals into its
two occupied MOs and its two virtual MOs:

.. code-block:: python

   from pyscf import gto
   from divi.backends import MaestroSimulator
   from divi.qprog import LASSQD, FragmentationConfig, FragmentSpec, SQDConfig
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer

   h4 = gto.M(
       atom="H 0 0 0; H 0 0 0.74; H 0 0 2.0; H 0 0 2.74",
       basis="sto-3g",
       verbose=0,
   )

   ensemble = LASSQD(
       h4,
       fragmentation=FragmentationConfig(
           active_spaces=[
               FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1),
               FragmentSpec(orbitals=(2, 3), n_alpha=1, n_beta=1),
           ],
       ),
       sqd=SQDConfig(n_batches=4, batch_size=128, n_recovery_iterations=1),
       optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
       max_iterations=20,
       seed=7,
       backend=MaestroSimulator(shots=2000),
   )
   ensemble.run(max_rounds=2)
   print(f"Energy: {ensemble.energy:.6f} Ha")

.. note::

   Occupied-vs-virtual is a poor split here — every H2 unit straddles both
   fragments. ``tutorials/chemistry/lassqd_h4.py`` compares it against cutting
   along the weakest coupling.

   The energy is a variational upper bound, so it sits *above* FCI/CASCI on the
   same active space. How far above depends on how strongly the fragments
   interact — see :ref:`lassqd-accuracy-characteristics` below.

Automatic Fragmentation
------------------------

Rather than specifying fragments by hand, LASSQD can select the active space
and split it into fragments for you. The automatic path has three independent
choices, all fields of
:class:`~divi.qprog.workflows.FragmentationConfig`: which orbitals are active,
how they partition into fragments, and what spin each fragment carries.

**Which orbitals.** Pass exactly one of these (each also mutually exclusive with
``active_spaces``):

- ``n_active_orbitals`` — total spatial orbitals to select around the
  HOMO-LUMO gap (``ceil(k / 2)`` highest occupied, ``floor(k / 2)`` lowest
  virtual). The right choice when the correlated orbitals sit at the frontier.
- ``active_orbitals`` — explicit MO column indices. Use this when the active
  space is defined by orbital *character* rather than energy: a transition
  metal's ``d`` manifold can sit well below the HOMO with its virtual partners
  well above the LUMO, where no frontier count reaches them. Pair it with a
  mean field whose orbitals already carry that character, such as one prepared
  with PySCF's AVAS.

**How they partition.** By default the selected space is localized and split by
greedily merging orbitals along their coupling strength, capped by
``max_orbitals_per_fragment`` (default 4), with ``coupling_threshold`` (default
``1e-3``) pruning weak edges. Alternatively pass ``fragment_atoms`` — one
sequence of atom indices per fragment — to assign each localized orbital to
whichever fragment owns the atom it sits on. That is how a localized active
space is usually specified: one fragment per metal centre.

**What spin.** Fragments default to closed shell. ``local_spins`` sets ``2S``
per fragment, in the order ``fragment_atoms`` names them, leaving each
fragment's electron count alone — so ``local_spins=[2, -2]`` makes two local
triplets aligned antiparallel, the antiferromagnetic arrangement of a coupled
dimer. The fragments must still sum to ``Sz = 0``. It requires
``fragment_atoms``, because coupling-graph fragment order depends on the
localization RNG and would not name a stable fragment.

.. code-block:: python

   from pyscf import gto
   from divi.backends import MaestroSimulator
   from divi.qprog import LASSQD, FragmentationConfig, SQDConfig
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer

   h4 = gto.M(
       atom="H 0 0 0; H 0 0 0.74; H 0 0 2.0; H 0 0 2.74",
       basis="sto-3g",
       verbose=0,
   )

   ensemble = LASSQD(
       h4,
       fragmentation=FragmentationConfig(
           n_active_orbitals=4, max_orbitals_per_fragment=2
       ),
       sqd=SQDConfig(n_batches=4, batch_size=128, n_recovery_iterations=1),
       optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
       max_iterations=2,
       seed=0,
       backend=MaestroSimulator(shots=200),
   )
   state = ensemble.initial_state()
   for fragment in state.fragments:
       print(fragment.spec.orbitals, fragment.spec.n_alpha, fragment.spec.n_beta)

.. _lassqd-rounds-and-results:

Rounds Are Macro-Cycles
-------------------------

LASSQD is a :class:`~divi.qprog.ensemble.ProgramEnsemble` workflow, so
:meth:`~divi.qprog.ensemble.ProgramEnsemble.run` drives it through the
``initial_state`` / ``is_complete`` / ``create_programs`` / ``update_state``
loop described in :doc:`program_ensembles`. For LASSQD, **one round is one
full macro-cycle**: every fragment's VQE runs once, SQD recovers each
fragment's ground state from the sampled distribution, the fragment RDMs are
reassembled into an active-space RDM, and the molecular orbitals are
re-optimized against it. ``run(max_rounds=N)`` caps the number of
macro-cycles; leave it as ``None`` to run until the energy converges.

A macro-cycle is considered complete once two consecutive rounds' total
energies differ by less than ``energy_tol`` (default ``1e-6`` Ha). After
``run()`` returns:

- ``ensemble.energy`` — the converged (or latest) total energy.
- ``ensemble.workflow_state`` — the latest
  :class:`~divi.qprog.workflows.LASSQDState`: ``mo_coeff``, per-fragment
  :class:`~divi.qprog.workflows.FragmentState` (spec, RDMs, converged VQE
  parameters), ``energy``, and ``previous_energy``.
- ``ensemble.round_history`` — one :class:`~divi.qprog.RoundRecord` per
  macro-cycle, each carrying its round number, program count, and
  circuit/runtime deltas.
- ``ensemble.round_reports`` — one
  :class:`~divi.qprog.workflows.LASSQDRoundReport` per macro-cycle: the energy
  and its change, each fragment's SQD subspace size, the orbital solve's
  iteration count, gradient norm and convergence flag, and per-stage wall clock.
  Recorded once a round's reduction finishes, so an interrupted run keeps every
  completed round — and a round that failed or was cancelled mid-reduction
  appears in ``round_history`` without a report. ``report.summary()`` renders one
  line per round.
- ``ensemble.stop_reason`` — a :class:`~divi.qprog.WorkflowStatus`.
  ``COMPLETE`` means the energy converged within ``energy_tol``;
  ``MAX_ROUNDS`` means ``run()`` stopped at the ``max_rounds`` cap before
  converging. ``FAILED`` and ``CANCELLED`` carry the same meaning as for any
  other ensemble workflow (see :doc:`program_ensembles`).

.. code-block:: python

   from pyscf import gto
   from divi.backends import MaestroSimulator
   from divi.qprog import (
       LASSQD,
       FragmentationConfig,
       FragmentSpec,
       SQDConfig,
       WorkflowStatus,
   )
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer

   h2 = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)

   ensemble = LASSQD(
       h2,
       fragmentation=FragmentationConfig(
           active_spaces=[FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)],
       ),
       sqd=SQDConfig(n_batches=4, batch_size=128, n_recovery_iterations=1),
       optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
       max_iterations=5,
       seed=0,
       backend=MaestroSimulator(shots=500),
   )
   ensemble.run(max_rounds=1)

   assert ensemble.stop_reason == WorkflowStatus.MAX_ROUNDS
   print(f"Stopped after {len(ensemble.round_history)} round(s) at "
         f"{ensemble.energy:.6f} Ha, reason={ensemble.stop_reason}")

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

Without retention, each recovery iteration diagonalizes only the configurations
it just sampled, so a determinant found early is lost as soon as sampling moves
on. ``carryover_cutoff`` keeps the ones carrying real weight — the determinants
of the winning batch whose coefficient exceeds that fraction of the largest —
and extends later iterations' subspaces with them (arXiv:2512.14936). It
defaults to ``1e-5``; pass ``None`` for conventional SQD:

.. code-block:: python

   from pyscf import gto
   from divi.backends import MaestroSimulator
   from divi.qprog import LASSQD, FragmentationConfig, FragmentSpec, SQDConfig
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer

   h4 = gto.M(atom="H 0 0 0; H 0 0 1.0; H 0 0 2.0; H 0 0 3.0",
              basis="sto-3g", verbose=0)

   # One four-orbital fragment spans 36 determinants, and a small batch_size
   # reaches only a few per iteration -- the regime where retention pays.
   ensemble = LASSQD(
       h4,
       fragmentation=FragmentationConfig(
           active_spaces=[FragmentSpec(orbitals=(0, 1, 2, 3), n_alpha=2, n_beta=2)],
       ),
       sqd=SQDConfig(
           n_batches=2,
           batch_size=4,
           n_recovery_iterations=4,
           carryover_cutoff=1e-2,
           max_carryover=64,
       ),
       optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
       max_iterations=3,
       seed=0,
       backend=MaestroSimulator(shots=500),
   )
   ensemble.run(max_rounds=1)

   print(f"subspaces {ensemble.round_reports[0].subspace_sizes} of 36")

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
particular orbital basis, and every round re-optimizes the orbitals, so carrying
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
be formed. Ignored on spin-polarized fragments, where exchanging the sectors is
not a symmetry.

``recovery_energy_tol`` and ``recovery_occupancies_tol`` end a fragment's
recovery once both its energy and its orbital occupancies stop moving. Both
default to ``0.0``, so recovery spends every iteration unless you opt in. Neither
is ``LASSQD``'s own ``energy_tol``, which ends the macro-cycle.

.. warning::

   Carryover improves the subspace non-monotonically, so a settled iteration does
   not mean the next one had nothing to add. Stopping early cost 2.6 mHa on the
   diiron complex of arXiv:2512.14936 and 11.7 mHa on a four-orbital H4 fragment,
   against a recovery loop that took 0.1 s per round there to the orbital solve's
   ~300 s. Enable it only when recovery dominates.

Choosing an Ansatz
--------------------

LASSQD defaults each fragment's VQE to
:class:`~divi.qprog.algorithms.UCCSDAnsatz`. Its first layer is seeded from
that fragment's own CCSD amplitudes rather than a random initial guess,
which improves convergence and gives SQD a better-covered sampled
distribution to recover the ground state from.
:class:`~divi.qprog.algorithms.LUCJAnsatz` is also available (pass
``ansatz=LUCJAnsatz()``). Each layer applies ``exp(K) exp(iJ) exp(-K)``, where
``K`` is a general orbital rotation (independent per spin sector) and ``J`` is a
diagonal Coulomb operator restricted to same-orbital opposite-spin pairs plus
same-spin neighbors — that restriction on ``J`` is what makes the ansatz
*local*. Its rotations are unitary rather than merely orthogonal, which
``exp(iJ)`` needs: under a real rotation the first-order energy correction is
imaginary and cancels, and the ansatz recovers far less correlation energy — 43%
against 94% on a four-orbital H4 fragment at one layer. That costs a phase per
Givens rotation, so LUCJ is more expensive than
``UCCSDAnsatz`` at every fragment size — 53 against 24 on a five-orbital
fragment. Prefer the default unless you have validated ``LUCJAnsatz`` against a
reference at your own fragment size.

Seeding works for both. ``UCCSDAnsatz``'s parameters *are* amplitudes, so each is
read straight off ``t1``/``t2``; ``LUCJAnsatz``'s are rotation and Coulomb
angles, so its seed comes from the leading term of the doubles tensor's double
factorization, which supplies both the rotation and the Coulomb weights.

Because SQD recovers correlation by diagonalizing in the *sampled* subspace, a
fragment ansatz that concentrates its amplitude on one determinant starves the
solver even when its own energy is low. Check the per-fragment subspace sizes in
:attr:`~divi.qprog.workflows.LASSQDRoundReport.subspace_sizes` against the
fragment's full determinant count before trusting an energy.

To match the circuit arXiv:2405.05068 and arXiv:2512.14936 run — the truncated
LUCJ form ``exp(K2) exp(-K1) exp(iJ1) exp(K1)`` on the Hartree-Fock determinant
— pass ``ansatz_kwargs={"trailing_rotation": True}`` with ``n_layers=1``. On a
five-orbital fragment that costs 103 parameters against 53 without it, where a
second full layer would cost 106.

Three more ``ansatz_kwargs`` trade expressiveness for parameters, should the
default prove too large to optimize at your fragment size. ``shared_spin_params``
drives both spin sectors from one set of rotation and same-spin Coulomb
parameters, imposing spin symmetry — 29 parameters on that five-orbital
fragment. ``rotation_depth`` truncates each rotation's brick-wall network to
that many half-layers instead of the ``n_orbitals`` a general rotation needs;
``rotation_depth=2`` also costs 29. ``same_spin_pairs`` and
``opposite_spin_pairs`` replace the Jastrow's local pattern with explicit
orbital pairs, from ``[]`` to every pair. Seeding follows all of these except
``shared_spin_params``, where the factorization has nothing to say — it gives each
sector its own rotation — so that flavor warns and starts from the optimizer's
own initialization.

Next Steps
------------

- The full tutorial in
  `tutorials/chemistry/lassqd_h4.py <https://github.com/QoroQuantum/divi/blob/main/tutorials/chemistry/lassqd_h4.py>`_
  runs the two-fragment H4 example to convergence and compares against CASCI.
- :doc:`program_ensembles` for the shared multi-round execution model,
  progress reporting, and circuit batching.
- :doc:`ground_state_energy_estimation_vqe` for single-fragment-sized active
  spaces that don't need fragmentation at all.
