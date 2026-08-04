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

A Single Fragment: the Variational Case
----------------------------------------

The simplest configuration puts the whole active space in one fragment. With
only one fragment there are no cross-fragment RDM blocks to approximate, so
the energy is variational and directly comparable to a full configuration
interaction (FCI) calculation on the same active space:

.. code-block:: python

   from pyscf import gto
   from divi.backends import MaestroSimulator
   from divi.qprog import LASSQD, FragmentSpec
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer

   mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)

   ensemble = LASSQD(
       mol,
       active_spaces=[FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)],
       optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
       max_iterations=5,
       n_batches=4,
       batch_size=128,
       n_sqd_iterations=1,
       seed=0,
       backend=MaestroSimulator(shots=500),
   )
   ensemble.run(max_rounds=2)
   print(f"Energy: {ensemble.energy:.6f} Ha")

.. important::

   Examples here run below the defaults (``n_batches=15``,
   ``batch_size=170``, ``n_sqd_iterations=6``) for speed. ``batch_size`` is the
   accuracy knob: configurations sampled per batch, and the subspace holds at
   most its square.

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
   from divi.qprog import LASSQD, FragmentSpec
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer

   h4 = gto.M(
       atom="H 0 0 0; H 0 0 0.74; H 0 0 2.0; H 0 0 2.74",
       basis="sto-3g",
       verbose=0,
   )

   ensemble = LASSQD(
       h4,
       active_spaces=[
           FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1),
           FragmentSpec(orbitals=(2, 3), n_alpha=1, n_beta=1),
       ],
       optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
       max_iterations=20,
       n_batches=4,
       batch_size=128,
       n_sqd_iterations=1,
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
and split it into fragments for you. Pass exactly one of ``n_active_orbitals``
or ``energy_window`` (mutually exclusive with each other and with
``active_spaces``):

- ``n_active_orbitals`` — total spatial orbitals to select around the
  HOMO-LUMO gap (``ceil(k / 2)`` highest occupied, ``floor(k / 2)`` lowest
  virtual). Use this when you want a fixed-size active space regardless of
  the molecule's orbital spacing.
- ``energy_window`` — an energy window, in Hartree, around the HOMO-LUMO gap:
  an occupied orbital qualifies when its energy is at least the HOMO energy
  minus the window, and a virtual orbital when its energy is at most the
  LUMO energy plus the window. Use this when the relevant orbitals are known
  to cluster within a specific energy range and you want the active-space
  size to adapt to the molecule rather than fixing it in advance.

Either way, ``max_orbitals_per_fragment`` (default 4) caps how many orbitals
each automatically built fragment can contain — the selected active space is
partitioned by localizing the occupied and virtual blocks independently and
greedily merging orbitals along their coupling strength, so a smaller cap
produces more, smaller fragments. ``coupling_threshold`` (default ``1e-3``)
is the relative edge-pruning threshold for that coupling graph.

.. code-block:: python

   from pyscf import gto
   from divi.backends import MaestroSimulator
   from divi.qprog import LASSQD
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer

   h4 = gto.M(
       atom="H 0 0 0; H 0 0 0.74; H 0 0 2.0; H 0 0 2.74",
       basis="sto-3g",
       verbose=0,
   )

   ensemble = LASSQD(
       h4,
       n_active_orbitals=4,
       max_orbitals_per_fragment=2,
       optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
       max_iterations=2,
       n_batches=4,
       batch_size=128,
       n_sqd_iterations=1,
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
- ``ensemble.stop_reason`` — a :class:`~divi.qprog.WorkflowStatus`.
  ``COMPLETE`` means the energy converged within ``energy_tol``;
  ``MAX_ROUNDS`` means ``run()`` stopped at the ``max_rounds`` cap before
  converging. ``FAILED`` and ``CANCELLED`` carry the same meaning as for any
  other ensemble workflow (see :doc:`program_ensembles`).

.. code-block:: python

   from pyscf import gto
   from divi.backends import MaestroSimulator
   from divi.qprog import LASSQD, FragmentSpec, WorkflowStatus
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer

   h2 = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)

   ensemble = LASSQD(
       h2,
       active_spaces=[FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)],
       optimizer=ScipyOptimizer(ScipyMethod.COBYLA),
       max_iterations=5,
       n_batches=4,
       batch_size=128,
       n_sqd_iterations=1,
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

Choosing an Ansatz
--------------------

LASSQD defaults each fragment's VQE to
:class:`~divi.qprog.algorithms.UCCSDAnsatz`. Its first layer is seeded from
that fragment's own CCSD amplitudes rather than a random initial guess,
which improves convergence and gives SQD a better-covered sampled
distribution to recover the ground state from.
:class:`~divi.qprog.algorithms.LUCJAnsatz` is also available (pass
``ansatz=LUCJAnsatz()``); it only uses fewer parameters per layer than
``UCCSDAnsatz`` on larger fragments (e.g. 16 vs. 26 at 8 qubits) — at 4
qubits it is more expensive (6 vs. 3).

Its accuracy depends strongly on fragment size, and on a **two-orbital
fragment it recovers nothing at all**: exact optimization from many random
starts returns
the Hartree-Fock determinant itself, about 20 mHa above the exact ground state
on H\ :sub:`4`'s fragments. Its sampled distribution then covers only the
reference determinant, which is the same "subspace collapsed to one
determinant" outcome as the small-sampling-budget case above, reached for a
different reason. Adding layers does not help there — 1, 2 and 3 layers agree
to 8 significant figures — because a two-orbital register has too little
structure for the ansatz to act on, not because the ansatz is inexpressive in
general: on a four-orbital fragment the same measurement recovers 47%, 66% and
88% of the correlation energy at 1, 2 and 3 layers. Raise ``n_layers`` before
concluding it cannot reach your fragment's correlated state, and prefer
``UCCSDAnsatz`` (the default) unless you have validated ``LUCJAnsatz`` against
a reference at your own fragment size.

To match the circuit arXiv:2405.05068 and arXiv:2512.14936 run — the truncated
LUCJ form ``exp(K2) exp(-K1) exp(iJ1) exp(K1)`` on the Hartree-Fock determinant
— pass ``ansatz_kwargs={"trailing_rotation": True}`` with ``n_layers=1``. On a
five-orbital fragment that costs 29 parameters against 21 without it, where a
second full layer would cost 42.

Next Steps
------------

- The full tutorial in
  `tutorials/chemistry/lassqd_h4.py <https://github.com/QoroQuantum/divi/blob/main/tutorials/chemistry/lassqd_h4.py>`_
  runs the two-fragment H4 example to convergence and compares against CASCI.
- :doc:`program_ensembles` for the shared multi-round execution model,
  progress reporting, and circuit batching.
- :doc:`ground_state_energy_estimation_vqe` for single-fragment-sized active
  spaces that don't need fragmentation at all.
