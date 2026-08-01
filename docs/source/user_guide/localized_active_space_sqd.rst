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
       n_batches=2,
       batch_size=4,
       n_sqd_iterations=1,
       seed=0,
       backend=MaestroSimulator(shots=500),
   )
   ensemble.run(max_rounds=2)
   print(f"Energy: {ensemble.energy:.6f} Ha")

.. important::

   Every code example on this page uses a deliberately small sampling
   budget (``n_batches``, ``batch_size``, ``n_sqd_iterations`` well below
   the constructor's defaults of 15, 170, and 6) so it runs quickly.
   ``lambda_penalty`` (default ``0.2``) is a further SQD sizing knob: the
   weight of the S² spin-contamination penalty applied before
   diagonalizing each fragment's projected Hamiltonian. A
   ``stop_reason`` of ``COMPLETE`` only means the energy stopped changing
   between rounds — it does **not** mean the energy is accurate. Run
   verbatim, the snippet above lands about 20 mHa above FCI — H2/STO-3G's
   entire correlation energy — meaning SQD's recovered subspace held only
   the reference (RHF) determinant and the reported energy is exactly the
   mean-field energy; that is directly recognizable, since the two values
   match. Reaching chemical accuracy takes a substantially larger budget;
   the measured result below (agreeing with FCI to ``2e-16`` Ha) uses
   ``n_batches=12, batch_size=32, n_sqd_iterations=3, max_iterations=60``
   at ``seed=7``.

On H2/STO-3G with that larger budget and seed, a converged single-fragment
run agrees with FCI to about ``2e-16`` Ha. Whether the correlated
determinants get captured depends on the seed as well as the budget: across
seeds ``{0, 1, 2, 3, 42}`` at that same budget, only one seed reproduced
this result, and the other four converged onto the mean-field energy with
``stop_reason == COMPLETE``. Raising ``max_iterations`` well beyond 60
changes nothing once a seed has landed on the mean-field plateau. Reaching
FCI is therefore possible but not guaranteed by budget alone. Fragmenting
the same active space into more than one piece moves the workflow into the
regime described in :ref:`lassqd-accuracy-characteristics`.

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
       max_iterations=5,
       n_batches=2,
       batch_size=4,
       n_sqd_iterations=1,
       seed=7,
       backend=MaestroSimulator(shots=200),
   )
   ensemble.run(max_rounds=2)
   # With two fragments this energy is not variational and is not
   # comparable to FCI/CASCI -- see Accuracy Characteristics below.
   print(f"Energy: {ensemble.energy:.6f} Ha")

.. warning::

   With more than one fragment, as above, the reported energy is **not
   variational** and is not comparable to FCI/CASCI on the same active
   space — see :ref:`lassqd-accuracy-characteristics` below.

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
       n_batches=2,
       batch_size=4,
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
       n_batches=2,
       batch_size=4,
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

**Fragmenting the active space is not variational.** Reassembling
per-fragment RDMs zeroes every cross-fragment 2-RDM block, so with more than
one fragment the reported ``energy`` is not comparable to a CASCI or FCI
calculation on the same active space and can fall substantially below it —
this is not a convergence failure. Measured on a linear H4 chain with two
2-orbital fragments, the converged energy sits about 1.47 Ha below CASCI. A
fixed-orbital-basis estimate of the zeroed-block error alone accounts for
only about 0.37 Ha of that gap; the remaining ~1.1 Ha accumulates because
each macro-cycle re-optimizes the molecular orbitals against a
non-N-representable RDM, and the self-consistency loop compounds that error
round over round until it converges by ``energy_tol`` onto the lower value.
The electron count and symmetry stay correct throughout — only the absolute
energy is affected. Do not use a multi-fragment LASSQD energy as a
drop-in replacement for a CASCI or FCI reference.

**A single fragment spanning the whole active space is the exception**: with
no cross-fragment blocks to zero, the functional is variational and the
result is directly comparable to FCI (see
`A Single Fragment: the Variational Case`_ above). This makes a
single-fragment run a useful sanity check when validating a new fragment
layout or a change to fragment-level settings.

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
qubits it is more expensive (6 vs. 3). On a minimal two-orbital fragment it
plateaus about 20 mHa above the exact ground state — again H2/STO-3G's
entire correlation energy — because this ansatz cannot represent the
correlated state at all, so its sampled distribution never covers anything
beyond the reference determinant; adding layers does not lift the plateau,
since more layers of an ansatz that can't reach the correlated state don't
help. This is the same "subspace collapsed to one determinant" outcome as
the small-sampling-budget case above, reached for a different reason: an
inexpressive ansatz rather than too few samples. Prefer ``UCCSDAnsatz``
(the default) unless you have already validated ``LUCJAnsatz`` against a
reference on your specific fragment size.

Next Steps
------------

- The full tutorial in
  `tutorials/chemistry/lassqd_h4.py <https://github.com/QoroQuantum/divi/blob/main/tutorials/chemistry/lassqd_h4.py>`_
  runs the two-fragment H4 example to convergence and compares against CASCI.
- :doc:`program_ensembles` for the shared multi-round execution model,
  progress reporting, and circuit batching.
- :doc:`ground_state_energy_estimation_vqe` for single-fragment-sized active
  spaces that don't need fragmentation at all.
