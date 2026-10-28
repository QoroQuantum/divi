Localised Active-Space SQD (LASSQD)
====================================

:class:`~divi.qprog.workflows.LASSQD` estimates the ground-state energy, and
optionally low-lying excited states (:ref:`lassqd-excited-states`), of a
molecule whose active space is too large for one circuit. It first partitions
the active space. Each macro-cycle then:

1. prepares each fragment's LUCJ circuit classically from CCSD, without
   optimisation by default
   (:class:`~divi.qprog.workflows.lassqd.LinearMethodPreparation` optionally refines
   it);
2. samples each final circuit once on a quantum backend;
3. recovers fragment states with sample-based quantum diagonalisation (SQD);
4. reassembles the fragment reduced density matrices (RDMs); and
5. re-optimises the molecular orbitals against the active-space RDM.

The cycle repeats until the total energy converges.

LASSQD requires the ``chem`` extra: ``pip install qoro-divi[chem]``. Build its
input with :meth:`~divi.qprog.problems.MolecularProblem.from_molecule`, passed
a PySCF ``gto.Mole`` or restricted mean-field object. LASSQD runs (or reuses)
RHF, or ROHF for an open-shell molecule (``spin`` set on the ``gto.Mole``).
Each macro-cycle re-optimises the orbitals in the atomic-orbital basis, so a
:class:`~divi.qprog.problems.MolecularProblem` built from a PennyLane
``qchem.Molecule`` or from bare integrals is rejected: neither carries one. A
restricted DFT object (``dft.RKS``, ``dft.ROKS``) supplies starting orbitals
only; its functional enters no energy.

**Open-shell molecules** (radicals, ions, high-spin ground states) take a
closed-shell core: the ``2S`` singly occupied orbitals, the highest occupied
ones in ROHF order, must all be active, and the fragments' ``n_alpha - n_beta``
must sum to ``spin``. Automatic fragmentation localises the singly occupied
orbitals apart from the doubly occupied ones and gives each fragment one
unpaired electron per singly occupied orbital it holds, unless ``local_spins``
sets otherwise. ``n_active_orbitals`` takes as many occupied orbitals as
virtual ones, so name a mostly singly occupied active space with
``active_orbitals``. Build the molecule with a non-negative ``spin``;
unrestricted mean fields (UHF, UKS) are rejected. A molecule's states all share
its ``S_z``, so lower-spin states of a high-spin ground state (the singlets of
triplet O2, say) need the molecule built with the lower ``spin``.

.. code-block:: python

   from pyscf import gto
   from divi.backends import MaestroSimulator
   from divi.qprog import LASSQD
   from divi.qprog.workflows.lassqd import FragmentationConfig, FragmentSpec
   from divi.qprog.problems import MolecularProblem

   # NH2, a doublet: orbitals 0-3 doubly occupied, 4 singly occupied in ROHF.
   mol = gto.M(
       atom="N 0 0 0; H 0 0.8 0.6; H 0 -0.8 0.6", basis="sto-3g", spin=1, verbose=0
   )
   ensemble = LASSQD(
       MolecularProblem.from_molecule(mol),
       fragmentation=FragmentationConfig(
           active_spaces=[FragmentSpec(orbitals=(3, 4, 5), n_alpha=2, n_beta=1)],
       ),
       seed=0,
       backend=MaestroSimulator(shots=2000),
   )
   ensemble.run(max_rounds=3)
   print(ensemble.energy)

:class:`~divi.qprog.workflows.LASSQD` subclasses
:class:`~divi.qprog.ensemble.ProgramEnsemble`; multi-round execution, progress
reporting and circuit batching are in
:doc:`../execution_workflows/program_ensembles`.

A Single Fragment: Comparable to FCI
--------------------------------------

With the whole active space in one fragment, the reassembled RDM is the
fragment's own, so the energy is comparable to full configuration interaction
(FCI) on the same active space:

.. dashboard-example: lassqd-ground-state

.. code-block:: python

   from pyscf import gto
   from divi.backends import MaestroSimulator
   from divi.qprog import LASSQD
   from divi.qprog.workflows.lassqd import FragmentationConfig, FragmentSpec, SQDConfig
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
seed a one-repetition spin-unbalanced LUCJ operator from each fragment's CCSD
amplitudes, fitted to the local interaction graph of arXiv:2512.14936, and send
only the final computational-basis sample to ``backend``. Fragment preparations
stay separate ensemble programs and can run in parallel. Non-finite CCSD
amplitudes or linear-method parameters raise ``RuntimeError`` naming the
fragment before a circuit is submitted. The LUCJ preparations fix the paper's
topology, including its final orbital rotation and local interaction pairs.

:class:`~divi.qprog.workflows.lassqd.CCSDPreparation`, the default, samples the
seeded operator without optimisation or a fragment statevector: its classical
cost is CCSD's, with one sampling job per fragment per round.

:class:`~divi.qprog.workflows.lassqd.LinearMethodPreparation` follows
arXiv:2512.14936: ffsim optimises the seeded operator with the linear method
before the sample. It works on the fragment's exact statevector, so it is
limited to fragments that can be simulated classically.

:class:`~divi.qprog.workflows.lassqd.VQEPreparation` runs a Divi ansatz in a
quantum optimisation loop: pass it an optimizer, and optionally another ansatz
(default :class:`~divi.qprog.algorithms.UCCSDAnsatz`;
:class:`~divi.qprog.algorithms.LUCJAnsatz` or another compatible one) and
``max_iterations`` (default 10). ``backend`` runs the expectation-value loop and
``sampling_backend`` can run the separate final SQD sample:

.. skip: next

.. code-block:: python

   from divi.backends import MaestroSimulator
   from divi.qprog import ScipyMethod, ScipyOptimizer, UCCSDAnsatz
   from divi.qprog.workflows.lassqd import VQEPreparation

   ensemble = LASSQD(
       ...,
       preparation=VQEPreparation(
           ScipyOptimizer(ScipyMethod.COBYLA), ansatz=UCCSDAnsatz()
       ),
       backend=MaestroSimulator(shots=500),
       sampling_backend=MaestroSimulator(force_sampling=True, shots=4000),
   )
   ensemble.run(max_rounds=2)

Without ``sampling_backend``, VQE optimisation and final sampling both use
``backend``; with the LUCJ preparations it overrides ``backend`` for the one
final sample. The configured strategy is ``ensemble.preparation``. The
``n_layers`` and ``ansatz_kwargs`` keywords configure only ``VQEPreparation``;
``LASSQD`` rejects them with the other preparations. Set the VQE iteration count
with ``VQEPreparation(max_iterations=...)``.

.. important::

   - Examples here run below :class:`~divi.qprog.workflows.lassqd.SQDConfig`'s
     defaults (``n_batches=15``, ``batch_size=170``,
     ``n_recovery_iterations=6``) for speed.
   - Each batch samples at most ``batch_size`` alpha and beta half-strings, so
     its subspace holds at most ``batch_size**2`` determinants. The ``n_batches``
     subspaces compete rather than pool.
   - ``carryover_cutoff`` is on by default; ``None`` gives conventional SQD,
     whose energies can oscillate (:ref:`lassqd-carryover`).
   - ``extension`` defaults to ``"auto"``, which runs ``"singles"`` (Ext-SQD,
     arXiv:2411.00468): once per round, every significant string gains those one
     electron move away. Fragments over 16384 determinants must set
     ``max_dim``, checked before any circuit runs. ``extension=None`` gives
     plain SQD.
   - ``stop_reason == COMPLETE`` means the energy stopped *changing*, not that
     it is accurate.
   - An energy equal to the mean field means the subspace held only the
     reference determinant; the workflow warns.

Explicit Fragment Specification
--------------------------------

:class:`~divi.qprog.workflows.lassqd.FragmentSpec` names one fragment: which spatial
orbitals belong to it and how many alpha and beta electrons are assigned to
it. Indices refer to the mean field's canonical molecular orbitals (RHF, or ROHF
for an open-shell molecule), not spatial positions. Fragments must be disjoint,
carry valid electron counts, and list occupied orbitals before virtual ones,
since the reference determinant fills orbitals in the order given. Pass known
splits through ``active_spaces``. For the four-orbital H4 tutorial:

.. code-block:: python

   from divi.qprog.workflows.lassqd import FragmentationConfig, FragmentSpec

   fragmentation = FragmentationConfig(
       active_spaces=[
           FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1),
           FragmentSpec(orbitals=(2, 3), n_alpha=1, n_beta=1),
       ],
   )

Automatic Fragmentation
------------------------

:class:`~divi.qprog.workflows.lassqd.FragmentationConfig` controls orbital
selection, partitioning and spin.

**Which orbitals.** Pass exactly one (each also exclusive with
``active_spaces``):

- ``n_active_orbitals`` — selects around the HOMO-LUMO gap.
- ``active_orbitals`` — explicit MO columns.

**Partitioning.** The default localises orbitals and merges them by coupling,
capped by ``max_orbitals_per_fragment`` (default 4); ``coupling_threshold``
(default ``1e-3``) prunes weak edges. ``fragment_atoms`` instead assigns each
localised orbital to an atom-defined fragment, and rejects both knobs.

**Spin.** ``local_spins`` gives signed ``2S`` values per atom-defined fragment,
summing to the molecule's ``spin`` (zero when closed-shell). It requires
``fragment_atoms``, since automatic graph-fragment order is unstable. It cannot
move an electron between fragments, as each fragment's electron count comes from
the localised orbitals it receives; for a broken-symmetry open-shell singlet, or
to choose which fragment carries a charge, use explicit
:class:`~divi.qprog.workflows.lassqd.FragmentSpec` entries.

.. code-block:: python

   from divi.qprog.workflows.lassqd import FragmentationConfig

   fragmentation = FragmentationConfig(
       n_active_orbitals=4,
       max_orbitals_per_fragment=2,
   )

.. _lassqd-rounds-and-results:

Rounds Are Macro-Cycles
-------------------------

One round prepares and samples every fragment, runs SQD recovery, reassembles
the RDM and optimises the orbitals. ``run(max_rounds=N)`` caps macro-cycles;
``None`` runs until consecutive energies differ by less than
``energy_tol`` (default ``1e-6`` Ha) and that round's orbital solve converged,
meaning its orbital-gradient norm is at most ``sqrt(energy_tol)``
(:attr:`~divi.qprog.workflows.lassqd.LASSQDRoundReport.orbital_converged`).
Before stopping, the lowest-state orbital updates check the orbital Hessian and,
at a saddle point, step off it and continue. Key results:

- ``ensemble.energy`` — the converged (or latest) total energy.
- ``ensemble.workflow_state`` — orbitals, fragment states, and current/previous
  energies.
- ``ensemble.round_history`` — program count and circuit/runtime deltas.
- ``ensemble.round_reports`` — energy change, SQD subspace sizes, orbital-solve
  diagnostics, and timings. ``report.summary()`` renders one line.
- ``ensemble.stop_reason`` — ``COMPLETE``, ``MAX_ROUNDS``, ``FAILED``, or
  ``CANCELLED``.

Reports are recorded after reduction, so an interrupted reduction can appear in
``round_history`` without a report. Lifecycle and cancellation are in
:doc:`../execution_workflows/program_ensembles`.

.. _lassqd-orbital-update:

Orbital Update
----------------

Each macro-cycle ends by re-optimising the orbitals against the reassembled
active-space RDM. ``orbital_update`` selects the solver:

- :class:`~divi.qprog.workflows.lassqd.SecondOrderOrbitalUpdate` (default) takes PySCF
  CIAH augmented-Hessian steps, with Hessian-vector products from finite
  differences of the analytic orbital gradient. ``max_iterations`` (default 50)
  caps the steps per macro-cycle.
- :class:`~divi.qprog.workflows.lassqd.LBFGSOrbitalUpdate` minimises with SciPy's
  L-BFGS-B. ``max_iterations`` defaults to ``None``, which leaves SciPy's own
  cap.
- :class:`~divi.qprog.workflows.lassqd.FixedOrbitals` keeps the orbitals as they are,
  so each round's energy is that of the recovered state on fixed orbitals, as in
  CASCI.
- :class:`~divi.qprog.workflows.lassqd.StateAveragedOrbitalUpdate` optimises the
  orbitals and the states together for a weighted average of the states
  ``SQDConfig.states`` requests, as state-averaged CASSCF does
  (:ref:`lassqd-excited-states`).

A capped solve returns its best orbitals and reports the round as not
converged, which blocks completion. Iteration count, gradient norm and
convergence flag per round are in ``ensemble.round_reports``; the solver is
``ensemble.orbital_update``.

.. _lassqd-excited-states:

Excited States
----------------

``SQDConfig(states=...)`` solves for several eigenstates at once, keyed by spin
multiplicity ``2S + 1``. On one fragment all come from one subspace on the same
orbitals. A singlet–triplet gap:

.. dashboard-example: lassqd-excited-states

.. code-block:: python

   from pyscf import gto
   from divi.backends import MaestroSimulator
   from divi.qprog import LASSQD
   from divi.qprog.workflows.lassqd import (
       FragmentationConfig,
       FragmentSpec,
       SQDConfig,
       StateAveragedOrbitalUpdate,
   )
   from divi.qprog.problems import MolecularProblem

   mol = gto.M(atom="H 0 0 0; H 0 0 1.5", basis="sto-3g", verbose=0)

   ensemble = LASSQD(
       MolecularProblem.from_molecule(mol),
       fragmentation=FragmentationConfig(
           active_spaces=[FragmentSpec(orbitals=(0, 1), n_alpha=1, n_beta=1)],
       ),
       sqd=SQDConfig(states={1: 2, 3: 1}),
       orbital_update=StateAveragedOrbitalUpdate(),
       seed=0,
       backend=MaestroSimulator(shots=2000),
   )
   ensemble.run(max_rounds=3)
   states = ensemble.electronic_states
   gap = states[(3, 0)].energy - states[(1, 0)].energy
   print(f"T1 - S0: {gap * 27.2114:.3f} eV")

- **Labels.** ``electronic_states`` maps ``(multiplicity, index)`` to an
  :class:`~divi.qprog.workflows.lassqd.ElectronicState`, ordered by energy.
  ``index`` counts within one spin from 0, in energy order after the first
  round's orbital update: S0 is ``(1, 0)``, S1 ``(1, 1)``, T1 ``(3, 0)``. Later
  rounds keep each label with the same state, matched by CI-vector overlap, so
  when states cross, labels and state-averaging weights follow the states, not
  the energy order. States across two fragments are labelled by energy order
  every round.
- **Orbitals.**
  :class:`~divi.qprog.workflows.lassqd.StateAveragedOrbitalUpdate` relaxes the orbitals
  for a weighted average of the states (default: each root equally, a triplet
  counting once), as SA-CASSCF does.
  :class:`~divi.qprog.workflows.lassqd.FixedOrbitals` keeps the input orbitals.
  The other updates relax for the lowest state only and warn.
- **Symmetry.** Build the molecule with
  ``symmetry=True`` (or pass a mean field with symmetry-adapted orbitals) and
  the state-averaged orbitals keep the abelian point group PySCF works in: the
  molecule's own group when abelian, D2h or C2v for a linear molecule, otherwise
  the largest abelian subgroup (NH3 keeps Cs, not C3v). A solve that breaks
  the abelian group warns.
- **Energies.** ``energy`` is always the lowest requested state's, of whichever
  spin. Under state averaging the macro-cycle converges on the average,
  reported as each round's ``averaged_energy``.
- **Coverage.** The circuit samples the ground state, so excited states rely on
  the default extension, ``"auto"`` (singles).
- **Spin** is targeted by ``lambda_penalty``. A root nearer another spin is left
  out, with a warning, so a subspace short of states of one spin returns fewer
  than were requested. A state can still be partly contaminated, so each
  reports its ``spin_square``, and a warning names any that misses.
- **Fragments.** With one fragment, everything above applies. With two, see
  :ref:`lassqd-states-across-fragments`; more are rejected.

.. _lassqd-states-across-fragments:

States Across Two Fragments
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A product of fragment states cannot describe an excitation that moves an
electron between fragments, or one shared by both. With two fragments,
``SQDConfig(states=...)`` instead mixes products, as LAS state interaction
(LASSI) does. Each fragment's SQD solve targets its lowest state. Each
fragment's spin multiplets are then found at its own electron count and one
electron more and fewer, in a determinant space grown from its samples. Finally
the products of the two fragments' multiplets are mixed by the whole
active-space Hamiltonian on the round's orbitals:

.. code-block:: python

   from pyscf import gto
   from divi.backends import MaestroSimulator
   from divi.qprog import LASSQD
   from divi.qprog.workflows.lassqd import (
       FixedOrbitals,
       FragmentationConfig,
       SQDConfig,
   )
   from divi.qprog.problems import MolecularProblem

   # Two H2 molecules 3 Angstrom apart, one per fragment.
   mol = gto.M(
       atom="H 0 0 0; H 0 0 0.74; H 3 0 0; H 3 0 0.74", basis="sto-3g", verbose=0
   )

   dimer = LASSQD(
       MolecularProblem.from_molecule(mol),
       fragmentation=FragmentationConfig(
           n_active_orbitals=4, fragment_atoms=[(0, 1), (2, 3)]
       ),
       sqd=SQDConfig(states={1: 2, 3: 1}),
       orbital_update=FixedOrbitals(),
       seed=0,
       backend=MaestroSimulator(shots=2000),
   )
   dimer.run(max_rounds=1)
   for label, state in dimer.electronic_states.items():
       print(label, f"{state.energy:.6f}", f"CT {state.charge_transfer:.3f}")

- **Accuracy.** The states are variational within the products kept, so they
  are limited by how far electrons may move between fragments, how many
  multiplets each fragment keeps, and how much of each fragment's space its
  samples reach. The workflow warns about each when it can raise the states.
- **Close contact.** ``SQDConfig.fragment_hops=2`` also moves two electrons.
  With one hop, the workflow suggests two
  when the lowest state has noticeable charge-transfer weight.
- **Fragment multiplets.** ``SQDConfig.fragment_roots`` caps how many of each
  fragment's lowest multiplets per electron count enter the products. The
  default takes all, or fewer when there would be too many products to mix
  densely. Leaving multiplets out can only raise the states, the lowest
  included, and warns.
- **Charge transfer.** A state's ``charge_transfer`` is its weight on products
  with an electron moved between fragments, in either direction. Between bonded
  fragments it also counts covalent sharing.
- **Orbitals.** ``StateAveragedOrbitalUpdate`` is rejected. The other updates
  relax for the lowest state, so excited states miss their orbital relaxation.
- **Results.** ``electronic_states`` holds the energies, ``spin_square`` and
  ``charge_transfer``. ``state_wavefunctions``, ``transition_properties`` and
  ``nevpt2`` need a single fragment and raise otherwise.
- **Cost.** The products are mixed densely and each fragment's multiplets are
  held as full-space CI vectors. A fragment past
  16384 determinants at one of its electron counts is rejected, and a round
  estimated to need more than ``mol.max_memory`` raises ``MemoryError``.

Wave Functions and Transitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each state's wave function, and the transitions between states, come from the
last completed round:

.. code-block:: python

   for label, wavefunction in ensemble.state_wavefunctions().items():
       print(label, wavefunction.leading_determinants(2))

   transition = ensemble.transition_properties()[((1, 0), (1, 1))]
   print(f"S0 -> S1: f = {transition.oscillator_strength:.3f}")

- **States.** :meth:`~divi.qprog.workflows.LASSQD.state_wavefunctions` returns
  a :class:`~divi.qprog.workflows.lassqd.StateWavefunction` per state, in
  ``electronic_states`` order: CI amplitudes over the subspace's alpha and beta
  strings and spin-traced 1- and 2-RDMs in PySCF's convention, all in the
  wave function's ``active_orbitals``. ``core_orbitals`` holds the doubly
  occupied orbitals outside the active space.
- **Transitions.** :meth:`~divi.qprog.workflows.LASSQD.transition_properties`
  returns a :class:`~divi.qprog.workflows.lassqd.StateTransition` for every pair
  of states of one multiplicity, keyed ``(initial, final)`` with the lower state
  first, so a crossing reorders the key. Each holds the transition 1-RDM,
  transition dipole, oscillator strength ``2/3 ΔE |μ|²`` and natural transition
  orbitals. Pairs of different multiplicities have no spin-free transition and
  are left out. The dipole's sign follows the states' arbitrary phases; its
  length does not.

Dynamic Correlation
~~~~~~~~~~~~~~~~~~~

:meth:`~divi.qprog.workflows.LASSQD.nevpt2` adds dynamic correlation to second order, with
PySCF's strongly contracted NEVPT2 on each state at the final orbitals:

.. code-block:: python

   corrections = ensemble.nevpt2()
   corrected = {
       label: state.energy + corrections[label]
       for label, state in ensemble.electronic_states.items()
   }

- **State-specific.** Each state is corrected on its own, as PySCF's
  ``mrpt.NEVPT`` does, so states close in energy that mix under the correction
  are not captured.
- **Order.** The correction can reorder states, a bright state dropping below a
  dark one, while the keys keep the uncorrected labels.
- **Degenerate sets.** Each component of an exactly degenerate set is an
  arbitrary mixture, and the state-specific correction depends on the mixture.
- **Classical active-space sizes.** PySCF needs each state's CI vector over the
  whole active space, which grows as the product of the alpha and beta string
  counts. The method
  raises ``MemoryError`` when one vector exceeds ``mol.max_memory``.
- **Open shells** are corrected the same way, from the ROHF-based states.

Wave functions, transitions and NEVPT2 each re-solve the final subspace at the
final orbitals on every call. The subspace is saved with the checkpoint, so all
three work after :meth:`~divi.qprog.ensemble.ProgramEnsemble.restore_state`.

Solvent
~~~~~~~

``solvent`` places the molecule in a PySCF continuum model (ddCOSMO, ddPCM, PCM
in any variant, or SMD) built on the same molecule, with its dielectric constant
set or, for SMD, its solvent named:

.. code-block:: python

   from pyscf import gto, solvent
   from divi.backends import MaestroSimulator
   from divi.qprog import LASSQD
   from divi.qprog.workflows.lassqd import (
       FixedOrbitals,
       FragmentationConfig,
       FragmentSpec,
       SQDConfig,
   )
   from divi.qprog.problems import MolecularProblem

   mol = gto.M(
       atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587", basis="sto-3g", verbose=0
   )
   water = solvent.ddCOSMO(mol)
   water.eps = 78.36

   ensemble = LASSQD(
       MolecularProblem.from_molecule(mol),
       fragmentation=FragmentationConfig(
           active_spaces=[FragmentSpec(orbitals=(3, 4, 5, 6), n_alpha=2, n_beta=2)],
       ),
       sqd=SQDConfig(states={1: 2}, extension="doubles"),
       orbital_update=FixedOrbitals(),
       solvent=water,
       seed=0,
       backend=MaestroSimulator(shots=2000),
   )
   ensemble.run(max_rounds=10)
   corrections = ensemble.nonequilibrium_corrections()
   vertical = {
       label: state.energy + corrections[label]
       for label, state in ensemble.electronic_states.items()
   }

- **Equilibrium field.** Each round adds the reaction field of the previous
  round's lowest requested state density to the one-electron Hamiltonian, so
  field, orbitals and CI converge together, as self-consistent ddCOSMO- or
  PCM-CASSCF does. The first round has no field, so the first ``energy_history``
  entry is the gas-phase energy. Every energy includes the solvation energy and
  is no variational bound; ``best_energy`` is then the last round's.
- **Which density.** The lowest requested state's, also under
  :class:`~divi.qprog.workflows.lassqd.StateAveragedOrbitalUpdate`, where
  PySCF's state-averaged CASSCF uses the averaged density. It sets the field
  for all states.
- **Vertical excitations.** Excited states sit in the lowest state's
  equilibrium field. :meth:`~divi.qprog.workflows.LASSQD.nonequilibrium_corrections`
  adds the solvent's fast, electronic part answering each state's density
  change at the optical dielectric constant, ``optical_eps`` (default 1.78). The correction
  is first order, without relaxation of the state in that response.
- **What the model covers.** ddCOSMO and PCM are electrostatics only; SMD adds
  its cavitation and dispersion term. :meth:`~divi.qprog.workflows.LASSQD.nevpt2`
  corrects the states in the converged field without feeding its correlated
  density back into it.
- **Comparing with PySCF.** PySCF's ddCOSMO- or PCM-CASSCF updates the field
  from a density that lags its converged one, so its energy can differ.
- **Checkpoints** keep the field, so a resumed run continues from it; one
  written with other solvent settings is refused. LASSQD works on a copy of the
  solvent model and leaves the one passed in untouched.

.. _lassqd-accuracy-characteristics:

Accuracy Characteristics
--------------------------

**The energy is a variational upper bound.** Reassembling the per-fragment RDMs
reproduces the RDMs of a product of fragment states, including the
cross-fragment Coulomb and exchange blocks, so ``energy`` is a genuine
expectation value and cannot fall below the exact CASSCF minimum for the same
active space, or FCI in the full basis. CASCI on fixed orbitals is not a bound,
since LASSQD re-optimises the orbitals. In a ``solvent`` the energy adds a
solvation energy and is no bound.

**What fragmenting costs is inter-fragment correlation.** A product of fragment
states cannot describe correlation *between* fragments.

.. _lassqd-carryover:

Carrying Configurations Between Iterations and Rounds
-------------------------------------------------------

Without retention, each recovery iteration diagonalises only the configurations
it just sampled, so a determinant found early is lost when sampling moves on.
``carryover_cutoff`` keeps the determinants of the winning batch whose
coefficient exceeds that fraction of the largest, and extends later iterations'
subspaces with them (arXiv:2512.14936). It defaults to ``1e-5``; ``None`` gives
conventional SQD and turns retention off within and across rounds:

.. code-block:: python

   from divi.qprog.workflows.lassqd import SQDConfig

   sqd = SQDConfig(
       n_batches=2,
       batch_size=4,
       n_recovery_iterations=4,
       carryover_cutoff=1e-2,
       max_carryover=64,
   )

Alpha and beta halves are retained separately and the subspace is rebuilt as
their product, which reintroduces determinant *combinations* never sampled
together. Strings are ranked by marginal weight over the whole subspace, not
only over determinants that cleared the cutoff.

``max_carryover`` caps retained strings per spin sector, and ``max_dim`` caps
each sector outright; see :class:`~divi.qprog.workflows.lassqd.SQDConfig` for
the resulting subspace bounds:

.. code-block:: python

   # At most 12 alpha x 12 beta = 144 determinants per batch, whatever the budget.
   config = SQDConfig(carryover_cutoff=1e-2, max_dim=12)

When a cap binds, strings are kept in priority order: reference, then carried by
descending weight, then this batch's halves by descending sample count.

.. warning::

   With both caps unset, the retained set and the subspace grow quadratically
   every iteration, as the relative cutoff prunes little. Past 16384
   determinants a failed eigenvalue solve has no dense fallback.

Retention also crosses macro-cycles: the strings retained from a fragment's best
result seed the first recovery iteration of its next round, then the cutoff
decides again. Every round re-optimises the orbitals, so carried strings are
first mapped into the new basis through the atomic-orbital overlap of the old
and new orbitals. A determinant in one basis is a superposition in another, so
the mapping keeps the nearest one: each new orbital takes the occupation of one
old orbital. ``carryover_mapping`` chooses the pairing:

- ``'assignment'`` (default) pairs old and new orbitals one-to-one by maximum
  total overlap, so every carried string keeps its electron count.
- ``'argmax'`` gives each new orbital the old orbital it overlaps most. Two new
  orbitals can then share an old one; strings whose electron count that changes
  are dropped.

.. code-block:: python

   sqd = SQDConfig(carryover_cutoff=1e-3, carryover_mapping="argmax")

.. _lassqd-subspace-floor:

Guaranteeing a Floor, and Stopping Early
------------------------------------------

Three further :class:`~divi.qprog.workflows.lassqd.SQDConfig` knobs shape the
subspace rather than its size.

``include_reference`` (on by default) keeps the fragment's aufbau reference
determinant in every batch, so a fragment's SQD energy cannot land above
its own reference. Turn it off to make the subspace exactly what was sampled.

``symmetrize_spin`` pools the alpha and beta halves, so a sampled ``|1001>``
also offers ``|0110>`` and both singlet and triplet combinations can form.
Ignored on spin-polarised fragments, where exchanging the sectors is not a
symmetry.

``recovery_energy_tol`` and ``recovery_occupancies_tol`` end a fragment's
recovery once both its energy and orbital occupancies stop moving. Both default
to ``0.0``, so recovery spends every iteration unless you opt in. Neither is
``LASSQD``'s own ``energy_tol``, which ends the macro-cycle.

Next Steps
------------

- The full tutorial in
  `tutorials/chemistry/lassqd_h4.py <https://github.com/QoroQuantum/divi/blob/main/tutorials/chemistry/lassqd_h4.py>`_
  runs the two-fragment H4 example to convergence and compares against CASCI.
- :doc:`ground_state_energy_estimation_vqe` for single-fragment-sized active
  spaces that don't need fragmentation at all.
