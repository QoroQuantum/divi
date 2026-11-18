LASSQD
======

:class:`~divi.qprog.ensembles.lassqd.LASSQD` partitions a molecule's active space
into fragments and prepares each for SQD. It takes a
:class:`~divi.qprog.ensembles.lassqd.Preparation`, an
:class:`~divi.qprog.ensembles.lassqd.OrbitalUpdate`, a
:class:`~divi.qprog.ensembles.lassqd.FragmentationConfig` and an
:class:`~divi.qprog.ensembles.lassqd.SQDConfig`. Requested excited states come
back as :class:`~divi.qprog.ensembles.lassqd.ElectronicState` records, with
:class:`~divi.qprog.ensembles.lassqd.StateWavefunction` and
:class:`~divi.qprog.ensembles.lassqd.StateTransition` for a single fragment.

Its building blocks also work on one active space, without fragmenting it.
:func:`~divi.qprog.ensembles.lassqd.prepare_lucj` builds a CCSD-seeded LUCJ
circuit for a :class:`~divi.qprog.problems.MolecularProblem`, and
:class:`~divi.qprog.ensembles.lassqd.SQDSolver` runs configuration recovery and
the projected diagonalisation on its samples, after
:func:`~divi.qprog.ensembles.lassqd.probs_to_sqd_bitstrings` puts them in SQD
order.

``LASSQD`` and its types live in ``divi.qprog.ensembles.lassqd``, which requires
the ``chem`` extra.

.. automodapi:: divi.qprog.ensembles.lassqd
   :headings: ~^
   :no-main-docstr:
   :no-inheritance-diagram:
   :no-inherited-members:
   :include-all-objects:
