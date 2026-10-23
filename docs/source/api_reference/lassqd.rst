LASSQD
======

:class:`~divi.qprog.workflows.LASSQD` partitions a molecule's active
space into fragments and prepares each fragment for SQD, configured by a
:class:`~divi.qprog.workflows.lassqd.Preparation`
(:class:`~divi.qprog.workflows.lassqd.CCSDPreparation`,
:class:`~divi.qprog.workflows.lassqd.LinearMethodPreparation` or
:class:`~divi.qprog.workflows.lassqd.VQEPreparation`), an
:class:`~divi.qprog.workflows.lassqd.OrbitalUpdate`
(:class:`~divi.qprog.workflows.lassqd.SecondOrderOrbitalUpdate`,
:class:`~divi.qprog.workflows.lassqd.StateAveragedOrbitalUpdate`,
:class:`~divi.qprog.workflows.lassqd.LBFGSOrbitalUpdate` or
:class:`~divi.qprog.workflows.lassqd.FixedOrbitals`),
:class:`~divi.qprog.workflows.lassqd.FragmentationConfig`, and
:class:`~divi.qprog.workflows.lassqd.SQDConfig`. Requested excited states come
back as :class:`~divi.qprog.workflows.lassqd.ElectronicState` records, with
their wave functions as :class:`~divi.qprog.workflows.lassqd.StateWavefunction`
and the transitions between them as
:class:`~divi.qprog.workflows.lassqd.StateTransition`.

The configuration and result types live in ``divi.qprog.workflows.lassqd``,
which requires the ``chem`` extra; :class:`~divi.qprog.workflows.LASSQD`
itself is imported from ``divi.qprog.workflows`` and documented with the other
workflows.

.. automodapi:: divi.qprog.workflows.lassqd
   :headings: ~^
   :no-main-docstr:
   :no-inheritance-diagram:
   :no-inherited-members:
   :include-all-objects:
   :skip: LASSQD
