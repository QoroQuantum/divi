LASSQD
======

:class:`~divi.qprog.workflows.LASSQD` partitions a molecule's active space
into fragments and prepares each for SQD. It takes a
:class:`~divi.qprog.workflows.lassqd.Preparation`, an
:class:`~divi.qprog.workflows.lassqd.OrbitalUpdate`, a
:class:`~divi.qprog.workflows.lassqd.FragmentationConfig` and an
:class:`~divi.qprog.workflows.lassqd.SQDConfig`. Requested excited states come
back as :class:`~divi.qprog.workflows.lassqd.ElectronicState` records, with
:class:`~divi.qprog.workflows.lassqd.StateWavefunction` and
:class:`~divi.qprog.workflows.lassqd.StateTransition` for a single fragment.

The types live in ``divi.qprog.workflows.lassqd``, which requires the ``chem``
extra; :class:`~divi.qprog.workflows.LASSQD` is imported from
``divi.qprog.workflows`` and documented with the other workflows.

.. automodapi:: divi.qprog.workflows.lassqd
   :headings: ~^
   :no-main-docstr:
   :no-inheritance-diagram:
   :no-inherited-members:
   :include-all-objects:
   :skip: LASSQD
