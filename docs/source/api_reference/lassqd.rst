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

``LASSQD`` and its types live in ``divi.qprog.ensembles.lassqd``, which requires
the ``chem`` extra.

.. automodapi:: divi.qprog.ensembles.lassqd
   :headings: ~^
   :no-main-docstr:
   :no-inheritance-diagram:
   :no-inherited-members:
   :include-all-objects:
