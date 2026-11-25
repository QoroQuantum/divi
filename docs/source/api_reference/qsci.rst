QSCI
====

:class:`~divi.qprog.ensembles.qsci.QSCI` samples bitstrings from quantum
programs, keeps the most probable as a seed subspace, grows it by each
determinant's Epstein-Nesbet second-order contribution, and diagonalises the
Hamiltonian projected onto it. Its result is a
:class:`~divi.qprog.ensembles.qsci.QSCIState`, with one
:class:`~divi.qprog.ensembles.qsci.ExpansionRecord` per expansion step, and
:func:`~divi.qprog.ensembles.qsci.extrapolate_pt2_to_zero` estimates the exact
energy from those records.

:class:`~divi.qprog.ensembles.qsci.QSCISolver` runs QSCI as a PySCF CASCI or
CASSCF solver, seeded by a VQE on the active space, and needs the ``chem``
extra.

``QSCI`` and its types live in ``divi.qprog.ensembles.qsci``.

.. automodapi:: divi.qprog.ensembles.qsci
   :headings: ~^
   :no-main-docstr:
   :no-inheritance-diagram:
   :no-inherited-members:
   :include-all-objects:
