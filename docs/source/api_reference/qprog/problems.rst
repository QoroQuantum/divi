Problems
========

QAOA and related solvers accept a :class:`~divi.qprog.problems.QAOAProblem`
instance that encapsulates the optimisation objective, mixer, initial state,
and solution decoding. Divi provides concrete classes for common graph and
binary optimisation problems.

:class:`~divi.qprog.algorithms.VQE` instead takes a
:class:`~divi.qprog.problems.HamiltonianProblem` — a qubit Hamiltonian plus
optional electron counts — or its subclass
:class:`~divi.qprog.problems.MolecularProblem`, built from spatial-orbital
integrals or a molecule via
:meth:`~divi.qprog.problems.MolecularProblem.from_molecule`.
:class:`~divi.qprog.workflows.VQEHyperparameterSweep` sweeps over either
problem type via its ``problems`` argument.
:class:`~divi.qprog.workflows.LASSQD` requires a
:class:`~divi.qprog.problems.MolecularProblem` built with ``from_molecule``
from a PySCF input, since it re-optimises molecular orbitals in the
atomic-orbital basis that only that constructor path retains.

.. automodapi:: divi.qprog.problems
   :headings: ~^
   :no-main-docstr:
   :no-inheritance-diagram:
   :no-inherited-members:
   :include-all-objects:
