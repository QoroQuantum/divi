Algorithms
==========

Divi provides implementations of popular quantum algorithms. VQE targets ground-state energy estimation; QAOA
and PCE target combinatorial optimisation; :class:`~divi.qprog.algorithms.TimeEvolution`
simulates Hamiltonian dynamics; :class:`~divi.qprog.algorithms.CustomVQA` lets you wrap
an arbitrary parameterised circuit as a variational program.

.. automodapi:: divi.qprog.algorithms
   :headings: ~^
   :no-main-docstr:
   :no-inheritance-diagram:
   :no-inherited-members:
   :include-all-objects:

Trotterization Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~

QAOA uses a trotterization strategy to evolve the cost Hamiltonian. The default
is :class:`~divi.hamiltonians.trotterization.ExactTrotterization`; :class:`~divi.hamiltonians.trotterization.QDrift`
provides randomised term sampling. See the :doc:`/api_reference/hamiltonians` reference page for full documentation.

.. autosummary::
   :nosignatures:

   divi.hamiltonians.trotterization.TrotterizationStrategy
   divi.hamiltonians.trotterization.ExactTrotterization
   divi.hamiltonians.trotterization.QDrift
