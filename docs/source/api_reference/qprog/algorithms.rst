Algorithms
==========

Divi provides implementations of popular quantum algorithms with a focus on
scalability and ease of use. VQE targets ground-state energy estimation; QAOA
and PCE target combinatorial optimization; :class:`~divi.qprog.algorithms.TimeEvolution`
simulates Hamiltonian dynamics; :class:`~divi.qprog.algorithms.CustomVQA` lets you wrap
an arbitrary parameterized circuit as a variational program.

.. automodapi:: divi.qprog.algorithms
   :headings: ~^
   :no-main-docstr:
   :no-inheritance-diagram:
   :no-inherited-members:
   :include-all-objects:

Trotterization Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~

QAOA uses a trotterization strategy to evolve the cost Hamiltonian. The default
is :class:`~divi.hamiltonians.ExactTrotterization`; :class:`~divi.hamiltonians.QDrift`
provides randomized sampling for shallower circuits at the cost of more circuits
per iteration. See the :doc:`/api_reference/hamiltonians` reference page for full documentation.

.. autosummary::
   :nosignatures:

   divi.hamiltonians.TrotterizationStrategy
   divi.hamiltonians.ExactTrotterization
   divi.hamiltonians.QDrift
