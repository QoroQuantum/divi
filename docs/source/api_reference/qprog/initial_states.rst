Initial States
==============

An initial state prepares the register before the first ansatz or evolution
layer. Algorithms such as QAOA, VQE and
:class:`~divi.qprog.algorithms.TimeEvolution` take one through their
``initial_state`` argument, and QAOA problems recommend one through
``recommended_initial_state``. Subclass
:class:`~divi.qprog.initial_states.InitialState` to define your own.

.. automodapi:: divi.qprog.initial_states
   :headings: ~^
   :no-main-docstr:
   :no-inheritance-diagram:
   :no-inherited-members:
   :include-all-objects:
