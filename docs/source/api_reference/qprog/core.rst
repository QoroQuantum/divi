Core Architecture
=================

:class:`~divi.qprog.QuantumProgram` is the abstract base shared by every quantum
algorithm in Divi. :class:`~divi.qprog.VariationalQuantumAlgorithm` extends it
with the variational-parameter machinery used by VQE, QAOA, PCE, and
:class:`~divi.qprog.algorithms.CustomVQA`. :class:`~divi.qprog.mixins.SolutionEntry`
is the uniform return type for decoded solutions.
:class:`~divi.qprog.EarlyStopping` terminates a variational optimisation once a
criterion is met: a patience-based plateau, a gradient-norm threshold, or
settled cost variance. :class:`~divi.qprog.StopReason` reports which one fired.

.. automodapi:: divi.qprog
   :headings: ~^
   :no-main-docstr:
   :no-inheritance-diagram:
   :no-inherited-members:
   :include-all-objects:
   :skip: ParamHistoryMode

Type Aliases
~~~~~~~~~~~~

.. autodata:: divi.qprog.variational_quantum_algorithm.ParamHistoryMode
   :no-value:

   Accepted values for
   :meth:`~divi.qprog.VariationalQuantumAlgorithm.param_history`:
   ``"all_evaluated"`` returns every parameter vector seen during optimisation;
   ``"best_per_iteration"`` returns only the best vector from each iteration.
