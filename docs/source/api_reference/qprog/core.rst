Core Architecture
=================

:class:`~divi.qprog.QuantumProgram` is the abstract base shared by every quantum
algorithm in Divi. :class:`~divi.qprog.VariationalQuantumAlgorithm` extends it
with the variational-parameter machinery used by VQE, QAOA, PCE, and
:class:`~divi.qprog.algorithms.CustomVQA`. :class:`~divi.qprog.SolutionEntry` is the
uniform return type for decoded solutions.

.. automodapi:: divi.qprog
   :headings: ~^
   :no-main-docstr:
   :no-inheritance-diagram:
   :no-inherited-members:
   :include-all-objects:
   :skip: VQE, QAOA, IterativeQAOA, InterpolationStrategy, PCE, TimeEvolution, CustomVQA
   :skip: Ansatz, HartreeFockAnsatz, UCCSDAnsatz, QCCAnsatz, QAOAAnsatz, HardwareEfficientAnsatz, GenericLayerAnsatz
   :skip: InitialState, ZerosState, OnesState, SuperpositionState, CustomPerQubitState, WState
   :skip: ScipyOptimizer, ScipyMethod, MonteCarloOptimizer, GridSearchOptimizer
   :skip: MoleculeTransformer, PartitioningProgramEnsemble, TimeEvolutionTrajectory, VQEHyperparameterSweep
   :skip: ProgramEnsemble, BatchConfig, BatchMode, EarlyStopping

Type Aliases
~~~~~~~~~~~~

.. autodata:: divi.qprog.variational_quantum_algorithm.ParamHistoryMode
   :no-value:

   Accepted values for
   :meth:`~divi.qprog.VariationalQuantumAlgorithm.param_history`:
   ``"all_evaluated"`` returns every parameter vector seen during optimization;
   ``"best_per_iteration"`` returns only the best vector from each iteration.
