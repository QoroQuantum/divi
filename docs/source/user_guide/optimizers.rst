Optimizers
==========

Divi provides built-in support for optimizing quantum programs using a range of optimization methods, each suited to different problem types and user requirements.

All optimizers can be accessed through the ``divi.qprog.optimizers`` module. Scipy-based optimizers rely on the :class:`~divi.qprog.optimizers.ScipyMethod` enum to specify the optimizer used.

Monte Carlo Optimization
-------------------------

The Monte Carlo [#kalos2008]_ method in Divi is a stochastic global optimization approach. It works by randomly sampling the parameter space and selecting configurations that minimize the target cost function. This method is particularly useful when:

- The optimization landscape is rugged or non-convex.
- Gradients are not available or are unreliable.
- A rough global search is preferred before local refinement.

Monte Carlo optimization can help identify promising regions in high-dimensional parameter spaces before applying more refined methods.

Configure :class:`~divi.qprog.optimizers.MonteCarloOptimizer` by passing ``population_size`` (the number of parameter sets evaluated per iteration) and optionally ``n_best_sets`` (how many top-performing sets are carried to the next iteration) to its constructor. The read-only ``n_param_sets`` property then reflects the configured population size.

SciPy Optimizers
----------------

Divi provides several SciPy-based optimizers through the :class:`~divi.qprog.optimizers.ScipyOptimizer` class:

Nelder-Mead
^^^^^^^^^^^

Nelder-Mead [#nelder1965]_ is a gradient-free, simplex-based optimization algorithm. It is ideal for local optimization in low to moderate dimensional spaces. The method iteratively refines a simplex (a geometrical figure defined by a set of parameter vectors) by evaluating cost function values and applying operations such as reflection, expansion, and contraction.

Use Nelder-Mead when:

- Your problem is continuous but noisy.
- Gradients are unavailable or expensive to compute.
- You are tuning parameters in a relatively low-dimensional space.

.. code-block:: python

   from divi.qprog.optimizers import ScipyOptimizer, ScipyMethod

   optimizer = ScipyOptimizer(method=ScipyMethod.NELDER_MEAD)

L-BFGS-B
^^^^^^^^

L-BFGS-B (Limited-memory Broyden–Fletcher–Goldfarb–Shanno with Bound constraints) [#zhu1997]_ is a quasi-Newton method that leverages gradient information to efficiently converge to a local minimum. In Divi, gradient calculation is performed using the parameter shift rule, a technique well-suited to quantum circuits that allows for analytical gradient computation by evaluating the function at shifted parameter values.

Divi computes these parameter shifts in parallel, significantly reducing wall-clock time for gradient evaluations.

Use L-BFGS-B when:

- You require fast convergence to a local minimum.
- Your cost function is smooth and differentiable.

.. code-block:: python

   optimizer = ScipyOptimizer(method=ScipyMethod.L_BFGS_B)

COBYLA
^^^^^^

COBYLA (Constrained Optimization BY Linear Approximations) [#powell1994]_ is a gradient-free, local optimization method—like Nelder-Mead—that supports nonlinear inequality constraints. It constructs successive linear approximations of the objective function and constraints, iteratively refining the solution within a trust region.

Use COBYLA when:

- Your optimization problem includes constraints.
- Gradients are inaccessible or too noisy.
- You seek a reliable optimizer for low to moderate-dimensional spaces.

COBYLA is also a good choice of optimizer when trying out :class:`~divi.qprog.algorithms.QAOA` for a new problem/experimenting, but your mileage may vary.

.. code-block:: python

   optimizer = ScipyOptimizer(method=ScipyMethod.COBYLA)

PyMOO Optimizers
----------------

Divi also supports evolutionary algorithms through PyMOO:

CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CMA-ES [#hansen2001]_ is a stochastic, derivative-free method for numerical optimization of non-linear or non-convex continuous optimization problems.

.. code-block:: python

   from divi.qprog.optimizers import PymooOptimizer, PymooMethod

   optimizer = PymooOptimizer(method=PymooMethod.CMAES)

Differential Evolution
^^^^^^^^^^^^^^^^^^^^^^

Differential Evolution [#storn1997]_ is a method that optimizes a problem by iteratively trying to improve a candidate solution with regard to a given measure of quality.

.. code-block:: python

   optimizer = PymooOptimizer(method=PymooMethod.DE)

Quantum Natural Gradient
------------------------

The :class:`~divi.qprog.optimizers.QNGOptimizer` performs regularized natural-gradient descent:

.. math::

   \theta \leftarrow \theta - \eta \, (G + \lambda I)^{-1} \nabla L,

where :math:`\nabla L` is the parameter-shift gradient and :math:`G` is a
positive-semidefinite metric tensor. Preconditioning by the inverse metric
rescales the gradient to follow the geometry of the quantum state manifold,
which can dramatically reduce the number of iterations needed to converge
compared to vanilla gradient descent — particularly in circuits with many
parameters or pronounced curvature variation.

**Metric estimators.** The optimizer is metric-agnostic: the metric is
produced by an injected :class:`~divi.qprog.optimizers.MetricEstimator`
strategy. Swapping the estimator changes the metric without touching the
optimizer itself. Two estimators are provided:

- :class:`~divi.qprog.optimizers.PullbackMetricEstimator` *(default)* —
  Hamiltonian-aware pullback metric,
  :math:`G_{ij} = \sum_r a_r^2 \,(\partial_i \langle P_r \rangle)(\partial_j \langle P_r \rangle)`,
  computed from the per-Pauli-term expectation gradients of the loss
  observable :math:`H = \sum_r a_r P_r`. The gradient and metric share one
  parameter-shift measurement pass (a single memoized evaluation), so there
  is no extra circuit overhead relative to a standard gradient step.
  Compatible with VQE, QAOA, and ``CustomVQA`` programs whose loss is the
  expectation value of the cost Hamiltonian. Rejects programs with a
  classical loss objective (PCE) or a supervised data-bound loss.

- :class:`~divi.qprog.optimizers.FubiniStudyMetricEstimator` — block-diagonal
  Fubini–Study metric (quantum geometric tensor), computed from the covariance
  of the Hermitian generators of each layer's parametric gates. Independent of
  the loss observable, so it applies to any program with a Pauli-rotation
  ansatz — including PCE. Provides only ``metric_fn``; the gradient falls back
  to the program's own parameter-shift rule. Rejects ansatze that use
  unsupported or composite-angle gates, and data-bound programs.

**Solver options.** The optimizer regularizes the metric before inverting to
prevent divergence along flat directions:

- ``solver="tikhonov"`` *(default)* — solves :math:`(G + \lambda I)\,\delta = \nabla L`
  via a Cholesky-based symmetric solve, exploiting the PSD structure.
- ``solver="pinv"`` — applies the Moore–Penrose pseudo-inverse of the raw
  (undamped) metric with cutoff ``rcond``.

**Usage** is the same as any optimizer — pass an instance via the
``optimizer=`` argument and call ``run()``:

.. code-block:: python

   from divi.qprog import VQE
   from divi.qprog.optimizers import QNGOptimizer

   vqe = VQE(
       molecule=molecule,
       backend=backend,
       optimizer=QNGOptimizer(step_size=0.1, regularization=1e-3),
       max_iterations=10,
   )
   vqe.run()

To switch to the Fubini–Study metric, inject a different estimator:

.. code-block:: python

   from divi.qprog.optimizers import FubiniStudyMetricEstimator, QNGOptimizer

   optimizer = QNGOptimizer(
       metric_estimator=FubiniStudyMetricEstimator(),
   )

.. note::

   QNG does not support checkpointing (``supports_checkpointing`` is
   ``False``). Passing ``checkpoint_config`` with a checkpoint directory to
   ``run()`` raises a ``ValueError`` upfront. The variational algorithm
   already checkpoints the parameter history, so optimizer-level state is
   not needed.

.. note::

   Both metric estimators compute the metric exactly per iteration and
   require additional circuit evaluations beyond a plain gradient step.
   QNG is best suited to simulators or small-to-moderate problems where
   the measurement overhead is acceptable. For large hardware runs with
   tight shot budgets, standard gradient-based optimizers may be more
   practical.

.. note::

   **QNG with QDrift.** When a QAOA program uses a stochastic
   :class:`~divi.hamiltonians.QDrift` trotterization, the gradient and the
   metric must be evaluated on the *same* sampled Hamiltonian — otherwise the
   natural-gradient step mixes mismatched operators. Divi guarantees this: the
   cost, gradient, and metric pipelines all draw the same QDrift batch within
   one optimizer evaluation (the sample is keyed on an internal per-evaluation
   counter) and resample on the next. The draw is reproducible from the
   ``QDrift(seed=...)`` you provide; with no seed it is still consistent within
   each evaluation but varies across runs. ``n_hamiltonians_per_iteration``
   controls how many independent samples are averaged per evaluation — higher
   values reduce variance at a proportional increase in circuits.

Use QNG when:

- You are running VQE, QAOA, or PCE on a simulator and want faster
  parameter convergence relative to vanilla gradient descent.
- Your landscape has strong curvature variation across parameters.
- You are using the pullback metric and want gradient + metric from a
  single measurement pass.

Grid Search
-----------

The :class:`~divi.qprog.optimizers.GridSearchOptimizer` performs an exhaustive evaluation of every
point on a user-defined parameter grid and returns the best-performing
combination. It is designed for low-dimensional parameter spaces (1–3
parameters) where you want full visibility into the loss landscape.

Use Grid Search when:

- You have a small number of variational parameters (e.g. QAOA with 1 layer: γ and β).
- You want to visualize the loss landscape.
- You need a deterministic, reproducible sweep.
- You want to warm-start a variational optimizer from the best grid point.

.. code-block:: python

   import numpy as np

   from divi.qprog.optimizers import GridSearchOptimizer

   # Auto-generate a 20×20 grid over [0, 2π] × [0, π]
   optimizer = GridSearchOptimizer(
       param_ranges=[(0, 2 * 3.14159), (0, 3.14159)],
       grid_points=20,
   )

   # Or supply an explicit grid
   optimizer = GridSearchOptimizer(
       param_grid=np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
   )

The grid is evaluated in a single pass regardless of ``max_iterations``.
A warning is issued if ``max_iterations > 1`` is supplied.

.. note::

   Grid search scales as ``grid_points ** n_params``, so it becomes
   impractical beyond ~3 parameters. For higher dimensions, use
   :class:`~divi.qprog.optimizers.MonteCarloOptimizer` or CMA-ES instead.

Choosing the Right Optimizer
----------------------------

**For :class:`~divi.qprog.algorithms.VQE`:**

- **QNG (pullback)**: Best when the landscape is smooth and you want faster per-iteration
  progress than L-BFGS-B at the cost of additional metric measurements; requires a
  Hamiltonian-expectation-value loss (not PCE)
- **L-BFGS-B**: Best for smooth, differentiable landscapes with good initial parameters
- **Monte Carlo**: Excellent for exploration and avoiding local minima
- **COBYLA**: Good for constrained problems or when gradients are unreliable
- **Nelder-Mead**: Robust choice for noisy or discontinuous landscapes

**For :class:`~divi.qprog.algorithms.QAOA`:**

- **Grid Search**: Best for 1–2 layer QAOA where you want full landscape visibility
- **QNG (pullback)**: Accelerates convergence in simulator runs; requires a cost
  Hamiltonian expectation loss
- **COBYLA**: Often the best starting point for :class:`~divi.qprog.algorithms.QAOA` problems
- **Nelder-Mead**: Good for noisy landscapes and parameter initialization
- **Monte Carlo**: Excellent for global exploration and avoiding barren plateaus
- **L-BFGS-B**: Use when you have good initial parameters and smooth landscapes

**For PyMOO Optimizers:**

- **CMA-ES**: Excellent for high-dimensional parameter spaces and when you need robust global optimization. Particularly effective for :class:`~divi.qprog.algorithms.VQE` with many parameters.
- **Differential Evolution**: Good for multimodal optimization landscapes and when you need to escape local minima. Works well for :class:`~divi.qprog.algorithms.QAOA` parameter optimization.

**For Hyperparameter Sweeps:**

- **Monte Carlo**: Best for initial exploration across parameter ranges
- **L-BFGS-B**: Use for fine-tuning after Monte Carlo exploration
- **Nelder-Mead**: Robust fallback when other methods fail
- **CMA-ES**: Excellent for high-dimensional sweeps with many parameters

**Quantum-Specific Considerations:**

- **Barren Plateaus**: Use :class:`~divi.qprog.optimizers.MonteCarloOptimizer` or CMA-ES to avoid getting trapped in flat regions
- **Parameter Initialization**: Start with small random values (typically [-0.1, 0.1]) for better convergence
- **Circuit Depth**: Deeper circuits benefit from more robust optimizers like CMA-ES or :class:`~divi.qprog.optimizers.MonteCarloOptimizer`
- **Noise Resilience**: Nelder-Mead and COBYLA are more robust to quantum noise than gradient-based methods
- **Natural Gradient**: Use :class:`~divi.qprog.optimizers.QNGOptimizer` on simulators when the circuit has pronounced
  curvature variation and you want metric-aware updates; not recommended for large hardware runs
  where extra metric measurements are expensive

Early Stopping
--------------

Long-running optimizations can waste resources once convergence has effectively
stalled.  Divi's :class:`~divi.qprog.early_stopping.EarlyStopping` controller lets you
terminate the loop automatically based on configurable criteria.

Pass an ``EarlyStopping`` instance to any variational algorithm:

.. code-block:: python

   from divi.qprog import VQE, EarlyStopping
   from divi.qprog.optimizers import ScipyOptimizer, ScipyMethod

   vqe = VQE(
       molecule=molecule,
       backend=backend,
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
       max_iterations=200,
       early_stopping=EarlyStopping(
           patience=10,
           min_delta=1e-5,
       ),
   )

   vqe.run()
   print(f"Stopped at iteration {vqe.current_iteration}")
   print(f"Reason: {vqe.stop_reason}")        # e.g. "patience_exceeded"
   print(f"Converged: {vqe.optimize_result.success}")  # False for early stop

Stopping Criteria
^^^^^^^^^^^^^^^^^

Three criteria are available and are evaluated **in priority order** after every
iteration.  The first one that fires stops the loop.

1. **Patience** *(always active)* — Stop when the cost has not improved by at
   least ``min_delta`` for ``patience`` consecutive iterations.

   .. code-block:: python

      EarlyStopping(patience=10, min_delta=1e-4)

2. **Gradient norm** *(optional)* — Stop when the L2 norm of the gradient falls
   below ``grad_norm_threshold``.  Only effective with gradient-based optimizers
   such as ``ScipyOptimizer(method=ScipyMethod.L_BFGS_B)``.

   .. code-block:: python

      EarlyStopping(patience=10, grad_norm_threshold=1e-6)

3. **Cost variance** *(optional)* — Stop when the rolling variance of the last
   ``variance_window`` cost values drops below ``variance_threshold``.  Useful
   for noisy landscapes where cost oscillates but no longer trends downward.

   .. code-block:: python

      EarlyStopping(
          patience=10,
          variance_window=20,
          variance_threshold=1e-8,
      )

All three criteria can be enabled simultaneously; the first one that triggers
will stop the loop.

After the Run
^^^^^^^^^^^^^

After ``run()`` completes, use :attr:`~divi.qprog.VariationalQuantumAlgorithm.stop_reason`
to determine *why* optimization ended:

- ``None`` — optimization ran to ``max_iterations`` without triggering early stopping
- ``"patience_exceeded"`` — cost plateaued
- ``"gradient_below_threshold"`` — gradient vanished
- ``"cost_variance_settled"`` — cost variance settled

The ``optimize_result`` attribute
is always populated and its ``message`` field includes the stop reason.

Inspecting Optimizer Results
----------------------------

After running a variational algorithm, you can inspect the raw result object
returned by the underlying optimizer via the
``optimize_result`` property.
This exposes optimizer-specific diagnostics such as:

- ``nfev`` – number of cost-function evaluations
- ``njev`` – number of Jacobian (gradient) evaluations (gradient-based optimizers)
- ``nit`` – number of iterations completed
- ``success`` – whether the optimizer converged
- ``message`` – convergence or termination message

.. skip: next

.. code-block:: python

   program.run()

   result = program.optimize_result
   if result is not None:
       print(f"Function evaluations: {result.nfev}")
       print(f"Converged: {result.success}")

.. note::

   ``optimize_result`` is always populated after :meth:`~divi.qprog.VariationalQuantumAlgorithm.run` completes.
   When optimization converges normally, ``success`` is ``True``.
   When early stopping or cancellation terminates the run, ``success`` is
   ``False`` and the ``message`` field describes the reason.  The available
   attributes depend on the optimizer; see :class:`scipy.optimize.OptimizeResult`
   for the full specification.

Next Steps
----------

- `tutorials/ <https://github.com/QoroQuantum/divi/tree/main/tutorials>`_ — runnable examples
- :doc:`ground_state_energy_estimation_vqe` and :doc:`combinatorial_optimization_qaoa_pce` — algorithm-specific guidance
- :doc:`program_ensembles` — optimizers in large-scale sweeps and ensembles

References
----------

.. [#kalos2008] Kalos, M. H., & Whitlock, P. A. (2008). *Monte Carlo Methods* (2nd ed.). Wiley-VCH.

.. [#nelder1965] Nelder, J. A., & Mead, R. (1965). A simplex method for function minimization. *The Computer Journal*, 7(4), 308–313.

.. [#zhu1997] Zhu, C., Byrd, R. H., Lu, P., & Nocedal, J. (1997). Algorithm 778: L-BFGS-B: Fortran subroutines for large-scale bound-constrained optimization. *ACM Transactions on Mathematical Software*, 23(4), 550–560.

.. [#powell1994] Powell, M. J. D. (1994). A direct search optimization method that models the objective and constraint functions by linear interpolation. In *Advances in Optimization and Numerical Analysis* (pp. 51–67). Springer.

.. [#hansen2001] Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation in evolution strategies. *Evolutionary Computation*, 9(2), 159–195.

.. [#storn1997] Storn, R., & Price, K. (1997). Differential evolution – a simple and efficient heuristic for global optimization over continuous spaces. *Journal of Global Optimization*, 11(4), 341–359.
