Optimizers
==========

Choose an optimizer from the properties of the run, not from an absolute
ranking. Shot noise, parameter count, gradient availability, and circuit budget
usually matter more than the algorithm name.

.. list-table:: Optimizer selection
   :header-rows: 1
   :widths: 20 25 25 30

   * - Start with
     - Suitable regime
     - Per-step cost
     - Main caveat
   * - COBYLA or Nelder–Mead
     - Small or moderate, gradient-free problems
     - Grows with parameter count
     - Local methods; repeated starts may find different minima.
   * - Monte Carlo or CMA-ES
     - Global exploration and multimodal landscapes
     - Population-sized batches
     - More evaluations before local convergence.
   * - SPSA or QN-SPSA
     - Many parameters on shot-based backends
     - Constant number of objective evaluations
     - Stochastic directions require tuning and averaging.
   * - QUIVER
     - Shot-based runs needing a tunable accuracy/budget trade-off
     - Controlled by direction count ``V``
     - Parameter-shift mode is exact only where the program provides an exact
       shift rule.
   * - L-BFGS-B or QNG
     - Smooth objectives with an exact program gradient
     - Parameter-shift and, for QNG, metric evaluations
     - Not supported by QAOA, which has no exact two-term parameter-shift rule.
   * - Grid search
     - One or two parameters, especially shallow QAOA
     - Exponential in parameter count
     - Impractical beyond roughly three parameters.

All optimizer classes are listed in the
:doc:`optimizer API reference <../api_reference/qprog/optimizers>`. SciPy
methods are selected with :class:`~divi.qprog.optimizers.ScipyMethod`. The
sections below explain the trade-offs and program-specific restrictions in
more detail.

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

.. note::

   On small or barren-plateau circuits the parameter-shift gradient can be
   near-zero at the start, so L-BFGS-B may hit its ``gtol`` and stop after a
   couple of iterations (a flat, uninformative loss trajectory). If that
   happens, switch to a gradient-free optimizer (COBYLA, Nelder-Mead) or the
   Monte Carlo optimizer for a more illustrative run.

.. note::

   :class:`~divi.qprog.algorithms.QAOA` has no exact parameter-shift rule. Each
   layer angle drives one rotation per Hamiltonian term, at an angle scaled by
   that term's coefficient, so its gradient is not recoverable from a single
   pair of shifted evaluations. Gradient-based optimizers — L-BFGS-B and the
   natural-gradient optimizers — raise on QAOA rather than return a wrong
   gradient. Use a gradient-free or evolutionary optimizer instead.

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

The :class:`~divi.qprog.optimizers.QNGOptimizer` preconditions the
parameter-shift gradient with a regularized metric:

.. math::

   \theta \leftarrow \theta - \eta \, (G + \lambda I)^{-1} \nabla L,

This can improve convergence when parameter directions have different
curvatures, at the cost of metric evaluations.

Choose a :class:`~divi.qprog.optimizers.MetricEstimator`:

- :class:`~divi.qprog.optimizers.PullbackMetricEstimator` *(default)* shares
  the Hamiltonian-gradient measurement pass. It supports expectation-valued
  VQE and CustomVQA, but not PCE or supervised data-bound losses.

- :class:`~divi.qprog.optimizers.FubiniStudyMetricEstimator` is
  observable-independent and supports PCE with Pauli-rotation ansatze. It
  rejects unsupported/composite-angle gates and data-bound programs.

``solver="tikhonov"`` regularizes flat directions; ``solver="pinv"`` uses a
pseudo-inverse with cutoff ``rcond``.

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

Use QNG on small-to-moderate exact-gradient problems where metric overhead is
acceptable. It does not support QAOA or checkpointing.

Simultaneous Perturbation (SPSA / QN-SPSA)
------------------------------------------

SPSA and QN-SPSA use a constant number of evaluations per step, making them
useful for many-parameter, shot-noisy circuits.

SPSA
^^^^

The :class:`~divi.qprog.optimizers.SPSAOptimizer` [#spall1992]_ perturbs all
parameters along one random :math:`\pm 1` direction and estimates the gradient
from two evaluations:

.. math::

   \hat g_k = \frac{f(\theta + c_k h) - f(\theta - c_k h)}{2 c_k}\, h,
   \qquad \theta \leftarrow \theta - a_k \hat g_k,

The two points share a batch, so stochastic costs use one sampled Hamiltonian.

.. code-block:: python

   from divi.qprog.optimizers import SPSAOptimizer

   optimizer = SPSAOptimizer(learning_rate=0.2, c=0.2)

Set ``c`` near the cost's shot-noise standard deviation, then tune
``learning_rate``. ``resamplings`` reduces gradient variance at proportional
cost. ``blocking`` spends an extra evaluation to reject steps that exceed
``blocking_tol`` times recent loss variation.

.. note::

   A 1000× loss increase warns once. Enable ``blocking``, raise
   ``regularization``, or lower ``learning_rate``.

.. note::

   Pass ``rng=`` for reproducible directions. ``exact_loss=True`` spends one
   extra evaluation for unbiased history and best-iterate selection. A QDrift
   seed does not seed SPSA directions.

QN-SPSA
^^^^^^^

The :class:`~divi.qprog.optimizers.QNSPSAOptimizer` [#gacon2021]_ preconditions
SPSA with a pluggable metric:

- :class:`~divi.qprog.optimizers.StochasticFidelityMetricEstimator` *(default)*
  adds four state-overlap evaluations per step to SPSA's two objective
  evaluations. It supports Qiskit-invertible ansatze, not data-bound programs.
  QDrift uses one fixed ansatz realization.

- Fubini–Study or pullback estimators provide exact metrics, but their cost
  scales with parameter count.

.. code-block:: python

   from divi.qprog.optimizers import QNSPSAOptimizer

   # Faithful stochastic-fidelity metric (default)
   optimizer = QNSPSAOptimizer(learning_rate=0.01, c=0.2, regularization=1e-3)

   # Or reuse an exact metric with the SPSA gradient
   from divi.qprog.optimizers import FubiniStudyMetricEstimator

   optimizer = QNSPSAOptimizer(
       metric_estimator=FubiniStudyMetricEstimator(),
   )

QN-SPSA usually needs a smaller ``learning_rate`` than SPSA. For unstable runs,
raise ``resamplings`` or ``regularization``, or enable ``blocking``.

.. note::

   Like QNG, neither SPSA nor QN-SPSA supports checkpointing
   (``supports_checkpointing`` is ``False``): their only persistent state is the
   parameter vector, which the variational algorithm already records. The
   per-step gains, blocking history, and running-average metric are recomputed
   each run.

QUIVER (Adaptive Directional Gradients)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~divi.qprog.optimizers.QUIVEROptimizer` [#coyle2026]_ estimates the
gradient from ``V`` random directional derivatives:

.. math::

   \tilde\nabla^{\mathsf F} f = \frac{1}{V}\sum_{\ell=1}^{V}
   \frac{f(\theta + \varepsilon v_\ell) - f(\theta - \varepsilon v_\ell)}
   {2\varepsilon}\, v_\ell,
   \qquad \theta \leftarrow \theta - a_k\,\tilde\nabla^{\mathsf F} f,

This costs :math:`2V` evaluations: ``V=1`` resembles SPSA; larger ``V`` trades
cost for precision.

QUIVER can adapt ``V`` and the per-direction shot count ``M``:

- ``V`` follows directional-sample spread.
- ``M`` follows measurement variance on sampling backends and stays fixed on
  native-expval backends.

.. code-block:: python

   from divi.qprog.optimizers import QUIVEROptimizer

   optimizer = QUIVEROptimizer(learning_rate=0.1, epsilon=0.1, V_init=2)

Use ``derivative_mode="parameter_shift"`` where an exact directional shift is
valid; ``adapt_V=False`` or ``adapt_M=False`` pins the budget. QUIVER supports
SPSA's ``blocking`` and ``exact_loss`` options, but not checkpointing.

.. note::

   Adaptive ``M`` disables circuit-template batching and cannot combine with
   ``shot_distribution``. Prefer fixed ``M`` when cloud submission overhead
   dominates.

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

Program-Specific Constraints
----------------------------

- :class:`~divi.qprog.algorithms.QAOA` has no exact two-term parameter-shift
  gradient. Use gradient-free SciPy methods, evolutionary methods, SPSA,
  QN-SPSA, QUIVER in finite-difference mode, or a low-dimensional grid search.
- VQE and ``CustomVQA`` can use exact-gradient methods only when their ansatz
  declares a valid parameter-shift spectrum.
- PCE uses a classical counts-based objective. Default pullback QNG does not
  apply; use the Fubini–Study estimator or a gradient-free optimizer.
- Checkpointing requires an optimizer with restorable state. See
  :doc:`resuming_long_runs` for the compatibility table.


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

After running a variational algorithm, the loss history is available directly
on the program object via
:attr:`~divi.qprog.VariationalQuantumAlgorithm.min_losses_per_iteration` and
:attr:`~divi.qprog.VariationalQuantumAlgorithm.losses_history` — see
:ref:`reading results <reading-results>` in core concepts for their
semantics and types.

Beyond the loss history, you can inspect the raw result object returned by the
underlying optimizer
via the ``optimize_result`` property.
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
- :doc:`../algorithms/ground_state_energy_estimation_vqe` and :doc:`../algorithms/combinatorial_optimization_qaoa_pce` — algorithm-specific guidance
- :doc:`program_ensembles` — optimizers in large-scale sweeps and ensembles

References
----------

.. [#kalos2008] Kalos, M. H., & Whitlock, P. A. (2008). *Monte Carlo Methods* (2nd ed.). Wiley-VCH.

.. [#nelder1965] Nelder, J. A., & Mead, R. (1965). A simplex method for function minimization. *The Computer Journal*, 7(4), 308–313.

.. [#zhu1997] Zhu, C., Byrd, R. H., Lu, P., & Nocedal, J. (1997). Algorithm 778: L-BFGS-B: Fortran subroutines for large-scale bound-constrained optimization. *ACM Transactions on Mathematical Software*, 23(4), 550–560.

.. [#powell1994] Powell, M. J. D. (1994). A direct search optimization method that models the objective and constraint functions by linear interpolation. In *Advances in Optimization and Numerical Analysis* (pp. 51–67). Springer.

.. [#hansen2001] Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation in evolution strategies. *Evolutionary Computation*, 9(2), 159–195.

.. [#storn1997] Storn, R., & Price, K. (1997). Differential evolution – a simple and efficient heuristic for global optimization over continuous spaces. *Journal of Global Optimization*, 11(4), 341–359.

.. [#spall1992] Spall, J. C. (1992). Multivariate stochastic approximation using a simultaneous perturbation gradient approximation. *IEEE Transactions on Automatic Control*, 37(3), 332–341.

.. [#gacon2021] Gacon, J., Zoufal, C., Carleo, G., & Woerner, S. (2021). Simultaneous perturbation stochastic approximation of the quantum Fisher information. *Quantum*, 5, 567.

.. [#coyle2026] Coyle, B., Raj, S., Umathe, V., Cherrat, E. A., & Kashefi, E. (2026). Adaptive directional gradients for parameterised quantum circuits. *arXiv preprint* arXiv:2606.09734.
