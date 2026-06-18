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

Simultaneous Perturbation (SPSA / QN-SPSA)
------------------------------------------

SPSA and its quantum-natural variant estimate their search direction from a
*constant* number of cost evaluations per step, independent of the number of
parameters. This makes them attractive for many-parameter, shot-noisy circuits
where the parameter-shift rule — which scales with the parameter count — is
prohibitively expensive.

SPSA
^^^^

The :class:`~divi.qprog.optimizers.SPSAOptimizer` [#spall1992]_ approximates the
gradient from just **two** cost evaluations per step by perturbing all
parameters simultaneously along a random Bernoulli :math:`\pm 1` direction
:math:`h`:

.. math::

   \hat g_k = \frac{f(\theta + c_k h) - f(\theta - c_k h)}{2 c_k}\, h,
   \qquad \theta \leftarrow \theta - a_k \hat g_k,

with decaying gains :math:`a_k = a/(A + k + 1)^\alpha` and
:math:`c_k = c/(k + 1)^\gamma`. The two perturbed points are evaluated as a
single batch, so a stochastic cost (e.g. a :class:`~divi.hamiltonians.QDrift`
QAOA cost) scores both against the same sampled Hamiltonian. SPSA is
gradient-free: it ignores any parameter-shift gradient the algorithm would
otherwise compute.

.. code-block:: python

   from divi.qprog.optimizers import SPSAOptimizer

   optimizer = SPSAOptimizer(learning_rate=0.2, c=0.2)

A good starting point is to set ``c`` near the standard deviation of the cost's
shot noise (so the finite difference clears the noise floor) and tune
``learning_rate`` from there. ``resamplings`` averages several gradient samples
per step to reduce variance (at proportional cost). The optional ``blocking``
guard performs *look-ahead* blocking — it evaluates the candidate's loss and
rejects the step if it would worsen the loss by more than ``blocking_tol`` times
the std of the recent window, otherwise accepting it. This prevents a single bad
estimate (or a divergent preconditioned step in QN-SPSA) from corrupting the run,
at the cost of one extra evaluation per step (plus one to seed the baseline).
``blocking_tol`` — not ``resamplings`` — is the knob that absorbs cost noise in
that single-evaluation accept/reject decision.

.. note::

   When ``blocking`` is off and a run diverges (most likely QN-SPSA in high
   dimensions), best-iterate tracking would otherwise return an early finite
   iterate with no signal that the run blew up. The optimizers emit a one-time
   ``UserWarning`` when the loss grows by more than 1000× its starting value —
   if you see it, enable ``blocking``, raise ``regularization``, or lower
   ``learning_rate``.

.. note::

   The perturbation directions are drawn from a random generator, so runs vary
   by default. Pass ``rng=`` (a ``numpy.random.Generator``) for reproducibility.
   The loss reported to the history is the average of the two perturbed
   evaluations — a free but ``O(c_k²)``-biased estimate of ``f(θ)``; set
   ``exact_loss=True`` to spend one extra unperturbed evaluation per step for the
   exact value (used for the recorded loss and best-iterate selection). Note that
   a ``QDrift(seed=...)`` seed fixes only the Hamiltonian sampling, not these
   perturbation directions.

QN-SPSA
^^^^^^^

The :class:`~divi.qprog.optimizers.QNSPSAOptimizer` [#gacon2021]_ combines the
SPSA gradient with a metric-preconditioned update, so it follows the geometry of
the state manifold like :class:`~divi.qprog.optimizers.QNGOptimizer` while
keeping a constant per-step circuit budget. Like QNG, the metric backend is
pluggable via the ``metric_estimator`` argument:

- :class:`~divi.qprog.optimizers.StochasticFidelityMetricEstimator` *(default)* —
  the faithful QN-SPSA metric, estimated from state-overlap fidelities
  :math:`F(\theta_1, \theta_2) = |\langle\psi(\theta_1)|\psi(\theta_2)\rangle|^2`
  (measured as the all-zeros probability of the compute-uncompute circuit
  :math:`U(\theta_1)\,U(\theta_2)^\dagger`) using two random directions and four
  overlap evaluations per step. The samples are accumulated into a running
  average seeded at the identity, conditioned as :math:`|\bar g| + \beta I`, and
  used to precondition the SPSA gradient. Because the overlap depends only on the
  ansatz state — not the loss observable — it applies to any qiskit-invertible
  ansatz, and rejects data-bound programs (and any ansatz qiskit cannot invert).
  For QDrift QAOA the metric is built from the fixed cost-ansatz realization
  captured at construction (it does not re-sample per evaluation), so it stays
  consistent across the run.

- :class:`~divi.qprog.optimizers.FubiniStudyMetricEstimator` or
  :class:`~divi.qprog.optimizers.PullbackMetricEstimator` — use the estimator's
  *exact* metric (as in QNG) while keeping the SPSA gradient. The metric cost
  then scales with the parameter count rather than staying constant.

.. code-block:: python

   from divi.qprog.optimizers import QNSPSAOptimizer

   # Faithful stochastic-fidelity metric (default)
   optimizer = QNSPSAOptimizer(learning_rate=0.01, c=0.2, regularization=1e-3)

   # Or reuse an exact metric with the SPSA gradient
   from divi.qprog.optimizers import FubiniStudyMetricEstimator

   optimizer = QNSPSAOptimizer(
       metric_estimator=FubiniStudyMetricEstimator(),
   )

QN-SPSA preconditions the step by the (inverse) metric, so it typically uses a
smaller raw ``learning_rate`` than plain SPSA. The constant per-step cost buys a
*noisier* metric estimate than QNG's exact metric. In high dimensions this noisy,
low-rank metric estimate can occasionally drive a divergent step; raise
``resamplings`` (less metric noise), increase ``regularization``, or enable
``blocking`` (which rejects such steps outright) if the optimization is unstable.

.. note::

   Like QNG, neither SPSA nor QN-SPSA supports checkpointing
   (``supports_checkpointing`` is ``False``): their only persistent state is the
   parameter vector, which the variational algorithm already records. The
   per-step gains, blocking history, and running-average metric are recomputed
   each run.

Use SPSA / QN-SPSA when:

- Your circuit has many parameters and parameter-shift gradients are too costly.
- The cost is shot-noisy and you want a method designed around stochastic
  evaluations.
- *(QN-SPSA)* You want metric-aware updates at a constant per-step circuit cost,
  trading the exact metric for a stochastic estimate.

QUIVER (Adaptive Directional Gradients)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :class:`~divi.qprog.optimizers.QUIVEROptimizer` [#coyle2026]_ reconstructs the
full gradient from ``V`` random Rademacher (:math:`\pm 1`) directional
derivatives, independent of the parameter count :math:`N`:

.. math::

   \tilde\nabla^{\mathsf F} f = \frac{1}{V}\sum_{\ell=1}^{V}
   \frac{f(\theta + \varepsilon v_\ell) - f(\theta - \varepsilon v_\ell)}
   {2\varepsilon}\, v_\ell,
   \qquad \theta \leftarrow \theta - a_k\,\tilde\nabla^{\mathsf F} f,

costing :math:`2V` evaluations per step. A single direction (:math:`V = 1`)
recovers SPSA; :math:`V = N` recovers the full parameter-shift gradient — so
``V`` dials between cheap-but-noisy and expensive-but-precise gradient estimates.

QUIVER also adapts ``V`` and the per-direction shot count ``M`` each step
(iCANS/gCANS-style), spending more directions when the gradient estimate is noisy
relative to its magnitude, and more shots when measurement noise dominates:

- ``V`` is driven by the spread of the ``V`` directional samples — it needs no
  backend variance, so it adapts on any cost function.
- ``M`` is driven by the measurement-variance estimate the cost closure exposes
  on shot-based backends. On native-expectation-value backends (no shot counts)
  it falls back to a fixed ``M`` and ``V``-from-spread only.

.. code-block:: python

   from divi.qprog.optimizers import QUIVEROptimizer

   optimizer = QUIVEROptimizer(learning_rate=0.1, epsilon=0.1, V_init=2)

Set ``derivative_mode="parameter_shift"`` to use a :math:`\pi/2` directional shift
instead of the finite-difference step ``epsilon``; set ``adapt_V=False`` /
``adapt_M=False`` to pin a fixed budget. QUIVER shares SPSA's ``blocking`` and
``exact_loss`` options, is gradient-free, and does not support checkpointing.

.. note::

   An adapting ``M`` delivers its per-evaluation shot budget to the backend as
   explicit per-circuit shot groups, which disables circuit-template batching for
   that submission. On template-capable backends (e.g. the Qoro cloud) this trades
   template reuse for shot adaptivity — prefer ``adapt_M=False`` there if
   submission overhead dominates, and reserve ``adapt_M`` for local shot-based
   simulators. ``adapt_M`` also assumes a uniform per-group shot count, so it is
   not combined with a configured ``shot_distribution`` (a warning is emitted).

Use QUIVER when:

- You want SPSA's constant-cost appeal but with a tunable accuracy/cost trade-off
  via ``V``.
- You want the per-step measurement budget to adapt automatically to shot noise.

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
- **QN-SPSA**: Metric-aware updates like QNG but at a constant per-step circuit
  cost; best for many-parameter circuits where exact gradients/metrics are too expensive
- **SPSA**: Gradient-free with two evaluations per step; best for many-parameter,
  shot-noisy circuits
- **QUIVER**: Forward-gradient generalization of SPSA with a tunable direction
  count ``V`` and adaptive shot allocation; best when you want to trade gradient
  accuracy against measurement budget on shot-based backends
- **L-BFGS-B**: Best for smooth, differentiable landscapes with good initial parameters
- **Monte Carlo**: Excellent for exploration and avoiding local minima
- **COBYLA**: Good for constrained problems or when gradients are unreliable
- **Nelder-Mead**: Robust choice for noisy or discontinuous landscapes

**For :class:`~divi.qprog.algorithms.QAOA`:**

- **Grid Search**: Best for 1–2 layer QAOA where you want full landscape visibility
- **QNG (pullback)**: Accelerates convergence in simulator runs; requires a cost
  Hamiltonian expectation loss
- **QN-SPSA / SPSA**: Best for deep, many-layer QAOA on shot-noisy backends, where
  the constant per-step cost beats parameter-shift gradients
- **QUIVER**: When you want SPSA's low per-step cost on shot-noisy QAOA but with a
  tunable direction count and adaptive shot allocation
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

.. [#spall1992] Spall, J. C. (1992). Multivariate stochastic approximation using a simultaneous perturbation gradient approximation. *IEEE Transactions on Automatic Control*, 37(3), 332–341.

.. [#gacon2021] Gacon, J., Zoufal, C., Carleo, G., & Woerner, S. (2021). Simultaneous perturbation stochastic approximation of the quantum Fisher information. *Quantum*, 5, 567.

.. [#coyle2026] Coyle, B., Raj, S., Umathe, V., Cherrat, E. A., & Kashefi, E. (2026). Adaptive directional gradients for parameterised quantum circuits. *arXiv preprint* arXiv:2606.09734.
