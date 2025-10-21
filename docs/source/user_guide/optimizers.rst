Optimizers
==========

Divi provides built-in support for optimizing quantum programs using three distinct methods, each suited to different problem types and user requirements.

All optimizers can be accessed through the `divi.qprog.optimizers` module. Scipy-based optimizers rely on the `ScipyMethod` enum to specify the optimizer used.

Monte Carlo Optimization
-------------------------

The Monte Carlo [#kalos2008]_ method in Divi is a stochastic global optimization approach. It works by randomly sampling the parameter space and selecting configurations that minimize the target cost function. This method is particularly useful when:

- The optimization landscape is rugged or non-convex.
- Gradients are not available or are unreliable.
- A rough global search is preferred before local refinement.

Monte Carlo optimization can help identify promising regions in high-dimensional parameter spaces before applying more refined methods.

In Divi, one can configure the optimizer; providing the size of the population `n_param_sets`, and the number of well-performing parameter sets to carry on to the subsequent iteration `n_best_sets`.

SciPy Optimizers
----------------

Divi provides several SciPy-based optimizers through the `ScipyOptimizer` class:

Nelder-Mead
-----------

Nelder-Mead [#nelder1965]_ is a gradient-free, simplex-based optimization algorithm. It is ideal for local optimization in low to moderate dimensional spaces. The method iteratively refines a simplex (a geometrical figure defined by a set of parameter vectors) by evaluating cost function values and applying operations such as reflection, expansion, and contraction.

Use Nelder-Mead when:

- Your problem is continuous but noisy.
- Gradients are unavailable or expensive to compute.
- You are tuning parameters in a relatively low-dimensional space.

.. code-block:: python

   from divi.qprog.optimizers import ScipyOptimizer, ScipyMethod

   optimizer = ScipyOptimizer(method=ScipyMethod.NELDER_MEAD)

L-BFGS-B
--------

L-BFGS-B (Limited-memory Broyden–Fletcher–Goldfarb–Shanno with Bound constraints) [#zhu1997]_ is a quasi-Newton method that leverages gradient information to efficiently converge to a local minimum. In Divi, gradient calculation is performed using the parameter shift rule, a technique well-suited to quantum circuits that allows for analytical gradient computation by evaluating the function at shifted parameter values.

Divi computes these parameter shifts in parallel, significantly reducing wall-clock time for gradient evaluations.

Use L-BFGS-B when:

- You require fast convergence to a local minimum.
- Your cost function is smooth and differentiable.

.. code-block:: python

   optimizer = ScipyOptimizer(method=ScipyMethod.L_BFGS_B)

COBYLA
------

COBYLA (Constrained Optimization BY Linear Approximations) [#powell1994]_ is a gradient-free, local optimization method—like Nelder-Mead—that supports nonlinear inequality constraints. It constructs successive linear approximations of the objective function and constraints, iteratively refining the solution within a trust region.

Use COBYLA when:

- Your optimization problem includes constraints.
- Gradients are inaccessible or too noisy.
- You seek a reliable optimizer for low to moderate-dimensional spaces.

COBYLA is also a good choice of optimizer when trying out QAOA for a new problem/experimenting, but your mileage may vary.

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

Choosing the Right Optimizer
----------------------------

**For VQE:**

- **L-BFGS-B**: Best for smooth, differentiable landscapes with good initial parameters
- **Monte Carlo**: Excellent for exploration and avoiding local minima
- **COBYLA**: Good for constrained problems or when gradients are unreliable
- **Nelder-Mead**: Robust choice for noisy or discontinuous landscapes

**For QAOA:**

- **COBYLA**: Often the best starting point for QAOA problems
- **Nelder-Mead**: Good for noisy landscapes and parameter initialization
- **Monte Carlo**: Excellent for global exploration and avoiding barren plateaus
- **L-BFGS-B**: Use when you have good initial parameters and smooth landscapes

**For PyMOO Optimizers:**

- **CMA-ES**: Excellent for high-dimensional parameter spaces and when you need robust global optimization. Particularly effective for VQE with many parameters.
- **Differential Evolution**: Good for multimodal optimization landscapes and when you need to escape local minima. Works well for QAOA parameter optimization.

**For Hyperparameter Sweeps:**

- **Monte Carlo**: Best for initial exploration across parameter ranges
- **L-BFGS-B**: Use for fine-tuning after Monte Carlo exploration
- **Nelder-Mead**: Robust fallback when other methods fail
- **CMA-ES**: Excellent for high-dimensional sweeps with many parameters

**Quantum-Specific Considerations:**

- **Barren Plateaus**: Use Monte Carlo or CMA-ES to avoid getting trapped in flat regions
- **Parameter Initialization**: Start with small random values (typically [-0.1, 0.1]) for better convergence
- **Circuit Depth**: Deeper circuits benefit from more robust optimizers like CMA-ES or Monte Carlo
- **Noise Resilience**: Nelder-Mead and COBYLA are more robust to quantum noise than gradient-based methods

Next Steps
----------

- 🔬 **Tutorials**: Try the runnable examples in the `tutorials/ <https://github.com/QoroQuantum/divi/tree/main/tutorials>`_ directory.
- ⚡ **Algorithm Guides**: Learn about :doc:`vqe` and :doc:`qaoa` for algorithm-specific guidance.
- ⚡ **Batching and Sweeps**: See how to use optimizers in large-scale computations in the :doc:`program_batches` guide.

References
----------

.. [#kalos2008] Kalos, M. H., & Whitlock, P. A. (2008). *Monte Carlo Methods* (2nd ed.). Wiley-VCH.

.. [#nelder1965] Nelder, J. A., & Mead, R. (1965). A simplex method for function minimization. *The Computer Journal*, 7(4), 308–313.

.. [#zhu1997] Zhu, C., Byrd, R. H., Lu, P., & Nocedal, J. (1997). Algorithm 778: L-BFGS-B: Fortran subroutines for large-scale bound-constrained optimization. *ACM Transactions on Mathematical Software*, 23(4), 550–560.

.. [#powell1994] Powell, M. J. D. (1994). A direct search optimization method that models the objective and constraint functions by linear interpolation. In *Advances in Optimization and Numerical Analysis* (pp. 51–67). Springer.

.. [#hansen2001] Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation in evolution strategies. *Evolutionary Computation*, 9(2), 159–195.

.. [#storn1997] Storn, R., & Price, K. (1997). Differential evolution – a simple and efficient heuristic for global optimization over continuous spaces. *Journal of Global Optimization*, 11(4), 341–359.
