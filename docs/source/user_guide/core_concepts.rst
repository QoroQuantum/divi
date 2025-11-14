Core Concepts
=============

This guide explains the fundamental concepts and architecture that make Divi work. Understanding these concepts will help you use Divi more effectively and build custom quantum algorithms.

The QuantumProgram Base Class
-----------------------------

All quantum algorithms in Divi inherit from the abstract base class :class:`QuantumProgram`, which provides a streamlined interface for all quantum programs. Its primary role is to establish a common structure for executing circuits and managing backend communication.

**Core Features:**

- **Backend Integration** 🔗 - Unified interface for simulators and hardware
- **Execution Lifecycle** 🔄 - Standardized methods for running programs
- **Result Handling** 📊 - A common structure for processing backend results
- **Error Handling** 🛡️ - Graceful handling of execution failures

**Key Properties:**

- ``total_circuit_count`` - Total circuits executed so far
- ``total_run_time`` - Cumulative execution time in seconds

The VariationalQuantumAlgorithm Class
---------------------------------------

For algorithms that rely on optimizing parameters, Divi provides the :class:`VariationalQuantumAlgorithm` class. This is the base class for algorithms like VQE and QAOA, and it extends `QuantumProgram` with advanced features for optimization and result tracking.

.. note::
   For complete API documentation of all properties and methods, see :doc:`../api_reference/qprog`.

Every variational quantum program in Divi follows a consistent lifecycle:

1. **Initialization** 🎯 - Set up your problem and algorithm parameters
2. **Circuit Generation** ⚡ - Create quantum circuits from your problem specification
3. **Optimization** 🔄 - Iteratively improve parameters to minimize cost functions
4. **Execution** 🚀 - Run circuits on quantum backends
5. **Result Processing** 📊 - Extract and analyze final results

Here's how a typical VQE program flows through this lifecycle:

.. code-block:: python

   from divi.qprog import VQE, HartreeFockAnsatz
   from divi.backends import ParallelSimulator
   from divi.qprog.optimizers import ScipyOptimizer, ScipyMethod

   # 1. Initialization - Define your quantum problem
   vqe = VQE(
       molecule=molecule,           # Your molecular system
       ansatz=HartreeFockAnsatz(),  # Quantum circuit template
       n_layers=2,                  # Circuit depth
       backend=ParallelSimulator(), # Where to run circuits
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),  # Choose optimizer
       seed=42                      # For reproducibility
   )

   # 2. Circuit Generation - Automatically creates circuits
   # (happens internally during run())

   # 3. Optimization - Iteratively improve parameters
   vqe.run()  # This handles steps 2-5 automatically!

   # 4. Execution & 5. Result Processing - All done!
   print(f"Ground state energy: {vqe.best_loss:.6f}")


**Key Features:**

- **Parameter Management** ⚙️ - Automatic initialization and validation of parameters
- **Optimization Loop** 🔄 - Built-in integration with classical optimizers
- **Loss Tracking** 📈 - Detailed history of loss values during optimization
- **Best Result Storage** 💾 - Automatic tracking of the best parameters and loss value found

**Key Properties:**

The most commonly accessed properties for result analysis:

- ``best_loss`` - The best (lowest) loss value found during optimization
- ``best_params`` - The parameters that achieved ``best_loss`` (may differ from final parameters)
- ``final_params`` - The parameters from the last optimization iteration
- ``min_losses_per_iteration`` - Convenience property returning minimum loss per iteration
- ``curr_params`` - Current parameters (can be set to customize initial values)

.. note::
   **Understanding ``best_params`` vs ``final_params``**: During optimization, Divi tracks the best
   loss value found across all iterations. ``best_params`` contains the parameters that achieved
   this best loss, while ``final_params`` contains the parameters from the final iteration.
   These may differ if the optimizer explores away from the best solution.

**Advanced Usage:**

.. code-block:: python

   # Access execution statistics
   print(f"Circuits executed: {vqe.total_circuit_count}")
   print(f"Total runtime: {vqe.total_run_time:.2f}s")

   # Examine optimization history
   for i, best_loss in enumerate(vqe.min_losses_per_iteration):
       print(f"Iteration {i}: {best_loss:.6f}")

   # Get the best parameters found during optimization
   best_params = vqe.best_params

**Warm-Starting and Pre-Training** 🔄
   For warm-starting or pre-training routines where you don't need final solution extraction,
   you can skip the final computation step:

   .. code-block:: python

      import numpy as np
      from divi.qprog.optimizers import MonteCarloOptimizer

      # Run optimization without final probability computation
      vqe.run(perform_final_computation=False)

      # Extract best parameters for reuse
      best_params = vqe.best_params  # Shape: (n_params,)

      # Reuse parameters in a new instance with the same optimizer configuration
      vqe2 = VQE(molecule=molecule, initial_params=best_params.reshape(1, -1))

      # If using a different optimizer, adapt to expected shape
      # For example, MonteCarloOptimizer expects (n_param_sets, n_params)
      optimizer = MonteCarloOptimizer(population_size=10)
      vqe3 = VQE(molecule=molecule, optimizer=optimizer)
      expected_shape = vqe3.get_expected_param_shape()  # (10, n_params)
      # Replicate best_params to match optimizer's n_param_sets
      adapted_params = np.tile(best_params, (expected_shape[0], 1))
      vqe3.curr_params = adapted_params

      # When you need the solution probabilities, run with final computation:
      vqe.run()  # This will perform final computation with best_params

Circuit Architecture
--------------------

Divi uses a two-tier circuit system for maximum efficiency:

**MetaCircuit** 🏗️
   Symbolic circuit templates with parameters that can be instantiated multiple times:

   .. code-block:: python

      from divi.circuits import MetaCircuit
      import pennylane as qml
      import sympy as sp

      # Define symbolic parameters
      params = sp.symarray("theta", 3)

      # Create parameterized circuit
      with qml.tape.QuantumTape() as tape:
          qml.RY(params[0], wires=0)
          qml.RX(params[1], wires=1)
          qml.CNOT(wires=[0, 1])
          qml.RY(params[2], wires=0)
          qml.expval(qml.PauliZ(0))

      # Create reusable template
      meta_circuit = MetaCircuit(tape, params)

      # Generate specific circuits
      circuit1 = meta_circuit.initialize_circuit_from_params([0.1, 0.2, 0.3])
      circuit2 = meta_circuit.initialize_circuit_from_params([0.4, 0.5, 0.6])

**Circuit** ⚡
   Concrete circuit instances with specific parameter values and QASM representations:

   .. code-block:: python

      # Each Circuit contains:
      print(f"Circuit ID: {circuit1.circuit_id}")
      print(f"Tags: {circuit1.tags}")
      print(f"QASM circuits: {len(circuit1.qasm_circuits)}")

      # Access the underlying PennyLane circuit
      pl_circuit = circuit1.main_circuit

Backend Abstraction
-------------------

Divi's backend system provides a unified interface for different execution environments:

**CircuitRunner Interface** 🎯
   All backends implement this common interface:

   .. code-block:: python

      class MyCustomBackend(CircuitRunner):
          def submit_circuits(self, circuits: dict[str, str]) -> Any:
              # Your custom execution logic here
              pass

**Available Backends:**

- **ParallelSimulator** 💻 - Local high-performance simulator
- **QoroService** ☁️ - Cloud quantum computing service

**Backend Selection:**

.. code-block:: python

   # For development and testing
   backend = ParallelSimulator(
       shots=1000,      # Measurement precision
       n_processes=4    # Parallel execution
   )

   # For production and real hardware
   backend = QoroService(
       auth_token="your-api-key",  # From environment or .env
       shots=1000
   )

   # Use the same quantum program with either backend!
   vqe = VQE(molecule=molecule, backend=backend)

Parameter Management
--------------------

Divi handles parameter optimization automatically, but you can also set custom initial parameters. This applies to all variational algorithms (VQE, QAOA, and custom implementations).

**Automatic Initialization** ⚡
   Parameters are randomly initialized between 0 and 2π when not specified:

   .. code-block:: python

      # VQE example
      vqe = VQE(molecule=molecule, n_layers=2)
      print(f"Parameters per layer: {vqe.n_params}")
      print(f"Total parameters: {vqe.n_params * vqe.n_layers}")

      # QAOA example
      from divi.qprog import QAOA, GraphProblem
      import networkx as nx
      qaoa = QAOA(problem=nx.bull_graph(), graph_problem=GraphProblem.MAXCUT, n_layers=2)
      print(f"QAOA parameters: {qaoa.n_params * qaoa.n_layers}")  # Always 2 params per layer

      # Access current parameters (triggers initialization if not set)
      curr_params = vqe.curr_params
      print(f"Shape: {curr_params.shape}")  # (n_param_sets, total_params)

**Setting Initial Parameters** 🎯
   You can set initial parameters in two ways: via the constructor or using the property setter.
   This is useful for warm-starting, pre-training, or using parameters from previous runs:

   .. code-block:: python

      import numpy as np

      # Method 1: Via constructor (recommended for initial setup)
      custom_params = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
      vqe = VQE(molecule=molecule, initial_params=custom_params, seed=42)

      # Method 2: Via property setter (useful for mid-run adjustments)
      qaoa = QAOA(problem=graph, graph_problem=GraphProblem.MAXCUT)
      qaoa.curr_params = np.array([[0.5, 0.3]])  # QAOA: (beta, gamma) per layer

      # Both methods work for any variational algorithm
      vqe.curr_params = custom_params  # Can also set after initialization

   .. note::
      Use the ``seed`` parameter in the constructor for reproducible parameter initialization
      when not providing custom initial parameters.

**Parameter Shape Requirements** 📐
   Parameters must match the expected shape based on your algorithm configuration.
   Use :meth:`get_expected_param_shape()` to validate shapes before setting parameters:

   .. code-block:: python

      # Verify the expected shape before setting parameters
      expected_shape = vqe.get_expected_param_shape()
      print(f"Expected shape: {expected_shape}")  # (n_param_sets, n_layers * n_params)

      # For VQE with 2 layers and 4 params per layer:
      # Shape: (n_param_sets, 8)
      # For QAOA with 3 layers (always 2 params per layer):
      # Shape: (n_param_sets, 6)

      # Use this to validate your parameters before setting them
      custom_params = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
      if custom_params.shape == expected_shape:
          vqe.curr_params = custom_params
      else:
          print(f"Shape mismatch! Expected {expected_shape}, got {custom_params.shape}")

**Parameter Validation** ✅
   Divi validates parameter shapes automatically and provides clear error messages:

   .. code-block:: python

      try:
          vqe.curr_params = np.array([[1, 2, 3]])  # Wrong shape
      except ValueError as e:
          print(f"Validation error: {e}")
          # "Initial parameters must have shape (1, 8), got (1, 3)"

**Multiple Parameter Sets** 🔄
   Some optimizers (like MonteCarloOptimizer) work with multiple parameter sets simultaneously.
   The first dimension represents the number of parameter sets. Always use :meth:`get_expected_param_shape()`
   to verify the correct shape before setting custom parameters:

   .. code-block:: python

      from divi.qprog.optimizers import MonteCarloOptimizer

      # Monte Carlo optimizer with 10 parameter sets
      optimizer = MonteCarloOptimizer(population_size=10)
      vqe = VQE(molecule=molecule, optimizer=optimizer, n_layers=2)

      # Get the expected shape (includes n_param_sets from optimizer)
      expected_shape = vqe.get_expected_param_shape()
      print(f"Expected shape: {expected_shape}")  # (10, 8) for 10 sets, 8 params

      # Parameters must match this shape exactly
      # Note: get_expected_param_shape() automatically uses optimizer.n_param_sets,
      # so you don't need to manually look up the optimizer's population size
      custom_params = np.random.uniform(0, 2*np.pi, expected_shape)
      vqe.curr_params = custom_params

Result Processing
-----------------

After execution, Divi provides rich result analysis capabilities:

**Loss History** 📈
   Track optimization progress over time. The ``losses_history`` property stores a list of
   dictionaries, where each dictionary maps parameter set indices to their loss values for that iteration.
   Use ``min_losses_per_iteration`` for a simplified view:

   .. code-block:: python

      # Plot convergence
      import matplotlib.pyplot as plt

      losses = vqe.min_losses_per_iteration
      plt.plot(losses)
      plt.xlabel('Iteration')
      plt.ylabel('Energy (Hartree)')
      plt.title('VQE Convergence')
      plt.show()

      # Access detailed loss history (useful for multi-parameter-set optimizers)
      # losses_history is a list[dict[int, float]] where each dict maps param_set_idx -> loss
      for iteration, loss_dict in enumerate(vqe.losses_history):
          print(f"Iteration {iteration}: {loss_dict}")

**Solution Probabilities** 🎲
   After optimization completes, access probability distributions for the best solution.
   For VQE and QAOA, the ``best_probs`` property contains a dictionary mapping bitstrings to their measurement
   probabilities (essentially a shots histogram from the final measurement). If you implement a custom
   variational algorithm, you are free to adjust this structure to suit your needs:

   .. code-block:: python

      # Get probability distribution for best parameters
      probs = vqe.best_probs
      if probs:  # Check if probabilities were computed
          most_likely_bitstring = max(probs, key=probs.get)
          probability = probs[most_likely_bitstring]
          print(f"Most likely solution: {most_likely_bitstring}")
          print(f"Probability: {probability:.4f}")

**Performance Metrics** ⚡
   Monitor execution efficiency:

   .. code-block:: python

      print(f"Total circuits: {vqe.total_circuit_count}")
      print(f"Total runtime: {vqe.total_run_time:.2f}s")
      print(f"Average time per circuit: {vqe.total_run_time / vqe.total_circuit_count:.3f}s")

Next Steps
----------

- 📖 **Algorithms**: Learn about specific algorithms in :doc:`vqe` and :doc:`qaoa`
- ⚡ **Backends**: Explore execution options in :doc:`backends`
- 🛠️ **Customization**: Create custom algorithms using the :doc:`../api_reference/qprog`
- 💡 **Examples**: See practical applications in the Tutorials section

Understanding these core concepts will help you leverage Divi's full power for your quantum computing projects!
