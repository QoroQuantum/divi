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

   # 1. Initialization - Define your quantum problem
   vqe = VQE(
       molecule=molecule,           # Your molecular system
       ansatz=HartreeFockAnsatz(),  # Quantum circuit template
       n_layers=2,                  # Circuit depth
       backend=ParallelSimulator()  # Where to run circuits
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

- ``losses_history`` - History of loss values from each optimization iteration
- ``initial_params`` - The starting parameters for the optimization. These allow you to set custom initial parameters for the optimization.
- ``final_params`` - The last set of parameters from the optimization
- ``best_params`` - The parameters that achieved the lowest loss
- ``best_loss`` - The best loss value recorded during the run

**Advanced Usage:**

.. code-block:: python

   # Access execution statistics
   print(f"Circuits executed: {vqe.total_circuit_count}")
   print(f"Total runtime: {vqe.total_run_time:.2f}s")

   # Examine optimization history
   for i, loss_dict in enumerate(vqe.losses_history):
       best_loss_in_iter = min(loss_dict.values())
       print(f"Iteration {i}: {best_loss_in_iter:.6f}")

   # Get the best parameters found during optimization
   best_params = vqe.best_params

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

Divi handles parameter optimization automatically, but you can also set custom initial parameters:

**Automatic Initialization** ⚡
   Parameters are randomly initialized between 0 and 2π:

   .. code-block:: python

      vqe = VQE(molecule=molecule, n_layers=2)
      print(f"Parameters per layer: {vqe.n_params}")
      print(f"Total parameters: {vqe.n_params * vqe.n_layers}")

      # Access current parameters
      initial_params = vqe.initial_params
      print(f"Shape: {initial_params.shape}")  # (n_sets, total_params)

**Custom Initial Parameters** 🎯
   Set specific starting points for better convergence:

   .. code-block:: python

      import numpy as np

      # Set custom initial parameters
      custom_params = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
      vqe.initial_params = custom_params

      # Verify the shape matches expectations
      expected_shape = vqe.get_expected_param_shape()
      print(f"Expected shape: {expected_shape}")

**Parameter Validation** ✅
   Divi validates parameter shapes automatically:

   .. code-block:: python

      try:
          vqe.initial_params = np.array([[1, 2, 3]])  # Wrong shape
      except ValueError as e:
          print(f"Validation error: {e}")
          # "Initial parameters must have shape (1, 6), got (1, 3)"

Result Processing
-----------------

After execution, Divi provides rich result analysis capabilities:

**Loss History** 📈
   Track optimization progress over time:

   .. code-block:: python

      # Plot convergence
      import matplotlib.pyplot as plt

      losses = [min(loss_dict.values()) for loss_dict in vqe.losses_history]
      plt.plot(losses)
      plt.xlabel('Iteration')
      plt.ylabel('Energy (Hartree)')
      plt.title('VQE Convergence')
      plt.show()

**Circuit Analysis** 🔍
   Examine which circuits were executed:

   .. code-block:: python

      circuits = vqe.circuits
      print(f"Total circuits: {len(circuits)}")

      for circuit in circuits[:3]:  # Show first 3
          print(f"Circuit {circuit.circuit_id}: {circuit.tags}")
          print(f"QASM length: {len(circuit.qasm_circuits[0])} characters")

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
