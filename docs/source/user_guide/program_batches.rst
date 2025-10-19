Program Batches and Workflows
===============================

Divi's program batch capabilities enable efficient execution of large-scale quantum computations across multiple problems, parameter sweeps, and optimization scenarios. This guide explains how to leverage Divi's parallelization features for maximum performance.

Program Batch Overview
-------------------------

**What are Program Batches?** 📦

Program batches allow you to run multiple quantum programs simultaneously, automatically distributing work across available computational resources. This is essential for:

- **Problem Decomposition** 🧩 - Breaking down large problems into smaller, solvable parts
- **Parameter Sweeps** 🔄 - Testing multiple parameter combinations
- **Molecular Studies** ⚗️ - Computing dissociation curves and reaction pathways
- **Algorithm Comparison** ⚖️ - Comparing different ansätze and optimizers
- **Large Problem Sets** 📊 - Solving multiple optimization instances

**Key Benefits:**

- **Automatic Parallelization** ⚡ - Utilizes all available CPU cores and quantum backends
- **Resource Optimization** 💰 - Maximizes hardware utilization and minimizes costs
- **Scalable Execution** 📈 - Handles problems from dozens to thousands of circuits
- **Progress Monitoring** 👀 - Real-time tracking of batch execution progress

VQE Hyperparameter Sweeps
-------------------------

The most common program batch scenario is running VQE across multiple molecular configurations:

**Basic Parameter Sweep** 🎯

.. code-block:: python

   import numpy as np
   import pennylane as qml
   from divi.qprog import VQEHyperparameterSweep, MoleculeTransformer, HartreeFockAnsatz
   from divi.qprog.optimizers import MonteCarloOptimizer
   from divi.backends import ParallelSimulator

   # Define your base molecule
   base_molecule = qml.qchem.Molecule(
       symbols=["H", "H"],
       coordinates=np.array([[0.0, 0.0, -0.6614], [0.0, 0.0, 0.6614]])
   )

   # Set up molecule transformer for bond length variations
   transformer = MoleculeTransformer(
       base_molecule=base_molecule,
       bond_modifiers=[-0.4, -0.2, 0.0, 0.2, 0.4]  # Bond length changes in Å
   )

   # Configure Monte Carlo optimizer for robustness
   mc_optimizer = MonteCarloOptimizer(
       n_param_sets=10,    # Try 10 parameter combinations per molecule
       n_best_sets=3       # Keep top 3 for next iteration
   )

   # Create batch VQE sweep
   vqe_sweep = VQEHyperparameterSweep(
       molecule_transformer=transformer,
       ansatze=[HartreeFockAnsatz()],  # Single ansatz for simplicity
       optimizer=mc_optimizer,
       max_iterations=25,
       backend=ParallelSimulator(shots=2000, n_processes=4)
   )

   # Execute the entire sweep
   vqe_sweep.create_programs()  # Generate all VQE instances
   vqe_sweep.run(blocking=True)  # Execute all programs in parallel

   # Analyze results
   results = vqe_sweep.aggregate_results()
   vqe_sweep.visualize_results()  # Show energy vs bond length plot

   print(f"Total circuits executed: {vqe_sweep.total_circuit_count}")

**Advanced Sweep Configuration** ⚙️

.. code-block:: python

   import numpy as np
   import pennylane as qml
   from divi.qprog import VQEHyperparameterSweep, MoleculeTransformer, HartreeFockAnsatz, UCCSDAnsatz, GenericLayerAnsatz
   from divi.qprog.optimizers import MonteCarloOptimizer
   from divi.backends import ParallelSimulator
   # Multiple ansätze comparison
   vqe_sweep = VQEHyperparameterSweep(
       molecule_transformer=transformer,
       ansatze=[
           HartreeFockAnsatz(),
           UCCSDAnsatz(),
           GenericLayerAnsatz([qml.RY, qml.RZ], entangling_layout="circular")
       ],
       optimizer=MonteCarloOptimizer(n_param_sets=5),
       max_iterations=50,
       backend=ParallelSimulator(shots=5000)
   )

   # Custom molecule transformations
   water_molecule = qml.qchem.Molecule(
       symbols=["O", "H", "H"],
       coordinates=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
   )
   transformer = MoleculeTransformer(
       base_molecule=water_molecule,
       atom_connectivity=[(0, 1), (0, 2)],  # Define molecular structure
       bonds_to_transform=[(0, 1)],          # Only modify O-H bonds
       bond_modifiers=[-0.1, 0.0, 0.1],     # Small perturbations
       alignment_atoms=[0]                   # Align to oxygen atom
   )

Problem Decomposition Workflows
-------------------------------

For problems that are too large to fit on a single quantum device, Divi provides workflows that automatically decompose the problem into smaller subproblems, solve them in parallel, and then combine the results into a final solution. This approach allows you to tackle large-scale optimization challenges that would otherwise be intractable.

Divi offers built-in support for two common decomposition scenarios:

- **Graph Partitioning**: For large graph problems like MaxCut or Minimum Vertex Cover.
- **QUBO Partitioning**: For large-scale QUBO (Quadratic Unconstrained Binary Optimization) problems.

Graph Partitioning QAOA
^^^^^^^^^^^^^^^^^^^^^^^

For large optimization problems that exceed quantum hardware limitations, Divi provides automatic graph partitioning:

**Basic Graph Partitioning** 🗺️

.. code-block:: python

   import networkx as nx
   from divi.qprog import GraphPartitioningQAOA, GraphProblem, PartitioningConfig
   from divi.qprog.optimizers import ScipyMethod, ScipyOptimizer

   # Create a large graph (too big for single quantum device)
   large_graph = nx.erdos_renyi_graph(50, 0.3)  # 50 nodes

   # Configure partitioning strategy
   config = PartitioningConfig(
       max_n_nodes_per_cluster=10,      # Maximum nodes per quantum partition
       minimum_n_clusters=3,             # Minimum partitions (optional)
       partitioning_algorithm="metis"    # Algorithm: "spectral", "metis", or "kernighan_lin"
   )

   # Create partitioned QAOA solver
   qaoa_partition = GraphPartitioningQAOA(
       graph_problem=GraphProblem.MAXCUT,
       graph=large_graph,
       n_layers=3,
       partitioning_config=config,
       optimizer=ScipyOptimizer(method=ScipyMethod.NELDER_MEAD),
       max_iterations=20,
       backend=ParallelSimulator()
   )

   # Execute workflow
   qaoa_partition.create_programs()    # Partition graph and create sub-problems
   qaoa_partition.run(blocking=True)   # Solve each partition in parallel

   # Combine results from all partitions
   final_solution, final_energy = qaoa_partition.aggregate_results()

   print(f"Final MaxCut value: {final_energy}")
   print(f"Total circuits: {qaoa_partition.total_circuit_count}")

**Partitioning Strategies** 🎲

Different partitioning algorithms for different graph structures:

.. code-block:: python

   # For regular graphs (grids, lattices)
   config_regular = PartitioningConfig(
       max_n_nodes_per_cluster=16,
       partitioning_algorithm="spectral",  # Good for regular structures
       minimum_n_clusters=None
   )

   # For irregular graphs (social networks, molecules)
   config_irregular = PartitioningConfig(
       max_n_nodes_per_cluster=12,
       partitioning_algorithm="metis",     # Excellent for irregular graphs
       minimum_n_clusters=4
   )

   # For very large graphs with community structure
   config_communities = PartitioningConfig(
       max_n_nodes_per_cluster=20,
       partitioning_algorithm="kernighan_lin",  # Preserves community structure
       minimum_n_clusters=None
   )

QUBO Partitioning
^^^^^^^^^^^^^^^^^

For large QUBO problems, Divi integrates with D-Wave's hybrid solvers:

**Large QUBO Problems** 📊

.. code-block:: python

   import dimod
   import hybrid
   from divi.qprog import QUBOPartitioningQAOA

   # Create large QUBO problem
   large_bqm = dimod.generators.gnp_random_bqm(
       n_variables=100,     # 100 binary variables
       n_interactions=0.3,  # 30% connectivity
       vartype="BINARY"
   )

   # Set up hybrid decomposition
   qubo_partition = QUBOPartitioningQAOA(
       qubo=large_bqm,
       decomposer=hybrid.EnergyImpactDecomposer(size=15),  # Decompose into size-15 chunks
       composer=hybrid.SplatComposer(),                    # Recombine solutions
       n_layers=3,
       optimizer=ScipyOptimizer(method=ScipyMethod.COBYLA),
       max_iterations=15,
       backend=ParallelSimulator()
   )

   # Execute partitioned computation
   qubo_partition.create_programs()
   qubo_partition.run()

   # Get final solution
   solution, energy = qubo_partition.aggregate_results()
   print(f"Final energy: {energy:.6f}")

Custom Batch Workflows
----------------------

You can create custom program batch workflows by inheriting from :class:`~divi.qprog.ProgramBatch`:

**Custom Batch Implementation** 🛠️

.. code-block:: python

   from divi.qprog import ProgramBatch, VQE
   from divi.backends import CircuitRunner, ParallelSimulator
   import pennylane as qml
   import numpy as np

   class CustomParameterSweep(ProgramBatch):
       def __init__(self, backend: CircuitRunner, molecules, parameters):
           super().__init__(backend)
           self.molecules = molecules
           self.parameters = parameters

       def create_programs(self):
           """Generate VQE programs for all molecule-parameter combinations"""
           super().create_programs()
           for i, (mol, params) in enumerate(zip(self.molecules, self.parameters)):
               vqe = VQE(
                   molecule=mol,
                   initial_params=params,
                   backend=self.backend
               )
               self._programs[f"sweep_{i}"] = vqe

       def aggregate_results(self):
           """Collect and analyze results from all programs"""
           super().aggregate_results()
           results = {}
           for program_id, program in self._programs.items():
               if program.losses_history:  # Check if program completed
                   final_loss = program.best_loss
                   results[program_id] = {
                       'energy': final_loss,
                       'params': program.best_params,
                       'circuits': program.total_circuit_count
                   }
           return results

   # Usage
   mol1 = qml.qchem.Molecule(symbols=["H", "H"], coordinates=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]))
   mol2 = qml.qchem.Molecule(symbols=["Li", "H"], coordinates=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]]))
   mol3 = qml.qchem.Molecule(symbols=["H", "F"], coordinates=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.92]]))
   molecules = [mol1, mol2, mol3]
   params1 = np.random.rand(4)
   params2 = np.random.rand(8)
   params3 = np.random.rand(12)
   parameters = [params1, params2, params3]

   # Use a local simulator
   local_backend = ParallelSimulator(n_processes=8)
   sweep = CustomParameterSweep(local_backend, molecules, parameters)
   sweep.create_programs()
   sweep.run(blocking=True)

   results = sweep.aggregate_results()
   print(results)

Parallel Execution Strategies
-----------------------------

Divi automatically optimizes parallel execution based on your backend and problem structure:

**Local Parallelization** 💻

.. code-block:: python

   import os
   from divi.backends import ParallelSimulator
   # Optimize for local execution
   backend = ParallelSimulator(
       n_processes=min(8, os.cpu_count()),  # Use available cores
       shots=1000,                          # Balance speed vs accuracy
       simulation_seed=42                   # Reproducible results
   )

   # For memory-intensive problems
   backend = ParallelSimulator(
       n_processes=2,        # Fewer processes to reduce memory usage
       shots=10000,         # More shots for better statistics
       qiskit_backend="statevector_simulator"  # Memory efficient
   )

**Cloud Parallelization** ☁️

.. code-block:: python

   from divi.backends import QoroService

   # Configure for cloud execution
   service = QoroService(
       polling_interval=5.0,     # Check job status every 5 seconds
       max_retries=1000,         # Allow long-running jobs
       use_circuit_packing=True  # Optimize circuit submission
   )

   # Submit large batches efficiently
   from divi.circuits import Circuit
   circuits = {"circ1": Circuit(), "circ2": Circuit()}
   if len(circuits) > 50:
       # Split into smaller batches for better queue management
       batch_size = 20
       for i in range(0, len(circuits), batch_size):
           batch = dict(list(circuits.items())[i:i+batch_size])
           job_id = service.submit_circuits(batch)

**Hybrid Execution** 🔄

Combine local and cloud execution for optimal performance:

.. code-block:: python

   # Use local simulator for development and small problems
   problem_size = 50
   molecule = qml.qchem.Molecule(symbols=["H", "H"], coordinates=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]))
   if problem_size < 100:
       backend = ParallelSimulator(n_processes=4)
   else:
       # Use cloud for large problems
       backend = QoroService()

   # Single interface works with both backends!
   vqe = VQE(molecule=molecule, backend=backend)
   vqe.run()

Progress Monitoring and Control
-------------------------------

Divi provides automatic progress tracking for long-running batches. When you execute a batch that contains compatible programs (like VQE or QAOA), a progress bar will be displayed in your console, showing the status of each program in real-time.

**Stopping a Batch** 🛑

You can gracefully stop a running batch at any time by pressing ``Ctrl+C``. Divi will catch the signal, attempt to cancel any pending programs, and allow any currently running programs to finish their current iteration before shutting down.

Performance Optimization
------------------------

**Memory Management** 🧠

.. code-block:: python

   # For memory-constrained systems
   backend = ParallelSimulator(
       n_processes=1,    # Single process to minimize memory
       shots=1000        # Reduce shots to save memory
   )

   # Process large batches in chunks
   large_problem_set = [
        qml.qchem.Molecule(symbols=["H", "H"], coordinates=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, d]]))
        for d in np.arange(0.5, 2.0, 0.1)
   ]
   def process_chunk(chunk):
        # Dummy processing function
        for mol in chunk:
            print(f"Processing molecule with bond length: {np.linalg.norm(mol.coordinates[0] - mol.coordinates[1]):.2f}")

   batch_size = 10
   for i in range(0, len(large_problem_set), batch_size):
       chunk = large_problem_set[i:i+batch_size]
       process_chunk(chunk)

**Execution Time Optimization** ⚡

.. code-block:: python

   import os
   from divi.backends import ParallelSimulator, QoroService
   # Balance speed vs accuracy
   backend = ParallelSimulator(
       n_processes=max(1, os.cpu_count() // 2),  # Use half available cores
       shots=2000,                               # Good balance of speed/accuracy
       qiskit_backend="qasm_simulator"          # Fastest simulator
   )

   # For cloud execution, optimize batch sizes
   service = QoroService(
       use_circuit_packing=True,    # Optimize circuit submission
       polling_interval=2.0         # Faster status checks
   )

Next Steps
----------

- ⚡ **Backend Optimization**: Learn about performance tuning in :doc:`backends`.
- 🛠️ **Custom Workflows**: Create your own batch processors using :class:`~divi.qprog.ProgramBatch`.
- 📊 **Result Analysis**: Learn about advanced result visualization, e.g., using :meth:`~divi.qprog.workflows.VQEHyperparameterSweep.visualize_results`.

Batch processing is where Divi's true power shines - enabling quantum computations that would be impractical with traditional approaches!
