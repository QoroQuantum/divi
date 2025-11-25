Checkpointing
=============

Checkpointing allows you to save the state of your quantum algorithm optimization at any point and resume from that exact state later. This is essential for long-running optimizations, debugging, and resuming interrupted computations.

Overview
--------

Divi's checkpointing system saves both the program state (parameters, losses, iteration count) and optimizer state (internal optimizer data) to disk. This allows you to:

- **Resume interrupted runs** - Continue optimization from where it stopped
- **Debug optimization** - Inspect intermediate states and parameters
- **Manage long runs** - Break up very long optimizations into manageable chunks
- **Adjust iteration targets** - Change `max_iterations` after loading to continue beyond the original target

Checkpointing is supported for all :class:`VariationalQuantumAlgorithm` subclasses (VQE, QAOA) and works with checkpointing-capable optimizers:

- :class:`MonteCarloOptimizer`
- :class:`PymooOptimizer` (CMAES and DE methods)

.. note::
   :class:`ScipyOptimizer` does not support checkpointing due to limitations in the underlying scipy optimization methods.

Basic Usage
-----------

Saving Checkpoints
^^^^^^^^^^^^^^^^^^

To enable checkpointing, pass a :class:`CheckpointConfig` object to the ``run()`` method:

.. code-block:: python

   from pathlib import Path
   from divi.qprog import VQE, HartreeFockAnsatz
   from divi.qprog.checkpointing import CheckpointConfig
   from divi.backends import ParallelSimulator
   import pennylane as qml

   # Create a molecule
   mol = qml.qchem.Molecule(
       symbols=["H", "H"],
       coordinates=np.array([[0.0, 0.0, -0.6614], [0.0, 0.0, 0.6614]])
   )

   # Create VQE program
   vqe = VQE(
       molecule=mol,
       ansatz=HartreeFockAnsatz(),
       n_layers=1,
       max_iterations=10,
       backend=ParallelSimulator(),
   )

   # Run with checkpointing enabled
   checkpoint_dir = Path("my_checkpoints")
   vqe.run(checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir))

By default, checkpoints are saved **every iteration**. Each checkpoint is stored in a subdirectory named ``checkpoint_{iteration:03d}`` (e.g., ``checkpoint_001``, ``checkpoint_002``).

Checkpoint Interval
^^^^^^^^^^^^^^^^^^^

To save checkpoints less frequently, set the ``checkpoint_interval`` parameter:

.. code-block:: python

   # Save checkpoint every 5 iterations
   vqe.run(
       checkpoint_config=CheckpointConfig(
           checkpoint_dir=checkpoint_dir,
           checkpoint_interval=5
       )
   )

Auto-Generated Checkpoint Directories
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can automatically generate a timestamped checkpoint directory:

.. code-block:: python

   # Creates a directory like "checkpoint_20250115_143022"
   config = CheckpointConfig.with_timestamped_dir()
   vqe.run(checkpoint_config=config)

   # Or with a checkpoint interval
   config = CheckpointConfig.with_timestamped_dir(checkpoint_interval=5)
   vqe.run(checkpoint_config=config)

Loading and Resuming
--------------------

To resume from a checkpoint, use the ``load_state()`` class method:

.. code-block:: python

   from divi.qprog import VQE

   # Load the latest checkpoint
   vqe_resumed = VQE.load_state(
       checkpoint_dir="my_checkpoints",
       backend=ParallelSimulator(),
       molecule=mol,  # Must provide original problem configuration
       ansatz=HartreeFockAnsatz(),
       n_layers=1,
   )

   # Continue optimization
   vqe_resumed.max_iterations = 20  # Set new target
   vqe_resumed.run()

**Important**: When loading from a checkpoint, you must provide all the original constructor arguments (problem definition, ansatz, etc.) because checkpoints only store **runtime state**, not the problem configuration.

Loading Specific Checkpoints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, ``load_state()`` loads the latest checkpoint. To load a specific checkpoint:

.. code-block:: python

   # Load checkpoint from iteration 5
   vqe_resumed = VQE.load_state(
       checkpoint_dir="my_checkpoints",
       backend=ParallelSimulator(),
       subdirectory="checkpoint_005",  # Specific checkpoint subdirectory
       molecule=mol,
       ansatz=HartreeFockAnsatz(),
       n_layers=1,
   )

Complete Example: QAOA with Checkpointing
------------------------------------------

Here's a complete example showing checkpointing with QAOA:

.. code-block:: python

   import networkx as nx
   from pathlib import Path
   from divi.qprog import QAOA, GraphProblem
   from divi.qprog.checkpointing import CheckpointConfig
   from divi.qprog.optimizers import PymooOptimizer, PymooMethod
   from divi.backends import ParallelSimulator

   # Create problem
   G = nx.bull_graph()
   checkpoint_dir = Path("qaoa_checkpoints")

   # Initial run - first half
   qaoa1 = QAOA(
       problem=G,
       graph_problem=GraphProblem.MAX_CLIQUE,
       n_layers=1,
       optimizer=PymooOptimizer(method=PymooMethod.CMAES, population_size=10),
       max_iterations=5,
       backend=ParallelSimulator(),
   )

   # Run with checkpointing
   qaoa1.run(checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir))

   # Later: Resume from checkpoint
   qaoa2 = QAOA.load_state(
       checkpoint_dir=checkpoint_dir,
       backend=ParallelSimulator(),
       problem=G,  # Must provide original problem
       graph_problem=GraphProblem.MAX_CLIQUE,
       n_layers=1,
   )

   # Continue optimization
   qaoa2.max_iterations = 10
   qaoa2.run()

   # Access results
   print(f"Best loss: {qaoa2.best_loss}")
   print(f"Solution: {qaoa2.solution}")

Managing Checkpoints
--------------------

Listing Checkpoints
^^^^^^^^^^^^^^^^^^^

You can list all checkpoints in a directory:

.. code-block:: python

   from divi.qprog.checkpointing import list_checkpoints

   checkpoints = list_checkpoints(Path("my_checkpoints"))
   for checkpoint in checkpoints:
       print(f"Iteration {checkpoint.iteration}: {checkpoint.path}")
       print(f"  Size: {checkpoint.size_bytes / 1024:.2f} KB")
       print(f"  Valid: {checkpoint.is_valid}")

Getting Checkpoint Information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get detailed information about a specific checkpoint:

.. code-block:: python

   from divi.qprog.checkpointing import get_checkpoint_info

   info = get_checkpoint_info(Path("my_checkpoints/checkpoint_005"))
   print(f"Iteration: {info.iteration}")
   print(f"Timestamp: {info.timestamp}")
   print(f"Size: {info.size_bytes} bytes")
   print(f"Valid: {info.is_valid}")

Finding the Latest Checkpoint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Get the path to the latest checkpoint:

.. code-block:: python

   from divi.qprog.checkpointing import get_latest_checkpoint

   latest = get_latest_checkpoint(Path("my_checkpoints"))
   if latest:
       print(f"Latest checkpoint: {latest}")

Cleaning Up Old Checkpoints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Remove old checkpoints, keeping only the most recent N:

.. code-block:: python

   from divi.qprog.checkpointing import cleanup_old_checkpoints

   # Keep only the 5 most recent checkpoints
   cleanup_old_checkpoints(Path("my_checkpoints"), keep_last_n=5)

Checkpoint Structure
--------------------

Each checkpoint is stored in a subdirectory with the following structure:

.. code-block:: text

   checkpoint_dir/
   ├── checkpoint_001/
   │   ├── program_state.json    # Program state (parameters, losses, etc.)
   │   └── optimizer_state.json  # Optimizer internal state
   ├── checkpoint_002/
   │   ├── program_state.json
   │   └── optimizer_state.json
   └── ...

The ``program_state.json`` file contains:

- Current iteration number
- Loss history
- Best parameters found so far
- Current parameters
- Random number generator state
- Algorithm-specific state (e.g., eigenstate for VQE, solution nodes for QAOA)

The ``optimizer_state.json`` file contains optimizer-specific data:

- For ``MonteCarloOptimizer``: Population, evaluated population, losses, RNG state
- For ``PymooOptimizer``: Serialized algorithm object and population

Best Practices
--------------

1. **Use meaningful checkpoint directory names** - Include experiment identifiers or timestamps
2. **Set appropriate checkpoint intervals** - For long runs, checkpoint every N iterations to save disk space
3. **Always provide problem configuration when loading** - Checkpoints don't store problem definitions
4. **Clean up old checkpoints** - Use ``cleanup_old_checkpoints()`` to manage disk space
5. **Verify checkpoint validity** - Check ``is_valid`` before resuming from a checkpoint
6. **Use auto-generated directories** - ``CheckpointConfig.with_timestamped_dir()`` prevents accidental overwrites

Error Handling
--------------

Checkpointing operations can raise several exceptions:

- :class:`CheckpointNotFoundError` - Checkpoint directory or file not found
- :class:`CheckpointCorruptedError` - Checkpoint file is invalid or corrupted
- :class:`RuntimeError` - Attempting to save checkpoint before any iterations complete
- :class:`ValueError` - Invalid checkpoint configuration

Always handle these exceptions appropriately:

.. code-block:: python

   from divi.qprog.checkpointing import (
       CheckpointNotFoundError,
       CheckpointCorruptedError,
   )

   try:
       vqe = VQE.load_state(checkpoint_dir, backend=backend, molecule=mol)
   except CheckpointNotFoundError as e:
       print(f"Checkpoint not found: {e}")
   except CheckpointCorruptedError as e:
       print(f"Checkpoint corrupted: {e}")

Limitations
-----------

- **ScipyOptimizer** does not support checkpointing
- Checkpoints are **not portable** across different Python versions or library versions
- Problem configuration must be **manually provided** when loading (not stored in checkpoint)
- Checkpoint files can be **large** for population-based optimizers (MonteCarlo, Pymoo)
