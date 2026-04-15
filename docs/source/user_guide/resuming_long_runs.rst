Resuming Long-Running or Interrupted Runs
=========================================

Checkpointing saves :class:`~divi.qprog.variational_quantum_algorithm.VariationalQuantumAlgorithm` run state to disk via :class:`~divi.qprog.checkpointing.CheckpointConfig` so you can resume after interruptions, inspect intermediate progress, or raise ``max_iterations`` after a reload.

Overview
--------

On each checkpoint, Divi writes **program** state (parameters, losses, iteration count, RNG state) and **optimizer** state (anything the optimizer needs to continue). That enables you to:

- **Resume interrupted runs** — continue from the last saved iteration
- **Debug** — inspect intermediate parameters and losses on disk
- **Chunk long jobs** — stop and restart without re-running from scratch
- **Raise iteration caps** — increase ``max_iterations`` after :meth:`~divi.qprog.variational_quantum_algorithm.VariationalQuantumAlgorithm.load_state`

Checkpointing is supported for all :class:`~divi.qprog.variational_quantum_algorithm.VariationalQuantumAlgorithm` subclasses (:class:`~divi.qprog.algorithms.VQE`, :class:`~divi.qprog.algorithms.QAOA`) and works with checkpointing-capable optimizers:

- :class:`~divi.qprog.optimizers.MonteCarloOptimizer`
- :class:`~divi.qprog.optimizers.PymooOptimizer` (CMAES and DE methods)

.. note::
   :class:`~divi.qprog.optimizers.ScipyOptimizer` does not support checkpointing due to limitations in the underlying scipy optimization methods.

Basic Usage
-----------

Saving Checkpoints
^^^^^^^^^^^^^^^^^^

To enable checkpointing, pass a :class:`~divi.qprog.checkpointing.CheckpointConfig` object to the ``run()`` method:

.. code-block:: python

   import numpy as np
   from pathlib import Path
   from divi.qprog import VQE, HartreeFockAnsatz
   from divi.qprog.checkpointing import CheckpointConfig
   from divi.backends import MaestroSimulator
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
       n_layers=2,
       max_iterations=10,
       backend=MaestroSimulator(),
   )

   # Run with checkpointing enabled
   checkpoint_dir = Path("my_checkpoints")
   vqe.run(checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir))

By default, checkpoints are saved **every iteration**. Each checkpoint is stored in a subdirectory named ``checkpoint_{iteration:03d}`` (e.g., ``checkpoint_001``, ``checkpoint_002``).

Checkpoint Interval
^^^^^^^^^^^^^^^^^^^

To save checkpoints less frequently, set the ``checkpoint_interval`` parameter:

.. invisible-code-block: python

   vqe = VQE(molecule=mol, ansatz=HartreeFockAnsatz(), n_layers=2,
             max_iterations=10, backend=MaestroSimulator())

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

.. invisible-code-block: python

   vqe = VQE(molecule=mol, ansatz=HartreeFockAnsatz(), n_layers=2,
             max_iterations=10, backend=MaestroSimulator())

.. code-block:: python

   # Creates a directory like "checkpoint_20250115_143022"
   config = CheckpointConfig.with_timestamped_dir()
   vqe.run(checkpoint_config=config)

Or with a checkpoint interval:

.. invisible-code-block: python

   vqe = VQE(molecule=mol, ansatz=HartreeFockAnsatz(), n_layers=2,
             max_iterations=10, backend=MaestroSimulator())

.. code-block:: python

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
       backend=MaestroSimulator(),
       molecule=mol,  # Must provide original problem configuration
       ansatz=HartreeFockAnsatz(),
       n_layers=2,
   )

   # Continue optimization
   vqe_resumed.max_iterations = 20  # Set new target
   vqe_resumed.run()

**Important**: When loading from a checkpoint, you must provide all the original constructor arguments (problem definition, ansatz, etc.) because checkpoints only store **runtime state**, not the problem configuration.

Loading Specific Checkpoints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, ``load_state()`` loads the latest checkpoint. To load a specific checkpoint:

.. skip: next

.. code-block:: python

   # Load checkpoint from iteration 5
   vqe_resumed = VQE.load_state(
       checkpoint_dir="my_checkpoints",
       backend=MaestroSimulator(),
       subdirectory="checkpoint_005",  # Specific checkpoint subdirectory
       molecule=mol,
       ansatz=HartreeFockAnsatz(),
       n_layers=2,
   )

Complete Example: :class:`~divi.qprog.algorithms.QAOA` with Checkpointing
--------------------------------------------------------------------------

Here's a complete example showing checkpointing with :class:`~divi.qprog.algorithms.QAOA`:

.. code-block:: python

   import networkx as nx
   from pathlib import Path
   from divi.qprog import QAOA
   from divi.qprog.problems import MaxCliqueProblem
   from divi.qprog.checkpointing import CheckpointConfig
   from divi.qprog.optimizers import PymooOptimizer, PymooMethod
   from divi.backends import MaestroSimulator

   # Create problem
   G = nx.bull_graph()
   checkpoint_dir = Path("qaoa_checkpoints")

   # Initial run - first half
   qaoa1 = QAOA(
       MaxCliqueProblem(G),
       n_layers=2,
       optimizer=PymooOptimizer(method=PymooMethod.CMAES, population_size=10),
       max_iterations=10,
       backend=MaestroSimulator(),
   )

   # Run with checkpointing
   qaoa1.run(checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir))

   # Later: Resume from checkpoint
   qaoa2 = QAOA.load_state(
       checkpoint_dir=checkpoint_dir,
       backend=MaestroSimulator(),
       problem=MaxCliqueProblem(G),  # Must provide original problem
       n_layers=2,
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

.. skip: next

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

.. skip: next

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

.. skip: next

.. code-block:: python

   from divi.qprog.checkpointing import get_latest_checkpoint

   latest = get_latest_checkpoint(Path("my_checkpoints"))
   if latest:
       print(f"Latest checkpoint: {latest}")

Cleaning Up Old Checkpoints
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Remove old checkpoints, keeping only the most recent N:

.. skip: next

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
- Algorithm-specific state (e.g., eigenstate for :class:`~divi.qprog.algorithms.VQE`, solution nodes for :class:`~divi.qprog.algorithms.QAOA`)

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

- :class:`~divi.qprog.checkpointing.CheckpointNotFoundError` - Checkpoint directory or file not found
- :class:`~divi.qprog.checkpointing.CheckpointCorruptedError` - Checkpoint file is invalid or corrupted
- :exc:`RuntimeError` — saving a checkpoint before any iteration has completed
- :exc:`ValueError` — invalid :class:`~divi.qprog.checkpointing.CheckpointConfig` or incompatible resume state

Handle load failures explicitly when you build tooling or CLIs:

.. skip: next

.. code-block:: python

   from pathlib import Path

   from divi.qprog import VQE, HartreeFockAnsatz
   from divi.qprog.checkpointing import (
       CheckpointCorruptedError,
       CheckpointNotFoundError,
   )
   from divi.backends import MaestroSimulator

   try:
       vqe = VQE.load_state(
           Path("my_checkpoints"),
           backend=MaestroSimulator(),
           molecule=mol,
           ansatz=HartreeFockAnsatz(),
           n_layers=2,
       )
   except CheckpointNotFoundError as e:
       print(f"Checkpoint not found: {e}")
   except CheckpointCorruptedError as e:
       print(f"Checkpoint corrupted: {e}")

Limitations
-----------

- **:class:`~divi.qprog.optimizers.ScipyOptimizer`** does not support checkpointing
- Checkpoints are **not portable** across different Python versions or library versions
- Problem configuration must be **manually provided** when loading (not stored in checkpoint)
- Checkpoint files can be **large** for population-based optimizers (MonteCarlo, Pymoo)

Next Steps
----------

- :doc:`core_concepts` — parameters, ``best_params`` vs ``final_params``, and warm-starting
- :doc:`optimizers` — which optimizers support resume and how ``run()`` interacts with checkpoints
- :doc:`visualization` — trajectories using ``losses_history`` / ``param_history`` after long runs
- :doc:`../api_reference/qprog/checkpointing` — ``CheckpointConfig``, ``list_checkpoints``, and exceptions
