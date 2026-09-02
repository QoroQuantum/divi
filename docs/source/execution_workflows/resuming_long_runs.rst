Resuming Long-Running or Interrupted Runs
=========================================

Checkpointing saves program state to disk via
:class:`~divi.qprog.checkpointing.CheckpointConfig` so you can resume after
interruptions, inspect intermediate progress, or raise ``max_iterations`` after
a reload. It covers both individual
:class:`~divi.qprog.variational_quantum_algorithm.VariationalQuantumAlgorithm`
runs and complete :class:`~divi.qprog.ensemble.ProgramEnsemble` workflows.

Overview
--------

On each checkpoint, Divi writes **program** state (parameters, losses, iteration count, RNG state) and **optimizer** state (anything the optimizer needs to continue). That enables you to:

- **Resume interrupted runs** — continue from the last saved iteration
- **Debug** — inspect intermediate parameters and losses on disk
- **Chunk long jobs** — stop and restart without re-running from scratch
- **Raise iteration caps** — increase ``max_iterations`` after :meth:`~divi.qprog.variational_quantum_algorithm.VariationalQuantumAlgorithm.load_state`
- **Resume ensemble rounds** — restore workflow state, accounting, and eligible
  child optimizers without repeating completed rounds

Checkpointing is available to variational programs such as
:class:`~divi.qprog.algorithms.VQE` and :class:`~divi.qprog.algorithms.QAOA`,
but the optimizer must also support restoring its state:

.. list-table:: Checkpoint compatibility
   :header-rows: 1
   :widths: 45 20 35

   * - Optimizer
     - Supported
     - Notes
   * - :class:`~divi.qprog.optimizers.MonteCarloOptimizer`
     - Yes
     - Restores the population and RNG state.
   * - :class:`~divi.qprog.optimizers.PymooOptimizer`
     - Yes
     - Supports CMA-ES and Differential Evolution state.
   * - :class:`~divi.qprog.optimizers.GridSearchOptimizer`
     - Yes
     - Restores the grid and any losses already evaluated.
   * - :class:`~divi.qprog.optimizers.ScipyOptimizer`
     - No
     - The underlying SciPy methods do not expose resumable state.
   * - :class:`~divi.qprog.optimizers.QNGOptimizer`
     - No
     - Rejects checkpoint configuration before the run starts.
   * - :class:`~divi.qprog.optimizers.SPSAOptimizer`,
       :class:`~divi.qprog.optimizers.QNSPSAOptimizer`, and
       :class:`~divi.qprog.optimizers.QUIVEROptimizer`
     - No
     - Their adaptive state is rebuilt rather than restored.

Basic Usage
-----------

Saving Checkpoints
^^^^^^^^^^^^^^^^^^

To enable checkpointing, pass a :class:`~divi.qprog.checkpointing.CheckpointConfig` object to the ``run()`` method:

.. dashboard-example: checkpointing

.. code-block:: python

   from pathlib import Path
   from divi.qprog import VQE, HartreeFockAnsatz
   from divi.qprog.checkpointing import CheckpointConfig
   from divi.qprog.optimizers import MonteCarloOptimizer
   from divi.backends import MaestroSimulator
   from pyscf import gto

   # Create a molecule
   mol = gto.M(atom="H 0 0 -0.6614; H 0 0 0.6614", basis="sto-3g", unit="Bohr")

   # Create VQE program
   vqe = VQE(
       molecule=mol,
       ansatz=HartreeFockAnsatz(),
       n_layers=2,
       max_iterations=10,
       optimizer=MonteCarloOptimizer(),
       backend=MaestroSimulator(),
   )

   # Run with checkpointing enabled
   checkpoint_dir = Path("my_checkpoints")
   vqe.run(checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir))

By default, checkpoints are saved **every iteration**. Each checkpoint is stored in a subdirectory named ``checkpoint_{iteration:03d}`` (e.g., ``checkpoint_001``, ``checkpoint_002``).

A run always checkpoints its final iteration, even when that iteration does not land on a ``checkpoint_interval`` boundary, and whether it ended by converging, by early stopping, or by cancellation. Nothing between the last interval boundary and the end of the run is lost.

Checkpoint Interval
^^^^^^^^^^^^^^^^^^^

To save checkpoints less frequently, set the ``checkpoint_interval`` parameter:

.. invisible-code-block: python

   vqe = VQE(molecule=mol, ansatz=HartreeFockAnsatz(), n_layers=2,
             max_iterations=10, optimizer=MonteCarloOptimizer(), backend=MaestroSimulator())

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
             max_iterations=10, optimizer=MonteCarloOptimizer(), backend=MaestroSimulator())

.. code-block:: python

   # Creates a directory like "checkpoint_20250115_143022"
   config = CheckpointConfig.with_timestamped_dir()
   vqe.run(checkpoint_config=config)

Or with a checkpoint interval:

.. invisible-code-block: python

   vqe = VQE(molecule=mol, ansatz=HartreeFockAnsatz(), n_layers=2,
             max_iterations=10, optimizer=MonteCarloOptimizer(), backend=MaestroSimulator())

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

By default, ``load_state()`` loads the latest checkpoint whose files are both present. A checkpoint left incomplete by an interrupted write is skipped rather than treated as the latest, so the last good checkpoint stays reachable. To load a specific checkpoint:

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

.. _resuming-ensembles:

Ensemble Workflows
------------------

Ensembles use the same :class:`~divi.qprog.checkpointing.CheckpointConfig`, but
restore onto a newly constructed ensemble instance because its constructor
defines the workflow and problem configuration:

.. skip: next

.. code-block:: python

   checkpoint_dir = Path("ensemble_checkpoints")
   config = CheckpointConfig(checkpoint_dir=checkpoint_dir)

   ensemble.run(max_rounds=6, checkpoint_config=config)

   resumed = make_the_same_ensemble(backend)
   resumed.restore_state(checkpoint_dir)
   resumed.run(max_rounds=6)

Completed rounds resume from their output state. Interrupted rounds resume from
their saved input state and reconstruct the entire program manifest before any
child state is applied. Children whose terminal results were saved are reused
without rerunning; otherwise eligible variational children continue from their
optimizer checkpoints, and unsupported or damaged children restart within that
round. See :ref:`ensemble-checkpointing` for the eligibility rules, custom
workflow-state hooks, and safety constraints.

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
   qaoa2.max_iterations = 20  # Raise the cumulative target beyond the saved run
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

:class:`~divi.qprog.algorithms.IterativeQAOA` optimises at one depth after another and restarts its iteration count at each one, so it nests that layout one level deeper — ``depth_01/checkpoint_001``, ``depth_02/checkpoint_001``, and so on. Depths therefore never overwrite one another, and ``load_state()`` on the top-level directory resolves to the deepest depth that has a complete checkpoint. Resuming continues the depth schedule from there rather than restarting at depth 1.

.. _ensemble-checkpoint-layout:

Ensemble checkpoint structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An ensemble checkpoint is organised by workflow round:

.. code-block:: text

   ensemble_checkpoint_dir/
   ├── round_001/
   │   ├── round_start.json       # Ordered child descriptors
   │   ├── round_completion.json  # Completed output and accounting
   │   ├── program_000/
   │   │   ├── checkpoint_.../       # Iterative VQA state
   │   │   └── program_completion.json # Terminal child state
   │   └── program_001/
   │       └── ...
   └── round_002/
       ├── round_start.json
       └── ...

``round_start.json`` records the ordered program IDs and types expected when the
interrupted round is reconstructed. The reconstructed structure must match
before child state is restored. ``create_programs(state)`` must therefore be
deterministic for a fixed ensemble configuration and restored state; workflows
with nondeterministic generation must persist and reuse those choices as part of
their workflow state.

Checkpointed :class:`~divi.qprog.workflows.PartitioningProgramEnsemble` runs
with :class:`~divi.qprog.problems.BinaryOptimizationProblem` require a
reproducible decomposer. Divi's seeded
:class:`~divi.qprog.problems.CommunityDecomposer` is supported; arbitrary
``hybrid`` decomposers are rejected because they cannot guarantee that a
reconstructed program slot represents the same subproblem.

Each child writes ``program_completion.json`` only after its terminal result is
available. That marker is preferred over iterative optimizer checkpoints, so a
completed child is not rerun even when its optimizer itself cannot checkpoint.
If terminal state is missing or invalid, Divi falls back to the latest complete
iterative checkpoint and then to a fresh child. ``round_completion.json`` is
written only after the round's state reduction succeeds, so its presence marks
a completed round. Workflows with state may place explicitly referenced
artifacts such as LASSQD's ``input_state.npz`` and ``output_state.npz``
alongside these markers.

The ``program_state.json`` file contains:

- Current iteration number
- Loss history
- Best parameters found so far
- Current parameters
- Random number generator state
- Algorithm-specific state (e.g., eigenstate for :class:`~divi.qprog.algorithms.VQE`, solution nodes for :class:`~divi.qprog.algorithms.QAOA`)

The ``optimizer_state.json`` file contains optimizer-specific data:

- For ``MonteCarloOptimizer``: Population, evaluated population, losses, RNG state
- For ``PymooOptimizer``: Serialised algorithm object and population
- For ``GridSearchOptimizer``: Parameter grid, evaluated losses, best point

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

- Optimizers marked unsupported in the compatibility table reject checkpointing
- Checkpoints are **not portable** across different Python versions or library versions
- Problem configuration must be **manually provided** when loading (not stored in checkpoint)
- Checkpoint files can be **large** for population-based optimizers (MonteCarlo, Pymoo)
- A child that is not eligible for mid-round restore restarts from the ensemble
  round's saved input state

Next Steps
----------

- :doc:`core_concepts` — parameters, ``best_params`` vs ``final_params``, and warm-starting
- :doc:`optimizers` — which optimizers support resume and how ``run()`` interacts with checkpoints
- :doc:`program_ensembles` — ensemble round semantics, child eligibility, and
  custom workflow-state checkpointing
- :doc:`visualization` — trajectories using ``losses_history`` / ``param_history`` after long runs
- :doc:`../api_reference/qprog/checkpointing` — ``CheckpointConfig``, ``list_checkpoints``, and exceptions
