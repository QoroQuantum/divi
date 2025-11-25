# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

"""Demonstrates checkpointing functionality for variational quantum algorithms.

This example shows how to:
- Save checkpoints during optimization
- Resume optimization from a checkpoint
- Adjust iteration targets after loading
- List and inspect checkpoints
"""

import shutil
from pathlib import Path

import numpy as np
import pennylane as qml

from divi.backends import ParallelSimulator
from divi.qprog import VQE, HartreeFockAnsatz
from divi.qprog.checkpointing import (
    CheckpointConfig,
    get_checkpoint_info,
    get_latest_checkpoint,
    list_checkpoints,
)
from divi.qprog.optimizers import MonteCarloOptimizer

if __name__ == "__main__":
    # Set up checkpoint directories
    checkpoint_dir = Path("checkpoint_demo")
    checkpoint_dir_interval = Path("checkpoint_demo_interval")

    try:
        # Create molecule
        mol = qml.qchem.Molecule(
            symbols=["H", "H"], coordinates=np.array([(0, 0, 0), (0, 0, 0.5)])
        )

        # Initial run - save checkpoints every iteration
        print("=" * 60)
        print("Step 1: Initial optimization run with checkpointing")
        print("=" * 60)

        vqe1 = VQE(
            molecule=mol,
            ansatz=HartreeFockAnsatz(),
            n_layers=1,
            optimizer=MonteCarloOptimizer(population_size=10),
            max_iterations=3,
            backend=ParallelSimulator(),
        )

        vqe1.run(checkpoint_config=CheckpointConfig(checkpoint_dir=checkpoint_dir))
        print(f"Initial run completed: {vqe1.current_iteration} iterations")
        print(f"Best energy: {vqe1.best_loss:.6f}")

        # List all checkpoints
        print("\n" + "=" * 60)
        print("Step 2: Listing available checkpoints")
        print("=" * 60)

        checkpoints = list_checkpoints(checkpoint_dir)
        print(f"Found {len(checkpoints)} checkpoint(s):")
        for cp in checkpoints:
            print(
                f"  Iteration {cp.iteration}: {cp.path.name} ({cp.size_bytes / 1024:.2f} KB)"
            )

        # Get latest checkpoint info
        latest = get_latest_checkpoint(checkpoint_dir)
        if latest:
            info = get_checkpoint_info(latest)
            print(f"\nLatest checkpoint details:")
            print(f"  Iteration: {info.iteration}")
            print(f"  Path: {info.path.name}")
            print(f"  Size: {info.size_bytes / 1024:.2f} KB")
            print(f"  Valid: {info.is_valid}")

        # Resume from checkpoint
        print("\n" + "=" * 60)
        print("Step 3: Resuming optimization from checkpoint")
        print("=" * 60)

        vqe2 = VQE.load_state(
            checkpoint_dir=checkpoint_dir,
            backend=ParallelSimulator(),
            molecule=mol,
            ansatz=HartreeFockAnsatz(),
            n_layers=1,
        )

        print(f"Loaded state from iteration {vqe2.current_iteration}")
        print(f"Resumed best energy: {vqe2.best_loss:.6f}")

        # Continue optimization with more iterations
        vqe2.max_iterations = 6  # Extend from 3 to 6 total iterations
        vqe2.run()

        print(f"\nResumed run completed: {vqe2.current_iteration} total iterations")
        print(f"Final best energy: {vqe2.best_loss:.6f}")
        print(f"Total circuits run: {vqe2.total_circuit_count}")

        # Demonstrate checkpoint interval
        print("\n" + "=" * 60)
        print("Step 4: Checkpointing with interval (every 2 iterations)")
        print("=" * 60)

        vqe3 = VQE(
            molecule=mol,
            ansatz=HartreeFockAnsatz(),
            n_layers=1,
            optimizer=MonteCarloOptimizer(population_size=10),
            max_iterations=5,
            backend=ParallelSimulator(),
        )

        vqe3.run(
            checkpoint_config=CheckpointConfig(
                checkpoint_dir=checkpoint_dir_interval, checkpoint_interval=2
            )
        )

        checkpoints_interval = list_checkpoints(checkpoint_dir_interval)
        print(f"Checkpoints saved (every 2 iterations): {len(checkpoints_interval)}")
        for cp in checkpoints_interval:
            print(f"  Iteration {cp.iteration}")

        print("\n" + "=" * 60)
        print("Checkpointing demo complete!")
        print("=" * 60)
    finally:
        # Clean up checkpoint directories
        for dir_path in [checkpoint_dir, checkpoint_dir_interval]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
                print(f"Cleaned up: {dir_path}")
