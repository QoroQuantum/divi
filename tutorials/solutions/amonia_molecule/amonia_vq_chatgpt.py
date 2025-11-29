from divi.backends import ParallelSimulator
from divi.qprog import GenericLayerAnsatz, UCCSDAnsatz
# Suggested for comparison:
# from divi.qprog import HartreeFockAnsatz

# Design multiple custom ansatze with varying complexity
# Example using GenericLayerAnsatz:
minimalist = GenericLayerAnsatz(
    gate_sequence=[...],  # e.g., [qml.RY]
    entangler=...,        # e.g., None or qml.CNOT
    entangling_layout=..., # e.g., None, "linear", or "all-to-all"
)

# Or implement custom ansatze from literature

# Test each custom ansatz on hamiltonian1
# Select the best trade-off

# Validate: Run best custom ansatz on both hamiltonian1 and hamiltonian2
# Confirm energies are degenerate (difference < 1 mHa)