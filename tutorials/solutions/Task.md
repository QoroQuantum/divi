# 🔬 Checkpoint 1: "Hydrogen Detective - Learning the Basics"

## The H₂ Mystery

You're given a hydrogen molecule (H₂) with a **suboptimal configuration**. Your mission: understand how `divi` works and find the VQE parameters (bond length, ansatz, and number of layers) that yield the **lowest ground state energy**.

## The H₂ Setup

```python
import numpy as np
import pennylane as qml

# The starting molecule (suboptimal configuration)
starting_h2 = qml.qchem.Molecule(
    symbols=["H", "H"], 
    coordinates=np.array([(0, 0, 0), (0, 0, 0.8)]),  # This bond length is not optimal!
    unit="bohr"
)
```

## Your H₂ Mission

1. **Learn the framework**: Understand `MoleculeTransformer`, `VQEHyperparameterSweep`, and the existing ansatze
2. **Find optimal parameters**: Determine the VQE parameters (bond length, ansatz, n_layers) that yield the lowest energy
3. **Design your investigation**: Test different bond lengths, ansatze, and layer counts to improve upon the suboptimal starting configuration
4. **Optimize your solution**: Balance accuracy and computational cost

## H₂ Success Criteria

Your solution should find the VQE parameters that produce the lowest energy. Submit:

- ✅ **Ansatze class**: The ansatz class(es) that achieved the lowest energy
- ✅ **Molecule coordinates**: The H₂ molecule configuration (as a numpy array of coordinates) that produced the lowest energy
- ✅ **Number of layers**: The `n_layers` parameter value that gave the lowest energy

## H₂ Hints

- Experimentally, H₂ has a bond length around 0.74 Å, but your goal is to find the configuration with lowest energy
- Try bond modifiers like `[0.5, 0.7, 0.9, 1.1, 1.3]` to explore different bond lengths
  - Note: `MoleculeTransformer` supports two modes: **scale mode** (multipliers) when all values are positive, or **delta mode** (additive changes in Å) when any value is zero or negative. See the [documentation](https://qoroquantum.github.io/divi) for details.
- The molecule needs 4 qubits for simulation

## Solution Scaffold

```python
from divi.backends import ParallelSimulator
from divi.qprog.workflows import MoleculeTransformer, VQEHyperparameterSweep
# Suggested ansatze (you can use others too):
# from divi.qprog import HartreeFockAnsatz, UCCSDAnsatz

# Create MoleculeTransformer with bond_modifiers (e.g., [0.5, 0.7, 0.9, 1.1, 1.3])
transformer = MoleculeTransformer(base_molecule=starting_h2, bond_modifiers=[...])

layer_ansatz = GenericLayerAnsatz(
    gate_sequence=[...],  # e.g., [qml.RY]
    entangler=...,        # e.g., None or qml.CNOT
    entangling_layout=..., # e.g., None, "linear", or "all-to-all"
)

# Create VQEHyperparameterSweep with your chosen ansatz/ansatze
# Suggested: [HartreeFockAnsatz(), UCCSDAnsatz()] for comparison
vqe_problem = VQEHyperparameterSweep(
    molecule_transformer=transformer,
    ansatze=[...],  # Choose your ansatz/ansatze here
    n_layers= ...,
    optimizer=...,
    max_iterations=...,
    backend=ParallelSimulator(shots=...)
)

# Create programs, run VQE, and aggregate results
vqe_problem.create_programs()
vqe_problem.run(blocking=True)
vqe_problem.aggregate_results()

# Find the parameters (bond length, ansatz, n_layers) that give lowest energy
# Extract optimal molecule coordinates from results
# Visualize results to show your findings
vqe_problem.visualize_results("line")
```

# 💧 Checkpoint 2: "Ammonia Apprentice - Forging an Efficient Tool"

## The Apprentice's Mystery

You've learned the basics on H₂. Now you move to a more complex molecule, Ammonia (NH₃). While the standard `UCCSDAnsatz` is accurate, it is too computationally expensive for demanding problems. Your mission is to forge a more efficient tool—a custom ansatz tailored for Ammonia. To prove your new tool is reliable, you must first match the accuracy of the standard method, and then use your ansatz to correctly confirm a known, fundamental property of Ammonia: that it exists in two distinct, "flipped" configurations that share the same lowest energy (degeneracy). This validated, efficient tool will be essential for the final challenge.

## The Apprentice's Setup

```python
# Define the active space for a 12-qubit simulation
# This freezes core 1s electrons and focuses on valence electrons
active_electrons = 8
active_orbitals = 6

# The two degenerate configurations of Ammonia (coordinates)
nh3_config1_coords = np.array(
    [
        (0, 0, 0), # N  
        (1.01, 0, 0), # H₁  
        (-0.5, 0.87, 0), # H₂  
        (-0.5, -0.87, 0) # H₃
    ]  
)

nh3_config2_coords = np.array(
    [
        (0, 0, 0),  # N (inverted)
        (-1.01, 0, 0),  # H₁
        (0.5, -0.87, 0),  # H₂
        (0.5, 0.87, 0),  # H₃
    ]
)

# Create molecule objects
nh3_molecule1 = qml.qchem.Molecule(
    symbols=["N", "H", "H", "H"],
    coordinates=nh3_config1_coords,
)

nh3_molecule2 = qml.qchem.Molecule(
    symbols=["N", "H", "H", "H"],
    coordinates=nh3_config2_coords,
)

# Build Hamiltonians with active space parameters
hamiltonian1, qubits = qml.qchem.molecular_hamiltonian(
    nh3_molecule1,
    active_electrons=active_electrons,
    active_orbitals=active_orbitals,
)

hamiltonian2, qubits = qml.qchem.molecular_hamiltonian(
    nh3_molecule2,
    active_electrons=active_electrons,
    active_orbitals=active_orbitals,
)

```

## Your Apprentice Mission

1. **Design custom ansatze**: Create multiple distinct ansatze (e.g., using `GenericLayerAnsatz` or custom implementations) with varying complexity
2. **Find the best tool**: Identify which ansatz provides the best trade-off between accuracy and efficiency
3. **Validate your tool**: Use your best ansatz to confirm both NH₃ configurations are degenerate (same energy)

## NH₃ Hints

- The 12-qubit active space (8 electrons, 6 orbitals) reduces computational complexity while maintaining chemical accuracy
- You can use `GenericLayerAnsatz` or implement custom ansatze from the literature (e.g., hardware-efficient, chemistry-inspired)
- Design ansatze with varying complexity: minimalist (few gates), balanced (moderate), and expressive (many gates)
- Test your ansatz with multiple random seeds to ensure robustness
- The two configurations should have energies within ~1 mHa of each other to confirm degeneracy

## Solution Scaffold

```python
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
```
