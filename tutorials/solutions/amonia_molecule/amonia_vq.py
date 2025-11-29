import pennylane as qml
import numpy as np
from divi.backends import ParallelSimulator
from divi.qprog import GenericLayerAnsatz, HartreeFockAnsatz, UCCSDAnsatz
from divi.qprog.workflows import MoleculeTransformer, VQEHyperparameterSweep
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
