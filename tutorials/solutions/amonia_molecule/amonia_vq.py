import pennylane as qml
import numpy as np
from divi.backends import ParallelSimulator
from divi.qprog import GenericLayerAnsatz, HartreeFockAnsatz, UCCSDAnsatz, VQE
from divi.qprog.optimizers import ScipyOptimizer, ScipyMethod
from divi.qprog.workflows import VQEHyperparameterSweep, MoleculeTransformer

# Different models: 

# 1. First, we test the most simple ansatz with only one rotation
simple = GenericLayerAnsatz(
    gate_sequence=[qml.RY], 
    entangler=qml.CNOT, 
    entangling_layout="linear" 
    )
# 2. Secondly, we test the more complex ansatz with one y- and one z- rotation
balanced = GenericLayerAnsatz(
    gate_sequence=[qml.RY, qml.RZ], 
    entangler=qml.CNOT, 
    entangling_layout="linear"
    )
# 3. Thirdly, we test the more complex ansatz with one y- and one z- rotation and entangling: all-to-all
expensive = GenericLayerAnsatz(
    gate_sequence=[qml.RY, qml.RZ], 
    entangler=qml.CNOT, 
    entangling_layout="all_to_all"
    )
# 3. Thirdly, we test the HF-ansatz
hf = HartreeFockAnsatz()
# 4. Lastly, we benchmark the results agains the accurate UCCSD - model
uccsd = UCCSDAnsatz()



# Define optimizer
optimizer = ScipyOptimizer(method=ScipyMethod.L_BFGS_B)


# Benchmarking 
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
    coordinates=nh3_config1_coords
)

nh3_molecule2 = qml.qchem.Molecule(
    symbols=["N", "H", "H", "H"],
    coordinates=nh3_config2_coords
)
"""
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
"""
bond_sweeps = np.array([0.0])
# Create a MoleculeTransformer to generate molecule variants
mol_transformer1 = MoleculeTransformer(
    base_molecule=nh3_molecule1,
    bond_modifiers=bond_sweeps,
)

mol_transformer2 = MoleculeTransformer(
    base_molecule=nh3_molecule2,
    bond_modifiers=bond_sweeps,
)
# Benchmark these ansätze on NH3

ansatze = [hf, balanced, uccsd, simple, expensive]  # add more if you want

optimizer = ScipyOptimizer(method=ScipyMethod.L_BFGS_B)

# Sweep object for geometry 1
sweep1 = VQEHyperparameterSweep(
    ansatze=ansatze,
    molecule_transformer=mol_transformer1,
    optimizer=optimizer,
    max_iterations=30,
    backend=ParallelSimulator(),
)

# Sweep object for geometry 2
sweep2 = VQEHyperparameterSweep(
    ansatze=ansatze,
    molecule_transformer=mol_transformer2,
    optimizer=optimizer,
    max_iterations=30,
    backend=ParallelSimulator(),
)


# Run sweeps for both geometries

print("\nRunning NH3 configuration 1 sweep…")
sweep1.create_programs()
sweep1.run()
results1 = sweep1.aggregate_results()

print("\nRunning NH3 configuration 2 sweep…")
sweep2.create_programs()
sweep2.run()
results2 = sweep2.aggregate_results()


# Extract and print final results

best_config_1, best_energy_1 = results1
best_config_2, best_energy_2 = results2

print("\n==============================")
print(" NH3 RESULTS (12-Qubit Space) ")
print("==============================")
print(f"Config 1 best ansatz: {best_config_1}")
print(f"Config 1 best energy: {best_energy_1:.8f} Ha")

print(f"\nConfig 2 best ansatz: {best_config_2}")
print(f"Config 2 best energy: {best_energy_2:.8f} Ha")

print("\nEnergy difference (degeneracy check):")
print(f"ΔE = {abs(best_energy_1 - best_energy_2):.8f} Ha")


# Plot results

# Note: These plots show all ansatz results for comparisons
sweep1.visualize_results("bar")
sweep2.visualize_results("bar")



