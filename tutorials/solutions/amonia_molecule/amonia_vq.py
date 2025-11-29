import pennylane as qml
import numpy as np
from divi.backends import ParallelSimulator
from divi.qprog import GenericLayerAnsatz, HartreeFockAnsatz, UCCSDAnsatz, VQE
from divi.qprog.optimizers import ScipyOptimizer, ScipyMethod
from divi.qprog.workflows import VQEHyperparameterSweep, MoleculeTransformer


class SimpleAnsatz(GenericLayerAnsatz):
    """Ansatz with a single RY rotation and linear entangling."""
    def __init__(self, *args, **kwargs):
        super().__init__(
            gate_sequence=[qml.RY],
            entangler=qml.CNOT,
            entangling_layout="linear",
            *args,
            **kwargs,
        )


class BalancedAnsatz(GenericLayerAnsatz):
    """Ansatz with RY + RZ rotations and linear entangling."""
    def __init__(self, *args, **kwargs):
        super().__init__(
            gate_sequence=[qml.RY, qml.RZ],
            entangler=qml.CNOT,
            entangling_layout="linear",
            *args,
            **kwargs,
        )


class ExpensiveAnsatz(GenericLayerAnsatz):
    """Ansatz with RY + RZ rotations and all-to-all entangling."""
    def __init__(self, *args, **kwargs):
        super().__init__(
            gate_sequence=[qml.RY, qml.RZ],
            entangler=qml.CNOT,
            entangling_layout="all_to_all",
            *args,
            **kwargs,
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


balanced = BalancedAnsatz()
simple = SimpleAnsatz()
expensive = ExpensiveAnsatz()
ansatze = [hf, balanced, uccsd, simple, expensive]
ansatze = [balanced, simple]  # add more if you want

optimizer = ScipyOptimizer(method=ScipyMethod.L_BFGS_B)
n_layers = 1
# Sweep object for geometry 1
energies = np.zeros((len(ansatze), 2))
circuit_counts = np.zeros((len(ansatze), 2))
backend = ParallelSimulator(shots=5000)
for i, ansatz in enumerate(ansatze):
    vqe1 = VQE(hamiltonian1, 
               n_layers=n_layers, 
               ansatz=ansatz, 
               max_iterations=50, 
               backend=backend)
    
    vqe1.run()
    energies[i, 0] = vqe1.best_loss
    circuit_counts[i, 0] = vqe1.total_circuit_count


"""
    vqe2 = VQE(hamiltonian2, 
            n_electrons=8, 
            n_layers=n_layers, 
            ansatz=ansatz, 
            max_iterations=50, 
            backend=backend)
    
    vqe2.run()
    energies[i, 1] = vqe2.losses[-1]
    circuit_counts[i, 1] = vqe2.total_circuit_count

"""
# Run sweeps for both geometries
print(energies)
print(circuit_counts)
