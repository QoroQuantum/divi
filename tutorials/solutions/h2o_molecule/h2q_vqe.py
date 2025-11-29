import numpy as np
import pennylane as qml
from divi.backends import ParallelSimulator
from divi.qprog import HartreeFockAnsatz, UCCSDAnsatz
from divi.qprog.optimizers import ScipyOptimizer, ScipyMethod
from divi.qprog.sweep import MoleculeTransformer, VQEHyperparameterSweep


### Remarks
#
# This is only to try out the different parameters
#


# Define the base H2 molecule
base_mol = qml.qchem.Molecule(
    symbols=["H", "H"],
    coordinates=np.array([(0, 0, 0), (0, 0, 0.735)]),
)

# Define bond length sweep (in bohr)
bond_lengths = np.linspace(0.4, 1.2, 5)

# Create a MoleculeTransformer to generate molecule variants
mol_transformer = MoleculeTransformer(
    base_molecule=base_mol,
    bond_modifiers=bond_lengths,
)

# Choose ansatze to sweep
ansatze = [HartreeFockAnsatz(), UCCSDAnsatz()]

# Define optimizer
optimizer = ScipyOptimizer(method=ScipyMethod.L_BFGS_B)

# Setup the hyperparameter sweep
vqe_sweep = VQEHyperparameterSweep(
    ansatze=ansatze,
    molecule_transformer=mol_transformer,
    optimizer=optimizer,
    max_iterations=100,
    backend=ParallelSimulator(),
)

# Create the programs (VQE runs for each ansatz & bond length)
vqe_sweep.create_programs()

# Execute the sweep (this runs all VQE programs)
vqe_sweep.run()

# Aggregate results to find the best energy and configuration
best_config, best_energy = vqe_sweep.aggregate_results()
print(f"Best configuration: {best_config}, Energy: {best_energy:.6f}")

# Visualize the results
vqe_sweep.visualize_results(graph_type="line")  # or graph_type="scatter"




#####
# Remarks: 
# UCCSDAnsatz: is the best for gaining the best accuracy
# Hartee-Fock: is enough for our use-case 
# 
#####



