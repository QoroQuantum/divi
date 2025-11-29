import numpy as np
import pennylane as qml

from divi.backends import ParallelSimulator
from divi.qprog import HartreeFockAnsatz, UCCSDAnsatz
from divi.qprog.optimizers import ScipyOptimizer, ScipyMethod
from divi.qprog.workflows import MoleculeTransformer, VQEHyperparameterSweep

# Define the base H2 molecule
base_mol = qml.qchem.Molecule(
    symbols=["H", "H"],
    coordinates=np.array([(0, 0, 0), (0, 0, 0.735)]),
    unit="angstrom",
)

# Bond-length sweep (relative modifiers)
bond_sweeps = np.linspace(-0.2, 0.2, 5)  # from -0.2 to +0.2 angstroms

# Create a MoleculeTransformer to generate molecule variants
mol_transformer = MoleculeTransformer(
    base_molecule=base_mol,
    bond_modifiers=bond_sweeps,
)

# Choose ansatze to sweep
# You can also add UCCSDAnsatz() here if you like
ansatze = [HartreeFockAnsatz()]

# Number of layers to sweep over
num_layers_list = [3]

# Define optimizer
optimizer = ScipyOptimizer(method=ScipyMethod.L_BFGS_B)

# Backend
backend = ParallelSimulator()

# Store best result per layer depth
best_results_by_layer = {}

for n_layers in num_layers_list:
    print(f"\n=== Running VQE sweep with n_layers = {n_layers} ===")

    # Setup the hyperparameter sweep for this specific layer depth
    vqe_sweep = VQEHyperparameterSweep(
        ansatze=ansatze,
        molecule_transformer=mol_transformer,
        optimizer=optimizer,
        max_iterations=100,
        backend=backend,
        n_layers=n_layers,   # <- fixed number of layers for this sweep
    )

    # Create the programs (VQE runs for each ansatz & bond length)
    vqe_sweep.create_programs()

    # Execute the sweep (this runs all VQE programs)
    vqe_sweep.run()

    # Aggregate results to find the best energy and configuration
    best_config, best_energy = vqe_sweep.aggregate_results()
    best_results_by_layer[n_layers] = (best_config, best_energy)

    print(f"Best configuration (layers={n_layers}): {best_config}")
    print(f"Best energy (layers={n_layers}): {best_energy:.6f} Ha")

    # Visualize the results for this layer depth
    # (e.g. one figure per depth)
    vqe_sweep.visualize_results(graph_type="line")  # or "scatter"


print("\n=== Summary over all layer depths ===")
for n_layers, (cfg, E) in best_results_by_layer.items():
    print(f"n_layers={n_layers}: E = {E:.6f} Ha, config = {cfg}")
