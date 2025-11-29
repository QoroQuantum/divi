import pennylane as qml
from divi.backends import ParallelSimulator
from divi.qprog.ansatze import GenericLayerAnsatz, HartreeFockAnsatz, UCCSDAnsatz
from divi.qprog.workflows import MoleculeTransformer, VQEHyperparameterSweep

# 1. Define your base molecule (example: H₂)
starting_h2 = {
    "symbols": ["H", "H"],
    "coordinates": [
        [0.0, 0.0, -0.35],
        [0.0, 0.0,  0.35],
    ]
}

# 2. Create MoleculeTransformer (example bond sweep)
transformer = MoleculeTransformer(
    base_molecule=starting_h2,
    bond_modifiers=[0.5, 0.7, 0.9, 1.1, 1.3]
)

# 3. Example generic ansatz
layer_ansatz = GenericLayerAnsatz(
    gate_sequence=[qml.RY],
    entangler=qml.CNOT,
    entangling_layout="linear"
)

# 4. Build hyperparameter-sweep VQE job
vqe_problem = VQEHyperparameterSweep(
    molecule_transformer=transformer,
    ansatze=[
        HartreeFockAnsatz(),
        UCCSDAnsatz(),
        layer_ansatz,
    ],
    n_layers=[1, 2, 3],
    optimizer="Adam",
    max_iterations=100,
    backend=ParallelSimulator(shots=0)  # analytic
)

# 5. Run the workflow
vqe_problem.create_programs()
vqe_problem.run(blocking=True)
vqe_problem.aggregate_results()

# 6. Visualize
vqe_problem.visualize_results("line")
