import numpy as np
import pennylane as qml
from pennylane import qchem
from typing import Any

from divi.backends import ParallelSimulator
from divi.qprog import HartreeFockAnsatz, GenericLayerAnsatz, UCCSDAnsatz, VQE
from divi.qprog.optimizers import ScipyOptimizer, ScipyMethod
from divi.qprog.workflows._vqe_sweep_extension import VQEHyperparameterSweepExtension
from divi.qprog.workflows import VQEHyperparameterSweep, MoleculeTransformer

if __name__ == "__main__":
    
    #H2 molecule definition
    opt_bond_length = 0.735
    h2_coords = np.array([(0, 0, 0), (0, 0, opt_bond_length)])
    h2_molecule = qml.qchem.Molecule(
        symbols=["H", "H"],
        coordinates=h2_coords,
        unit="angstrom",
    )

    # VQE algorithm setup
    ansatze_h2 = [HartreeFockAnsatz()]
    bond_sweeps = [0.75]
    mol_transformer = MoleculeTransformer(
        base_molecule=h2_molecule,
        bond_modifiers=bond_sweeps,
    )
    n_layers_list = [1, 2]  # Example layer counts to sweep over
    
    # Sweep over the parameters including n_layers
    sweep = VQEHyperparameterSweepExtension(
        ansatze=ansatze_h2,
        molecule_transformer=mol_transformer,
        optimizer=ScipyOptimizer(method=ScipyMethod.L_BFGS_B),
        max_iterations=3,
        backend=ParallelSimulator(shots=4000),
        n_layers_list=n_layers_list
    )

    sweep.create_programs()
    sweep.run()

    best_cfg, best_E = sweep.aggregate_results()
