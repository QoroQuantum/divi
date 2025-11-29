import numpy as np
import pennylane as qml

from divi.backends import ParallelSimulator
from divi.qprog import HartreeFockAnsatz, GenericLayerAnsatz, UCCSDAnsatz
from divi.qprog.optimizers import ScipyOptimizer, ScipyMethod
from divi.qprog.workflows import VQEHyperparameterSweep, MoleculeTransformer

class MoleculeEnergyCalc:
    """
    This class calculates the ground energy for molecules using VQE sweeps.
    Supports multiple ansätze and geometries.
    """
    def __init__(self, molecules, bond_sweeps=None, ansatze=None, max_iterations=50):
        """
        molecules: list of qml.qchem.Molecule objects
        bond_sweeps: array of bond modifiers for MoleculeTransformer
        ansatze: list of ansatz objects
        """
        self.molecules = molecules
        self.bond_sweeps = bond_sweeps if bond_sweeps is not None else np.array([0.0])
        self.ansatze = ansatze if ansatze is not None else [HartreeFockAnsatz()] # or any generic one
        self.max_iterations = max_iterations
        self.optimizer = ScipyOptimizer(method=ScipyMethod.L_BFGS_B)
        self.backend = ParallelSimulator()
        self.results_by_molecule = {}

    def run_sweeps(self):
        for idx, mol in enumerate(self.molecules):
            print(f"\n=== Running VQE sweep for molecule {idx+1}/{len(self.molecules)} ===")
            mol_transformer = MoleculeTransformer(base_molecule=mol, bond_modifiers=self.bond_sweeps)

            vqe_sweep = VQEHyperparameterSweep(
                ansatze=self.ansatze,
                molecule_transformer=mol_transformer,
                optimizer=self.optimizer,
                max_iterations=self.max_iterations,
                backend=self.backend,
            )

            vqe_sweep.create_programs()
            vqe_sweep.run()
            best_config, best_energy = vqe_sweep.aggregate_results()
            self.results_by_molecule[idx] = (best_config, best_energy)

            print(f"Best configuration for molecule {idx+1}: {best_config}")
            print(f"Best energy for molecule {idx+1}: {best_energy:.8f} Ha")

            # Optional: visualize per molecule
            vqe_sweep.visualize_results("bar")

    def summary(self):
        print("\n=== Summary ===")
        for idx, (cfg, E) in self.results_by_molecule.items():
            print(f"Molecule {idx+1}: Best energy = {E:.8f} Ha, Best ansatz = {cfg}")


if __name__ == "__main__":
    # Base H2 molecule:
    base_mol = qml.qchem.Molecule(
        symbols=["H", "H"],
        coordinates=np.array([(0, 0, 0), (0, 0, 0.735)]),
        unit="angstrom",
    )

    # Sweep parameters
    bond_sweeps = (-0.1, 0.1, 5)
    ansatze = [HartreeFockAnsatz()]  # You can add UCCSDAnsatz()
    
    # Initialize and run the calculation
    h2_calc = MoleculeEnergyCalc(
        molecules=[base_mol],
        bond_sweeps=bond_sweeps,
        ansatze=ansatze,
        max_iterations=50,
    )

    h2_calc.run_sweeps()
    h2_calc.summary()

    # Two degenerate NH3 geometries
    nh3_coords1 = np.array([(0, 0, 0), (1.01, 0, 0), (-0.5, 0.87, 0), (-0.5, -0.87, 0)])
    nh3_coords2 = np.array([(0, 0, 0), (-1.01, 0, 0), (0.5, -0.87, 0), (0.5, 0.87, 0)])

    nh3_mol1 = qml.qchem.Molecule(symbols=["N", "H", "H", "H"], coordinates=nh3_coords1)
    nh3_mol2 = qml.qchem.Molecule(symbols=["N", "H", "H", "H"], coordinates=nh3_coords2)

    molecules = [nh3_mol1, nh3_mol2]

    # Define ansätze
    ansatze = [
        HartreeFockAnsatz(),
        GenericLayerAnsatz([qml.RY], entangler=qml.CNOT, entangling_layout="linear"),
        GenericLayerAnsatz([qml.RY, qml.RZ], entangler=qml.CNOT, entangling_layout="linear"),
        GenericLayerAnsatz([qml.RY, qml.RZ], entangler=qml.CNOT, entangling_layout="all_to_all"),
        UCCSDAnsatz()
    ]

    # Run calculation
    nh3_calc = MoleculeEnergyCalc(molecules=molecules, bond_sweeps=np.array([0.0]), ansatze=ansatze, max_iterations=30)
    nh3_calc.run_sweeps()
    nh3_calc.summary()
