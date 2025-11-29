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
    def __init__(self, molecules, bond_sweeps=None, ansatze=None, n_layers_list=None, max_iterations=50):
        """
        molecules: list of qml.qchem.Molecule objects
        bond_sweeps: array of bond modifiers for MoleculeTransformer
        ansatze: list of ansatz objects
        """
        self.molecules = molecules
        self.bond_sweeps = bond_sweeps if bond_sweeps is not None else np.array([0.0])
        self.ansatze = ansatze if ansatze is not None else [HartreeFockAnsatz()] # or any generic one
        self.max_iterations = max_iterations
        #list of layer depths for GenericLayerAnsatz-like circuits
        self.n_layers_list = n_layers_list if n_layers_list is not None else [1]
        
        self.optimizer = ScipyOptimizer(method=ScipyMethod.L_BFGS_B)
        self.backend = ParallelSimulator()
        self.results_by_molecule = {}

    def _clone_ansatz_with_layers(self, ansatz, n_layers):
        """
        Internal helper. Creates a NEW ansatz object with the desired n_layers.
        Safely handles HartreeFockAnsatz & UCCSDAnsatz (no layers).
        """
        if hasattr(ansatz, "n_layers"):
            return ansatz.__class__(
                gate_sequence=getattr(ansatz, "gate_sequence", None),
                entangler=getattr(ansatz, "entangler", None),
                entangling_layout=getattr(ansatz, "entangling_layout", None),
                n_layers=n_layers,
            )
        return ansatz  # This loop is not accessible for HF / UCCSDAnsatz
    
    def run_sweeps(self):
        """
        
        """
        for idx, mol in enumerate(self.molecules):

            print(f"\n=== Running molecule {idx+1}/{len(self.molecules)} ===")
            self.results_by_molecule[idx] = {}

            # Create molecule deformation sweep
            mol_transformer = MoleculeTransformer(
                base_molecule=mol,
                bond_modifiers=self.bond_sweeps,
            )

            #### Loop over layer depths
            for n_layers in self.n_layers_list:
                print(f"\n---- n_layers = {n_layers} ----")

                # Prepare ansätze for this specific number of layers
                ansatze_this_round = [
                    self._clone_ansatz_with_layers(ansatz, n_layers)
                    for ansatz in self.ansatze
                ]

                # Construct the sweep
                vqe_sweep = VQEHyperparameterSweep(
                    ansatze=ansatze_this_round,
                    molecule_transformer=mol_transformer,
                    optimizer=self.optimizer,
                    max_iterations=self.max_iterations,
                    backend=self.backend,
                )

                vqe_sweep.create_programs()
                vqe_sweep.run()
                best_config, best_energy = vqe_sweep.aggregate_results()

                #### Store results by depth
                self.results_by_molecule[idx][n_layers] = (best_config, best_energy)

                print(f"Best config (n_layers={n_layers}): {best_config}")
                print(f"Best energy (n_layers={n_layers}): {best_energy:.8f} Ha")

                # Optional visualisation
                vqe_sweep.visualize_results("bar")

    def summary(self):
        print("\n=== Summary ===")
        for idx, (cfg, E) in self.results_by_molecule.items():
            print(f"Molecule {idx+1}: Best energy = {E:.8f} Ha, Best ansatz = {cfg}")

class SimpleAnsatz(GenericLayerAnsatz):
    def __init__(self, *args, **kwargs):
        super().__init__(
            gate_sequence=[qml.RY],
            entangler=qml.CNOT,
            entangling_layout="linear",
            *args,
            **kwargs,
        )

class BalancedAnsatz(GenericLayerAnsatz):
    def __init__(self, *args, **kwargs):
        super().__init__(
            gate_sequence=[qml.RY, qml.RZ],
            entangler=qml.CNOT,
            entangling_layout="linear",
            *args,
            **kwargs,
        )

class ExpensiveAnsatz(GenericLayerAnsatz):
    def __init__(self, *args, **kwargs):
        super().__init__(
            gate_sequence=[qml.RY, qml.RZ],
            entangler=qml.CNOT,
            entangling_layout="all_to_all",
            *args,
            **kwargs,
        )


if __name__ == "__main__":
    # 1. H2 - Molecule
    # 1.1 coodinates & initial configuration & ansatz:
    h2_coords = np.array([(0, 0, 0), (0, 0, 0.735)])
    ansatze_h2 = [HartreeFockAnsatz()]  # You can add UCCSDAnsatz()
    base_mol_h2 = qml.qchem.Molecule(symbols=["H", "H"], coordinates=h2_coords, unit="angstrom",)
    # 1.2 Initialize and run the calculation
    h2_calc = MoleculeEnergyCalc(
        molecules=[base_mol_h2],
        bond_sweeps=(-0.1, 0.1, 5),
        ansatze=ansatze_h2,
        max_iterations=50,
    )
    h2_calc.run_sweeps()
    h2_calc.summary()
 
    # 2. NH3 - Molecule
    # 2.1 coodinates & initial configuration & ansatz:
    nh3_config1_coords = np.array([
        (0, 0, 0),
        (1.01, 0, 0),
        (-0.5, 0.87, 0),
        (-0.5, -0.87, 0),
    ])
    nh3_config2_coords = np.array([
        (0, 0, 0),
        (-1.01, 0, 0),
        (0.5, -0.87, 0),
        (0.5, 0.87, 0),
    ])
    mol1_nh3 = qml.qchem.Molecule(symbols=["N","H","H","H"], coordinates=nh3_config1_coords)
    mol2_nh3 = qml.qchem.Molecule(symbols=["N","H","H","H"], coordinates=nh3_config2_coords)
    ansatze_nh3 = [
        HartreeFockAnsatz(),
        BalancedAnsatz(),
        SimpleAnsatz(),
        ExpensiveAnsatz(),
        UCCSDAnsatz(),
    ]

    # 2.2 Initialize and run the calculation
    nh3_calc = MoleculeEnergyCalc(
        molecules=[mol1_nh3, mol2_nh3],
        bond_sweeps=np.array([0.0]),
        ansatze=ansatze_nh3,
        n_layers_list=[1, 2, 3, 4],
        max_iterations=50,
    )
    nh3_calc.run_sweeps()
    nh3_calc.summary()
