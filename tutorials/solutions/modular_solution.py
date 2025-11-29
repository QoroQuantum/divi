import numpy as np
import pennylane as qml

from divi.backends import ParallelSimulator
from divi.qprog import HartreeFockAnsatz, GenericLayerAnsatz, UCCSDAnsatz, VQE
from divi.qprog.optimizers import ScipyOptimizer, ScipyMethod
from divi.qprog.workflows import VQEHyperparameterSweep, MoleculeTransformer

class MoleculeEnergyCalc:
    """
    This class calculates the ground energy for molecules using VQE sweeps.
    Supports multiple ansätze and geometries.
    """
    def __init__(self, molecules, 
                 bond_sweeps=None, 
                 ansatze=None, 
                 n_layers_list=None, 
                 hamiltonians = [], 
                 max_iterations=50):
        """
        molecules: list of qml.qchem.Molecule objects
        bond_sweeps: array of bond modifiers for MoleculeTransformer
        ansatze: list of ansatz objects
        n_layers_list: list of number of layers -> stays empty for HF and UCCSD ansatz
        """
        self.molecules = molecules
        self.bond_sweeps = bond_sweeps if bond_sweeps is not None else np.array([0.0])
        self.ansatze = ansatze if ansatze is not None else [HartreeFockAnsatz()] # or any generic one
        self.max_iterations = max_iterations
        #list of layer depths for GenericLayerAnsatz-like circuits
        self.n_layers_list = n_layers_list if n_layers_list is not None else [1]
        self.hamiltonians = hamiltonians or [] 
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
    
    def run_hamiltonian_vqe(self):
        """
        Run VQE directly on a list of explicit Hamiltonians (no geometry sweep).
        """
        print("\n=== Running VQE on Explicit Hamiltonians ===")
        self.results["hamiltonians"] = {}

        for h_idx, H in enumerate(self.hamiltonians):
            self.results["hamiltonians"][h_idx] = {}

            for n_layers in self.n_layers_list:
                print(f"\n--- Hamiltonian {h_idx+1}, n_layers={n_layers} ---")

                energies = []
                circuits = []

                for ansatz in self.ansatze:
                    ans = self._clone_with_layers(ansatz, n_layers)

                    vqe = VQE(
                        H,
                        ansatz=ans,
                        backend=self.backend,
                        n_layers=n_layers,
                        max_iterations=self.max_iterations,
                    )

                    vqe.run()

                    energies.append(vqe.best_loss)
                    circuits.append(vqe.total_circuit_count)

                self.results["hamiltonians"][h_idx][n_layers] = (energies, circuits)

    def run_geometry_sweeps(self):
        """
        Classical VQEHyperparameterSweep workflow with MoleculeTransformer.
        Works for H2 or general molecules with bond sweeps.
        """
        print("\n=== Running Geometry-Based Sweeps ===")
        self.results["molecules"] = {}

        for mol_idx, mol in enumerate(self.molecules):
            self.results["molecules"][mol_idx] = {}

            mol_transformer = MoleculeTransformer(
                base_molecule=mol,
                bond_modifiers=self.bond_sweeps,
            )

            for n_layers in self.n_layers_list:
                print(f"\n--- Molecule {mol_idx+1}, n_layers={n_layers} ---")

                ans_list = [self._clone_with_layers(a, n_layers) for a in self.ansatze]

                sweep = VQEHyperparameterSweep(
                    ansatze=ans_list,
                    molecule_transformer=mol_transformer,
                    optimizer=self.optimizer,
                    max_iterations=self.max_iterations,
                    backend=self.backend,
                )

                sweep.create_programs()
                sweep.run()

                best_cfg, best_E = sweep.aggregate_results()

                self.results["molecules"][mol_idx][n_layers] = (best_cfg, best_E)


    def summary(self):
        print("\n===== SUMMARY =====")
        print(self.results)


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
    print("H2-Molecule Energy Calculation started...")
    # H₂ definition
    h2_coords = np.array([(0, 0, 0), (0, 0, 0.735)])
    h2_molecule = qml.qchem.Molecule(
        symbols=["H", "H"],
        coordinates=h2_coords,
        unit="angstrom",
    )

    # Only HF ansatz for H₂ in this example
    ansatze_h2 = [HartreeFockAnsatz()]

    h2_calc = MoleculeEnergyCalc(
        molecules=[h2_molecule],
        hamiltonians=None,             
        bond_sweeps=(-0.1, 0.1, 5),
        ansatze=ansatze_h2,
        max_iterations=50,
        n_layers_list=[1, 2],
    )

    h2_calc.run_geometry_sweeps()
    print("\nH₂ Summary:")
    h2_calc.summary()

    # --- NH₃ definitions ---
    print("NH3-Molecule Energy Calculation started...")
    nh3_coords1 = np.array([
        (0, 0, 0),
        (1.01, 0, 0),
        (-0.5, 0.87, 0),
        (-0.5, -0.87, 0),
    ])

    nh3_coords2 = np.array([
        (0, 0, 0),
        (-1.01, 0, 0),
        (0.5, -0.87, 0),
        (0.5, 0.87, 0),
    ])

    nh3_molecule_1 = qml.qchem.Molecule(
        symbols=["N", "H", "H", "H"],
        coordinates=nh3_coords1,
    )
    nh3_molecule_2 = qml.qchem.Molecule(
        symbols=["N", "H", "H", "H"],
        coordinates=nh3_coords2,
    )

    # Active space
    active_electrons = 8
    active_orbitals = 6

    # Build explicit Hamiltonians
    H1, _ = qml.qchem.molecular_hamiltonian(
        nh3_molecule_1,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )
    H2, _ = qml.qchem.molecular_hamiltonian(
        nh3_molecule_2,
        active_electrons=active_electrons,
        active_orbitals=active_orbitals,
    )

    # --- Define ansätze for NH₃ ---
    ansatze_nh3 = [
        HartreeFockAnsatz(),
        SimpleAnsatz(),
        BalancedAnsatz(),
        ExpensiveAnsatz(),
        UCCSDAnsatz(),
    ]

    nh3_calc = MoleculeEnergyCalc(
        hamiltonians=[H1, H2],     # <-- Direct Hamiltonian VQE  #### New!
        molecules=None,            # <-- No geometry sweep needed
        ansatze=ansatze_nh3,
        max_iterations=40,
        n_layers_list=[1, 2],
        backend=ParallelSimulator(shots=4000),
    )

    nh3_calc.run_hamiltonian_vqe()
    print("\nNH₃ Summary:")
    nh3_calc.summary()
