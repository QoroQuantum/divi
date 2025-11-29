import numpy as np
import pennylane as qml
from pennylane import qchem

from divi.backends import ParallelSimulator
from divi.qprog import HartreeFockAnsatz, GenericLayerAnsatz, UCCSDAnsatz, VQE
from divi.qprog.optimizers import ScipyOptimizer, ScipyMethod
from divi.qprog.workflows import VQEHyperparameterSweep, MoleculeTransformer

from typing import Any



class MoleculeEnergyCalc:
    """
    Calculates ground state energies using VQE, 
    for geometry sweeps OR explicit Hamiltonians.
    """

    def __init__(self, n_electrons = None, molecules=None, bond_sweeps=None,
                 ansatze=None, n_layers_list=None,
                 hamiltonians=None, max_iterations=50, visualize=False):

        # Create results dictionary
        self.results = {}
        self.n_electrons = n_electrons
        self.molecules = molecules if molecules is not None else []
        self.bond_sweeps = bond_sweeps if bond_sweeps is not None else np.array([0.0])
        self.ansatze = ansatze if ansatze is not None else [HartreeFockAnsatz()]
        self.n_layers_list = n_layers_list if n_layers_list is not None else [1]
        self.hamiltonians = hamiltonians if hamiltonians is not None else []
        self.max_iterations = max_iterations
        self.visualize = visualize 

        # Always create backend internally
        self.backend = ParallelSimulator(shots=4000)

        # Shared optimizer
        self.optimizer = ScipyOptimizer(method=ScipyMethod.L_BFGS_B)

    # unified clone helper
    def _clone_with_layers(self, ansatz, n_layers):
        """
        Returns a NEW ansatz instance with n_layers (if supported).
        HF and UCCSD ignore n_layers.
        """
        if hasattr(ansatz, "n_layers"):
            return ansatz.__class__(
                gate_sequence=getattr(ansatz, "gate_sequence", None),
                entangler=getattr(ansatz, "entangler", None),
                entangling_layout=getattr(ansatz, "entangling_layout", None),
                n_layers=n_layers,
            )
        return ansatz  # HF, UCCSD


    def run_hamiltonian_vqe(self):
        """
        Hamiltonian VQE
        """
        print("\n=== Running VQE on Explicit Hamiltonians ===")
        self.results["hamiltonians"] = {}

        for h_idx, H in enumerate(self.hamiltonians):
            self.results["hamiltonians"][h_idx] = {}

            for n_layers in self.n_layers_list:
                print(f"\n--- Hamiltonian {h_idx+1}, n_layers={n_layers} ---")

                energies = []
                circuits = []
                n_electrons = self.n_electrons

                for ansatz in self.ansatze:
                    ans = self._clone_with_layers(ansatz, n_layers)

                    vqe = VQE(
                        H,
                        ansatz=ans,
                        backend=self.backend,
                        max_iterations=self.max_iterations,
                        n_electrons=n_electrons
                    )

                    vqe.run()

                    energies.append(vqe.best_loss)
                    circuits.append(vqe.total_circuit_count)

                self.results["hamiltonians"][h_idx][n_layers] = (
                    energies, circuits
                )


    def run_geometry_sweeps(self):
        """
        geometry sweep mode
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

                ans_list = [
                    self._clone_with_layers(a, n_layers) for a in self.ansatze
                ]

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
                
                if self.visualize:
                # Visualize the results for this layer depth
                # (e.g. one figure per depth)
                    sweep.visualize_results(graph_type="line")  # or "scatter"
        


    
    def summary(self):
        print("\n===== SUMMARY =====")
        for section, data in self.results.items():
            print(f"\n> {section.upper()}")
            print(data)



class HFLayerAnsatz(GenericLayerAnsatz):
    """
    GenericLayerAnsatz on top of a Hartree-Fock reference state.

    Usage:
        ansatz = HFLayerAnsatz(
            gate_sequence=[qml.RY, qml.RZ],
            entangler=qml.CNOT,
            entangling_layout="linear",
        )

        # later, when building:
        ops = ansatz.build(
            params,
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_electrons=n_electrons,  # or hf_state=...
        )
    """

    def build(
        self,
        params: Any,
        n_qubits: int,
        n_layers: int,
        **kwargs: Any,
    ) -> list[qml.operation.Operator]:
        # Option A: user directly passes a bitstring/array as hf_state
        hf_state = kwargs.pop("hf_state", None)

        # Option B: derive HF state from number of electrons
        if hf_state is None:
            n_electrons = kwargs.get("n_electrons", None)
            if n_electrons is None:
                raise ValueError(
                    "HFLayerAnsatz.build requires either `hf_state` or `n_electrons` "
                    "in kwargs."
                )
            hf_state = qchem.hf_state(n_electrons, n_qubits)

        wires = list(range(n_qubits))

        # 1) HF preparation as the very first operation
        operations: list[qml.operation.Operator] = [
            qml.BasisState(hf_state, wires=wires)
        ]

        # 2) All the usual layers from GenericLayerAnsatz
        layer_ops = super().build(params, n_qubits=n_qubits, n_layers=n_layers, **kwargs)

        return operations + layer_ops


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
    # 1. Minimalistic ansatz with only one Y-Rotation and a linear CNOT entangler
    minimal = HFLayerAnsatz(gate_sequence=[qml.RY],
            entangler=qml.CNOT,
            entangling_layout="linear",),
    # 2. Balanced ansatz with two Rotations (Y, Z) and a linear CNOT entangler
    balanced = HFLayerAnsatz(gate_sequence=[qml.RY, qml.RZ],
                             entangler=qml.CNOT,
                             entangling_layout="linear",),
    # 3. Expensive ansatz with two Rotations (Y, Z) and a linear CNOT entangler
    expensive = HFLayerAnsatz(gate_sequence=[qml.RY, qml.RZ],
                              entangler=qml.CNOT,
                              entangling_layout="all_to_all",)


    ansatze_nh3 = [
        HartreeFockAnsatz(),
        minimal,
        balanced,
        expensive,
        UCCSDAnsatz(),            
    ]

    nh3_calc = MoleculeEnergyCalc(
        hamiltonians=[H1, H2],     
        molecules=None,            
        ansatze=ansatze_nh3,
        max_iterations=40,
        n_layers_list=[1, 2],
        n_electrons=active_electrons,
    )

    nh3_calc.run_hamiltonian_vqe()
    print("\nNH₃ Summary:")
    nh3_calc.summary()
