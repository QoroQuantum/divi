import pennylane as qml

from pennylane import numpy as np
from qprog.quantum_program import QuantumProgram


class VQE(QuantumProgram):
    def __init__(self, symbols, bond_lengths, coordinate_structure, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.symbols = symbols
        self.bond_lengths = bond_lengths
        self.coordinate_structure = coordinate_structure
        assert len(self.coordinate_structure) == len(
            self.symbols), "The number of symbols must match the number of coordinates"
        self.hamiltonian_ops = self._generate_hamiltonian_operations()

    def _generate_hamiltonian_operations(self):
        hamiltonian_ops = []
        for bond_length in self.bond_lengths:
            # Generate the Hamiltonian for the given bond length
            coordinates = []
            for coord_str in self.coordinate_structure:
                coordinates.append(
                    [coord_str[0] * bond_length, coord_str[1] * bond_length, coord_str[2] * bond_length])
            coordinates = np.array(coordinates, requires_grad=False)
            molecule = qml.qchem.Molecule(self.symbols, coordinates)
            hamiltonian = qml.qchem.molecular_hamiltonian(molecule)
            hamiltonian_ops.append(hamiltonian)
        return hamiltonian_ops
