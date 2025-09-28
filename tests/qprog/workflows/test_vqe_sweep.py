# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from itertools import product

import numpy as np
import pennylane as qml
import pytest
from qprog_contracts import verify_basic_program_batch_behaviour
from scipy.spatial.distance import pdist, squareform

from divi.qprog.algorithms import RYAnsatz, UCCSDAnsatz
from divi.qprog.optimizers import MonteCarloOptimizer
from divi.qprog.workflows import (
    MoleculeTransformer,
    VQEHyperparameterSweep,
    _vqe_sweep,
)


@pytest.fixture
def h2_molecule():
    """Fixture for a simple H2 molecule with a bond length of 0.74 Å."""
    symbols = ["H", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    return qml.qchem.Molecule(symbols, coordinates)


@pytest.fixture
def water_molecule():
    """Fixture for a water molecule (H2O), which has a non-linear structure."""
    symbols = ["O", "H", "H"]
    # Standard coordinates for H2O with the Oxygen atom at the origin
    coordinates = np.array(
        [[0.0000, 0.0000, 0.0000], [0.757, 0.586, 0.0000], [-0.757, 0.586, 0.0000]]
    )
    return qml.qchem.Molecule(symbols, coordinates)


def get_pairwise_distances(molecule):
    """Helper function to calculate a symmetric matrix of all pairwise atomic distances."""
    return squareform(pdist(molecule.coordinates))


class TestMoleculeTransformerValidation:
    """Tests for the validation logic in MoleculeTransformer's __post_init__."""

    def test_successful_initialization(self, h2_molecule):
        """Test that the class can be initialized without errors with valid inputs."""
        transformer = MoleculeTransformer(
            base_molecule=h2_molecule,
            bond_modifiers=[0.9, 1.1],
            atom_connectivity=[(0, 1)],
            bonds_to_transform=[(0, 1)],
            alignment_atoms=[0, 1],
        )

        assert transformer.base_molecule == h2_molecule
        assert transformer.bond_modifiers == [0.9, 1.1]
        assert transformer.atom_connectivity == [(0, 1)]
        assert transformer.bonds_to_transform == [(0, 1)]
        assert transformer.alignment_atoms == [0, 1]
        assert transformer._mode == "scale"

    def test_invalid_base_molecule_type(self):
        """Test that a ValueError is raised for an invalid base_molecule type."""
        with pytest.raises(
            ValueError, match="is expected to be a Pennylane `Molecule` instance"
        ):
            MoleculeTransformer(base_molecule="not_a_molecule", bond_modifiers=[1.1])

    def test_non_numeric_bond_modifiers(self, h2_molecule):
        """Test ValueError for non-numeric values in bond_modifiers."""
        with pytest.raises(ValueError, match="should be a sequence of floats"):
            MoleculeTransformer(base_molecule=h2_molecule, bond_modifiers=[1.0, "a"])

    def test_duplicate_bond_modifiers(self, h2_molecule):
        """Test ValueError for duplicate values in bond_modifiers."""
        with pytest.raises(ValueError, match="contains duplicate values"):
            MoleculeTransformer(base_molecule=h2_molecule, bond_modifiers=[1.1, 1.1])

    def test_mode_detection(self, h2_molecule):
        """Test that the transformation mode is correctly detected."""
        # All positive values should result in 'scale' mode
        mt_scale = MoleculeTransformer(
            base_molecule=h2_molecule, bond_modifiers=[0.9, 1.2]
        )
        assert mt_scale._mode == "scale"

        # A zero value should trigger 'delta' mode
        mt_delta_zero = MoleculeTransformer(
            base_molecule=h2_molecule, bond_modifiers=[0.0, 0.2]
        )
        assert mt_delta_zero._mode == "delta"

        # A negative value should trigger 'delta' mode
        mt_delta_neg = MoleculeTransformer(
            base_molecule=h2_molecule, bond_modifiers=[-0.1, 0.1]
        )
        assert mt_delta_neg._mode == "delta"

    def test_default_atom_connectivity(self, h2_molecule):
        """Test that atom_connectivity defaults to a simple chain if not provided."""
        mt = MoleculeTransformer(base_molecule=h2_molecule, bond_modifiers=[1.1])
        assert mt.atom_connectivity == ((0, 1),)

    def test_out_of_bounds_atom_connectivity(self, h2_molecule):
        """Test ValueError for out-of-bounds indices in atom_connectivity."""
        with pytest.raises(ValueError, match="atom indices"):
            MoleculeTransformer(
                base_molecule=h2_molecule,
                bond_modifiers=[1.1],
                atom_connectivity=[(0, 2)],
            )

    def test_default_bonds_to_transform(self, h2_molecule):
        """Test that bonds_to_transform defaults to the full atom_connectivity list."""
        connectivity = [(0, 1)]
        mt = MoleculeTransformer(
            base_molecule=h2_molecule,
            bond_modifiers=[1.1],
            atom_connectivity=connectivity,
        )
        assert mt.bonds_to_transform == connectivity

    def test_empty_bonds_to_transform(self, h2_molecule):
        """Test ValueError if bonds_to_transform is empty."""
        with pytest.raises(ValueError, match="`bonds_to_transform` cannot be empty"):
            MoleculeTransformer(
                base_molecule=h2_molecule,
                bond_modifiers=[1.1],
                bonds_to_transform=[],
            )

    def test_bonds_to_transform_not_subset(self, h2_molecule):
        """Test ValueError if bonds_to_transform is not a subset of atom_connectivity."""
        with pytest.raises(ValueError, match="is not a subset of"):
            MoleculeTransformer(
                base_molecule=h2_molecule,
                bond_modifiers=[1.1],
                atom_connectivity=[(0, 1)],
                bonds_to_transform=[(0, 2)],  # This bond is not in connectivity
            )

    def test_out_of_bounds_alignment_atoms(self, h2_molecule):
        """Test ValueError for out-of-bounds indices in alignment_atoms."""
        with pytest.raises(ValueError, match="need to be in range"):
            MoleculeTransformer(
                base_molecule=h2_molecule, bond_modifiers=[1.1], alignment_atoms=[0, 2]
            )


class TestMoleculeTransformerGeneration:
    """Tests for the molecule generation logic in MoleculeTransformer."""

    def test_generate_scale_mode(self, water_molecule):
        """Test molecule generation in 'scale' mode correctly scales bond lengths."""
        bond_modifiers = [0.5, 1.5]
        mt = MoleculeTransformer(
            base_molecule=water_molecule,
            bond_modifiers=bond_modifiers,
            atom_connectivity=[(0, 1), (0, 2)],
            bonds_to_transform=[(0, 1)],  # Only transform the first O-H bond
        )
        assert mt._mode == "scale"

        variants = mt.generate()

        assert set(variants.keys()) == set(bond_modifiers)
        original_dist = np.linalg.norm(
            water_molecule.coordinates[1] - water_molecule.coordinates[0]
        )

        dist_0_5 = np.linalg.norm(
            variants[0.5].coordinates[1] - variants[0.5].coordinates[0]
        )
        assert np.isclose(dist_0_5, original_dist * 0.5)

        dist_1_5 = np.linalg.norm(
            variants[1.5].coordinates[1] - variants[1.5].coordinates[0]
        )
        assert np.isclose(dist_1_5, original_dist * 1.5)

    def test_generate_delta_mode(self, water_molecule):
        """Test molecule generation in 'delta' mode correctly adds to bond lengths."""
        bond_modifiers = [-0.1, 0.2]
        mt = MoleculeTransformer(
            base_molecule=water_molecule,
            bond_modifiers=bond_modifiers,
            atom_connectivity=[(0, 1), (0, 2)],
            bonds_to_transform=[(0, 1)],  # Only transform the first O-H bond
        )
        assert mt._mode == "delta"
        variants = mt.generate()

        assert set(variants.keys()) == set(bond_modifiers)
        original_dist = np.linalg.norm(
            water_molecule.coordinates[1] - water_molecule.coordinates[0]
        )

        dist_neg_0_1 = np.linalg.norm(
            variants[-0.1].coordinates[1] - variants[-0.1].coordinates[0]
        )
        assert np.isclose(dist_neg_0_1, original_dist - 0.1)

        dist_pos_0_2 = np.linalg.norm(
            variants[0.2].coordinates[1] - variants[0.2].coordinates[0]
        )
        assert np.isclose(dist_pos_0_2, original_dist + 0.2)

    def test_generate_handles_identity_transforms(self, mocker, water_molecule):
        """Test that a delta modifier of 0.0 and a scale modifier of 1
        results in the original coordinates."""
        spy = mocker.spy(_vqe_sweep, "_transform_bonds")

        mt = MoleculeTransformer(
            base_molecule=water_molecule, bond_modifiers=[0.0, 0.1]
        )
        variants = mt.generate()
        assert 0.0 in variants
        assert np.allclose(variants[0.0].coordinates, water_molecule.coordinates)
        assert spy.call_count == 1

        mt = MoleculeTransformer(
            base_molecule=water_molecule, bond_modifiers=[1.0, 1.5]
        )
        variants = mt.generate()
        assert 1.0 in variants
        assert np.allclose(variants[1.0].coordinates, water_molecule.coordinates)
        assert spy.call_count == 2

    def test_bond_length_cannot_be_zero(self, water_molecule):
        """Test RuntimeError when a transformation results in a zero bond length."""
        original_dist = np.linalg.norm(
            water_molecule.coordinates[1] - water_molecule.coordinates[0]
        )
        mt_delta = MoleculeTransformer(
            base_molecule=water_molecule,
            bond_modifiers=[-original_dist],
            atom_connectivity=[(0, 1), (0, 2)],
            bonds_to_transform=[(0, 1)],
        )
        with pytest.raises(RuntimeError, match="New bond length can't be zero"):
            mt_delta.generate()

    def test_transformation_propagates_correctly_on_water(self, water_molecule):
        """
        Test that transforming one bond in a non-linear molecule (H2O) correctly
        updates other pairwise distances while leaving untransformed bonds unchanged.
        """
        scale_factor = 1.5
        mt = MoleculeTransformer(
            base_molecule=water_molecule,
            bond_modifiers=[scale_factor],
            atom_connectivity=[(0, 1), (0, 2)],  # O-H1 and O-H2 bonds
            bonds_to_transform=[(0, 1)],  # Only stretch the O-H1 bond
        )
        variants = mt.generate()
        transformed_mol = variants[scale_factor]

        # Get original and new pairwise distances
        original_distances = get_pairwise_distances(water_molecule)
        transformed_distances = get_pairwise_distances(transformed_mol)

        # 1. Check the bond that was transformed (O-H1, indices 0-1)
        original_OH1_dist = original_distances[0, 1]
        transformed_OH1_dist = transformed_distances[0, 1]
        assert np.isclose(transformed_OH1_dist, original_OH1_dist * scale_factor)

        # 2. Check the bond that was NOT transformed (O-H2, indices 0-2)
        original_OH2_dist = original_distances[0, 2]
        transformed_OH2_dist = transformed_distances[0, 2]
        assert np.isclose(transformed_OH2_dist, original_OH2_dist)

        # 3. Check the distance between atoms not directly bonded (H1-H2, indices 1-2)
        # This distance should have changed because H1 was moved.
        original_HH_dist = original_distances[1, 2]
        transformed_HH_dist = transformed_distances[1, 2]
        assert not np.isclose(transformed_HH_dist, original_HH_dist)


@pytest.fixture
def vqe_sweep(default_test_simulator, h2_molecule):
    """Fixture to create a VQEHyperparameterSweep instance with the new interface."""
    bond_modifiers = [0.9, 1.0, 1.1]
    ansatze = [UCCSDAnsatz(), RYAnsatz()]
    optimizer = MonteCarloOptimizer(n_param_sets=10, n_best_sets=3)
    max_iterations = 10

    transformer = MoleculeTransformer(
        base_molecule=h2_molecule,
        bond_modifiers=bond_modifiers,
    )

    return VQEHyperparameterSweep(
        ansatze=ansatze,
        molecule_transformer=transformer,
        optimizer=optimizer,
        max_iterations=max_iterations,
        backend=default_test_simulator,
    )


class TestVQEHyperparameterSweep:
    """A test class to group all tests for the VQEHyperparameterSweep."""

    def test_verify_basic_behaviour(self, mocker, vqe_sweep):
        """Test that the sweep conforms to basic batch program behavior."""
        verify_basic_program_batch_behaviour(mocker, vqe_sweep)

    def test_correct_number_of_programs_created(self, mocker, vqe_sweep):
        """Test that the correct number of VQE programs are instantiated with the correct parameters."""
        mocker.patch("divi.qprog.VQE")
        bond_modifiers = vqe_sweep.molecule_transformer.bond_modifiers

        vqe_sweep.create_programs()

        assert len(vqe_sweep.programs) == len(bond_modifiers) * len(vqe_sweep.ansatze)

        assert all(
            (ansatz, modifier) in vqe_sweep.programs
            for ansatz, modifier in product(vqe_sweep.ansatze, bond_modifiers)
        )

        for program in vqe_sweep.programs.values():
            assert isinstance(program.optimizer, MonteCarloOptimizer)
            assert program.max_iterations == 10
            assert program.backend.shots == 5000
            assert program.molecule.symbols == ["H", "H"]

    def test_results_aggregated_correctly(self, mocker, vqe_sweep):
        """Test that results from multiple VQE runs are aggregated to find the minimum energy."""
        mocker.patch("divi.qprog.VQE")

        mock_program_1 = mocker.MagicMock()
        mock_program_1.losses = [{0: -1.2}]
        mock_program_2 = mocker.MagicMock()
        mock_program_2.losses = [{0: -1.1}]

        vqe_sweep.programs = {
            (UCCSDAnsatz, 0.9): mock_program_1,
            (RYAnsatz, 1.0): mock_program_2,
        }

        smallest_key, smallest_value = vqe_sweep.aggregate_results()

        assert smallest_key == (UCCSDAnsatz, 0.9)
        assert smallest_value == -1.2

    def test_visualize_results_line_plot_data(self, mocker, vqe_sweep):
        """Test that the line plot visualization is called with the correct data."""
        mocker.patch("divi.qprog.VQE")
        mock_plot = mocker.patch("matplotlib.pyplot.plot")
        mocker.patch("matplotlib.pyplot.show")
        mocker.patch("matplotlib.pyplot.legend")
        mocker.patch("matplotlib.pyplot.xlabel")
        mocker.patch("matplotlib.pyplot.ylabel")

        # Setup mock programs with predictable energy values
        # Energy = -(modifier * 10 + ansatz_index)
        mock_programs = {}
        for ansatz_idx, ansatz in enumerate(vqe_sweep.ansatze):
            for modifier in vqe_sweep.molecule_transformer.bond_modifiers:
                mock_program = mocker.MagicMock()
                mock_program.losses = [{0: -(modifier * 10 + ansatz_idx)}]
                mock_programs[(ansatz, modifier)] = mock_program
        vqe_sweep.programs = mock_programs

        vqe_sweep.visualize_results(graph_type="line")

        # Check that plot was called for each ansatz
        assert mock_plot.call_count == len(vqe_sweep.ansatze)

        # Construct expected calls
        call_uccsd = mocker.call(
            [0.9, 1.0, 1.1], [-9.0, -10.0, -11.0], label="UCCSDAnsatz", color="blue"
        )
        call_ry = mocker.call(
            [0.9, 1.0, 1.1], [-10.0, -11.0, -12.0], label="RYAnsatz", color="g"
        )

        mock_plot.assert_has_calls([call_uccsd, call_ry], any_order=True)

    def test_visualize_results_with_invalid_graph_type(self, mocker, vqe_sweep):
        """Test that providing an invalid graph type raises a ValueError."""
        mocker.patch("divi.qprog.VQE")
        mock_show = mocker.patch("matplotlib.pyplot.show")

        mock_program = mocker.MagicMock()
        mock_program.losses = [{0: -1.0}]
        vqe_sweep.programs = {
            (
                vqe_sweep.ansatze[0],
                vqe_sweep.molecule_transformer.bond_modifiers[0],
            ): mock_program
        }

        with pytest.raises(ValueError, match="Invalid graph type"):
            vqe_sweep.visualize_results(graph_type="some_invalid_type")

        mock_show.assert_not_called()
