# SPDX-FileCopyrightText: 2025 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from itertools import product

import numpy as np
import pennylane as qml
import pytest
from scipy.spatial.distance import pdist, squareform

from divi.qprog.algorithms import GenericLayerAnsatz, HartreeFockAnsatz, UCCSDAnsatz
from divi.qprog.optimizers import MonteCarloOptimizer
from divi.qprog.workflows import (
    MoleculeTransformer,
    VQEHyperparameterSweep,
    _vqe_sweep,
)
from tests.qprog.qprog_contracts import verify_basic_program_batch_behaviour


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

    def test_duplicate_atom_connectivity(self, h2_molecule):
        """Test ValueError for duplicate values in atom_connectivity."""
        with pytest.raises(ValueError, match="contains duplicate values"):
            MoleculeTransformer(
                base_molecule=h2_molecule,
                bond_modifiers=[1.1],
                atom_connectivity=[(0, 1), (0, 1)],  # Duplicate
            )

    def test_alignment_functionality(self, water_molecule):
        """Test that alignment is applied when alignment_atoms is specified."""
        bond_modifiers = [1.2]
        mt = MoleculeTransformer(
            base_molecule=water_molecule,
            bond_modifiers=bond_modifiers,
            atom_connectivity=[(0, 1), (0, 2)],
            bonds_to_transform=[(0, 1)],
            alignment_atoms=[0, 1],  # Align on first two atoms
        )

        variants = mt.generate()
        transformed_mol = variants[1.2]

        # The molecule should be generated successfully with alignment
        assert len(transformed_mol.symbols) == 3
        assert transformed_mol.coordinates.shape == (3, 3)


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
    ansatze = [HartreeFockAnsatz(), GenericLayerAnsatz([qml.RY])]
    optimizer = MonteCarloOptimizer(population_size=5, n_best_sets=2)
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
            (ansatz.name, modifier) in vqe_sweep.programs
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
        mock_program_1.losses_history = [{0: -1.2}]
        mock_program_1.best_loss = -1.2
        mock_program_2 = mocker.MagicMock()
        mock_program_2.losses_history = [{0: -1.1}]
        mock_program_2.best_loss = -1.1

        uccsd_instance = UCCSDAnsatz()
        generic_ry_instance = GenericLayerAnsatz([qml.RY])

        vqe_sweep.programs = {
            (uccsd_instance, 0.9): mock_program_1,
            (generic_ry_instance, 1.0): mock_program_2,
        }

        smallest_key, smallest_value = vqe_sweep.aggregate_results()

        assert smallest_key == (uccsd_instance, 0.9)
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
                mock_program.losses_history = [{0: -(modifier * 10 + ansatz_idx)}]
                mock_program.best_loss = -(modifier * 10 + ansatz_idx)
                mock_programs[(ansatz.name, modifier)] = mock_program
        vqe_sweep.programs = mock_programs

        vqe_sweep.visualize_results(graph_type="line")

        # Check that plot was called for each ansatz
        assert mock_plot.call_count == len(vqe_sweep.ansatze)

        # Construct expected calls
        call_hartree_fock = mocker.call(
            [0.9, 1.0, 1.1],
            [-9.0, -10.0, -11.0],
            label="HartreeFockAnsatz",
            color="blue",
        )
        call_ry = mocker.call(
            [0.9, 1.0, 1.1],
            [-10.0, -11.0, -12.0],
            label="GenericLayerAnsatz",
            color="g",
        )

        mock_plot.assert_has_calls([call_hartree_fock, call_ry], any_order=True)

    def test_visualize_results_with_invalid_graph_type(self, mocker, vqe_sweep):
        """Test that providing an invalid graph type raises a ValueError."""
        mocker.patch("divi.qprog.VQE")
        mock_show = mocker.patch("matplotlib.pyplot.show")

        mock_program = mocker.MagicMock()
        mock_program.losses_history = [{0: -1.0}]
        vqe_sweep.programs = {
            (
                vqe_sweep.ansatze[0],
                vqe_sweep.molecule_transformer.bond_modifiers[0],
            ): mock_program
        }

        with pytest.raises(ValueError, match="Invalid graph type"):
            vqe_sweep.visualize_results(graph_type="some_invalid_type")

        mock_show.assert_not_called()

    def test_visualize_results_scatter_plot(self, mocker, vqe_sweep):
        """Test scatter plot visualization functionality."""
        mocker.patch("divi.qprog.VQE")
        mock_scatter = mocker.patch("matplotlib.pyplot.scatter")
        mocker.patch("matplotlib.pyplot.show")
        mocker.patch("matplotlib.pyplot.legend")
        mocker.patch("matplotlib.pyplot.xlabel")
        mocker.patch("matplotlib.pyplot.ylabel")

        # Setup mock programs
        mock_programs = {}
        for ansatz_idx, ansatz in enumerate(vqe_sweep.ansatze):
            for modifier in vqe_sweep.molecule_transformer.bond_modifiers:
                mock_program = mocker.MagicMock()
                mock_program.best_loss = -(modifier * 10 + ansatz_idx)
                mock_programs[(ansatz.name, modifier)] = mock_program
        vqe_sweep.programs = mock_programs

        vqe_sweep.visualize_results(graph_type="scatter")

        # Check that scatter was called for each ansatz
        assert mock_scatter.call_count == len(vqe_sweep.ansatze)

    def test_visualize_results_missing_programs(self, mocker, vqe_sweep):
        """Test visualization behavior with missing programs."""
        mocker.patch("divi.qprog.VQE")
        mock_scatter = mocker.patch("matplotlib.pyplot.scatter")
        mocker.patch("matplotlib.pyplot.show")
        mocker.patch("matplotlib.pyplot.legend")
        mocker.patch("matplotlib.pyplot.xlabel")
        mocker.patch("matplotlib.pyplot.ylabel")

        # Setup mock programs with some missing
        mock_programs = {}
        for ansatz_idx, ansatz in enumerate(vqe_sweep.ansatze):
            # Only add programs for first two modifiers, skip the third
            for modifier in vqe_sweep.molecule_transformer.bond_modifiers[:2]:
                mock_program = mocker.MagicMock()
                mock_program.best_loss = -(modifier * 10 + ansatz_idx)
                mock_programs[(ansatz.name, modifier)] = mock_program
        vqe_sweep.programs = mock_programs

        # This should not raise an error, just skip missing programs
        vqe_sweep.visualize_results(graph_type="scatter")

        # Should still call scatter for available programs
        assert mock_scatter.call_count == len(vqe_sweep.ansatze)

    def test_visualize_results_with_executor(self, mocker, vqe_sweep):
        """Test visualization calls join() when executor is present."""
        mocker.patch("divi.qprog.VQE")
        mock_join = mocker.patch.object(vqe_sweep, "join")
        mocker.patch("matplotlib.pyplot.plot")
        mocker.patch("matplotlib.pyplot.show")
        mocker.patch("matplotlib.pyplot.legend")
        mocker.patch("matplotlib.pyplot.xlabel")
        mocker.patch("matplotlib.pyplot.ylabel")

        # Set up a mock executor
        mock_executor = mocker.MagicMock()
        vqe_sweep._executor = mock_executor

        # Setup mock programs
        mock_programs = {}
        for ansatz_idx, ansatz in enumerate(vqe_sweep.ansatze):
            for modifier in vqe_sweep.molecule_transformer.bond_modifiers:
                mock_program = mocker.MagicMock()
                mock_program.best_loss = -(modifier * 10 + ansatz_idx)
                mock_programs[(ansatz.name, modifier)] = mock_program
        vqe_sweep.programs = mock_programs

        vqe_sweep.visualize_results(graph_type="line")

        # Should call join() when executor is present
        mock_join.assert_called_once()


class TestHelperFunctions:
    """Tests for helper functions in _vqe_sweep module."""

    def test_safe_normalize_edge_cases(self):
        """Test _safe_normalize with edge cases."""
        from divi.qprog.workflows._vqe_sweep import _safe_normalize

        # Test with zero vector
        zero_vec = np.array([0.0, 0.0, 0.0])
        result = _safe_normalize(zero_vec)
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

        # Test with very small vector
        small_vec = np.array([1e-8, 1e-8, 1e-8])
        result = _safe_normalize(small_vec)
        np.testing.assert_array_almost_equal(result, expected)

        # Test with custom fallback
        custom_fallback = np.array([0.0, 1.0, 0.0])
        result = _safe_normalize(zero_vec, fallback=custom_fallback)
        np.testing.assert_array_almost_equal(result, custom_fallback)

        # Test with normal vector
        normal_vec = np.array([3.0, 4.0, 0.0])
        result = _safe_normalize(normal_vec)
        expected = np.array([0.6, 0.8, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_compute_angle_edge_cases(self):
        """Test _compute_angle with edge cases."""
        from divi.qprog.workflows._vqe_sweep import _compute_angle

        # Test with parallel vectors
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([2.0, 0.0, 0.0])
        angle = _compute_angle(v1, v2)
        assert np.isclose(angle, 0.0)

        # Test with antiparallel vectors
        v2 = np.array([-2.0, 0.0, 0.0])
        angle = _compute_angle(v1, v2)
        assert np.isclose(angle, 180.0)

        # Test with perpendicular vectors
        v2 = np.array([0.0, 1.0, 0.0])
        angle = _compute_angle(v1, v2)
        assert np.isclose(angle, 90.0)

        # Test with very small vectors
        v1 = np.array([1e-8, 1e-8, 0.0])
        v2 = np.array([1e-8, 0.0, 0.0])
        angle = _compute_angle(v1, v2)
        assert 0 <= angle <= 180

    def test_compute_dihedral_edge_cases(self):
        """Test _compute_dihedral with edge cases."""
        from divi.qprog.workflows._vqe_sweep import _compute_dihedral

        # Test with collinear vectors
        b0 = np.array([1.0, 0.0, 0.0])
        b1 = np.array([2.0, 0.0, 0.0])
        b2 = np.array([3.0, 0.0, 0.0])
        dihedral = _compute_dihedral(b0, b1, b2)
        assert np.isclose(dihedral, 0.0)

        # Test with very small vectors
        b0 = np.array([1e-8, 1e-8, 0.0])
        b1 = np.array([1e-8, 0.0, 0.0])
        b2 = np.array([0.0, 1e-8, 0.0])
        dihedral = _compute_dihedral(b0, b1, b2)
        assert np.isclose(dihedral, 0.0)

        # Test with normal case
        b0 = np.array([1.0, 0.0, 0.0])
        b1 = np.array([0.0, 1.0, 0.0])
        b2 = np.array([0.0, 0.0, 1.0])
        dihedral = _compute_dihedral(b0, b1, b2)
        assert isinstance(dihedral, float)

    def test_find_refs_edge_cases(self):
        """Test _find_refs with edge cases."""
        from divi.qprog.workflows._vqe_sweep import _find_refs

        # Test with no valid references
        adj = [[2], [2], [0, 1]]
        placed = {0, 1}
        gp, ggp = _find_refs(adj, placed, 1, 2)
        assert gp is None  # No valid references
        assert ggp is None

        # Test with valid references
        adj = [[1], [0, 2], [1]]
        placed = {0, 1, 2}
        gp, ggp = _find_refs(adj, placed, 1, 2)
        assert gp == 0  # 0 is in adj[1] and in placed
        assert ggp is None  # No grandparent found


class TestZMatrixConversion:
    """Tests for Z-matrix conversion functions."""

    def test_cartesian_to_zmatrix_empty_coords(self):
        """Test _cartesian_to_zmatrix with empty coordinates."""
        from divi.qprog.workflows._vqe_sweep import _cartesian_to_zmatrix

        empty_coords = np.array([]).reshape(0, 3)
        connectivity = []

        with pytest.raises(ValueError, match="Cannot convert empty coordinate array"):
            _cartesian_to_zmatrix(empty_coords, connectivity)

    def test_cartesian_to_zmatrix_single_atom(self):
        """Test _cartesian_to_zmatrix with single atom."""
        from divi.qprog.workflows._vqe_sweep import _cartesian_to_zmatrix

        coords = np.array([[0.0, 0.0, 0.0]])
        connectivity = []

        zmatrix = _cartesian_to_zmatrix(coords, connectivity)
        assert len(zmatrix) == 1
        assert zmatrix[0].bond_ref is None
        assert zmatrix[0].angle_ref is None
        assert zmatrix[0].dihedral_ref is None

    def test_zmatrix_to_cartesian_edge_cases(self):
        """Test _zmatrix_to_cartesian with edge cases."""
        from divi.qprog.workflows._vqe_sweep import _zmatrix_to_cartesian, _ZMatrixEntry

        # Test with empty Z-matrix
        empty_zmatrix = []
        coords = _zmatrix_to_cartesian(empty_zmatrix)
        assert coords.shape == (0, 3)

        # Test with single atom
        single_atom_zmatrix = [_ZMatrixEntry(None, None, None, None, None, None)]
        coords = _zmatrix_to_cartesian(single_atom_zmatrix)
        assert coords.shape == (1, 3)
        np.testing.assert_array_almost_equal(coords[0], [0.0, 0.0, 0.0])

        # Test with two atoms
        two_atom_zmatrix = [
            _ZMatrixEntry(None, None, None, None, None, None),
            _ZMatrixEntry(0, None, None, 1.0, None, None),
        ]
        coords = _zmatrix_to_cartesian(two_atom_zmatrix)
        assert coords.shape == (2, 3)
        np.testing.assert_array_almost_equal(coords[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(coords[1], [1.0, 0.0, 0.0])

    def test_cartesian_to_zmatrix_with_dihedral(self):
        """Test Z-matrix conversion with 4+ atoms requiring dihedral angles."""
        from divi.qprog.workflows._vqe_sweep import _cartesian_to_zmatrix

        # Create a 4-atom chain molecule to ensure dihedral references are found
        coords = np.array(
            [
                [0.0, 0.0, 0.0],  # C1 at origin
                [1.0, 0.0, 0.0],  # C2 along x-axis
                [2.0, 0.0, 0.0],  # C3 along x-axis
                [2.0, 1.0, 0.0],  # C4 with dihedral angle
            ]
        )
        connectivity = [(0, 1), (1, 2), (2, 3)]  # Chain connectivity

        zmatrix = _cartesian_to_zmatrix(coords, connectivity)

        # Check that dihedral angles are calculated for 4th atom
        assert len(zmatrix) == 4
        assert (
            zmatrix[3].dihedral is not None
        )  # This should trigger dihedral calculation
        assert zmatrix[3].bond_ref == 2
        assert zmatrix[3].angle_ref == 1
        assert zmatrix[3].dihedral_ref == 0

    def test_zmatrix_to_cartesian_four_atoms(self):
        """Test Z-matrix to Cartesian conversion with 4+ atoms."""
        from divi.qprog.workflows._vqe_sweep import _zmatrix_to_cartesian, _ZMatrixEntry

        # Create Z-matrix with 4 atoms to test the 4+ atom placement loop
        zmatrix = [
            _ZMatrixEntry(None, None, None, None, None, None),  # Atom 0
            _ZMatrixEntry(0, None, None, 1.0, None, None),  # Atom 1
            _ZMatrixEntry(0, 1, None, 1.0, 90.0, None),  # Atom 2
            _ZMatrixEntry(0, 1, 2, 1.0, 90.0, 0.0),  # Atom 3 with dihedral
        ]

        coords = _zmatrix_to_cartesian(zmatrix)

        # Verify 4 atoms are placed correctly
        assert coords.shape == (4, 3)
        assert np.all(np.isfinite(coords))  # All coordinates should be finite

        # Check that atoms are placed at expected positions
        np.testing.assert_array_almost_equal(coords[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(coords[1], [1.0, 0.0, 0.0])
        # Atoms 2 and 3 should be placed using the 4+ atom logic

    def test_zmatrix_to_cartesian_with_none_references(self):
        """Test Z-matrix conversion with None references in 4+ atom case."""
        from divi.qprog.workflows._vqe_sweep import _zmatrix_to_cartesian, _ZMatrixEntry

        # Test edge case where some references are None
        zmatrix = [
            _ZMatrixEntry(None, None, None, None, None, None),  # Atom 0
            _ZMatrixEntry(0, None, None, 1.0, None, None),  # Atom 1
            _ZMatrixEntry(0, 1, None, 1.0, 90.0, None),  # Atom 2
            _ZMatrixEntry(None, None, None, 1.0, 90.0, 0.0),  # Atom 3 with None refs
        ]

        coords = _zmatrix_to_cartesian(zmatrix)

        # Should handle None references gracefully
        assert coords.shape == (4, 3)
        assert np.all(np.isfinite(coords))

    def test_zmatrix_to_cartesian_zero_angles(self):
        """Test Z-matrix conversion with zero angles and dihedrals."""
        from divi.qprog.workflows._vqe_sweep import _zmatrix_to_cartesian, _ZMatrixEntry

        # Test with zero angles and dihedrals
        zmatrix = [
            _ZMatrixEntry(None, None, None, None, None, None),  # Atom 0
            _ZMatrixEntry(0, None, None, 1.0, None, None),  # Atom 1
            _ZMatrixEntry(0, 1, None, 1.0, 0.0, None),  # Atom 2 with zero angle
            _ZMatrixEntry(
                0, 1, 2, 1.0, 0.0, 0.0
            ),  # Atom 3 with zero angle and dihedral
        ]

        coords = _zmatrix_to_cartesian(zmatrix)

        # Should handle zero angles correctly
        assert coords.shape == (4, 3)
        assert np.all(np.isfinite(coords))


class TestBondTransformation:
    """Tests for bond transformation functions."""

    def test_transform_bonds_zero_length_error(self):
        """Test _transform_bonds raises RuntimeError for zero bond length."""
        from divi.qprog.workflows._vqe_sweep import _transform_bonds, _ZMatrixEntry

        zmatrix = [
            _ZMatrixEntry(None, None, None, None, None, None),
            _ZMatrixEntry(0, None, None, 1.0, None, None),
        ]
        bonds_to_transform = [(0, 1)]

        # Test scale mode that results in zero length
        with pytest.raises(RuntimeError, match="New bond length can't be zero"):
            _transform_bonds(zmatrix, bonds_to_transform, 0.0, "scale")

        # Test delta mode that results in zero length
        with pytest.raises(RuntimeError, match="New bond length can't be zero"):
            _transform_bonds(zmatrix, bonds_to_transform, -1.0, "delta")


class TestKabschAlignment:
    """Tests for Kabsch alignment algorithm."""

    def test_kabsch_align_identical_points(self):
        """Test _kabsch_align with identical point sets."""
        from divi.qprog.workflows._vqe_sweep import _kabsch_align

        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        aligned = _kabsch_align(points, points)
        np.testing.assert_array_almost_equal(aligned, points)

    def test_kabsch_align_translation(self):
        """Test _kabsch_align with translated point sets."""
        from divi.qprog.workflows._vqe_sweep import _kabsch_align

        P = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        Q = P + np.array([10.0, 20.0, 30.0])  # Translation
        aligned = _kabsch_align(P, Q)
        np.testing.assert_array_almost_equal(aligned, Q)

    def test_kabsch_align_rotation(self):
        """Test _kabsch_align with rotated point sets."""
        from divi.qprog.workflows._vqe_sweep import _kabsch_align

        # Simple rotation around Z-axis
        angle = np.pi / 4
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])

        P = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        Q = P @ R.T
        aligned = _kabsch_align(P, Q)

        # Check that the alignment produces reasonable results
        assert aligned.shape == Q.shape
        # Verify that the function runs without error and produces output
        assert not np.allclose(aligned, P)  # Should be different from original
        assert np.all(np.isfinite(aligned))  # Should be finite values

    def test_kabsch_align_with_reference_atoms(self):
        """Test _kabsch_align with reference atom subset."""
        from divi.qprog.workflows._vqe_sweep import _kabsch_align

        P = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        Q = P + np.array([10.0, 20.0, 30.0])

        # Align only first two atoms
        aligned = _kabsch_align(P, Q, reference_atoms_idx=slice(0, 2))

        # First two atoms should be aligned
        np.testing.assert_array_almost_equal(aligned[:2], Q[:2])
        # Third atom should be transformed but not necessarily aligned
        assert aligned.shape == P.shape


class TestMathematicalCorrectnessIssues:
    """Test cases for confirmed mathematical issues that were fixed."""

    def test_kabsch_reflection_handling_fixed(self):
        """Test that Kabsch algorithm finds the best rotation, not a reflection."""
        from divi.qprog.workflows._vqe_sweep import _kabsch_align

        # Create a right-handed coordinate system for P (non-planar)
        P = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )

        # Create Q as a reflection of P (a left-handed system)
        Q = np.copy(P)
        Q[:, 2] *= -1.0  # Reflection across xy plane

        # Align P to Q
        P_aligned = _kabsch_align(P, Q)

        # A naive Kabsch implementation might return a rotation matrix with det=-1,
        # which is a reflection. This would make P_aligned very close to Q.
        # The corrected implementation should find the best *proper rotation* (det=+1),
        # which will not perfectly align P to the reflected Q.
        # Therefore, we assert that the aligned points are NOT close to the reflected target points.
        is_close = np.allclose(P_aligned, Q, atol=1e-7)
        assert (
            not is_close
        ), "Kabsch alignment should produce a proper rotation, not a reflection."

        # For a proper rotation, the RMSD should be non-zero in this case.
        rmsd = np.sqrt(np.mean(np.sum((P_aligned - Q) ** 2, axis=1)))
        assert (
            rmsd > 0.1
        ), f"Expected non-zero RMSD when avoiding reflection, but got {rmsd}"

    def test_physical_validation_fixed(self):
        """Test that zero bond lengths are now properly rejected."""
        from divi.qprog.workflows._vqe_sweep import _zmatrix_to_cartesian, _ZMatrixEntry

        # Create Z-matrix with physically impossible geometry
        # Bond length of 0.0 should be invalid
        zmatrix = [
            _ZMatrixEntry(None, None, None, None, None, None),
            _ZMatrixEntry(0, None, None, 0.0, None, None),  # Zero bond length!
        ]

        # This should raise an error for physically impossible geometry
        with pytest.raises(ValueError, match="Bond length for atom 1 must be positive"):
            _zmatrix_to_cartesian(zmatrix)
