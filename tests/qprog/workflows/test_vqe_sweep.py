# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

from itertools import product

import numpy as np
import pytest
from qiskit.circuit.library import RYGate
from qiskit.quantum_info import SparsePauliOp
from scipy.spatial.distance import pdist, squareform

from divi.qprog import ReportingLevel
from divi.qprog.algorithms import GenericLayerAnsatz, HartreeFockAnsatz, UCCSDAnsatz
from divi.qprog.checkpointing import CheckpointConfig
from divi.qprog.optimizers import MonteCarloOptimizer, SPSAOptimizer
from divi.qprog.problems import HamiltonianProblem
from divi.qprog.workflows import (
    MoleculeTransformer,
    VQEHyperparameterSweep,
    _vqe_sweep,
)
from divi.qprog.workflows._vqe_sweep import (
    _cartesian_to_zmatrix,
    _compute_angle,
    _kabsch_align,
    _safe_normalize,
    _transform_bonds,
    _zmatrix_to_cartesian,
    _ZMatrixEntry,
)
from tests.qprog._program_contracts import verify_basic_program_ensemble_behaviour


@pytest.fixture
def h2_molecule(qp):
    """Fixture for a simple H2 molecule with a bond length of 0.74 Å."""
    symbols = ["H", "H"]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    return qp.qchem.Molecule(symbols, coordinates)


@pytest.fixture
def gto():
    """PySCF's molecule builder; skips without the ``chem`` extra."""
    return pytest.importorskip("pyscf.gto")


@pytest.fixture
def scf():
    """PySCF's mean-field solvers; skips without the ``chem`` extra."""
    return pytest.importorskip("pyscf.scf")


@pytest.fixture
def pyscf_h2_molecule(gto):
    """H2 at a 1.4 Bohr bond length, as a PySCF molecule."""
    return gto.M(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", unit="Bohr")


@pytest.fixture
def water_molecule(qp):
    """Fixture for a water molecule (H2O), which has a non-linear structure."""
    symbols = ["O", "H", "H"]
    # Standard coordinates for H2O with the Oxygen atom at the origin
    coordinates = np.array(
        [[0.0000, 0.0000, 0.0000], [0.757, 0.586, 0.0000], [-0.757, 0.586, 0.0000]]
    )
    return qp.qchem.Molecule(symbols, coordinates)


def get_pairwise_distances(coords_or_molecule):
    """Helper function to calculate a symmetric matrix of all pairwise atomic distances."""
    if hasattr(coords_or_molecule, "coordinates"):
        coords = coords_or_molecule.coordinates
    else:
        coords = coords_or_molecule
    return squareform(pdist(coords))


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
            ValueError, match="PennyLane `qchem.Molecule` or a PySCF `gto.Mole`"
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

    @pytest.mark.parametrize("bond", [(0, 2), (2, 0), (-1, 0), (1, -1)])
    def test_out_of_bounds_atom_connectivity(self, h2_molecule, bond):
        """Test ValueError for out-of-bounds indices in atom_connectivity."""
        with pytest.raises(ValueError, match="atom indices"):
            MoleculeTransformer(
                base_molecule=h2_molecule,
                bond_modifiers=[1.1],
                atom_connectivity=[bond],
            )

    def test_reversed_bond_is_in_bounds(self, h2_molecule):
        mt = MoleculeTransformer(
            base_molecule=h2_molecule, bond_modifiers=[1.1], atom_connectivity=[(1, 0)]
        )
        assert mt.atom_connectivity == [(1, 0)]

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

    def test_ring_closing_bond_to_transform_raises(self, water_molecule):
        with pytest.raises(ValueError, match=r"Bonds \[\(2, 1\)\] close a ring"):
            MoleculeTransformer(
                base_molecule=water_molecule,
                bond_modifiers=[1.1],
                atom_connectivity=[(0, 1), (0, 2), (2, 1)],
                bonds_to_transform=[(0, 1), (2, 1)],
            )

    def test_spanning_tree_bond_in_a_ring_is_accepted(self, water_molecule):
        mt = MoleculeTransformer(
            base_molecule=water_molecule,
            bond_modifiers=[1.1],
            atom_connectivity=[(0, 1), (0, 2), (2, 1)],
            bonds_to_transform=[(0, 1)],
        )
        assert mt.bonds_to_transform == [(0, 1)]

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

    def test_generate_from_pyscf_molecule(self, gto, pyscf_h2_molecule):
        """A PySCF base sweeps into PySCF variants, in the same Bohr geometry.

        Both molecule types expose coordinates in Bohr, so the same modifier
        scales the bond identically whichever stack supplied the molecule.
        """
        variants = MoleculeTransformer(
            base_molecule=pyscf_h2_molecule, bond_modifiers=[0.5, 1.5]
        ).generate()

        assert set(variants) == {0.5, 1.5}
        for modifier, variant in variants.items():
            assert isinstance(variant, gto.Mole)
            bond = np.linalg.norm(variant.atom_coords()[1] - variant.atom_coords()[0])
            assert np.isclose(bond, 1.4 * modifier)

        # The base must survive untouched — set_geom_ defaults to in-place.
        assert np.isclose(np.linalg.norm(pyscf_h2_molecule.atom_coords()[1]), 1.4)

    def test_angstrom_pyscf_molecule_keeps_its_unit(self, gto, recwarn):
        """Variants of an Angstrom molecule stay in Angstrom, without a PySCF warning."""
        base = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", unit="Angstrom")

        variant = MoleculeTransformer(
            base_molecule=base, bond_modifiers=[1.5]
        ).generate()[1.5]

        assert variant.unit == "Angstrom"
        bond = np.linalg.norm(variant.atom_coords()[1] - variant.atom_coords()[0])
        assert np.isclose(bond, 1.5 * np.linalg.norm(base.atom_coords()[1]))
        assert not recwarn.list

    def test_pyscf_mean_field_is_reduced_to_its_molecule(
        self, gto, scf, pyscf_h2_molecule
    ):
        """A converged mean field belongs to one geometry, so sweep its molecule."""
        transformer = MoleculeTransformer(
            base_molecule=scf.RHF(pyscf_h2_molecule), bond_modifiers=[1.5]
        )

        assert isinstance(transformer.base_molecule, gto.Mole)
        assert isinstance(transformer.generate()[1.5], gto.Mole)

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
def vqe_sweep_ansatze():
    """Shared ansatze for VQE sweep tests."""
    return [HartreeFockAnsatz(), GenericLayerAnsatz([RYGate])]


@pytest.fixture
def vqe_sweep_optimizer():
    """Shared optimizer for VQE sweep tests."""
    return MonteCarloOptimizer(population_size=5, n_best_sets=2)


@pytest.fixture
def vqe_sweep_max_iterations():
    """Shared max_iterations for VQE sweep tests."""
    return 10


@pytest.fixture
def h2_problem(h2_molecule, qp):
    """Fixture for a problem wrapping an H2 molecular Hamiltonian."""
    hamiltonian, _ = qp.qchem.molecular_hamiltonian(h2_molecule)
    return HamiltonianProblem(hamiltonian)


def _assert_common_program_settings(program, max_iterations):
    assert isinstance(program.optimizer, MonteCarloOptimizer)
    assert program.max_iterations == max_iterations
    assert program.backend.shots == 5000


@pytest.fixture
def vqe_sweep(
    default_test_simulator,
    h2_molecule,
    vqe_sweep_ansatze,
    vqe_sweep_optimizer,
    vqe_sweep_max_iterations,
):
    """Fixture to create a VQEHyperparameterSweep instance with molecule_transformer."""
    bond_modifiers = [0.9, 1.0, 1.1]

    transformer = MoleculeTransformer(
        base_molecule=h2_molecule,
        bond_modifiers=bond_modifiers,
    )

    return VQEHyperparameterSweep(
        ansatze=vqe_sweep_ansatze,
        molecule_transformer=transformer,
        problems=None,
        optimizer=vqe_sweep_optimizer,
        max_iterations=vqe_sweep_max_iterations,
        backend=default_test_simulator,
    )


@pytest.fixture
def vqe_sweep_problems(
    default_test_simulator,
    h2_problem,
    vqe_sweep_ansatze,
    vqe_sweep_optimizer,
    vqe_sweep_max_iterations,
):
    """Fixture to create a VQEHyperparameterSweep instance with problems."""
    problems = [h2_problem]

    return VQEHyperparameterSweep(
        ansatze=vqe_sweep_ansatze,
        molecule_transformer=None,
        problems=problems,
        optimizer=vqe_sweep_optimizer,
        max_iterations=vqe_sweep_max_iterations,
        backend=default_test_simulator,
    )


class TestVQEHyperparameterSweep:
    """A test class to group all tests for the VQEHyperparameterSweep."""

    def test_child_checkpoints_use_distinct_derived_directories(
        self, vqe_sweep_problems, tmp_path
    ):
        vqe_sweep_problems.create_programs()
        vqe_sweep_problems._round_index = 1
        original = CheckpointConfig(checkpoint_dir=tmp_path, checkpoint_interval=2)

        session = vqe_sweep_problems._prepare_checkpoint_session(
            original, vqe_sweep_problems._save_round_input_state(None, tmp_path)
        )
        configs = session.iterative_config_by_program

        assert set(configs) == set(vqe_sweep_problems.programs.values())
        paths = [config.checkpoint_dir for config in configs.values()]
        assert len(set(paths)) == len(paths)
        assert {path.name for path in paths} == {
            f"program_{slot:03d}" for slot in range(len(paths))
        }
        assert {path.parent.name for path in paths} == {"round_001"}
        assert all(config.checkpoint_interval == 2 for config in configs.values())
        assert original.checkpoint_dir == tmp_path

    def test_sampling_backend_is_owned_by_ensemble(
        self,
        default_test_simulator,
        sampling_test_simulator,
        h2_problem,
        vqe_sweep_ansatze,
        vqe_sweep_optimizer,
        vqe_sweep_max_iterations,
    ):
        sweep = VQEHyperparameterSweep(
            ansatze=vqe_sweep_ansatze,
            problems=[h2_problem],
            optimizer=vqe_sweep_optimizer,
            max_iterations=vqe_sweep_max_iterations,
            backend=default_test_simulator,
            sampling_backend=sampling_test_simulator,
        )

        sweep.create_programs()

        assert sweep.sampling_backend is sampling_test_simulator
        assert all(
            program.sampling_backend is None for program in sweep.programs.values()
        )

    @pytest.mark.parametrize(
        "reporting_kwargs, expected_level",
        [
            pytest.param({}, ReportingLevel.COMPACT, id="default"),
            pytest.param({"reporting_level": "off"}, ReportingLevel.OFF, id="off"),
        ],
    )
    def test_children_receive_sweep_configuration(
        self,
        default_test_simulator,
        h2_problem,
        vqe_sweep_ansatze,
        reporting_kwargs,
        expected_level,
    ):
        """Children get the sweep's ansatz, a fresh optimizer copy and forwarded kwargs."""
        template = SPSAOptimizer()
        sweep = VQEHyperparameterSweep(
            ansatze=vqe_sweep_ansatze,
            problems=[h2_problem],
            optimizer=template,
            max_iterations=3,
            backend=default_test_simulator,
            seed=11,
            precision=5,
            **reporting_kwargs,
        )

        sweep.create_programs()

        assert sweep.reporting_level == expected_level
        optimizers = []
        for ansatz in vqe_sweep_ansatze:
            program = sweep.programs[(ansatz.name, 0)]
            assert program.ansatz is ansatz
            assert program.max_iterations == 3
            assert program._seed == 11
            assert program._precision == 5
            assert program.backend is default_test_simulator
            assert isinstance(program.optimizer, SPSAOptimizer)
            optimizers.append(program.optimizer)
        assert len({id(o) for o in optimizers + [template]}) == len(optimizers) + 1

    def test_verify_basic_behaviour(self, vqe_sweep, mocker):
        """Test that the sweep conforms to basic batch program behavior."""
        verify_basic_program_ensemble_behaviour(vqe_sweep, mocker)

    def test_correct_number_of_programs_created_molecule_transformer(
        self, vqe_sweep, vqe_sweep_max_iterations
    ):
        """Test that the correct number of VQE programs are created with molecule_transformer."""
        bond_modifiers = vqe_sweep.molecule_transformer.bond_modifiers
        ansatze = vqe_sweep.ansatze

        vqe_sweep.create_programs()

        # Expected count is the cartesian product of ansatze and bond_modifiers (2 * 3 = 6)
        expected_count = len(ansatze) * len(bond_modifiers)
        assert len(vqe_sweep.programs) == expected_count

        # Verify that all expected program keys exist
        # Program keys are tuples of (ansatz.name, bond_modifier_value)
        assert all(
            (ansatz.name, modifier) in vqe_sweep.programs
            for ansatz, modifier in product(ansatze, bond_modifiers)
        )

        for program in vqe_sweep.programs.values():
            _assert_common_program_settings(program, vqe_sweep_max_iterations)
            assert program.n_qubits == 4

    def test_sweep_over_a_pyscf_transformer_builds_real_programs(
        self,
        pyscf_h2_molecule,
        default_test_simulator,
        vqe_sweep_optimizer,
        vqe_sweep_max_iterations,
    ):
        """The feature boundary: a PySCF transformer drives a whole sweep.

        Nothing is mocked here — each variant has to survive VQE's own
        molecule handling and produce a real cost Hamiltonian.
        """
        bond_modifiers = [1.0, 1.2]
        sweep = VQEHyperparameterSweep(
            ansatze=[HartreeFockAnsatz()],
            molecule_transformer=MoleculeTransformer(
                base_molecule=pyscf_h2_molecule, bond_modifiers=bond_modifiers
            ),
            optimizer=vqe_sweep_optimizer,
            max_iterations=vqe_sweep_max_iterations,
            backend=default_test_simulator,
        )

        sweep.create_programs()

        assert len(sweep.programs) == len(bond_modifiers)
        for modifier in bond_modifiers:
            program = sweep.programs[(HartreeFockAnsatz().name, modifier)]
            assert program.n_qubits == 4
            assert isinstance(program.cost_hamiltonian, SparsePauliOp)

    def test_correct_number_of_programs_created_problems(
        self, vqe_sweep_problems, vqe_sweep_max_iterations
    ):
        """Test that the correct number of VQE programs are created with problems."""
        problems = vqe_sweep_problems.problems
        ansatze = vqe_sweep_problems.ansatze

        vqe_sweep_problems.create_programs()

        # Expected count is the cartesian product of ansatze and problems (2 * 1 = 2)
        expected_count = len(ansatze) * len(problems)
        assert len(vqe_sweep_problems.programs) == expected_count

        # Verify that all expected program keys exist
        # Program keys are tuples of (ansatz.name, problem_index)
        assert all(
            (ansatz.name, p_id) in vqe_sweep_problems.programs
            for ansatz, p_id in product(ansatze, range(len(problems)))
        )

        for program in vqe_sweep_problems.programs.values():
            _assert_common_program_settings(program, vqe_sweep_max_iterations)
            assert program.cost_hamiltonian is not None

    def test_problem_dict_keys_use_ids(
        self,
        default_test_simulator,
        h2_problem,
        vqe_sweep_ansatze,
        vqe_sweep_optimizer,
        vqe_sweep_max_iterations,
    ):
        """Test that dict problem inputs use the dict keys as program IDs."""
        problems = {"h0": h2_problem}

        vqe_sweep = VQEHyperparameterSweep(
            ansatze=vqe_sweep_ansatze,
            molecule_transformer=None,
            problems=problems,
            optimizer=vqe_sweep_optimizer,
            max_iterations=vqe_sweep_max_iterations,
            backend=default_test_simulator,
        )

        vqe_sweep.create_programs()

        expected_count = len(vqe_sweep_ansatze) * len(problems)
        assert len(vqe_sweep.programs) == expected_count
        assert all(
            (ansatz.name, p_id) in vqe_sweep.programs
            for ansatz, p_id in product(vqe_sweep_ansatze, problems.keys())
        )

    def test_hamiltonian_constant_only_raises(
        self,
        qp,
        default_test_simulator,
        vqe_sweep_optimizer,
        vqe_sweep_max_iterations,
    ):
        """Test that constant-only Hamiltonians are rejected during program creation."""
        vqe_sweep = VQEHyperparameterSweep(
            ansatze=[HartreeFockAnsatz()],
            molecule_transformer=None,
            problems=[HamiltonianProblem(qp.Identity(0))],
            optimizer=vqe_sweep_optimizer,
            max_iterations=vqe_sweep_max_iterations,
            backend=default_test_simulator,
        )

        with pytest.raises(
            ValueError, match="Hamiltonian contains only constant terms"
        ):
            vqe_sweep.create_programs()

    def test_molecule_transformer_and_problems_rejected(
        self,
        qp,
        default_test_simulator,
        h2_molecule,
        vqe_sweep_optimizer,
        vqe_sweep_max_iterations,
    ):
        """Test that providing both molecule_transformer and problems raises."""
        transformer = MoleculeTransformer(
            base_molecule=h2_molecule,
            bond_modifiers=[1.0],
        )

        with pytest.raises(ValueError, match="supports either a molecule sweep"):
            VQEHyperparameterSweep(
                ansatze=[HartreeFockAnsatz()],
                molecule_transformer=transformer,
                problems=[HamiltonianProblem(qp.PauliZ(0))],
                optimizer=vqe_sweep_optimizer,
                max_iterations=vqe_sweep_max_iterations,
                backend=default_test_simulator,
            )

    def test_results_aggregated_correctly(self, mocker, vqe_sweep):
        """Test that results from multiple VQE runs are aggregated to find the minimum energy."""

        mock_program_1 = mocker.MagicMock()
        mock_program_1.losses_history = [{0: -1.2}]
        mock_program_1.best_loss = -1.2
        mock_program_2 = mocker.MagicMock()
        mock_program_2.losses_history = [{0: -1.1}]
        mock_program_2.best_loss = -1.1

        uccsd_instance = UCCSDAnsatz()
        generic_ry_instance = GenericLayerAnsatz([RYGate])

        vqe_sweep.programs = {
            (uccsd_instance, 0.9): mock_program_1,
            (generic_ry_instance, 1.0): mock_program_2,
        }

        smallest_key, smallest_value = vqe_sweep.aggregate_results()

        assert smallest_key == (uccsd_instance, 0.9)
        assert smallest_value == -1.2

    @pytest.mark.parametrize(
        "plot_kwargs",
        [
            pytest.param({}, id="default"),
            pytest.param({"graph_type": "line"}, id="line"),
        ],
    )
    def test_visualize_results_line_plot_data(self, mocker, vqe_sweep, plot_kwargs):
        """Test that the line plot visualization is called with the correct data."""
        mock_plot = mocker.patch("matplotlib.pyplot.plot")
        mock_scatter = mocker.patch("matplotlib.pyplot.scatter")
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

        vqe_sweep.visualize_results(**plot_kwargs)

        mock_scatter.assert_not_called()
        assert mock_plot.call_count == len(vqe_sweep.ansatze)

        # Collect per-call data
        calls_by_label = {}
        for call in mock_plot.call_args_list:
            label = call.kwargs["label"]
            calls_by_label[label] = {
                "x": call.args[0],
                "y": call.args[1],
                "color": call.kwargs["color"],
            }

        # Each ansatz gets its correct data series
        assert calls_by_label["HartreeFockAnsatz"]["x"] == [0.9, 1.0, 1.1]
        assert calls_by_label["HartreeFockAnsatz"]["y"] == [-9.0, -10.0, -11.0]
        assert calls_by_label["GenericLayerAnsatz"]["x"] == [0.9, 1.0, 1.1]
        assert calls_by_label["GenericLayerAnsatz"]["y"] == [-10.0, -11.0, -12.0]

        # Each ansatz gets a distinct color
        colors = [c["color"] for c in calls_by_label.values()]
        assert len(set(colors)) == len(colors)

    def test_visualize_results_with_invalid_graph_type(self, mocker, vqe_sweep):
        """Test that providing an invalid graph type raises a ValueError."""
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

    @pytest.mark.parametrize(
        "n_modifiers",
        [
            pytest.param(None, id="all_programs"),
            pytest.param(2, id="missing_programs"),
        ],
    )
    def test_visualize_results_scatter_plot(self, mocker, vqe_sweep, n_modifiers):
        """One scatter per ansatz, over the bond modifiers that have programs."""
        mock_scatter = mocker.patch("matplotlib.pyplot.scatter")
        mocker.patch("matplotlib.pyplot.show")
        mocker.patch("matplotlib.pyplot.legend")
        mocker.patch("matplotlib.pyplot.xlabel")
        mocker.patch("matplotlib.pyplot.ylabel")

        available = list(vqe_sweep.molecule_transformer.bond_modifiers[:n_modifiers])
        mock_programs = {}
        for ansatz_idx, ansatz in enumerate(vqe_sweep.ansatze):
            for modifier in available:
                mock_program = mocker.MagicMock()
                mock_program.best_loss = -(modifier * 10 + ansatz_idx)
                mock_programs[(ansatz.name, modifier)] = mock_program
        vqe_sweep.programs = mock_programs

        vqe_sweep.visualize_results(graph_type="scatter")

        assert mock_scatter.call_count == len(vqe_sweep.ansatze)
        for ansatz_idx, call in enumerate(mock_scatter.call_args_list):
            assert call.args[0] == available
            assert call.args[1] == [-(m * 10 + ansatz_idx) for m in available]
            assert call.kwargs["label"] == vqe_sweep.ansatze[ansatz_idx].name

    def test_visualize_results_with_executor(self, mocker, vqe_sweep):
        """Test visualization calls join() when executor is present."""
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


def test_safe_normalize_edge_cases():
    """Test _safe_normalize with edge cases."""
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


def test_compute_angle_edge_cases():
    """Test _compute_angle with edge cases."""
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


class TestZMatrixConversion:
    """Tests for Z-matrix conversion functions."""

    def test_cartesian_to_zmatrix_empty_coords(self):
        """Test _cartesian_to_zmatrix with empty coordinates."""
        empty_coords = np.array([]).reshape(0, 3)
        connectivity = []

        with pytest.raises(ValueError, match="Cannot convert empty coordinate array"):
            _cartesian_to_zmatrix(empty_coords, connectivity)

    def test_cartesian_to_zmatrix_single_atom(self):
        """Test _cartesian_to_zmatrix with single atom."""
        coords = np.array([[0.0, 0.0, 0.0]])
        connectivity = []

        zmatrix = _cartesian_to_zmatrix(coords, connectivity)
        assert len(zmatrix) == 1
        assert zmatrix[0].bond_ref is None
        assert zmatrix[0].angle_ref is None
        assert zmatrix[0].dihedral_ref is None

    def test_zmatrix_to_cartesian_edge_cases(self):
        """Test _zmatrix_to_cartesian with edge cases."""
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

    def test_zero_bond_length_raises(self):
        """Z-matrix entries with zero bond length are rejected as unphysical."""
        zmatrix = [
            _ZMatrixEntry(None, None, None, None, None, None),
            _ZMatrixEntry(0, None, None, 0.0, None, None),  # Zero bond length!
        ]

        with pytest.raises(ValueError, match="Bond length for atom 1 must be positive"):
            _zmatrix_to_cartesian(zmatrix)


def test_transform_bonds_zero_length_error():
    """Test _transform_bonds raises RuntimeError for zero bond length."""
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


def _rotation(axis, degrees):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    theta = np.radians(degrees)
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * k @ k


_RNG_POINTS = np.random.default_rng(7).normal(size=(6, 3))


class TestKabschAlignment:
    """Tests for Kabsch alignment algorithm."""

    @pytest.mark.parametrize(
        "P, Q",
        [
            pytest.param(
                np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
                np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
                id="identical",
            ),
            pytest.param(
                np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                np.array([[11.0, 22.0, 33.0], [14.0, 25.0, 36.0]]),
                id="translation",
            ),
            pytest.param(
                _RNG_POINTS,
                _RNG_POINTS @ _rotation([1.0, 2.0, -0.5], 73.0).T + [3.0, -1.0, 2.0],
                id="rotation+translation",
            ),
        ],
    )
    def test_kabsch_align_recovers_target(self, P, Q):
        """_kabsch_align maps P exactly onto a rigidly displaced Q."""
        np.testing.assert_allclose(_kabsch_align(P, Q), Q, atol=1e-10)

    def test_kabsch_align_with_reference_atoms(self):
        """Test _kabsch_align with reference atom subset."""
        P = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        Q = P + np.array([10.0, 20.0, 30.0])

        # Align only first two atoms
        aligned = _kabsch_align(P, Q, reference_atoms_idx=slice(0, 2))

        # First two atoms should be aligned
        np.testing.assert_array_almost_equal(aligned[:2], Q[:2])
        # Third atom should be transformed but not necessarily aligned
        assert aligned.shape == P.shape


def _iupac_dihedral(a, b, c, d):
    b0, b1, b2 = b - a, c - b, d - c
    n1, n2 = np.cross(b0, b1), np.cross(b1, b2)
    m1 = np.cross(n1, b1 / np.linalg.norm(b1))
    return np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))


def _signed_volumes(coords):
    """Signed volume of every atom quadruple; equal values mean equal handedness."""
    n = len(coords)
    return np.array(
        [
            np.linalg.det(coords[[j, k, l]] - coords[i])
            for i in range(n)
            for j in range(i + 1, n)
            for k in range(j + 1, n)
            for l in range(k + 1, n)
        ]
    )


def _assert_congruent(actual, expected, atol=1e-10):
    """Same shape up to a proper rigid motion: distances and handedness agree."""
    np.testing.assert_allclose(
        get_pairwise_distances(actual), get_pairwise_distances(expected), atol=atol
    )
    np.testing.assert_allclose(
        _signed_volumes(actual), _signed_volumes(expected), atol=atol
    )


def _chain(bonds, angles, dihedrals):
    """Unbranched chain from bond lengths, bond angles and IUPAC dihedrals."""
    coords = [np.zeros(3), np.array([bonds[0], 0.0, 0.0])]
    for bond, angle in zip(bonds[1:], angles):
        a, b = coords[-2], coords[-1]
        direction = _rotation([0.0, 0.0, 1.0], 180.0 - angle) @ (b - a)
        coords.append(b + bond * direction / np.linalg.norm(direction))
    coords = np.array(coords)
    for i, target in enumerate(dihedrals, start=3):
        a, b, c = coords[i - 3], coords[i - 2], coords[i - 1]
        twist = _iupac_dihedral(a, b, c, coords[i]) - target
        coords[i:] = (coords[i:] - c) @ _rotation(c - b, twist).T + c
    return coords


_CHAIN_COORDS = _chain(
    bonds=[1.4, 1.1, 1.6, 1.2], angles=[109.5, 120.0, 100.0], dihedrals=[60.0, -130.0]
)
_CHAIN_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4)]
_NH3_COORDS = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.77, 0.0, -0.72],
        [-0.885, 1.533, -0.72],
        [-0.885, -1.533, -0.72],
    ]
)
_NH3_EDGES = [(0, 1), (0, 2), (0, 3)]


def test_chain_fixture_has_requested_dihedrals():
    assert np.isclose(_iupac_dihedral(*_CHAIN_COORDS[:4]), 60.0)
    assert np.isclose(_iupac_dihedral(*_CHAIN_COORDS[1:]), -130.0)


@pytest.mark.parametrize(
    "coords, connectivity",
    [
        pytest.param(
            np.array([[0.3, -1.0, 2.0], [1.5, 0.2, 0.4]]), [(0, 1)], id="diatomic"
        ),
        pytest.param(
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0]]),
            [(0, 1), (1, 2)],
            id="triangle",
        ),
        pytest.param(
            np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]]),
            [(0, 1), (0, 2)],
            id="water",
        ),
        pytest.param(_CHAIN_COORDS, _CHAIN_EDGES, id="chain"),
        pytest.param(
            _CHAIN_COORDS, [(3, 4), (1, 0), (3, 2), (2, 1)], id="chain_out_of_order"
        ),
        pytest.param(
            _chain([1.4, 0.05, 1.6, 0.02], [109.5, 120.0, 100.0], [60.0, -130.0]),
            _CHAIN_EDGES,
            id="chain_short_bonds",
        ),
        pytest.param(_NH3_COORDS, _NH3_EDGES, id="nh3"),
        pytest.param(
            _NH3_COORDS[[1, 2, 0, 3]], [(0, 2), (1, 2), (2, 3)], id="nh3_permuted"
        ),
        pytest.param(
            np.array(
                [[0.0, 0.0, 0.0], [2.2, 0.0, 0.0], [4.4, 0.0, 0.0], [4.4, 1.9, 0.7]]
            ),
            [(0, 1), (1, 2), (2, 3)],
            id="collinear_prefix",
        ),
    ],
)
def test_zmatrix_roundtrip_is_exact(coords, connectivity):
    """Any connected tree rebuilds to the same shape, handedness included."""
    rebuilt = _zmatrix_to_cartesian(_cartesian_to_zmatrix(coords, connectivity))

    _assert_congruent(rebuilt, coords)


@pytest.mark.parametrize(
    "bonds_to_transform, expected_scale",
    [
        pytest.param(_NH3_EDGES, np.where(np.eye(4), 1.0, 1.1), id="all_bonds"),
        pytest.param(
            [(0, 1)],
            np.array(
                [
                    [1.0, 1.1, 1.0, 1.0],
                    [1.1, 1.0, np.nan, np.nan],
                    [1.0, np.nan, 1.0, 1.0],
                    [1.0, np.nan, 1.0, 1.0],
                ]
            ),
            id="one_bond",
        ),
    ],
)
def test_bond_scan_on_branched_molecule(qp, bonds_to_transform, expected_scale):
    """Scanning NH3 changes only the scanned bonds; the rest of the shape holds."""
    molecule = qp.qchem.Molecule(["N", "H", "H", "H"], _NH3_COORDS)
    variant = MoleculeTransformer(
        base_molecule=molecule,
        bond_modifiers=[1.1],
        atom_connectivity=_NH3_EDGES,
        bonds_to_transform=bonds_to_transform,
    ).generate()[1.1]

    original = get_pairwise_distances(_NH3_COORDS)
    ratio = np.divide(
        get_pairwise_distances(variant.coordinates),
        original,
        out=np.ones_like(original),
        where=original > 0,
    )
    fixed = ~np.isnan(expected_scale)
    np.testing.assert_allclose(ratio[fixed], expected_scale[fixed], atol=1e-10)


def test_bond_scan_with_root_not_first_in_bfs_order(qp):
    """Water numbered H, H, O scales its bonds without assuming index order."""
    coords = np.array([[1.43, 1.11, 0.0], [-1.43, 1.11, 0.0], [0.0, 0.0, 0.0]])
    molecule = qp.qchem.Molecule(["H", "H", "O"], coords)
    variant = MoleculeTransformer(
        base_molecule=molecule,
        bond_modifiers=[1.2],
        atom_connectivity=[(0, 2), (1, 2)],
    ).generate()[1.2]

    _assert_congruent(variant.coordinates, coords * 1.2)


@pytest.mark.parametrize(
    "n_atoms, connectivity",
    [
        pytest.param(4, [(0, 1), (2, 3)], id="disconnected"),
        pytest.param(4, [(0, 1), (1, 2)], id="atom_left_out"),
    ],
)
def test_connectivity_must_span_every_atom(qp, n_atoms, connectivity):
    molecule = qp.qchem.Molecule(["H"] * n_atoms, _NH3_COORDS[:n_atoms])
    with pytest.raises(ValueError, match="must connect every atom"):
        MoleculeTransformer(
            base_molecule=molecule,
            bond_modifiers=[1.1],
            atom_connectivity=connectivity,
        )


def test_kabsch_align_keeps_handedness_against_a_mirror_image():
    """Aligning onto a reflection still moves the source by a proper rotation."""
    mirror = _RNG_POINTS * [1.0, 1.0, -1.0]

    aligned = _kabsch_align(_RNG_POINTS, mirror)

    _assert_congruent(aligned, _RNG_POINTS)
