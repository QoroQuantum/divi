# SPDX-FileCopyrightText: 2025-2026 Qoro Quantum Ltd <divi@qoroquantum.de>
#
# SPDX-License-Identifier: Apache-2.0

import copy
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from itertools import product
from typing import Any, Literal, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from divi._optional import optional_module
from divi.hamiltonians._term_ops import ObservableInput
from divi.qprog import VQE, Ansatz, ProgramEnsemble, ReportingLevel
from divi.qprog.optimizers import MonteCarloOptimizer, Optimizer


def _is_pyscf_molecule(molecule) -> bool:
    gto = optional_module("pyscf.gto")
    return gto is not None and isinstance(molecule, gto.Mole)


def _is_pennylane_molecule(molecule) -> bool:
    qp = optional_module("pennylane")
    return qp is not None and isinstance(molecule, qp.qchem.Molecule)


def _normalise_molecule(molecule):
    """Reduce a PySCF mean field to the molecule its geometry belongs to."""
    scf = optional_module("pyscf.scf")
    return (
        molecule.mol
        if scf is not None and isinstance(molecule, scf.hf.SCF)
        else molecule
    )


def _geometry_of(molecule) -> npt.NDArray:
    """Atomic coordinates in Bohr — the native unit of both molecule types."""
    if _is_pyscf_molecule(molecule):
        return np.asarray(molecule.atom_coords())
    return np.asarray(molecule.coordinates)


def _atom_count(molecule) -> int:
    if _is_pyscf_molecule(molecule):
        return int(molecule.natm)
    return len(molecule.symbols)


def _with_geometry(molecule, coordinates: npt.NDArray):
    """``molecule`` moved to ``coordinates`` (Bohr), leaving the original alone."""
    if _is_pyscf_molecule(molecule):
        return molecule.set_geom_(np.asarray(coordinates), unit="Bohr", inplace=False)
    variant = copy.copy(molecule)
    variant.coordinates = coordinates
    return variant


class _ZMatrixEntry(NamedTuple):
    bond_ref: int | None
    angle_ref: int | None
    dihedral_ref: int | None
    bond_length: float | None
    angle: float | None
    dihedral: float | None


# --- Helper functions ---
def _safe_normalize(v, fallback=None):
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        if fallback is None:
            fallback = np.array([1.0, 0.0, 0.0])
        return fallback / np.linalg.norm(fallback)
    return v / norm


def _compute_angle(v1, v2):
    dot = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


def _compute_dihedral(b0, b1, b2):
    n1 = np.cross(b0, b1)
    n2 = np.cross(b0, b2)
    if np.linalg.norm(n1) < 1e-6 or np.linalg.norm(n2) < 1e-6:
        return 0.0
    dot = np.dot(n1, n2) / (np.linalg.norm(n1) * np.linalg.norm(n2))
    angle = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    if np.dot(b1, n2) < 0:
        angle *= -1
    return angle


def _find_refs(adj, placed, parent, child):
    gp = next((n for n in adj[parent] if n != child and n in placed), None)
    ggp = None
    if gp is not None:
        ggp = next((n for n in adj[gp] if n != parent and n in placed), None)
    return gp, ggp


# --- Main functions ---
def _cartesian_to_zmatrix(
    coords: npt.NDArray[np.float64], connectivity: list[tuple[int, int]]
) -> list[_ZMatrixEntry]:
    num_atoms = len(coords)
    if num_atoms == 0:
        raise ValueError(
            "Cannot convert empty coordinate array to Z-matrix: molecule must have at least one atom."
        )

    adj = [[] for _ in range(num_atoms)]
    for i, j in connectivity:
        adj[i].append(j)
        adj[j].append(i)

    zmatrix_entries = {0: _ZMatrixEntry(None, None, None, None, None, None)}
    q = deque([0])
    placed_atoms = {0}

    while q:
        parent_idx = q.popleft()
        for child_idx in adj[parent_idx]:
            if child_idx in placed_atoms:
                continue
            placed_atoms.add(child_idx)
            q.append(child_idx)

            bond_len = np.linalg.norm(coords[child_idx] - coords[parent_idx])
            gp, ggp = _find_refs(adj, placed_atoms, parent_idx, child_idx)

            angle = None
            if gp is not None:
                angle = _compute_angle(
                    coords[child_idx] - coords[parent_idx],
                    coords[gp] - coords[parent_idx],
                )

            dihedral = None
            if gp is not None and ggp is not None:
                dihedral = _compute_dihedral(
                    coords[parent_idx] - coords[gp],
                    coords[child_idx] - coords[parent_idx],
                    coords[ggp] - coords[gp],
                )

            zmatrix_entries[child_idx] = _ZMatrixEntry(
                parent_idx, gp, ggp, float(bond_len), angle, dihedral
            )

    return [zmatrix_entries[i] for i in range(num_atoms)]


def _zmatrix_to_cartesian(z_matrix: list[_ZMatrixEntry]) -> npt.NDArray[np.float64]:
    n_atoms = len(z_matrix)
    coords = np.zeros((n_atoms, 3))

    if n_atoms == 0:
        return coords

    for i, entry in enumerate(z_matrix[1:], start=1):
        if entry.bond_length is not None and entry.bond_length <= 0:
            raise ValueError(
                f"Bond length for atom {i} must be positive, got {entry.bond_length}"
            )

    # --- First atom at origin ---
    coords[0] = np.array([0.0, 0.0, 0.0])

    # --- Second atom along +X axis ---
    if n_atoms > 1:
        coords[1] = np.array([z_matrix[1].bond_length, 0.0, 0.0])

    # --- Third atom in XY plane ---
    if n_atoms > 2:
        entry = z_matrix[2]
        r = entry.bond_length
        theta = np.radians(entry.angle) if entry.angle is not None else 0.0

        a1 = coords[entry.bond_ref]
        a2 = coords[entry.angle_ref]

        v = a2 - a1
        v /= np.linalg.norm(v)
        # fixed perpendicular in XY plane, fallback handled inline
        perp = np.array([-v[1], v[0], 0.0])
        perp /= np.linalg.norm(perp) if np.linalg.norm(perp) > 1e-6 else 1.0

        coords[2] = a1 + r * (np.cos(theta) * v + np.sin(theta) * perp)

    for i, entry in enumerate(z_matrix[3:], start=3):
        a1 = coords[entry.bond_ref] if entry.bond_ref is not None else np.zeros(3)
        a2 = coords[entry.angle_ref] if entry.angle_ref is not None else np.zeros(3)
        a3 = (
            coords[entry.dihedral_ref]
            if entry.dihedral_ref is not None
            else np.zeros(3)
        )

        r = entry.bond_length

        theta = np.radians(entry.angle) if entry.angle is not None else 0.0
        phi = np.radians(entry.dihedral) if entry.dihedral is not None else 0.0

        b1 = _safe_normalize(a1 - a2, fallback=np.array([1.0, 0.0, 0.0]))
        b2 = a3 - a2
        n = _safe_normalize(np.cross(b1, b2), fallback=np.array([0.0, 0.0, 1.0]))
        nc = np.cross(n, b1)

        coords[i] = a1 + r * (
            -np.cos(theta) * b1 + np.sin(theta) * (np.cos(phi) * nc + np.sin(phi) * n)
        )

    return coords


def _transform_bonds(
    zmatrix: list[_ZMatrixEntry],
    bonds_to_transform: list[tuple[int, int]],
    value: float,
    transform_type: Literal["scale", "delta"] = "scale",
) -> list[_ZMatrixEntry]:
    """
    Transform specified bonds in a Z-matrix.

    Args:
        zmatrix: List of _ZMatrixEntry.
        bonds_to_transform: List of (atom1, atom2) tuples specifying bonds.
        value: Multiplier or additive value.
        transform_type: "scale" or "add".

    Returns:
        New Z-matrix with transformed bond lengths.
    """
    bonds_set = {tuple(sorted(b)) for b in bonds_to_transform}

    new_zmatrix = []
    for i, entry in enumerate(zmatrix):
        if (
            entry.bond_ref is not None
            and entry.bond_length is not None
            and tuple(sorted((i, entry.bond_ref))) in bonds_set
        ):
            old_length = entry.bond_length
            new_length = (
                old_length * value if transform_type == "scale" else old_length + value
            )
            if new_length == 0.0:
                raise RuntimeError(
                    "New bond length can't be zero after transformation."
                )
            new_zmatrix.append(entry._replace(bond_length=new_length))
        else:
            new_zmatrix.append(entry)
    return new_zmatrix


def _kabsch_align(
    P_in: npt.NDArray[np.float64],
    Q_in: npt.NDArray[np.float64],
    reference_atoms_idx=slice(None),
) -> npt.NDArray[np.float64]:
    """
    Align point set P onto Q using the Kabsch algorithm.

    Parameters
    ----------
    P : (N, D) npt.NDArray[np.float64]. Source coordinates.
    Q : (N, D) npt.NDArray[np.float64]. Target coordinates.

    Returns
    -------
    P_aligned : (N, D) npt.NDArray[np.float64]
        P rotated and translated onto Q.
    """

    P = P_in[reference_atoms_idx, :]
    Q = Q_in[reference_atoms_idx, :]

    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)

    Pc = np.mean(P, axis=0)
    Qc = np.mean(Q, axis=0)

    P_centered = P - Pc
    Q_centered = Q - Qc

    H = P_centered.T @ Q_centered
    U, _, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # Reflections have det = -1; flipping Vt's last row restores a rotation.
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = Qc - Pc @ R

    P_aligned = P_in @ R + t

    P_aligned[np.abs(P_aligned) < 1e-12] = 0.0

    return P_aligned


@dataclass(frozen=True, eq=True)
class MoleculeTransformer:
    """
    A class for transforming molecular structures by modifying bond lengths.

    This class generates variants of a base molecule by adjusting bond lengths
    according to specified modifiers. The modification mode is detected automatically.

    Variants are emitted as the same type as ``base_molecule``, so a PennyLane
    molecule sweeps into PennyLane molecules and a PySCF one into PySCF ones.

    Attributes:
        base_molecule: The reference molecule used as a template for generating
            variants — a PennyLane ``qchem.Molecule`` or a PySCF ``gto.Mole``.
            A PySCF mean field is reduced to the molecule it was built from.
        bond_modifiers: A list of values used to adjust bond lengths. The class will generate
            **one new molecule for each modifier** in this list. The modification
            mode is detected automatically:
            - **Scale mode**: If all values are positive, they are used as scaling
            factors (e.g., 1.1 for a 10% increase).
            - **Delta mode**: If any value is zero or negative, all values are
            treated as additive changes to the bond length, in Bohr — the unit
            both PennyLane and PySCF store coordinates in.
        atom_connectivity: A sequence of atom index pairs specifying the bonds in the molecule.
            If not provided, a chain structure will be assumed
            e.g.: `[(0, 1), (1, 2), (2, 3), ...]`.
        bonds_to_transform: A subset of `atom_connectivity` that specifies the bonds to modify.
            If None, all bonds will be transformed.
        alignment_atoms: Indices of atoms onto which to align the orientation of the resulting
            variants of the molecule. Only useful for visualisation and debugging.
            If None, no alignment is carried out.
    """

    base_molecule: Any
    bond_modifiers: Sequence[float]
    atom_connectivity: Sequence[tuple[int, int]] | None = None
    bonds_to_transform: Sequence[tuple[int, int]] | None = None
    alignment_atoms: Sequence[int] | None = None

    _mode: Literal["scale", "delta"] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(
            self, "base_molecule", _normalise_molecule(self.base_molecule)
        )
        # PySCF first, so a PySCF sweep never imports PennyLane to reject it.
        if not (
            _is_pyscf_molecule(self.base_molecule)
            or _is_pennylane_molecule(self.base_molecule)
        ):
            raise ValueError(
                "`base_molecule` is expected to be a PennyLane `qchem.Molecule` "
                "or a PySCF `gto.Mole` instance."
            )

        if not all(isinstance(x, (float, int)) for x in self.bond_modifiers):
            raise ValueError("`bond_modifiers` should be a sequence of floats.")
        if len(set(self.bond_modifiers)) < len(self.bond_modifiers):
            raise ValueError("`bond_modifiers` contains duplicate values.")
        object.__setattr__(
            self,
            "_mode",
            "scale" if all(v > 0 for v in self.bond_modifiers) else "delta",
        )

        n_symbols = _atom_count(self.base_molecule)
        atom_connectivity: Sequence[tuple[int, int]]
        if self.atom_connectivity is None:
            atom_connectivity = tuple(zip(range(n_symbols), range(1, n_symbols)))
            object.__setattr__(self, "atom_connectivity", atom_connectivity)
        else:
            atom_connectivity = self.atom_connectivity
            if len(set(atom_connectivity)) < len(atom_connectivity):
                raise ValueError("`atom_connectivity` contains duplicate values.")

            if not all(
                0 <= a < n_symbols and 0 <= b < n_symbols for a, b in atom_connectivity
            ):
                raise ValueError(
                    "`atom_connectivity` should be a sequence of tuples of"
                    " atom indices in (0, len(molecule.symbols))"
                )

        if self.bonds_to_transform is None:
            object.__setattr__(self, "bonds_to_transform", atom_connectivity)
        else:
            bonds_to_transform = self.bonds_to_transform
            if len(bonds_to_transform) == 0:
                raise ValueError("`bonds_to_transform` cannot be empty.")
            if not set(bonds_to_transform).issubset(atom_connectivity):
                raise ValueError(
                    "`bonds_to_transform` is not a subset of `atom_connectivity`"
                )

        if self.alignment_atoms is not None and not all(
            0 <= idx < n_symbols for idx in self.alignment_atoms
        ):
            raise ValueError(
                "`alignment_atoms` need to be in range (0, len(molecule.symbols))"
            )

    def generate(self) -> dict[float, Any]:
        """Bond-modified variants, each the same molecule type as the base."""
        variants = {}
        original_coords = _geometry_of(self.base_molecule)
        mode = self._mode

        atom_connectivity = list(self.atom_connectivity or ())
        bonds_to_transform = list(self.bonds_to_transform or ())

        z_matrix = _cartesian_to_zmatrix(original_coords, atom_connectivity)

        for value in self.bond_modifiers:
            if (value == 0 and mode == "delta") or (value == 1 and mode == "scale"):
                transformed_coords = original_coords.copy()
            else:
                transformed_z_matrix = _transform_bonds(
                    z_matrix, bonds_to_transform, value, mode
                )

                transformed_coords = _zmatrix_to_cartesian(transformed_z_matrix)

                if self.alignment_atoms is not None:
                    transformed_coords = _kabsch_align(
                        transformed_coords, original_coords, self.alignment_atoms
                    )

            variants[value] = _with_geometry(self.base_molecule, transformed_coords)

        return variants


class VQEHyperparameterSweep(ProgramEnsemble):
    """Allows user to carry out a grid search across different values
    for the ansatz and the bond length used in a VQE program.
    """

    def __init__(
        self,
        ansatze: Sequence[Ansatz],
        molecule_transformer: MoleculeTransformer | None = None,
        hamiltonians: (
            Sequence[ObservableInput] | Mapping[Any, ObservableInput] | None
        ) = None,
        optimizer: Optimizer | None = None,
        max_iterations: int = 10,
        **kwargs,
    ):
        """
        Initialise a VQE hyperparameter sweep.

        Parameters
        ----------
        ansatze: Sequence[Ansatz]
            A sequence of ansatz circuits to test.
        hamiltonians: Sequence[ObservableInput] | Mapping[Any, ObservableInput], optional
            The Hamiltonians to use for the VQE runs — any form ``to_spo``
            accepts (PennyLane operator, ``SparsePauliOp``, Pauli-string dict,
            or OpenFermion ``QubitOperator``). A mapping keys each program by
            its own key instead of by position. If ``None``
            (the default), no Hamiltonians are provided explicitly and
            molecule-based VQE runs will use their default Hamiltonians.
        molecule_transformer: MoleculeTransformer | None, optional
            A `MoleculeTransformer` object defining the configuration for
            generating the molecule variants. If ``None`` (the default),
            the provided ``hamiltonians`` are used directly and no molecular
            transformation is performed.
        optimizer: Optimizer
            The optimisation algorithm for the VQE runs.
        max_iterations: int
            The maximum number of optimizer iterations for each VQE run.
        **kwargs
            Forwarded to the parent class. ``reporting_level`` accepts a
            :class:`~divi.qprog.ReportingLevel` controlling how much live
            progress is shown. ``sampling_backend`` optionally selects a
            separate backend for final solution sampling.
        """
        super().__init__(
            backend=kwargs.pop("backend"),
            sampling_backend=kwargs.pop("sampling_backend", None),
            reporting_level=kwargs.pop("reporting_level", ReportingLevel.COMPACT),
        )

        self.molecule_transformer = molecule_transformer
        self.ansatze = ansatze
        self.hamiltonians = hamiltonians
        self.max_iterations = max_iterations

        if molecule_transformer is not None and hamiltonians is not None:
            raise ValueError(
                "VQEHyperparameterSweep supports either a molecule sweep "
                "(via molecule_transformer) or a Hamiltonian sweep (via hamiltonians), "
                "but not both."
            )

        if molecule_transformer is None and not hamiltonians:
            raise ValueError(
                "At least one of molecule_transformer or hamiltonians must be provided."
            )

        self._optimizer_template = (
            optimizer if optimizer is not None else MonteCarloOptimizer()
        )

        self._constructor = partial(
            VQE,
            max_iterations=self.max_iterations,
            backend=self.backend,
            **kwargs,
        )

    def create_programs(self, state=None):
        """
        Create VQE programs for all combinations of ansätze and molecule variants.

        Generates molecule variants using the configured MoleculeTransformer, then
        creates a VQE program for each (ansatz, molecule_variant) pair.

        Note:
            Program IDs are tuples of (ansatz_name, bond_modifier_value).
        """
        super().create_programs()

        # Molecule sweep or Hamiltonian sweep
        if self.molecule_transformer is not None:
            molecule_variants = self.molecule_transformer.generate()
            sweep_items = (
                (modifier, molecule, None)
                for modifier, molecule in molecule_variants.items()
            )
        else:
            hamiltonians = self.hamiltonians or ()
            if isinstance(hamiltonians, dict):
                h_items = hamiltonians.items()
            else:
                h_items = enumerate(hamiltonians)
            sweep_items = ((h_id, None, hamiltonian) for h_id, hamiltonian in h_items)

        for ansatz, (item_id, molecule, hamiltonian) in product(
            self.ansatze, sweep_items
        ):
            _job_id = (ansatz.name, item_id)
            self._programs[_job_id] = self._constructor(
                molecule=molecule,
                hamiltonian=hamiltonian,
                ansatz=ansatz,
                optimizer=self._optimizer_template.copy(),
            )

    def aggregate_results(self):
        """
        Find the best ansatz and bond configuration from all VQE runs.

        Compares the final energies across all ansatz/molecule combinations
        and returns the configuration that achieved the lowest ground state energy.

        Returns:
            tuple: A tuple containing:
                - best_config (tuple): (ansatz_name, bond_modifier) of the best result.
                - best_energy (float): The lowest energy achieved.

        Raises:
            RuntimeError: If programs haven't been run or have empty losses.
        """
        super().aggregate_results()

        all_energies = {key: prog.best_loss for key, prog in self.programs.items()}

        smallest_key = min(all_energies, key=lambda k: all_energies[k])
        smallest_value = all_energies[smallest_key]

        return smallest_key, smallest_value

    def visualize_results(self, graph_type: Literal["line", "scatter"] = "line"):
        """
        Visualise the results of the VQE problem.
        """
        if graph_type not in ["line", "scatter"]:
            raise ValueError(
                f"Invalid graph type: {graph_type}. Choose between 'line' and 'scatter'."
            )

        if (transformer := self.molecule_transformer) is None:
            raise RuntimeError(
                "visualize_results currently supports molecule-transformer sweeps only; "
                "visualisation for hamiltonians-only sweeps is not implemented."
            )

        if self._executor is not None:
            self.join()

        # Every configured ansatz, not only those whose programs completed.
        unique_ansatze = self.ansatze

        colors = ["blue", "g", "r", "c", "m", "y", "k"]
        color_map = {
            ansatz: colors[i % len(colors)] for i, ansatz in enumerate(unique_ansatze)
        }

        if graph_type == "scatter":
            for ansatz in unique_ansatze:
                modifiers = []
                energies = []
                for modifier in transformer.bond_modifiers:
                    program_key = (ansatz.name, modifier)
                    if program_key in self._programs:
                        modifiers.append(modifier)
                        energies.append(self._programs[program_key].best_loss)

                plt.scatter(
                    modifiers,
                    energies,
                    color=color_map[ansatz],
                    label=ansatz.name,
                )

        elif graph_type == "line":
            for ansatz in unique_ansatze:
                energies = []
                for modifier in transformer.bond_modifiers:
                    energies.append(self._programs[(ansatz.name, modifier)].best_loss)

                plt.plot(
                    transformer.bond_modifiers,
                    energies,
                    label=ansatz.name,
                    color=color_map[ansatz],
                )

        plt.xlabel("Scale Factor" if transformer._mode == "scale" else "Bond Δ")
        plt.ylabel("Energy level")
        plt.legend()
        plt.show()
